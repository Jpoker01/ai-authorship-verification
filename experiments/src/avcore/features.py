"""BoW and TF-IDF features of text pairs, built from token counts that are computed once per process.

The texts of the cleaned jsonl files are counted once per lowercase value, streamed line by
line, so the texts are never held in memory. Every Optuna trial then selects its vocabulary
and weights these counts without reading the texts again. The features are equal to the
output of the scikit-learn vectorizers fitted on the same texts.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity

SPLITS = ("train", "validation", "test")
PAIRS_COLUMNS = ["id", "same"]

# a vocabulary is fitted on the first splits and applied to the second one
TUNING_SPLITS = (("train",), "validation")
FINAL_SPLITS = (("train", "validation"), "test")
HELD_OUT_SPLITS = (("train",), "test")

@dataclass(frozen=True)
class SplitMatrices:
    """Matrices of one split; row i of both matrices belongs to pair i.

    Attributes:
        pairs_df: Id and label (same) of every pair.
        first: Vector of the first text of every pair.
        second: Vector of the second text of every pair.
    """

    pairs_df: pd.DataFrame
    first: sp.csr_matrix
    second: sp.csr_matrix


@dataclass(frozen=True)
class PairMatrices:
    """Vectors of both texts of every pair of the train, validation and test split.

    Attributes:
        columns: Name of every column (a term).
        settings: How the vectors were built.
        splits: Matrices of every split.
    """

    columns: list[str]
    settings: dict[str, object]
    splits: dict[str, SplitMatrices]

@dataclass(frozen=True)
class PairFeatures:
    """Features of the fit pairs and of the target pairs.

    Attributes:
        X_fit: Features of the first and of the second text of every fit pair.
        X_target: Features of the first and of the second text of every target pair.
        fit_pairs_df: Id and label of every fit pair.
        target_pairs_df: Id and label of every target pair.
        vectorizer: Fitted scikit-learn vectorizer that gives the same features from raw texts,
            None for features that no vectorizer builds (graph vectors).
    """

    X_fit: tuple[sp.csr_matrix, sp.csr_matrix]
    X_target: tuple[sp.csr_matrix, sp.csr_matrix]
    fit_pairs_df: pd.DataFrame
    target_pairs_df: pd.DataFrame
    vectorizer: CountVectorizer | TfidfVectorizer | None

def read_pair_labels(path: Path) -> pd.DataFrame:
    """Read the id and the label of every pair of a cleaned jsonl file, without its texts.

    Args:
        path: Cleaned jsonl file with the keys id, pair and same.

    Returns:
        One row per line, with the columns id and same.
    """
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            rows.append({"id": record["id"], "same": record["same"]})
    return pd.DataFrame(rows, columns=PAIRS_COLUMNS)

def _iter_side_texts(path: Path, side: int) -> Iterator[str]:
    """Yield text number side (0 or 1) of every pair, one line of the file at a time."""
    with path.open(encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)["pair"][side]


def iter_pair_texts(path: Path) -> Iterator[str]:
    """Yield the first text of every pair, then the second text of every pair.

    Args:
        path: Cleaned jsonl file with the keys id, pair and same.

    Returns:
        Twice as many texts as pairs, read from the file twice and never held in memory.
    """
    return chain(_iter_side_texts(path, 0), _iter_side_texts(path, 1))

def split_sides(pairs_df: pd.DataFrame, vectors: sp.csr_matrix) -> SplitMatrices:
    """Split the rows built from iter_pair_texts back into the first and the second texts.

    Args:
        pairs_df: The pairs the texts came from, with the columns id and same.
        vectors: One row per text, all first texts, then all second texts.

    Returns:
        The id and label of every pair with the vectors of its first and second text.

    Raises:
        ValueError: If there are not exactly two rows per pair.
    """
    n_pairs = len(pairs_df)
    if vectors.shape[0] != 2 * n_pairs:
        raise ValueError(f"{vectors.shape[0]} rows for {n_pairs} pairs.")
    return SplitMatrices(
        pairs_df=pairs_df[PAIRS_COLUMNS],
        first=vectors[:n_pairs],
        second=vectors[n_pairs:],
    )

def build_token_counts(
    cleaned_paths: Mapping[str, Path], *, lowercase: bool
) -> PairMatrices:
    """Count the words of every text once, streamed from the cleaned jsonl files.

    The columns are all terms of the training and validation texts, so the counts serve
    a vectorizer fitted on the training texts as well as one fitted on both; a term of
    the test texts only is never a column.

    Args:
        cleaned_paths: Cleaned jsonl file of the train, validation and test split.
        lowercase: Whether the texts are lowercased before they are split into tokens.

    Returns:
        The token counts of both texts of every pair of the three splits.

    Raises:
        ValueError: If a split is missing or unknown, or a file has another number of
            lines when it is read again.
    """
    if set(cleaned_paths) != set(SPLITS):
        raise ValueError(f"The splits have to be {list(SPLITS)}.")
    labels = {name: read_pair_labels(cleaned_paths[name]) for name in SPLITS}
    n_train = len(labels["train"])
    n_validation = len(labels["validation"])

    # one count of the training and validation texts; its rows are the train first texts,
    # the train second texts, the validation first texts and the validation second texts
    vectorizer = CountVectorizer(lowercase=lowercase, dtype=np.int32)
    fit_texts = chain(
        iter_pair_texts(cleaned_paths["train"]),
        iter_pair_texts(cleaned_paths["validation"]),
    )
    fit_counts = vectorizer.fit_transform(fit_texts)
    # fit_transform leaves the entries of a row in the order the terms were first seen
    fit_counts.sort_indices()
    if fit_counts.shape[0] != 2 * (n_train + n_validation):
        raise ValueError(
            f"{fit_counts.shape[0]} rows for {n_train + n_validation} pairs.")

    # the test texts are only counted on these columns, so their own terms are dropped
    test_counts = vectorizer.transform(iter_pair_texts(cleaned_paths["test"]))

    # every side is sliced straight from fit_counts, so no second copy of a split is made
    train_second_start = n_train
    validation_first_start = 2 * n_train
    validation_second_start = 2 * n_train + n_validation
    splits = {
        "train": SplitMatrices(
            pairs_df=labels["train"],
            first=fit_counts[:train_second_start],
            second=fit_counts[train_second_start:validation_first_start],
        ),
        "validation": SplitMatrices(
            pairs_df=labels["validation"],
            first=fit_counts[validation_first_start:validation_second_start],
            second=fit_counts[validation_second_start:],
        ),
        "test": split_sides(labels["test"], test_counts),
    }
    return PairMatrices(
        columns=vectorizer.get_feature_names_out().tolist(),
        settings={"lowercase": lowercase},
        splits=splits,
    )

def _check_splits(fit_splits: Sequence[str], target_split: str) -> None:
    """Raise ValueError unless the splits are the tuning, the final or the held-out combination."""
    combination = (tuple(fit_splits), target_split)
    if combination not in (TUNING_SPLITS, FINAL_SPLITS, HELD_OUT_SPLITS):
        raise ValueError(
            f"fit_splits {list(fit_splits)} with target_split {target_split!r} is not allowed."
        )

def _select_columns(
    fit_counts: sp.csr_matrix,
    *,
    max_features: int | None,
    min_df: float,
    max_df: float,
) -> np.ndarray:
    """Return the columns a scikit-learn vectorizer keeps when fitted on the fit documents.

    Args:
        fit_counts: Token counts, one row per fit document (1 for a present term if binary).
        max_features: Number of most frequent terms to keep, None for all.
        min_df: Lowest document frequency of a kept term, a share (float) or a count (int).
        max_df: Highest document frequency of a kept term, a share (float) or a count (int).

    Returns:
        The kept column numbers, ascending.

    Raises:
        ValueError: If max_df is below min_df or no term is kept.
    """
    n_documents, n_columns = fit_counts.shape
    # every stored entry is one term in one document, so this counts the documents per term
    document_frequencies = np.bincount(fit_counts.indices, minlength=n_columns)
    # an int is a number of documents, a float a share of them
    min_documents = min_df if isinstance(min_df,
                                         Integral) else min_df * n_documents
    max_documents = max_df if isinstance(max_df,
                                         Integral) else max_df * n_documents
    if max_documents < min_documents:
        raise ValueError("max_df corresponds to < documents than min_df")

    # a term that is only in other documents is not in the vocabulary of the fit documents
    columns = np.flatnonzero(
        (document_frequencies > 0)
        & (document_frequencies >= min_documents)
        & (document_frequencies <= max_documents)
    )
    if max_features is not None and len(columns) > max_features:
        # int64 as in scikit-learn on every platform (a sum of int32 counts is int32 on Windows)
        term_counts = np.asarray(
            fit_counts.sum(axis=0, dtype=np.int64)).ravel()
        # the same call on the same values as in scikit-learn, so ties fall the same way
        most_frequent_first = (-term_counts[columns]).argsort()
        columns = np.sort(columns[most_frequent_first[:max_features]])

    if len(columns) == 0:
        raise ValueError(
            "After pruning, no terms remain. Try a lower min_df or a higher max_df."
        )
    return columns


def _vocabulary(token_counts: PairMatrices, columns: np.ndarray) -> dict[str, int]:
    """Return the kept terms with their feature numbers, as a scikit-learn vocabulary."""
    return {token_counts.columns[column]: index for index, column in enumerate(columns)}

def _stack_fit_documents(
    token_counts: PairMatrices, fit_splits: Sequence[str]
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """Return one row per fit document (all first texts, then all second texts) and the fit pairs.

    The rows are stacked once, so a trial holds a single copy of the fit counts.
    """
    parts = [token_counts.splits[name] for name in fit_splits]
    fit_counts = sp.vstack(
        [part.first for part in parts] + [part.second for part in parts], format="csr"
    )
    fit_pairs_df = pd.concat([part.pairs_df for part in parts], ignore_index=True)
    return fit_counts, fit_pairs_df

def _bow_matrix(
    counts: sp.csr_matrix, columns: np.ndarray, binary: bool
) -> sp.csr_matrix:
    """Return the kept columns as CountVectorizer returns them (int64, 1 for a present term if binary)."""
    X = counts[:, columns].astype(np.int64)
    if binary:
        X.data.fill(1)
    return X

def build_bow_features(
    token_counts: PairMatrices,
    fit_splits: Sequence[str],
    target_split: str,
    *,
    max_features: int | None,
    min_df: float,
    max_df: float,
    binary: bool,
) -> PairFeatures:
    """Build the BoW features of the fit and of the target pairs from the token counts.

    Equal to a CountVectorizer with these arguments fitted on both texts of the fit pairs.

    Args:
        token_counts: Output of build_token_counts.
        fit_splits: ["train"] while tuning and for the held-out test scores,
            ["train", "validation"] for the final model.
        target_split: "validation" while tuning, "test" for the final model
            and for the held-out test scores.
        max_features: Number of most frequent terms to keep, None for all.
        min_df: Lowest document frequency of a kept term, a share (float) or a count (int).
        max_df: Highest document frequency of a kept term, a share (float) or a count (int).
        binary: Whether a term is counted at most once per text.

    Returns:
        The features of both texts of the fit and of the target pairs.

    Raises:
        ValueError: If the splits are not the tuning, the final or the held-out
            combination, max_df is below min_df or no term is kept.
    """
    _check_splits(fit_splits, target_split)
    fit_counts, fit_pairs_df = _stack_fit_documents(token_counts, fit_splits)
    n_fit_pairs = len(fit_pairs_df)
    target = token_counts.splits[target_split]
    if binary:
        fit_counts.data.fill(1)
    columns = _select_columns(
        fit_counts, max_features=max_features, min_df=min_df, max_df=max_df
    )
    vectorizer = CountVectorizer(
        lowercase=token_counts.settings["lowercase"],
        vocabulary=_vocabulary(token_counts, columns),
        binary=binary,
    )
    return PairFeatures(
        X_fit=(
            _bow_matrix(fit_counts[:n_fit_pairs], columns, binary),
            _bow_matrix(fit_counts[n_fit_pairs:], columns, binary),
        ),
        X_target=(
            _bow_matrix(target.first, columns, binary),
            _bow_matrix(target.second, columns, binary),
        ),
        fit_pairs_df=fit_pairs_df,
        target_pairs_df=target.pairs_df,
        vectorizer=vectorizer,
    )



def build_tfidf_features(
    token_counts: PairMatrices,
    fit_splits: Sequence[str],
    target_split: str,
    *,
    max_features: int | None,
    min_df: float,
    sublinear_tf: bool,
) -> PairFeatures:
    """Build the TF-IDF features of the fit and of the target pairs from the token counts.

    Equal to a TfidfVectorizer with these arguments fitted on both texts of the fit pairs.

    Args:
        token_counts: Output of build_token_counts.
        fit_splits: ["train"] while tuning and for the held-out test scores,
            ["train", "validation"] for the final model.
        target_split: "validation" while tuning, "test" for the final model
            and for the held-out test scores.
        max_features: Number of most frequent terms to keep, None for all.
        min_df: Lowest document frequency of a kept term, a share (float) or a count (int).
        sublinear_tf: Whether a count is replaced by 1 + log(count).

    Returns:
        The features of both texts of the fit and of the target pairs.

    Raises:
        ValueError: If the splits are not the tuning, the final or the held-out
            combination or no term is kept.
    """
    _check_splits(fit_splits, target_split)
    fit_counts, fit_pairs_df = _stack_fit_documents(token_counts, fit_splits)
    n_fit_pairs = len(fit_pairs_df)
    target = token_counts.splits[target_split]
    columns = _select_columns(
        fit_counts, max_features=max_features, min_df=min_df, max_df=1.0
    )
    kept_counts = fit_counts[:, columns]
    # the counts of all columns are not needed any more and can be freed before the weighting
    del fit_counts
    transformer = TfidfTransformer(sublinear_tf=sublinear_tf)
    transformer.fit(kept_counts)
    vectorizer = TfidfVectorizer(
        lowercase=token_counts.settings["lowercase"],
        vocabulary=_vocabulary(token_counts, columns),
        sublinear_tf=sublinear_tf,
    )
    vectorizer.idf_ = transformer.idf_
    return PairFeatures(
        X_fit=(
            transformer.transform(kept_counts[:n_fit_pairs]),
            transformer.transform(kept_counts[n_fit_pairs:]),
        ),
        X_target=(
            transformer.transform(target.first[:, columns]),
            transformer.transform(target.second[:, columns]),
        ),
        fit_pairs_df=fit_pairs_df,
        target_pairs_df=target.pairs_df,
        vectorizer=vectorizer,
    )



def compute_absolute_difference_features(
    X_pair: tuple[sp.csr_matrix, sp.csr_matrix],
) -> sp.csr_matrix:
    """Return |first - second| for every pair, one row per pair.

    Args:
        X_pair: Features of the first and of the second text of every pair.

    Returns:
        The absolute difference of the two feature vectors of every pair.
    """
    X_1, X_2 = X_pair
    return abs(X_1 - X_2)


def compute_element_wise_features(
    X_pair: tuple[sp.csr_matrix, sp.csr_matrix],
) -> sp.csr_matrix:
    """Return first * second, feature by feature, for every pair, one row per pair.

    Args:
        X_pair: Features of the first and of the second text of every pair.

    Returns:
        The element-wise product of the two feature vectors of every pair.
    """
    X_1, X_2 = X_pair
    return X_1.multiply(X_2)



def compute_cosine_similarities(
    X_pair: tuple[sp.csr_matrix | np.ndarray, sp.csr_matrix | np.ndarray],
) -> np.ndarray:
    """Return the cosine similarity of the two texts of every pair.

    Args:
        X_pair: Vectors of the first and of the second text of every pair.

    Returns:
        One similarity per pair; 0 when one of the two vectors has only zeros.
    """
    X_1, X_2 = X_pair
    similarities = [
        cosine_similarity(X_1[row : row + 1], X_2[row : row + 1])[0, 0]
        for row in range(X_1.shape[0])
    ]
    return np.array(similarities)
