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

    vectorizer = CountVectorizer(lowercase=lowercase, dtype=np.int32)
    fit_texts = chain(
        iter_pair_texts(cleaned_paths["train"]),
        iter_pair_texts(cleaned_paths["validation"]),
    )
    fit_counts = vectorizer.fit_transform(fit_texts)
    # fit_transform leaves the entries of a row in the order the terms were first seen
    fit_counts.sort_indices()
    test_counts = vectorizer.transform(iter_pair_texts(cleaned_paths["test"]))

    n_train = len(labels["train"])
    n_validation = len(labels["validation"])
    if fit_counts.shape[0] != 2 * (n_train + n_validation):
        raise ValueError(f"{fit_counts.shape[0]} rows for {n_train + n_validation} pairs.")
    # the sides are sliced straight from fit_counts, so no second copy of a split is made
    validation_start = 2 * n_train
    splits = {
        "train": SplitMatrices(
            pairs_df=labels["train"],
            first=fit_counts[:n_train],
            second=fit_counts[n_train:validation_start],
        ),
        "validation": SplitMatrices(
            pairs_df=labels["validation"],
            first=fit_counts[validation_start : validation_start + n_validation],
            second=fit_counts[validation_start + n_validation :],
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