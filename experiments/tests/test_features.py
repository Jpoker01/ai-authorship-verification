from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from avcore import features as features_module
from avcore.data import write_pairs_jsonl
from avcore.features import (
    SPLITS,
    PairFeatures,
    PairMatrices,
    SplitMatrices,
    build_bow_features,
    build_tfidf_features,
    build_token_counts,
    compute_absolute_difference_features,
    compute_cosine_similarities,
    compute_element_wise_features,
    read_pair_labels,
    split_sides,
)

TUNING = (["train"], "validation")
FINAL = (["train", "validation"], "test")
HELD_OUT = (["train"], "test")
OTHER_SPLIT_COMBINATIONS = [
    (["train", "test"], "validation"),
    (["test"], "validation"),
    (["validation"], "test"),
    (["train", "validation"], "validation"),
]


def _random_pairs(n_pairs: int, seed: int, prefix: str) -> pd.DataFrame:
    """Random pairs of a small vocabulary: many terms share a count, the same word comes in two cases.

    Pair i has the id f"{prefix}{i}" and is a same-author pair when i is even.
    """
    rng = np.random.default_rng(seed)
    words = [f"w{index}" for index in range(120)] + [f"W{index}" for index in range(40)]
    weights = 1 / np.arange(1, len(words) + 1)
    weights = weights / weights.sum()
    records = []
    for index in range(n_pairs):
        texts = []
        for _ in range(2):
            n_words = int(rng.integers(5, 60))
            text_words = rng.choice(words, size=n_words, p=weights)
            texts.append(" ".join(text_words))
        records.append(
            {"id": f"{prefix}{index}", "pair": texts, "same": index % 2 == 0},
        )
    return pd.DataFrame(records)


def _write_splits(
    pairs_by_split: dict[str, pd.DataFrame],
    folder: Path,
) -> dict[str, Path]:
    """Write every split as a cleaned jsonl file and return the paths by split."""
    paths = {}
    for name, pairs_df in pairs_by_split.items():
        paths[name] = folder / f"{name}.jsonl"
        write_pairs_jsonl(pairs_df, paths[name])
    return paths


def _texts(pairs_df: pd.DataFrame) -> list[str]:
    """First text of every pair, then the second text of every pair - the order a vectorizer is fitted on."""
    first_texts = [pair[0] for pair in pairs_df["pair"]]
    second_texts = [pair[1] for pair in pairs_df["pair"]]
    return first_texts + second_texts


def _side(pairs_df: pd.DataFrame, side: int) -> list[str]:
    """Text number side (0 or 1) of every pair."""
    return [pair[side] for pair in pairs_df["pair"]]


def _assert_same_matrix(actual: sp.csr_matrix, expected: sp.csr_matrix) -> None:
    """Both are CSR matrices with the same shape, dtype and the same three arrays."""
    assert actual.format == "csr"
    assert expected.format == "csr"
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual.indptr, expected.indptr)
    assert np.array_equal(actual.indices, expected.indices)
    assert np.array_equal(actual.data, expected.data)


@pytest.fixture(scope="module")
def pairs_by_split() -> dict[str, pd.DataFrame]:
    """Random train (60), validation (20) and test (25) pairs."""
    return {
        "train": _random_pairs(60, seed=1, prefix="train"),
        "validation": _random_pairs(20, seed=2, prefix="validation"),
        "test": _random_pairs(25, seed=3, prefix="test"),
    }


@pytest.fixture(scope="module")
def cleaned_paths(
    pairs_by_split: dict[str, pd.DataFrame],
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    """pairs_by_split written as cleaned jsonl files."""
    return _write_splits(pairs_by_split, tmp_path_factory.mktemp("cleaned"))


@pytest.fixture(scope="module")
def token_counts(cleaned_paths: dict[str, Path]) -> PairMatrices:
    """Token counts of the cleaned files without lowercasing."""
    return build_token_counts(cleaned_paths, lowercase=False)


def _assert_pairs_of_rows(
    features: PairFeatures,
    fit_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> None:
    """The returned pairs are the fit pairs (numbered from 0) and the target pairs, in the order of the matrix rows."""
    assert features.fit_pairs_df.index.tolist() == list(range(len(fit_df)))
    assert features.fit_pairs_df["id"].tolist() == fit_df["id"].tolist()
    assert features.fit_pairs_df["same"].tolist() == fit_df["same"].tolist()
    assert features.target_pairs_df["id"].tolist() == target_df["id"].tolist()
    assert features.target_pairs_df["same"].tolist() == target_df["same"].tolist()


def _fit_and_target(
    pairs_by_split: dict[str, pd.DataFrame],
    stage: tuple[list[str], str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the raw pairs of a stage: the pairs to fit on and the pairs the fit is applied to.

    A stage is (fit splits, target split), e.g. FINAL = (["train", "validation"], "test")
    gives the train pairs followed by the validation pairs, and the test pairs.
    """
    fit_split_names, target_split_name = stage

    fit_dfs = []
    for name in fit_split_names:
        fit_dfs.append(pairs_by_split[name])
    fit_df = pd.concat(fit_dfs, ignore_index=True)

    target_df = pairs_by_split[target_split_name]
    return fit_df, target_df


@pytest.mark.parametrize("lowercase", [False, True])
def test_build_token_counts_equal_count_vectorizer_on_the_texts(
    pairs_by_split: dict[str, pd.DataFrame],
    cleaned_paths: dict[str, Path],
    lowercase: bool,
) -> None:
    """The counts streamed from the files equal CountVectorizer fitted on the train + validation texts in memory."""
    fit_df = pd.concat(
        [pairs_by_split["train"], pairs_by_split["validation"]],
        ignore_index=True,
    )
    reference = CountVectorizer(lowercase=lowercase, dtype=np.int32)
    reference.fit(_texts(fit_df))

    token_counts = build_token_counts(cleaned_paths, lowercase=lowercase)

    assert token_counts.columns == reference.get_feature_names_out().tolist()
    assert token_counts.settings == {"lowercase": lowercase}
    for name in SPLITS:
        split = token_counts.splits[name]
        pairs_df = pairs_by_split[name]
        assert split.pairs_df["id"].tolist() == pairs_df["id"].tolist()
        assert split.pairs_df["same"].tolist() == pairs_df["same"].tolist()
        _assert_same_matrix(split.first, reference.transform(_side(pairs_df, 0)))
        _assert_same_matrix(split.second, reference.transform(_side(pairs_df, 1)))


@pytest.mark.parametrize("stage", [TUNING, FINAL, HELD_OUT])
@pytest.mark.parametrize(
    ("max_features", "min_df", "max_df", "binary"),
    [
        (None, 1, 1.0, False),
        (None, 0.05, 0.7, False),
        (None, 3, 0.9, True),
        (None, 2, 30, False),
        (40, 1e-5, 1.0, False),
        (80, 1e-5, 0.95, True),
        (80, 0.0, 1.0, False),
    ],
)
def test_build_bow_features_equal_the_fitted_count_vectorizer(
    pairs_by_split: dict[str, pd.DataFrame],
    token_counts: PairMatrices,
    stage: tuple[list[str], str],
    max_features: int | None,
    min_df: float,
    max_df: float,
    binary: bool,
) -> None:
    """The features of the fit and target pairs equal a CountVectorizer fitted on the fit texts."""
    fit_splits, target_split = stage
    fit_df, target_df = _fit_and_target(pairs_by_split, stage)
    reference = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        lowercase=False,
        binary=binary,
    )
    reference.fit(_texts(fit_df))

    features = build_bow_features(
        token_counts,
        fit_splits,
        target_split,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        binary=binary,
    )

    _assert_same_matrix(features.X_fit[0], reference.transform(_side(fit_df, 0)))
    _assert_same_matrix(features.X_fit[1], reference.transform(_side(fit_df, 1)))
    _assert_same_matrix(features.X_target[0], reference.transform(_side(target_df, 0)))
    _assert_same_matrix(features.X_target[1], reference.transform(_side(target_df, 1)))
    _assert_pairs_of_rows(features, fit_df, target_df)


@pytest.mark.parametrize("stage", [TUNING, FINAL, HELD_OUT])
@pytest.mark.parametrize(
    ("max_features", "min_df", "sublinear_tf"),
    [
        (None, 1, True),
        (None, 0.05, False),
        (40, 1e-5, True),
        (80, 1e-5, False),
        (80, 0.0, True),
    ],
)
def test_build_tfidf_features_equal_the_fitted_tfidf_vectorizer(
    pairs_by_split: dict[str, pd.DataFrame],
    token_counts: PairMatrices,
    stage: tuple[list[str], str],
    max_features: int | None,
    min_df: float,
    sublinear_tf: bool,
) -> None:
    """The features of the fit and target pairs equal a TfidfVectorizer fitted on the fit texts."""
    fit_splits, target_split = stage
    fit_df, target_df = _fit_and_target(pairs_by_split, stage)
    reference = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        lowercase=False,
        sublinear_tf=sublinear_tf,
    )
    reference.fit(_texts(fit_df))

    features = build_tfidf_features(
        token_counts,
        fit_splits,
        target_split,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )

    _assert_same_matrix(features.X_fit[0], reference.transform(_side(fit_df, 0)))
    _assert_same_matrix(features.X_fit[1], reference.transform(_side(fit_df, 1)))
    _assert_same_matrix(features.X_target[0], reference.transform(_side(target_df, 0)))
    _assert_same_matrix(features.X_target[1], reference.transform(_side(target_df, 1)))
    _assert_pairs_of_rows(features, fit_df, target_df)


@pytest.mark.parametrize("lowercase", [False, True])
@pytest.mark.parametrize("representation", ["bow", "tfidf"])
def test_features_vectorizer_gives_the_same_features_from_raw_texts(
    pairs_by_split: dict[str, pd.DataFrame],
    cleaned_paths: dict[str, Path],
    tmp_path: Path,
    representation: str,
    lowercase: bool,
) -> None:
    """The returned vectorizer, saved and loaded again, turns raw texts into the same features."""
    fit_splits, target_split = FINAL
    target_df = pairs_by_split[target_split]
    vectorizer_path = tmp_path / "vectorizer.pkl"
    max_features = 80
    min_df = 1e-5
    max_df = 0.95
    binary = True
    sublinear_tf = True
    token_counts = build_token_counts(cleaned_paths, lowercase=lowercase)
    if representation == "bow":
        features = build_bow_features(
            token_counts,
            fit_splits,
            target_split,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            binary=binary,
        )
    else:
        features = build_tfidf_features(
            token_counts,
            fit_splits,
            target_split,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
        )

    joblib.dump(features.vectorizer, vectorizer_path)
    vectorizer = joblib.load(vectorizer_path)

    _assert_same_matrix(vectorizer.transform(_side(target_df, 0)), features.X_target[0])
    _assert_same_matrix(vectorizer.transform(_side(target_df, 1)), features.X_target[1])


def test_build_token_counts_ignores_terms_of_the_test_texts(tmp_path: Path) -> None:
    """A term that is only in the test texts is never a column, so no vocabulary can select it."""
    pairs_by_split = {
        "train": pd.DataFrame(
            [{"id": "t0", "pair": ["apple pear", "apple plum"], "same": True}],
        ),
        "validation": pd.DataFrame(
            [{"id": "v0", "pair": ["zebra apple", "pear pear"], "same": False}],
        ),
        "test": pd.DataFrame(
            [{"id": "e0", "pair": ["yak apple", "yak yak"], "same": True}],
        ),
    }
    cleaned_paths = _write_splits(pairs_by_split, tmp_path)
    lowercase = False

    token_counts = build_token_counts(cleaned_paths, lowercase=lowercase)

    assert token_counts.columns == ["apple", "pear", "plum", "zebra"]
    assert token_counts.splits["test"].second.nnz == 0


@pytest.mark.parametrize("stage", [TUNING, FINAL, HELD_OUT])
@pytest.mark.parametrize(
    ("max_features", "min_df", "sublinear_tf"),
    [
        (None, 1, True),
        (None, 0.05, False),
        (40, 1e-5, True),
        (80, 1e-5, False),
        (80, 0.0, True),
    ],
)
def test_build_tfidf_features_equal_the_fitted_tfidf_vectorizer(
    pairs_by_split: dict[str, pd.DataFrame],
    token_counts: PairMatrices,
    stage: tuple[list[str], str],
    max_features: int | None,
    min_df: float,
    sublinear_tf: bool,
) -> None:
    """The features of the fit and target pairs equal a TfidfVectorizer fitted on the fit texts."""
    fit_splits, target_split = stage
    fit_df, target_df = _fit_and_target(pairs_by_split, stage)
    reference = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        lowercase=False,
        sublinear_tf=sublinear_tf,
    )
    reference.fit(_texts(fit_df))

    features = build_tfidf_features(
        token_counts,
        fit_splits,
        target_split,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )

    _assert_same_matrix(features.X_fit[0], reference.transform(_side(fit_df, 0)))
    _assert_same_matrix(features.X_fit[1], reference.transform(_side(fit_df, 1)))
    _assert_same_matrix(features.X_target[0], reference.transform(_side(target_df, 0)))
    _assert_same_matrix(features.X_target[1], reference.transform(_side(target_df, 1)))
    _assert_pairs_of_rows(features, fit_df, target_df)


@pytest.mark.parametrize("lowercase", [False, True])
@pytest.mark.parametrize("representation", ["bow", "tfidf"])
def test_features_vectorizer_gives_the_same_features_from_raw_texts(
    pairs_by_split: dict[str, pd.DataFrame],
    cleaned_paths: dict[str, Path],
    tmp_path: Path,
    representation: str,
    lowercase: bool,
) -> None:
    """The returned vectorizer, saved and loaded again, turns raw texts into the same features."""
    fit_splits, target_split = FINAL
    target_df = pairs_by_split[target_split]
    vectorizer_path = tmp_path / "vectorizer.pkl"
    max_features = 80
    min_df = 1e-5
    max_df = 0.95
    binary = True
    sublinear_tf = True
    token_counts = build_token_counts(cleaned_paths, lowercase=lowercase)
    if representation == "bow":
        features = build_bow_features(
            token_counts,
            fit_splits,
            target_split,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            binary=binary,
        )
    else:
        features = build_tfidf_features(
            token_counts,
            fit_splits,
            target_split,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
        )

    joblib.dump(features.vectorizer, vectorizer_path)
    vectorizer = joblib.load(vectorizer_path)

    _assert_same_matrix(vectorizer.transform(_side(target_df, 0)), features.X_target[0])
    _assert_same_matrix(vectorizer.transform(_side(target_df, 1)), features.X_target[1])

