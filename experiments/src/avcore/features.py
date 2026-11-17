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