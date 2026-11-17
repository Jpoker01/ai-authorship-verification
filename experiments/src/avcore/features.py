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
