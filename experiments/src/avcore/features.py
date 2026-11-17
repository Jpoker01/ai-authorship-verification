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