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

