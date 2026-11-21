"""The six classifiers of the pairwise notebooks - their search spaces, fixed params and predictions.

Every pairwise notebook (BoW and TF-IDF, ISG) tunes the same six classifiers with the same
search spaces. A classifier decides with its own predict, as in v1; its score (probability
of the same author, or the decision function of LinearSVC) is kept for every pair.
"""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from avcore.tracking import PREDICTION_COLUMNS

SEED = 42

CLASSIFIER_CLASSES: dict[str, type[BaseEstimator]] = {
    "multinomial_naive_bayes": MultinomialNB,
    "logistic_regression": LogisticRegression,
    "linear_svc": LinearSVC,
    "k-nearest_neighbors": KNeighborsClassifier,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}

CLASSIFIER_NAMES = list(CLASSIFIER_CLASSES)

# per param: (kind, *arguments), suggested in this order
# kinds: categorical (flat choices, storable in study settings), int, float, float_log
# name -> (kind, *arguments), suggested in this order; kind is categorical, int, float or
CLASSIFIER_SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "multinomial_naive_bayes": {
        "alpha": ("float_log", 1e-3, 1.0),
    },
    "logistic_regression": {
        "C": ("float_log", 0.01, 10_000.0),
        "penalty": ("categorical", "l1", "l2"),
    },
    "linear_svc": {
        "C": ("float_log", 0.01, 10_000.0),
        "loss": ("categorical", "hinge", "squared_hinge"),
        "class_weight": ("categorical", None, "balanced"),
    },
    "k-nearest_neighbors": {
        "n_neighbors": ("int", 1, 50),
        "weights": ("categorical", "uniform", "distance"),
        "metric": ("categorical", "euclidean", "manhattan", "cosine"),
    },
    "random_forest": {
        "n_estimators": ("int", 100, 600),
        "max_depth": ("int", 5, 50),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    },
    "xgboost": {
        "eta": ("float", 0.01, 0.3),
        "max_depth": ("int", 3, 12),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "reg_lambda": ("float_log", 1e-3, 10.0),
        "reg_alpha": ("float_log", 1e-3, 1.0),
    },
}

CLASSIFIER_FIXED_PARAMS: dict[str, dict[str, object]] = {
    "multinomial_naive_bayes": {},
    "logistic_regression": {
        "solver": "liblinear",
        "max_iter": 5000,
        "random_state": SEED,
    },
    "linear_svc": {
        "penalty": "l2",
        "dual": "auto",
        "max_iter": 5000,
        "random_state": SEED,
    },
    "k-nearest_neighbors": {"algorithm": "auto"},
    "random_forest": {"random_state": SEED},
    "xgboost": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 300,
        "tree_method": "auto",
        "random_state": SEED,
    },
}

# classifiers that use n_jobs - RF and kNN via joblib, XGBoost its own threads;
# LogisticRegression uses them only for more than two classes
THREADED_CLASSIFIERS = [
    "k-nearest_neighbors",
    "random_forest",
    "xgboost",
]
# a classifier without probabilities is scored by its decision function
DECISION_FUNCTION_CLASSIFIERS = ["linear_svc"]
