import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
import scipy.sparse as sp
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from avcore import classifiers, runs, tracking
from avcore.classifiers import (
    CLASSIFIER_FIXED_PARAMS,
    CLASSIFIER_NAMES,
    CLASSIFIER_SEARCH_SPACES,
    SEED,
    PairPredictions,
    build_classifier,
    build_prediction_table,
    predict_pairs,
    suggest_params,
)

# the scikit-learn method that gives the score of every classifier
SCORE_METHODS = {
    "multinomial_naive_bayes": "predict_proba",
    "logistic_regression": "predict_proba",
    "linear_svc": "decision_function",
    "k-nearest_neighbors": "predict_proba",
    "random_forest": "predict_proba",
    "xgboost": "predict_proba",
}


class RecordingTrial:
    """A stand-in for an Optuna trial that records every suggestion and returns the low end."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def suggest_categorical(self, name: str, choices: list) -> object:
        self.calls.append((name, CategoricalDistribution(choices)))
        return choices[0]

    def suggest_int(self, name: str, low: int, high: int) -> int:
        self.calls.append((name, IntDistribution(low, high)))
        return low

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
    ) -> float:
        self.calls.append((name, FloatDistribution(low, high, log=log)))
        return low


def _pairs_data(n_pairs: int, seed: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """Non-negative sparse pair features (MNB needs counts) with a bool label, as in the notebooks.

    Pair i is a same-author pair (True) when i is even; same pairs have larger values in the
    first five columns.
    """
    rng = np.random.default_rng(seed)
    y = np.arange(n_pairs) % 2 == 0
    X = rng.poisson(1.0, size=(n_pairs, 20)).astype(np.float64)
    X[y, :5] += rng.poisson(3.0, size=(int(y.sum()), 5))
    return sp.csr_matrix(X), y


@pytest.fixture(scope="module")
def pairs_data() -> tuple[sp.csr_matrix, np.ndarray]:
    """60 training pairs."""
    return _pairs_data(n_pairs=60, seed=1)


def _low_params(classifier_name: str) -> dict[str, object]:
    """The low end of every searched hyperparameter (a forest with 10 trees, to be fast)."""
    params = suggest_params(RecordingTrial(), CLASSIFIER_SEARCH_SPACES[classifier_name])
    if classifier_name == "random_forest":
        params["n_estimators"] = 10
    return params


def _fitted(
    classifier_name: str,
    X: sp.csr_matrix,
    y: np.ndarray,
    n_jobs: int = 1,
) -> object:
    """The classifier with the low end of every searched hyperparameter, fitted on X, y."""
    params = _low_params(classifier_name)
    classifier = build_classifier(classifier_name, params, n_jobs=n_jobs)
    classifier.fit(X, y)
    return classifier


def test_classifier_tables_name_the_same_six_classifiers() -> None:
    """Every table holds the six classifiers of CLASSIFIER_NAMES, in the same order."""
    expected_names = list(SCORE_METHODS)

    assert CLASSIFIER_NAMES == expected_names
    assert list(CLASSIFIER_SEARCH_SPACES) == expected_names
    assert list(CLASSIFIER_FIXED_PARAMS) == expected_names
    # the classifiers whose fit or predict uses threads (LogReg does not for two classes)
    assert classifiers.THREADED_CLASSIFIERS == [
        "k-nearest_neighbors",
        "random_forest",
        "xgboost",
    ]
    assert classifiers.DECISION_FUNCTION_CLASSIFIERS == ["linear_svc"]


def test_fixed_params_are_the_agreed_params() -> None:
    """The fixed params - seeds of the four random classifiers, liblinear, dual="auto", no n_jobs."""
    expected = {
        "multinomial_naive_bayes": {},
        "logistic_regression": {
            "solver": "liblinear",
            "max_iter": 100,
            "random_state": 42,
        },
        "linear_svc": {
            "penalty": "l2",
            "dual": "auto",
            "max_iter": 5000,
            "random_state": 42,
        },
        "k-nearest_neighbors": {"algorithm": "auto"},
        "random_forest": {"random_state": 42},
        "xgboost": {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": 300,
            "tree_method": "auto",
            "random_state": 42,
        },
    }

    assert CLASSIFIER_FIXED_PARAMS == expected


def test_linear_classifiers_search_c_up_to_1e3() -> None:
    """LogReg and LinearSVC search C log-scaled in [0.01, 1e3]; LogReg no longer searches the solver."""
    expected_logreg = {
        "C": ("float_log", 0.01, 1_000.0),
        "penalty": ("categorical", "l1", "l2"),
    }
    expected_svc = {
        "C": ("float_log", 0.01, 1_000.0),
        "loss": ("categorical", "hinge", "squared_hinge"),
        "class_weight": ("categorical", None, "balanced"),
    }

    assert CLASSIFIER_SEARCH_SPACES["logistic_regression"] == expected_logreg
    assert CLASSIFIER_SEARCH_SPACES["linear_svc"] == expected_svc
