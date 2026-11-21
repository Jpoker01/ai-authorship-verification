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
from sklearn.pipeline import Pipeline
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
        "C": ("float_log", 0.01, 1_000.0),
        "penalty": ("categorical", "l1", "l2"),
    },
    "linear_svc": {
        "C": ("float_log", 0.01, 1_000.0),
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
        "max_iter": 100,
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


def suggest_params(
    trial: optuna.Trial,
    search_space: Mapping[str, tuple],
) -> dict[str, object]:
    """Suggest every hyperparameter of a search space, in its order.

    Args:
        trial: The running trial.
        search_space: Name -> (kind, *arguments); kind is categorical, int, float or float_log.

    Returns:
        The suggested value of every hyperparameter.

    Raises:
        ValueError: If a kind is unknown.
    """
    params: dict[str, object] = {}
    for name, (kind, *arguments) in search_space.items():
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(arguments))
        elif kind == "int":
            params[name] = trial.suggest_int(name, *arguments)
        elif kind == "float":
            params[name] = trial.suggest_float(name, *arguments)
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, *arguments, log=True)
        else:
            raise ValueError(f"Unknown kind {kind!r} of {name}.")
    return params


def _check_name(classifier_name: str) -> None:
    """Raise ValueError unless classifier_name is one of CLASSIFIER_NAMES."""
    if classifier_name not in CLASSIFIER_CLASSES:
        raise ValueError(
            f"Unknown classifier {classifier_name!r}, not one of {CLASSIFIER_NAMES}.",
        )


def build_classifier(
    classifier_name: str,
    params: Mapping[str, object],
    *,
    n_jobs: int,
    scaler: TransformerMixin | None = None,
) -> BaseEstimator:
    """Return an unfitted classifier with the searched and the fixed hyperparameters.

    Args:
        classifier_name: One of CLASSIFIER_NAMES.
        params: The searched hyperparameters - exactly the names of its search space.
        n_jobs: Threads of a classifier of THREADED_CLASSIFIERS.
        scaler: Fitted on the fit pairs in front of the classifier when given (a copy of it).

    Returns:
        The classifier, or a pipeline of the scaler and the classifier.

    Raises:
        ValueError: If the classifier is unknown or params are not exactly its searched
            hyperparameters (e.g. a vectorizer param of the best trial, or a missing one).
    """
    _check_name(classifier_name)
    searched_names = sorted(CLASSIFIER_SEARCH_SPACES[classifier_name])
    if sorted(params) != searched_names:
        raise ValueError(
            f"{classifier_name} takes the params {searched_names}, not {sorted(params)}.",
        )
    all_params = {**params, **CLASSIFIER_FIXED_PARAMS[classifier_name]}
    if classifier_name in THREADED_CLASSIFIERS:
        all_params["n_jobs"] = n_jobs
    classifier = CLASSIFIER_CLASSES[classifier_name](**all_params)
    if scaler is None:
        return classifier
    return Pipeline([("scaler", clone(scaler)), ("classifier", classifier)])


@dataclass(frozen=True)
class PairPredictions:
    """Decision and score of a classifier for every pair.

    Decide by pred, not by a threshold on score - a tie (probability 0.5, decision
    function 0) is a different author.

    Attributes:
        score: Probability of the same author, or the LinearSVC decision function.
        pred: classifier.predict - 0 = different author, 1 = same author.
    """

    score: np.ndarray
    pred: np.ndarray


def predict_pairs(
    classifier: BaseEstimator,
    classifier_name: str,
    X: sp.csr_matrix,
) -> PairPredictions:
    """Return the decision and the score of a fitted classifier for every pair.

    Args:
        classifier: The fitted classifier or pipeline of build_classifier, fitted on the
            labels 0 / 1 or False / True.
        classifier_name: Its name, one of CLASSIFIER_NAMES.
        X: One row per pair.

    Returns:
        The 0 / 1 decision of classifier.predict and the score of every pair.

    Raises:
        ValueError: If the classifier is unknown or was not fitted on the labels 0 and 1.
    """
    _check_name(classifier_name)
    if isinstance(classifier, Pipeline):
        estimator = classifier.named_steps["classifier"]
    else:
        estimator = classifier
    if estimator.classes_.tolist() != [0, 1]:
        raise ValueError(f"The classes are {estimator.classes_.tolist()}, not [0, 1].")
    if classifier_name == "random_forest":
        estimator.set_params(n_jobs=1)
    if classifier_name == "k-nearest_neighbors":
        # to ensure reproducibility of experiments (at the cost of lower speed)
        prediction_settings = config_context(enable_cython_pairwise_dist=False)
    else:
        prediction_settings = nullcontext()  # no special setting

    with prediction_settings:
        # predict returns bool for bool labels; 0 = different author, 1 = same author
        pred = classifier.predict(X).astype(int)
        if classifier_name in DECISION_FUNCTION_CLASSIFIERS:
            score = classifier.decision_function(X)
        else:
            same_author_column = 1  # the columns follow classes_, checked to be [0, 1]
            score = classifier.predict_proba(X)[:, same_author_column]

    # xgboost returns float32 - float64 keeps its exact value in the prediction table
    return PairPredictions(score=score.astype(np.float64), pred=pred)


def build_prediction_table(
    pairs_df: pd.DataFrame,
    predictions: PairPredictions,
) -> pd.DataFrame:
    """Return id, same, score and pred of every pair, 0 = different author, 1 = same author.

    Args:
        pairs_df: The pairs that were predicted
        predictions: Output of predict_pairs for these pairs.

    Returns:
        One row per pair with the columns tracking.PREDICTION_COLUMNS.

    Raises:
        ValueError: If an id repeats, there is not one prediction per pair or a score is NaN.
    """
    if not pairs_df["id"].is_unique:
        raise ValueError("The pairs hold a repeated id.")
    n_pairs = len(pairs_df)
    if len(predictions.score) != n_pairs or len(predictions.pred) != n_pairs:
        raise ValueError(
            f"{len(predictions.score)} scores and {len(predictions.pred)} decisions "
            f"for {n_pairs} pairs.",
        )
    if np.isnan(predictions.score).any():
        raise ValueError("A score is NaN.")
    return pd.DataFrame(
        {
            "id": pairs_df["id"].to_numpy(),
            "same": pairs_df["same"].astype(int).to_numpy(),
            "score": predictions.score,
            "pred": predictions.pred,
        },
        columns=PREDICTION_COLUMNS,
    )
