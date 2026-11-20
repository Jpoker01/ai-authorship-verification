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