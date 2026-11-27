import re
import warnings
from functools import lru_cache
from typing import NamedTuple

import joblib
import numpy as np
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator
from sklearn.exceptions import InconsistentVersionWarning

from .config import CLASSIFIER_PATH, MODEL_DIR, VECTORIZER_PATH


class Models(NamedTuple):
    """Fitted vectorizer and classifier the predictions are made with."""

    vectorizer: BaseEstimator
    classifier: BaseEstimator


@lru_cache(maxsize=1)
def load_models() -> Models:
    """Load the vectorizer and classifier from disk on the first call only.

    Returns:
        The loaded models, the same object on every later call.

    Raises:
        FileNotFoundError: If a model file is missing.
        InconsistentVersionWarning: If a model file was saved with another
            scikit-learn version.
        ValueError: If a loaded object is not a vectorizer or a classifier.
    """
    vectorizer_path = MODEL_DIR / VECTORIZER_PATH
    classifier_path = MODEL_DIR / CLASSIFIER_PATH

    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer model not found at {vectorizer_path}. ")

    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier model not found at {classifier_path}. ")

    # A model saved with another scikit-learn version can silently give other scores
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        vectorizer = joblib.load(vectorizer_path)
        classifier = joblib.load(classifier_path)

    if not hasattr(vectorizer, "transform"):
        raise ValueError(
            "Loaded vectorizer doesn't have transform method. "
            "Please ensure the correct model file is saved.",
        )
    if not hasattr(classifier, "predict_proba"):
        raise ValueError(
            "Loaded classifier doesn't have predict_proba method. "
            "Please ensure the correct model file is saved.",
        )

    return Models(vectorizer=vectorizer, classifier=classifier)


def clean_text(text: str) -> str:
    """
    Clean the text the same way as the training data
    (experiments/notebooks/dataset/v1/transformer_dataset_preprocessing.ipynb).
    """
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r"\[[^]]*\]", "", soup.get_text())
    pattern = r"[^a-zA-z0-9\s,']"
    text = re.sub(pattern, "", text)
    return text


def predict_probability(text1: str, text2: str) -> float:
    """
    Predict the probability that two texts are written by the same author.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Probability that texts are from the same author (0.0 to 1.0)

    Raises:
        FileNotFoundError: If model files are not found
        ValueError: If loaded models are invalid
    """
    models = load_models()

    text1 = clean_text(text1)
    text2 = clean_text(text2)

    text1_embedding = models.vectorizer.transform([text1])
    text2_embedding = models.vectorizer.transform([text2])

    features = np.abs(text1_embedding - text2_embedding)

    probabilities = models.classifier.predict_proba(features)[0, 1]

    # Return probability of same author (index 1)
    same_author_probability = float(probabilities)

    return same_author_probability


def predict(text1: str, text2: str) -> dict:
    """
    Predict authorship and return probability.

    Args:
        text1: First text to compare
        text2: Second text to compare

    Returns:
        Dictionary with same_author_probability
    """
    probability = predict_probability(text1, text2)

    return {"same_author_probability": probability}
