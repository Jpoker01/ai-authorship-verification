import re
import warnings
from functools import lru_cache
from typing import NamedTuple

import joblib
import numpy as np
from bs4 import BeautifulSoup
from bs4.exceptions import ParserRejectedMarkup
from sklearn.base import BaseEstimator
from sklearn.exceptions import InconsistentVersionWarning

from .config import (
    CLASSIFIER_PATH,
    MAX_LESS_THAN_SIGNS,
    MAX_OPENING_BRACKETS,
    MODEL_DIR,
    VECTORIZER_PATH,
)

# [^]] = any character except "]", so every [...] block is removed
_BRACKET_BLOCK = re.compile(r"\[[^]]*\]")
_NOT_KEPT = re.compile(r"[^a-zA-Z0-9\s,']")


class UnsupportedTextError(ValueError):
    """Raised when a submitted text cannot be turned into input for the model."""


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
    """Remove HTML markup, [...] blocks and every character outside the kept set.

    Args:
        text: One raw PAN text.

    Returns:
        The text with only ASCII letters, digits, whitespace, commas and apostrophes.
    """
    text_html_parsed = BeautifulSoup(text, "html.parser").get_text()
    text_no_brackets = _BRACKET_BLOCK.sub("", text_html_parsed)
    text_desired_characters = _NOT_KEPT.sub("", text_no_brackets)
    return text_desired_characters


def clean_submitted_text(text: str) -> str:
    """Clean one submitted text, refusing input that makes the cleaning very slow.

    Args:
        text: One text sent to the API.

    Returns:
        The cleaned text.

    Raises:
        UnsupportedTextError: If the text has more than MAX_LESS_THAN_SIGNS "<",
            markup the parser rejects or more than MAX_OPENING_BRACKETS "[".
    """
    # Counted before parsing, so a text full of unclosed tags never reaches the parser
    if text.count("<") > MAX_LESS_THAN_SIGNS:
        raise UnsupportedTextError(
            f'The text has more than {MAX_LESS_THAN_SIGNS} "<" characters.',
        )
    try:
        text_html_parsed = BeautifulSoup(text, "html.parser").get_text()
    except ParserRejectedMarkup as error:
        raise UnsupportedTextError(
            "The text has markup that cannot be parsed.",
        ) from error
    # Counted after parsing, because entities such as &#91; turn into "[" there
    if text_html_parsed.count("[") > MAX_OPENING_BRACKETS:
        raise UnsupportedTextError(
            f'The text has more than {MAX_OPENING_BRACKETS} "[" characters.',
        )
    return clean_text(text)


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
        UnsupportedTextError: If a text cannot be cleaned safely
    """
    models = load_models()

    text1 = clean_submitted_text(text1)
    text2 = clean_submitted_text(text2)

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
