import joblib
import numpy as np

from .config import MODEL_DIR, VECTORIZER_PATH, CLASSIFIER_PATH

#module variables
_vectorizer = None
_classifier = None

# variable to track if models are loaded and prevent reloading
_models_loaded = False

def _load_models():
    """Load the TF-IDF vectorizer and classifier from disk."""
    global _vectorizer, _classifier, _models_loaded

    vectorizer_path = MODEL_DIR / VECTORIZER_PATH
    classifier_path = MODEL_DIR / CLASSIFIER_PATH

    if _models_loaded:
        return

    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Vectorizer model not found at {vectorizer_path}. "
        )

    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Classifier model not found at {classifier_path}. "
        )

    # Try loading with joblib (used in experiments)
    _vectorizer = joblib.load(vectorizer_path)
    _classifier = joblib.load(classifier_path)

    # Verify the loaded objects are correct types
    if not hasattr(_vectorizer, 'transform'):
        raise ValueError(
            "Loaded vectorizer doesn't have transform method. "
            "Please ensure the correct model file is saved."
        )
    if not hasattr(_classifier, 'predict_proba'):
        raise ValueError(
            "Loaded classifier doesn't have predict_proba method. "
            "Please ensure the correct model file is saved."
        )

    _models_loaded = True


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
    # Load models on first call - if already loaded, return from the function
    _load_models()

    text1_embedding = _vectorizer.transform([text1])
    text2_embedding  = _vectorizer.transform([text2])

    features = np.abs(text1_embedding  - text2_embedding )

    probabilities = _classifier.predict_proba(features)[0, 1]

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

    return {
        "same_author_probability": probability
    }