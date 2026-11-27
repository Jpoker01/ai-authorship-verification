from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
CLASSIFIER_PATH = MODEL_DIR / "classifier.pkl"

MAX_TEXT_LENGTH = 50_000
MIN_TEXT_LENGTH = 5_000

# Python's html.parser slows down quadratically with the number of unclosed tags
MAX_LESS_THAN_SIGNS = 300
# Python's re scans from every "[" to the next "]", so unclosed "[" cost quadratic time
MAX_OPENING_BRACKETS = 300
