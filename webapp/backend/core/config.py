from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"
VECTORIZER_PATH = MODEL_DIR  / "vectorizer.pkl"
CLASSIFIER_PATH = MODEL_DIR  / "classifier.pkl"

MAX_TEXT_LENGTH = 50_000
MIN_TEXT_LENGTH = 5_000