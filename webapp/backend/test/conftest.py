import json
import random
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import app


@pytest.fixture
def client():
    """
    Create a test client for the FastAPI application.
    """
    return TestClient(app)


@pytest.fixture
def mock_predict():
    """
    Mack the predict function from predict router to avoid loading the actual model
    """
    with patch('core.ml.predict') as mock_predict:
        yield mock_predict

@pytest.fixture
def sample_text():
    filepath = Path(__file__).parent / "data" / "five_samples.json"
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample = random.choice(data)
    sample_text_key = random.choice(["text1", "text2"])

    return {
        "text": sample[sample_text_key],
    }

@pytest.fixture
def sample_texts():
    filepath = Path(__file__).parent / "data" / "five_samples.json"
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample = random.choice(data)

    return {
        "text1": sample["text1"],
        "text2": sample["text2"]
    }
