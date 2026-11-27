from unittest.mock import patch

import joblib
import pytest
from core import ml
from fastapi.testclient import TestClient
from main import app
from sklearn.exceptions import InconsistentVersionWarning

OTHER_SKLEARN_VERSION = "0.0.0"


@pytest.fixture
def unloaded_models() -> None:
    """Force the next call to load the model files from disk."""
    ml.load_models.cache_clear()


def test_load_models_accepts_same_sklearn_version(unloaded_models):
    """Model files saved with the installed scikit-learn version load."""
    models = ml.load_models()
    assert isinstance(models, ml.Models)


@pytest.mark.parametrize("path_name", ["VECTORIZER_PATH", "CLASSIFIER_PATH"])
def test_load_models_refuses_each_file_from_other_sklearn_version(
    path_name,
    tmp_path,
    monkeypatch,
    unloaded_models,
):
    """Either model file saved with another scikit-learn version is refused."""
    real_path = getattr(ml, path_name)
    model = joblib.load(real_path)
    other_version_path = tmp_path / real_path.name
    with patch("sklearn.base.__version__", OTHER_SKLEARN_VERSION):
        joblib.dump(model, other_version_path)
    monkeypatch.setattr(ml, path_name, other_version_path)
    with pytest.raises(InconsistentVersionWarning):
        ml.load_models()


def test_load_models_returns_the_same_models_on_every_call(unloaded_models) -> None:
    """The model files are loaded only once, later calls return the same models."""
    first_models = ml.load_models()
    second_models = ml.load_models()
    assert second_models is first_models


def test_startup_loads_the_models() -> None:
    """The models are loaded when the application starts, not on the first request."""
    with patch("core.ml.load_models") as mock_load_models, TestClient(app):
        mock_load_models.assert_called_once_with()


def test_startup_fails_when_the_models_cannot_be_loaded() -> None:
    """A model file that cannot be loaded stops the application before the first request."""
    with (
        patch("core.ml.load_models", side_effect=FileNotFoundError),
        pytest.raises(FileNotFoundError),
        TestClient(app),
    ):
        pass


def test_predict_probability_uses_the_cached_models() -> None:
    """A prediction takes the models from load_models instead of reading the files again."""
    text = "the cat sat on the mat and looked at the dog " * 20
    with patch("core.ml.load_models", wraps=ml.load_models) as spy_load_models:
        ml.predict_probability(text, text)
    spy_load_models.assert_called_once_with()
