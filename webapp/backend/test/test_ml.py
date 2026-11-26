from unittest.mock import patch

import pytest
from core import ml
from sklearn.exceptions import InconsistentVersionWarning


@pytest.fixture
def unloaded_models(monkeypatch):
    """Force the next call to load the model files from disk."""
    monkeypatch.setattr(ml, "_models_loaded", False)


def test_load_models_refuses_other_sklearn_version(unloaded_models):
    """Model files saved with another scikit-learn version are refused."""
    other_version = "0.0.0"
    with (
        patch("sklearn.base.__version__", other_version),
        pytest.raises(InconsistentVersionWarning),
    ):
        ml._load_models()


def test_load_models_accepts_same_sklearn_version(unloaded_models):
    """Model files saved with the installed scikit-learn version load."""
    ml._load_models()
    assert ml._models_loaded
