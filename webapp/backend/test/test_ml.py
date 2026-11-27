from unittest.mock import patch

import joblib
import pytest
from core import ml
from sklearn.exceptions import InconsistentVersionWarning

OTHER_SKLEARN_VERSION = "0.0.0"


@pytest.fixture
def unloaded_models(monkeypatch):
    """Force the next call to load the model files from disk."""
    monkeypatch.setattr(ml, "_models_loaded", False)


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
        ml._load_models()


def test_load_models_accepts_same_sklearn_version(unloaded_models):
    """Model files saved with the installed scikit-learn version load."""
    ml._load_models()
    assert ml._models_loaded
