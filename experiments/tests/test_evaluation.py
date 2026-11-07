import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from avcore.evaluation import (
    ClassificationMetrics,
    evaluate_results,
    evaluate_classification_thresholds,
    select_best_threshold
)

Y_TRUE = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
Y_PRED = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 0])

def test_evaluate_results_matches_sklearn() -> None:
    """Every metric equals the sklearn value in percent, rounded to 2 decimals."""
    metrics = evaluate_results(Y_TRUE, Y_PRED)
    assert metrics.accuracy == round(accuracy_score(Y_TRUE, Y_PRED) * 100, 2)
    assert metrics.precision == round(precision_score(Y_TRUE, Y_PRED) * 100, 2)
    assert metrics.recall == round(recall_score(Y_TRUE, Y_PRED) * 100, 2)
    assert metrics.f1 == round(f1_score(Y_TRUE, Y_PRED) * 100, 2)

def test_evaluate_results_fields_are_named() -> None:
    """The fields are read by name, so accuracy and F1 cannot be swapped by unpacking."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 1])  # accuracy 50, f1 66.67
    metrics = evaluate_results(y_true, y_pred)
    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.accuracy == 50.0
    assert metrics.f1 == 66.67

def test_evaluate_results_no_positive_prediction_gives_zero() -> None:
    """Precision, recall and F1 are 0 when nothing is predicted as same author, without a warning."""
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0, 0, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = evaluate_results(y_true, y_pred)
    assert metrics.precision == 0
    assert metrics.recall == 0
    assert metrics.f1 == 0

def test_evaluate_results_no_same_author_pair_gives_zero() -> None:
    """Recall and F1 are 0 when y_true has no same-author pair, without a warning."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = evaluate_results(y_true, y_pred)
    assert metrics.recall == 0
    assert metrics.f1 == 0

def test_evaluate_results_returns_plain_floats() -> None:
    """The fields are plain floats, not np.float64."""
    metrics = evaluate_results(Y_TRUE, Y_PRED)
    assert all(type(value) is float for value in metrics.to_dict().values())

def test_evaluate_results_accepts_bool_labels() -> None:
    """Bool y_true (the 'same' column) and int y_pred give the same result as ints."""
    metrics = evaluate_results(Y_TRUE, Y_PRED)
    metrics_bool = evaluate_results(Y_TRUE.astype(bool), Y_PRED)
    assert metrics == metrics_bool

def test_evaluate_results_rejects_length_mismatch() -> None:
    """Inputs of different length raise ValueError."""
    y_true = np.array([1])
    y_pred = np.array([0, 0, 1])
    with pytest.raises(ValueError):
        evaluate_results(y_true, y_pred)


