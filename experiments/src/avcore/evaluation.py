"""Evaluation metrics and the classification threshold search shared by all notebooks."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1"]


@dataclass(frozen=True)
class ClassificationMetrics:
    """Classification metrics in percentage, rounded to two decimal places."""

    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        """Return the metrics as a dictionary which will then be consumed by MLflow or DataFrame row."""
        return asdict(self)


def _to_percent(value: float) -> float:
    """Convert the fraction value to percentage rounded to two decimal places"""
    return float(round(value * 100, 2))


def evaluate_results(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    average: str = "binary",
) -> ClassificationMetrics:
    """Compute accuracy, precision, recall and f1 in percentage.

    Args:
        y_true: True labels, 0 / False = different author, 1 / True = same author.
        y_pred: Predicted labels in the same encoding.
        average: sklearn averaging; "binary" scores the positive class (same author).

    Returns:
        The four metrics in percentage, rounded to two decimal places.

    Raises:
        ValueError: If the inputs are empty or of different length.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        raise ValueError("y_true is an empty array")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true has shape {y_true.shape}, and y_pred has shape {y_pred.shape}"
        )

    return ClassificationMetrics(
        accuracy=_to_percent(accuracy_score(y_true, y_pred)),
        precision=_to_percent(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        recall=_to_percent(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        f1=_to_percent(f1_score(y_true, y_pred, average=average, zero_division=0)),
    )

def evaluate_classification_thresholds(
    y_true: npt.ArrayLike,
    scores: npt.ArrayLike,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Evaluate every threshold, predicting "same author" when score >= threshold.

    Run it on validation data only - the selected threshold is then applied to test unchanged.

    Args:
        y_true: True labels, 0 / False = different author, 1 / True = same author.
        scores: Same-author scores (probability or cosine similarity), higher = more likely the same author.
        thresholds: Thresholds to evaluate, in the order of the output rows.

    Returns:
        One row per threshold with the columns threshold, accuracy, precision, recall, f1
        (percent, rounded) and f1_unrounded (fraction, used only to rank the thresholds).

    Raises:
        ValueError: If thresholds is empty, y_true, scores or thresholds contain NaN,
            or y_true and scores differ in length.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    thresholds_array = np.asarray(thresholds, dtype=float)
    if thresholds_array.size == 0:
        raise ValueError("thresholds is empty.")
    for name, values in [
        ("y_true", y_true),
        ("scores", scores),
        ("thresholds", thresholds_array),
    ]:
        if pd.isna(values).any():
            raise ValueError(f"{name} contains NaN.")

    rows = []
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        metrics = evaluate_results(y_true, y_pred)
        rows.append(
            {
                "threshold": threshold,
                **metrics.to_dict(),
                "f1_unrounded": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows, columns=["threshold", *METRIC_COLUMNS, "f1_unrounded"])