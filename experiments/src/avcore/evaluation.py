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