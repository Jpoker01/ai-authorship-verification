"""Evaluation metrics and the classification threshold search shared by all notebooks."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1"]