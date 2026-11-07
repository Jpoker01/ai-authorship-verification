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