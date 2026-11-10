"""Training and scoring of the transformer classification heads, resumable after a crash.

A Kubernetes Job pod that is restarted runs the notebook from the top again. The trial that
was running is repeated with the same hyperparameters (tuning.py), and train_classifier
continues it from its last checkpoint instead of from the first step.
"""

import hashlib
import logging
import os
import pickle
import time
import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.optimization import get_linear_schedule_with_warmup

from avcore.data import RANDOM_STATE

logger = logging.getLogger(__name__)

# v1 limit of all three transformers (ModernBERT could take 8 192)
MAX_LENGTH = 4096
CHECKPOINT_FILE = "checkpoint.pt"
CHECKPOINT_MINUTES = 60

@dataclass(frozen=True)
class TrainingSettings:
    """Hyperparameters of one training run (one Optuna trial or the final training)."""

    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    warmup_ratio: float
    seed: int = RANDOM_STATE


@dataclass(frozen=True)
class TrainingResult:
    """What a training run reports.

    Attributes:
        epoch_losses: Mean training loss of every epoch.
        pruning_f1s: Weighted F1 at argmax on validation after every epoch (empty without validation).
        validation_scores_df: score_pairs output of the last epoch, or None without validation.
    """

    epoch_losses: list[float]
    pruning_f1s: list[float]
    validation_scores_df: pd.DataFrame | None

def checkpoint_key_for(trial: optuna.Trial) -> str:
    """Name the checkpoint of a trial by its study and position, which a repeated trial keeps.

    Raises:
        ValueError: If the trial has no position, i.e. it was not started by tuning.run_study.
    """
    if "position" not in trial.user_attrs:
        raise ValueError("The trial has no position - start the study with tuning.run_study.")
    # trial.number changes when a crashed trial is repeated, the position does not
    return f"{trial.study.study_name}-position-{trial.user_attrs['position']}"