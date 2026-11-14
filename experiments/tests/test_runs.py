"""Tests of avcore.runs on constructed studies and folders; nothing is trained."""

import hashlib
import json
import sqlite3
from pathlib import Path

import optuna
import pandas as pd
import pytest
import sqlalchemy
from optuna.trial import TrialState, create_trial

from avcore import runs, tracking, tuning

optuna.logging.set_verbosity(optuna.logging.ERROR)

SEED = 42
SECONDS_ATTR = "training_seconds"


def build_study(trials: list[optuna.trial.FrozenTrial]) -> optuna.Study:
    """Return an in-memory study that maximizes and holds the given trials."""
    study = optuna.create_study(direction="maximize")
    for trial in trials:
        study.add_trial(trial)
    return study


def sha256_of(path: Path) -> str:
    """Return the sha256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_run_dir(run_dir: Path, params: dict[str, str]) -> None:
    """Write a result folder that holds only its params file."""
    run_dir.mkdir(parents=True)
    (run_dir / tracking.PARAMS_FILE).write_text(json.dumps(params), encoding="utf-8")
