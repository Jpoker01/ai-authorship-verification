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


def test_select_best_trial_takes_the_highest_value() -> None:
    """The complete trial with the highest value is the best trial."""
    trials = [
        create_trial(state=TrialState.COMPLETE, value=54.55),
        create_trial(state=TrialState.COMPLETE, value=57.14),
        create_trial(state=TrialState.COMPLETE, value=50.0),
    ]
    study = build_study(trials)
    assert runs.select_best_trial(study).number == 1


def test_select_best_trial_takes_the_earliest_trial_of_equal_values() -> None:
    """Of trials with the same value the one with the lowest number wins."""
    trials = [
        create_trial(state=TrialState.COMPLETE, value=54.55),
        create_trial(state=TrialState.COMPLETE, value=57.14),
        create_trial(state=TrialState.COMPLETE, value=57.14),
        create_trial(state=TrialState.COMPLETE, value=57.14),
    ]
    study = build_study(trials)
    assert runs.select_best_trial(study).number == 1


def test_select_best_trial_ignores_pruned_and_failed_trials() -> None:
    """A pruned or failed trial is never the best trial, whatever it reported."""
    trials = [
        create_trial(
            state=TrialState.PRUNED,
            value=99.0,
            intermediate_values={0: 99.0},
        ),
        create_trial(state=TrialState.FAIL, intermediate_values={0: 98.0}),
        create_trial(state=TrialState.COMPLETE, value=40.0),
    ]
    study = build_study(trials)
    assert runs.select_best_trial(study).number == 2

def test_select_best_trial_without_a_complete_trial_raises() -> None:
    """A study with pruned trials only has no best trial."""
    trials = [create_trial(state=TrialState.PRUNED, intermediate_values={0: 0.5})]
    study = build_study(trials)
    with pytest.raises(ValueError, match="holds no complete trial"):
        runs.select_best_trial(study)

def test_sum_training_seconds_adds_complete_pruned_and_failed_trials() -> None:
    """The search time is the sum over the complete, pruned and failed trials."""
    trials = [
        create_trial(state=TrialState.COMPLETE, value=1.0, user_attrs={SECONDS_ATTR: 10.0}),
        create_trial(
            state=TrialState.PRUNED,
            intermediate_values={0: 0.5},
            user_attrs={SECONDS_ATTR: 5.0},
        ),
        create_trial(state=TrialState.FAIL, user_attrs={SECONDS_ATTR: 3.0}),
    ]
    study = build_study(trials)
    assert runs.sum_training_seconds(study, SECONDS_ATTR) == 18.0

def test_sum_training_seconds_leaves_out_a_crashed_attempt() -> None:
    """The time of a crashed attempt is not added - the repeated trial holds it."""
    trials = [
        create_trial(state=TrialState.FAIL, user_attrs={SECONDS_ATTR: 7.0, "crashed": True}),
        create_trial(state=TrialState.COMPLETE, value=1.0, user_attrs={SECONDS_ATTR: 10.0}),
    ]
    study = build_study(trials)
    assert runs.sum_training_seconds(study, SECONDS_ATTR) == 10.0

def test_sum_training_seconds_leaves_out_a_running_trial() -> None:
    """A trial that is still running has not ended, so its time is not counted yet."""
    trials = [
        create_trial(state=TrialState.RUNNING, user_attrs={SECONDS_ATTR: 99.0}),
        create_trial(state=TrialState.COMPLETE, value=1.0, user_attrs={SECONDS_ATTR: 10.0}),
    ]
    study = build_study(trials)
    assert runs.sum_training_seconds(study, SECONDS_ATTR) == 10.0

def test_sum_training_seconds_counts_a_trial_without_a_time_as_zero() -> None:
    """A trial that failed before its first epoch has no stored time."""
    trials = [
        create_trial(state=TrialState.FAIL),
        create_trial(state=TrialState.COMPLETE, value=1.0, user_attrs={SECONDS_ATTR: 4.0}),
    ]
    study = build_study(trials)
    assert runs.sum_training_seconds(study, SECONDS_ATTR) == 4.0

def test_sum_training_seconds_without_any_stored_time_raises() -> None:
    """A study whose trials never stored the attribute does not report 0 seconds."""
    trials = [create_trial(state=TrialState.COMPLETE, value=1.0)]
    study = build_study(trials)
    with pytest.raises(ValueError, match="holds no trial with training_seconds"):
        runs.sum_training_seconds(study, SECONDS_ATTR)

def test_build_predictions_score_equal_to_the_threshold_is_same_author() -> None:
    """A pair is predicted as the same author when its score is at least the threshold."""
    pairs_df = pd.DataFrame({"id": ["a", "b", "c"]})
    scores_df = pd.DataFrame({"id": ["a", "b", "c"], "same": [0, 1, 1], "score": [0.3, 0.5, 0.7]})
    threshold = 0.5
    predictions_df = runs.build_predictions(pairs_df, scores_df, threshold)
    assert predictions_df["pred"].tolist() == [0, 1, 1]

def test_build_predictions_has_the_prediction_columns_and_integer_labels() -> None:
    """The table has exactly the columns of a prediction table, both labels as integers."""
    pairs_df = pd.DataFrame({"id": ["a", "b"]})
    scores_df = pd.DataFrame(
        {"id": ["a", "b"], "same": [True, False], "score": [0.9, 0.2], "argmax": [1, 0]}
    )
    threshold = 0.5
    predictions_df = runs.build_predictions(pairs_df, scores_df, threshold)
    assert list(predictions_df.columns) == tracking.PREDICTION_COLUMNS
    assert pd.api.types.is_integer_dtype(predictions_df["same"])
    assert pd.api.types.is_integer_dtype(predictions_df["pred"])
    assert predictions_df["same"].tolist() == [1, 0]
    assert predictions_df["score"].tolist() == [0.9, 0.2]