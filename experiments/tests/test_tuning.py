"""Tests of the seeded, resumable Optuna studies on small SQLite files."""

import math
from pathlib import Path

import optuna
import pytest
from optuna.trial import TrialState

from avcore.tuning import FINISHED_STATES, open_study, run_study

optuna.logging.set_verbosity(optuna.logging.ERROR)


def _pruner() -> optuna.pruners.SuccessiveHalvingPruner:
    """The pruner of the v1 transformer notebooks."""
    return optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=3,
        min_early_stopping_rate=0,
    )


def _objective(trial: optuna.Trial) -> float:
    """Quadratic toy objective that reports 3 steps and can be pruned after step 1."""
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", 1, 5)
    value = 0.0
    for step in range(3):
        value = -((x - 2) ** 2) - y + step
        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return value


def _finished(study: optuna.Study) -> list[tuple[dict, TrialState]]:
    """Params and state of every complete or pruned trial, in trial order."""
    finished_trials = study.get_trials(states=FINISHED_STATES)
    params_and_states = []
    for trial in finished_trials:
        params_and_states.append((trial.params, trial.state))
    return params_and_states


def _crash_mid_trial(study: optuna.Study) -> None:
    """Start a trial and leave it RUNNING, as a killed pod does."""
    trial = study.ask()
    trial.suggest_float("x", -10, 10)
    trial.suggest_int("y", 1, 5)

def test_open_study_names_the_study_after_the_file_and_maximizes(
    tmp_path: Path,
) -> None:
    """The SQLite file is created, the study is named after it and maximizes."""
    study_path = tmp_path / "toy_v2.db"
    study = open_study(study_path)
    assert study_path.exists()
    assert study.study_name == "toy_v2"
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE

def test_open_study_rejects_another_seed(tmp_path: Path) -> None:
    """The seed stored on creation cannot be changed on a later open."""
    open_study(tmp_path / "toy_v2.db")
    with pytest.raises(ValueError, match="Seed 7 differs"):
        open_study(tmp_path / "toy_v2.db", seed=7)

def test_same_seed_gives_the_same_params(tmp_path: Path) -> None:
    """Two studies with the same seed suggest the same params, also after the 10 random trials."""
    first_study = open_study(tmp_path / "first_v2.db", pruner=_pruner())
    run_study(first_study, _objective, n_trials=15)
    second_study = open_study(tmp_path / "second_v2.db", pruner=_pruner())
    run_study(second_study, _objective, n_trials=15)
    assert [trial.params for trial in first_study.trials] == [
        trial.params for trial in second_study.trials
    ]

def test_run_study_counts_only_complete_and_pruned(tmp_path: Path) -> None:
    """A trial that returns NaN fails and does not count towards n_trials."""

    def objective(trial: optuna.Trial) -> float:
        value = _objective(trial)
        if trial.number == 2:
            return math.nan
        return value

    study = open_study(tmp_path / "toy_v2.db")
    run_study(study, objective, n_trials=5)
    assert len(study.get_trials(states=FINISHED_STATES)) == 5
    assert study.trials[2].state == TrialState.FAIL
    assert len(study.trials) == 6