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
    trial = study.ask()
    trial.suggest_float("x", -10, 10)
    trial.suggest_int("y", 1, 5)
