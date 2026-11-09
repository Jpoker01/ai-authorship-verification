"""Seeded, resumable Optuna studies shared by all notebooks.

Every study lives in its own SQLite file, runs one trial at a time and stops at
N_TRIALS finished (complete + pruned) trials. A study killed in the middle of a
trial resumes with the same suggestions as an uninterrupted run.
"""

from collections.abc import Callable
from pathlib import Path

import optuna
from optuna.pruners import BasePruner
from optuna.study import MaxTrialsCallback, StudyDirection
from optuna.trial import FrozenTrial, TrialState

from avcore.data import RANDOM_STATE

N_TRIALS = 30
FINISHED_STATES = (TrialState.COMPLETE, TrialState.PRUNED)
NOT_FAILED_STATES = tuple(state for state in TrialState if state != TrialState.FAIL)


class _StudyWithoutFailedTrials:
    """Read-only view of a study that hides its failed trials from a pruner."""

    def __init__(self, study: optuna.study.Study) -> None:
        self._study = study

    def __getattr__(self, name: str) -> object:
        return getattr(self._study, name)

    def get_trials(
        self,
        deepcopy: bool = True,
        states: tuple[TrialState, ...] | None = None,
    ) -> list[FrozenTrial]:
        if states is None:
            states = NOT_FAILED_STATES
        return self._study.get_trials(deepcopy=deepcopy, states=states)
