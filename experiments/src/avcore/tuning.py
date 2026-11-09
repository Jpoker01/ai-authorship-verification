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


class IgnoreFailedTrialsPruner(BasePruner):
    """Pruner that decides like the given pruner, but ignores failed trials.

    A failed trial (crashed pod, out of memory) never finished, so the values it
    reported before failing do not compete with the other trials.
    """

    def __init__(self, pruner: BasePruner) -> None:
        self._pruner = pruner

    def prune(self, study: optuna.Study, trial: FrozenTrial) -> bool:
        return self._pruner.prune(_StudyWithoutFailedTrials(study), trial)


def _position(study: optuna.Study) -> int:
    """Count the trials that are not crashes - the position of the next trial."""
    trials = study.get_trials(deepcopy=False)
    crashed_trials = [trial for trial in trials if trial.user_attrs.get("crashed")]
    return len(trials) - len(crashed_trials)


def _reseed_sampler(study: optuna.Study, trial: FrozenTrial | None = None) -> None:
    """Create the sampler for the next trial from the stored seed and the position."""
    seed = study.user_attrs["seed"] + _position(study)
    study.sampler = optuna.samplers.TPESampler(seed=seed)
