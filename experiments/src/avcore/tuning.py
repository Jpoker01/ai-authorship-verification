"""Seeded, resumable Optuna studies shared by all notebooks.

Every study lives in its own SQLite file, runs one trial at a time and stops at
N_TRIALS finished (complete + pruned) trials. A study killed in the middle of a
trial resumes with the same suggestions as an uninterrupted run.
"""

from collections.abc import Callable
from pathlib import Path

import optuna
from optuna.pruners import BasePruner, HyperbandPruner
from optuna.study import MaxTrialsCallback, StudyDirection
from optuna.trial import FrozenTrial, TrialState

from avcore.data import RANDOM_STATE

N_TRIALS = 30
FINISHED_STATES = (TrialState.COMPLETE, TrialState.PRUNED)
ENDED_STATES = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
NOT_FAILED_STATES = (
    TrialState.RUNNING,
    TrialState.COMPLETE,
    TrialState.PRUNED,
    TrialState.WAITING,
)


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
        # get all trials that are not FAILED
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
    """Count the ended trials that are not crashes - the position of the next trial."""
    ended_trials = study.get_trials(deepcopy=False, states=ENDED_STATES)
    crashed_trials = [
        trial for trial in ended_trials if trial.user_attrs.get("crashed")
    ]
    return len(ended_trials) - len(crashed_trials)


def _reseed_sampler(study: optuna.Study, trial: FrozenTrial | None = None) -> None:
    """Create the sampler for the next trial from the stored seed and the position."""
    seed = study.user_attrs["seed"] + _position(study)
    study.sampler = optuna.samplers.TPESampler(seed=seed)


def open_study(
    study_path: Path,
    pruner: BasePruner | None = None,
    seed: int = RANDOM_STATE,
) -> optuna.Study:
    """Create or load the study stored in study_path and fail its crashed trials.

    Args:
        study_path: Absolute path of the SQLite file; its stem is the study name.
        pruner: Pruner of the study, not Hyperband; failed trials are hidden from it.
        seed: Seed of the first trial, stored in the study on creation.

    Returns:
        The study, maximizing, with the sampler for its next trial.

    Raises:
        ValueError: If study_path is relative, pruner is a HyperbandPruner, the
            loaded study does not maximize, or seed differs from the stored one.
        FileNotFoundError: If the folder of study_path does not exist.
    """
    if not study_path.is_absolute():
        raise ValueError(f"{study_path} is not an absolute path.")
    if not study_path.parent.is_dir():
        raise FileNotFoundError(f"{study_path.parent} does not exist.")
    # its brackets follow the trial number, which a retried crash shifts
    if isinstance(pruner, HyperbandPruner):
        raise ValueError("HyperbandPruner does not resume a crashed study correctly.")
    if pruner is not None:
        pruner = IgnoreFailedTrialsPruner(pruner)
    storage = optuna.storages.RDBStorage(f"sqlite:///{study_path}")
    study = optuna.create_study(
        study_name=study_path.stem,
        storage=storage,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    if study.direction != StudyDirection.MAXIMIZE:
        raise ValueError(f"Study {study_path.stem} does not maximize.")

    stored_seed = study.user_attrs.get("seed")
    if stored_seed is None:
        study.set_user_attr("seed", seed)
    elif stored_seed != seed:
        raise ValueError(f"Seed {seed} differs from the stored seed {stored_seed}.")

    # one process per study, so a trial still running at open is a crashed one
    study_id = storage.get_study_id_from_name(study_path.stem)
    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        trial_id = storage.get_trial_id_from_study_id_trial_number(
            study_id,
            trial.number,
        )
        storage.set_trial_user_attr(trial_id, "crashed", True)
        study.tell(trial.number, state=TrialState.FAIL)
    _reseed_sampler(study)
    return study


def _raise_if_too_many_failures(study: optuna.Study, n_trials: int) -> None:
    """Raise when n_trials trials failed without a crash - the study can not finish."""
    failed_trials = study.get_trials(deepcopy=False, states=(TrialState.FAIL,))
    failures = [trial for trial in failed_trials if not trial.user_attrs.get("crashed")]
    if len(failures) >= n_trials:
        raise RuntimeError(
            f"Study {study.study_name}: {len(failures)} trials failed without a crash.",
        )


def run_study(
    study: optuna.Study,
    objective: Callable[[optuna.Trial], float],
    n_trials: int = N_TRIALS,
    catch: tuple[type[Exception], ...] = (),
) -> None:
    """Run trials one at a time until the study has n_trials finished trials.

    Args:
        study: Study returned by open_study.
        objective: Function that trains and evaluates one trial.
        n_trials: Number of complete + pruned trials to reach.
        catch: Exceptions that fail one trial and let the study continue.

    Raises:
         RuntimeError: If the study holds n_trials failed trials that are not
            crashes (caught errors, NaN values), also from earlier runs.
    """
    finished_trials = study.get_trials(deepcopy=False, states=FINISHED_STATES)
    # optimize would start one more trial before the callback stops it
    if len(finished_trials) >= n_trials:
        return
    _raise_if_too_many_failures(study, n_trials)

    # used only to store the correct slot number in optuna database
    def objective_with_position(trial: optuna.Trial) -> float:
        trial.set_user_attr("position", _position(study))
        try:
            return objective(trial)  # run the training as before
        except (optuna.TrialPruned, *catch):
            raise  # pruned, or an error you listed in catch: unchanged
        except BaseException:
            trial.set_user_attr("crashed", True)  # anything else: mark it as a crash
            raise

    # without it, a search space that always runs out of memory loops forever
    def stop_after_failed_trials(study: optuna.Study, trial: FrozenTrial) -> None:
        _raise_if_too_many_failures(study, n_trials)

    study.optimize(
        objective_with_position,
        n_jobs=1,
        catch=catch,
        callbacks=[
            MaxTrialsCallback(n_trials, states=FINISHED_STATES),
            _reseed_sampler,
            stop_after_failed_trials,
        ],
    )
