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


def test_open_study_rejects_a_relative_path() -> None:
    """A relative path would create another study in another working directory."""
    with pytest.raises(ValueError, match="not an absolute path"):
        open_study(Path("toy_v2.db"))


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


def test_run_study_again_adds_no_trials(tmp_path: Path) -> None:
    """A finished study run again (a re-run of the notebook) gets no new trials."""
    run_study(open_study(tmp_path / "toy_v2.db"), _objective, n_trials=5)
    study = open_study(tmp_path / "toy_v2.db")
    run_study(study, _objective, n_trials=5)
    assert len(study.trials) == 5


def test_crash_mid_trial_repeats_the_params_at_the_same_position(
    tmp_path: Path,
) -> None:
    """A trial killed mid-run fails and is repeated; the study ends as an uninterrupted one."""
    uninterrupted = open_study(tmp_path / "uninterrupted_v2.db", pruner=_pruner())
    run_study(uninterrupted, _objective, n_trials=15)
    study = open_study(tmp_path / "toy_v2.db", pruner=_pruner())
    run_study(study, _objective, n_trials=12)
    _crash_mid_trial(study)
    study = open_study(tmp_path / "toy_v2.db", pruner=_pruner())
    run_study(study, _objective, n_trials=15)
    crashed_trial = study.trials[12]
    repeated_trial = study.trials[13]
    assert crashed_trial.state == TrialState.FAIL
    assert crashed_trial.user_attrs["crashed"] is True
    assert repeated_trial.params == crashed_trial.params
    assert repeated_trial.user_attrs["position"] == 12
    assert _finished(study) == _finished(uninterrupted)


def test_resume_after_a_finished_trial_repeats_nothing(tmp_path: Path) -> None:
    """A study stopped between trials and opened again continues as an uninterrupted one."""
    uninterrupted = open_study(tmp_path / "uninterrupted_v2.db", pruner=_pruner())
    run_study(uninterrupted, _objective, n_trials=15)
    run_study(
        open_study(tmp_path / "toy_v2.db", pruner=_pruner()),
        _objective,
        n_trials=12,
    )
    study = open_study(tmp_path / "toy_v2.db", pruner=_pruner())
    run_study(study, _objective, n_trials=15)
    assert _finished(study) == _finished(uninterrupted)


def test_caught_error_fails_one_trial_and_continues(tmp_path: Path) -> None:
    """An error in catch fails only its trial, and the next trial gets new params."""

    def objective(trial: optuna.Trial) -> float:
        if trial.number == 3:
            trial.suggest_float("x", -10, 10)
            trial.suggest_int("y", 1, 5)
            raise MemoryError("out of memory")
        return _objective(trial)

    study = open_study(tmp_path / "toy_v2.db", pruner=_pruner())
    run_study(study, objective, n_trials=8, catch=(MemoryError,))
    assert len(study.get_trials(states=FINISHED_STATES)) == 8
    assert study.trials[3].state == TrialState.FAIL
    assert study.trials[4].params != study.trials[3].params


def test_failed_trial_does_not_take_part_in_pruning(tmp_path: Path) -> None:
    """A failed trial's high step value does not prune the trials after it."""
    study = open_study(tmp_path / "toy_v2.db", pruner=_pruner())
    failed_trial = study.ask()
    failed_trial.report(10.0, 1)
    failed_trial.should_prune()
    study.tell(failed_trial, state=TrialState.FAIL)
    should_prune = []
    for value in (1.0, 2.0):
        trial = study.ask()
        trial.report(value, 1)
        should_prune.append(trial.should_prune())
        study.tell(trial, value)
    assert should_prune == [False, False]


def test_interrupted_trial_repeats_the_params_at_the_same_position(
    tmp_path: Path,
) -> None:
    """A trial stopped by an error outside catch is repeated; the study ends as an uninterrupted one."""
    uninterrupted = open_study(tmp_path / "uninterrupted_v2.db", pruner=_pruner())
    run_study(uninterrupted, _objective, n_trials=15)
    interrupted_numbers = []

    def interrupted_objective(trial: optuna.Trial) -> float:
        if trial.number == 5 and not interrupted_numbers:
            interrupted_numbers.append(trial.number)
            trial.suggest_float("x", -10, 10)
            trial.suggest_int("y", 1, 5)
            raise KeyboardInterrupt
        return _objective(trial)

    study = open_study(tmp_path / "toy_v2.db", pruner=_pruner())
    with pytest.raises(KeyboardInterrupt):
        run_study(study, interrupted_objective, n_trials=15)
    study = open_study(tmp_path / "toy_v2.db", pruner=_pruner())
    run_study(study, interrupted_objective, n_trials=15)
    assert study.trials[5].user_attrs["crashed"] is True
    assert study.trials[6].params == study.trials[5].params
    assert _finished(study) == _finished(uninterrupted)


def test_open_study_rejects_a_minimizing_study(tmp_path: Path) -> None:
    """A loaded study that minimizes would rank the trials the wrong way round."""
    study_path = tmp_path / "toy_v2.db"
    optuna.create_study(
        study_name=study_path.stem,
        storage=f"sqlite:///{study_path}",
        direction="minimize",
    )
    with pytest.raises(ValueError, match="does not maximize"):
        open_study(study_path)


def test_open_study_rejects_a_missing_folder(tmp_path: Path) -> None:
    """A missing folder fails with a clear error instead of a SQLite one."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        open_study(tmp_path / "missing" / "toy_v2.db")


def test_run_study_stops_after_n_trials_failed_trials(tmp_path: Path) -> None:
    """Failed trials stop the study at n_trials, also when it is run again."""

    def failing_objective(trial: optuna.Trial) -> float:
        if trial.number < 10:
            raise MemoryError
        return _objective(trial)

    study = open_study(tmp_path / "toy_v2.db", pruner=_pruner())
    with pytest.raises(RuntimeError, match="^Study toy_v2: 3 trials failed"):
        run_study(study, failing_objective, n_trials=3, catch=(MemoryError,))
    study = open_study(tmp_path / "toy_v2.db", pruner=_pruner())
    with pytest.raises(RuntimeError, match="^Study toy_v2: 3 trials failed"):
        run_study(study, failing_objective, n_trials=3, catch=(MemoryError,))
    assert [trial.state for trial in study.trials] == [TrialState.FAIL] * 3


def test_only_caught_errors_count_as_failed_trials(tmp_path: Path) -> None:
    """Two crashes, one caught error and two pruned trials stay below the limit of 3."""
    study = open_study(tmp_path / "toy_v2.db", pruner=_pruner())
    _crash_mid_trial(study)
    study = open_study(tmp_path / "toy_v2.db", pruner=_pruner())
    _crash_mid_trial(study)
    study = open_study(tmp_path / "toy_v2.db", pruner=_pruner())
    calls = []

    def objective_failing_then_pruned(trial: optuna.Trial) -> float:
        calls.append(trial.number)
        if len(calls) == 1:
            raise MemoryError
        if len(calls) <= 3:
            raise optuna.TrialPruned()
        return _objective(trial)

    run_study(study, objective_failing_then_pruned, n_trials=3, catch=(MemoryError,))
    assert len(_finished(study)) == 3
