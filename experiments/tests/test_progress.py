"""Tests of avcore.progress - CPU only."""

import datetime
import json
import logging
import os

import optuna
import pytest

from avcore import progress


class FakeClock:
    """Monotonic clock that moves only when the test says so."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        """Return the seconds the test has set."""
        return self.seconds


@pytest.fixture
def clock() -> FakeClock:
    """A fake clock at 0 s."""
    return FakeClock()


@pytest.fixture
def progress_file(tmp_path, clock: FakeClock) -> progress.ProgressFile:
    """A progress file in a logs folder that does not exist yet, on the fake clock."""
    return progress.ProgressFile(
        tmp_path / "logs" / "run.progress.json", "run", clock=clock
    )


def read(progress_file: progress.ProgressFile) -> dict:
    """Return the parsed content of the progress file."""
    return json.loads(progress_file.path.read_text(encoding="utf-8"))


def test_start_stage_writes_the_run_the_stage_and_its_fields_at_once(
    progress_file: progress.ProgressFile,
):
    """A new stage is in the file before the next update, with the host and two times."""
    progress_file.start_stage("tuning", trial=3, n_trials=30)
    content = read(progress_file)
    assert content["run"] == "run"
    assert content["stage"] == "tuning"
    assert (content["trial"], content["n_trials"]) == (3, 30)
    assert {"host", "stage_started_at", "updated_at"} <= content.keys()


def test_update_writes_only_after_min_seconds(
    progress_file: progress.ProgressFile, clock: FakeClock
):
    """An update 29 s after the last write is kept in memory, the one at 30 s is written."""
    progress_file.start_stage("tuning")
    clock.seconds = 29.0
    progress_file.update(step=1)
    assert "step" not in read(progress_file)
    clock.seconds = 30.0
    progress_file.update(step=2)
    assert read(progress_file)["step"] == 2


def test_update_with_force_writes_at_once(
    progress_file: progress.ProgressFile, clock: FakeClock
):
    """force=True writes even right after the last write."""
    progress_file.start_stage("tuning")
    clock.seconds = 1.0
    progress_file.update(force=True, step=5)
    assert read(progress_file)["step"] == 5


def test_start_stage_drops_the_fields_of_the_previous_stage(
    progress_file: progress.ProgressFile,
):
    """The epoch of the tuning is not shown in the test scoring."""
    progress_file.start_stage("tuning", trial=3)
    progress_file.update(force=True, epoch=2)
    progress_file.start_stage("test scoring", n_pairs=10)
    content = read(progress_file)
    assert "epoch" not in content and "trial" not in content
    assert content["n_pairs"] == 10

def test_the_file_is_replaced_and_no_temporary_file_is_left(
    progress_file: progress.ProgressFile, monkeypatch
):
    """The text goes to a temporary file that os.replace moves over the progress file."""
    targets = []
    real_replace = os.replace

    def recording_replace(source, target):
        targets.append(target)
        real_replace(source, target)

    monkeypatch.setattr(progress.os, "replace", recording_replace)
    progress_file.start_stage("tuning")
    progress_file.update(force=True, step=1)
    assert targets == [progress_file.path, progress_file.path]
    assert [path.name for path in progress_file.path.parent.iterdir()] == [
        "run.progress.json"
    ]


def test_the_temporary_file_carries_the_host_and_the_process(
    progress_file: progress.ProgressFile, monkeypatch
):
    """Two pods with the same process id never write into the same temporary file."""
    sources = []
    real_replace = os.replace

    def recording_replace(source, target):
        sources.append(source.name)
        real_replace(source, target)

    monkeypatch.setattr(
        progress.socket, "gethostname", lambda: "train-longformer-ab12c"
    )
    monkeypatch.setattr(progress.os, "replace", recording_replace)
    progress_file.start_stage("tuning")
    assert sources == [f"run.progress.json.train-longformer-ab12c.{os.getpid()}.tmp"]
    assert read(progress_file)["host"] == "train-longformer-ab12c"


def test_the_times_are_utc_with_their_offset(progress_file: progress.ProgressFile):
    """The pod and dt may run in other time zones - every time carries +00:00."""
    progress_file.start_stage("tuning")
    content = read(progress_file)
    assert content["stage_started_at"].endswith("+00:00")
    assert content["updated_at"].endswith("+00:00")


def test_a_failed_write_is_logged_once_and_the_run_goes_on(
    progress_file: progress.ProgressFile, monkeypatch, caplog
):
    """Two failed writes give one warning and no exception."""

    def fail(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(progress.os, "replace", fail)
    with caplog.at_level(logging.WARNING, logger="avcore.progress"):
        progress_file.start_stage("tuning")
        progress_file.update(force=True, step=1)
    assert len(caplog.records) == 1
    assert "disk full" in caplog.records[0].getMessage()


def test_a_write_after_failed_writes_is_logged(
    progress_file: progress.ProgressFile, monkeypatch, caplog
):
    """A failed write, then a working one - a warning for each, the file holds the last state."""

    def fail(source, target):
        raise OSError("disk full")

    with caplog.at_level(logging.WARNING, logger="avcore.progress"):
        # the first write fails
        monkeypatch.setattr(progress.os, "replace", fail)
        progress_file.start_stage("tuning")
        # the volume works again - the real os.replace is back
        monkeypatch.undo()
        progress_file.update(force=True, step=2)

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert "not written" in messages[0]
    assert messages[1].endswith("written again.")
    assert read(progress_file)["step"] == 2

def test_a_value_json_can_not_hold_raises(progress_file: progress.ProgressFile):
    """A wrong field stops the run in its first stage, not silently in the file."""
    with pytest.raises(TypeError):
        progress_file.start_stage("tuning", trial=object())


def test_trial_fields_count_the_finished_trials_and_their_mean_time():
    """Two finished trials of 10 s and 30 s give finished_trials 2 and a mean of 20 s; a failed trial is not counted."""
    study = optuna.create_study()
    start = datetime.datetime(2026, 11, 17, 12, 0, 0)
    for seconds, state in [(10, "COMPLETE"), (30, "PRUNED"), (99, "FAIL")]:
        finished = optuna.trial.create_trial(
            state=optuna.trial.TrialState[state],
            value=0.5 if state == "COMPLETE" else None,
        )
        finished.datetime_start = start
        finished.datetime_complete = start + datetime.timedelta(seconds=seconds)
        study.add_trial(finished)
    trial = study.ask()
    fields = progress.trial_fields(trial, n_trials=30)
    assert fields == {
        "trial": 3,
        "finished_trials": 2,
        "n_trials": 30,
        "mean_trial_seconds": 20.0,
    }


def test_trial_fields_of_the_first_trial_have_no_mean_time():
    """Before the first finished trial the mean is None, not a division by zero."""
    trial = optuna.create_study().ask()
    assert progress.trial_fields(trial, n_trials=30)["mean_trial_seconds"] is None
