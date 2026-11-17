"""Tests of avcore.progress - CPU only."""

import json
import logging
import os

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
    """A failure, then a working write - one warning for each, the file holds the last state."""
    real_replace = os.replace

    def fail(source, target):
        raise OSError("disk full")

    with caplog.at_level(logging.WARNING, logger="avcore.progress"):
        monkeypatch.setattr(progress.os, "replace", fail)
        progress_file.start_stage("tuning")
        monkeypatch.setattr(progress.os, "replace", real_replace)
        progress_file.update(force=True, step=2)
    assert [
        record.getMessage().endswith("written again.") for record in caplog.records
    ] == [False, True]
    assert read(progress_file)["step"] == 2