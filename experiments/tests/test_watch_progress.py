"""Tests of the live view experiments/k8s/watch_progress.py - CPU only, no kubectl."""

import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

# experiments/k8s/watch_progress.py - k8s is a folder, not an installed package
WATCHER_PATH = Path(__file__).parents[1] / "k8s" / "watch_progress.py"


def load_watcher():
    """Import the watcher script by its path, as a module named watch_progress."""
    spec = importlib.util.spec_from_file_location("watch_progress", WATCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


watch_progress = load_watcher()

# a fixed "now" (17.11.2026 12:00 UTC, in seconds) so no test depends on the clock
NOW = datetime(2026, 11, 17, 12, 0, 0, tzinfo=UTC).timestamp()
# the pod the fake progress files name as their host
POD = "train-longformer-ab12c"


def at(seconds):
    """ISO time seconds after NOW, as the progress file writes it."""
    return datetime.fromtimestamp(NOW + seconds, UTC).isoformat(timespec="seconds")


def training(step, updated=0, **fields):
    """A progress file of the tuning, trial 7, epoch 2 of 3, 100 steps per epoch."""
    return {
        "run": "run",
        "host": POD,
        "stage": "tuning",
        "stage_started_at": at(-600),
        "trial": 7,
        "finished_trials": 5,
        "n_trials": 30,
        "mean_trial_seconds": 3600.0,
        "epoch": 2,
        "n_epochs": 3,
        "step": step,
        "n_steps": 100,
        "loss": 0.5,
        "updated_at": at(updated),
        **fields,
    }

RUNNING = {POD: "Running"}


def rows_of(progress, pods=RUNNING):
    """The rows of one progress file at NOW, with a fresh speed measurement."""
    return watch_progress.describe(progress, pods, watch_progress.SpeedTracker(), NOW)


def lines_of(progress, pods=RUNNING):
    """The texts of the rows of one progress file."""
    return [row.text for row in rows_of(progress, pods)]


def test_the_training_lines_show_the_pod_the_trial_and_the_epoch():
    """The pod phase, the finished trials, the step bar and the time since the update are shown."""
    text = "\n".join(lines_of(training(50, updated=-120)))
    assert f"pod {POD} Running" in text
    assert "trial 7, 5 of 30 finished" in text
    assert "epoch 2 / 3, step 50 / 100" in text
    assert "updated 2min ago" in text


def test_the_last_step_of_a_trial_epoch_shows_the_validation_scoring():
    """step == n_steps in the tuning means the run scores the validation pairs of the epoch."""
    lines = lines_of(training(100), pods={})
    assert "  scoring the validation pairs of the epoch" in lines
    assert "pod gone" in lines[0]


def test_the_final_training_shows_no_validation_scoring():
    """The final training scores nothing between its epochs; its stage shows how long it runs."""
    final = training(100, stage="final training", stage_started_at=at(-900))
    for key in ["trial", "finished_trials", "n_trials", "mean_trial_seconds"]:
        del final[key]
    lines = lines_of(final)
    assert "  scoring the validation pairs of the epoch" not in lines
    assert "  final training - for 15min" in lines


def test_the_remaining_time_comes_from_two_samples_of_the_same_pod():
    """60 steps in 60 s leave 90 of the 300 steps = 90 s; a new pod starts the measurement again."""
    tracker = watch_progress.SpeedTracker()
    assert tracker.remaining_seconds(training(50)) is None
    # epoch 3, step 10 = step 210 of 300
    assert tracker.remaining_seconds(
        training(10, updated=60, epoch=3)
    ) == pytest.approx(90)
    restarted = training(20, updated=120, epoch=3, host="train-longformer-zz99")
    assert tracker.remaining_seconds(restarted) is None


def test_the_tuning_estimate_counts_the_running_trial_once():
    """5 of 30 finished, trial 7 running - 24 later trials at 1h each plus its 50min left (10min of 1h done)."""
    text = "\n".join(lines_of(training(50)))
    assert "the tuning ends in about 24h 50min" in text


def test_the_trial_and_the_step_rows_carry_their_bar():
    """The trial row has 5 of 30, the step row 50 of 100; text rows have no bar."""
    bars = [(row.done, row.total) for row in rows_of(training(50)) if row.total is not None]
    assert bars == [(5, 30), (50, 100)]

def test_half_written_old_and_foreign_files_are_skipped(tmp_path):
    """Not yet JSON, untouched for over 24h, or without the progress fields - not shown."""
    (tmp_path / "new.progress.json").write_text(
        json.dumps(training(1)), encoding="utf-8"
    )
    (tmp_path / "half.progress.json").write_text('{"run": "ha', encoding="utf-8")
    (tmp_path / "list.progress.json").write_text("[]", encoding="utf-8")
    (tmp_path / "other.progress.json").write_text('{"run": "x"}', encoding="utf-8")
    old = tmp_path / "old.progress.json"
    old.write_text(json.dumps(training(1)), encoding="utf-8")
    os.utime(old, (NOW - 25 * 3600, NOW - 25 * 3600))
    assert len(watch_progress.read_progress_files(tmp_path, NOW)) == 1


def test_a_pod_without_a_progress_file_is_listed():
    """A pod waiting for a GPU writes no file yet and is still shown."""
    pods = {POD: "Running", "train-bigbird-x1": "Pending"}
    rows = watch_progress.pods_without_progress(pods, [training(1)])
    assert [row.text for row in rows] == [
        "Pods without a progress file yet:",
        "  train-bigbird-x1 Pending",
    ]
    assert watch_progress.pods_without_progress(None, []) == []


def test_a_failed_kubectl_is_shown_and_the_progress_still_is(monkeypatch):
    """Without kubectl the view says so and shows the progress anyway."""
    monkeypatch.setenv("PATH", "")
    pods = watch_progress.read_pods()
    lines = lines_of(training(50), pods=pods)
    assert pods is None and "kubectl not reachable" in lines[0]


def test_the_view_renders_the_files_of_the_folder(tmp_path, monkeypatch):
    """build_view draws one run from a folder, without kubectl, into a terminal of 120 columns."""
    monkeypatch.setenv("PATH", "")
    (tmp_path / "run.progress.json").write_text(
        json.dumps(training(50, updated=0)), encoding="utf-8"
    )
    monkeypatch.setattr(watch_progress.time, "time", lambda: NOW)
    console = watch_progress.Console(width=120, record=True, color_system=None)
    console.print(watch_progress.build_view(watch_progress.SpeedTracker(), tmp_path))
    text = console.export_text()
    assert "kubectl not reachable" in text and "epoch 2 / 3, step 50 / 100" in text


def test_a_kubectl_answer_that_is_not_json_is_shown_as_not_reachable(
    tmp_path, monkeypatch
):
    """An error page or a login prompt instead of JSON does not stop the view."""
    kubectl = tmp_path / "kubectl"
    kubectl.write_text(
        "#!/bin/sh\necho 'error: You must be logged in'\n", encoding="utf-8"
    )
    kubectl.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert watch_progress.read_pods() is None

def test_a_new_trial_in_the_same_pod_starts_the_measurement_again():
    """The next trial of a pod is measured from its own first sample, not from the trial before."""
    tracker = watch_progress.SpeedTracker()
    tracker.remaining_seconds(training(50))
    tracker.remaining_seconds(training(10, updated=60, epoch=3))
    next_trial = {"trial": 8, "stage_started_at": at(100)}
    assert tracker.remaining_seconds(training(5, updated=120, epoch=1, **next_trial)) is None
    # 30 steps in 30 s, step 35 of 300 - 265 steps left
    assert tracker.remaining_seconds(
        training(35, updated=150, epoch=1, **next_trial)
    ) == pytest.approx(265)