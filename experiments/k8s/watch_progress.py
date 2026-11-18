"""Live view of every running experiment on dt - reads ~/logs/*.progress.json and kubectl.

Run from a terminal on the dt hub server, in ~/diploma_thesis/experiments:
".venv/bin/python k8s/watch_progress.py" (launch.sh starts it after the kubectl apply).
Ctrl+C or a closed terminal ends ONLY this view, never a Job.
"""

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

LOG_DIR = Path.home() / "logs"
REFRESH_SECONDS = 10
# progress files untouched for longer are finished or abandoned runs
SHOWN_HOURS = 24
BAR_WIDTH = 30
# the fields every progress file has - other files in ~/logs are skipped
REQUIRED_FIELDS = {"run", "host", "stage", "stage_started_at", "updated_at"}

# styles laid over the finished text of a row - the text itself never changes
ROW_STYLES = [
    (re.compile(r"^\S.*"), "bold"),  # a line without indent is a run or a section
    (re.compile(r"\b(Running|Succeeded)$"), "green"),
    (re.compile(r"\bPending$"), "yellow"),
    (re.compile(r"(pod gone|Failed|kubectl not reachable)$"), "red"),
    (re.compile(r"(?<=in about ).*?(?= \(|$)"), "bold"),  # only the duration
    (re.compile(r"^  updated .*|\(.*\)"), "dim"),
]


@dataclass
class Row:
    """One line of the view; a line with done and total gets a bar."""

    text: str
    done: int | None = None
    total: int | None = None


def format_duration(seconds: float) -> str:
    """Return a duration as "2h 05min", "14min" or "under 1min"."""
    minutes = int(seconds // 60)
    if minutes < 1:
        return "under 1min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}min" if hours else f"{minutes}min"


def read_progress_files(log_dir: Path, now: float) -> list[dict]:
    """Return the progress files changed in the last SHOWN_HOURS, sorted by file name.

    Args:
        log_dir: Folder of the progress files.
        now: Current time in seconds since the epoch.

    Returns:
        The parsed files; a file being replaced right now, or one without the REQUIRED_FIELDS,
        is skipped until the next refresh.
    """
    progress = []
    for path in sorted(log_dir.glob("*.progress.json")):
        try:
            if now - path.stat().st_mtime > SHOWN_HOURS * 3600:
                continue
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(content, dict) and REQUIRED_FIELDS <= content.keys():
            progress.append(content)
    return progress


def read_pods() -> dict[str, str] | None:
    """Return pod name -> "Running" / "Pending" / ... of every dt-v2 pod, None if kubectl fails."""
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-l", "app=dt-v2", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        pods = json.loads(result.stdout)["items"]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, KeyError):
        return None
    return {pod["metadata"]["name"]: pod["status"].get("phase", "?") for pod in pods}

class SpeedTracker:
    """Steps per second of every run, measured from the progress files this view has seen."""

    def __init__(self) -> None:
        self.first_seen: dict[tuple, tuple[float, int]] = {}

    def remaining_seconds(self, progress: dict) -> float | None:
        """Estimate the seconds left in the current trial or training, None before 2 samples.

        Args:
            progress: One parsed progress file with epoch, n_epochs, step and n_steps.

        Returns:
            Remaining steps divided by the speed since the first sample of this stage and pod;
            a validation scoring between the two samples counts in the speed.
        """
        if "step" not in progress:
            return None
        steps_done = (progress["epoch"] - 1) * progress["n_steps"] + progress["step"]
        steps_total = progress["n_epochs"] * progress["n_steps"]
        updated = datetime.fromisoformat(progress["updated_at"]).timestamp()
        # identifies the pod training
        key = (progress["run"], progress["host"], progress["stage_started_at"])
        first = self.first_seen.setdefault(key, (updated, steps_done))
        if updated <= first[0] or steps_done <= first[1]:
            return None
        speed = (steps_done - first[1]) / (updated - first[0])
        return (steps_total - steps_done) / speed


def describe(
    progress: dict,
    pods: dict[str, str] | None,
    tracker: SpeedTracker,
    now: float,
) -> list[Row]:
    """Return the rows of one run.

    Args:
        progress: One parsed progress file.
        pods: Pod phases from read_pods, None if kubectl failed.
        tracker: Speed of the runs over the refreshes.
        now: Current time in seconds since the epoch.

    Returns:
        The run and its pod, the stage, the bars and the estimates.
    """
    pod = progress["host"]
    phase = "kubectl not reachable" if pods is None else pods.get(pod, "pod gone")
    rows = [Row(f"{progress['run']}  -  pod {pod} {phase}")]
    stage = progress["stage"]
    stage_seconds = (
        now - datetime.fromisoformat(progress["stage_started_at"]).timestamp()
    )
    if "trial" in progress:
        rows.append(
            Row(
                f"  {stage} - trial {progress['trial']}, {progress['finished_trials']} of "
                f"{progress['n_trials']} finished",
                progress["finished_trials"],
                progress["n_trials"],
            )
        )
    else:
        rows.append(Row(f"  {stage} - for {format_duration(stage_seconds)}"))
    if "step" in progress:
        rows.append(
            Row(
                f"  epoch {progress['epoch']} / {progress['n_epochs']}, step {progress['step']} / "
                f"{progress['n_steps']}, loss {progress['loss']:.4f}",
                progress["step"],
                progress["n_steps"],
            )
        )
        # only the trials score the validation pairs after each epoch, not the final training
        if stage == "tuning" and progress["step"] == progress["n_steps"]:
            rows.append(Row("  scoring the validation pairs of the epoch"))
    if "pairs_done" in progress:
        rows.append(
            Row(
                f"  pairs {progress['pairs_done']} / {progress['n_pairs']}",
                progress["pairs_done"],
                progress["n_pairs"],
            )
        )
    remaining = tracker.remaining_seconds(progress)
    if remaining is not None:
        rows.append(Row(f"  this training ends in about {format_duration(remaining)}"))
    if progress.get("mean_trial_seconds"):
        # the running trial with its own estimate, the trials after it with the mean
        later_trials = progress["n_trials"] - progress["finished_trials"] - 1
        running = remaining
        if running is None:
            running = max(progress["mean_trial_seconds"] - stage_seconds, 0)
        tuning_seconds = running + later_trials * progress["mean_trial_seconds"]
        rows.append(
            Row(
                f"  the tuning ends in about {format_duration(tuning_seconds)} "
                "(the final training not included)"
            )
        )
    updated = datetime.fromisoformat(progress["updated_at"]).timestamp()
    rows.append(Row(f"  updated {format_duration(now - updated)} ago"))
    return rows


def pods_without_progress(pods: dict[str, str] | None, shown: list[dict]) -> list[Row]:
    """Return a row for every dt-v2 pod no shown progress file names, e.g. one waiting for a GPU.

    Args:
        pods: Pod phases from read_pods, None if kubectl failed.
        shown: The progress files of the view.

    Returns:
        "Pods without a progress file yet:" and one row per pod, or nothing.
    """
    hosts = {progress["host"] for progress in shown}
    waiting = sorted(name for name in pods or {} if name not in hosts)
    if not waiting:
        return []
    return [
        Row("Pods without a progress file yet:"),
        *[Row(f"  {name} {pods[name]}") for name in waiting],
    ]

def build_view(tracker: SpeedTracker, log_dir: Path = LOG_DIR) -> Group:
    """Read the progress files and the pods once and return the view as a table.

    Args:
        tracker: Speed of the runs over the refreshes.
        log_dir: Folder of the progress files.

    Returns:
        A header and one table row per Row - the text, and a bar where the row has one.
    """
    now = time.time()
    pods = read_pods()
    shown = read_progress_files(log_dir, now)
    rows = []
    for progress in shown:
        rows += [*describe(progress, pods, tracker, now), Row("")]
    rows += pods_without_progress(pods, shown)
    if not rows:
        rows = [
            Row(f"No progress file changed in the last {SHOWN_HOURS}h in {log_dir}.")
        ]
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True)
    table.add_column(width=BAR_WIDTH)
    table.add_column(justify="right")
    for row in rows:
        text = Text(row.text)
        for pattern, style in ROW_STYLES:
            text.highlight_regex(pattern, style)
        if row.total is None:
            table.add_row(text, "", "")
        else:
            percent = f"{100 * row.done // row.total}%" if row.total else ""
            table.add_row(
                text,
                ProgressBar(row.total, row.done, width=BAR_WIDTH),
                percent,
            )
    header = Text(
        f"{datetime.now(UTC).astimezone():%H:%M:%S %Z} - Ctrl+C ends this view, never a Job\n"
    )
    return Group(header, table)


def main() -> None:
    """Redraw the view every REFRESH_SECONDS until Ctrl+C."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--once", action="store_true", help="print the view once and stop"
    )
    args = parser.parse_args()
    console = Console()
    tracker = SpeedTracker()
    if args.once:
        console.print(build_view(tracker))
        return
    try:
        with Live(build_view(tracker), console=console, auto_refresh=False) as live:
            while True:
                time.sleep(REFRESH_SECONDS)
                live.update(build_view(tracker), refresh=True)
    except KeyboardInterrupt:
        console.print("View stopped - the Jobs keep running.")


if __name__ == "__main__":
    main()
