"""Live view of every running experiment on dt - reads ~/logs/*.progress.json and kubectl.

Run from a terminal on the dt hub server, in the launch checkout ~/launch/<tag>/experiments:
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
# the compact view has one bar per run, so a shorter one keeps the row on the screen
LINE_BAR_WIDTH = 16
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


def pod_phase(progress: dict, pods: dict[str, str] | None) -> str:
    """Return the phase of the pod of a run, or why there is none."""
    if pods is None:
        return "kubectl not reachable"
    return pods.get(progress["host"], "pod gone")


def seconds_since(iso_time: str, now: float) -> float:
    """Return the seconds from an ISO timestamp of a progress file to now."""
    return now - datetime.fromisoformat(iso_time).timestamp()


def estimate_tuning_seconds(
    progress: dict,
    remaining: float | None,
    stage_seconds: float,
) -> float | None:
    """Return the seconds until the tuning ends, None before the first trial has finished.

    Args:
        progress: One parsed progress file.
        remaining: Seconds left in the running trial from SpeedTracker, None if unknown.
        stage_seconds: Seconds since the running trial started.

    Returns:
        The running trial with its own estimate (or the mean trial time), the trials after it
        with the mean trial time.
    """
    if not progress.get("mean_trial_seconds"):
        return None
    later_trials = progress["n_trials"] - progress["finished_trials"] - 1
    running = remaining
    if running is None:
        running = max(progress["mean_trial_seconds"] - stage_seconds, 0)
    return running + later_trials * progress["mean_trial_seconds"]


@dataclass
class Line:
    """One run in the compact view - one table row; the innermost counter gets the bar."""

    run: str
    phase: str
    stage: str
    position: str
    done: int | None
    total: int | None
    estimate: str
    updated: str


def summarize(
    progress: dict,
    pods: dict[str, str] | None,
    tracker: SpeedTracker,
    now: float,
) -> Line:
    """Return one run as one line of the compact view.

    Args:
        progress: One parsed progress file.
        pods: Pod phases from read_pods, None if kubectl failed.
        tracker: Speed of the runs over the refreshes.
        now: Current time in seconds since the epoch.

    Returns:
        The run, its pod phase, the stage, the trials, epoch, step (one step = one batch) and
        pairs it has reached, the bar of the innermost of them and the estimates.
    """
    phase = pod_phase(progress, pods)
    stage_seconds = seconds_since(progress["stage_started_at"], now)
    parts = []
    done = total = None
    if "trial" in progress:
        parts.append(f"trials {progress['finished_trials']}/{progress['n_trials']}")
        done, total = progress["finished_trials"], progress["n_trials"]
    if "step" in progress:
        parts.append(
            f"epoch {progress['epoch']}/{progress['n_epochs']}, "
            f"step {progress['step']}/{progress['n_steps']}",
        )
        done, total = progress["step"], progress["n_steps"]
        # only the trials score the validation pairs after each epoch, not the final training
        if progress["stage"] == "tuning" and progress["step"] == progress["n_steps"]:
            parts.append("validation scoring")
    if "pairs_done" in progress:
        split = f"{progress['split']} " if "split" in progress else ""
        parts.append(f"{split}pairs {progress['pairs_done']}/{progress['n_pairs']}")
        done, total = progress["pairs_done"], progress["n_pairs"]
    remaining = tracker.remaining_seconds(progress)
    tuning_seconds = estimate_tuning_seconds(progress, remaining, stage_seconds)
    estimates = []
    if remaining is not None:
        estimates.append(f"training {format_duration(remaining)}")
    if tuning_seconds is not None:
        estimates.append(f"tuning {format_duration(tuning_seconds)}")
    stage = progress["stage"]
    if not estimates:
        # without an estimate, the time in the stage shows that the run moves
        stage = f"{stage} ({format_duration(stage_seconds)})"
    return Line(
        run=progress["run"],
        phase=phase,
        stage=stage,
        position=", ".join(parts),
        done=done,
        total=total,
        estimate=", ".join(estimates),
        updated=format_duration(seconds_since(progress["updated_at"], now)),
    )


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
    rows = [Row(f"{progress['run']}  -  pod {pod} {pod_phase(progress, pods)}")]
    stage = progress["stage"]
    stage_seconds = seconds_since(progress["stage_started_at"], now)
    if "trial" in progress:
        rows.append(
            Row(
                f"  {stage} - trial {progress['trial']}, {progress['finished_trials']} of "
                f"{progress['n_trials']} finished",
                progress["finished_trials"],
                progress["n_trials"],
            ),
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
            ),
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
            ),
        )
    remaining = tracker.remaining_seconds(progress)
    if remaining is not None:
        rows.append(Row(f"  this training ends in about {format_duration(remaining)}"))
    tuning_seconds = estimate_tuning_seconds(progress, remaining, stage_seconds)
    if tuning_seconds is not None:
        rows.append(
            Row(
                f"  the tuning ends in about {format_duration(tuning_seconds)} "
                "(the final training not included)",
            ),
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
    # a finished pod has nothing left to wait for
    waiting = sorted(
        name
        for name, phase in (pods or {}).items()
        if name not in hosts and phase != "Succeeded"
    )
    if not waiting:
        return []
    return [
        Row("Pods without a progress file yet:"),
        *[Row(f"  {name} {pods[name]}") for name in waiting],
    ]


def styled(text: str) -> Text:
    """Return the text with ROW_STYLES laid over it."""
    styled_text = Text(text)
    for pattern, style in ROW_STYLES:
        styled_text.highlight_regex(pattern, style)
    return styled_text


def bar_cells(
    done: int | None,
    total: int | None,
    width: int = BAR_WIDTH,
) -> tuple[ProgressBar | str, str]:
    """Return the bar and the percent of a counter, two empty cells without one."""
    if total is None:
        return "", ""
    percent = f"{100 * done // total}%" if total else ""
    return ProgressBar(total, done, width=width), percent


def rows_table(rows: list[Row]) -> Table:
    """Return the detailed rows as a table - the text, and a bar where the row has one."""
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True)
    table.add_column(width=BAR_WIDTH)
    table.add_column(justify="right")
    for row in rows:
        table.add_row(styled(row.text), *bar_cells(row.done, row.total))
    return table


def lines_table(lines: list[Line], width: int | None = None) -> Table:
    """Return the compact view - one run per row, with a header.

    Args:
        lines: The runs, one Line each.
        width: Columns of the terminal; the run names are cut first so that the
            counters and the estimates always stay on the screen. None leaves them whole.

    Returns:
        The table.
    """
    other_cells = [
        [line.phase, line.stage, line.position, line.estimate, line.updated]
        for line in lines
    ]
    # the five text columns, the bar, the percent and one space on each side of every column
    other_width = sum(
        max(len(cell) for cell in column) for column in zip(*other_cells, strict=True)
    )
    other_width += LINE_BAR_WIDTH + len("100%") + 2 * 8
    run_width = None if width is None else max(width - other_width, 16)
    table = Table(box=None, padding=(0, 1), header_style="dim")
    table.add_column("run", no_wrap=True, overflow="ellipsis", max_width=run_width)
    table.add_column("pod", no_wrap=True)
    table.add_column("stage", no_wrap=True)
    table.add_column("reached", no_wrap=True)
    table.add_column("", width=LINE_BAR_WIDTH)
    table.add_column("", justify="right", no_wrap=True)
    table.add_column("ends in about", no_wrap=True)
    table.add_column("last write", no_wrap=True, style="dim")
    for line in lines:
        table.add_row(
            Text(line.run.removeprefix("JP_").removesuffix("_v2"), style="bold"),
            styled(line.phase),
            line.stage,
            line.position,
            *bar_cells(line.done, line.total, LINE_BAR_WIDTH),
            line.estimate,
            line.updated,
        )
    return table


def build_view(
    tracker: SpeedTracker,
    log_dir: Path = LOG_DIR,
    *,
    show_finished: bool = False,
    detail: bool = False,
    width: int | None = None,
) -> Group:
    """Read the progress files and the pods once and return the view.

    Args:
        tracker: Speed of the runs over the refreshes.
        log_dir: Folder of the progress files.
        show_finished: Also show the runs whose stage is done; otherwise they are only counted.
        detail: Several rows per run (pod name, loss, every estimate) instead of one.
        width: Columns of the terminal, for the compact view; None leaves the run names whole.

    Returns:
        A header, the runs, the pods without a progress file and the count of hidden runs.
    """
    now = time.time()
    pods = read_pods()
    shown = read_progress_files(log_dir, now)
    active = [
        progress for progress in shown if show_finished or progress["stage"] != "done"
    ]
    parts = [
        Text(
            f"{datetime.now(UTC).astimezone():%H:%M:%S %Z} - Ctrl+C ends this view, never a Job\n",
        ),
    ]
    if detail:
        rows = []
        for progress in active:
            rows += [*describe(progress, pods, tracker, now), Row("")]
        parts.append(rows_table(rows))
    elif active:
        lines = [summarize(progress, pods, tracker, now) for progress in active]
        parts.append(lines_table(lines, width))
    # the pods of hidden runs are not waiting, so every read file counts
    extra_rows = pods_without_progress(pods, shown)
    n_hidden = len(shown) - len(active)
    if n_hidden:
        extra_rows.append(Row(f"{n_hidden} finished runs hidden (--all shows them)"))
    if not shown:
        extra_rows.append(
            Row(f"No progress file changed in the last {SHOWN_HOURS}h in {log_dir}."),
        )
    if extra_rows:
        parts.append(rows_table([Row(""), *extra_rows]))
    return Group(*parts)


def main() -> None:
    """Redraw the view every REFRESH_SECONDS until Ctrl+C."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--once",
        action="store_true",
        help="print the view once and stop",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="also show the finished runs",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="several rows per run instead of one",
    )
    args = parser.parse_args()
    console = Console()
    tracker = SpeedTracker()

    def view() -> Group:
        return build_view(
            tracker,
            show_finished=args.all,
            detail=args.detail,
            width=console.width,
        )

    if args.once:
        console.print(view())
        return
    try:
        with Live(view(), console=console, auto_refresh=False) as live:
            while True:
                time.sleep(REFRESH_SECONDS)
                live.update(view(), refresh=True)
    except KeyboardInterrupt:
        console.print("View stopped - the Jobs keep running.")


if __name__ == "__main__":
    main()
