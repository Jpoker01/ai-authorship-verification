"""Live view of every running experiment on dt - reads ~/logs/*.progress.json and kubectl.

Run from a terminal on the dt hub server, in ~/diploma_thesis/experiments:
".venv/bin/python k8s/watch_progress.py" (launch.sh starts it after the kubectl apply).
Ctrl+C or a closed terminal ends ONLY this view, never a Job.
"""

import argparse
import json
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