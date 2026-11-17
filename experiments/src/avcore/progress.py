"""Progress file of a running experiment, read by the watcher on dt - never by the run itself.

The run writes where it stands (stage, trial, epoch, step, pairs) into ONE small JSON file in
~/logs. Nothing in the run reads the file, so it can not change a result; a failed write is
logged once and the run goes on.
"""

import json
import logging
import os
import socket
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import optuna

from avcore.tuning import FINISHED_STATES

logger = logging.getLogger(__name__)

MIN_SECONDS = 30.0


def trial_fields(trial: optuna.Trial, n_trials: int) -> dict[str, object]:
    """Return the fields of the stage "tuning" for a trial that starts.

    Args:
        trial: The trial that starts.
        n_trials: Finished trials the study needs.

    Returns:
        trial (its number), finished_trials, n_trials and mean_trial_seconds (the mean wall
        time of the finished trials, None before the first one).
    """
    finished = trial.study.get_trials(deepcopy=False, states=FINISHED_STATES)
    seconds = [
        (done.datetime_complete - done.datetime_start).total_seconds()
        for done in finished
    ]
    return {
        "trial": trial.number,
        "finished_trials": len(finished),
        "n_trials": n_trials,
        "mean_trial_seconds": sum(seconds) / len(seconds) if seconds else None,
    }


def _now() -> str:
    """Return the current UTC time as an ISO 8601 text, to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")

class ProgressFile:
    """Writes the progress of one run into path, at most every min_seconds."""

    def __init__(
        self,
        path: Path,
        run: str,
        *,
        min_seconds: float = MIN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Remember where to write; nothing is written before the first stage.

        Args:
            path: The progress file, e.g. ~/logs/<run name>.progress.json.
            run: Run name shown by the watcher.
            min_seconds: Shortest time between two writes of update.
            clock: Seconds of a monotonic clock - replaced in the tests.
        """
        self.path = path
        self.run = run
        self.min_seconds = min_seconds
        self.clock = clock
        self.state: dict = {}
        self.last_write: float | None = None
        self.write_failed = False

    def start_stage(self, stage: str, **fields: object) -> None:
        """Start a stage, e.g. "tuning" or "test scoring", and write the file at once.

        Args:
            stage: Name of the stage.
            **fields: Values of the stage, e.g. trial=3, n_trials=30; the fields of the
                previous stage are dropped.
        """
        self.state = {"stage": stage, "stage_started_at": _now(), **fields}
        self._write()

    def update(self, *, force: bool = False, **fields: object) -> None:
        """Merge fields into the current stage; write when min_seconds have passed.

        Args:
            force: Write now, e.g. after the last step of an epoch.
            **fields: Values that changed, e.g. epoch=1, step=120.
        """
        self.state.update(fields)
        since_write = (
            None if self.last_write is None else self.clock() - self.last_write
        )
        if force or since_write is None or since_write >= self.min_seconds:
            self._write()

    def _write(self) -> None:
        """Replace the file in one step, so the watcher never reads half a file."""
        # the pod name in a Kubernetes Job - a new name means a restarted pod
        host = socket.gethostname()
        content = {
            "run": self.run,
            "host": host,
            **self.state,
            "updated_at": _now(),
        }
        # a value json can not hold stops the run here, in its first stage
        text = json.dumps(content, indent=2)
        self.last_write = self.clock()
        # the host too - two pods can run their kernel under the same process id
        temporary_path = self.path.with_name(
            f"{self.path.name}.{host}.{os.getpid()}.tmp"
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(text, encoding="utf-8")
            os.replace(temporary_path, self.path)
        except OSError as error:
            if not self.write_failed:
                logger.warning(
                    f"Progress file {self.path} not written, the run goes on: {error}"
                )
            self.write_failed = True
            return
        if self.write_failed:
            logger.warning(f"Progress file {self.path} written again.")
        self.write_failed = False