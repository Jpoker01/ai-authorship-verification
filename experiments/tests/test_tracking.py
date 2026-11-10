import hashlib
import os
import subprocess
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import pytest

from avcore import tracking

def _git(repo_dir, *args):
    """Run a git command in a test repo, with a committer name and email."""
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            *args,
        ],
        check=True,
        capture_output=True,
    )

@pytest.fixture(autouse=True)
def isolated_environment():
    """Restore os.environ and the tracking URI that the tested functions change."""
    tracking_uri = mlflow.get_tracking_uri()
    with mock.patch.dict(os.environ):
        yield
    mlflow.set_tracking_uri(tracking_uri)

@pytest.fixture
def project_dir(tmp_path):
    """experiments/ folder of a git repo with one commit pushed to a remote."""
    remote_dir = tmp_path / "remote.git"
    repo_dir = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "--bare", str(remote_dir)], check=True)
    subprocess.run(
        ["git", "clone", "-q", str(remote_dir), str(repo_dir)],
        check=True,
        capture_output=True,
    )
    project = repo_dir / "experiments"
    (project / "src" / "avcore").mkdir(parents=True)
    (project / "uv.lock").write_bytes(b"lock\r\n")
    (project / "src" / "avcore" / "__init__.py").write_text("")
    _git(repo_dir, "add", ".")
    _git(repo_dir, "commit", "-q", "-m", "init")
    _git(repo_dir, "push", "-q", "origin", "HEAD")
    return project

@pytest.fixture
def context():
    """A small run context, as collect_run_context returns it."""
    return tracking.RunContext(environment={"commit": "abc"}, packages={"pandas": "2.3.3"})

@pytest.fixture
def experiment(tmp_path):
    """An experiment in a local sqlite store, with its artifacts in tmp_path."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment(
        "test_v2", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    return "test_v2"

def _predictions_df():
    """A valid predictions table with two pairs."""
    return pd.DataFrame(
        {"id": ["a", "b"], "same": [1, 0], "score": [0.9, 0.2], "pred": [1, 0]}
    )


def _write(run_dir, context, **changes):
    """Write a valid run folder, with some arguments replaced."""
    arguments = {
        "params": {"lr": 1e-5},
        "metrics": {"test_f1": 92.38},
        "tables": {"predictions-test": _predictions_df()},
        "figures": {},
    }
    tracking.write_run_files(run_dir, context=context, **{**arguments, **changes})