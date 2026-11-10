import hashlib
import os
import subprocess
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
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

def test_collect_run_context_records_commit_lock_and_data_hashes(project_dir, tmp_path):
    """The context holds HEAD, the LF sha256 of uv.lock and the sha256 of each data file."""
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"id": "a"}\n')
    run_context = tracking.collect_run_context(
        project_dir, [data_path], package_dir=project_dir / "src" / "avcore"
    )
    head = subprocess.run(
        ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert run_context.environment["commit"] == head
    assert run_context.environment["uv_lock_sha256"] == hashlib.sha256(b"lock\n").hexdigest()
    assert (
        run_context.environment["data_sha256.train.jsonl"]
        == hashlib.sha256(b'{"id": "a"}\n').hexdigest()
    )
    expected_keys = {"started_at", "python_version", "cgroup_cpu_max",
                     "version.mlflow", "gpu_name"}
    assert expected_keys <= run_context.environment.keys()
    assert "pandas" in run_context.packages

@pytest.mark.parametrize("change", ["modified", "untracked_hidden", "unpushed"])
def test_collect_run_context_rejects_an_unclean_repo(project_dir, change):
    """A modified file, a hidden untracked file or an unpushed commit stop the run."""
    if change == "modified":
        (project_dir / "uv.lock").write_text("changed\n")
    elif change == "untracked_hidden":
        _git(project_dir, "config", "status.showUntrackedFiles", "no")
        (project_dir / "new_module.py").write_text("x = 1\n")
    else:
        _git(project_dir, "commit", "-q", "--allow-empty", "-m", "local only")
    with pytest.raises(RuntimeError):
        tracking.collect_run_context(
            project_dir, [], package_dir=project_dir / "src" / "avcore"
        )

def test_collect_run_context_rejects_a_package_outside_the_project(
    project_dir, tmp_path
):
    """An avcore imported from another folder than project_dir stops the run."""
    with pytest.raises(RuntimeError, match="imported from"):
        tracking.collect_run_context(
            project_dir, [], package_dir=tmp_path / "other" / "avcore"
        )

@pytest.mark.parametrize(
    ("run_name", "changes"),
    [
        ("run_v2", {"metrics": {"test_f1": float("nan")}}),
        ("run_v2", {"metrics": {"test_f1": True}}),
        ("run_v2", {"params": {"commit": "replaced"}}),
        ("run_v2", {"params": {"note": "x" * 251}}),
        ("run_v2", {"params": {"f1@0.5": 1}}),
        (
            "run_v2",
            {"tables": {"trials": pd.DataFrame({"pair": [["text one", "text two"]]})}},
        ),
        ("run_v2", {"tables": {"trials": pd.DataFrame({"note": ["x" * 101]})}}),
        ("run_v2", {"tables": {"sub/trials": pd.DataFrame({"a": [1]})}}),
        (
            "run_v2",
            {
                "tables": {
                    "predictions-test": _predictions_df().rename(
                        columns={"score": "propability_same_author"}
                    )
                }
            },
        ),
        ("run_v2", {"tables": {"predictions-test": _predictions_df().assign(id="a")}}),
        ("run_v2", {"tables": {"test_predictions": _predictions_df()}}),
        ("run_v2", {"tables": {"predictions-test": _predictions_df().assign(same=2)}}),
        ("run_v2", {"tables": {"predictions-test": _predictions_df().assign(score=float("nan"))}}),
        ("run", {}),
        ("run x_v2", {}),
        ("run_v2", {"figures": {"sub/confusion_matrix": None}}),
        ("run_v2", {"tables": {"trials": pd.DataFrame({"note": [np.bytes_(b"x" * 101)]})}}),
    ],
)
def test_write_run_files_rejects_invalid_input_and_writes_nothing(
    tmp_path, context, run_name, changes
):
    """Every invalid input raises ValueError before any file or folder is written."""
    with pytest.raises(ValueError):
        _write(tmp_path / run_name, context, **changes)
    assert list(tmp_path.iterdir()) == []

def test_write_run_files_refuses_an_existing_folder(tmp_path, context):
    """An existing run folder is never written into."""
    (tmp_path / "run_v2").mkdir()
    with pytest.raises(FileExistsError):
        _write(tmp_path / "run_v2", context)

def test_log_run_files_uploads_the_saved_files(tmp_path, context, experiment):
    """The run gets the folder name, the context and params, the metrics and every file."""
    run_dir = tmp_path / "longformer_v2"
    figure, _ = plt.subplots()
    _write(run_dir, context, figures={"confusion_matrix-test": figure})
    run = mlflow.get_run(tracking.log_run_files(run_dir, experiment))
    assert run.info.run_name == "longformer_v2"
    assert run.data.params == {"commit": "abc", "lr": "1e-05"}
    assert run.data.metrics == {"test_f1": 92.38}
    artifacts = {
        item.path for item in mlflow.MlflowClient().list_artifacts(run.info.run_id)
    }
    assert artifacts == {
        "context.json",
        "packages.json",
        "params.json",
        "metrics.json",
        "manifest.json",
        "predictions-test.csv",
        "confusion_matrix-test.png",
    }


def test_log_run_files_returns_the_finished_run_on_a_second_call(
    tmp_path, context, experiment
):
    """Uploading the same folder twice gives one run."""
    run_dir = tmp_path / "longformer_v2"
    _write(run_dir, context)
    first_run_id = tracking.log_run_files(run_dir, experiment)
    assert tracking.log_run_files(run_dir, experiment) == first_run_id
    assert (
        len(mlflow.search_runs(experiment_names=[experiment], output_format="list"))
        == 1
    )

@pytest.mark.parametrize("change", ["extra", "modified"])
def test_log_run_files_refuses_files_outside_the_manifest(
    tmp_path, context, experiment, change
):
    """An extra or a changed file in the folder stops the upload."""
    run_dir = tmp_path / "longformer_v2"
    _write(run_dir, context)
    if change == "extra":
        (run_dir / "notes.txt").write_text("text")
    else:
        (run_dir / "metrics.json").write_text('{"test_f1": 99.0}')
    with pytest.raises(RuntimeError, match="manifest"):
        tracking.log_run_files(run_dir, experiment)


def test_log_run_files_requires_an_existing_experiment(tmp_path, context, experiment):
    """An unknown experiment name stops the upload instead of creating it."""
    run_dir = tmp_path / "longformer_v2"
    _write(run_dir, context)
    with pytest.raises(RuntimeError, match="does not exist"):
        tracking.log_run_files(run_dir, "missing_v2")


def test_connect_tracking_requires_databricks_credentials(tmp_path, monkeypatch):
    """A missing token raises before MLflow is touched."""
    for name in tracking.DATABRICKS_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    # the check has to fail before MLflow is touched - a real connection retries for minutes
    monkeypatch.setattr(
        mlflow,
        "set_tracking_uri",
        mock.Mock(side_effect=AssertionError("MLflow reached")),
    )
    env_path = tmp_path / ".env"
    env_path.write_text("DATABRICKS_HOST=https://example.invalid\n")
    with pytest.raises(RuntimeError, match="DATABRICKS_TOKEN"):
        tracking.connect_tracking(env_path)

def test_env_example_lists_the_required_variables():
    """The .env template lists exactly the credentials that connect_tracking requires."""
    lines = (
        (Path(__file__).parents[1] / ".env.example")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    names = [line.split("=")[0] for line in lines if line and not line.startswith("#")]
    assert names == tracking.DATABRICKS_VARIABLES

def test_log_run_files_uploads_again_after_a_failed_upload(
    tmp_path, context, experiment, monkeypatch
):
    """A failed upload does not count - the next call uploads the folder as a new run."""
    run_dir = tmp_path / "longformer_v2"
    _write(run_dir, context)
    monkeypatch.setattr(mlflow, "log_artifacts", mock.Mock(side_effect=OSError("no network")))
    with pytest.raises(OSError):
        tracking.log_run_files(run_dir, experiment)
    monkeypatch.undo()
    run_id = tracking.log_run_files(run_dir, experiment)
    runs = mlflow.search_runs(experiment_names=[experiment], output_format="list")
    assert sorted(run.info.status for run in runs) == ["FAILED", "FINISHED"]
    assert mlflow.get_run(run_id).info.status == "FINISHED"

def test_collect_run_context_rejects_repeated_data_file_names(project_dir, tmp_path):
    """Two data files with the same name would share one data_sha256 key."""
    data_paths = [tmp_path / "a" / "train.jsonl", tmp_path / "b" / "train.jsonl"]
    for data_path in data_paths:
        data_path.parent.mkdir()
        data_path.write_text("{}\n")
    with pytest.raises(ValueError, match="not unique"):
        tracking.collect_run_context(
            project_dir, data_paths, package_dir=project_dir / "src" / "avcore"
        )

def test_write_run_files_accepts_missing_values_and_times(tmp_path, context):
    """None, NaT and times - as in an Optuna trials table - are plain cells."""
    trials_df = pd.DataFrame(
        {
            "number": [0, 1],
            "datetime_start": pd.to_datetime(["2026-11-10 10:00", "2026-11-10 11:00"]),
            "duration": pd.to_timedelta(["1h", None]),
            "state": ["COMPLETE", None],
        }
    )
    _write(tmp_path / "run_v2", context, tables={"trials": trials_df})
    assert (tmp_path / "run_v2" / "trials.csv").is_file()

def test_connect_tracking_loads_the_env_file_over_the_shell(tmp_path, monkeypatch):
    """The .env values win over stale shell variables and run URLs are not printed."""
    monkeypatch.setenv("DATABRICKS_TOKEN", "stale")
    env_path = tmp_path / ".env"
    env_path.write_text("DATABRICKS_HOST=https://example.invalid\nDATABRICKS_TOKEN=fresh\n")
    tracking.connect_tracking(env_path, tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    assert os.environ["DATABRICKS_TOKEN"] == "fresh"
    assert os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] == "true"

def test_connect_tracking_requires_the_env_file(tmp_path):
    """A missing .env file raises instead of loading nothing."""
    with pytest.raises(FileNotFoundError):
        tracking.connect_tracking(
            tmp_path / ".env", tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}"
        )

def test_write_run_files_rejects_a_relative_folder(tmp_path, context, monkeypatch):
    """A relative run folder raises and nothing is written into the current folder."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        _write(Path("run_v2"), context)
    assert list(tmp_path.iterdir()) == []