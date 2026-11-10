"""Run records of the v2 experiments - context, local result files and their MLflow upload."""

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import numbers
import os
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.figure import Figure

EXPERIMENT_NAME = "/Shared/JP_authorship_verification_dt_v2"
SMOKE_EXPERIMENT_NAME = f"{EXPERIMENT_NAME}_smoke"

# 0 = different author, 1 = same author; score: higher = more likely the same author
PREDICTION_COLUMNS = ["id", "same", "score", "pred"]
PREDICTION_TABLES = ["predictions-test", "predictions-validation"]
PACKAGES = [
    "scikit-learn",
    "xgboost",
    "optuna",
    "numpy",
    "pandas",
    "mlflow",
    "torch",
    "transformers",
]
DATABRICKS_VARIABLES = ["DATABRICKS_HOST", "DATABRICKS_TOKEN"]
PACKAGE_DIR = Path(__file__).resolve().parent

# Databricks limits keys and param values to 250 characters; the local stores accept more
MAX_KEY_LENGTH = 250
MAX_PARAM_LENGTH = 250
# a PAN text has about 21 000 characters, ids and hyperparameter values far fewer
MAX_TABLE_TEXT_LENGTH = 100
_KEY_PATTERN = re.compile(r"[\w./\- ]+")
_FILE_NAME_PATTERN = re.compile(r"[\w.-]+")

CONTEXT_FILE = "context.json"
PACKAGES_FILE = "packages.json"
PARAMS_FILE = "params.json"
METRICS_FILE = "metrics.json"
MANIFEST_FILE = "manifest.json"
MANIFEST_TAG = "dt.manifest_sha256"

@dataclass(frozen=True)
class RunContext:
    """Environment of a run.

    Attributes:
        environment: Commit, hashes, versions and hardware, logged as MLflow params.
        packages: Every installed distribution with its version, saved as a file.
    """

    environment: dict[str, str]
    packages: dict[str, str]

def _git(project_dir: Path, *args: str) -> str:
    """Run a read-only git command in project_dir and return its stripped output."""
    result = subprocess.run(
        ["git", "-C", str(project_dir), *args], capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()

def _sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file, read in chunks."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()

def _version(package: str) -> str:
    """Return the installed version of a package, or "not installed"."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"

def _cgroup_value(name: str) -> str:
    """Return a cgroup v2 limit of this container (cpu.max, memory.max), or "none"."""
    path = Path("/sys/fs/cgroup") / name
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "none"

def _gpu_params() -> dict[str, str]:
    """Return CUDA, cuDNN and GPU of torch, "none" without torch or without a GPU."""
    params = {
        "cuda_version": "none",
        "cudnn_version": "none",
        "gpu_count": "0",
        "gpu_name": "none",
        "gpu_vram_gb": "none",
    }
    # the default group has no torch, the traditional notebooks run without it
    if importlib.util.find_spec("torch") is None:
        return params

    import torch

    params["cuda_version"] = str(torch.version.cuda)
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        params["cudnn_version"] = str(torch.backends.cudnn.version())
        params["gpu_count"] = str(torch.cuda.device_count())
        params["gpu_name"] = properties.name
        params["gpu_vram_gb"] = str(
            round(properties.total_memory / 1024 ** 3, 2))
    return params


def _check_repo(project_dir: Path, package_dir: Path) -> None:
    """Raise RuntimeError unless the repo is committed, pushed and the source of avcore."""
    status = _git(project_dir, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"The repo has uncommitted changes:\n{status}")
    if not _git(project_dir, "branch", "--remotes", "--contains", "HEAD"):
        raise RuntimeError("HEAD is not on any remote branch - push it first.")
    if not package_dir.resolve().is_relative_to(project_dir.resolve()):
        raise RuntimeError(f"avcore is imported from {package_dir}, not from {project_dir}.")

def collect_run_context(
    project_dir: Path,
    data_paths: Sequence[Path],
    *,
    package_dir: Path = PACKAGE_DIR,
) -> RunContext:
    """Collect the commit, lock and data hashes, versions and hardware of a run.

    Call it once at the top of a notebook (it hashes every data file).

    Args:
        project_dir: The experiments folder of the repo (holds uv.lock).
        data_paths: Every data file the run reads.
        package_dir: Folder avcore is imported from, has to lie inside project_dir.

    Returns:
        The run context.

    Raises:
        RuntimeError: If the repo is not clean, not pushed or not the source of avcore.
        ValueError: If two data files share a file name.
    """
    _check_repo(project_dir, package_dir)
    names = [path.name for path in data_paths]
    if len(set(names)) != len(names):
        raise ValueError(f"Data file names are not unique: {names}")

    # the same lock hashes the same on Windows (CRLF checkout) and Linux
    lock_bytes = (project_dir / "uv.lock").read_bytes().replace(b"\r\n", b"\n")
    environment = {
        "commit": _git(project_dir, "rev-parse", "HEAD"),
        "uv_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": str(os.cpu_count()),
        "cgroup_cpu_max": _cgroup_value("cpu.max"),
        "cgroup_memory_max": _cgroup_value("memory.max"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        **{f"data_sha256.{path.name}": _sha256(path) for path in data_paths},
        **{f"version.{package}": _version(package) for package in PACKAGES},
        **_gpu_params(),
    }
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    return RunContext(environment=environment, packages=dict(sorted(packages.items())))

def _check_key(kind: str, key: object) -> None:
    """Raise ValueError if a key is not a valid MLflow / Databricks key."""
    if not isinstance(key, str):
        raise ValueError(f"{kind} key {key!r} is not a string.")
    if len(key) > MAX_KEY_LENGTH:
        raise ValueError(f"{kind} key {key[:20]!r}... is longer than {MAX_KEY_LENGTH} characters.")
    if not _KEY_PATTERN.fullmatch(key):
        raise ValueError(f"{kind} key {key!r} has a character MLflow does not allow.")

def _check_file_name(name: str) -> None:
    """Raise ValueError if a table or figure name is not a plain file name."""
    if not _FILE_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"File name {name!r} is not allowed.")

def _is_plain_cell(value: object) -> bool:
    """Return True for a number, a bool, a time, a missing value or a short string."""
    if isinstance(value, str):
        return len(value) <= MAX_TABLE_TEXT_LENGTH
    # bool is a subclass of int, so True / False count as numbers too
    is_python_number = isinstance(value, (int, float))
    # not np.generic - np.bytes_ is one too and would skip the length check
    is_numpy_number = isinstance(value, (np.number, np.bool_))
    # an Optuna trials table holds start / end times and None for unfinished values
    is_time = isinstance(value, (pd.Timestamp, pd.Timedelta))
    is_missing = value is None or value is pd.NA or value is pd.NaT
    return is_python_number or is_numpy_number or is_time or is_missing

def _check_table(name: str, table_df: pd.DataFrame) -> None:
    """Raise ValueError for a bad file name,
    a cell that is not a number or a short string, or a bad prediction table.
    """
    _check_file_name(name)
    for column in table_df.columns:
        cells = [column, *table_df[column].astype(object)]
        if not all(_is_plain_cell(cell) for cell in cells):
            raise ValueError(f"Table {name} column {str(column)[:20]!r} holds a list, an object or a long text.")
    if "prediction" in name:
        _check_predictions(name, table_df)

def _check_predictions(name: str, predictions_df: pd.DataFrame) -> None:
    """Raise ValueError unless a prediction table has the agreed name, columns and values."""
    if name not in PREDICTION_TABLES or list(
            predictions_df.columns) != PREDICTION_COLUMNS:
        raise ValueError(
            f"Prediction table {name} has to be one of {PREDICTION_TABLES} with columns {PREDICTION_COLUMNS}.")
    if not predictions_df["id"].is_unique:
        raise ValueError(f"Table {name} has repeated ids.")
    if not predictions_df[["same", "pred"]].isin([0, 1]).all(axis=None):
        raise ValueError(
            f"Table {name} has a same or pred value that is not 0 / 1.")
    if not np.isfinite(
            predictions_df["score"].to_numpy(dtype=float)).all():
        raise ValueError(f"Table {name} has a score that is not finite.")

def _check_params(context: RunContext, params: Mapping[str, object]) -> None:
    """Raise ValueError if a param repeats a context key, has a bad key or is too long."""
    overlap = sorted(set(context.environment) & set(params))
    if overlap:
        raise ValueError(f"Params repeat context keys: {overlap}")
    for key, value in {**context.environment, **params}.items():
        _check_key("Param", key)
        if len(str(value)) > MAX_PARAM_LENGTH:
            raise ValueError(f"Param {key} is longer than {MAX_PARAM_LENGTH} characters.")


def _check_metrics(metrics: Mapping[str, float]) -> None:
    """Raise ValueError if a metric has a bad key or is not a finite number."""
    for key, value in metrics.items():
        _check_key("Metric", key)
        # bool is a number in Python, but True is not a metric
        is_number = isinstance(value, numbers.Real) and not isinstance(value, bool)
        if not is_number or not math.isfinite(value):
            raise ValueError(f"Metric {key} is not a finite number: {value!r}")

def write_run_files(
    run_dir: Path,
    *,
    context: RunContext,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    tables: Mapping[str, pd.DataFrame],
    figures: Mapping[str, Figure],
) -> None:
    """Write everything a run reports into a new folder - all checks first, then all files.

    The files go to run_dir.partial, which is renamed to run_dir once complete.

    Args:
        run_dir: Absolute folder that does not exist yet; its name is the run name and contains "_v2".
        context: Output of collect_run_context.
        params: Hyperparameters and settings, at most MAX_PARAM_LENGTH characters each.
        metrics: Finite numbers.
        tables: Saved as <name>.csv; prediction tables are PREDICTION_TABLES with PREDICTION_COLUMNS.
        figures: Saved as <name>.png.

    Raises:
        FileExistsError: If run_dir or run_dir.partial already exists.
        ValueError: If any input fails a check (nothing is written then).
    """
    if not run_dir.is_absolute() or "_v2" not in run_dir.name:
        raise ValueError(
            f"Run folder {run_dir} has to be absolute and contain _v2.")
    _check_file_name(run_dir.name)
    _check_params(context, params)
    _check_metrics(metrics)
    for name, table_df in tables.items():
        _check_table(name, table_df)
    for name in figures:
        _check_file_name(name)
    if run_dir.exists():
        raise FileExistsError(run_dir)

    partial_dir = run_dir.with_name(f"{run_dir.name}.partial")
    partial_dir.mkdir(parents=True, exist_ok=False)
    json_files = {
        CONTEXT_FILE: context.environment,
        PACKAGES_FILE: context.packages,
        PARAMS_FILE: {key: str(value) for key, value in params.items()},
        METRICS_FILE: {key: float(value) for key, value in metrics.items()},
    }
    for file_name, content in json_files.items():
        (partial_dir / file_name).write_text(json.dumps(content, indent=2),
                                             encoding="utf-8")
    for name, table_df in tables.items():
        table_df.to_csv(partial_dir / f"{name}.csv", index=False)
    for name, figure in figures.items():
        figure.savefig(partial_dir / f"{name}.png", bbox_inches="tight")

    manifest = {
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {path.name: _sha256(path) for path in
                  sorted(partial_dir.iterdir())},
    }
    (partial_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2),
                                             encoding="utf-8")
    partial_dir.rename(run_dir)

def connect_tracking(env_path: Path, tracking_uri: str = "databricks") -> None:
    """Point MLflow at the tracking server, with the credentials from the .env file.

    Args:
        env_path: The .env file with DATABRICKS_HOST and DATABRICKS_TOKEN.
        tracking_uri: "databricks", or a local URI in the tests.

    Raises:
        FileNotFoundError: If env_path does not exist.
        RuntimeError: If tracking_uri is "databricks" and a credential is missing.
    """
    if not env_path.is_file():
        raise FileNotFoundError(env_path)
    # the .env file wins over a stale variable exported in the shell
    load_dotenv(env_path, override=True)
    if tracking_uri == "databricks":
        missing = [name for name in DATABRICKS_VARIABLES if
                   not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"Missing in {env_path.name}: {missing}")
    # otherwise every run prints its URL, with the workspace host
    os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
    mlflow.set_tracking_uri(tracking_uri)

def log_run_files(run_dir: Path, experiment_name: str) -> str:
    """Upload a folder written by write_run_files as one MLflow run named after the folder.

    Calling it again for the same folder is safe: a finished upload returns its existing
    run, a failed one is uploaded again as a new run (e.g. from dt when the Job pod had no network).

    Args:
        run_dir: Folder written by write_run_files.
        experiment_name: EXPERIMENT_NAME or SMOKE_EXPERIMENT_NAME; has to exist already.

    Returns:
        The MLflow run id.

    Raises:
        RuntimeError: If the folder does not match its manifest or the experiment does not exist.
    """
    manifest_path = run_dir / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    found = {
        path.name: _sha256(path)
        for path in run_dir.iterdir()
        if path.name != MANIFEST_FILE
    }
    if found != manifest["files"]:
        raise RuntimeError(
            f"The files in {run_dir} do not match its manifest.")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment {experiment_name} does not exist.")

    manifest_sha256 = _sha256(manifest_path)
    finished_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`{MANIFEST_TAG}` = '{manifest_sha256}' and attributes.status = 'FINISHED'",
        output_format="list",
    )
    if finished_runs:
        return finished_runs[0].info.run_id

    context = json.loads((run_dir / CONTEXT_FILE).read_text(encoding="utf-8"))
    params = json.loads((run_dir / PARAMS_FILE).read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / METRICS_FILE).read_text(encoding="utf-8"))
    with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=run_dir.name,
            tags={MANIFEST_TAG: manifest_sha256},
    ) as run:
        mlflow.log_params({**context, **params})
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(run_dir))
    return run.info.run_id
