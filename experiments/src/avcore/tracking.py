"""Run records of the v2 experiments - context, local result files and their MLflow upload."""

import hashlib
import importlib.metadata
import importlib.util
import json
import math
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
        params: Commit, hashes, versions and hardware, logged as MLflow params.
        packages: Every installed distribution with its version, saved as a file.
    """

    params: dict[str, str]
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
    params = {
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
    return RunContext(params=params, packages=dict(sorted(packages.items())))