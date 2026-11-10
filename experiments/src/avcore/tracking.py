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