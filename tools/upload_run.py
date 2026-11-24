"""Upload every finished run folder in ~/results_v2 to Databricks, one MLflow run per folder.

A folder that was uploaded before returns its existing run, so the script can run again and
again. On dt, from ~/diploma_thesis, every 30min:
    nohup bash -c 'while true; do MLFLOW_DISABLE_TELEMETRY=true PYTHONDONTWRITEBYTECODE=1 \\
        experiments/.venv/bin/python tools/upload_run.py; sleep 1800; done' \\
        >> ~/logs/uploads_v2.log 2>&1 &
"""

import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from avcore import tracking
from mlflow.exceptions import MlflowException

RESULTS_DIR = Path.home() / "results_v2"
ENV_FILE = Path.home() / "diploma_thesis" / "experiments" / ".env"

logger = logging.getLogger(__name__)


def upload_run_dir(run_dir: Path) -> str:
    """Upload one run folder to the MLflow experiment named in its params.json.

    Returns:
        The MLflow run id.
    """
    manifest_path = run_dir / tracking.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = json.loads((run_dir / tracking.PARAMS_FILE).read_text(encoding="utf-8"))
    # opening a folder in JupyterLab adds .ipynb_checkpoints, which log_run_files refuses
    with tempfile.TemporaryDirectory() as staging_root:
        staging_dir = Path(staging_root) / run_dir.name
        staging_dir.mkdir()
        shutil.copy2(manifest_path, staging_dir)
        for name in manifest["files"]:
            shutil.copy2(run_dir / name, staging_dir)
        return tracking.log_run_files(staging_dir, params["mlflow_experiment"])


def main() -> int:
    """Upload every finished run folder and log one line per folder.

    Returns:
        1 if any folder failed, otherwise 0.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tracking.connect_tracking(ENV_FILE)
    failed = 0
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        # write_run_files writes into <run>.partial and renames it once complete
        if run_dir.name.endswith(".partial"):
            continue
        if not (run_dir / tracking.MANIFEST_FILE).is_file():
            continue
        try:
            run_id = upload_run_dir(run_dir)
        except (MlflowException, OSError, RuntimeError, KeyError, ValueError) as error:
            failed += 1
            logger.error(f"FAIL  {run_dir.name} - {type(error).__name__} - {error}")
            continue
        logger.info(f"OK    {run_dir.name} run={run_id}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
