"""Upload one finished run folder of a launch to Databricks as one MLflow run.

The folder is first checked against its manifest and against the proof block of
the pod that wrote it; nothing is uploaded when a check fails. The upload goes
from a staging copy that holds only the manifest files, and a second call for
the same folder returns the same run.

Run on dt from ~/diploma_thesis with the venv of experiments, never with uv run:
    PYTHONDONTWRITEBYTECODE=1 experiments/.venv/bin/python tools/upload_run.py \\
        --run-dir ~/results_v2/<run> \\
        --pod-log ~/logs/<pod>-<commit>.log --launch-dir ~/launch/<tag> \\
        --tag <tag> --env-file ~/diploma_thesis/experiments/.env [--check-only]
"""

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO
from urllib.parse import urlparse

from avcore import tracking
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

EXPERIMENT_ID = "1405921385660545"
STAGING_ROOT = Path.home() / "upload_v2"
UPLOAD_LOG = Path.home() / "logs" / "uploads_v2.log"
FROZEN_LIST = "experiments/frozen_files.sha256"
TRACKING_FROZEN_PATH = "src/avcore/tracking.py"
# the clocks of the pod and of dt may differ a little
CLOCK_TOLERANCE = timedelta(minutes=5)
# a Job pod is <job>-<5 random characters>, its log <pod>-<short commit>.log
POD_LOG_PATTERN = re.compile(
    r"(?P<job>[a-z0-9-]+)-[a-z0-9]{5}-(?P<commit>[0-9a-f]{7,40})\.log",
)
FAMILY_PREFIXES = {
    "train-": "train",
    "embed-": "embed",
    "llm-": "llm",
    "trad-": "traditional",
}

PROOF_LINE_PATTERNS = {
    "pod": re.compile(
        r"pod=\S+ commit=(?P<commit>[0-9a-f]+) started=(?P<started>\S+) .*",
    ),
    "cgroup": re.compile(
        r"cpu\.max=\[(?P<cpu_max>[^\]]*)\] memory\.max=\[(?P<memory_max>[^\]]*)\]"
        r"( OMP_NUM_THREADS=\[(?P<omp>[^\]]*)\])?",
    ),
    "status": re.compile(r"status=\[(?P<status>.*)\]"),
    "tag": re.compile(r"tag=\[(?P<tags>[^\]]*)\] describe=\[.*\]"),
    "lock": re.compile(r"lock=(?P<sha256>[0-9a-f]{64})"),
    "venv": re.compile(r"venv=(?P<sha256>[0-9a-f]{64})"),
    # a differing file adds its sha256sum message after "bad=[", maybe over several lines
    "frozen": re.compile(
        r"frozen=\d+ entries, exit=(?P<exit>\d+), list=(?P<list_sha256>[0-9a-f]{64}) "
        r"bad=\[(?P<bad>.*)",
    ),
    "data_match": re.compile(r"data=(?P<result>MATCH|DIFFERENT)"),
    "papermill": re.compile(r"papermill started=(?P<started>\S+)"),
}
DATA_LINE_PATTERN = re.compile(r"(?P<sha256>[0-9a-f]{64}) (?P<name>\S+)")
# the gpu_check line - torch version, CUDA version, GPU name, compute capability
GPU_LINE_PATTERN = re.compile(r"\S+ \S+ (?P<gpu>.+) \(\d+, \d+\)")
PARAM_FIELD_PATTERN = re.compile(r"(?P<key>\w+)=\[(?P<value>[^\]]*)\]")
PARAMS_LINE_STARTS = ("params SMOKE=[", "params MODEL_ID=[", "NOTEBOOK=[")


class CheckFailed(Exception):
    """A check of the run folder or of its proof block failed."""


def parse_arguments() -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pod-log", type=Path, required=True)
    parser.add_argument("--launch-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--tracking-uri", default="databricks")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    """Return the content of a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    """Return the sha256 of a file."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    """Log a passed check, or raise CheckFailed with its message.

    Raises:
        CheckFailed: If the condition is False.
    """
    if not condition:
        raise CheckFailed(message)
    logger.info(f"OK    {message}")


def run_git(launch_dir: Path, *args: str) -> bytes:
    """Run a read-only git command in the launch checkout and return its output.

    Raises:
        subprocess.CalledProcessError: If git fails, e.g. the tag does not exist.
    """
    completed = subprocess.run(
        ["git", "-C", str(launch_dir), *args],
        capture_output=True,
        check=True,
        # git must not write lock files into a checkout a Job runs from
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    return completed.stdout


def redact(text: str) -> str:
    """Replace the Databricks host, its hostname and the token in a text."""
    for variable in tracking.DATABRICKS_VARIABLES:
        value = os.environ.get(variable)
        if value:
            text = text.replace(value, f"<{variable}>")
    # network errors name the bare hostname, without https://
    hostname = urlparse(os.environ.get("DATABRICKS_HOST", "")).hostname
    if hostname:
        text = text.replace(hostname, "<host>")
    return text


class RedactingFilter(logging.Filter):
    """Redact every log record, also those of MLflow and urllib3."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Replace the message of the record by its redacted text and keep it."""
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def check_run_folder(run_dir: Path) -> dict:
    """Check the folder against its manifest and return the manifest.

    Args:
        run_dir: Folder written by tracking.write_run_files in ~/results_v2.

    Returns:
        The content of manifest.json.

    Raises:
        CheckFailed: If the folder is unfinished, a manifest file is missing or
            changed, or the run belongs to another experiment than the real one.
    """
    partial_dir = run_dir.with_name(f"{run_dir.name}.partial")
    require(not partial_dir.exists(), f"F1 no unfinished copy {partial_dir.name}")
    manifest_path = run_dir / tracking.MANIFEST_FILE
    require(manifest_path.is_file(), f"F1 {run_dir.name} has its manifest")
    manifest = read_json(manifest_path)
    for name, expected_sha256 in manifest["files"].items():
        path = run_dir / name
        require(path.is_file(), f"F1 {name} exists")
        require(sha256_of(path) == expected_sha256, f"F1 {name} equals the manifest")
    for path in sorted(run_dir.iterdir()):
        is_known = path.name in manifest["files"] or path.name == tracking.MANIFEST_FILE
        # e.g. .ipynb_checkpoints of JupyterLab - left out of the staging copy
        if not is_known:
            logger.info(f"NOTE  {path.name} is not in the manifest, not uploaded")
    params = read_json(run_dir / tracking.PARAMS_FILE)
    experiment = params.get("mlflow_experiment")
    require(experiment == tracking.EXPERIMENT_NAME, f"F2 experiment {experiment}")
    return manifest


def find_family(job_name: str) -> str:
    """Return the experiment family of a Job name.

    Raises:
        CheckFailed: If the Job name belongs to no known family.
    """
    for prefix, family in FAMILY_PREFIXES.items():
        if job_name.startswith(prefix):
            return family
    raise CheckFailed(f"P0 Job {job_name} belongs to no known family")


def read_proof_line(line: str, proof: dict) -> None:
    """Store the fields of one line of the proof block in proof."""
    if line.startswith("START REFUSED"):
        proof["refused"] = True
    if line.startswith(PARAMS_LINE_STARTS):
        proof["params"] = line
    data_match = DATA_LINE_PATTERN.fullmatch(line)
    if data_match:
        proof["data"][data_match["name"]] = data_match["sha256"]
    gpu_match = GPU_LINE_PATTERN.fullmatch(line)
    if gpu_match:
        proof["gpu"] = gpu_match["gpu"]
    for name, pattern in PROOF_LINE_PATTERNS.items():
        match = pattern.fullmatch(line)
        if match:
            proof[name] = match.groupdict()


def read_proof_block(pod_log: Path) -> dict:
    """Return the fields of the proof block - the lines up to "papermill started=".

    Returns:
        One dict of named fields per name of PROOF_LINE_PATTERNS, plus "data"
        (file name to sha256), "params" (the params line), "gpu" (GPU name, only
        GPU Jobs) and "refused" (True after a START REFUSED line).
    """
    proof = {"data": {}, "params": "", "refused": False}
    with pod_log.open(encoding="utf-8", errors="replace") as log_file:
        for raw_line in log_file:
            read_proof_line(raw_line.rstrip("\n"), proof)
            if "papermill" in proof:
                break
    return proof


def require_proof_fields(proof: dict) -> None:
    """Check that every line of the proof block is there and the start was not refused.

    Raises:
        CheckFailed: If a line is missing or the pod refused its start.
    """
    for name in PROOF_LINE_PATTERNS:
        require(name in proof, f"P0 proof block has its {name} line")
    require(not proof["refused"], "P0 the pod did not refuse its start")


def read_pod_start(pod_log: Path) -> datetime | None:
    """Return the start time in the pod line of a pod log, None without one."""
    with pod_log.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = PROOF_LINE_PATTERNS["pod"].fullmatch(line.rstrip("\n"))
            if match:
                return datetime.fromisoformat(match["started"])
    return None


def check_later_pods(
    pod_log: Path,
    pod_started: datetime,
    job_name: str,
    run_started: datetime,
) -> None:
    """Check that no later pod of the same Job started before the run did.

    Args:
        pod_log: Log of the pod under check.
        pod_started: Start time of that pod, from its proof block.
        job_name: Name of the Job of the pod.
        run_started: Start time of the run.

    Raises:
        CheckFailed: If such a pod exists - it, not this one, wrote the folder.
    """
    for other_log in sorted(pod_log.parent.glob("*.log")):
        other_match = POD_LOG_PATTERN.fullmatch(other_log.name)
        # the job name is matched whole: llm-granite is not llm-granite-r2
        if (
            other_log == pod_log
            or other_match is None
            or other_match["job"] != job_name
        ):
            continue
        other_started = read_pod_start(other_log)
        if other_started is None:
            continue
        is_later_pod = pod_started < other_started <= run_started
        require(not is_later_pod, f"P0 no later pod {other_log.name} before the run")


def check_pod_times(
    pod_log: Path,
    pod_started: datetime,
    run_started: datetime,
    run_written: datetime,
) -> None:
    """Check that the run started after the pod and was written while the pod logged.

    Raises:
        CheckFailed: If the times do not fit.
    """
    log_written = datetime.fromtimestamp(pod_log.stat().st_mtime, tz=UTC)
    require(pod_started <= run_started, "P0 the pod started before the run")
    require(run_started <= run_written, "P0 the run started before its manifest")
    is_written_in_time = run_written <= log_written + CLOCK_TOLERANCE
    require(is_written_in_time, "P0 the manifest was written while the pod logged")


def check_run_printed(pod_log: Path, run_name: str) -> None:
    """Check that the notebook of this pod printed the path of this run folder.

    Raises:
        CheckFailed: If the path is not printed after "papermill started=".
    """
    path_pattern = re.compile(rf"/results_v2/{re.escape(run_name)}(\s|$)")
    is_notebook_output = False
    is_printed = False
    with pod_log.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if line.startswith("papermill started="):
                is_notebook_output = True
            if is_notebook_output and path_pattern.search(line):
                is_printed = True
                break
    require(is_printed, f"P0 the notebook of the pod printed {run_name}")


def check_commit(proof: dict, context: dict, launch_dir: Path, tag: str) -> None:
    """Check that the pod and the run ran the commit of the launch tag.

    Raises:
        CheckFailed: If a commit differs.
        subprocess.CalledProcessError: If the tag does not exist.
    """
    short_commit = proof["pod"]["commit"]
    require(context["commit"].startswith(short_commit), f"P1 pod commit {short_commit}")
    tag_commit = run_git(launch_dir, "rev-parse", f"{tag}^{{commit}}")
    is_tag_commit = context["commit"] == tag_commit.decode().strip()
    require(is_tag_commit, f"P1 run commit = tag {tag}")


def check_launch_state(proof: dict, frozen_list: bytes, tag: str) -> None:
    """Check the tag, the clean tree, the frozen files and the data of the pod.

    Raises:
        CheckFailed: If one of them is not as at the launch.
    """
    pod_tags = proof["tag"]["tags"].split()
    require(tag in pod_tags, f"P2 pod HEAD has the tag {tag}")
    require(proof["status"]["status"] == "", "P2 pod tree clean")
    require(proof["frozen"]["exit"] == "0", "P2 frozen files exit 0")
    require(proof["frozen"]["bad"] == "]", "P2 no frozen file differs")
    list_sha256 = hashlib.sha256(frozen_list).hexdigest()
    is_launch_list = proof["frozen"]["list_sha256"] == list_sha256
    require(is_launch_list, "P2 frozen list = the list at the tag")
    require(proof["data_match"]["result"] == "MATCH", "P2 data=MATCH")


def read_param_fields(params_line: str) -> dict:
    """Return the KEY=[value] fields of the params line."""
    return {
        match["key"]: match["value"]
        for match in PARAM_FIELD_PATTERN.finditer(params_line)
    }


def check_full_run(proof: dict, family: str, run_name: str) -> None:
    """Check in the params line that the pod ran a full run, not a smoke or pilot run.

    Raises:
        CheckFailed: If the params line belongs to a smoke or pilot run.
    """
    params_line = proof["params"]
    if family == "traditional":
        require('"SMOKE": false' in params_line, "P2 traditional SMOKE false")
        return
    fields = read_param_fields(params_line)
    if family == "llm":
        require(fields.get("MODE") == "test", "P2 llm MODE=[test]")
        require(fields.get("MAX_TOKENS") == "", "P2 llm MAX_TOKENS=[]")
        expected_run_tag = "r2" if "_r2_" in run_name else ""
        is_run_tag = fields.get("RUN_TAG") == expected_run_tag
        require(is_run_tag, f"P2 llm RUN_TAG=[{expected_run_tag}]")
        return
    require(fields.get("SMOKE") == "False", f"P2 {family} SMOKE=[False]")


def compute_venv_sha256(packages: dict) -> str:
    """Return the venv hash of the proof block, built from packages.json of the run."""
    lines = [f"{name.lower()}=={version}" for name, version in packages.items()]
    # the pod prints the sorted lines and one newline at the end
    text = "\n".join(sorted(lines)) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_environment(proof: dict, context: dict, packages: dict, family: str) -> None:
    """Check the lock, the venv and the cgroup limits of the pod against the run.

    Raises:
        CheckFailed: If one of them differs.
    """
    require(
        proof["lock"]["sha256"] == context["uv_lock_sha256"],
        "P3 lock = uv_lock_sha256",
    )
    venv_sha256 = compute_venv_sha256(packages)
    require(proof["venv"]["sha256"] == venv_sha256, "P4 venv = packages.json")
    cgroup = proof["cgroup"]
    require(cgroup["cpu_max"] == context["cgroup_cpu_max"], "P6 cpu.max = run")
    require(cgroup["memory_max"] == context["cgroup_memory_max"], "P6 memory.max = run")
    if family == "traditional":
        require(cgroup["omp"] == context["omp_num_threads"], "P6 OMP_NUM_THREADS = run")


def check_gpu(proof: dict, context: dict, family: str) -> None:
    """Check that a GPU run ran on the GPU of the proof block, a traditional one on none.

    Raises:
        CheckFailed: If the GPU differs.
    """
    gpu_name = context["gpu_name"]
    if family == "traditional":
        require(gpu_name == "none", "P6 no GPU in a traditional run")
        return
    require(proof.get("gpu") == gpu_name, f"P6 GPU {gpu_name}")


def read_frozen_list(frozen_list: bytes) -> dict:
    """Return the entries of frozen_files.sha256 as relative path to sha256."""
    frozen = {}
    for line in frozen_list.decode("utf-8").splitlines():
        parts = line.split()
        # comment lines start with "#"
        is_entry = len(parts) == 2 and len(parts[0]) == 64
        if is_entry:
            frozen[parts[1]] = parts[0]
    return frozen


def check_data(proof: dict, context: dict, frozen: dict) -> None:
    """Check every data file of the run against the proof block or the frozen list.

    Raises:
        CheckFailed: If a hash differs or a data file is in neither.
    """
    frozen_by_name = {Path(path).name: sha256 for path, sha256 in frozen.items()}
    for key, run_sha256 in context.items():
        if not key.startswith("data_sha256."):
            continue
        name = key.removeprefix("data_sha256.")
        # the LLM prompt files are not in the proof block, they are frozen files
        if name in proof["data"]:
            require(proof["data"][name] == run_sha256, f"P5 {name} = proof block")
        elif name in frozen_by_name:
            require(frozen_by_name[name] == run_sha256, f"P5 {name} = frozen list")
        else:
            raise CheckFailed(
                f"P5 {name} is in neither the proof block nor the frozen list",
            )


def check_tracking_module(frozen: dict) -> None:
    """Check that the tracking module this upload imports is the frozen one.

    Raises:
        CheckFailed: If it differs from its line in the frozen list.
    """
    require(TRACKING_FROZEN_PATH in frozen, "U2 tracking.py is in the frozen list")
    module_sha256 = sha256_of(Path(tracking.__file__))
    require(
        module_sha256 == frozen[TRACKING_FROZEN_PATH],
        "U2 tracking.py = frozen line",
    )


def check_proof(args: argparse.Namespace, manifest: dict) -> None:
    """Run every check of the proof block against the run folder.

    Raises:
        CheckFailed: If a check fails.
    """
    log_match = POD_LOG_PATTERN.fullmatch(args.pod_log.name)
    require(log_match is not None, f"P0 {args.pod_log.name} is a pod log name")
    family = find_family(log_match["job"])
    context = read_json(args.run_dir / tracking.CONTEXT_FILE)
    packages = read_json(args.run_dir / tracking.PACKAGES_FILE)
    frozen_list = run_git(args.launch_dir, "show", f"{args.tag}:{FROZEN_LIST}")
    frozen = read_frozen_list(frozen_list)
    proof = read_proof_block(args.pod_log)
    require_proof_fields(proof)
    run_started = datetime.fromisoformat(context["started_at"])
    run_written = datetime.fromisoformat(manifest["written_at"])
    pod_started = datetime.fromisoformat(proof["pod"]["started"])
    check_pod_times(args.pod_log, pod_started, run_started, run_written)
    check_later_pods(args.pod_log, pod_started, log_match["job"], run_started)
    check_run_printed(args.pod_log, args.run_dir.name)
    check_commit(proof, context, args.launch_dir, args.tag)
    check_launch_state(proof, frozen_list, args.tag)
    check_full_run(proof, family, args.run_dir.name)
    check_environment(proof, context, packages, family)
    check_gpu(proof, context, family)
    check_data(proof, context, frozen)
    check_tracking_module(frozen)


def take_lock(lock_file: IO) -> None:
    """Lock the open lock file, so a second upload can not run at the same time.

    Raises:
        CheckFailed: If another upload holds the lock.
    """
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise CheckFailed("another upload is running") from None


def check_staged_copy(
    staging_dir: Path,
    run_dir: Path,
    manifest: dict,
    names: list[str],
) -> None:
    """Check that an existing staging copy holds exactly the files of the run folder.

    Raises:
        CheckFailed: If a file is missing, extra or different.
    """
    staged_names = sorted(path.name for path in staging_dir.iterdir())
    require(staged_names == sorted(names), "U3 staging copy holds the manifest files")
    for name in names:
        # check_run_folder proved the run folder files equal to the manifest
        run_sha256 = manifest["files"].get(name) or sha256_of(run_dir / name)
        require(
            sha256_of(staging_dir / name) == run_sha256,
            f"U3 staged {name} = run folder",
        )


def stage_run_folder(run_dir: Path, manifest: dict) -> Path:
    """Copy the manifest files and the manifest into STAGING_ROOT / <run name>.

    Returns:
        The staging copy; an existing one is reused after its check.

    Raises:
        CheckFailed: If an existing staging copy differs from the run folder.
    """
    staging_dir = STAGING_ROOT / run_dir.name
    names = sorted(manifest["files"])
    names.append(tracking.MANIFEST_FILE)
    if staging_dir.exists():
        check_staged_copy(staging_dir, run_dir, manifest, names)
        return staging_dir
    partial_dir = staging_dir.with_name(f"{staging_dir.name}.partial")
    # left by a copy that was cut off - the lock keeps every other upload out
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir()
    for name in names:
        shutil.copy2(run_dir / name, partial_dir / name)
    partial_dir.rename(staging_dir)
    logger.info(f"OK    U3 staged {len(names)} files in {staging_dir}")
    return staging_dir


def connect(env_file: Path, tracking_uri: str) -> MlflowClient:
    """Connect to the tracking server and check that the experiment is the known one.

    Raises:
        CheckFailed: If the experiment has another id.
    """
    tracking.connect_tracking(env_file, tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(tracking.EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id if experiment else None
    require(experiment_id == EXPERIMENT_ID, f"0 experiment id {experiment_id}")
    return client


def verify_run(
    client: MlflowClient,
    run_id: str,
    staging_dir: Path,
    manifest_sha256: str,
) -> None:
    """Compare the uploaded run with the staging copy.

    Raises:
        CheckFailed: If the run differs from the staging copy.
    """
    run = client.get_run(run_id)
    params = read_json(staging_dir / tracking.CONTEXT_FILE)
    params.update(read_json(staging_dir / tracking.PARAMS_FILE))
    metrics = read_json(staging_dir / tracking.METRICS_FILE)
    require(run.info.status == "FINISHED", "V2 run FINISHED")
    require(run.info.run_name == staging_dir.name, "V2 run name = folder name")
    require(
        run.data.tags.get(tracking.MANIFEST_TAG) == manifest_sha256,
        "V2 manifest tag",
    )
    require(run.data.params == params, "V2 params = context + params")
    require(run.data.metrics == metrics, "V2 metrics = metrics.json")
    uploaded_sizes = {
        artifact.path: artifact.file_size for artifact in client.list_artifacts(run_id)
    }
    staged_sizes = {path.name: path.stat().st_size for path in staging_dir.iterdir()}
    require(uploaded_sizes == staged_sizes, "V3 artifacts = staging copy")


def check_other_runs(
    client: MlflowClient,
    run_id: str,
    run_name: str,
    manifest_sha256: str,
) -> None:
    """Check that one run is FINISHED and list every other run of the same name.

    Raises:
        CheckFailed: If more than one FINISHED run has the manifest tag.
    """
    tag_filter = f"tags.`{tracking.MANIFEST_TAG}` = '{manifest_sha256}'"
    finished_runs = client.search_runs(
        experiment_ids=[EXPERIMENT_ID],
        filter_string=f"{tag_filter} and attributes.status = 'FINISHED'",
    )
    require(len(finished_runs) == 1, "V4 exactly one FINISHED run")
    same_name_runs = client.search_runs(
        experiment_ids=[EXPERIMENT_ID],
        filter_string=f"attributes.run_name = '{run_name}'",
    )
    # a killed upload or an older launch with the same run name
    for run in same_name_runs:
        if run.info.run_id != run_id:
            logger.info(f"NOTE  V4 other run {run.info.run_id} {run.info.status}")


def write_upload_line(run_name: str, manifest_sha256: str, run_id: str) -> None:
    """Append the result of an upload to UPLOAD_LOG."""
    written_at = datetime.now(UTC).isoformat(timespec="seconds")
    line = (
        f"{written_at} {run_name} manifest={manifest_sha256} run={run_id} verify=OK\n"
    )
    with UPLOAD_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(line)


def upload(args: argparse.Namespace, run_dir: Path, manifest: dict) -> None:
    """Stage, upload and verify the run folder; the caller holds the lock.

    Raises:
        CheckFailed: If the staging copy or the uploaded run is not right.
    """
    staging_dir = stage_run_folder(run_dir, manifest)
    client = connect(args.env_file, args.tracking_uri)
    run_id = tracking.log_run_files(staging_dir, tracking.EXPERIMENT_NAME)
    manifest_sha256 = sha256_of(staging_dir / tracking.MANIFEST_FILE)
    verify_run(client, run_id, staging_dir, manifest_sha256)
    check_other_runs(client, run_id, staging_dir.name, manifest_sha256)
    write_upload_line(run_dir.name, manifest_sha256, run_id)
    logger.info(f"UPLOADED {run_dir.name} run={run_id}")


def run(args: argparse.Namespace) -> None:
    """Check the folder and its proof block, then upload it under the lock.

    Raises:
        CheckFailed: If a check fails.
    """
    # one resolved path for every later use, check_proof reads args.run_dir
    args.run_dir = run_dir = args.run_dir.resolve()
    manifest = check_run_folder(run_dir)
    check_proof(args, manifest)
    if args.check_only:
        logger.info("CHECKS PASSED - nothing uploaded (--check-only)")
        return
    STAGING_ROOT.mkdir(exist_ok=True)
    # closing the file at the end of the block releases the lock
    with (STAGING_ROOT / ".lock").open("w") as lock_file:
        take_lock(lock_file)
        upload(args, run_dir, manifest)


def main() -> int:
    """Run the upload and return 0 when done, 1 when a check or the upload failed."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[handler])
    # MLflow logs through a handler of its own, not through the root handler
    for mlflow_handler in logging.getLogger("mlflow").handlers:
        mlflow_handler.addFilter(RedactingFilter())
    args = parse_arguments()
    try:
        run(args)
    except CheckFailed as error:
        logger.error(f"FAIL  {error}")
        return 1
    # the last resort of the command line - a traceback would pass no filter
    except Exception as error:  # noqa: BLE001
        logger.error(f"FAIL  {type(error).__name__} - {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
