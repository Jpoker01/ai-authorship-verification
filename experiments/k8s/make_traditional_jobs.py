"""Write one Job manifest per run of the traditional notebooks from traditional-job.yaml.

The manifests go into a new folder below the git-ignored experiments/outputs, numbered in the order
they are applied (the longest runs first), with a sha256 list that is checked right before the apply.

Usage (in the launch checkout, experiments folder):
    .venv/bin/python k8s/make_traditional_jobs.py --launch-tag v2-launch-3-traditional outputs/k8s/launch
    .venv/bin/python k8s/make_traditional_jobs.py --smoke g1 outputs/k8s/smoke-g1
    .venv/bin/python k8s/make_traditional_jobs.py --launch-tag ... --memory trad-tfidf-absdiff-xgb=160Gi outputs/k8s/retry-1
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

K8S_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = K8S_DIR.parent / "outputs"
TEMPLATE_PATH = K8S_DIR / "traditional-job.yaml"
PAIRWISE_NOTEBOOK = "traditional/bow_tfidf_pairwise_classification.ipynb"
COSINE_NOTEBOOK = "traditional/bow_tfidf_cosine_similarity.ipynb"
REPRESENTATIONS = ["tfidf", "bow"]
COMBINATIONS = ["absolute_difference", "element-wise"]
# the order of the apply - the longest runs first
CLASSIFIERS = [
    "linear_svc",
    "k-nearest_neighbors",
    "xgboost",
    "random_forest",
    "logistic_regression",
    "multinomial_naive_bayes",
]
THREADED_CLASSIFIERS = ["k-nearest_neighbors", "random_forest", "xgboost"]
JOB_NAME_PARTS = {
    "absolute_difference": "absdiff",
    "element-wise": "elementwise",
    "linear_svc": "lsvc",
    "k-nearest_neighbors": "knn",
    "xgboost": "xgb",
    "random_forest": "rf",
    "logistic_regression": "lr",
    "multinomial_naive_bayes": "mnb",
}
LIGHT_RESOURCES = {"cpu": 2, "memory": "48Gi", "min_free_gb": 5}
THREADED_RESOURCES = {"cpu": 8, "memory": "96Gi", "min_free_gb": 20}
SMOKE_RESOURCES = {"cpu": 1, "memory": "8Gi", "min_free_gb": 2}
SMOKE_TRIALS = 4
PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")
# a Kubernetes Job name - lowercase letters, digits and "-", at most 63 characters
JOB_NAME_PATTERN = re.compile(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")


@dataclass
class Run:
    """One run of a notebook - its Job name, notebook, parameters and resources."""

    job_name: str
    notebook: str
    params: dict[str, object]
    cpu: int
    memory: str
    min_free_gb: int


def make_run(
    name_parts: list[str],
    notebook: str,
    params: dict[str, object],
    resources: dict[str, object],
    smoke_tag: str,
) -> Run:
    """Return one run, with the smoke settings added when smoke_tag is set."""
    if smoke_tag:
        params = dict(params)
        params["N_TRIALS"] = SMOKE_TRIALS
        params["RUN_TAG"] = smoke_tag
        name_parts = name_parts + [smoke_tag]
    return Run(
        job_name="-".join(name_parts),
        notebook=notebook,
        params=params,
        cpu=resources["cpu"],
        memory=resources["memory"],
        min_free_gb=resources["min_free_gb"],
    )


def build_runs(smoke_tag: str) -> list[Run]:
    """Return the 26 runs in the order of the apply.

    Args:
        smoke_tag: "" for the launch, otherwise the RUN_TAG of the smoke round (e.g. "g1-1a2b3c4").

    Returns:
        24 pairwise runs (the longest classifiers first) and the 2 cosine runs.
    """
    runs = []
    for classifier in CLASSIFIERS:
        for representation in REPRESENTATIONS:
            for combination in COMBINATIONS:
                if smoke_tag:
                    resources = SMOKE_RESOURCES
                elif classifier in THREADED_CLASSIFIERS:
                    resources = THREADED_RESOURCES
                else:
                    resources = LIGHT_RESOURCES
                n_jobs = 1
                if classifier in THREADED_CLASSIFIERS:
                    n_jobs = resources["cpu"]
                params = {
                    "SMOKE": bool(smoke_tag),
                    "REPRESENTATION": representation,
                    "COMBINATION": combination,
                    "CLASSIFIER": classifier,
                    "N_JOBS": n_jobs,
                }
                name_parts = [
                    "trad",
                    representation,
                    JOB_NAME_PARTS[combination],
                    JOB_NAME_PARTS[classifier],
                ]
                run = make_run(
                    name_parts=name_parts,
                    notebook=PAIRWISE_NOTEBOOK,
                    params=params,
                    resources=resources,
                    smoke_tag=smoke_tag,
                )
                runs.append(run)
    for representation in REPRESENTATIONS:
        resources = LIGHT_RESOURCES
        if smoke_tag:
            resources = SMOKE_RESOURCES
        params = {"SMOKE": bool(smoke_tag), "REPRESENTATION": representation}
        name_parts = ["trad", representation, "cosine"]
        run = make_run(
            name_parts=name_parts,
            notebook=COSINE_NOTEBOOK,
            params=params,
            resources=resources,
            smoke_tag=smoke_tag,
        )
        runs.append(run)
    return runs


def apply_memory_overrides(runs: list[Run], overrides: list[str]) -> None:
    """Set the memory of the named runs, e.g. "trad-tfidf-absdiff-xgb=160Gi".

    Raises:
        SystemExit: If an override names no run or has no "=".
    """
    runs_by_name = {}
    for run in runs:
        runs_by_name[run.job_name] = run
    for override in overrides:
        job_name, separator, memory = override.partition("=")
        has_separator = separator == "="
        is_generated_run = job_name in runs_by_name
        is_memory = re.fullmatch(r"\d+Gi", memory) is not None
        if not (has_separator and is_generated_run and is_memory):
            sys.exit(
                f"--memory {override!r} - expected <job name>=<n>Gi of a generated run",
            )
        runs_by_name[job_name].memory = memory


def render(template: str, run: Run, launch_tag: str) -> str:
    """Return the manifest of one run - the template with every {{...}} field filled."""
    fields = {
        "{{JOB_NAME}}": run.job_name,
        "{{CPU}}": str(run.cpu),
        "{{MEMORY}}": run.memory,
        "{{LAUNCH_TAG}}": launch_tag,
        "{{NOTEBOOK}}": run.notebook,
        "{{PARAMS_JSON}}": json.dumps(run.params),
        "{{MIN_FREE_GB}}": str(run.min_free_gb),
    }
    manifest = template
    for field, value in fields.items():
        manifest = manifest.replace(field, value)
    leftover = PLACEHOLDER.findall(manifest)
    if leftover:
        sys.exit(f"{run.job_name}: unfilled fields {leftover}")
    return manifest


def check_runs(runs: list[Run]) -> None:
    """Refuse Job names Kubernetes would reject and two runs with the same name.

    Raises:
        SystemExit: On an invalid or repeated Job name.
    """
    names = set()
    for run in runs:
        if not JOB_NAME_PATTERN.fullmatch(run.job_name):
            sys.exit(f"{run.job_name!r} is not a valid Job name")
        if run.job_name in names:
            sys.exit(f"{run.job_name!r} twice")
        names.add(run.job_name)


def check_threaded_classifiers() -> None:
    """Refuse a THREADED_CLASSIFIERS list that differs from the one of avcore.classifiers."""
    from avcore import classifiers

    if sorted(THREADED_CLASSIFIERS) != sorted(classifiers.THREADED_CLASSIFIERS):
        sys.exit(
            "THREADED_CLASSIFIERS differs from avcore.classifiers.THREADED_CLASSIFIERS",
        )


def current_commit() -> str:
    """Return the short hash of the commit the generator runs from."""
    result = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=K8S_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def write_manifests(
    runs: list[Run],
    template: str,
    launch_tag: str,
    out_dir: Path,
) -> None:
    """Write the numbered manifests and their sha256 list into a new folder.

    Raises:
        SystemExit: If out_dir already exists.
    """
    if out_dir.exists():
        sys.exit(
            f"{out_dir} exists - a generator run never overwrites earlier manifests",
        )
    out_dir.mkdir(parents=True)
    hash_lines = []
    for position, run in enumerate(runs, start=1):
        path = out_dir / f"{position:02d}-{run.job_name}.yaml"
        manifest = render(template, run, launch_tag)
        path.write_text(manifest, encoding="utf-8")
        digest = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
        hash_lines.append(f"{digest}  {path.name}\n")
    (out_dir / "manifests.sha256").write_text("".join(hash_lines), encoding="utf-8")


def main() -> None:
    """Write the manifests of one launch or smoke round from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--launch-tag",
        help="the tag of the launch checkout, e.g. v2-launch-3-traditional",
    )
    mode.add_argument(
        "--smoke",
        choices=["g1", "g2"],
        help="a smoke round; the commit is added to RUN_TAG",
    )
    parser.add_argument(
        "--memory",
        action="append",
        default=[],
        help="<job name>=<n>Gi, repeatable",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="a new folder below experiments/outputs",
    )
    args = parser.parse_args()

    check_threaded_classifiers()
    if not args.out_dir.resolve().is_relative_to(OUTPUTS_DIR):
        sys.exit(
            f"out_dir has to be below {OUTPUTS_DIR} - git ignores only that folder",
        )
    smoke_tag = ""
    if args.smoke:
        smoke_tag = f"{args.smoke}-{current_commit()}"
    launch_tag = ""
    if args.launch_tag:
        launch_tag = args.launch_tag
    runs = build_runs(smoke_tag)
    check_runs(runs)
    apply_memory_overrides(runs, args.memory)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    write_manifests(
        runs=runs,
        template=template,
        launch_tag=launch_tag,
        out_dir=args.out_dir,
    )
    print(f"{len(runs)} manifests in {args.out_dir}")


if __name__ == "__main__":
    main()
