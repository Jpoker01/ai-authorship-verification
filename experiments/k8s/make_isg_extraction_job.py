"""Write the Job manifest of the ISG extraction notebook from isg-extraction-job.yaml.

The manifest goes into a new folder below the git-ignored experiments/outputs, with a sha256 list that
is checked right before the apply. Run it from a launch checkout inside the home of the hub server; the
Job mounts that home and runs the checkout the generator ran from.

Usage (in the launch checkout, experiments folder):
    .venv/bin/python k8s/make_isg_extraction_job.py --home-claim <pvc> --launch-tag <tag> outputs/k8s/isg-launch
    .venv/bin/python k8s/make_isg_extraction_job.py --home-claim <pvc> --smoke g1 outputs/k8s/isg-smoke-g1
    .venv/bin/python k8s/make_isg_extraction_job.py ... --memory isg-extraction=96Gi outputs/k8s/isg-retry-1

The home PVC is the claim whose volume appears in "df ~":
    kubectl get pvc -o custom-columns=NAME:.metadata.name,VOLUME:.spec.volumeName
"""

import argparse
import sys
from pathlib import Path

from make_traditional_jobs import (
    CLAIM_NAME_PATTERN,
    OUTPUTS_DIR,
    Run,
    apply_memory_overrides,
    check_runs,
    current_commit,
    pod_repo_path,
    write_manifests,
)

K8S_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = K8S_DIR / "isg-extraction-job.yaml"
NOTEBOOK = "graph/isg_vector_extraction.ipynb"
# one worker per CPU; a worker holds about 1,2 GB, the parent about 10 GB
RESOURCES = {"cpu": 16, "memory": "48Gi", "min_free_gb": 10}
SMOKE_RESOURCES = {"cpu": 1, "memory": "8Gi", "min_free_gb": 2}


def build_runs(smoke_tag: str) -> list[Run]:
    """Return the one run of the extraction notebook.

    Args:
        smoke_tag: "" for the launch, otherwise the RUN_TAG of the smoke round (e.g. "g1-1a2b3c4").
    """
    resources = RESOURCES
    if smoke_tag:
        resources = SMOKE_RESOURCES
    params = {"SMOKE": bool(smoke_tag), "WORKERS": resources["cpu"]}
    name_parts = ["isg", "extraction"]
    if smoke_tag:
        params["RUN_TAG"] = smoke_tag
        name_parts.append(smoke_tag)
    run = Run(
        job_name="-".join(name_parts),
        notebook=NOTEBOOK,
        params=params,
        cpu=resources["cpu"],
        omp_threads=1,
        memory=resources["memory"],
        min_free_gb=resources["min_free_gb"],
    )
    return [run]


def main() -> None:
    """Write the manifest of the launch or of a smoke round from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--launch-tag",
        help="the tag of the launch checkout, e.g. v2-launch-4-isg-extraction",
    )
    mode.add_argument(
        "--smoke",
        choices=["g1", "g2"],
        help="a smoke round; the commit is added to RUN_TAG",
    )
    parser.add_argument(
        "--home-claim",
        required=True,
        help="the PVC of the hub server's home, mounted by the Job",
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

    if not args.out_dir.resolve().is_relative_to(OUTPUTS_DIR):
        sys.exit(
            f"out_dir has to be below {OUTPUTS_DIR} - git ignores only that folder",
        )
    if not CLAIM_NAME_PATTERN.fullmatch(args.home_claim):
        sys.exit(f"{args.home_claim!r} is not a valid PVC name")
    smoke_tag = ""
    if args.smoke:
        smoke_tag = f"{args.smoke}-{current_commit()}"
    launch_tag = ""
    if args.launch_tag:
        launch_tag = args.launch_tag
    runs = build_runs(smoke_tag)
    check_runs(runs)
    apply_memory_overrides(runs, args.memory)
    shared_fields = {
        "{{LAUNCH_TAG}}": launch_tag,
        "{{HOME_CLAIM}}": args.home_claim,
        "{{REPO}}": pod_repo_path(),
    }
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    write_manifests(
        runs=runs,
        template=template,
        shared_fields=shared_fields,
        out_dir=args.out_dir,
    )
    print(f"{len(runs)} manifest in {args.out_dir}")


if __name__ == "__main__":
    main()
