"""Write one Job manifest per ISG run (12 pairwise, 1 cosine) from traditional-job.yaml.

The runs load the vectors of the extraction notebook, so the generator refuses to write them until the
vector folder of the same round has its manifest. The manifests go into a new folder below the
git-ignored experiments/outputs, with a sha256 list that is checked right before the apply. Run it from
a launch checkout inside the home of the hub server; the Jobs mount that home and run the checkout the
generator ran from.

Usage (in the launch checkout, experiments folder):
    .venv/bin/python k8s/make_isg_jobs.py --home-claim <pvc> --launch-tag <tag> outputs/k8s/isg-runs
    .venv/bin/python k8s/make_isg_jobs.py --home-claim <pvc> --smoke g1 outputs/k8s/isg-runs-smoke-g1
    .venv/bin/python k8s/make_isg_jobs.py ... --memory isg-absdiff-lsvc=64Gi outputs/k8s/isg-retry-1

The home PVC is the claim whose volume appears in "df ~":
    kubectl get pvc -o custom-columns=NAME:.metadata.name,VOLUME:.spec.volumeName
"""

import argparse
import sys
from pathlib import Path

from make_traditional_jobs import (
    CLAIM_NAME_PATTERN,
    CLASSIFIERS,
    COMBINATIONS,
    JOB_NAME_PARTS,
    LIGHT_RESOURCES,
    OUTPUTS_DIR,
    SMOKE_RESOURCES,
    TEMPLATE_PATH,
    THREADED_RESOURCES,
    Run,
    apply_memory_overrides,
    check_runs,
    current_commit,
    make_run,
    pod_repo_path,
    write_manifests,
)

from avcore.classifiers import THREADED_CLASSIFIERS
from avcore.graph import MANIFEST_FILE

PAIRWISE_NOTEBOOK = "graph/isg_pairwise_classification.ipynb"
COSINE_NOTEBOOK = "graph/isg_cosine_similarity.ipynb"
# the cosine run holds the vectors and two similarity tables only
COSINE_RESOURCES = {"cpu": 2, "memory": "16Gi", "min_free_gb": 5}
# the folders isg_vector_extraction.ipynb writes
VECTORS_DIR = Path.home() / "isg_vectors_v2"
SMOKE_VECTORS_ROOT = Path.home() / "isg_vectors_v2_smoke"
TRADITIONAL_LABEL = "family: traditional"
GRAPH_LABEL = "family: graph"


def build_runs(smoke_tag: str) -> list[Run]:
    """Return the 13 runs in the order of the apply.

    Args:
        smoke_tag: "" for the launch, otherwise the RUN_TAG of the smoke round (e.g. "g1-1a2b3c4").

    Returns:
        12 pairwise runs (the longest classifiers first) and the cosine run.
    """
    runs = []
    for classifier in CLASSIFIERS:
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
            omp_threads = resources["cpu"]
            # each joblib thread of kNN starts its own OpenMP team
            if classifier == "k-nearest_neighbors":
                omp_threads = 1
            params = {
                "SMOKE": bool(smoke_tag),
                "COMBINATION": combination,
                "CLASSIFIER": classifier,
                "N_JOBS": n_jobs,
            }
            name_parts = [
                "isg",
                JOB_NAME_PARTS[combination],
                JOB_NAME_PARTS[classifier],
            ]
            run = make_run(
                name_parts=name_parts,
                notebook=PAIRWISE_NOTEBOOK,
                params=params,
                resources=resources,
                omp_threads=omp_threads,
                smoke_tag=smoke_tag,
            )
            runs.append(run)
    resources = COSINE_RESOURCES
    if smoke_tag:
        resources = SMOKE_RESOURCES
    params = {"SMOKE": bool(smoke_tag)}
    name_parts = ["isg", "cosine"]
    # not make_run - it adds N_TRIALS, which the cosine notebook does not have
    if smoke_tag:
        params["RUN_TAG"] = smoke_tag
        name_parts.append(smoke_tag)
    cosine_run = Run(
        job_name="-".join(name_parts),
        notebook=COSINE_NOTEBOOK,
        params=params,
        cpu=resources["cpu"],
        omp_threads=resources["cpu"],
        memory=resources["memory"],
        min_free_gb=resources["min_free_gb"],
    )
    runs.append(cosine_run)
    return runs


def check_vectors(smoke_tag: str) -> None:
    """Refuse to write the runs before the extraction of the same round is complete.

    Args:
        smoke_tag: "" for the launch, otherwise the RUN_TAG of the smoke round.

    Raises:
        SystemExit: If the vector folder has no manifest, so every run would fail at its load.
    """
    vectors_dir = VECTORS_DIR
    if smoke_tag:
        vectors_dir = SMOKE_VECTORS_ROOT / smoke_tag
    manifest_path = vectors_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        sys.exit(
            f"{manifest_path} is missing - the extraction Job of this round is not Complete",
        )


def relabel(template: str) -> str:
    """Return the traditional template with the label of the graph family.

    Raises:
        SystemExit: If the template does not hold the label exactly twice (Job and pod).
    """
    if template.count(TRADITIONAL_LABEL) != 2:
        sys.exit(f"{TEMPLATE_PATH.name} does not hold {TRADITIONAL_LABEL!r} twice")
    return template.replace(TRADITIONAL_LABEL, GRAPH_LABEL)


def main() -> None:
    """Write the manifests of the launch or of a smoke round from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--launch-tag",
        help="the tag of the launch checkout, e.g. v2-launch-5-graph",
    )
    mode.add_argument(
        "--smoke",
        choices=["g1", "g2"],
        help="a smoke round; the commit is added to RUN_TAG",
    )
    parser.add_argument(
        "--home-claim",
        required=True,
        help="the PVC of the hub server's home, mounted by every Job",
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
    check_vectors(smoke_tag)
    runs = build_runs(smoke_tag)
    check_runs(runs)
    apply_memory_overrides(runs, args.memory)
    shared_fields = {
        "{{LAUNCH_TAG}}": launch_tag,
        "{{HOME_CLAIM}}": args.home_claim,
        "{{REPO}}": pod_repo_path(),
    }
    template = relabel(TEMPLATE_PATH.read_text(encoding="utf-8"))
    write_manifests(
        runs=runs,
        template=template,
        shared_fields=shared_fields,
        out_dir=args.out_dir,
    )
    print(f"{len(runs)} manifests in {args.out_dir}")


if __name__ == "__main__":
    main()
