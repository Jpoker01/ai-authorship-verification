#!/usr/bin/env bash
# Apply the given Job manifests, then show the live view of every run in this terminal.
# Ctrl+C or a closed terminal ends ONLY the view - the Jobs run on. The view alone, again:
#   .venv/bin/python k8s/watch_progress.py   (in ~/diploma_thesis/experiments)
# Run on the dt hub server from ~/diploma_thesis: bash experiments/k8s/launch.sh <manifest>...
set -euo pipefail
experiments="$(cd "$(dirname "$0")/.." && pwd)"
for manifest in "$@"; do
    kubectl apply -f "$manifest"
done
exec "$experiments/.venv/bin/python" "$experiments/k8s/watch_progress.py"