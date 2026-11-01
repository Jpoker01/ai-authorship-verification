#!/usr/bin/env bash
# Run the thesis re-run notebooks one after another, in the background, with logs.
#
# Usage (from the repository root on Metacentrum):
#     chmod +x run_experiments.sh
#     nohup ./run_experiments.sh > run_experiments.log 2>&1 &
#
# Progress:  tail -f run_experiments.log          (overall)
#            tail -f experiments/logs/<notebook>.log   (current notebook)
#
# Each notebook is executed in place with nbconvert, so its outputs are saved into the .ipynb.
# A failed notebook does NOT stop the queue; the failure is written to the log and the next one starts.
# To skip notebooks, comment them out in the NOTEBOOKS list below.

set -u

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
NB_ROOT="$REPO_ROOT/experiments/notebooks"
LOG_DIR="$REPO_ROOT/experiments/logs"
mkdir -p "$LOG_DIR"

# Order: deployed model first, then the other traditional notebooks, then the cheap cosine baselines.
NOTEBOOKS=(
    "traditional/20251208_JP_traditional_word_TF-IDF_absolute_difference.ipynb"
    "traditional/20251208_JP_traditional_word_TF-IDF_element-wise.ipynb"
    "traditional/20251208_JP_traditional_word_BOW_absolute_difference.ipynb"
    "traditional/20251208_JP_traditional_word_BOW_element-wise.ipynb"
    "traditional/20251208_JP_traditional_word_TF-IDF_embedding.ipynb"
    "traditional/20251208_JP_traditional_word_BOW_embedding.ipynb"
    # Graph notebooks need the ISG vectors (*.npz) in data/01_processed first:
    # "graph/20251127_JP_integrated_syntactic_graph_absolute_difference.ipynb"
    # "graph/20251127_JP_integrated_syntactic_graph_element-wise.ipynb"
)

echo "=== run started $(date '+%Y-%m-%d %H:%M:%S') on $(hostname) ==="

for nb in "${NOTEBOOKS[@]}"; do
    dir="$NB_ROOT/$(dirname "$nb")"
    file="$(basename "$nb")"
    log="$LOG_DIR/${file%.ipynb}.log"

    if [[ ! -f "$dir/$file" ]]; then
        echo "[$(date '+%H:%M:%S')] MISSING  $nb"
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START    $nb"
    start=$(date +%s)

    # Notebooks resolve paths relative to their own folder, so run from there.
    (
        cd "$dir" && jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=-1 \
            --ExecutePreprocessor.kernel_name=python3 \
            "$file"
    ) > "$log" 2>&1
    status=$?

    elapsed=$(( $(date +%s) - start ))
    if [[ $status -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE     $nb  ($((elapsed / 3600))h $(((elapsed % 3600) / 60))min)"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED   $nb  exit=$status after $((elapsed / 60))min, see $log"
        grep -m1 -A3 "Error\|Exception" "$log" | sed 's/^/        /'
    fi
done

echo "=== run finished $(date '+%Y-%m-%d %H:%M:%S') ==="
