#!/bin/bash
# XGBoost + PRAGMA embedding features on NCC. Reuses a pre-trained checkpoint, so
# run scripts/ncc_pragma.sh first (or at least its pretrain stage).
#
#     CHECKPOINT=runs/pragma/pretrain_large.pt sbatch scripts/ncc_hybrid.sh
#
# Options (set before sbatch):
#     CHECKPOINT   pre-trained weights to embed with   (default: runs/pragma/pretrain.pt)
#     TARGETS      which targets to run                (default: "vol spread")
#     HORIZON      minutes ahead to predict            (default: 30)
#     PCA          embedding components given to XGBoost, 0 = all  (default: 32)
#
# For each target: hybrid fit, then the comparison table (xgboost, pragma finetune/probe
# if their prediction files exist, and xgboost+pragma).

#SBATCH --job-name=hybrid
#SBATCH -N 1
#SBATCH -c 4                   # the short QOS rejects more CPUs per job
#SBATCH --gres=gpu:ampere:1    # embeddings are computed on the GPU
#SBATCH --mem=28G
#SBATCH -p tpg-gpu-small       # partition: ug- undergrad, tpg- taught PG, res- research
#SBATCH --qos=short            # QOS: must allow the time limit below
#SBATCH -t 04:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
export PYTHONUNBUFFERED=1

CHECKPOINT="${CHECKPOINT:-runs/pragma/pretrain.pt}"
TARGETS="${TARGETS:-vol spread}"
HORIZON="${HORIZON:-30}"
PCA="${PCA:-32}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

if [ ! -f data/processed/instruments.csv ]; then
    echo "data/processed/ is missing: copy it over first (see scripts/ncc_pragma.sh)" >&2
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "$CHECKPOINT not found: pre-train first (scripts/ncc_pragma.sh)" >&2
    exit 1
fi
if [ ! -x .venv-ncc/bin/python ]; then
    python3 -m venv .venv-ncc
    .venv-ncc/bin/python -m pip install --quiet --upgrade pip
    .venv-ncc/bin/python -m pip install --quiet -r requirements.txt
fi
PY=.venv-ncc/bin/python

echo "host: $(hostname)   started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "checkpoint: $CHECKPOINT   pca: $PCA   horizon: $HORIZON"

for target in $TARGETS; do
    echo; echo "=== $target: XGBoost + PRAGMA embeddings ==="
    $PY -m models.pragma.hybrid --target "$target" --horizon "$HORIZON" \
        --checkpoint "$CHECKPOINT" --pca "$PCA"

    echo; echo "=== $target: comparison ==="
    $PY -m models.pragma.compare --target "$target" --horizon "$HORIZON"
done

echo; echo "finished: $(date)"
