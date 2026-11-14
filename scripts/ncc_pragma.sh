#!/bin/bash
# Train and evaluate the PRAGMA-style model on NCC (Durham's GPU cluster).
#
# Submit from the repo root, on the NCC head node:
#
#     sbatch scripts/ncc_pragma.sh
#
# Before the first run, copy the processed data over (data/ is gitignored), e.g.
# from your laptop:
#
#     rsync -av data/processed/ <user>@ncc1.clients.dur.ac.uk:<repo path>/data/processed/
#
# Watch progress with `squeue -u $USER` and `tail -f slurm-<jobid>.out`.
# Options (set before sbatch, e.g. `TARGETS=vol sbatch scripts/ncc_pragma.sh`):
#     TARGETS         which targets to run            (default: "vol spread")
#     HORIZON         minutes ahead to predict        (default: 30)
#     SKIP_PRETRAIN   set to 1 to reuse runs/pragma/pretrain.pt
#
# If sbatch rejects the partition or QOS, run `sinfo` and `sacctmgr show qos`
# to see what your account may use, and change the two lines marked below.

#SBATCH --job-name=pragma
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:ampere:1    # NCC requires a GPU type: pascal, turing or ampere
#SBATCH --mem=28G
#SBATCH -p ug-gpu-small          # partition: ug- undergrad, tpg- taught PG, res- research
#SBATCH --qos=short              # QOS: must allow the time limit below
#SBATCH -t 08:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail

TARGETS="${TARGETS:-vol spread}"
HORIZON="${HORIZON:-30}"
SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"

# sbatch runs a copy of this script, so go to where the job was submitted from.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

if [ ! -f data/processed/instruments.csv ]; then
    echo "data/processed/ is missing: copy it over first (see the top of this script)" >&2
    exit 1
fi

# One-off environment setup. Kept separate from a laptop .venv.
if [ ! -x .venv-ncc/bin/python ]; then
    python3 -m venv .venv-ncc
    .venv-ncc/bin/python -m pip install --quiet --upgrade pip
    .venv-ncc/bin/python -m pip install --quiet -r requirements.txt
fi
PY=.venv-ncc/bin/python

echo "host: $(hostname)   started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$PY -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

if [ "$SKIP_PRETRAIN" != "1" ]; then
    echo; echo "=== pretrain ==="
    $PY -m models.pragma.train pretrain --preset gpu
fi

for target in $TARGETS; do
    echo; echo "=== $target: fine-tune from the pre-trained model ==="
    $PY -m models.pragma.train finetune --target "$target" --horizon "$HORIZON"

    echo; echo "=== $target: linear probe on frozen embeddings ==="
    $PY -m models.pragma.train probe --target "$target" --horizon "$HORIZON"

    echo; echo "=== $target: same model without pre-training ==="
    $PY -m models.pragma.train finetune --target "$target" --horizon "$HORIZON" \
        --from-scratch --preset gpu

    echo; echo "=== $target: comparison with XGBoost ==="
    $PY -m models.pragma.compare --target "$target" --horizon "$HORIZON"
done

echo; echo "finished: $(date)"
