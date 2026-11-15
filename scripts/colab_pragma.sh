#!/bin/bash
# Train and evaluate the PRAGMA-style model on Google Colab.
#
# In a Colab notebook (Runtime > Change runtime type > GPU), run these cells:
#
#     from google.colab import drive
#     drive.mount('/content/drive')
#
#     !git clone -b feat/pragma https://github.com/amelie-major/Durhack_2026.git
#     %cd Durhack_2026
#     !bash scripts/colab_pragma.sh
#
# Before the first run, upload the laptop's data/processed/ folder to Google
# Drive as  MyDrive/durhack/processed/  (it is gitignored, so the clone has no data).
#
# Checkpoints, predictions and a log are written to MyDrive/durhack/runs/, so
# they survive the Colab session ending. If the session dies after pre-training,
# rerun with SKIP_PRETRAIN=1 to carry on from the saved checkpoint.
#
# Options (put them before bash, e.g. `!TARGETS=vol bash scripts/colab_pragma.sh`):
#     DRIVE_DIR       Drive folder for data and results (default: /content/drive/MyDrive/durhack)
#     TARGETS         which targets to run               (default: "vol spread")
#     HORIZON         minutes ahead to predict           (default: 30)
#     SKIP_PRETRAIN   set to 1 to reuse the saved pretrain.pt
#     PRETRAIN_STEPS  shorten or lengthen pre-training   (default: the preset's 8,000)
#     PRESET          colab (120-bar context, faster), gpu (240-bar, as on NCC) or smoke
#                                                        (default: colab)
#
# If the GPU runs out of memory, halve batch_size in models/pragma/config.py.

set -euo pipefail

DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/durhack}"
TARGETS="${TARGETS:-vol spread}"
HORIZON="${HORIZON:-30}"
SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-}"
PRESET="${PRESET:-colab}"

cd "$(dirname "$0")/.."

# --- results: keep them on Drive so a disconnect does not lose them ----------
if [ -d "$DRIVE_DIR" ]; then
    mkdir -p "$DRIVE_DIR/runs" runs
    if [ ! -L runs/pragma ]; then
        rm -rf runs/pragma
        ln -s "$DRIVE_DIR/runs" runs/pragma
    fi
    echo "results: $DRIVE_DIR/runs"
else
    mkdir -p runs/pragma
    echo "WARNING: $DRIVE_DIR not found (Drive not mounted?)." \
         "Results stay in runs/pragma and are lost when the session ends." >&2
fi
exec > >(tee -a runs/pragma/colab.log) 2>&1

# --- data: copy from Drive to local disk once (reading from Drive is slow) ----
if [ ! -f data/processed/instruments.csv ]; then
    if [ -f "$DRIVE_DIR/processed/instruments.csv" ]; then
        echo "copying data from $DRIVE_DIR/processed ..."
        mkdir -p data
        cp -r "$DRIVE_DIR/processed" data/processed
    else
        echo "No data found. Upload the laptop's data/processed/ folder to" \
             "$DRIVE_DIR/processed/ (or put it at data/processed/ here)." >&2
        exit 1
    fi
fi

python -m pip install --quiet -r requirements.txt

echo "started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
if [ "$PRESET" != "smoke" ] && ! python -c "import torch, sys; sys.exit(not torch.cuda.is_available())"; then
    echo "No GPU. In Colab choose Runtime > Change runtime type > GPU, then rerun." >&2
    exit 1
fi

if [ "$SKIP_PRETRAIN" != "1" ]; then
    echo; echo "=== pretrain ==="
    python -m models.pragma.train pretrain --preset "$PRESET" ${PRETRAIN_STEPS:+--steps "$PRETRAIN_STEPS"}
fi

for target in $TARGETS; do
    echo; echo "=== $target: fine-tune from the pre-trained model ==="
    python -m models.pragma.train finetune --target "$target" --horizon "$HORIZON"

    echo; echo "=== $target: linear probe on frozen embeddings ==="
    python -m models.pragma.train probe --target "$target" --horizon "$HORIZON"

    echo; echo "=== $target: same model without pre-training ==="
    python -m models.pragma.train finetune --target "$target" --horizon "$HORIZON" \
        --from-scratch --preset "$PRESET"

    echo; echo "=== $target: comparison with XGBoost ==="
    python -m models.pragma.compare --target "$target" --horizon "$HORIZON"
done

echo; echo "finished: $(date)"
