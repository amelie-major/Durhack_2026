#!/bin/bash
# Refresh the BETA hybrid predictions (XGBoost + PRAGMA embeddings) shown on the website.
#
#     scripts/run_hybrid_beta.sh          # from the repo root
#
# The hybrid model and its code live on the models/hybrid-h30 branch, not on main.
# This keeps a separate checkout of that branch (HYBRID_REPO), pulls it, and runs:
#
#     OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python -m models.pragma.hybrid_predict \
#         --model exported_models/hybrid_<target>_h30 --data data/processed --last 1 --device cpu
#
# for target = spread and vol. Results go to data/hybrid/{spread,vol}_now.csv, which
# `python -m live.server` serves at /api/hybrid (shown with a Beta badge).
#
# Options (environment variables):
#     HYBRID_REPO    checkout of models/hybrid-h30   (default: ../Durhack_hybrid)
#     HYBRID_REMOTE  where to clone it from          (default: this repo's origin URL)
#     PY             python with torch + xgboost     (default: .venv/bin/python, else python3)
#     TARGETS        which models to run             (default: "spread vol")
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HYBRID_REPO="${HYBRID_REPO:-$ROOT/../Durhack_hybrid}"
HYBRID_REMOTE="${HYBRID_REMOTE:-$(git -C "$ROOT" remote get-url origin 2>/dev/null || echo https://github.com/amelie-major/Durhack_2026.git)}"
BRANCH="models/hybrid-h30"
TARGETS="${TARGETS:-spread vol}"
if [ -z "${PY:-}" ]; then
    if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="python3"; fi
fi
case "$PY" in /*) ;; */*) PY="$ROOT/$PY" ;; esac   # make a relative path absolute

if [ ! -d "$ROOT/data/processed" ]; then
    echo "data/processed is missing: run loader.py first" >&2
    exit 1
fi

if [ -d "$HYBRID_REPO/.git" ] || [ -f "$HYBRID_REPO/.git" ]; then
    echo "updating $HYBRID_REPO ($BRANCH) ..."
    git -C "$HYBRID_REPO" pull --ff-only "$HYBRID_REMOTE" "$BRANCH"
else
    echo "cloning $BRANCH into $HYBRID_REPO ..."
    git clone --branch "$BRANCH" --single-branch "$HYBRID_REMOTE" "$HYBRID_REPO"
fi

mkdir -p "$ROOT/data/hybrid"
for target in $TARGETS; do
    echo "predicting $target (beta hybrid model) ..."
    (cd "$HYBRID_REPO" && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE "$PY" -m models.pragma.hybrid_predict \
        --model "exported_models/hybrid_${target}_h30" --data "$ROOT/data/processed" \
        --last 1 --out "$ROOT/data/hybrid/${target}_now.csv" --device cpu)
done
echo "done: data/hybrid/*_now.csv (served at /api/hybrid by python -m live.server)"
