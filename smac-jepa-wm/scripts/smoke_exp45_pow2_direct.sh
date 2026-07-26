#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

: "${MANIFEST:?Set MANIFEST to an external JEPA dataset manifest}"
: "${EXP40_CHECKPOINT:?Set EXP40_CHECKPOINT to an external Exp-40 checkpoint}"
OUT_DIR="${OUT_DIR:-runs/smoke_exp45_pow2_direct}"
rm -rf "$OUT_DIR"

MANIFEST="$MANIFEST" \
EXP40_CHECKPOINT="$EXP40_CHECKPOINT" \
OUT_DIR="$OUT_DIR" \
EPOCHS=1 \
SAMPLES_PER_EPOCH="${SMOKE_SAMPLES_PER_EPOCH:-32}" \
BATCH_SIZE="${SMOKE_BATCH_SIZE:-4}" \
NUM_WORKERS=0 \
DEVICE="${SMOKE_DEVICE:-cuda}" \
WANDB=0 \
POW2_WARMUP_STEPS=0 \
POW2_HIDDEN_DIM="${SMOKE_POW2_HIDDEN_DIM:-96}" \
POW2_DIRECT_WEIGHT=0.10 \
POW2_COMPOSITION_WEIGHT=0.05 \
POW2_SHARED_HEAD_WEIGHT=0.10 \
bash scripts/run_exp45_pow2_direct_train.sh

echo "[OK] Pow2 smoke completed"
