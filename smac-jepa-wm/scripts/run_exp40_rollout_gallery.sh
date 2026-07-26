#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JEPA_ROOT="${JEPA_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${CHECKPOINT:?Set CHECKPOINT to an external Exp-40 checkpoint}"
: "${MANIFEST:?Set MANIFEST to an external evaluation manifest}"
SPLIT="${SPLIT:-eval}"
MAX_BATCHES="${MAX_BATCHES:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TOP_K="${TOP_K:-5}"
SEED="${SEED:-123}"
DEVICE="${DEVICE:-cuda}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
: "${OUT_DIR:?Set OUT_DIR to a fresh rollout-gallery output directory}"
LOG_FILE="${LOG_FILE:-$OUT_DIR/run.log}"

if [[ ! -d "$JEPA_ROOT" ]]; then
  echo "ERROR: JEPA repository not found: $JEPA_ROOT" >&2
  exit 2
fi
[[ -f "$CHECKPOINT" ]] || { echo "ERROR: Exp40 checkpoint not found: $CHECKPOINT" >&2; exit 2; }
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: evaluation manifest not found: $MANIFEST" >&2
  exit 2
fi
if [[ ! -f "$JEPA_ROOT/eval_jepa_exp31_exp33_anchored.py" ]]; then
  echo "ERROR: anchored evaluator missing: $JEPA_ROOT/eval_jepa_exp31_exp33_anchored.py" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
cd "$JEPA_ROOT"

PYTHON_BIN="${PYTHON:-${PY:-python}}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: Python not found: $PYTHON_BIN" >&2; exit 2; }

printf '[repo]       %s\n' "$JEPA_ROOT"
printf '[checkpoint] %s\n' "$CHECKPOINT"
printf '[manifest]   %s (%s)\n' "$MANIFEST" "$SPLIT"
printf '[output]     %s\n' "$OUT_DIR"
printf '[scale]      max_batches=%s batch_size=%s rollout_starts_per_item=20 horizon=15\n' "$MAX_BATCHES" "$BATCH_SIZE"
printf '[python]     %s\n' "$PYTHON_BIN"

env LD_LIBRARY_PATH="" \
    PYTHONPATH="$JEPA_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$JEPA_ROOT/tools/eval_exp40_rollout_gallery.py" \
      --checkpoint "$CHECKPOINT" \
      --manifest "$MANIFEST" \
      --split "$SPLIT" \
      --out-dir "$OUT_DIR" \
      --horizon 15 \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --max-batches "$MAX_BATCHES" \
      --top-k "$TOP_K" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --amp \
      2>&1 | tee "$LOG_FILE"

printf '\n[OK] rollout gallery: %s\n' "$OUT_DIR"
