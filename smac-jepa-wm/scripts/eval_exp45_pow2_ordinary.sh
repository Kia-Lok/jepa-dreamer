#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON:-${PY:-python}}"

: "${CHECKPOINT:=${1:-}}"
: "${CHECKPOINT:?Set CHECKPOINT to an external Exp-45 checkpoint}"
: "${MANIFEST:?Set MANIFEST to an external evaluation manifest}"
SPLIT="${SPLIT:-eval}"
OUT_DIR="${OUT_DIR:-eval_outputs/exp45_pow2_direct/ordinary}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_BATCHES="${MAX_BATCHES:-300}"
PROBE_EPOCHS="${PROBE_EPOCHS:-5}"
PROBE_MAX_BATCHES="${PROBE_MAX_BATCHES:-100}"
PROBE_SAMPLES="${PROBE_SAMPLES:-8000}"

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || fail "Checkpoint not found: $CHECKPOINT"
[[ -f "$MANIFEST" ]] || fail "Manifest not found: $MANIFEST"
mkdir -p "$OUT_DIR" "$OUT_DIR/r2aware_probes" "$OUT_DIR/hidden_compatible_probes"

find_ordinary_eval() {
  local path
  for path in \
    "${ORDINARY_EVAL:-}" \
    eval_rnn_seqmem_dreamer_probe_r2aware.py \
    tools/eval_rnn_seqmem_dreamer_probe_r2aware.py \
    eval_rnn_seqmem_dreamer_probe.py; do
    if [[ -n "$path" && -f "$path" ]]; then echo "$path"; return 0; fi
  done
  return 1
}
ORDINARY_EVAL_PATH="$(find_ordinary_eval || true)"
[[ -n "$ORDINARY_EVAL_PATH" ]] || fail "Could not find eval_rnn_seqmem_dreamer_probe_r2aware.py"
find_anchored_eval() {
  local path
  for path in     "${ANCHORED_EVAL:-}"     eval_jepa_exp31_exp33_anchored.py     tools/eval_jepa_exp31_exp33_anchored.py; do
    if [[ -n "$path" && -f "$path" ]]; then echo "$path"; return 0; fi
  done
  return 1
}
ANCHORED_EVAL_PATH="$(find_anchored_eval || true)"
[[ -n "$ANCHORED_EVAL_PATH" ]] || fail "Could not find eval_jepa_exp31_exp33_anchored.py"

SANITIZED="$OUT_DIR/exp45_exp40_base_eval.pt"
"$PYTHON_BIN" tools/make_exp40_eval_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --out "$SANITIZED"

# Trusted apples-to-apples Exp40 recursive evaluation at H=5.
R2_AWARE_BASE_EVAL="$ORDINARY_EVAL_PATH" \
"$PYTHON_BIN" tools/eval_rnn_seqmem_dreamer_probe_r2aware_anchored.py \
  --manifest "$MANIFEST" --split "$SPLIT" \
  --checkpoint "$SANITIZED" \
  --out-dir "$OUT_DIR/recursive_h5" \
  --summary-out "$OUT_DIR/recursive_h5_summary.csv" \
  --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --device "$DEVICE" \
  --max-batches "$MAX_BATCHES" --window-mode sequential \
  --target-mode full --enemy-visibility-mask --eval-rollout-horizon 5 \
  --diagnostics --probe-decoder --probe-dir "$OUT_DIR/r2aware_probes" \
  --probe-epochs "$PROBE_EPOCHS" \
  --probe-max-batches-per-epoch "$PROBE_MAX_BATCHES" \
  --probe-samples-per-epoch "$PROBE_SAMPLES" \
  --thresholds 0.01 0.05 0.10

# The targeted hidden evaluator loads a probe with the meaningful_features_v2
# naming/feature-mask contract produced by this anchored evaluator.
"$PYTHON_BIN" "$ANCHORED_EVAL_PATH" \
  --manifest "$MANIFEST" --split "$SPLIT" \
  --checkpoint "$SANITIZED" \
  --out-dir "$OUT_DIR/anchored_h5" \
  --summary-out "$OUT_DIR/anchored_h5_summary.csv" \
  --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --device "$DEVICE" \
  --max-batches "$MAX_BATCHES" --window-mode sequential \
  --target-mode full --enemy-visibility-mask --eval-rollout-horizon 5 \
  --diagnostics --probe-decoder --probe-dir "$OUT_DIR/hidden_compatible_probes" \
  --probe-epochs "$PROBE_EPOCHS" \
  --probe-max-batches-per-epoch "$PROBE_MAX_BATCHES" \
  --probe-samples-per-epoch "$PROBE_SAMPLES" \
  --thresholds 0.01 0.05 0.10

# New branch: direct powers and arbitrary binary compositions.
"$PYTHON_BIN" tools/eval_pow2_direct.py \
  --manifest "$MANIFEST" --split "$SPLIT" \
  --checkpoint "$CHECKPOINT" \
  --base-evaluator "$ORDINARY_EVAL_PATH" \
  --out "$OUT_DIR/pow2_direct_and_binary.json" \
  --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --device "$DEVICE" \
  --max-batches "$MAX_BATCHES" \
  --power-horizons "1 2 4 8 16" \
  --binary-horizons "3 5 9 13 15 16" \
  --exact-horizons "3 5 9 13 15"

echo "[OK] ordinary recursive + direct evaluation: $OUT_DIR"
