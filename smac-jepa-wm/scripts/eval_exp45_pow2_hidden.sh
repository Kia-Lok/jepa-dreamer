#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON:-${PY:-python}}"

: "${CHECKPOINT:=${1:-}}"
: "${CHECKPOINT:?Set CHECKPOINT to an external Exp-45 checkpoint}"
: "${MANIFEST:?Set MANIFEST to an external evaluation manifest}"
SPLIT="${SPLIT:-eval}"
OUT_DIR="${OUT_DIR:-eval_outputs/exp45_pow2_direct/hidden}"
ORDINARY_OUT_DIR="${ORDINARY_OUT_DIR:-eval_outputs/exp45_pow2_direct/ordinary}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DIRECT_MAX_BATCHES="${DIRECT_MAX_BATCHES:-300}"
NATURAL_HIDDEN_TARGETS="${NATURAL_HIDDEN_TARGETS:-3000}"
NATURAL_HIDDEN_SCAN_BATCHES="${NATURAL_HIDDEN_SCAN_BATCHES:-5000}"
CONTROLLED_MAX_BATCHES="${CONTROLLED_MAX_BATCHES:-150}"
CONTROLLED_TARGETS="${CONTROLLED_TARGETS:-10000}"

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || fail "Checkpoint not found: $CHECKPOINT"
[[ -f "$MANIFEST" ]] || fail "Manifest not found: $MANIFEST"
mkdir -p "$OUT_DIR"

find_file() {
  local explicit="$1"; shift
  local path
  for path in "$explicit" "$@"; do
    if [[ -n "$path" && -f "$path" ]]; then echo "$path"; return 0; fi
  done
  return 1
}

ORDINARY_EVAL_PATH="$(find_file "${ORDINARY_EVAL:-}" \
  eval_rnn_seqmem_dreamer_probe_r2aware.py \
  tools/eval_rnn_seqmem_dreamer_probe_r2aware.py \
  eval_rnn_seqmem_dreamer_probe.py || true)"
HIDDEN_EVAL_PATH="$(find_file "${HIDDEN_EVAL:-}" \
  eval_jepa_hidden_belief_exp31_exp33.py \
  tools/eval_jepa_hidden_belief_exp31_exp33.py || true)"
[[ -n "$ORDINARY_EVAL_PATH" ]] || fail "Could not find ordinary evaluator needed by direct metrics"
[[ -n "$HIDDEN_EVAL_PATH" ]] || fail "Could not find eval_jepa_hidden_belief_exp31_exp33.py"

SANITIZED="${SANITIZED:-$ORDINARY_OUT_DIR/exp45_exp40_base_eval.pt}"
if [[ ! -f "$SANITIZED" ]]; then
  mkdir -p "$(dirname "$SANITIZED")"
  "$PYTHON_BIN" tools/make_exp40_eval_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --out "$SANITIZED"
fi

PROBE_DIR="${PROBE_DIR:-$ORDINARY_OUT_DIR/hidden_compatible_probes}"
if ! find "$PROBE_DIR" -type f -name '*.pt' -print -quit 2>/dev/null | grep -q .; then
  fail "No probe checkpoint found in $PROBE_DIR. Run scripts/eval_exp45_pow2_ordinary.sh first."
fi

# Trusted hidden-belief suite for the unchanged Exp40 recursive branch.
"$PYTHON_BIN" "$HIDDEN_EVAL_PATH" \
  --manifest "$MANIFEST" --split "$SPLIT" \
  --checkpoint "$SANITIZED" \
  --out-dir "$OUT_DIR/recursive_hidden_h5" \
  --probe-dir "$PROBE_DIR" \
  --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --device "$DEVICE" \
  --eval-rollout-horizon 5 --target-mode full \
  --natural-hidden-eval \
  --natural-hidden-target-entity-times "$NATURAL_HIDDEN_TARGETS" \
  --natural-hidden-max-scan-batches "$NATURAL_HIDDEN_SCAN_BATCHES" \
  --controlled-occlusion-eval \
  --controlled-occlusion-max-batches "$CONTROLLED_MAX_BATCHES" \
  --controlled-occlusion-target-entity-times "$CONTROLLED_TARGETS" \
  --controlled-occlusion-spans 1 3 5 \
  --thresholds 0.01 0.05 0.10

# The direct evaluator reports a dedicated natural_hidden_enemy subset at every
# trained power and every requested binary horizon. Controlled occlusion remains
# the standard recursive H=5 suite above; it is not silently conflated with the
# direct branch.
"$PYTHON_BIN" tools/eval_pow2_direct.py \
  --manifest "$MANIFEST" --split "$SPLIT" \
  --checkpoint "$CHECKPOINT" \
  --base-evaluator "$ORDINARY_EVAL_PATH" \
  --out "$OUT_DIR/pow2_natural_hidden.json" \
  --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --device "$DEVICE" \
  --max-batches "$DIRECT_MAX_BATCHES" \
  --power-horizons "1 2 4 8 16" \
  --binary-horizons "3 5 9 13 15 16" \
  --exact-horizons "3 5 9 13 15"

echo "[OK] hidden recursive + direct natural-hidden evaluation: $OUT_DIR"
