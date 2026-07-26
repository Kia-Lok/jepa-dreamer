#!/usr/bin/env bash
# Resilient form of the user-provided Exp45 forecast-JEPA pipeline.
# Safe stages continue when a previous stage fails; every result is recorded.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="${ROOT:-$(dirname "$REPO")}"
JEPA_ROOT="${JEPA_ROOT:-$ROOT/smac-jepa-wm}"
VENV="${VENV:-$ROOT/.venv}"
: "${MANIFEST:?Set MANIFEST to an external JEPA dataset manifest}"
: "${EXP40_CHECKPOINT:?Set EXP40_CHECKPOINT to an external Exp-40 checkpoint}"
EPOCHS="${EPOCHS:-5}"; SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"; NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1}"; DEVICE="${DEVICE:-cuda}"; WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-SMAC-JEPA-losses}"
WANDB_NAME="${WANDB_NAME:-exp45-pow2-direct-1-2-4-8-16}"
AMP="${AMP:-1}"
POW2_DIRECT_WEIGHT="${POW2_DIRECT_WEIGHT:-0.10}"
POW2_COMPOSITION_WEIGHT="${POW2_COMPOSITION_WEIGHT:-0.05}"
POW2_SHARED_HEAD_WEIGHT="${POW2_SHARED_HEAD_WEIGHT:-0.10}"
POW2_HIDDEN_DIM="${POW2_HIDDEN_DIM:-384}"
POW2_WARMUP_STEPS="${POW2_WARMUP_STEPS:-2000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-300}"; PROBE_EPOCHS="${PROBE_EPOCHS:-5}"
PROBE_MAX_BATCHES="${PROBE_MAX_BATCHES:-100}"; PROBE_SAMPLES="${PROBE_SAMPLES:-8000}"
DIRECT_MAX_BATCHES="${DIRECT_MAX_BATCHES:-300}"
NATURAL_HIDDEN_TARGETS="${NATURAL_HIDDEN_TARGETS:-3000}"
NATURAL_HIDDEN_SCAN_BATCHES="${NATURAL_HIDDEN_SCAN_BATCHES:-5000}"
CONTROLLED_MAX_BATCHES="${CONTROLLED_MAX_BATCHES:-150}"
CONTROLLED_TARGETS="${CONTROLLED_TARGETS:-10000}"
RESUME="${RESUME:-}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PIPE_DIR="${PIPE_DIR:-$ROOT/exp45_pow2_full_pipeline_$STAMP}"
RUN_DIR="${RUN_DIR:-$JEPA_ROOT/runs/rnn_seqmem_exp45_pow2_direct_$STAMP}"
EVAL_ROOT="${EVAL_ROOT:-$PIPE_DIR/eval}"
ORDINARY_OUT_DIR="$EVAL_ROOT/ordinary"; HIDDEN_OUT_DIR="$EVAL_ROOT/hidden"
LOG_DIR="$PIPE_DIR/logs"; STATUS_FILE="$PIPE_DIR/status.tsv"
STRICT_EXIT="${STRICT_EXIT:-0}"
mkdir -p "$PIPE_DIR" "$LOG_DIR" "$ORDINARY_OUT_DIR" "$HIDDEN_OUT_DIR"
printf 'stage\tstatus\ttimestamp\texit\tdetail\n' > "$STATUS_FILE"

FAILURES=0
now(){ date -Is; }
record(){ printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$(now)" "$3" "${4:-}" | tee -a "$STATUS_FILE"; }
run_stage(){
  local stage="$1" log="$2"; shift 2
  record "$stage" STARTED - ""
  "$@" 2>&1 | tee "$log"
  local code=${PIPESTATUS[0]}
  if (( code == 0 )); then record "$stage" COMPLETED 0 ""; else record "$stage" FAILED "$code" "$log"; FAILURES=$((FAILURES+1)); fi
  return "$code"
}

# Setup is checked without aborting the outer forecast->RL pipeline.
SETUP_OK=1
[[ -d "$ROOT" ]] || { record setup FAILED 2 "ROOT missing: $ROOT"; SETUP_OK=0; }
[[ -d "$JEPA_ROOT" ]] || { record setup FAILED 2 "JEPA repo missing: $JEPA_ROOT"; SETUP_OK=0; }
[[ -f "$VENV/bin/activate" ]] || { record setup FAILED 2 "venv missing: $VENV"; SETUP_OK=0; }
if (( SETUP_OK == 0 )); then
  FAILURES=$((FAILURES+1))
  record pipeline COMPLETED_WITH_FAILURES 0 "$PIPE_DIR"
  (( STRICT_EXIT == 1 )) && exit 1 || exit 0
fi

# The forecast implementation is canonical source, never a runtime-installed bundle.
for required in \
  smac_jepa/train_jepa_exp45_pow2_direct.py \
  scripts/static_audit_exp45_pow2.sh \
  scripts/run_exp45_pow2_direct_train.sh \
  scripts/eval_exp45_pow2_ordinary.sh \
  scripts/eval_exp45_pow2_hidden.sh; do
  if [[ ! -f "$JEPA_ROOT/$required" ]]; then
    record source_verify FAILED 2 "missing canonical source: $JEPA_ROOT/$required"
    FAILURES=$((FAILURES + 1))
  fi
done

# shellcheck disable=SC1090
if ! source "$VENV/bin/activate"; then
  record setup FAILED 2 "could not activate virtual environment: $VENV"
  FAILURES=$((FAILURES + 1))
  record pipeline COMPLETED_WITH_FAILURES 0 "$PIPE_DIR"
  (( STRICT_EXIT == 1 )) && exit 1 || exit 0
fi
if ! cd "$JEPA_ROOT"; then
  record setup FAILED 2 "could not enter JEPA repo: $JEPA_ROOT"
  FAILURES=$((FAILURES + 1))
  record pipeline COMPLETED_WITH_FAILURES 0 "$PIPE_DIR"
  (( STRICT_EXIT == 1 )) && exit 1 || exit 0
fi
export PYTHONPATH="$JEPA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
printf '[root]       %s\n[jepa_root]  %s\n[venv]       %s\n[manifest]   %s\n[exp40]      %s\n[pipe_dir]   %s\n' \
  "$ROOT" "$JEPA_ROOT" "$VENV" "$MANIFEST" "$EXP40_CHECKPOINT" "$PIPE_DIR"

AUDIT_OK=1
for path in "$MANIFEST" "$EXP40_CHECKPOINT" scripts/static_audit_exp45_pow2.sh scripts/run_exp45_pow2_direct_train.sh scripts/eval_exp45_pow2_ordinary.sh scripts/eval_exp45_pow2_hidden.sh; do
  [[ -e "$path" ]] || { record preflight FAILED 2 "missing $JEPA_ROOT/$path"; AUDIT_OK=0; }
done
if (( AUDIT_OK )); then
  run_stage static_audit "$LOG_DIR/static_audit.log" bash ./scripts/static_audit_exp45_pow2.sh || AUDIT_OK=0
else
  FAILURES=$((FAILURES+1))
fi

CHECKPOINT="$RUN_DIR/checkpoint.pt"
TRAIN_OK=0
if (( AUDIT_OK )); then
  train_env=(
    "MANIFEST=$MANIFEST" "EXP40_CHECKPOINT=$EXP40_CHECKPOINT" "OUT_DIR=$RUN_DIR"
    "PYTHON=$VENV/bin/python"
    "EPOCHS=$EPOCHS" "SAMPLES_PER_EPOCH=$SAMPLES_PER_EPOCH" "BATCH_SIZE=$BATCH_SIZE"
    "NUM_WORKERS=$NUM_WORKERS" "SEED=$SEED" "DEVICE=$DEVICE" "WANDB=$WANDB"
    "WANDB_PROJECT=$WANDB_PROJECT" "WANDB_NAME=$WANDB_NAME" "AMP=$AMP"
    "POW2_DIRECT_WEIGHT=$POW2_DIRECT_WEIGHT" "POW2_COMPOSITION_WEIGHT=$POW2_COMPOSITION_WEIGHT"
    "POW2_SHARED_HEAD_WEIGHT=$POW2_SHARED_HEAD_WEIGHT" "POW2_HIDDEN_DIM=$POW2_HIDDEN_DIM"
    "POW2_WARMUP_STEPS=$POW2_WARMUP_STEPS"
  )
  [[ -z "$RESUME" ]] || train_env+=("RESUME=$RESUME")
  run_stage training "$LOG_DIR/train.log" env "${train_env[@]}" bash ./scripts/run_exp45_pow2_direct_train.sh && TRAIN_OK=1 || true
else
  record training SKIPPED - "static audit/preflight failed"
fi

# A partial checkpoint is still useful for diagnostics. Never fabricate one.
if [[ -s "$CHECKPOINT" ]]; then
  record checkpoint AVAILABLE 0 "$CHECKPOINT"
  run_stage ordinary_eval "$LOG_DIR/eval_ordinary.log" env \
    CHECKPOINT="$CHECKPOINT" MANIFEST="$MANIFEST" SPLIT=eval OUT_DIR="$ORDINARY_OUT_DIR" \
    DEVICE="$DEVICE" BATCH_SIZE="$BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" MAX_BATCHES="$EVAL_MAX_BATCHES" \
    PROBE_EPOCHS="$PROBE_EPOCHS" PROBE_MAX_BATCHES="$PROBE_MAX_BATCHES" PROBE_SAMPLES="$PROBE_SAMPLES" \
    bash ./scripts/eval_exp45_pow2_ordinary.sh || true
  run_stage hidden_eval "$LOG_DIR/eval_hidden.log" env \
    CHECKPOINT="$CHECKPOINT" MANIFEST="$MANIFEST" SPLIT=eval OUT_DIR="$HIDDEN_OUT_DIR" \
    ORDINARY_OUT_DIR="$ORDINARY_OUT_DIR" DEVICE="$DEVICE" BATCH_SIZE="$BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" \
    DIRECT_MAX_BATCHES="$DIRECT_MAX_BATCHES" NATURAL_HIDDEN_TARGETS="$NATURAL_HIDDEN_TARGETS" \
    NATURAL_HIDDEN_SCAN_BATCHES="$NATURAL_HIDDEN_SCAN_BATCHES" CONTROLLED_MAX_BATCHES="$CONTROLLED_MAX_BATCHES" \
    CONTROLLED_TARGETS="$CONTROLLED_TARGETS" bash ./scripts/eval_exp45_pow2_hidden.sh || true
else
  record checkpoint MISSING 2 "$CHECKPOINT"
  record ordinary_eval SKIPPED - "no checkpoint"
  record hidden_eval SKIPPED - "no checkpoint"
  FAILURES=$((FAILURES+1))
fi

state=COMPLETED; (( FAILURES > 0 )) && state=COMPLETED_WITH_FAILURES
record pipeline "$state" 0 "$PIPE_DIR"
echo "[FORECAST PIPELINE] $PIPE_DIR"
echo "[STATUS] $STATUS_FILE"
(( STRICT_EXIT == 1 && FAILURES > 0 )) && exit 1
exit 0
