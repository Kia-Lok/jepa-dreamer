#!/usr/bin/env bash
# Exp-45 forecast -> ordinary Tactical-v1.2 AC -> Option-Critic V9.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ROOT="${ROOT:-$(dirname "$REPO")}"
JEPA_ROOT="${JEPA_ROOT:-$ROOT/smac-jepa-wm}"
VENV="${VENV:-$ROOT/.venv}"
PY="${PY:-python}"
: "${EXP40_CHECKPOINT:?Set EXP40_CHECKPOINT to an external Exp-40 checkpoint}"
: "${TACTICAL_V12_CHECKPOINT:?Set TACTICAL_V12_CHECKPOINT to an external Tactical-v1.2 checkpoint}"
: "${JEPA_CHECKPOINT:?Set JEPA_CHECKPOINT to the Exp-40 checkpoint used by RL}"
: "${MANIFEST:?Set MANIFEST to an external JEPA dataset manifest}"
: "${PIPE_DIR:?Set PIPE_DIR to a fresh pipeline output directory}"
SOURCE_RUN_META="${SOURCE_RUN_META:-}"
EXPECTED_SOURCE_CHECKPOINT_SHA256="${EXPECTED_SOURCE_CHECKPOINT_SHA256:-}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
STRICT_EXIT="${STRICT_EXIT:-1}"

[[ ! -e "$PIPE_DIR" ]] || { echo "[FAIL] PIPE_DIR already exists: $PIPE_DIR" >&2; exit 1; }
mkdir -p "$PIPE_DIR/logs"
STATUS="$PIPE_DIR/status.tsv"
printf 'stage\tstatus\ttimestamp\texit\n' > "$STATUS"
FAILURES=0

run_stage() {
  local name="$1" log="$2"; shift 2
  printf '%s\tSTARTED\t%s\t-\n' "$name" "$(date '+%Y-%m-%dT%H:%M:%S%z')" | tee -a "$STATUS"
  "$@" 2>&1 | tee "$log"
  local code=${PIPESTATUS[0]}
  if (( code == 0 )); then
    printf '%s\tCOMPLETED\t%s\t0\n' "$name" "$(date '+%Y-%m-%dT%H:%M:%S%z')" | tee -a "$STATUS"
  else
    FAILURES=$((FAILURES + 1))
    printf '%s\tFAILED\t%s\t%s\n' "$name" "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$code" | tee -a "$STATUS"
  fi
  return "$code"
}
should_continue() { (( CONTINUE_ON_FAILURE == 1 || FAILURES == 0 )); }

run_stage forecast_jepa "$PIPE_DIR/logs/forecast.log" env \
  ROOT="$ROOT" REPO="$REPO" JEPA_ROOT="$JEPA_ROOT" VENV="$VENV" \
  EXP40_CHECKPOINT="$EXP40_CHECKPOINT" MANIFEST="$MANIFEST" \
  PIPE_DIR="$PIPE_DIR/forecast" RUN_DIR="$PIPE_DIR/forecast/train" STRICT_EXIT=1 \
  bash "$REPO/scripts/run_exp45_full_train_eval_resilient.sh" || true

if should_continue; then
  run_stage actor_critic_h15_800k "$PIPE_DIR/logs/actor_critic.log" env \
    REPO="$REPO" PY="$PY" SOURCE_CHECKPOINT="$TACTICAL_V12_CHECKPOINT" \
    SOURCE_RUN_META="$SOURCE_RUN_META" \
    EXPECTED_SOURCE_CHECKPOINT_SHA256="$EXPECTED_SOURCE_CHECKPOINT_SHA256" \
    JEPA_CHECKPOINT="$JEPA_CHECKPOINT" RUN_DIR="$PIPE_DIR/actor_critic_h15_800k" \
    FINAL_STEP=800000 bash "$REPO/scripts/run_actor_critic_h15_800k.sh" || true
else
  printf 'actor_critic_h15_800k\tSKIPPED_PREVIOUS_FAILURE\t%s\t-\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" | tee -a "$STATUS"
fi

if should_continue; then
  run_stage option_critic_v9_h15_800k "$PIPE_DIR/logs/option_critic.log" env \
    REPO="$REPO" PY="$PY" SOURCE_CHECKPOINT="$TACTICAL_V12_CHECKPOINT" \
    SOURCE_RUN_META="$SOURCE_RUN_META" \
    EXPECTED_SOURCE_CHECKPOINT_SHA256="$EXPECTED_SOURCE_CHECKPOINT_SHA256" \
    JEPA_CHECKPOINT="$JEPA_CHECKPOINT" RUN_DIR="$PIPE_DIR/option_critic_v9_h15_800k" \
    FINAL_STEP=800000 bash "$REPO/scripts/run_option_critic_v9_anchor_safe_800k.sh" || true
else
  printf 'option_critic_v9_h15_800k\tSKIPPED_PREVIOUS_FAILURE\t%s\t-\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" | tee -a "$STATUS"
fi

STATE=COMPLETED
(( FAILURES > 0 )) && STATE=COMPLETED_WITH_FAILURES
printf 'pipeline\t%s\t%s\t%s\n' "$STATE" "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$FAILURES" | tee -a "$STATUS"
(( STRICT_EXIT == 1 && FAILURES > 0 )) && exit 1
