#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
SKIP_DEPENDENCY_CHECKS="${SKIP_DEPENDENCY_CHECKS:-0}"
ALLOW_MISSING_JEPA_DATA_SOURCE="${ALLOW_MISSING_JEPA_DATA_SOURCE:-0}"

fail() { echo "[FAIL] $*" >&2; exit 1; }
command -v "$PY" >/dev/null 2>&1 || fail "Python not found: $PY"

JEPA_SHELL=(
  run_exp40_dreamer_event_balanced.sh
  run_exp40_rollout_gallery.sh
  run_exp45_pow2_direct_train.sh
  eval_exp45_pow2_all.sh
  eval_exp45_pow2_ordinary.sh
  eval_exp45_pow2_hidden.sh
  smoke_exp45_pow2_direct.sh
  static_audit_exp45_pow2.sh
)
DREAMER_SHELL=(
  run_exp45_full_train_eval_resilient.sh
  run_tactical_v1_2_2m.sh
  static_audit_tactical_v1_2.sh
  run_actor_critic_h15_800k.sh
  static_audit_actor_critic_h15_800k.sh
  run_option_critic_v9_anchor_safe_800k.sh
  static_audit_option_critic_v9_anchor_safe.sh
  run_forecast_then_actor_critic_h15_then_option_critic_v9.sh
)

for file in "${JEPA_SHELL[@]}"; do bash -n "$ROOT/smac-jepa-wm/scripts/$file"; done
for file in "${DREAMER_SHELL[@]}"; do bash -n "$ROOT/smac-dreamer/scripts/$file"; done

"$PY" -m py_compile \
  "$ROOT/smac-jepa-wm/smac_jepa/train_jepa_exp40_dreamer.py" \
  "$ROOT/smac-jepa-wm/smac_jepa/train_jepa_exp31_exp35.py" \
  "$ROOT/smac-jepa-wm/smac_jepa/anchored_belief_memory.py" \
  "$ROOT/smac-jepa-wm/smac_jepa/train_jepa_exp45_pow2_direct.py" \
  "$ROOT/smac-jepa-wm/smac_jepa/pow2_direct_predictor.py" \
  "$ROOT/smac-jepa-wm/tools/audit_exp45_pow2_checkpoint.py" \
  "$ROOT/smac-jepa-wm/tools/eval_pow2_direct.py" \
  "$ROOT/smac-jepa-wm/tools/eval_exp40_rollout_gallery.py" \
  "$ROOT/smac-dreamer/scripts/audit_actor_critic_h15_800k.py" \
  "$ROOT/smac-dreamer/scripts/audit_option_critic_v9_anchor_safe.py"

if [[ ! -d "$ROOT/smac-jepa-wm/smac_jepa/data" ]]; then
  [[ "$ALLOW_MISSING_JEPA_DATA_SOURCE" == 1 ]] ||
    fail "required JEPA loader source is absent: smac-jepa-wm/smac_jepa/data"
  echo "[BLOCKED] JEPA loader source is absent; import tests cannot run" >&2
fi

if [[ "$SKIP_DEPENDENCY_CHECKS" != 1 ]]; then
  cd "$ROOT/smac-jepa-wm"
  "$PY" -m smac_jepa.train_jepa_exp40_dreamer --help >/dev/null
  "$PY" -m smac_jepa.train_jepa_exp45_pow2_direct --help >/dev/null
  "$PY" -m pytest -q tests/test_pow2_checkpoint_sanitizer.py tests/test_pow2_direct_predictor.py

  cd "$ROOT/smac-dreamer"
  "$PY" scripts/train_r2dreamer_smaclite_multimap.py --help >/dev/null
  "$PY" scripts/evaluate_multimap.py --help >/dev/null
  SKIP_CHECKPOINT_AUDIT=1 PY="$PY" bash scripts/static_audit_tactical_v1_2.sh
  SKIP_CHECKPOINT_AUDIT=1 PY="$PY" bash scripts/static_audit_actor_critic_h15_800k.sh
  SKIP_CHECKPOINT_AUDIT=1 PY="$PY" bash scripts/static_audit_option_critic_v9_anchor_safe.sh
fi

if rg -n \
  'CURRENT_[A-Z0-9_]+\.txt|BUNDLE_ZIP|combined-upload|20260709_083104' \
  "$ROOT/smac-jepa-wm/scripts" "$ROOT/smac-dreamer/scripts"; then
  fail "stale runtime pointer, bundle dependency, or historical path remains"
fi

echo "[OK] retained workflow validation passed"
