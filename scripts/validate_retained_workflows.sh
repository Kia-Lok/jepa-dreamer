#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JEPA_ROOT="$ROOT/smac-jepa-wm"
DREAMER_ROOT="$ROOT/smac-dreamer"
PY="${PY:-python}"

fail() { echo "[FAIL] $*" >&2; exit 1; }
command -v "$PY" >/dev/null 2>&1 || fail "Python not found: $PY"
command -v rg >/dev/null 2>&1 || fail "ripgrep is required"

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

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

for file in "${JEPA_SHELL[@]}"; do bash -n "$JEPA_ROOT/scripts/$file"; done
for file in "${DREAMER_SHELL[@]}"; do bash -n "$DREAMER_ROOT/scripts/$file"; done

cd "$JEPA_ROOT"
JEPA_DATA_FILES=(
  smac_jepa/data/__init__.py
  smac_jepa/data/dataset.py
  smac_jepa/data/markov_rollout_dataset.py
  smac_jepa/data/markov_rollout_visibility_dataset.py
)
for file in "${JEPA_DATA_FILES[@]}"; do
  [[ -s "$file" ]] || fail "Missing JEPA loader source: $file"
done

"$PY" -m py_compile \
  "${JEPA_DATA_FILES[@]}" \
  smac_jepa/train_jepa_exp40_dreamer.py \
  smac_jepa/train_jepa_exp31_exp35.py \
  smac_jepa/anchored_belief_memory.py \
  smac_jepa/train_jepa_exp45_pow2_direct.py \
  smac_jepa/pow2_direct_predictor.py \
  scripts/validate_exp33_dreamer_checkpoint.py \
  tools/audit_exp45_pow2_checkpoint.py \
  tools/eval_pow2_direct.py \
  tools/eval_exp40_rollout_gallery.py

"$PY" - <<'PY'
from smac_jepa.data import (
    SMACJEPADataset,
    load_manifest,
    load_manifest_all,
    load_npz_metadata,
)
from smac_jepa.data.markov_rollout_dataset import MarkovRolloutSMACJEPADataset
from smac_jepa.data.markov_rollout_visibility_dataset import (
    VisibilityMarkovRolloutSMACJEPADataset,
)

assert SMACJEPADataset is not None
assert load_manifest is not None
assert load_manifest_all is not None
assert load_npz_metadata is not None
assert MarkovRolloutSMACJEPADataset is not None
assert VisibilityMarkovRolloutSMACJEPADataset is not None
assert VisibilityMarkovRolloutSMACJEPADataset.explicit_visibility_mask_version >= 1
print("[OK] JEPA dataloader imports and visibility contract")
PY

"$PY" -m pytest -q tests/test_dataset_windows.py
"$PY" tests/test_exp33_memory_contract.py
"$PY" -m smac_jepa.train_jepa_exp40_dreamer --help >/dev/null
"$PY" -m smac_jepa.train_jepa_exp45_pow2_direct --help >/dev/null
"$PY" -m pytest -q \
  tests/test_pow2_checkpoint_sanitizer.py \
  tests/test_pow2_direct_predictor.py

cd "$DREAMER_ROOT"
"$PY" scripts/train_r2dreamer_smaclite_multimap.py --help >/dev/null
"$PY" scripts/evaluate_multimap.py --help >/dev/null
"$PY" scripts/preflight_jepa_training.py --help >/dev/null

"$PY" -m pytest -q \
  tests/test_tactical_policy_v1_2.py \
  tests/test_option_critic_v9_core.py \
  tests/test_option_critic_v9_migration.py \
  tests/test_option_critic_v9_auxiliary.py

# These cases specify pre-V9 progressive slot locks and learned/cumulative
# termination behavior that V9 intentionally replaced. Keep the tests, but run
# the same still-applicable legacy subset enforced by the canonical V9 audit.
LEGACY_EXCLUDE='not test_locked_child_delta_cannot_change_source_policy_before_unlock and not test_minimum_duration_is_never_violated and not test_preservation_phase_reselects_at_every_eligible_state and not test_eligible_probability_uses_fixed_hazard_without_preservation_override and not test_eval_uses_deterministic_cumulative_hazard and not test_initialized_bounded_termination_matches_fixed_hazard and not test_executed_termination_probability_has_warmup_gate_and_smooth_cap_gradients and not test_locked_slots_have_zero_probability_and_zero_pg_maturity and not test_child_learned_termination_waits_for_slot_maturity and not test_v1_2_migration_has_eight_capacity_but_only_two_active_anchors and not test_progressive_slot_unlock_and_specialization_are_continuous and not test_v1_2_migration_reselects_every_state_at_step_zero'

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/external/r2dreamer:$PWD/tests${PYTHONPATH:+:$PYTHONPATH}" \
"$PY" -m pytest -q \
  tests/test_hierarchical_options.py \
  tests/test_option_critic_math.py \
  tests/test_hierarchical_auxiliary.py \
  tests/test_hierarchy_migration.py \
  -k "$LEGACY_EXCLUDE"

SKIP_CHECKPOINT_AUDIT=1 PY="$PY" bash scripts/static_audit_tactical_v1_2.sh
SKIP_CHECKPOINT_AUDIT=1 PY="$PY" bash scripts/static_audit_actor_critic_h15_800k.sh
SKIP_CHECKPOINT_AUDIT=1 PY="$PY" bash scripts/static_audit_option_critic_v9_anchor_safe.sh

if rg -n \
  'CURRENT_[A-Z0-9_]+\.txt|BUNDLE_ZIP|combined-upload|rnn_seqmem_exp40_event_balanced_5ep_20260709_083104' \
  "$JEPA_ROOT/scripts" "$DREAMER_ROOT/scripts"; then
  fail "stale runtime pointer, bundle dependency, or historical path remains"
fi

echo "[OK] all retained workflows passed checkpoint-free validation"
