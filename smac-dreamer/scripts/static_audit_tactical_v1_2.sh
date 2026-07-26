#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_tactical_mixture_v1_2.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
SOURCE_RUN_META="${SOURCE_RUN_META:-}"
EXPECTED_SOURCE_CHECKPOINT_SHA256="${EXPECTED_SOURCE_CHECKPOINT_SHA256:-}"
SKIP_CHECKPOINT_AUDIT="${SKIP_CHECKPOINT_AUDIT:-0}"
cd "$REPO"
"$PY" -m py_compile \
  external/r2dreamer/dreamer.py \
  external/r2dreamer/tactical_policy.py \
  scripts/train_r2dreamer_smaclite_multimap.py \
  scripts/audit_tactical_v1_2.py \
  scripts/assert_tactical_v1_2_metrics.py
"$PY" -m pytest -q tests/test_tactical_policy_v1_2.py
if [[ "$SKIP_CHECKPOINT_AUDIT" != 1 ]]; then
  [[ -s "$CHECKPOINT" ]] || { echo "[FAIL] checkpoint missing: $CHECKPOINT" >&2; exit 1; }
  args=(--repo "$REPO" --config "$CONFIG" --checkpoint "$CHECKPOINT")
  [[ -z "$SOURCE_RUN_META" ]] || args+=(--source-run-meta "$SOURCE_RUN_META")
  [[ -z "$EXPECTED_SOURCE_CHECKPOINT_SHA256" ]] ||
    args+=(--expected-checkpoint-sha256 "$EXPECTED_SOURCE_CHECKPOINT_SHA256")
  "$PY" scripts/audit_tactical_v1_2.py "${args[@]}"
else
  "$PY" - "$REPO/$CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
assert bool(cfg.tactical_mixture.enabled)
assert int(cfg.tactical_mixture.num_tactics) == 2
assert int(cfg.validation.every) == 200000
PY
fi
echo "[OK] Tactical Mixture v1.2 static audit passed"
