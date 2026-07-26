#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-$(dirname "$REPO")/.venv/bin/python}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_tactical_mixture_v1_2_actor_critic_h15_800k.yaml}"
SOURCE_CONFIG="${SOURCE_CONFIG:-configs/r2_2100_jepa_tactical_mixture_v1_2.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
EXPECTED_SOURCE_CHECKPOINT_SHA256="${EXPECTED_SOURCE_CHECKPOINT_SHA256:-}"
SKIP_CHECKPOINT_AUDIT="${SKIP_CHECKPOINT_AUDIT:-0}"

[[ -x "$PY" ]] || { echo "[FAIL] Python missing: $PY" >&2; exit 1; }
[[ -f "$REPO/$CONFIG" ]] || { echo "[FAIL] config missing: $REPO/$CONFIG" >&2; exit 1; }
[[ -f "$REPO/$SOURCE_CONFIG" ]] || { echo "[FAIL] source config missing: $REPO/$SOURCE_CONFIG" >&2; exit 1; }

bash -n "$REPO/scripts/run_actor_critic_h15_800k.sh"
bash -n "$REPO/scripts/run_forecast_then_actor_critic_h15_then_option_critic_v9.sh"
"$PY" -m py_compile "$REPO/scripts/audit_actor_critic_h15_800k.py"
if [[ "$SKIP_CHECKPOINT_AUDIT" != 1 ]]; then
  [[ -s "$CHECKPOINT" ]] || { echo "[FAIL] checkpoint missing: $CHECKPOINT" >&2; exit 1; }
  args=(--repo "$REPO" --config "$CONFIG" --source-config "$SOURCE_CONFIG" --checkpoint "$CHECKPOINT")
  [[ -z "$EXPECTED_SOURCE_CHECKPOINT_SHA256" ]] ||
    args+=(--expected-sha256 "$EXPECTED_SOURCE_CHECKPOINT_SHA256")
  "$PY" "$REPO/scripts/audit_actor_critic_h15_800k.py" "${args[@]}"
else
  "$PY" - "$REPO/$CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
assert int(cfg.imag_horizon) == 15
assert bool(cfg.tactical_mixture.enabled)
assert not bool(cfg.hierarchical_options.enabled)
assert bool(cfg.validation.run_at_start) and int(cfg.validation.every) == 200000
PY
fi

echo "[OK] Tactical-v1.2 ordinary actor-critic H=15 static audit passed"
