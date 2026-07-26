#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-python}"
CONFIG="${CONFIG:-configs/r2_2100_jepa_tactical_mixture_v1_2_actor_critic_h15_800k.yaml}"
SOURCE_CONFIG="${SOURCE_CONFIG:-configs/r2_2100_jepa_tactical_mixture_v1_2.yaml}"
FINAL_STEP="${FINAL_STEP:-800000}"
: "${SOURCE_CHECKPOINT:?Set SOURCE_CHECKPOINT to a Tactical-v1.2 checkpoint}"
: "${JEPA_CHECKPOINT:?Set JEPA_CHECKPOINT to a compatible Exp-40 checkpoint}"
: "${RUN_DIR:?Set RUN_DIR to a fresh actor-critic output directory}"
SOURCE_RUN_META="${SOURCE_RUN_META:-}"
EXPECTED_SOURCE_CHECKPOINT_SHA256="${EXPECTED_SOURCE_CHECKPOINT_SHA256:-}"

command -v "$PY" >/dev/null 2>&1 || { echo "[FAIL] Python missing: $PY" >&2; exit 1; }
[[ -d "$REPO" ]] || { echo "[FAIL] repo missing: $REPO" >&2; exit 1; }
[[ -s "$SOURCE_CHECKPOINT" ]] || { echo "[FAIL] source checkpoint missing: $SOURCE_CHECKPOINT" >&2; exit 1; }
[[ -s "$JEPA_CHECKPOINT" ]] || { echo "[FAIL] JEPA checkpoint missing: $JEPA_CHECKPOINT" >&2; exit 1; }
[[ "$FINAL_STEP" =~ ^[0-9]+$ ]] && (( FINAL_STEP == 800000 )) || {
  echo "[FAIL] comparison baseline requires exactly 800000 new environment steps" >&2
  exit 1
}
[[ ! -e "$RUN_DIR" ]] || { echo "[FAIL] RUN_DIR already exists: $RUN_DIR" >&2; exit 1; }
if [[ -n "$SOURCE_RUN_META" ]]; then
  [[ -s "$SOURCE_RUN_META" ]] || { echo "[FAIL] source metadata missing: $SOURCE_RUN_META" >&2; exit 1; }
fi
if pgrep -af 'train_r2dreamer_smaclite_multimap.py' >/dev/null; then
  echo "[FAIL] another multimap trainer is already active" >&2
  exit 1
fi

cd "$REPO"
audit_env=(REPO="$REPO" PY="$PY" CONFIG="$CONFIG" SOURCE_CONFIG="$SOURCE_CONFIG"
  CHECKPOINT="$SOURCE_CHECKPOINT" SKIP_CHECKPOINT_AUDIT=0)
[[ -z "$EXPECTED_SOURCE_CHECKPOINT_SHA256" ]] ||
  audit_env+=("EXPECTED_SOURCE_CHECKPOINT_SHA256=$EXPECTED_SOURCE_CHECKPOINT_SHA256")
env "${audit_env[@]}" bash scripts/static_audit_actor_critic_h15_800k.sh

mkdir -p "$RUN_DIR"
SOURCE_SHA="$("$PY" - "$SOURCE_CHECKPOINT" <<'PY'
import hashlib, sys
with open(sys.argv[1], "rb") as stream:
    print(hashlib.file_digest(stream, "sha256").hexdigest())
PY
)"
META_VALUE=null
[[ -z "$SOURCE_RUN_META" ]] || META_VALUE="$SOURCE_RUN_META"
cat > "$RUN_DIR/SOURCE_LINEAGE.txt" <<EOF
source_checkpoint=$SOURCE_CHECKPOINT
source_checkpoint_sha256=$SOURCE_SHA
source_run_meta=$META_VALUE
jepa_checkpoint=$JEPA_CHECKPOINT
architecture=tactical_mixture_v1_2_ordinary_actor_critic
option_critic=disabled
imag_horizon=15
new_environment_steps=$FINAL_STEP
EOF

printf '[repo] %s\n[source] %s\n[jepa] %s\n[run_dir] %s\n' \
  "$REPO" "$SOURCE_CHECKPOINT" "$JEPA_CHECKPOINT" "$RUN_DIR"
"$PY" scripts/train_r2dreamer_smaclite_multimap.py \
  --config "$CONFIG" \
  --resume "$SOURCE_CHECKPOINT" \
  --resume-start-step 0 \
  --jepa-checkpoint "$JEPA_CHECKPOINT" \
  --logdir "$RUN_DIR" \
  --steps "$FINAL_STEP" \
  2>&1 | tee "$RUN_DIR/train.log"
