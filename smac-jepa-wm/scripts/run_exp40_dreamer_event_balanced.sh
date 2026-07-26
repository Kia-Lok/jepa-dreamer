#!/usr/bin/env bash
set -Eeuo pipefail
export SMAC_JEPA_EXP34_TWO_MASK_LOSS=1
export SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT="${SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT:-3.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JEPA_ROOT="${JEPA_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON:-${PY:-python}}"
: "${MANIFEST:?Set MANIFEST to an external JEPA dataset manifest}"
: "${OUT_DIR:?Set OUT_DIR to a fresh Exp-40 output directory}"
cd "$JEPA_ROOT"
EPOCHS="${EPOCHS:-5}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-SMAC-JEPA-losses}"
WANDB_NAME="${WANDB_NAME:-exp40-event-balanced}"
fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || fail "Manifest not found: $MANIFEST"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python not found: $PYTHON_BIN"
printf '[jepa_root] %s\n[python]    %s\n[manifest]  %s\n[out_dir]   %s\n' \
  "$JEPA_ROOT" "$PYTHON_BIN" "$MANIFEST" "$OUT_DIR"
resume_args=()
if [[ -n "${RESUME:-}" ]]; then [[ -f "$RESUME" ]] || fail "Requested resume checkpoint does not exist: $RESUME"; resume_args=(--resume "$RESUME"); mkdir -p "$OUT_DIR"; else if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -print -quit | grep -q .; then fail "OUT_DIR is not empty: $OUT_DIR"; fi; mkdir -p "$OUT_DIR"; fi
wandb_args=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-name "$WANDB_NAME")
if [[ "${WANDB:-1}" == "0" ]]; then wandb_args=(--no-wandb); fi
unset SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO || true
env LD_LIBRARY_PATH="" SMAC_JEPA_ANCHOR_GATE_INIT="${SMAC_JEPA_ANCHOR_GATE_INIT:--3.0}" SMAC_JEPA_ANCHOR_DELTA_SCALE="${SMAC_JEPA_ANCHOR_DELTA_SCALE:-0.25}" SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE="${SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE:-0.10}" SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT="${SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT:-0.002}" \
"$PYTHON_BIN" -m smac_jepa.train_jepa_exp40_dreamer \
  --manifest "$MANIFEST" --split train --out-dir "$OUT_DIR" \
  --model-size default --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" \
  --rollout-window 20 --rollout-horizon 5 --window-mode random --samples-per-epoch "$SAMPLES_PER_EPOCH" \
  --temporal-loss lambda --td-lambda 0.9 --action-conditioned-memory --rollout-memory-dim 322 \
  --one-step-weight 0.5 --target-mode full \
  --r2-dyn-scale 1.0 --r2-rep-scale 0.05 --r2-barlow-scale 0.025 --r2-barlow-lambda 0.0005 \
  --r2-latent-normalize --sigreg-weight 0.005 --decoder-weight 0.005 --presence-weight 0.01 \
  --ema-target-encoder --ema-momentum 0.996 \
  --delta-loss-weight 0.06 --event-dynamics-weight 1.25 --event-dynamics-threshold 0.01 \
  --event-balanced-sampling --event-fraction 0.50 --event-pool-fraction 0.25 \
  --enemy-observation-dropout 0.0 --occlusion-mode contiguous --contiguous-occlusion-spans 1 3 5 --occlusion-spans-per-sample 2 \
  --hidden-reconstruction-weight 0.03 --last-seen-anchor-weight 0.04 --last-seen-change-threshold 0.01 \
  --hidden-presence-weight 0.02 --reappearance-consistency-weight 0.02 \
  --hidden-change-residual-weight 0.0 --hidden-change-gate-weight 0.0 \
  --inverse-dynamics-weight 0.01 --inverse-dynamics-hidden-dim 256 \
  --memory-barlow-scale 0.0 --no-residual-state-decoder --no-direct-action-fusion \
  --aux-loss-warmup-steps 2000 --lr-warmup-steps 2000 --grad-clip 1.0 --checkpoint-every-steps 250 \
  --seed "$SEED" --device cuda --amp "${wandb_args[@]}" "${resume_args[@]}"
[[ -s "$OUT_DIR/checkpoint.pt" ]] || fail "Training exited without producing $OUT_DIR/checkpoint.pt"
"$PYTHON_BIN" scripts/validate_exp33_dreamer_checkpoint.py "$OUT_DIR/checkpoint.pt" \
  --jepa-root "$JEPA_ROOT" --dreamer-root "$JEPA_ROOT/../smac-dreamer"
