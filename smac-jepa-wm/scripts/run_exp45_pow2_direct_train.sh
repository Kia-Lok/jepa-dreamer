#!/usr/bin/env bash
set -Eeuo pipefail

export SMAC_JEPA_EXP34_TWO_MASK_LOSS=1
export SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT="${SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT:-3.0}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON:-${PY:-python}}"

: "${MANIFEST:?Set MANIFEST to an external JEPA dataset manifest}"
OUT_DIR="${OUT_DIR:-runs/rnn_seqmem_exp45_pow2_direct_1_2_4_8_16}"
: "${EXP40_CHECKPOINT:?Set EXP40_CHECKPOINT to an external Exp-40 checkpoint}"
EPOCHS="${EPOCHS:-5}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT="${WANDB_PROJECT:-SMAC-JEPA-losses}"
WANDB_NAME="${WANDB_NAME:-exp45-pow2-direct-1-2-4-8-16}"
POW2_DIRECT_WEIGHT="${POW2_DIRECT_WEIGHT:-0.10}"
POW2_COMPOSITION_WEIGHT="${POW2_COMPOSITION_WEIGHT:-0.05}"
POW2_SHARED_HEAD_WEIGHT="${POW2_SHARED_HEAD_WEIGHT:-0.10}"
POW2_HIDDEN_DIM="${POW2_HIDDEN_DIM:-384}"
POW2_WARMUP_STEPS="${POW2_WARMUP_STEPS:-2000}"

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || fail "Manifest not found: $MANIFEST"
[[ -f smac_jepa/train_jepa_exp40_dreamer.py ]] || fail "Missing installed Exp40 wrapper"
[[ -f smac_jepa/train_jepa_exp45_pow2_direct.py ]] || fail "Missing canonical Exp-45 trainer source"

resume_args=()
init_args=()
if [[ -n "${RESUME:-}" ]]; then
  [[ -f "$RESUME" ]] || fail "Resume checkpoint not found: $RESUME"
  resume_args=(--resume "$RESUME")
  mkdir -p "$OUT_DIR"
else
  [[ -f "$EXP40_CHECKPOINT" ]] || fail "Exp40 checkpoint not found: $EXP40_CHECKPOINT"
  init_args=(--pow2-init-from-exp40 "$EXP40_CHECKPOINT")
  if [[ -d "$OUT_DIR" ]] && find "$OUT_DIR" -mindepth 1 -print -quit | grep -q .; then
    fail "OUT_DIR is not empty: $OUT_DIR"
  fi
  mkdir -p "$OUT_DIR"
fi

wandb_args=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-name "$WANDB_NAME")
if [[ "${WANDB:-1}" == "0" ]]; then wandb_args=(--no-wandb); fi
amp_args=(--amp)
if [[ "${AMP:-1}" == "0" ]]; then amp_args=(--no-amp); fi

unset SMAC_JEPA_FORCE_ANCHOR_GATE_ZERO || true

env \
  LD_LIBRARY_PATH="" \
  SMAC_JEPA_ANCHOR_GATE_INIT="${SMAC_JEPA_ANCHOR_GATE_INIT:--3.0}" \
  SMAC_JEPA_ANCHOR_DELTA_SCALE="${SMAC_JEPA_ANCHOR_DELTA_SCALE:-0.25}" \
  SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE="${SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE:-0.10}" \
  SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT="${SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT:-0.002}" \
"$PYTHON_BIN" -m smac_jepa.train_jepa_exp45_pow2_direct \
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
  --pow2-horizons 1 2 4 8 16 --pow2-base-rollout-horizon 5 \
  --pow2-direct-weight "$POW2_DIRECT_WEIGHT" \
  --pow2-composition-weight "$POW2_COMPOSITION_WEIGHT" \
  --pow2-shared-head-weight "$POW2_SHARED_HEAD_WEIGHT" \
  --pow2-hidden-dim "$POW2_HIDDEN_DIM" \
  --pow2-action-embed-dim 48 --pow2-slot-embed-dim 32 --pow2-residual-scale 0.25 \
  --pow2-warmup-steps "$POW2_WARMUP_STEPS" \
  --seed "$SEED" --device "$DEVICE" "${amp_args[@]}" \
  "${wandb_args[@]}" "${init_args[@]}" "${resume_args[@]}"

[[ -s "$OUT_DIR/checkpoint.pt" ]] || fail "Training exited without $OUT_DIR/checkpoint.pt"
"$PYTHON_BIN" tools/audit_exp45_pow2_checkpoint.py "$OUT_DIR/checkpoint.pt"
echo "[OK] Exp45 checkpoint: $OUT_DIR/checkpoint.pt"
