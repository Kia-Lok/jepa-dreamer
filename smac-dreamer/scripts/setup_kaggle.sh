#!/usr/bin/env bash
set -euo pipefail

SRC="${1:-/kaggle/input/smac-dreamer/smac-dreamer}"
DST="${2:-/kaggle/working/smac-dreamer}"

echo "[kaggle] source: ${SRC}"
echo "[kaggle] dest  : ${DST}"
mkdir -p "$(dirname "${DST}")"
if [[ ! -d "${DST}" ]]; then
  cp -R "${SRC}" "${DST}"
fi
cd "${DST}"

echo "[kaggle] installed torch:"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
PY

echo "[kaggle] installing project deps without reinstalling torch"
python -m pip install -e . --no-deps
python -m pip install \
  cloudpickle gymnasium hydra-core numpy omegaconf ruamel.yaml scikit-learn \
  rtree tensorboard tensordict torchrl wandb

echo "[kaggle] CUDA preflight for FP32 Kaggle config"
PYTHONPATH=src:external/r2dreamer:external/smaclite python - <<'PY'
from smacdreamer.cuda_preflight import run_cuda_preflight
run_cuda_preflight("cuda:0", "float32")
PY

cat <<'TXT'
[kaggle] setup complete.

Set W&B secrets/environment before training:
  WANDB_API_KEY   (Kaggle secret; never commit)
  WANDB_PROJECT
  WANDB_ENTITY
  WANDB_MODE=online

Train:
  python scripts/train_r2dreamer_smaclite_multimap.py --config configs/r2_650_kaggle.yaml
TXT
