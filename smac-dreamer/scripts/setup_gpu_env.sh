#!/usr/bin/env bash
# Set up the smac-r2 conda env for a CLOUD GPU box (Kubeflow notebook, A40, CUDA 12.x).
#
# Mirrors docs/phase1b_windows_conda_setup.md but for Linux + CUDA torch. Installs
# Miniconda into your HOME (which on a Kubeflow notebook lives on the persistent
# workspace volume, so the env survives notebook restarts). Idempotent: re-running
# skips steps that are already done.
#
# Usage (from the repo root, in the notebook terminal):
#   bash scripts/setup_gpu_env.sh
#   conda activate smac-r2          # then run training (see docs)
set -euo pipefail

ENV_NAME=smac-r2
PY_VER=3.11
# Driver 535 (CUDA 12.2) runs cu126 torch fine via CUDA minor-version compatibility.
# If `torch.cuda.is_available()` is False after install, retry with cu124 or the
# default PyPI wheel (drop --index-url).
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu126"

echo "==> 1/6  Ensure Miniconda is installed (into \$HOME, persists on the workspace PVC)"
if ! command -v conda >/dev/null 2>&1; then
  if [ ! -d "$HOME/miniconda3" ]; then
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm -f /tmp/miniconda.sh
  fi
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
# Make `conda activate` work in this non-interactive shell.
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "==> 2/6  Create the $ENV_NAME env (Python $PY_VER)"
if ! conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  conda create -y -n "$ENV_NAME" python="$PY_VER"
fi
conda activate "$ENV_NAME"
python --version

echo "==> 3/6  Native SMAClite dep (libspatialindex) via conda-forge — no root needed"
conda install -y -c conda-forge libspatialindex rtree

echo "==> 4/6  PyTorch (CUDA) — pinned to the r2dreamer version"
pip install "torch==2.8.0" --index-url "$TORCH_CUDA_INDEX"

echo "==> 5/6  R2-Dreamer + SMAClite Python deps (exact pins)"
pip install \
  "torchrl==0.9.2" "tensordict==0.9.1" "gymnasium==1.2.0" "numpy==1.26.0" \
  "omegaconf" "einops==0.3.0" "ruamel.yaml==0.17.4" "tensorboard>=2.20,<3" \
  "hydra-core==1.3.2" "cloudpickle" "wandb"
pip install scikit-learn scipy numba pygame

echo "==> 6/6  Verify imports + GPU visibility"
export SDL_VIDEODRIVER=dummy   # headless pygame (no display on a notebook pod)
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA NOT visible — request a GPU on the notebook server, "
                     "or retry torch with a different cu** index (see top of script).")
import sys, pathlib
root = pathlib.Path.cwd()
for p in ("src", "external/r2dreamer", "external/smaclite"):
    sys.path.insert(0, str(root / p))
import smaclite, tools, buffer, dreamer, trainer  # noqa: F401
from smacdreamer.r2dreamer_factory import make_smaclite_multimap_envs  # noqa: F401
print("all imports OK — env ready")
PY

echo
echo "Done. Next:"
echo "  conda activate $ENV_NAME"
echo "  export SDL_VIDEODRIVER=dummy"
echo "  python scripts/train_r2dreamer_smaclite_multimap.py --config configs/multimap_gpu.yaml"
