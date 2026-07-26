# Phase 1A Environment Setup — Miniconda on Linux Cloud GPU

This guide configures a Python virtual environment (via **Miniconda**) on a **Linux cloud
GPU** terminal to run **Phase 1A** of the R2-Dreamer migration:

- the Gymnasium-compatible `SMACliteDreamerEnv`,
- the factorised action codec,
- the Phase 1A test suite (`tests/`),
- the Gymnasium smoke test (`scripts/smoke_test_gym_smaclite_env.py`).

> **Scope note.** Phase 1A is the **environment + action interface** only. It does **not**
> require PyTorch / TorchRL / TensorDict — those belong to the R2-Dreamer model (Phase 1B+).
> To keep the Phase 1A environment small and fast, the core steps install only what Phase
> 1A needs (`gymnasium`, `numpy`, the SMAClite simulator, `pytest`). The **full
> R2-Dreamer stack** (torch 2.8 + CUDA, torchrl, tensordict, hydra, …) is in the
> [optional section](#optional-install-the-full-r2-dreamer-stack-phase-1b) at the end.

Python version is pinned to **3.11** to match `external/r2dreamer/pyproject.toml`
(`requires-python = ">=3.11,<3.12"`), so the same conda env carries forward to later phases.

---

## 0. Prerequisites

- A Linux shell on the cloud GPU instance (bash).
- The repository checked out, e.g. at `~/smac-dreamer`. Adjust `PROJECT_DIR` below.
- For Phase 1A the GPU is **not** required (the env + tests are CPU-only). The GPU matters
  from Phase 1B onward.

```bash
# Set once per shell session. Change this to wherever you cloned the repo.
export PROJECT_DIR=~/smac-dreamer
cd "$PROJECT_DIR"

# Confirm you are on the migration branch with a clean tree.
git rev-parse --abbrev-ref HEAD     # expect: r2dreamer
git status --short                  # expect: no unexpected changes
```

---

## 1. Install Miniconda (skip if `conda` already works)

```bash
# Download and install Miniconda for Linux x86_64.
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Make `conda` available in the current shell and initialise it for future shells.
source ~/miniconda3/bin/activate
conda init bash
# Reload the shell config so `conda activate` works without restarting the terminal.
source ~/.bashrc
```

Verify:

```bash
conda --version
```

---

## 2. Create and activate the conda environment

```bash
# Create a Python 3.11 env named "smacdreamer" (matches r2dreamer's requires-python).
conda create -y -n smacdreamer python=3.11

conda activate smacdreamer
python --version          # expect: Python 3.11.x
```

---

## 3. Install Phase 1A dependencies

The SMAClite simulator (`external/smaclite`) requires `numpy`, `gymnasium`, `pygame`,
`scikit-learn`, and `Rtree` (which needs the system `libspatialindex` library). `Rtree`
wheels on PyPI bundle the native library, but installing it via conda-forge is the most
reliable on a headless cloud box. `numba` and `scipy` are pulled in transitively by some
SMAClite code paths; we install them explicitly to avoid surprises.

```bash
# Native library for Rtree (spatial index) — most robust via conda-forge.
conda install -y -c conda-forge libspatialindex rtree

# Python deps for Phase 1A + the SMAClite simulator.
# numpy/gymnasium versions are aligned with external/r2dreamer/pyproject.toml so this env
# is forward-compatible with later phases.
pip install \
  "numpy==1.26.0" \
  "gymnasium==1.2.0" \
  "scikit-learn" \
  "scipy" \
  "numba" \
  "pygame" \
  "pytest"
```

> **Headless note.** `pygame` is only used by SMAClite for optional rendering. The env,
> tests, and smoke test run without a display. If `pygame` import ever complains about a
> missing video device, set a dummy driver: `export SDL_VIDEODRIVER=dummy`.

---

## 4. Make the project and SMAClite importable

Phase 1A imports `smacdreamer.*` from `src/` and the vendored simulator from
`external/smaclite`. It deliberately does **not** put `external/dreamerv3` on the path (so
the environment stays JAX-free). The test suite's `tests/conftest.py` already adds `src`
and `external/smaclite` to `sys.path`, and the smoke-test script does the same — so for
running the tests/smoke test you do **not** need to set `PYTHONPATH` manually.

For ad-hoc `python` / `ipython` sessions outside the tests, set:

```bash
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/external/smaclite"
```

(Optional, cleaner alternative — install both as editable packages so no `PYTHONPATH` is
needed at all:)

```bash
pip install -e "$PROJECT_DIR/external/smaclite"
# The project itself has no setup.py yet; importing via PYTHONPATH=src is the supported path.
```

---

## 5. Run the Phase 1A test suite

```bash
cd "$PROJECT_DIR"
python -m pytest tests/ -v
```

**Expected (with the SMAClite simulator installed): all tests pass** — the 18 pure-NumPy
codec tests plus the 14 env tests and 6 padding tests (which were skipped on a machine
without the simulator).

To run only the simulator-free codec tests (fast sanity check):

```bash
python -m pytest tests/test_action_codec.py -v
```

---

## 6. Run the Gymnasium smoke test

```bash
cd "$PROJECT_DIR"
python scripts/smoke_test_gym_smaclite_env.py --scenario 2s3z
```

This resets the env, samples valid factorised one-hot actions, runs one full episode, and
prints: total reward, episode length, battle outcome, invalid-action count,
masking-failure count, and the final observation shapes. It imports **no** JAX / Elements /
Embodied / Portal / DreamerV3.

Try other built-in scenarios too, e.g.:

```bash
python scripts/smoke_test_gym_smaclite_env.py --scenario 3s5z
```

---

## 7. Quick verification checklist

```bash
# Python is 3.11 and in the conda env.
python -c "import sys; print(sys.version)"

# Core deps import.
python -c "import numpy, gymnasium, sklearn, scipy, numba, pytest; print('phase1a deps OK')"

# SMAClite simulator imports and registers its gym ids.
PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/external/smaclite" \
  python -c "import smaclite; print('smaclite OK')"

# The migrated env + codec import JAX-free.
PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/external/smaclite" python - <<'PY'
import sys
from smacdreamer.envs.action_codec import FactorisedActionCodec
from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
forbidden = [m for m in ("jax","elements","embodied","portal","dreamerv3") if m in sys.modules]
print("forbidden modules:", forbidden or "NONE (good)")
c = FactorisedActionCodec(num_agents=3, num_actions=4)
print("codec round trip:", c.decode(c.encode([1,2,3])) == [1,2,3])
PY
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | `pip install scikit-learn` (step 3 missed). |
| `Rtree`/`libspatialindex` load error | `conda install -y -c conda-forge libspatialindex rtree` (step 3). |
| `pygame` errors about a video device | `export SDL_VIDEODRIVER=dummy` before running. |
| `ModuleNotFoundError: No module named 'smacdreamer'` | Set `PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/external/smaclite"`, or run via `pytest` from the repo root. |
| Env tests still **skipped** | The suite skips env/padding tests when `import smaclite` fails. Re-run the step-7 `smaclite OK` check and fix the underlying import error. |
| `numpy` version conflicts later | Phase 1A pins `numpy==1.26.0` to match r2dreamer; don't upgrade it. |

---

## Optional: install the full R2-Dreamer stack (Phase 1B+)

Phase 1A does not need these, but the **same** `smacdreamer` conda env should host them
when you move to model integration. Install PyTorch built for the instance's CUDA version
(check with `nvidia-smi`), then the rest of the r2dreamer deps.

```bash
conda activate smacdreamer

# PyTorch 2.8.0 with CUDA 12.1 wheels (adjust cu121 to match the box's CUDA).
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121

# Remaining R2-Dreamer runtime deps (versions from external/r2dreamer/pyproject.toml).
pip install \
  "torchrl==0.9.2" \
  "tensordict==0.9.1" \
  "hydra-core==1.3.2" \
  "einops==0.3.0" \
  "ruamel.yaml==0.17.4" \
  "tensorboard>=2.20,<3" \
  "wandb"

# Sanity check GPU visibility.
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

> Do not pin `numpy` above `1.26.x` — r2dreamer requires `numpy==1.26.0`.

---

## Environment summary

| Component | Phase 1A value | Notes |
|---|---|---|
| Conda env name | `smacdreamer` | Python 3.11 |
| Python | 3.11 | matches `r2dreamer` `requires-python` |
| numpy | 1.26.0 | pinned for forward-compat with r2dreamer |
| gymnasium | 1.2.0 | r2dreamer-aligned |
| SMAClite deps | scikit-learn, scipy, numba, pygame, Rtree (+ libspatialindex) | from `external/smaclite/setup.py` (+ transitive) |
| Test runner | pytest | `python -m pytest tests/` |
| GPU required for 1A | No | needed from Phase 1B |
| PyTorch/TorchRL/TensorDict | **not installed in 1A** | see optional section for 1B+ |
```
