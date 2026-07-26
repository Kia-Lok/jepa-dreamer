# Phase 1B Environment Setup — Conda on Windows (Command Prompt)

This guide creates a Conda virtual environment on **Windows** and runs the **Phase 1B
single-map R2-Dreamer training** to confirm the current workflow still works:

```
python scripts\train_r2dreamer_smaclite_debug.py --scenario 2s_vs_1sc --steps 500
```

All commands are written for **Command Prompt (`cmd.exe`)**, not PowerShell.

> **Why a CPU debug run.** `scripts\train_r2dreamer_smaclite_debug.py` is hard-coded to
> `device="cpu"` and `compile=False` because `torch.compile`/Triton is unavailable on
> Windows. This run verifies the collect → world-model update → actor-critic loop end to
> end on CPU with a small model (`deter=256`, `units=128`). It is a correctness check, not
> a performance run.

---

## What the run needs (verified against the code)

| Dependency | Pin (from `external/r2dreamer/pyproject.toml`) | Why |
|---|---|---|
| Python | 3.11 (`>=3.11,<3.12`) | r2dreamer requires-python |
| torch | 2.8.0 | model / training |
| torchrl | 0.9.2 | replay buffer (`buffer.py`) |
| tensordict | 0.9.1 | transition format |
| gymnasium | 1.2.0 | env interface |
| numpy | 1.26.0 | arrays (do **not** upgrade to 2.x) |
| omegaconf | (any 2.3.x) | the debug script builds an OmegaConf config directly |
| tensorboard | >=2.20,<3 | `tools.Logger` (default logger) |
| cloudpickle | (any) | `envs/parallel.py` worker processes |
| einops, ruamel.yaml | 0.3.0 / 0.17.4 | imported by r2dreamer (not at load time, but install for safety) |
| wandb | (any) | only if you pass `--wandb-project` |
| **SMAClite sim** | scikit-learn, scipy, numba, pygame, Rtree (+ libspatialindex) | `external/smaclite` runtime |

The script puts `src`, `external\r2dreamer`, and `external\smaclite` on `sys.path` itself,
so **no `PYTHONPATH` is required** to run it.

---

## 0. Open Command Prompt at the project root

```bat
cd /d "c:\Users\User\OneDrive - National University of Singapore\Documents\smac-dreamer"
```

(`/d` lets `cd` change drives if needed. Keep the quotes — the path has spaces.)

Confirm the branch and that the JAX cleanup is in place:

```bat
git rev-parse --abbrev-ref HEAD
git status --short
```

---

## 1. Install Miniconda (skip if already installed)

1. Download the Windows x86_64 installer from:
   `https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe`
2. Run it. Recommended options:
   - Install for **Just Me**.
   - Default location `%USERPROFILE%\miniconda3`.
   - Leave "Add Miniconda to PATH" **unchecked** (use the Anaconda Prompt instead — the
     supported way on Windows).

After installing, **close this Command Prompt** and open **"Anaconda Prompt (miniconda3)"**
from the Start menu. All remaining commands run in the Anaconda Prompt (it is a
`cmd.exe` with conda activated).

Verify:

```bat
conda --version
```

> If you prefer a normal Command Prompt, you can instead initialise conda for cmd once:
> `"%USERPROFILE%\miniconda3\Scripts\conda.exe" init cmd.exe`, then open a fresh
> Command Prompt. The Anaconda Prompt is simpler and is assumed below.

---

## 2. Create and activate the environment (Python 3.11)

```bat
conda create -y -n smac-r2 python=3.11
conda activate smac-r2
python --version
```

Expected: `Python 3.11.x`.

Re-enter the project directory inside the Anaconda Prompt:

```bat
cd /d "c:\Users\User\OneDrive - National University of Singapore\Documents\smac-dreamer"
```

---

## 3. Install the SMAClite simulator's native + Python deps

`Rtree` needs the native `libspatialindex` library; conda-forge is the reliable source on
Windows.

```bat
conda install -y -c conda-forge libspatialindex rtree
```

---

## 4. Install PyTorch (CPU) — exact pin first

Try the exact pinned CPU build first:

```bat
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

**If that fails** (no matching Windows CPU wheel for 2.8.0), fall back to the newest CPU
build and note the change in your run log:

```bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify torch imports and reports CPU:

```bat
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
```

---

## 5. Install the remaining R2-Dreamer + project dependencies

Exact pins first:

```bat
pip install torchrl==0.9.2 tensordict==0.9.1 gymnasium==1.2.0 numpy==1.26.0 omegaconf einops==0.3.0 "ruamel.yaml==0.17.4" "tensorboard>=2.20,<3" cloudpickle
```

Then the SMAClite Python deps:

```bat
pip install scikit-learn scipy numba pygame
```

Optional — only if you want Weights & Biases logging:

```bat
pip install wandb
```

### If the exact pins don't resolve on Windows

`torchrl`/`tensordict` versions are tightly coupled to the torch version. If pip reports a
resolution error, relax **torchrl + tensordict together** to the build that matches your
installed torch, e.g.:

```bat
pip install torchrl tensordict
```

`tensordict` and `torchrl` must come from the same release line as torch — if torch fell
back in step 4, let pip pick matching `torchrl`/`tensordict` here. Record any version that
differs from the pins above; the Kubeflow image will still use the exact pins.

---

## 6. Verify the environment imports cleanly (no JAX)

```bat
python -c "import torch, torchrl, tensordict, gymnasium, numpy, omegaconf, tensorboard, cloudpickle; print('core OK')"
python -c "import sklearn, scipy, numba, pygame; print('smaclite deps OK')"
```

Confirm the SMAClite simulator and the r2dreamer modules import via the script's own path
setup:

```bat
python -c "import sys, pathlib; r=pathlib.Path.cwd(); [sys.path.insert(0,str(r/p)) for p in (r'src', r'external\r2dreamer', r'external\smaclite')]; import smaclite, tools, buffer, dreamer, trainer; from smacdreamer.r2dreamer_factory import make_smaclite_envs; print('all imports OK (JAX-free)')"
```

> Headless note: if `pygame` complains about a video device, set `set SDL_VIDEODRIVER=dummy`
> before running. The training run does not render.

---

## 7. Run the Phase 1B single-map training

The map of interest is **2s_vs_1sc** (the working single-map run). 500 steps:

```bat
python scripts\train_r2dreamer_smaclite_debug.py --scenario 2s_vs_1sc --steps 500 --logdir logs\r2dreamer\2s_vs_1sc_500
```

With Weights & Biases logging (matches the existing workflow):

```bat
python scripts\train_r2dreamer_smaclite_debug.py --scenario 2s_vs_1sc --steps 500 --logdir logs\r2dreamer\2s_vs_1sc_500 --wandb-project smac-r2dreamer --wandb-run 2s_vs_1sc-500
```

**Phase 1B acceptance criteria (from the script docstring):**
- Script starts without import errors.
- World-model losses (`rew`, `con`, `dyn`, `rep`) appear in the logs after ~10 env steps.
- A checkpoint is written to `<logdir>\latest.pt` on completion.
- No crash for the full `--steps` run.

Check the outputs:

```bat
dir logs\r2dreamer\2s_vs_1sc_500
type logs\r2dreamer\2s_vs_1sc_500\metrics.jsonl
```

(`latest.pt` should exist; `metrics.jsonl` should contain the loss scalars. With
`--wandb-project`, scalars also appear in the W&B run.)

---

## 8. (Optional) run the Phase 1A tests in this env

The same env also runs the test suite and the Gymnasium smoke test:

```bat
pip install pytest
python -m pytest tests\ -v
python scripts\smoke_test_gym_smaclite_env.py --scenario 2s_vs_1sc
```

With the simulator installed, the env/padding tests that previously skipped should now run.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `'conda' is not recognized` | Use the **Anaconda Prompt**, or run `"%USERPROFILE%\miniconda3\Scripts\conda.exe" init cmd.exe` and reopen Command Prompt. |
| `No module named 'sklearn'` | `pip install scikit-learn` (step 5). |
| `Rtree` / `libspatialindex` load error | `conda install -y -c conda-forge libspatialindex rtree` (step 3). |
| `pygame` video-device error | `set SDL_VIDEODRIVER=dummy` before running. |
| `Could not find a version that satisfies torch==2.8.0` | Drop the pin: `pip install torch --index-url https://download.pytorch.org/whl/cpu` (step 4 fallback). |
| `torchrl`/`tensordict` ABI / version mismatch with torch | Reinstall both unpinned so pip matches your torch: `pip install torchrl tensordict`. |
| `numpy` 2.x conflicts | Keep `numpy==1.26.0`; reinstall it if something upgraded it. |
| `No module named 'tools'` / `'dreamer'` | Run the script from the **project root** so its `sys.path` setup resolves `external\r2dreamer`. |
| Worker process / pickling error on `env-num>1` | Use `--env-num 1` (default) on Windows; spawn-based ParallelEnv is most reliable single-process for the debug run. |

---

## Environment summary

| Component | Value |
|---|---|
| Conda env | `smac-r2` (Python 3.11) |
| Device | CPU (Windows; `compile=False`) |
| Entry point | `scripts\train_r2dreamer_smaclite_debug.py` |
| Phase 1B map | `2s_vs_1sc`, 500 steps |
| Success signal | losses logged + `<logdir>\latest.pt` written, no crash |
| Forward-compat | same pins as `external\r2dreamer\pyproject.toml`; Kubeflow uses CUDA torch instead of CPU |
```
