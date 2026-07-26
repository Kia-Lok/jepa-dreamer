# Cloud GPU Setup — Kubeflow Notebook Server (NVIDIA A40, CUDA)

This guide runs **multimap R2-Dreamer × SMAClite training on a cloud GPU** using a
**Kubeflow Notebook server** (interactive terminal, code pulled from GitHub — no Docker,
no PyTorchJob, no registry).

It is the cloud/Linux/GPU twin of [`phase1b_windows_conda_setup.md`](phase1b_windows_conda_setup.md).
The same entry point is used; only the device, deps, and config differ.

Target GPU: **NVIDIA A40, 46 GB, driver 535.161 (CUDA 12.2)**.

---

## 0. One-time code fix already applied

`scripts/train_r2dreamer_smaclite_multimap.py` previously only moved **three** config
fields to the chosen device. The reused debug config hard-codes `device="cpu"` in many
nested places (`buffer`, `storage_device`, `encoder`, `decoder`, and every head). On CPU
that is invisible, but on GPU it causes a **CUDA/CPU mismatch crash**. The script now
calls `_propagate_device(config, cfg.device)` to set **every** device field. Nothing for
you to do — just be aware that `device: cuda:0` in the YAML now fully takes effect.

---

## 1. Create the Kubeflow Notebook server (with the GPU attached)

In the Kubeflow UI → **Notebooks → New Notebook**:

| Field | Value | Why |
|---|---|---|
| **Image** | A CUDA-capable image (e.g. a `jupyter-pytorch-cuda` / `*-cuda-full` base) | Must have NVIDIA userspace libs. We install our own conda env on top, so the image's torch version doesn't matter. |
| **CPU / RAM** | ≥ 4 CPU, ≥ 16 GB | `env_num: 4` spawns parallel SMAClite workers. |
| **GPUs → Number** | **1** | Without this the A40 is not scheduled into the pod. |
| **GPUs → Vendor** | **NVIDIA** (`nvidia.com/gpu`) | |
| **Workspace Volume** | New PVC, **≥ 20 GB** | This is your persistent home — see Storage below. |

Create it, wait for **Running**, then **Connect** and open a **Terminal**
(New → Terminal in JupyterLab).

Confirm the GPU is actually in the pod:

```bash
nvidia-smi          # should list the A40, ~46 GB, driver 535.161
```

If `nvidia-smi` is missing or shows no GPU, the notebook was created **without** a GPU —
delete it and recreate with GPUs = 1.

---

## 2. Storage — where checkpoints + logs live (recommended setup)

**Recommendation: workspace PVC as the primary store + Weights & Biases for metrics.**

- The **workspace volume** (your `$HOME`, where you clone the repo) is a PVC that
  **persists across notebook stop/start**. Writing `logs/.../latest.pt` there is durable
  for the lifetime of the notebook server. This is the simplest correct choice and needs
  no extra credentials.
- Caveat: if you **delete the notebook server**, that PVC is usually deleted with it. So
  for anything you must not lose:
  - Enable **W&B** (`wandb.project` in the config) so all metrics stream off-pod, and
  - When a run finishes, **download `latest.pt`** (right-click in the JupyterLab file
    browser → Download), or push it to object storage / W&B artifacts.
- Installing Miniconda into `$HOME` (step 3) also lands it on this PVC, so the **env
  itself survives restarts** — you only run the setup once.

If your cluster offers a separate, longer-lived **data volume**, mount it too and point
`logdir:` at it. Object storage (S3/GCS) is only worth the extra credential wiring if you
expect to recreate notebook servers often.

---

## 3. Get the code + build the conda env

In the notebook terminal:

```bash
cd ~
git clone https://github.com/<your-account>/<your-repo>.git smac-dreamer
cd smac-dreamer

bash scripts/setup_gpu_env.sh
```

`scripts/setup_gpu_env.sh` (created for this setup) installs Miniconda into `$HOME`,
creates the `smac-r2` Python 3.11 env, installs **CUDA torch 2.8.0** plus the exact
R2-Dreamer + SMAClite pins, and verifies that `torch.cuda.is_available()` is `True`.

> **CUDA wheel note.** The script uses the `cu126` torch wheel, which runs on your 535
> driver via CUDA minor-version compatibility. If verification reports CUDA **not**
> available, edit `TORCH_CUDA_INDEX` at the top of the script to `cu124` (or drop the
> `--index-url` for the default PyPI build) and re-run.

Activate it for the rest of the session:

```bash
conda activate smac-r2
export SDL_VIDEODRIVER=dummy     # headless pygame; no display on a notebook pod
```

---

## 4. Provide the `easy500` map folder

`configs/multimap_gpu.yaml` points `maps_folder` at `configs/maps/easy500`, **which is
not in the repo** (only `easy100` and `500map_v1` are). Do one of:

- Copy/generate your 500-map set into `configs/maps/easy500/` (as `*.json` scenarios), **or**
- Edit `maps_folder` in `configs/multimap_gpu.yaml` to an existing folder, e.g.
  `configs/maps/easy100`, for a first run.

The factory scans the folder, derives padding from the TRAIN-max, and prints the
train/held-out split at startup — sanity-check those numbers before committing GPU hours.

---

## 5. Run training

```bash
# Quick GPU smoke (few minutes) — proves the cuda loop end-to-end:
python scripts/train_r2dreamer_smaclite_multimap.py --config configs/multimap_gpu.yaml --steps 1000

# Full run (uses `steps:` from the config):
python scripts/train_r2dreamer_smaclite_multimap.py --config configs/multimap_gpu.yaml
```

For a long run that survives terminal/SSH drops, use the JupyterLab terminal plus `nohup`
(or `tmux` if the image has it):

```bash
nohup python scripts/train_r2dreamer_smaclite_multimap.py \
  --config configs/multimap_gpu.yaml > logs/run.out 2>&1 &
tail -f logs/run.out
```

Watch GPU use in a second terminal: `watch -n2 nvidia-smi`.

### Tuning (`configs/multimap_gpu.yaml`)
The A40 has 46 GB, so the defaults (`deter: 2048`, `units: 256`, `batch_size: 16`,
`batch_length: 64`, `env_num: 4`) leave plenty of head-room — raise `batch_size`,
`env_num`, or `steps` as needed. Keep `gamma` equal to the agent discount.

---

## 6. Verify success / retrieve results

- **W&B** (if `wandb.project` set): world-model losses, `invalid-action`,
  per-term `log_reward_*`, and periodic `episode/eval_battle_won` +
  `episode/eval_reward_original` on the held-out maps.
- **On disk:** `logs/r2dreamer/multimap_gpu/latest.pt` (checkpoint, every 10 min) and
  `run_config.json` (resolved reward + padding). Download `latest.pt` from the JupyterLab
  file browser when done.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nvidia-smi` not found / no GPU | Notebook created without a GPU — recreate with GPUs = 1, vendor NVIDIA. |
| `torch.cuda.is_available()` is `False` | Switch `TORCH_CUDA_INDEX` to `cu124` (or default PyPI) in `setup_gpu_env.sh` and re-run. |
| `RuntimeError: ... cuda ... cpu ...` mismatch | Ensure you pulled the repo **with** the `_propagate_device` fix (step 0); `device:` must be `cuda:0`. |
| `Rtree` / `libspatialindex` load error | `conda install -y -c conda-forge libspatialindex rtree` (the script does this). |
| `pygame` video-device error | `export SDL_VIDEODRIVER=dummy`. |
| Maps folder empty / startup assert on padding | Point `maps_folder` at a folder that actually contains `*.json` maps (step 4). |
| Worker/pickling error at `env_num>1` | Lower `env_num` to 1 to isolate, then raise once the loop is verified. |
| Lost work after deleting the notebook | The workspace PVC was deleted with it — use W&B + download `latest.pt` before deleting (step 2). |

---

## Summary

| Component | Value |
|---|---|
| Platform | Kubeflow Notebook server, interactive terminal |
| GPU | NVIDIA A40 46 GB (driver 535 / CUDA 12.2), `device: cuda:0` |
| Env | `smac-r2` conda (Python 3.11), Miniconda in `$HOME` on the workspace PVC |
| Setup | `bash scripts/setup_gpu_env.sh` |
| Entry point | `scripts/train_r2dreamer_smaclite_multimap.py --config configs/multimap_gpu.yaml` |
| Storage | workspace PVC (primary) + W&B (durability) |
| Maps | provide `configs/maps/easy500/` (or repoint `maps_folder`) |
