# R2-Dreamer × SMAClite

Training pipeline for **R2-Dreamer** (a decoder-free DreamerV3-family world-model agent) on the
**SMAClite** simulator, treating the multi-agent environment as a **single-agent centralised
control** problem: one Dreamer agent drives all allied units.

Project code lives under `src/`, `scripts/`, and `configs/`; the upstream agent/simulator are in
`external/r2dreamer` and `external/smaclite`.

---

## Quick start

```bash
conda activate smac-r2          # see docs/cloud_kubeflow_notebook_gpu_setup.md for env setup
wandb login                     # once, for online logging

# memory + headless env vars (inherited by spawned env workers)
export MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy

# production 2M-step run (GPU)
python scripts/train_r2dreamer_smaclite_multimap.py --config configs/r2_650.yaml

# post-training: evaluate a checkpoint on a blind split
python scripts/evaluate_multimap.py --config configs/r2_650.yaml \
    --checkpoint logs/r2dreamer/r2_650/best_val_macro_winrate.pt --split blind_iid
```

Optional frozen-JEPA backend work lives behind `world_model.backend: jepa` and
requires installing the local `smac-jepa-wm` package plus providing a real JEPA
checkpoint. Synthetic safety tests run without those files, but long JEPA
training should wait until the combined preflight and 5,000-step smoke run pass:

```bash
python scripts/preflight_jepa_training.py \
  --checkpoint /path/to/checkpoint.pt \
  --episode-npz /path/to/episode.npz \
  --config configs/r2_650_jepa.yaml \
  --device cpu \
  --report-json logs/jepa_preflight_report.json
```

See [`docs/jepa_r2dreamer_integration.md`](docs/jepa_r2dreamer_integration.md).

See [`docs/CHANGES.md`](docs/CHANGES.md) for the full feature changelog and
[`docs/diagnostics/`](docs/diagnostics/) for diagnostic reports.

---

## Training pipeline overview

End to end, from launch to artifacts.

### 1. Entry point & config
`scripts/train_r2dreamer_smaclite_multimap.py --config configs/r2_650.yaml`. One YAML drives
everything: dataset folders, model size, reward, observation mode, masking, validation cadence,
replay, AMP, and W&B. The model/buffer/trainer hyperparameters are built by
`train_r2dreamer_smaclite_debug.make_config()` and then overridden from the YAML (device
propagated recursively, buffer capacity/backend, `action_masking`, `amp_dtype`, …).

**Preflight:** `cuda_preflight` validates the GPU (BF16 support, wheel/arch match, tiny tensor
test) **before** the expensive discovery pass — it fails loudly rather than wasting a run.

### 2. Dataset & map discovery
Explicit folders (no ratio split): `maps.{train, validation, blind_iid, blind_compositional}`.
Training discovers **only** `train`; `validation` is held out for checkpoint selection; the blind
splits are never touched during training. Discovery (`map_discovery.discover_folders`) probes each
map in a **recycled subprocess pool** (avoids the SMAClite native-memory blow-up), validates it,
sets the model **padding from the TRAIN-max** (`max_agents/enemies/actions`), and safety-net-checks
that every map fits.

### 3. Environment (centralised control)
One Dreamer agent drives **all allied units** as a single centralised controller.
`SMACliteDreamerEnv` wraps SMAClite as a gymnasium env:

- **Action** = one categorical per allied agent slot (factorised `A` groups of `C` actions),
  padded to `max_agents × max_actions`.
- **Observation** (`observation.mode: structured`) = canonical per-entity blocks (self / ally /
  enemy features, movement, `avail_actions`, agent slot/alive masks, entity masks) with **fixed
  semantic positions across all maps**, HP and shields in separate dims, and a **global unit-type
  vocabulary**.
- **Reward** = the swappable reward (`smaclite_default`; `dense_v3` / `ally_ehp_v4` also available);
  the original SMAClite return is always tracked for selection.
- **Parallelism**: `env_num` spawn-worker envs (`ParallelEnv`), each sampling maps
  `shuffled_round_robin`; **workers recycle every `max_episodes_per_worker` episodes** to reclaim
  SMAClite's per-reset native memory leak (the fix for long-run OOM).

### 4. The R2-Dreamer model
A DreamerV3-family world model + actor-critic, single LaProp optimizer (`lr 4e-5`, AGC, warmup):

- **World model**: encoder (MLP over the flattened structured obs) → **RSSM** (recurrent latent,
  discrete stochastic state) + reward & continuation heads. Representation is learned by
  R2-Dreamer's **decoder-free redundancy-reduction (Barlow)** objective — no observation
  reconstruction.
- **Actor-critic**: factorised multi-one-hot actor + critic, trained in **latent imagination**.
- **Masking heads**: `avail_head` + `alive_head` predict the next available-action / alive masks
  (BCE), used to mask actions inside imagination.

### 5. Training loop (`ValidationTrainer.begin`, per step)
1. **Collect**: step the envs; the policy acts via `Dreamer.act()`. With masking on, invalid
   actions are set to −∞ before sampling and padded/dead agents are forced to NOOP, so **only valid
   actions are ever executed**. Transitions (obs, executed action, latent state) go to replay.
2. **Replay**: large-capacity buffer with **CPU / memmap storage** (off-RAM), sampled to GPU on
   demand; updates run at a fixed collect:update ratio.
3. **World-model update** (`Dreamer._cal_grad`, one weighted loss, one backward): dynamics +
   representation KL, reward, continuation, Barlow, and the auxiliary avail/alive BCE — on **real**
   replay data.
4. **Actor-critic in imagination**: roll the policy forward in latent space; actions masked by the
   **predicted** avail/alive masks; reset/terminal states excluded as imagination starts; λ-returns
   drive the actor (policy-gradient + entropy) and critic. AMP autocast is fp16 or **bf16** per
   `amp_dtype` (bf16 avoids overflow on the large observation).

### 6. Validation & checkpoint selection
Every `validation.every` steps (e.g. 100k; not at start), `ValidationTrainer.eval` runs a dedicated
**map × fixed-seed** pass over the validation maps with the **original** reward: it logs per-map +
macro/micro win rate, original return, length, timeout rate, and ally/enemy effective-HP. It saves
**`best_val_macro_winrate.pt`** selected by macro win rate (tie-break macro original return) —
**never** shaped return. `latest.pt` is written periodically by wall-clock.

### 7. Logging & artifacts
W&B (project + run name from config): world-model / actor-critic losses, mask precision/recall +
pre/post-mask invalid diagnostics, `val/macro_*` + `val/micro_*`, and system/memory telemetry.
Artifacts under `logs/r2dreamer/<run>/`: `latest.pt`, `best_val_macro_winrate.pt`, `run_meta.json`
(exact obs mode + model dims so standalone eval rebuilds the model), and the memmap replay scratch.

### 8. Post-training evaluation
`scripts/evaluate_multimap.py --checkpoint best_val_macro_winrate.pt --split blind_iid|blind_compositional`
reconstructs the exact model from `run_meta.json` and reports per-map + macro/micro metrics on the
blind splits.

**One-line mental model:** discover train maps → centralised structured-obs SMAClite envs (recycled
workers) → collect masked-valid actions into a CPU/memmap replay → R2-Dreamer learns a decoder-free
world model and masks the actor-critic in imagination → select checkpoints by held-out macro win
rate.

---

## Key configs
- `configs/r2_650.yaml` — production: explicit folders, structured obs, action masking, bf16,
  500k/memmap replay, validation every 100k.
- `configs/ablation/{A,B,C,D}_*.yaml` — reward ablation (`smaclite_default` / `dense_v3` /
  `ally_ehp_v4`), identical except the reward block.
- `configs/multimap_gpu.yaml`, `configs/multimap.yaml` — GPU / CPU-debug variants.
