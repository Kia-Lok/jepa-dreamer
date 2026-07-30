# JEPA-Dreamer for SMACLite

Research code for learning an entity-level JEPA world model from SMACLite trajectories and integrating it with R2-Dreamer for centralised multi-agent control.

The repository contains:

- data collection and dataset auditing tools;
- a visibility-aware JEPA world model;
- a direct multi-horizon forecast extension;
- R2-Dreamer training with a frozen JEPA checkpoint;
- tactical policy training;
- ordinary actor-critic training;
- hierarchical Option-Critic training;
- evaluation and checkpoint-validation utilities.

Collected datasets, checkpoints, replay buffers, logs and experiment outputs are intentionally not included.

## Main models

| Component | Role |
|---|---|
| **Best working model** | Visibility-aware JEPA world model with anchored recurrent memory, action conditioning and hidden-entity belief retention. This is the default checkpoint for Dreamer training. |
| **Experimental forecast model** | Direct multi-horizon extension trained to predict future latent states at power-of-two horizons. Intended for forecasting experiments rather than as the default RL world model. |
| **R2-Dreamer integration** | Uses a frozen JEPA checkpoint to provide latent world-model features for policy learning. |
| **Tactical policy** | Intermediate policy used as the source checkpoint for controlled actor-critic and hierarchical comparisons. |
| **Ordinary actor-critic** | Non-hierarchical comparison policy trained from the tactical source checkpoint. |
| **Option-Critic** | Hierarchical policy with anchored source behaviour and multiple trainable options. |

Experiment identifiers remain in filenames for compatibility, but the recommended workflow is described by model role rather than experiment number.

## Repository layout

```text
smac-jepa-wm/
├── simulator/          SMACLite trajectory collection
├── smac_jepa/          JEPA models, datasets and trainers
├── scripts/            JEPA training and evaluation launchers
├── tools/              checkpoint audits and forecast evaluation
├── splits/             dataset manifests
└── tests/              JEPA and forecast tests

smac-dreamer/
├── configs/            Dreamer, tactical and hierarchy configurations
├── external/           retained R2-Dreamer and SMACLite source
├── scripts/            training, evaluation and audit launchers
├── src/                JEPA–Dreamer integration
└── tests/              Dreamer, tactical and hierarchy tests

scripts/
└── validate_retained_workflows.sh
```

## Requirements

Recommended:

- Python 3.11
- Linux
- NVIDIA GPU for full training
- CUDA-compatible PyTorch
- `uv`
- SMACLite
- optional Weights & Biases account

Create the environment from the repository root:

```bash
git clone https://github.com/Kia-Lok/jepa-dreamer.git
cd jepa-dreamer

uv venv --python 3.11
source .venv/bin/activate
```

Install the combined environment from the repository root:

```bash
uv pip install -r requirements.txt
```

SMACLite's Rtree dependency also requires the native SpatialIndex library.

Optional W&B login:

```bash
wandb login
```

# 1. Collect training data

The JEPA collector runs SMACLite episodes, records entity states and joint actions, and stores trajectories as compressed `.npz` files.

Runtime datasets are written under:

```text
smac-jepa-wm/data/
```

This directory is ignored by Git.

## Collect one built-in scenario

```bash
cd smac-jepa-wm

uv run python simulator/collect_smaclite_data.py   --env-key smaclite:smaclite/2s3z-v0   --episodes 100   --max-steps 120   --out data/2s3z_random.npz   --seed 1
```

Important arguments:

- `--env-key`: SMACLite environment identifier;
- `--episodes`: number of episodes to collect;
- `--max-steps`: maximum recorded steps per episode;
- `--out`: output `.npz` file;
- `--seed`: collection seed.

## Collect from generated R2-2100 maps

The generated-map collector reads one directory of JSON configurations and creates:

- one `.npz` file per collected configuration;
- a train/evaluation manifest;
- consistent padded entity and action dimensions.

Example using the retained training-map directory:

```bash
uv run python simulator/collect_generated_configs.py   --config-dir ../smac-dreamer/configs/maps/r2_2100/configs/train   --out-dir data/r2_2100   --manifest-out splits/r2_2100_seed1.json   --episodes 64   --max-steps 120   --seed 1
```

For other splits, replace the config directory with the corresponding retained map directory.

The resulting manifest contains the dataset paths used by the trainers. Keep the manifest, but do not commit the generated `.npz` files.

## Audit the collected dataset

```bash
uv run python -m smac_jepa.audit_dataset   --manifest splits/r2_2100_seed1.json   --out reports/r2_2100_audit.json
```

The audit checks dataset readability, dimensions, episode lengths, masks, action widths and padding consistency.

# 2. Train the best working model

The recommended JEPA model uses:

- explicit visibility masks;
- separate invalid-slot and observation masks;
- anchored recurrent belief memory;
- action conditioning;
- event-balanced sampling;
- hidden-state retention for temporarily unobserved entities.

Run from `smac-jepa-wm`:

```bash
MANIFEST="$(pwd)/splits/r2_2100_seed1.json" OUT_DIR="$(pwd)/runs/best_working_model" WANDB=1 bash scripts/run_exp40_dreamer_event_balanced.sh
```

Disable W&B with `WANDB=0`.

Expected output:

```text
smac-jepa-wm/runs/best_working_model/checkpoint.pt
```

The launcher validates the produced checkpoint before completing.

## Evaluate recursive rollout quality

```bash
CHECKPOINT="$(pwd)/runs/best_working_model/checkpoint.pt" MANIFEST="$(pwd)/splits/r2_2100_seed1.json" OUT_DIR="$(pwd)/runs/best_working_model_rollout_eval" bash scripts/run_exp40_rollout_gallery.sh
```

This evaluates recursive prediction quality over multiple rollout horizons and produces summary metrics and visualisation-ready outputs.

# 3. Train the experimental forecast model

The forecast model starts from the best working checkpoint and learns direct latent predictions at horizons:

```text
1, 2, 4, 8 and 16 steps
```

It is intended for forecast experiments and should not replace the default JEPA checkpoint in ordinary Dreamer training unless explicitly being tested.

Run from the repository root:

```bash
cd ..

EXP40_CHECKPOINT="$(pwd)/smac-jepa-wm/runs/best_working_model/checkpoint.pt" MANIFEST="$(pwd)/smac-jepa-wm/splits/r2_2100_seed1.json" PIPE_DIR="$(pwd)/outputs/forecast_pipeline" WANDB=1 bash smac-dreamer/scripts/run_exp45_full_train_eval_resilient.sh
```

The pipeline performs:

1. checkpoint validation;
2. direct multi-horizon training;
3. ordinary forecast evaluation;
4. hidden-state forecast evaluation;
5. checkpoint auditing;
6. result collection under `PIPE_DIR`.

Use `WANDB=0` for local runs without logging.

# 4. Train R2-Dreamer policies

Dreamer training uses a frozen JEPA checkpoint. The recommended default is the best working model checkpoint.

The repository does not include source checkpoints, so all RL launchers require explicit checkpoint paths.

## Generic multimap training

```bash
python smac-dreamer/scripts/train_r2dreamer_smaclite_multimap.py   --config smac-dreamer/configs/r2_2100_jepa_tactical_mixture_v1_2.yaml   --jepa-checkpoint /absolute/path/to/best_working_model/checkpoint.pt   --run-dir /absolute/path/to/new_run
```

Inspect all supported arguments with:

```bash
python smac-dreamer/scripts/train_r2dreamer_smaclite_multimap.py --help
```

## Train the tactical source policy

```bash
SOURCE_CHECKPOINT=/absolute/path/to/non_tactical_source.pt JEPA_CHECKPOINT=/absolute/path/to/best_working_model/checkpoint.pt RUN_DIR=/absolute/path/to/tactical_source_run bash smac-dreamer/scripts/run_tactical_v1_2_2m.sh
```

The source checkpoint must be compatible with the retained R2-Dreamer architecture.

## Train the ordinary actor-critic comparison

```bash
SOURCE_CHECKPOINT=/absolute/path/to/tactical_source_checkpoint.pt JEPA_CHECKPOINT=/absolute/path/to/best_working_model/checkpoint.pt RUN_DIR=/absolute/path/to/actor_critic_run FINAL_STEP=800000 bash smac-dreamer/scripts/run_actor_critic_h15_800k.sh
```

This launcher performs a controlled non-hierarchical continuation from the tactical source checkpoint.

## Train the Option-Critic policy

```bash
SOURCE_CHECKPOINT=/absolute/path/to/tactical_source_checkpoint.pt JEPA_CHECKPOINT=/absolute/path/to/best_working_model/checkpoint.pt RUN_DIR=/absolute/path/to/option_critic_run FINAL_STEP=800000 bash smac-dreamer/scripts/run_option_critic_v9_anchor_safe_800k.sh
```

The launcher validates that the source checkpoint:

- is loadable;
- contains the expected agent state;
- represents the tactical source architecture;
- has not already been converted into an Option-Critic checkpoint.

The ordinary actor-critic and Option-Critic launchers intentionally enforce the same number of new environment steps.

# 5. Run the full forecast and RL comparison pipeline

The combined launcher runs:

1. experimental forecast training and evaluation;
2. ordinary actor-critic training;
3. Option-Critic training.

```bash
EXP40_CHECKPOINT=/absolute/path/to/best_working_model/checkpoint.pt TACTICAL_V12_CHECKPOINT=/absolute/path/to/tactical_source_checkpoint.pt JEPA_CHECKPOINT=/absolute/path/to/best_working_model/checkpoint.pt MANIFEST=/absolute/path/to/manifest.json PIPE_DIR=/absolute/path/to/combined_pipeline bash smac-dreamer/scripts/run_forecast_then_actor_critic_h15_then_option_critic_v9.sh
```

Use separate output directories for every run. Existing run directories are rejected to avoid accidental overwrite or resume ambiguity.

# 6. Evaluate a Dreamer checkpoint

```bash
python smac-dreamer/scripts/evaluate_multimap.py   --config smac-dreamer/configs/r2_2100_jepa_option_critic_8_v9_anchor_safe_h15_800k.yaml   --checkpoint /absolute/path/to/best_val_macro_winrate.pt   --split blind_iid   --output /absolute/path/to/eval_blind_iid.json
```

Common evaluation splits:

```text
validation
blind_iid
blind_compositional
```

Keep `run_meta.json` beside the checkpoint where possible. It records architecture, padding and lineage information used during evaluation.

Inspect evaluator arguments with:

```bash
python smac-dreamer/scripts/evaluate_multimap.py --help
```

# 7. Checkpoint and integration validation

Inspect a JEPA checkpoint:

```bash
python smac-dreamer/scripts/inspect_jepa_checkpoint.py   /absolute/path/to/best_working_model/checkpoint.pt
```

Run JEPA preflight checks:

```bash
python smac-dreamer/scripts/preflight_jepa_training.py   --config smac-dreamer/configs/r2_2100_jepa_tactical_mixture_v1_2.yaml   --jepa-checkpoint /absolute/path/to/best_working_model/checkpoint.pt
```

Validate JEPA–Dreamer integration:

```bash
python smac-dreamer/scripts/validate_jepa_r2_integration.py
```

Validate JEPA token parity:

```bash
python smac-dreamer/scripts/validate_jepa_token_parity.py
```

# 8. Repository validation

Run the retained workflow validator from the repository root:

```bash
PY="$(pwd)/.venv/bin/python" bash scripts/validate_retained_workflows.sh
```

This checks:

- dataloader source availability;
- dataset and visibility-mask contracts;
- JEPA trainer imports;
- forecast tests;
- shell syntax;
- Dreamer trainer imports;
- tactical policy tests;
- Option-Critic tests;
- checkpoint-free static audits;
- stale path and bundle references.

Run the project test suites separately:

```bash
uv run pytest smac-jepa-wm/tests -q
uv run pytest smac-dreamer/tests -q
```

Some environment-dependent tests may require SMACLite assets, CUDA or additional simulator setup.

# Runtime files

The following are generated during collection, training or evaluation and should not be committed:

```text
smac-jepa-wm/data/       collected trajectories
smac-jepa-wm/runs/       JEPA checkpoints and evaluations
smac-jepa-wm/reports/    dataset audit reports
smac-dreamer/logs/       Dreamer checkpoints and training logs
checkpoints/             manually saved model weights
replay/                  replay and memmap storage
wandb/                   local W&B state
outputs/                 combined pipeline outputs
```

The Python dataloader source under:

```text
smac-jepa-wm/smac_jepa/data/
```

is part of the repository and must remain tracked.

# Notes

- Use the **best working model** checkpoint for ordinary Dreamer experiments.
- Treat the **experimental forecast model** as a separate forecasting research path.
- Do not reuse output directories.
- Do not commit datasets, checkpoints, replay buffers or logs.
- Keep all map and training configs under `smac-dreamer/configs/`.
- Use absolute checkpoint paths for RL launchers.
- Run `scripts/validate_retained_workflows.sh` after modifying training code.

The commands above are the retained workflow reference for this combined repository.
