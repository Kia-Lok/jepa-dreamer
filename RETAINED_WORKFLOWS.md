# Retained Workflows

No datasets or checkpoints are bundled. All paths below must refer to external
inputs and fresh output directories. Set `PY`, `PYTHON`, `ROOT`, `JEPA_ROOT`, or
`VENV` when the defaults do not match the active environment.

## Exp-40 JEPA

```bash
MANIFEST=/path/to/manifest.json \
OUT_DIR=/path/to/output/exp40 \
WANDB=0 \
bash smac-jepa-wm/scripts/run_exp40_dreamer_event_balanced.sh
```

## Exp-45 Forecast

```bash
EXP40_CHECKPOINT=/path/to/exp40/checkpoint.pt \
MANIFEST=/path/to/manifest.json \
PIPE_DIR=/path/to/output/exp45_pipeline \
WANDB=0 \
bash smac-dreamer/scripts/run_exp45_full_train_eval_resilient.sh
```

## Tactical-v1.2 Source

```bash
SOURCE_CHECKPOINT=/path/to/non_tactical_source.pt \
JEPA_CHECKPOINT=/path/to/exp40/checkpoint.pt \
RUN_DIR=/path/to/output/tactical_v1_2 \
bash smac-dreamer/scripts/run_tactical_v1_2_2m.sh
```

## Ordinary Actor-Critic H=15 / 800k

```bash
SOURCE_CHECKPOINT=/path/to/tactical_v1_2_checkpoint.pt \
JEPA_CHECKPOINT=/path/to/exp40/checkpoint.pt \
RUN_DIR=/path/to/output/actor_critic_h15_800k \
FINAL_STEP=800000 \
bash smac-dreamer/scripts/run_actor_critic_h15_800k.sh
```

## Option-Critic V9 Anchor-Safe, 8 Slots, H=15 / 800k

```bash
SOURCE_CHECKPOINT=/path/to/tactical_v1_2_checkpoint.pt \
JEPA_CHECKPOINT=/path/to/exp40/checkpoint.pt \
RUN_DIR=/path/to/output/option_critic_v9_h15_800k \
FINAL_STEP=800000 \
bash smac-dreamer/scripts/run_option_critic_v9_anchor_safe_800k.sh
```

`SOURCE_RUN_META` is optional lineage metadata.
`EXPECTED_SOURCE_CHECKPOINT_SHA256` optionally enforces exact reproduction.

## Combined Sequential Pipeline

```bash
EXP40_CHECKPOINT=/path/to/exp40/checkpoint.pt \
TACTICAL_V12_CHECKPOINT=/path/to/tactical_v1_2_checkpoint.pt \
JEPA_CHECKPOINT=/path/to/exp40/checkpoint.pt \
MANIFEST=/path/to/manifest.json \
PIPE_DIR=/path/to/output/forecast_ac_option_v9 \
bash smac-dreamer/scripts/run_forecast_then_actor_critic_h15_then_option_critic_v9.sh
```

## Generic Multimap

```bash
python smac-dreamer/scripts/train_r2dreamer_smaclite_multimap.py --help
python smac-dreamer/scripts/evaluate_multimap.py --help
```

Run checkpoint-free repository validation with:

```bash
PY=/path/to/python bash scripts/validate_retained_workflows.sh
```
