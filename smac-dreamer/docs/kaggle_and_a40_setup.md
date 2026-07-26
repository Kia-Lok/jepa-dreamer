# R2-Dreamer SMAClite Long Runs

## A40

Use BF16 config:

```bash
python scripts/train_r2dreamer_smaclite_multimap.py \
  --config configs/r2_650.yaml
```

`configs/r2_650.yaml` requests `amp_dtype: bfloat16`. The script fails clearly if
the selected CUDA device does not support BF16.

## Kaggle P100/T4

Upload or mount the repository as a Kaggle input, then run:

```bash
bash scripts/setup_kaggle.sh /kaggle/input/<dataset-name>/smac-dreamer /kaggle/working/smac-dreamer
cd /kaggle/working/smac-dreamer
python scripts/train_r2dreamer_smaclite_multimap.py \
  --config configs/r2_650_kaggle.yaml
```

`configs/r2_650_kaggle.yaml` uses `amp_dtype: float32`, which disables autocast and
GradScaler.

## W&B

Do not store API keys in the repository. On Kaggle, use notebook secrets and set:

```python
from kaggle_secrets import UserSecretsClient
import os

secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = secrets.get_secret("WANDB_PROJECT")
os.environ["WANDB_ENTITY"] = secrets.get_secret("WANDB_ENTITY")
os.environ["WANDB_MODE"] = "online"
```

You can also pass non-secret overrides:

```bash
python scripts/train_r2dreamer_smaclite_multimap.py \
  --config configs/r2_650_kaggle.yaml \
  --wandb-project YOUR_PROJECT \
  --wandb-entity YOUR_ENTITY
```

## Writable paths

The Kaggle config writes logs/checkpoints and replay memmaps under `/kaggle/working`.
Do not put replay memmaps under `/kaggle/input`.
