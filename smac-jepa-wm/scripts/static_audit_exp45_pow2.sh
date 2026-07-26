#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON:-${PY:-python}}"

"$PYTHON_BIN" -m py_compile \
  smac_jepa/pow2_direct_predictor.py \
  smac_jepa/train_jepa_exp45_pow2_direct.py \
  tools/make_exp40_eval_checkpoint.py \
  tools/audit_exp45_pow2_checkpoint.py \
  tools/eval_pow2_direct.py \
  tools/eval_rnn_seqmem_dreamer_probe_r2aware_anchored.py

bash -n scripts/*.sh

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
trainer = Path('smac_jepa/train_jepa_exp45_pow2_direct.py').read_text()
predictor = Path('smac_jepa/pow2_direct_predictor.py').read_text()
required_trainer = [
    '_exp40._patch_for_exp33_dreamer()',
    'base_kwargs["rollout_horizon"] = int(_CFG["base_horizon"])',
    'args.rollout_horizon = int(_CFG["max_horizon"])',
    'target_encoder(target_entity_seq, target_entity_mask_seq)',
    'PowerOfTwoDirectPredictor',
    'pow2_predictor_state',
    'pow2_dataset_rollout_horizon',
]
required_predictor = [
    'self.decoder = nn.LSTMCell',
    'self.power_heads = nn.ModuleDict',
    'def predict_binary(',
    'action_seq[:, offset : offset + block]',
]
missing = [item for item in required_trainer if item not in trainer]
missing += [item for item in required_predictor if item not in predictor]
if missing:
    raise SystemExit(f'[FAIL] missing source contracts: {missing}')
if 'pred_latent.detach()' in trainer:
    raise SystemExit('[FAIL] forbidden predictor detach found')
print('[OK] source contract audit passed')
PY

PYTHONPATH=. "$PYTHON_BIN" -m pytest -q tests/test_pow2_direct_predictor.py tests/test_pow2_checkpoint_sanitizer.py
echo "[OK] Exp45 Pow2 static/unit audit passed"
