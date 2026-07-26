#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
: "${CHECKPOINT:=${1:-}}"
: "${CHECKPOINT:?Set CHECKPOINT to an external Exp-45 checkpoint}"
CHECKPOINT="$CHECKPOINT" bash scripts/eval_exp45_pow2_ordinary.sh
CHECKPOINT="$CHECKPOINT" bash scripts/eval_exp45_pow2_hidden.sh
echo "[OK] all Exp45 evaluations completed"
