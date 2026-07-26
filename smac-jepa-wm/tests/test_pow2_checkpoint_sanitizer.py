from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "tools" / "make_exp40_eval_checkpoint.py"
spec = importlib.util.spec_from_file_location("checkpoint_sanitizer", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sanitizer_removes_only_pow2_branch() -> None:
    checkpoint = {
        "model_state": {
            "encoder.weight": torch.ones(2, 2),
            "pow2_predictor.head.weight": torch.zeros(2, 2),
        },
        "pow2_predictor_state": {"head.weight": torch.zeros(2, 2)},
        "resolved_config": {
            "rollout_horizon": 16,
            "pow2_base_rollout_horizon": 5,
            "pow2_direct_predictor": True,
        },
        "config": {"rollout_horizon": 16},
    }
    result = module.sanitize(checkpoint, None)
    assert set(result["model_state"]) == {"encoder.weight"}
    assert "pow2_predictor_state" not in result
    assert result["resolved_config"]["rollout_horizon"] == 5
    assert result["resolved_config"]["dreamer_compatible"] is True
    assert result["config"]["rollout_horizon"] == 5
    assert "pow2_predictor_state" in checkpoint
