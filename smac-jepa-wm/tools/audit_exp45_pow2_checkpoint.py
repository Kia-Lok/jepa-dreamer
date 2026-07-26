from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


def load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit_exp45_pow2_checkpoint.py CHECKPOINT")
    path = Path(sys.argv[1])
    if not path.is_file():
        raise SystemExit(f"missing checkpoint: {path}")
    checkpoint = load(path)
    cfg = checkpoint.get("resolved_config", checkpoint.get("config", {}))
    model_state = checkpoint.get("model_state", {})
    predictor = checkpoint.get("pow2_predictor_state", {})
    failures: list[str] = []

    expected = [1, 2, 4, 8, 16]
    if not bool(cfg.get("pow2_direct_predictor", False)):
        failures.append("pow2_direct_predictor flag is missing")
    if [int(x) for x in cfg.get("pow2_horizons", [])] != expected:
        failures.append(f"pow2_horizons != {expected}: {cfg.get('pow2_horizons')}")
    if int(cfg.get("pow2_base_rollout_horizon", -1)) != 5:
        failures.append("base recursive horizon is not 5")
    if int(cfg.get("pow2_dataset_rollout_horizon", -1)) != 16:
        failures.append("dataset/direct horizon is not 16")
    if not any(str(key).startswith("pow2_predictor.") for key in model_state):
        failures.append("model_state lacks pow2_predictor.* keys")
    if not predictor:
        failures.append("top-level pow2_predictor_state is missing")
    for horizon in expected:
        prefix = f"power_heads.{horizon}."
        if not any(str(key).startswith(prefix) for key in predictor):
            failures.append(f"missing trained head h={horizon}")
    if "memory_module_state" not in checkpoint:
        failures.append("memory_module_state missing")
    if "target_encoder_state" not in checkpoint:
        failures.append("target_encoder_state missing")
    if not bool(cfg.get("r2_latent_normalize", False)):
        failures.append("r2_latent_normalize is false")

    if failures:
        print("[FAIL] Exp45 checkpoint audit")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    parameter_count = sum(int(value.numel()) for value in predictor.values())
    print("[OK] Exp45 checkpoint audit passed")
    print(f"[OK] checkpoint={path}")
    print(f"[OK] power_horizons={expected}")
    print(f"[OK] predictor_parameters={parameter_count}")
    print(f"[OK] global_step={checkpoint.get('global_step', -1)}")


if __name__ == "__main__":
    main()
