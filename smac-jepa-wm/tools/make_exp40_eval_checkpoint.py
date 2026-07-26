from __future__ import annotations

"""Create a strict Exp40-compatible view of an Exp45 Pow2 checkpoint.

The Exp45 checkpoint stores the original Exp40 model parameters plus an extra
``pow2_predictor`` module. Existing ordinary and hidden-belief evaluators build
plain ``SMACJEPA`` and therefore correctly reject those extra keys. This tool
removes only the new branch and restores the trusted recursive rollout horizon
in checkpoint metadata. The source checkpoint is never modified.
"""

import argparse
import copy
from pathlib import Path
from typing import Any

import torch


def safe_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--base-rollout-horizon",
        type=int,
        default=None,
        help="Override the saved pow2_base_rollout_horizon (normally 5).",
    )
    return parser.parse_args()


def sanitize(checkpoint: dict[str, Any], base_horizon: int | None) -> dict[str, Any]:
    out = copy.deepcopy(checkpoint)
    model_state = out.get("model_state")
    if not isinstance(model_state, dict):
        raise RuntimeError("Checkpoint does not contain model_state")

    pow2_keys = [str(key) for key in model_state if str(key).startswith("pow2_predictor.")]
    if not pow2_keys and not out.get("pow2_predictor_state"):
        raise RuntimeError("Checkpoint does not appear to contain an Exp45 Pow2 predictor")
    out["model_state"] = {
        key: value
        for key, value in model_state.items()
        if not str(key).startswith("pow2_predictor.")
    }
    out.pop("pow2_predictor_state", None)

    resolved = out.setdefault("resolved_config", {})
    fallback = out.get("config", {}) if isinstance(out.get("config"), dict) else {}
    saved_base_h = resolved.get(
        "pow2_base_rollout_horizon",
        fallback.get("pow2_base_rollout_horizon", 5),
    )
    base_h = int(saved_base_h if base_horizon is None else base_horizon)
    if base_h < 1:
        raise RuntimeError("base rollout horizon must be positive")

    update = {
        "rollout_horizon": base_h,
        "training_regime": "markov_rollout_rnn_seqmem_r2offline",
        "objective_family": "r2offline",
        "pow2_eval_sanitized": True,
        "pow2_direct_predictor": False,
        "dreamer_compatible": True,
        "dreamer_max_imagination_horizon": base_h,
        "sanitized_from_pow2_checkpoint": True,
    }
    resolved.update(update)
    if isinstance(out.get("config"), dict):
        out["config"].update(update)

    metadata = out.setdefault("metadata", {})
    metadata.update(
        {
            "pow2_eval_sanitized": True,
            "removed_pow2_model_keys": len(pow2_keys),
            "source_training_regime": "exp40_plus_pow2_direct_v1",
        }
    )
    return out


def main() -> None:
    args = parse_args()
    source = Path(args.checkpoint).expanduser().resolve()
    destination = Path(args.out).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Checkpoint not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = safe_load(source)
    sanitized = sanitize(checkpoint, args.base_rollout_horizon)
    torch.save(sanitized, destination)
    print(f"[OK] source={source}")
    print(f"[OK] output={destination}")
    print(f"[OK] model_keys={len(sanitized['model_state'])}")
    print(
        "[OK] rollout_horizon="
        f"{sanitized['resolved_config']['rollout_horizon']} dreamer_compatible=True"
    )


if __name__ == "__main__":
    main()
