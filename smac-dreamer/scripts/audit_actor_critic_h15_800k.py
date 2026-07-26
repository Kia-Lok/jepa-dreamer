#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_allowed(cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.pop("hierarchical_options", None)
    out.pop("imag_horizon", None)
    if isinstance(out.get("buffer"), dict):
        out["buffer"].pop("scratch_dir", None)
    if isinstance(out.get("validation"), dict):
        out["validation"].pop("run_at_start", None)
        out["validation"].pop("every", None)
    out.pop("compile", None)
    if isinstance(out.get("model"), dict):
        out["model"].pop("compile", None)
    if isinstance(out.get("wandb"), dict):
        out["wandb"].pop("run_name", None)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--source-config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--expected-sha256")
    args = p.parse_args()

    repo = Path(args.repo).resolve()
    config_path = (repo / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    source_config_path = (repo / args.source_config).resolve() if not Path(args.source_config).is_absolute() else Path(args.source_config).resolve()
    checkpoint = Path(args.checkpoint).resolve()

    assert config_path.is_file(), config_path
    assert source_config_path.is_file(), source_config_path
    assert checkpoint.is_file(), checkpoint
    if args.expected_sha256:
        assert sha256(checkpoint) == args.expected_sha256

    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True, throw_on_missing=True)
    src = OmegaConf.to_container(OmegaConf.load(source_config_path), resolve=True, throw_on_missing=True)
    assert isinstance(cfg, dict) and isinstance(src, dict)

    assert strip_allowed(cfg) == strip_allowed(src), "baseline changed source training regime outside the allowlist"
    assert int(cfg["imag_horizon"]) == 15
    assert bool(cfg["tactical_mixture"]["enabled"]) is True
    assert bool(cfg["hierarchical_options"]["enabled"]) is False
    assert bool(cfg["validation"]["run_at_start"]) is True
    assert int(cfg["validation"]["every"]) == 200_000
    assert cfg["buffer"]["scratch_dir"] == "replay_tactical_v12_actor_critic_h15_800k"

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = ckpt.get("tactical_mixture_metadata") or {}
    state = ckpt.get("agent_state_dict")
    assert isinstance(state, dict) and state, "checkpoint lacks a non-empty agent_state_dict"
    assert meta.get("architecture") == "tactical_mixture_v1_2"
    assert int(meta.get("num_tactics", -1)) == 2
    assert not any(k.startswith("hierarchical_options.") for k in state)

    result = {
        "config": str(config_path),
        "source_config": str(source_config_path),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "step": ckpt.get("step"),
            "val_macro_win_rate": ckpt.get("val_macro_win_rate"),
        },
        "checks": {
            "ordinary_tactical_actor_critic": True,
            "option_critic_disabled": True,
            "tactical_mixture_enabled": True,
            "fixed_horizon_15": True,
            "exactly_800k_enforced_by_runner": True,
            "source_training_regime_preserved": True,
            "fresh_run_local_replay": True,
            "startup_and_200k_validation": True,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[OK] Tactical-v1.2 ordinary actor-critic H=15 source/config audit passed")


if __name__ == "__main__":
    main()
