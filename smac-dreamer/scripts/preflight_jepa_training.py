#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "external" / "r2dreamer", ROOT / "external" / "smaclite", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from omegaconf import OmegaConf

from smacdreamer.jepa.checkpoint import load_frozen_jepa_checkpoint, sha256_file, validate_metadata, _arch_from, _checkpoint_contract
from validate_jepa_r2_integration import run_integration_parity
from validate_jepa_token_parity import _episode_counts, _load_checkpoint_contract, run_token_parity


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _padding_from_config(config, episode_meta: dict, config_path: pathlib.Path) -> dict:
    padding = config.get("padding")
    if padding:
        return {
            "max_agents": int(padding["max_agents"]),
            "max_enemies": int(padding["max_enemies"]),
            "max_actions": int(padding["max_actions"]),
        }
    maps = config.get("maps") or {}
    train_folder = maps.get("train")
    if train_folder:
        folder = pathlib.Path(str(train_folder))
        if not folder.is_absolute():
            folder = (config_path.parent.parent / folder).resolve()
        if folder.exists():
            from smacdreamer.envs.map_discovery import discover_folders
            _, _, pad = discover_folders(
                str(folder),
                str(folder),
                padding_override=None,
                obs_mode=str((config.get("observation") or {}).get("mode", "structured")),
                isolate_probe=True,
                verbose=False,
            )
            return {
                "max_agents": int(pad.max_agents),
                "max_enemies": int(pad.max_enemies),
                "max_actions": int(pad.max_actions),
            }
        raise FileNotFoundError(
            f"cannot derive JEPA runtime padding: configured train map folder does not exist: {folder}. "
            "Provide a valid maps.train source or an explicit YAML padding block."
        )
    raise ValueError(
        "cannot derive JEPA runtime padding: config has no explicit padding block and no maps.train source. "
        "Provide one of them before running preflight."
    )


def derive_runtime_metadata(config_path: str | pathlib.Path, episode_npz: str | pathlib.Path, ckpt_cfg, vis):
    config_path = pathlib.Path(config_path)
    config = OmegaConf.load(config_path)
    episode_meta = _episode_counts(pathlib.Path(episode_npz))
    pad = _padding_from_config(config, episode_meta, config_path)
    dynamic = max(int(episode_meta["ally_state_feat_size"]), int(episode_meta["enemy_state_feat_size"]), 1)
    static_dim = int(episode_meta["static_dim"])
    entity_static_dim = int(episode_meta["entity_static_feat_size"])
    live = {
        "mode": "entity",
        "n_agents": int(episode_meta["n_agents"]),
        "n_enemies": int(episode_meta["n_enemies"]),
        "max_agents": int(pad["max_agents"]),
        "max_enemies": int(pad["max_enemies"]),
        "max_actions": int(pad["max_actions"]),
        "token_dim": int(dynamic + entity_static_dim),
        "dynamic_token_dim": int(dynamic),
        "static_dim": static_dim,
        "entity_static_feat_size": entity_static_dim,
        "ally_state_feat_size": int(episode_meta["ally_state_feat_size"]),
        "enemy_state_feat_size": int(episode_meta["enemy_state_feat_size"]),
        "ally_has_shields": bool(episode_meta.get("ally_has_shields", False)),
        "enemy_has_shields": bool(episode_meta.get("enemy_has_shields", False)),
        "num_unit_types": int(episode_meta.get("num_unit_types", 0)),
        "n_actions": int(pad["max_actions"]),
        "latent_dim": int(ckpt_cfg.get("latent_dim", 64)),
        "memory_dim": int(ckpt_cfg.get("rollout_memory_dim", ckpt_cfg.get("memory_dim", 128))),
        "action_conditioned_memory": bool(ckpt_cfg.get("action_conditioned_memory", False)),
        "latent_normalization": ckpt_cfg.get("latent_normalization", ckpt_cfg.get("latent_normalize", "none")),
    }
    live.update(vis.metadata())
    return live, config


def _resolve_horizon(config, ckpt_cfg, cli_horizon, allow_override: bool):
    config_horizon = int(config.get("imag_horizon"))
    checkpoint_horizon = ckpt_cfg.get("rollout_horizon")
    checkpoint_horizon = int(checkpoint_horizon) if checkpoint_horizon is not None else None
    if cli_horizon is None:
        validated = config_horizon
    else:
        validated = int(cli_horizon)
        if validated != config_horizon and not allow_override:
            raise ValueError(
                f"--rollout-horizon={validated} differs from config imag_horizon={config_horizon}; "
                "pass --allow-rollout-horizon-override to validate a different horizon"
            )
    if checkpoint_horizon is not None and config_horizon > checkpoint_horizon:
        raise ValueError(
            f"config imag_horizon={config_horizon} exceeds checkpoint rollout_horizon={checkpoint_horizon}"
        )
    return config_horizon, checkpoint_horizon, validated


def run_preflight(args) -> dict:
    checkpoint = pathlib.Path(args.checkpoint)
    episode = pathlib.Path(args.episode_npz)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not episode.exists():
        raise FileNotFoundError(f"episode-npz not found: {episode}")
    meta, cfg, vis = _load_checkpoint_contract(checkpoint)
    live, config = derive_runtime_metadata(args.config, episode, cfg, vis)
    arch = _arch_from(meta, cfg)
    contract = _checkpoint_contract(meta, cfg, arch)
    validate_metadata(contract, live)
    config_horizon, checkpoint_horizon, validated_horizon = _resolve_horizon(
        config,
        cfg,
        args.rollout_horizon,
        bool(args.allow_rollout_horizon_override),
    )
    core, memory, info = load_frozen_jepa_checkpoint(
        checkpoint,
        map_location=torch.device(args.device),
        live_metadata=live,
    )
    frozen_count = sum(p.numel() for p in list(core.parameters()) + list(memory.parameters()))
    assert frozen_count > 0
    assert all(not p.requires_grad for p in list(core.parameters()) + list(memory.parameters()))

    token = run_token_parity(checkpoint, episode, int(args.step))
    integration = run_integration_parity(
        checkpoint,
        episode,
        device=args.device,
        rollout_horizon=int(validated_horizon),
        step=int(args.step),
    )
    return {
        "result": "pass",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(pathlib.Path(args.config)),
        "episode_npz": str(episode),
        "device": args.device,
        "config_horizon": config_horizon,
        "checkpoint_horizon": checkpoint_horizon,
        "validated_horizon": validated_horizon,
        "checkpoint_metadata": meta,
        "runtime_metadata": live,
        "checkpoint_config": cfg,
        "visibility": vis.metadata(),
        "frozen_parameter_count": frozen_count,
        "token_parity": {
            "max_error": token.max_error,
            "comparisons": token.comparisons,
        },
        "integration_parity": {
            "max_error": integration.max_error,
            "comparisons": integration.comparisons,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run all JEPA R2-Dreamer preflight checks before smoke training.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episode-npz", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rollout-horizon", type=int, default=None)
    ap.add_argument("--allow-rollout-horizon-override", action="store_true")
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--report-json", required=True)
    args = ap.parse_args()
    report_path = pathlib.Path(args.report_json)
    try:
        report = run_preflight(args)
    except Exception as exc:
        report = {
            "result": "fail",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "checkpoint": args.checkpoint,
            "config": args.config,
            "episode_npz": args.episode_npz,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print("JEPA R2-DREAMER PREFLIGHT: FAIL", file=sys.stderr)
        raise SystemExit(1) from exc
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("JEPA R2-DREAMER PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
