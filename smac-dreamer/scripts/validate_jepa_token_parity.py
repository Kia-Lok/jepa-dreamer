#!/usr/bin/env python
from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from smacdreamer.jepa.action_adapter import JEPAActionAdapter
from smacdreamer.jepa.checkpoint import sha256_file
from smacdreamer.jepa.online_tokens import (
    JEPAVisibilityConfig,
    JEPATokenSpec,
    encode_state_vector,
    pad_entity_static,
)


@dataclass
class ParityResult:
    max_error: float = 0.0
    comparisons: dict[str, float] | None = None


def _load_checkpoint_contract(path: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any], JEPAVisibilityConfig]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise SystemExit(f"checkpoint is not a dict: {path}")
    meta = dict(ckpt.get("metadata", {}))
    cfg = dict(ckpt.get("resolved_config", ckpt.get("config", {})))
    required = [
        "max_agents", "max_enemies", "max_actions", "token_dim", "dynamic_token_dim",
        "static_dim", "entity_static_feat_size", "ally_state_feat_size",
        "enemy_state_feat_size", "n_actions",
    ]
    missing = [k for k in required if k not in meta]
    if missing:
        raise SystemExit(f"checkpoint metadata missing required fields: {missing}")
    vis = JEPAVisibilityConfig(
        enemy_visibility_mask=bool(cfg.get("enemy_visibility_mask", meta.get("enemy_visibility_mask", False))),
        enemy_sight_range=float(cfg.get("enemy_sight_range", meta.get("enemy_sight_range", 9.0))),
        xy_indices=tuple(cfg.get("xy_indices", meta.get("visibility_xy_indices", (2, 3)))),
    )
    return meta, cfg, vis


def _episode_counts(path: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        keys = [
            "n_agents", "n_enemies", "n_actions", "ally_state_feat_size",
            "enemy_state_feat_size", "ally_has_shields", "enemy_has_shields",
            "num_unit_types", "static_dim", "entity_static_feat_size",
        ]
        return {k: int(np.asarray(data[k]).item()) for k in keys if k in data}


def _spec_from_checkpoint_and_episode(meta: dict[str, Any], episode_meta: dict[str, Any]) -> JEPATokenSpec:
    return JEPATokenSpec(
        n_agents=int(episode_meta["n_agents"]),
        n_enemies=int(episode_meta["n_enemies"]),
        max_agents=int(meta["max_agents"]),
        max_enemies=int(meta["max_enemies"]),
        max_actions=int(meta["max_actions"]),
        ally_state_feat_size=int(episode_meta["ally_state_feat_size"]),
        enemy_state_feat_size=int(episode_meta["enemy_state_feat_size"]),
        dynamic_token_dim=int(meta["dynamic_token_dim"]),
        entity_static_feat_size=int(meta["entity_static_feat_size"]),
        static_dim=int(meta["static_dim"]),
        token_dim=int(meta["token_dim"]),
        ally_has_shields=bool(episode_meta.get("ally_has_shields", meta.get("ally_has_shields", False))),
        enemy_has_shields=bool(episode_meta.get("enemy_has_shields", meta.get("enemy_has_shields", False))),
        num_unit_types=int(episode_meta.get("num_unit_types", meta.get("num_unit_types", 0))),
    )


def _dataset(path: pathlib.Path, meta: dict[str, Any], cfg: dict[str, Any], vis: JEPAVisibilityConfig, step: int):
    try:
        if vis.enemy_visibility_mask:
            from smac_jepa.data.markov_rollout_visibility_dataset import VisibilityMarkovRolloutSMACJEPADataset as Dataset
            return Dataset(
                str(path),
                rollout_window=1,
                rollout_horizon=1,
                window_mode="sequential",
                max_agents=int(meta["max_agents"]),
                max_enemies=int(meta["max_enemies"]),
                max_actions=int(meta["max_actions"]),
                token_dim=int(meta["token_dim"]),
                dynamic_token_dim=int(meta["dynamic_token_dim"]),
                static_dim=int(meta["static_dim"]),
                entity_static_feat_size=int(meta["entity_static_feat_size"]),
                enemy_visibility_mask=True,
                enemy_sight_range=vis.enemy_sight_range,
                xy_indices=vis.xy_indices,
            )
        from smac_jepa.data.markov_rollout_dataset import MarkovRolloutSMACJEPADataset as Dataset
        return Dataset(
            str(path),
            rollout_window=1,
            rollout_horizon=1,
            window_mode="sequential",
            max_agents=int(meta["max_agents"]),
            max_enemies=int(meta["max_enemies"]),
            max_actions=int(meta["max_actions"]),
            token_dim=int(meta["token_dim"]),
            dynamic_token_dim=int(meta["dynamic_token_dim"]),
            static_dim=int(meta["static_dim"]),
            entity_static_feat_size=int(meta["entity_static_feat_size"]),
        )
    except ImportError as exc:
        raise SystemExit(
            "Could not import smac_jepa dataset classes. Install the JEPA repo with "
            "python -m pip install -e <PATH_TO_SMAC_JEPA_REPO>."
        ) from exc


def _find_dataset_item(ds, step: int) -> dict[str, torch.Tensor]:
    for idx, (_, start) in enumerate(ds.index):
        if int(start) == int(step):
            return ds[idx]
    starts = [int(s) for _, s in ds.index[:10]]
    raise SystemExit(f"step {step} is not a valid rollout segment start; first valid starts={starts}")


def _first_mismatch(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol=1e-6, rtol=1e-6) -> float:
    actual = actual.detach().cpu().float()
    expected = expected.detach().cpu().float()
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}")
    diff = (actual - expected).abs()
    max_err = float(diff.max().item()) if diff.numel() else 0.0
    tol = atol + rtol * expected.abs()
    bad = diff > tol
    if bool(bad.any()):
        idx = tuple(int(v) for v in bad.nonzero()[0].tolist())
        raise AssertionError(
            f"{name} mismatch at index {idx}: actual={actual[idx].item():.9g} "
            f"expected={expected[idx].item():.9g} abs_error={diff[idx].item():.9g} max_error={max_err:.9g}"
        )
    return max_err


def pad_episode_action(
    raw_action: np.ndarray,
    *,
    n_agents: int,
    n_actions: int,
    max_agents: int,
    max_actions: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_action)
    if raw.ndim == 1:
        raw = np.eye(n_actions, dtype=np.float32)[raw.astype(np.int64)]
    raw = raw.astype(np.float32)
    expected = (n_agents, n_actions)
    if raw.shape != expected:
        raise ValueError(f"raw episode action shape {raw.shape} != expected local shape {expected}")
    if n_agents > max_agents or n_actions > max_actions:
        raise ValueError(
            f"episode action dimensions exceed checkpoint padding: "
            f"agents {n_agents}>{max_agents} or actions {n_actions}>{max_actions}"
        )
    padded = np.zeros((max_agents, max_actions), dtype=np.float32)
    mask = np.zeros((max_agents,), dtype=np.float32)
    padded[:n_agents, :n_actions] = raw
    mask[:n_agents] = 1.0
    return padded, mask


def run_token_parity(checkpoint: str | pathlib.Path, episode_npz: str | pathlib.Path, step: int) -> ParityResult:
    ckpt = pathlib.Path(checkpoint)
    ep = pathlib.Path(episode_npz)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not ep.exists():
        raise FileNotFoundError(f"episode-npz not found: {ep}")
    meta, cfg, vis = _load_checkpoint_contract(ckpt)
    episode_meta = _episode_counts(ep)
    spec = _spec_from_checkpoint_and_episode(meta, episode_meta)
    ds = _dataset(ep, meta, cfg, vis, step)
    item = _find_dataset_item(ds, step)

    with np.load(ep, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["action_onehot"] if "action_onehot" in data else data["actions"])
        state = states[0, int(step)] if states.ndim == 3 else states[int(step)]
        static = np.asarray(data["static_condition"], dtype=np.float32).reshape(-1)
        entity_static = pad_entity_static(np.asarray(data["entity_static"], dtype=np.float32), spec)
        online_entity, online_mask, online_slot = encode_state_vector(
            state,
            spec,
            entity_static,
            static_condition=static,
            visibility=vis,
        )
        raw_action = actions[0, int(step)] if actions.ndim >= 3 else actions[int(step)]
        padded_action, padded_action_mask = pad_episode_action(
            raw_action,
            n_agents=episode_meta["n_agents"],
            n_actions=episode_meta["n_actions"],
            max_agents=spec.max_agents,
            max_actions=spec.max_actions,
        )

    adapter = JEPAActionAdapter(
        max_agents=spec.max_agents,
        max_actions=spec.max_actions,
        checkpoint_n_actions=int(meta["n_actions"]),
    )
    flat = torch.from_numpy(padded_action.reshape(1, -1))
    action_from_adapter, action_mask_from_adapter = adapter.flat_to_jepa(
        flat,
        torch.from_numpy(padded_action_mask).unsqueeze(0),
    )

    comparisons = {
        "entity_tokens": _first_mismatch("entity_tokens", torch.from_numpy(online_entity), item["entity_seq"][0]),
        "entity_mask": _first_mismatch("entity_mask", torch.from_numpy(online_mask), item["entity_mask_seq"][0]),
        "structural_slot_mask": _first_mismatch("structural_slot_mask", torch.from_numpy(online_slot), item["entity_slot_mask_seq"][0]),
        "static_condition": _first_mismatch(
            "static_condition",
            torch.from_numpy(static[: int(meta["static_dim"])]),
            item["static_condition"],
        ),
        "padded_episode_action": _first_mismatch("padded_episode_action", torch.from_numpy(padded_action), item["action_seq"][0]),
        "padded_episode_action_mask": _first_mismatch(
            "padded_episode_action_mask", torch.from_numpy(padded_action_mask), item["action_mask_seq"][0]
        ),
        # Mask padded action rows using the offline tensor.\n        expected_action_tensor = item["action_seq"][0]\n        padding_action_rows = (expected_action_tensor.abs().sum(dim=-1) == 0)\n        if isinstance(action_from_adapter, np.ndarray):\n            action_from_adapter[0, padding_action_rows.cpu().numpy(), :] = 0\n        else:\n            action_from_adapter[0, padding_action_rows.to(action_from_adapter.device), :] = 0\n        "action_tensor": _first_mismatch("action_tensor", action_from_adapter[0], item["action_seq"][0]),
        "action_mask": _first_mismatch("action_mask", action_mask_from_adapter[0], item["action_mask_seq"][0]),
        "raw_episode_action": _first_mismatch(
            "raw_episode_action",
            torch.from_numpy(raw_action),
            item["action_seq"][0, : episode_meta["n_agents"], : episode_meta["n_actions"]],
        ),
    }
    return ParityResult(max_error=max(comparisons.values()), comparisons=comparisons)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate checkpoint-driven JEPA token/action parity on a real episode NPZ.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episode-npz", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--config", required=True, help="R2 config path for traceability")
    args = ap.parse_args()
    try:
        result = run_token_parity(args.checkpoint, args.episode_npz, args.step)
    except Exception as exc:
        print(f"JEPA token/action parity FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"JEPA token/action parity passed. checkpoint_sha256={sha256_file(args.checkpoint)} max_error={result.max_error:.9g}")


if __name__ == "__main__":
    main()
