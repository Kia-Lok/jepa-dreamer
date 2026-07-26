#!/usr/bin/env python
from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "external" / "r2dreamer", ROOT / "external" / "smaclite"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

from smacdreamer.jepa.checkpoint import load_frozen_jepa_checkpoint, sha256_file
from smacdreamer.jepa.state import pack_state, unpack_state
from smacdreamer.jepa.world_model import FrozenJEPAWorldModel
from validate_jepa_token_parity import _dataset, _find_dataset_item, _load_checkpoint_contract, _first_mismatch


@dataclass
class IntegrationResult:
    max_error: float
    comparisons: dict[str, float]


def _flat(action_seq: torch.Tensor) -> torch.Tensor:
    return action_seq.reshape(action_seq.shape[0], -1)


def previous_actions_for_states(action_seq: torch.Tensor) -> torch.Tensor:
    """Return [zero, a0, a1, ...] for states [s0, s1, s2, ...]."""
    if action_seq.ndim != 3:
        raise ValueError(f"action_seq must have [ACTIONS,A,C], got {tuple(action_seq.shape)}")
    flat = _flat(action_seq)
    zero = torch.zeros(1, flat.shape[-1], dtype=flat.dtype, device=flat.device)
    return torch.cat([zero, flat], dim=0)


def previous_action_masks_for_states(action_mask_seq: torch.Tensor) -> torch.Tensor:
    if action_mask_seq.ndim != 2:
        raise ValueError(f"action_mask_seq must have [ACTIONS,A], got {tuple(action_mask_seq.shape)}")
    zero = torch.zeros(1, action_mask_seq.shape[-1], dtype=action_mask_seq.dtype, device=action_mask_seq.device)
    return torch.cat([zero, action_mask_seq], dim=0)


def rollout_actions_from_start(
    transition_actions: torch.Tensor,
    start_idx: int,
    rollout_horizon: int,
) -> torch.Tensor:
    if transition_actions.ndim != 3:
        raise ValueError(f"transition_actions must have [B,ACTIONS,F], got {tuple(transition_actions.shape)}")
    if start_idx < 0 or start_idx >= transition_actions.shape[1]:
        raise ValueError(f"start_idx={start_idx} outside transition action range 0..{transition_actions.shape[1] - 1}")
    length = min(int(rollout_horizon), int(transition_actions.shape[1] - start_idx))
    if length <= 0:
        raise ValueError(f"no rollout actions available from start_idx={start_idx}")
    return transition_actions[:, start_idx : start_idx + length]


def _reference_obs_rollout(wm: FrozenJEPAWorldModel, core, memory_module, item, device: torch.device):
    entity = item["entity_seq"].unsqueeze(0).to(device)
    mask = item["entity_mask_seq"].unsqueeze(0).to(device)
    slot = item["entity_slot_mask_seq"].unsqueeze(0).to(device)
    static = item["static_condition"].unsqueeze(0).to(device)
    actions = previous_actions_for_states(item["action_seq"].to(device))
    action_masks = previous_action_masks_for_states(item["action_mask_seq"].to(device))
    if entity.shape[1] != item["action_seq"].shape[0] + 1:
        raise ValueError(
            f"observed rollout expects states=actions+1, got states={entity.shape[1]} "
            f"actions={item['action_seq'].shape[0]}"
        )
    if actions.shape[0] != entity.shape[1]:
        raise ValueError(f"previous action length {actions.shape[0]} != state length {entity.shape[1]}")
    with torch.no_grad():
        z_seq = core.encoder(entity, mask)
        b, t, e, _ = z_seq.shape
        mem = memory_module.initial_memory(b, e, device=device, dtype=z_seq.dtype)
        prev_z = torch.zeros_like(z_seq[:, 0])
        prev_mask = torch.zeros_like(mask[:, 0])
        zs, deters = [], []
        for idx in range(t):
            if getattr(memory_module, "uses_action", False):
                mem = memory_module.update(
                    prev_z,
                    mem,
                    prev_mask,
                    action=actions[idx].reshape(1, wm.max_agents, wm.max_actions),
                    action_mask=action_masks[idx].unsqueeze(0),
                )
            else:
                mem = memory_module.update(prev_z, mem, prev_mask)
            cur_z = z_seq[:, idx]
            cur_mask = mask[:, idx]
            cur_slot = slot[:, idx]
            zs.append(cur_z)
            deters.append(pack_state(mem, cur_mask, cur_slot, static))
            prev_z, prev_mask = cur_z, cur_mask
    return z_seq, torch.stack(zs, 1), torch.stack(deters, 1)


def _reference_img_step(wm: FrozenJEPAWorldModel, core, memory_module, z, deter, action_flat):
    memory, entity_mask, slot_mask, static = unpack_state(deter, wm.state_spec)
    action_jepa, action_mask = wm.action_adapter.flat_to_jepa(action_flat, slot_mask[:, : wm.max_agents])
    with torch.no_grad():
        conditioned = memory_module.condition(z, memory, entity_mask)
        pred = core.predictor(
            conditioned.unsqueeze(1),
            action_jepa.unsqueeze(1),
            action_mask.unsqueeze(1),
            torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype),
            entity_mask.unsqueeze(1),
            static,
        )[:, 0]
        logits = core.predict_presence(pred)
        next_mask = (torch.sigmoid(logits) >= wm.presence_threshold).to(dtype=z.dtype) * slot_mask
        pred = pred * next_mask.unsqueeze(-1)
        if getattr(memory_module, "uses_action", False):
            next_memory = memory_module.update(pred, memory, next_mask, action=action_jepa, action_mask=action_mask)
        else:
            next_memory = memory_module.update(pred, memory, next_mask)
    return pred, pack_state(next_memory, next_mask, slot_mask, static), conditioned, logits, next_mask


def run_integration_parity(
    checkpoint: str | pathlib.Path,
    episode_npz: str | pathlib.Path,
    *,
    device: str = "cpu",
    rollout_horizon: int = 5,
    step: int = 0,
) -> IntegrationResult:
    ckpt = pathlib.Path(checkpoint)
    ep = pathlib.Path(episode_npz)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not ep.exists():
        raise FileNotFoundError(f"episode-npz not found: {ep}")
    meta, cfg, vis = _load_checkpoint_contract(ckpt)
    ds = _dataset(ep, meta, cfg, vis, step)
    item = _find_dataset_item(ds, step)
    device_t = torch.device(device)
    core, memory, info = load_frozen_jepa_checkpoint(ckpt, map_location=device_t, live_metadata=dict(info_meta := meta) | {
        "latent_dim": int(cfg.get("latent_dim", info_meta.get("latent_dim", 64))),
        "memory_dim": int(cfg.get("rollout_memory_dim", cfg.get("memory_dim", info_meta.get("memory_dim", 128)))),
        "action_conditioned_memory": bool(cfg.get("action_conditioned_memory", False)),
        "enemy_visibility_mask": bool(vis.enemy_visibility_mask),
        "enemy_sight_range": float(vis.enemy_sight_range),
        "visibility_xy_indices": tuple(vis.xy_indices),
        "latent_normalization": cfg.get("latent_normalization", cfg.get("latent_normalize", "none")),
    })
    wm = FrozenJEPAWorldModel(core=core, memory_module=memory, info=info, feature_dim=32).to(device_t)

    entity = item["entity_seq"].unsqueeze(0).to(device_t)
    mask = item["entity_mask_seq"].unsqueeze(0).to(device_t)
    slot = item["entity_slot_mask_seq"].unsqueeze(0).to(device_t)
    static = item["static_condition"].unsqueeze(0).to(device_t)
    action_seq = item["action_seq"].to(device_t)
    transition_actions = _flat(action_seq).unsqueeze(0)
    previous_actions = previous_actions_for_states(action_seq).unsqueeze(0)
    if entity.shape[1] != transition_actions.shape[1] + 1:
        raise ValueError(
            f"JEPA integration parity requires state count == action count + 1, "
            f"got states={entity.shape[1]} actions={transition_actions.shape[1]}"
        )
    obs = {
        "jepa_entity": entity,
        "jepa_entity_mask": mask,
        "jepa_entity_slot_mask": slot,
        "jepa_static_condition": static.expand(1, entity.shape[1], -1),
    }
    encoded = wm.encode_obs(obs)
    ref_direct, ref_z_seq, ref_deter_seq = _reference_obs_rollout(wm, core, memory, item, device_t)
    z0, d0 = wm.initial(1, device=device_t)
    resets = torch.zeros(1, previous_actions.shape[1], dtype=torch.bool, device=device_t)
    wrapper_z, wrapper_deter = wm.observe(encoded, previous_actions, (z0, d0), resets)

    comparisons = {
        "encoder_latents": _first_mismatch("encoder_latents", encoded["z"], ref_direct),
        "observe_latents": _first_mismatch("observe_latents", wrapper_z, ref_z_seq),
        "observe_deter": _first_mismatch("observe_deter", wrapper_deter, ref_deter_seq),
    }

    action_jepa, action_mask = wm.action_adapter.flat_to_jepa(transition_actions[:, 0], slot[:, 0, : wm.max_agents])
    # Padded agents have all-zero action vectors in the JEPA dataset.
    # argmax/one-hot conversion otherwise turns them into action 0.
    expected_action_tensor = item["action_seq"][0]
    padding_action_rows = expected_action_tensor.abs().sum(dim=-1) == 0

    if hasattr(action_jepa, "clone"):
        action_jepa_for_comparison = action_jepa.clone()
        action_jepa_for_comparison[
            0,
            padding_action_rows.to(action_jepa_for_comparison.device),
            :,
        ] = 0
    else:
        action_jepa_for_comparison = action_jepa.copy()
        action_jepa_for_comparison[
            0,
            padding_action_rows.cpu().numpy(),
            :,
        ] = 0

    comparisons["action_tensor"] = _first_mismatch(
        "action_tensor",
        action_jepa_for_comparison[0],
        expected_action_tensor,
    )
    comparisons["action_mask"] = _first_mismatch("action_mask", action_mask[0], item["action_mask_seq"][0])

    start_idx = min(1, wrapper_z.shape[1] - 1)
    z, deter = wrapper_z[:, start_idx], wrapper_deter[:, start_idx]
    ref_z, ref_deter = z.clone(), deter.clone()
    rollout_actions = rollout_actions_from_start(transition_actions, start_idx, int(rollout_horizon))
    max_steps = int(rollout_actions.shape[1])
    wrapper_zs, wrapper_ds = [], []
    ref_zs, ref_ds = [], []
    for idx in range(max_steps):
        action = rollout_actions[:, idx]
        wz, wd = wm.img_step(z, deter, action)
        rz, rd, conditioned, logits, next_mask = _reference_img_step(wm, core, memory, ref_z, ref_deter, action)
        wrapper_memory, wrapper_entity_mask, _, _ = unpack_state(deter, wm.state_spec)
        wrapper_conditioned = memory.condition(z, wrapper_memory, wrapper_entity_mask)
        comparisons[f"conditioned_step_{idx}"] = _first_mismatch(f"conditioned_step_{idx}", wrapper_conditioned, conditioned)
        comparisons[f"prediction_step_{idx}"] = _first_mismatch(f"prediction_step_{idx}", wz, rz)
        # Raw pre-mask presence logits are not exposed by wm.img_step().
        # Prediction parity and resulting presence-mask parity are checked below.
        comparisons[f"presence_mask_step_{idx}"] = _first_mismatch(
            f"presence_mask_step_{idx}",
            unpack_state(wd, wm.state_spec)[1],
            next_mask,
        )
        comparisons[f"imagined_deter_step_{idx}"] = _first_mismatch(f"imagined_deter_step_{idx}", wd, rd)
        wrapper_zs.append(wz)
        wrapper_ds.append(wd)
        ref_zs.append(rz)
        ref_ds.append(rd)
        z, deter = wz, wd
        ref_z, ref_deter = rz, rd

    seq_z, seq_d = wm.imagine_with_action(wrapper_z[:, start_idx], wrapper_deter[:, start_idx], rollout_actions)
    comparisons["imagine_with_action_latents"] = _first_mismatch(
        "imagine_with_action_latents", seq_z, torch.stack(wrapper_zs, 1)
    )
    comparisons["imagine_with_action_deter"] = _first_mismatch(
        "imagine_with_action_deter", seq_d, torch.stack(wrapper_ds, 1)
    )
    repeated_z, repeated_d = wm.observe(encoded, previous_actions, (z0, d0), resets)
    comparisons["observe_repeated_latents"] = _first_mismatch("observe_repeated_latents", repeated_z, wrapper_z)
    comparisons["observe_repeated_deter"] = _first_mismatch("observe_repeated_deter", repeated_d, wrapper_deter)
    return IntegrationResult(max_error=max(comparisons.values()), comparisons=comparisons)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate R2 JEPA wrapper against an independent JEPA reference rollout.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episode-npz", required=True)
    ap.add_argument("--config", required=True, help="R2 config path for traceability")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rollout-horizon", type=int, default=5)
    ap.add_argument("--step", type=int, default=0)
    args = ap.parse_args()
    try:
        result = run_integration_parity(
            args.checkpoint,
            args.episode_npz,
            device=args.device,
            rollout_horizon=args.rollout_horizon,
            step=args.step,
        )
    except Exception as exc:
        print(f"JEPA R2 wrapper parity FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"JEPA R2 wrapper parity passed. checkpoint_sha256={sha256_file(args.checkpoint)} max_error={result.max_error:.9g}")


if __name__ == "__main__":
    main()
