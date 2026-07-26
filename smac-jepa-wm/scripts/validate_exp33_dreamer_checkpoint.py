from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--jepa-root", type=Path, default=Path("."))
    parser.add_argument("--dreamer-root", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.jepa_root.resolve()))
    checkpoint = safe_load(args.checkpoint)
    if not isinstance(checkpoint, dict):
        raise SystemExit("Checkpoint is not a dictionary")

    for key in ("model_state", "memory_module_state", "metadata"):
        if key not in checkpoint:
            raise SystemExit(f"Checkpoint is missing {key!r}")

    cfg = dict(checkpoint.get("resolved_config", checkpoint.get("config", {})))
    metadata = dict(checkpoint["metadata"])
    state = checkpoint["memory_module_state"]

    anchored = bool(cfg.get("anchored_belief_memory", False)) or any(
        str(key).startswith("hidden_gate_net.") for key in state
    )
    checks = {
        "anchored": anchored,
        "memory_architecture": cfg.get("memory_architecture")
        == "anchored_ordered_action_latent_filter_v1",
        "action_conditioned_memory": bool(cfg.get("action_conditioned_memory", False)),
        "presence_rollout_mode_soft": cfg.get("presence_rollout_mode") == "soft",
        "target_mode_full": cfg.get("target_mode") == "full",
        "latent_normalized": bool(cfg.get("r2_latent_normalize", False)),
        "ema_target_encoder": bool(cfg.get("ema_target_encoder", False)),
        "rollout_horizon_5": int(cfg.get("rollout_horizon", -1)) == 5,
        "dreamer_contract": int(cfg.get("dreamer_integration_contract_version", -1)) == 1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("Checkpoint contract failed: " + ", ".join(failed))

    from smac_jepa.anchored_belief_memory import (
        AnchoredActionConditionedEntityRolloutGRUMemory,
    )

    latent_dim = int(cfg.get("latent_dim", 192))
    memory_dim = int(cfg.get("rollout_memory_dim", cfg.get("memory_dim", 322)))
    n_actions = int(metadata["n_actions"])
    max_agents = int(metadata["max_agents"])
    hidden_dim = cfg.get("rollout_memory_hidden_dim")
    residual = not bool(cfg.get("rollout_memory_no_residual", False))

    memory = AnchoredActionConditionedEntityRolloutGRUMemory(
        latent_dim=latent_dim,
        memory_dim=memory_dim,
        n_actions=n_actions,
        max_agents=max_agents,
        hidden_dim=hidden_dim,
        residual=residual,
    )
    memory.load_state_dict(state, strict=True)

    entities = int(metadata["max_agents"]) + int(metadata["max_enemies"])
    batch = 2
    mem = memory.initial_memory(batch, entities, device=torch.device("cpu"), dtype=torch.float32)
    z = torch.randn(batch, entities, latent_dim)
    z[:, max_agents + 1 :] = 0.0
    entity_mask = torch.ones(batch, entities)
    action = torch.zeros(batch, max_agents, dtype=torch.long)
    action_mask = torch.ones(batch, max_agents)
    conditioned = memory.condition(z, mem, entity_mask)
    updated = memory.update(
        z,
        mem,
        entity_mask,
        action=action,
        action_mask=action_mask,
    )
    assert conditioned.shape == (batch, entities, latent_dim)
    assert updated.shape == (batch, entities, memory_dim)
    assert torch.isfinite(conditioned).all() and torch.isfinite(updated).all()

    if args.dreamer_root is not None:
        sys.path.insert(0, str((args.dreamer_root / "src").resolve()))
        from smacdreamer.jepa.checkpoint import load_frozen_jepa_checkpoint

        _, loaded_memory, info = load_frozen_jepa_checkpoint(
            args.checkpoint,
            map_location="cpu",
            strict=True,
        )
        if type(loaded_memory).__name__ != type(memory).__name__:
            raise SystemExit(
                "Dreamer reconstructed the wrong memory class: "
                f"{type(loaded_memory).__name__}"
            )
        if info.memory_dim != memory_dim:
            raise SystemExit("Dreamer memory dimension mismatch")

    print("Exp33 Dreamer checkpoint contract: PASS")
    print(f"checkpoint={args.checkpoint}")
    print(f"latent_dim={latent_dim}")
    print(f"memory_dim={memory_dim}")
    print(f"recurrent_dim={memory.recurrent_dim}")
    print(f"entities={entities}")
    print("presence_rollout_mode=soft")


if __name__ == "__main__":
    main()
