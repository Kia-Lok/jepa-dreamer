from __future__ import annotations

"""Anchored-memory compatibility launcher for the R2-aware evaluator."""

import importlib.util
import os
from pathlib import Path
from typing import Any

import torch


def load_base():
    requested = os.environ.get(
        "R2_AWARE_BASE_EVAL", "eval_rnn_seqmem_dreamer_probe.py"
    )
    path = Path(requested).expanduser()
    if not path.is_file():
        path = Path.cwd() / path
    if not path.is_file():
        raise SystemExit(f"R2-aware base evaluator not found: {requested}")
    spec = importlib.util.spec_from_file_location("_r2aware_base", path.resolve())
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()


def build_memory_module(
    checkpoint: dict[str, Any],
    dataset: Any,
    device: torch.device,
) -> torch.nn.Module:
    cfg = base.get_config(checkpoint)
    memory_state = checkpoint.get("memory_module_state", {})
    anchored = bool(cfg.get("anchored_belief_memory", False)) or any(
        str(key).startswith("hidden_gate_net.") for key in memory_state
    )
    if not anchored:
        return base._pow2_original_build_memory_module(checkpoint, dataset, device)

    from smac_jepa.anchored_belief_memory import (
        AnchoredActionConditionedEntityRolloutGRUMemory,
    )

    metadata = checkpoint.get("metadata", {})
    module = AnchoredActionConditionedEntityRolloutGRUMemory(
        latent_dim=int(cfg["latent_dim"]),
        memory_dim=int(cfg["rollout_memory_dim"]),
        n_actions=int(
            cfg.get("n_actions", metadata.get("n_actions", dataset.metadata.n_actions))
        ),
        max_agents=int(
            cfg.get(
                "max_agents",
                metadata.get("max_agents", dataset.metadata.max_agents),
            )
        ),
        hidden_dim=cfg.get("rollout_memory_hidden_dim", None),
        residual=not bool(cfg.get("rollout_memory_no_residual", False)),
    ).to(device)
    if not memory_state:
        raise RuntimeError("Anchored checkpoint lacks memory_module_state")
    module.load_state_dict(memory_state, strict=True)
    module.eval()
    return module


def main() -> None:
    base._pow2_original_build_memory_module = base.build_memory_module
    base.build_memory_module = build_memory_module
    base.main()


if __name__ == "__main__":
    main()
