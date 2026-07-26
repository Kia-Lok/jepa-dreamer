from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .memory import ActionConditionedEntityRolloutGRUMemory, EntityRolloutGRUMemory as CompatEntityRolloutGRUMemory


class JEPADependencyError(ImportError):
    pass


class JEPACompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class JEPACheckpointInfo:
    path: str
    sha256: str
    metadata: dict[str, Any]
    resolved_config: dict[str, Any]
    training_regime: str | None
    action_conditioned_memory: bool
    latent_dim: int
    memory_dim: int
    n_actions: int


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_jepa():
    try:
        from smac_jepa.jepa import SMACJEPA
        try:
            from smac_jepa.modules.rollout_memory import EntityRolloutGRUMemory
        except ImportError:
            EntityRolloutGRUMemory = CompatEntityRolloutGRUMemory
        from smac_jepa.anchored_belief_memory import (
            AnchoredActionConditionedEntityRolloutGRUMemory,
        )
        return (
            SMACJEPA,
            EntityRolloutGRUMemory,
            AnchoredActionConditionedEntityRolloutGRUMemory,
        )
    except ImportError as exc:
        raise JEPADependencyError(
            "world_model.backend='jepa' requires the external smac-jepa-wm package. "
            "Install it with: python -m pip install -e \"<PATH_TO_SMAC_JEPA_REPO>\""
        ) from exc


def _required(checkpoint: dict[str, Any], key: str):
    if key not in checkpoint:
        raise KeyError(f"JEPA checkpoint missing required key {key!r}")
    return checkpoint[key]


def _cfg_get(cfg: dict[str, Any], *names: str, default=None):
    for name in names:
        if name in cfg and cfg[name] is not None:
            return cfg[name]
    return default


def _arch_from(metadata: dict[str, Any], cfg: dict[str, Any]) -> dict[str, int | float | str]:
    latent_dim = int(_cfg_get(cfg, "latent_dim", default=64))
    hidden_dim = int(_cfg_get(cfg, "hidden_dim", default=128))
    action_dim = int(_cfg_get(cfg, "action_dim", default=64))
    num_heads = int(_cfg_get(cfg, "num_heads", default=2))
    encoder_layers = int(_cfg_get(cfg, "encoder_layers", default=1))
    action_layers = int(_cfg_get(cfg, "action_layers", default=1))
    predictor_layers = int(_cfg_get(cfg, "predictor_layers", default=1))
    max_context_len = int(_cfg_get(cfg, "max_context_len", "context_len", default=32))
    return {
        "state_dim": int(metadata["state_dim"]),
        "n_agents": int(metadata["n_agents"]),
        "n_actions": int(metadata["n_actions"]),
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "action_dim": action_dim,
        "num_heads": num_heads,
        "mode": str(metadata.get("mode", "entity")),
        "max_agents": int(metadata["max_agents"]),
        "max_enemies": int(metadata["max_enemies"]),
        "max_actions": int(metadata["max_actions"]),
        "token_dim": int(metadata["token_dim"]),
        "decoder_weight": float(_cfg_get(cfg, "decoder_weight", default=1.0)),
        "encoder_layers": encoder_layers,
        "action_layers": action_layers,
        "predictor_layers": predictor_layers,
        "max_context_len": max_context_len,
        "static_dim": int(metadata.get("static_dim", 0)),
    }


VALIDATION_FIELDS = (
        "mode",
        "n_agents",
        "n_enemies",
        "max_agents",
        "max_enemies",
        "max_actions",
        "token_dim",
        "dynamic_token_dim",
        "static_dim",
        "entity_static_feat_size",
        "ally_state_feat_size",
        "enemy_state_feat_size",
        "ally_has_shields",
        "enemy_has_shields",
        "num_unit_types",
        "n_actions",
        "latent_dim",
        "memory_dim",
        "action_conditioned_memory",
        "enemy_visibility_mask",
        "enemy_sight_range",
        "visibility_xy_indices",
        "latent_normalization",
    )


def _checkpoint_contract(metadata: dict[str, Any], cfg: dict[str, Any], arch: dict[str, Any]) -> dict[str, Any]:
    contract = dict(metadata)
    contract.setdefault("latent_dim", int(arch["latent_dim"]))
    contract.setdefault("memory_dim", int(cfg.get("rollout_memory_dim", cfg.get("memory_dim", 128))))
    contract.setdefault("action_conditioned_memory", bool(cfg.get("action_conditioned_memory", False)))
    contract.setdefault("enemy_visibility_mask", bool(cfg.get("enemy_visibility_mask", False)))
    contract.setdefault("enemy_sight_range", float(cfg.get("enemy_sight_range", 9.0)))
    contract.setdefault("visibility_xy_indices", tuple(cfg.get("xy_indices", cfg.get("visibility_xy_indices", (2, 3)))))
    contract.setdefault("latent_normalization", cfg.get("latent_normalization", cfg.get("latent_normalize", "none")))
    return contract


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return tuple(a) == tuple(b)
    return a == b


def validate_metadata(metadata: dict[str, Any], live: dict[str, Any]) -> None:
    """Validate a live map against the checkpoint's padded JEPA contract.

    The checkpoint stores dataset-wide padded capacities and a canonical
    feature schema. A particular live map may contain fewer entities and
    may omit optional raw features such as shields.
    """
    mismatches = []

    # These fields require special handling rather than strict equality.
    special_fields = {
        "n_agents",
        "n_enemies",
        "n_actions",
        "ally_state_feat_size",
        "enemy_state_feat_size",
        "ally_has_shields",
        "enemy_has_shields",
    }

    # Fixed architectural/schema fields must still match exactly.
    for field in VALIDATION_FIELDS:
        if field in special_fields:
            continue

        if field not in metadata:
            mismatches.append(
                (field, "<missing in checkpoint>",
                 live.get(field, "<missing in runtime>"))
            )
            continue

        if field not in live:
            mismatches.append(
                (field, metadata[field], "<missing in runtime>")
            )
            continue

        if not _same_value(metadata[field], live[field]):
            mismatches.append(
                (field, metadata[field], live[field])
            )

    # A live map may use fewer entities/actions than the padded capacity.
    capacity_fields = (
        ("n_agents", "max_agents"),
        ("n_enemies", "max_enemies"),
        ("n_actions", "max_actions"),
    )

    for live_field, capacity_field in capacity_fields:
        if live_field not in live:
            mismatches.append(
                (live_field, metadata.get(capacity_field, "<missing>"),
                 "<missing in runtime>")
            )
            continue

        capacity = metadata.get(
            capacity_field,
            metadata.get(live_field),
        )

        if capacity is None:
            mismatches.append(
                (capacity_field, "<missing in checkpoint>",
                 live[live_field])
            )
            continue

        if int(live[live_field]) > int(capacity):
            mismatches.append(
                (
                    live_field,
                    f"capacity={int(capacity)}",
                    int(live[live_field]),
                )
            )

    # Shield support is directional:
    # checkpoint=True, live=False is valid because the missing shield
    # channel is represented as zero.
    # checkpoint=False, live=True is invalid.
    for side in ("ally", "enemy"):
        shield_field = f"{side}_has_shields"
        size_field = f"{side}_state_feat_size"

        if shield_field not in metadata:
            mismatches.append(
                (shield_field, "<missing in checkpoint>",
                 live.get(shield_field, "<missing in runtime>"))
            )
            continue

        if shield_field not in live:
            mismatches.append(
                (shield_field, metadata[shield_field],
                 "<missing in runtime>")
            )
            continue

        checkpoint_has_shields = bool(metadata[shield_field])
        live_has_shields = bool(live[shield_field])

        if live_has_shields and not checkpoint_has_shields:
            mismatches.append(
                (
                    shield_field,
                    checkpoint_has_shields,
                    live_has_shields,
                )
            )

        if size_field not in metadata:
            mismatches.append(
                (size_field, "<missing in checkpoint>",
                 live.get(size_field, "<missing in runtime>"))
            )
            continue

        if size_field not in live:
            mismatches.append(
                (size_field, metadata[size_field],
                 "<missing in runtime>")
            )
            continue

        checkpoint_size = int(metadata[size_field])
        live_size = int(live[size_field])

        # SMACLite omits the raw shield value on maps without shields.
        # The canonical JEPA representation restores it as a zero channel.
        canonical_live_size = live_size
        if checkpoint_has_shields and not live_has_shields:
            canonical_live_size += 1

        if canonical_live_size != checkpoint_size:
            mismatches.append(
                (
                    size_field,
                    checkpoint_size,
                    f"raw={live_size}, canonical={canonical_live_size}",
                )
            )

    if mismatches:
        lines = [
            "JEPA checkpoint is incompatible with the live R2 environment:"
        ]
        lines.extend(
            f"  {field}: checkpoint={ckpt!r} live={value!r}"
            for field, ckpt, value in mismatches
        )
        raise JEPACompatibilityError("\n".join(lines))



def load_frozen_jepa_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device,
    live_metadata: dict[str, Any] | None = None,
    strict: bool = True,
):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"JEPA checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError("JEPA checkpoint must be a dict")
    model_state = _required(checkpoint, "model_state")
    memory_state = _required(checkpoint, "memory_module_state")
    metadata = dict(_required(checkpoint, "metadata"))
    cfg = dict(checkpoint.get("resolved_config", checkpoint.get("config", {})))
    (SMACJEPA, EntityRolloutGRUMemory, AnchoredActionConditionedEntityRolloutGRUMemory) = _import_jepa()
    arch = _arch_from(metadata, cfg)
    contract = _checkpoint_contract(metadata, cfg, arch)
    if live_metadata is not None:
        validate_metadata(contract, live_metadata)
    model = SMACJEPA(**arch)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Strict JEPA model_state load failed:\n"
            f"  missing={list(missing)}\n  unexpected={list(unexpected)}"
        )

    action_conditioned = bool(cfg.get("action_conditioned_memory", False))
    memory_dim = int(cfg.get("rollout_memory_dim", cfg.get("memory_dim", 128)))
    hidden = cfg.get("rollout_memory_hidden_dim", None)
    residual = not bool(cfg.get("rollout_memory_no_residual", False))
    anchored = bool(cfg.get("anchored_belief_memory", False)) or any(
        str(key).startswith("hidden_gate_net.") for key in memory_state
    )
    if anchored:
        if not action_conditioned:
            raise JEPACompatibilityError(
                "Anchored Exp33 checkpoint must set action_conditioned_memory=True"
            )
        memory = AnchoredActionConditionedEntityRolloutGRUMemory(
            latent_dim=int(arch["latent_dim"]),
            memory_dim=memory_dim,
            n_actions=int(metadata["n_actions"]),
            max_agents=int(metadata["max_agents"]),
            hidden_dim=hidden,
            residual=residual,
        )
    elif action_conditioned:
        memory = ActionConditionedEntityRolloutGRUMemory(
            latent_dim=int(arch["latent_dim"]),
            memory_dim=memory_dim,
            n_actions=int(metadata["n_actions"]),
            hidden_dim=hidden,
            residual=residual,
        )
    else:
        memory = EntityRolloutGRUMemory(
            latent_dim=int(arch["latent_dim"]),
            memory_dim=memory_dim,
            hidden_dim=hidden,
            residual=residual,
        )
    missing, unexpected = memory.load_state_dict(memory_state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Strict JEPA memory_module_state load failed:\n"
            f"  missing={list(missing)}\n  unexpected={list(unexpected)}"
        )
    for module in (model, memory):
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
    info = JEPACheckpointInfo(
        path=str(path),
        sha256=sha256_file(path),
        metadata=contract,
        resolved_config=cfg,
        training_regime=cfg.get("training_regime"),
        action_conditioned_memory=action_conditioned,
        latent_dim=int(arch["latent_dim"]),
        memory_dim=memory_dim,
        n_actions=int(metadata["n_actions"]),
    )
    return model, memory, info
