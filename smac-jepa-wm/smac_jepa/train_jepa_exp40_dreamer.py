from __future__ import annotations

"""Train the full Exp33 anchored JEPA for the combined R2-Dreamer backend.

This is deliberately a thin compatibility layer over the full June-29
``train_jepa_exp31_exp35`` trainer. It preserves the complete Exp33 memory
architecture instead of replacing it with the legacy plain GRU.
"""

import os
from typing import Any

import torch

from . import train_jepa_exp31_exp35 as _base
from .anchored_belief_memory import (
    AnchoredActionConditionedEntityRolloutGRUMemory,
)

_MEMORY_ARCH = "anchored_ordered_action_latent_filter_v1"
_CONTRACT_VERSION = 1


def _safe_load(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _require_full_base_trainer() -> None:
    required = (
        "main",
        "parse_args",
        "markov_rollout_rnn_losses",
        "ActionConditionedEntityRolloutGRUMemory",
        "get_model_preset",
    )
    missing = [name for name in required if not hasattr(_base, name)]
    if missing:
        raise SystemExit(
            "The full Exp31-Exp33 base trainer is missing APIs: "
            + ", ".join(missing)
            + ". Copy the latest working smac_jepa/train_jepa_exp31_exp35.py "
              "into this repository before running this wrapper."
        )


def _patch_for_exp33_dreamer() -> None:
    _require_full_base_trainer()

    original_parse_args = _base.parse_args
    original_loss = _base.markov_rollout_rnn_losses
    original_torch_save = torch.save

    def parse_args():
        args = original_parse_args()
        args.anchored_belief_memory = True
        args.anchored_belief_version = 1
        args.anchor_gate_init = float(
            os.environ.get("SMAC_JEPA_ANCHOR_GATE_INIT", "-3.0")
        )
        args.anchor_delta_scale = float(
            os.environ.get("SMAC_JEPA_ANCHOR_DELTA_SCALE", "0.25")
        )
        args.anchor_hidden_correction_scale = float(
            os.environ.get(
                "SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE", "0.10"
            )
        )
        args.anchor_gate_sparsity_weight = float(
            os.environ.get("SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT", "0.002")
        )

        latent_dim = int(
            args.latent_dim
            or _base.get_model_preset(args.model_size).latent_dim
        )
        recurrent_dim = int(args.rollout_memory_dim) - latent_dim - 2
        args.anchored_recurrent_dim = recurrent_dim

        errors: list[str] = []
        if not args.action_conditioned_memory:
            errors.append("--action-conditioned-memory is required")
        if recurrent_dim < 16:
            errors.append(
                "--rollout-memory-dim must be at least latent_dim + 18; "
                f"got latent_dim={latent_dim}, memory_dim={args.rollout_memory_dim}"
            )
        if args.target_mode != "full":
            errors.append("--target-mode full is required for hidden-belief learning")
        if not args.r2_latent_normalize:
            errors.append("--r2-latent-normalize is required")
        if not args.ema_target_encoder:
            errors.append("--ema-target-encoder is required")
        if args.occlusion_mode != "contiguous":
            errors.append("--occlusion-mode contiguous is required")
        if int(args.rollout_horizon) != 5:
            errors.append("Exp33/R2 parity currently requires --rollout-horizon 5")
        # This experiment allows inverse dynamics as a small action-sensitivity auxiliary.
        if float(args.memory_barlow_scale) != 0.0:
            errors.append("memory Barlow is not part of the retained Exp33 recipe")
        if bool(args.residual_state_decoder):
            errors.append("residual state decoding is not part of Exp33")
        if bool(args.direct_action_fusion):
            errors.append("direct action fusion is not part of Exp33")
        # This experiment intentionally allows event-balanced sampling.

        if args.init_from:
            errors.append(
                "Exp33 changes the memory layout and must start from scratch; "
                "--init-from is not allowed"
            )
        if args.resume:
            checkpoint = _safe_load(args.resume)
            cfg = checkpoint.get("resolved_config", checkpoint.get("config", {}))
            state = checkpoint.get("memory_module_state", {})
            anchored = bool(cfg.get("anchored_belief_memory", False)) or any(
                str(key).startswith("hidden_gate_net.") for key in state
            )
            if not anchored:
                errors.append("--resume must point to an anchored Exp33 checkpoint")

        if errors:
            raise SystemExit("Invalid Exp33-Dreamer configuration:\n- " + "\n- ".join(errors))
        return args

    def anchored_loss(*args, **kwargs):
        memory_module = args[1]
        if hasattr(memory_module, "reset_auxiliary_statistics"):
            memory_module.reset_auxiliary_statistics()
        losses = original_loss(*args, **kwargs)
        gate_mean = memory_module.gate_open_mean()
        weight = float(
            os.environ.get("SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT", "0.002")
        )
        gate_penalty = gate_mean * weight
        losses["anchor_gate_open_mean"] = gate_mean.detach()
        losses["anchor_gate_sparsity_loss"] = gate_penalty
        losses["total_loss"] = losses["total_loss"] + gate_penalty
        return losses

    def dreamer_contract_save(obj, *save_args, **save_kwargs):
        if isinstance(obj, dict):
            state = obj.get("memory_module_state", {})
            anchored_state = any(
                str(key).startswith("hidden_gate_net.") for key in state
            )
            if anchored_state:
                cfg = obj.setdefault("resolved_config", {})
                cfg.update(
                    {
                        "anchored_belief_memory": True,
                        "anchored_belief_version": 1,
                        "memory_architecture": _MEMORY_ARCH,
                        "memory_state_layout": "recurrent|anchor|seen|age",
                        "anchor_gate_init": float(
                            os.environ.get("SMAC_JEPA_ANCHOR_GATE_INIT", "-3.0")
                        ),
                        "anchor_delta_scale": float(
                            os.environ.get("SMAC_JEPA_ANCHOR_DELTA_SCALE", "0.25")
                        ),
                        "anchor_hidden_correction_scale": float(
                            os.environ.get(
                                "SMAC_JEPA_ANCHOR_HIDDEN_CORRECTION_SCALE", "0.10"
                            )
                        ),
                        "anchor_gate_sparsity_weight": float(
                            os.environ.get(
                                "SMAC_JEPA_ANCHOR_GATE_SPARSITY_WEIGHT", "0.002"
                            )
                        ),
                        # The standalone blocker-fixed rollout propagates the
                        # probability from the presence head, not an oracle mask.
                        "presence_rollout_mode": "soft",
                        "dreamer_integration_contract_version": _CONTRACT_VERSION,
                        "dreamer_compatible": True,
                        "dreamer_backend": "frozen_exp33_anchored_jepa",
                        "dreamer_max_imagination_horizon": int(
                            cfg.get("rollout_horizon", 5)
                        ),
                    }
                )
                metadata = obj.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata.update(
                        {
                            "memory_architecture": _MEMORY_ARCH,
                            "anchored_belief_memory": True,
                            "anchored_belief_version": 1,
                            "presence_rollout_mode": "soft",
                        }
                    )
        return original_torch_save(obj, *save_args, **save_kwargs)

    _base.parse_args = parse_args
    _base.markov_rollout_rnn_losses = anchored_loss
    _base.ActionConditionedEntityRolloutGRUMemory = (
        AnchoredActionConditionedEntityRolloutGRUMemory
    )
    _base.torch.save = dreamer_contract_save


def main() -> None:
    print("[EXP40] Exp39 + event-balanced/action-effective sampling")
    os.environ.setdefault("SMAC_JEPA_EXP34_TWO_MASK_LOSS", "1")
    os.environ.setdefault("SMAC_JEPA_PRESENCE_NEG_CLASS_WEIGHT", "3.0")
    _patch_for_exp33_dreamer()
    _base.main()


if __name__ == "__main__":
    main()
