from __future__ import annotations

"""Exp45: Exp40 plus direct power-of-two latent jump prediction.

This is a compatibility layer over the *installed* Exp40 trainer. It deliberately
reuses Exp40's encoder, anchored action-conditioned memory, EMA target encoder,
R2-offline losses, event-balanced sampler, delta/event losses, and inverse action
pressure. The only added trainable mechanism is ``PowerOfTwoDirectPredictor``.

The data segment is extended to the largest direct horizon (default 16), while
Exp40's original recursive loss remains fixed at horizon 5. Direct targets are
supervised at 1, 2, 4, 8, and 16 steps from the same real context and real action
sequence. A weak sampled composition loss trains 2=1+1, 4=2+2, etc., making
binary inference such as 9=8+1 less out-of-distribution.
"""

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn

from .pow2_direct_predictor import (
    PowerOfTwoDirectPredictor,
    canonical_pow2_horizons,
    masked_latent_mse,
    normalize_entity_latent,
)
from . import train_jepa_exp40_dreamer as _exp40
from . import train_jepa_exp31_exp35 as _base


_CFG: dict[str, Any] = {}
_INIT_CHECKPOINT: dict[str, Any] | None = None
_TARGET_ENCODER_INITIALIZED = False
_LOSS_CALL_COUNT = 0


def _safe_load(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pow2-horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--pow2-base-rollout-horizon", type=int, default=5)
    parser.add_argument("--pow2-direct-weight", type=float, default=0.10)
    parser.add_argument("--pow2-composition-weight", type=float, default=0.05)
    parser.add_argument("--pow2-shared-head-weight", type=float, default=0.10)
    parser.add_argument("--pow2-hidden-dim", type=int, default=384)
    parser.add_argument("--pow2-action-embed-dim", type=int, default=48)
    parser.add_argument("--pow2-slot-embed-dim", type=int, default=32)
    parser.add_argument("--pow2-dropout", type=float, default=0.0)
    parser.add_argument("--pow2-residual-scale", type=float, default=0.25)
    parser.add_argument(
        "--pow2-horizon-weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional weights matching --pow2-horizons. Defaults to log2(h)+1.",
    )
    parser.add_argument("--pow2-warmup-steps", type=int, default=2000)
    parser.add_argument(
        "--pow2-init-from-exp40",
        default=None,
        help="Load Exp40 model/memory/projector/EMA weights but start a fresh optimizer.",
    )
    parser.add_argument(
        "--pow2-composition-sample-one",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample one composition scale per batch instead of evaluating every scale.",
    )
    return parser


def _extract_custom_args() -> tuple[argparse.Namespace, list[str]]:
    parser = _custom_parser()
    custom, remaining = parser.parse_known_args(sys.argv[1:])
    return custom, remaining


def _normalize_weights(horizons: tuple[int, ...], values: list[float] | None) -> dict[int, float]:
    if values is None:
        raw = [float(h.bit_length()) for h in horizons]  # log2(h)+1
    else:
        if len(values) != len(horizons):
            raise SystemExit("--pow2-horizon-weights must match --pow2-horizons length")
        raw = [float(v) for v in values]
    if any(v < 0 for v in raw) or sum(raw) <= 0:
        raise SystemExit("Power-of-two horizon weights must be nonnegative with positive sum")
    total = sum(raw)
    return {h: v / total for h, v in zip(horizons, raw)}


def _prepare_exp40_patch() -> None:
    # Install the exact current Exp40 contract first (anchored memory + Exp40 allowances).
    if not hasattr(_exp40, "_patch_for_exp33_dreamer"):
        raise SystemExit(
            "Installed train_jepa_exp40_dreamer.py lacks _patch_for_exp33_dreamer. "
            "Reinstall the known-good real Exp39/40/41 bundle first."
        )
    _exp40._patch_for_exp33_dreamer()


def _patch_for_pow2() -> None:
    global _INIT_CHECKPOINT

    original_parse_args = _base.parse_args
    original_loss = _base.markov_rollout_rnn_losses
    original_model_cls = _base.SMACJEPA
    original_memory_cls = _base.ActionConditionedEntityRolloutGRUMemory
    original_torch_save = torch.save

    original_r2_projector_cls = getattr(_base, "R2PosteriorProjector", None)
    original_memory_projector_cls = getattr(_base, "R2MemoryProjector", None)

    def parse_args() -> argparse.Namespace:
        global _CFG, _INIT_CHECKPOINT, _LOSS_CALL_COUNT
        custom, remaining = _extract_custom_args()
        horizons = canonical_pow2_horizons(custom.pow2_horizons)
        base_h = int(custom.pow2_base_rollout_horizon)
        if base_h < 1:
            raise SystemExit("--pow2-base-rollout-horizon must be positive")
        if max(horizons) < base_h:
            raise SystemExit("Largest power-of-two horizon must be >= base rollout horizon")
        if max(horizons) > 32:
            raise SystemExit("This implementation caps the largest direct horizon at 32")
        if float(custom.pow2_direct_weight) <= 0:
            raise SystemExit("--pow2-direct-weight must be positive")
        if float(custom.pow2_composition_weight) < 0:
            raise SystemExit("--pow2-composition-weight must be nonnegative")

        saved_argv = sys.argv
        try:
            # Exp40's own guard must still see its trusted recursive horizon (5).
            sys.argv = [saved_argv[0], *remaining]
            args = original_parse_args()
        finally:
            sys.argv = saved_argv

        if int(args.rollout_horizon) != base_h:
            raise SystemExit(
                "Pass --rollout-horizon equal to --pow2-base-rollout-horizon "
                f"({base_h}) so the Exp40 contract is validated before extension."
            )
        if args.resume and custom.pow2_init_from_exp40:
            raise SystemExit("Use either --resume for an Exp45 checkpoint or --pow2-init-from-exp40, not both")
        if args.resume:
            resume_checkpoint = _safe_load(args.resume)
            resume_cfg = resume_checkpoint.get(
                "resolved_config", resume_checkpoint.get("config", {})
            )
            if not bool(resume_cfg.get("pow2_direct_predictor", False)):
                raise SystemExit("--resume must point to an Exp45 Pow2 checkpoint")
            _LOSS_CALL_COUNT = int(resume_checkpoint.get("global_step", 0))

        init_path = custom.pow2_init_from_exp40
        if init_path:
            path = Path(init_path)
            if not path.is_file():
                raise SystemExit(f"Exp40 initialization checkpoint does not exist: {path}")
            _INIT_CHECKPOINT = _safe_load(path)
            init_cfg = _INIT_CHECKPOINT.get("resolved_config", _INIT_CHECKPOINT.get("config", {}))
            if not bool(init_cfg.get("r2_latent_normalize", False)):
                raise SystemExit("Initialization checkpoint is not an R2-normalized Exp40-family checkpoint")
            if int(init_cfg.get("rollout_horizon", 5)) != base_h:
                raise SystemExit(
                    f"Initialization checkpoint rollout_horizon={init_cfg.get('rollout_horizon')} "
                    f"does not match base horizon {base_h}"
                )

        horizon_weights = _normalize_weights(horizons, custom.pow2_horizon_weights)
        _CFG = {
            "horizons": horizons,
            "max_horizon": max(horizons),
            "base_horizon": base_h,
            "direct_weight": float(custom.pow2_direct_weight),
            "composition_weight": float(custom.pow2_composition_weight),
            "shared_head_weight": float(custom.pow2_shared_head_weight),
            "hidden_dim": int(custom.pow2_hidden_dim),
            "action_embed_dim": int(custom.pow2_action_embed_dim),
            "slot_embed_dim": int(custom.pow2_slot_embed_dim),
            "dropout": float(custom.pow2_dropout),
            "residual_scale": float(custom.pow2_residual_scale),
            "horizon_weights": horizon_weights,
            "warmup_steps": max(0, int(custom.pow2_warmup_steps)),
            "init_from": str(init_path) if init_path else None,
            "composition_sample_one": bool(custom.pow2_composition_sample_one),
        }

        # The dataset must expose targets/actions out to the largest direct horizon.
        # The wrapped base loss below explicitly restores Exp40's recursive h=5.
        args.rollout_horizon = int(_CFG["max_horizon"])
        args.pow2_horizons = list(horizons)
        args.pow2_max_horizon = int(_CFG["max_horizon"])
        args.pow2_base_rollout_horizon = base_h
        args.pow2_direct_weight = float(_CFG["direct_weight"])
        args.pow2_composition_weight = float(_CFG["composition_weight"])
        args.pow2_shared_head_weight = float(_CFG["shared_head_weight"])
        args.pow2_horizon_weights = horizon_weights
        args.pow2_warmup_steps = int(_CFG["warmup_steps"])
        args.pow2_init_from_exp40 = _CFG["init_from"]
        return args

    class Pow2SMACJEPA(original_model_cls):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if _INIT_CHECKPOINT is not None:
                # Load before registering the new branch so the original strict Exp40
                # state dict remains a valid initialization source.
                original_model_cls.load_state_dict(
                    self, _INIT_CHECKPOINT["model_state"], strict=True
                )
            latent_dim = int(kwargs.get("latent_dim", getattr(self, "latent_dim", 192)))
            n_actions = int(kwargs.get("n_actions"))
            max_agents = int(kwargs.get("max_agents", kwargs.get("n_agents")))
            max_enemies = int(kwargs.get("max_enemies", 0))
            self.pow2_predictor = PowerOfTwoDirectPredictor(
                latent_dim=latent_dim,
                n_actions=n_actions,
                max_agents=max_agents,
                max_entities=max_agents + max_enemies,
                horizons=_CFG["horizons"],
                hidden_dim=_CFG["hidden_dim"],
                action_embed_dim=_CFG["action_embed_dim"],
                slot_embed_dim=_CFG["slot_embed_dim"],
                dropout=_CFG["dropout"],
                residual_scale=_CFG["residual_scale"],
            )

        def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
            # Exp40 initialization lacks only the newly added branch. Exp45 resume
            # checkpoints contain it and remain strict.
            has_pow2 = any(str(k).startswith("pow2_predictor.") for k in state_dict)
            if has_pow2:
                return super().load_state_dict(state_dict, strict=strict)
            result = super().load_state_dict(state_dict, strict=False)
            bad_missing = [k for k in result.missing_keys if not k.startswith("pow2_predictor.")]
            if result.unexpected_keys or bad_missing:
                raise RuntimeError(
                    "Non-Pow2 checkpoint incompatibility: "
                    f"missing={bad_missing}, unexpected={result.unexpected_keys}"
                )
            return result

    class InitializedMemory(original_memory_cls):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if _INIT_CHECKPOINT is not None:
                state = _INIT_CHECKPOINT.get("memory_module_state")
                if state is None:
                    raise RuntimeError("Exp40 initialization checkpoint lacks memory_module_state")
                self.load_state_dict(state, strict=True)

    def make_initialized_projector(name: str, original_cls: type[nn.Module] | None):
        if original_cls is None:
            return None

        class InitializedProjector(original_cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                if _INIT_CHECKPOINT is not None:
                    state = _INIT_CHECKPOINT.get(name)
                    if state is not None:
                        self.load_state_dict(state, strict=True)

        return InitializedProjector

    def pow2_loss(*loss_args: Any, **loss_kwargs: Any) -> dict[str, torch.Tensor]:
        global _TARGET_ENCODER_INITIALIZED, _LOSS_CALL_COUNT
        model = loss_args[0]
        memory_module = loss_args[1]
        target_encoder = loss_args[4] if len(loss_args) > 4 else None
        batch = next(
            (item for item in loss_args if isinstance(item, dict) and "entity_seq" in item),
            None,
        )
        if batch is None:
            raise RuntimeError("Could not locate batch argument in Exp40 loss call")

        if (
            _INIT_CHECKPOINT is not None
            and target_encoder is not None
            and not _TARGET_ENCODER_INITIALIZED
        ):
            target_state = _INIT_CHECKPOINT.get("target_encoder_state")
            if target_state is not None:
                target_encoder.load_state_dict(target_state, strict=True)
            _TARGET_ENCODER_INITIALIZED = True

        base_kwargs = dict(loss_kwargs)
        base_kwargs["rollout_horizon"] = int(_CFG["base_horizon"])
        losses = original_loss(*loss_args, **base_kwargs)

        direct = _pow2_direct_losses(
            model=model,
            memory_module=memory_module,
            target_encoder=target_encoder,
            batch=batch,
            rollout_window=int(loss_kwargs["rollout_window"]),
            r2_latent_normalize=bool(loss_kwargs.get("r2_latent_normalize", True)),
        )
        _LOSS_CALL_COUNT += 1
        warmup_steps = int(_CFG["warmup_steps"])
        warmup = 1.0 if warmup_steps <= 0 else min(1.0, _LOSS_CALL_COUNT / warmup_steps)
        direct_weight = float(_CFG["direct_weight"]) * warmup
        composition_weight = float(_CFG["composition_weight"]) * warmup
        shared_weight = float(_CFG["shared_head_weight"]) * warmup
        losses["total_loss"] = (
            losses["total_loss"]
            + direct_weight * direct["pow2_direct_loss"]
            + composition_weight * direct["pow2_composition_loss"]
            + shared_weight * direct["pow2_shared_loss"]
        )
        losses.update(direct)
        losses["pow2_active_weight"] = torch.as_tensor(
            direct_weight, device=losses["total_loss"].device
        )
        losses["pow2_active_composition_weight"] = torch.as_tensor(
            composition_weight, device=losses["total_loss"].device
        )
        return losses

    def contract_save(obj: Any, *save_args: Any, **save_kwargs: Any):
        if isinstance(obj, dict) and "model_state" in obj:
            state = obj["model_state"]
            pow2_state = {
                str(k).removeprefix("pow2_predictor."): v
                for k, v in state.items()
                if str(k).startswith("pow2_predictor.")
            }
            obj["pow2_predictor_state"] = pow2_state
            cfg = obj.setdefault("resolved_config", {})
            cfg.update(
                {
                    "training_regime": "exp40_plus_pow2_direct_v1",
                    "objective_family": "r2offline_exp40_pow2_direct",
                    "pow2_direct_predictor": True,
                    "pow2_predictor_version": 1,
                    "pow2_horizons": list(_CFG["horizons"]),
                    "pow2_max_horizon": int(_CFG["max_horizon"]),
                    "pow2_base_rollout_horizon": int(_CFG["base_horizon"]),
                    "pow2_horizon_weights": _CFG["horizon_weights"],
                    "pow2_direct_weight": float(_CFG["direct_weight"]),
                    "pow2_composition_weight": float(_CFG["composition_weight"]),
                    "pow2_shared_head_weight": float(_CFG["shared_head_weight"]),
                    "pow2_hidden_dim": int(_CFG["hidden_dim"]),
                    "pow2_action_embed_dim": int(_CFG["action_embed_dim"]),
                    "pow2_slot_embed_dim": int(_CFG["slot_embed_dim"]),
                    "pow2_dropout": float(_CFG["dropout"]),
                    "pow2_residual_scale": float(_CFG["residual_scale"]),
                    "pow2_warmup_steps": int(_CFG["warmup_steps"]),
                    "pow2_init_from_exp40": _CFG["init_from"],
                    "pow2_dataset_rollout_horizon": int(_CFG["max_horizon"]),
                    "pow2_binary_composable": True,
                    "pow2_composed_horizon_unbounded_by_head": True,
                    "pow2_composed_horizon_note": "reuse largest trained block, subject to available action sequence",
                    "pow2_dynamic_exact_shared_head": True,
                    "pow2_dynamic_exact_horizons": list(range(1, int(_CFG["max_horizon"]) + 1)),
                    "pow2_dynamic_exact_sampling": "rotating_one_horizon_per_batch",
                    "pow2_direct_uses_real_action_sequence": True,
                    "pow2_direct_consumes_predicted_latent_inside_block": False,
                    "dreamer_compatible": False,
                    "dreamer_compatibility_note": (
                        "Strip pow2_predictor.* keys and restore rollout_horizon=5 for "
                        "ordinary Exp40/R2 loading; direct jump planning is experimental."
                    ),
                }
            )
            obj.setdefault("metadata", {}).update(
                {
                    "pow2_direct_predictor": True,
                    "pow2_horizons": list(_CFG["horizons"]),
                }
            )
        return original_torch_save(obj, *save_args, **save_kwargs)

    _base.parse_args = parse_args
    _base.SMACJEPA = Pow2SMACJEPA
    _base.ActionConditionedEntityRolloutGRUMemory = InitializedMemory
    if original_r2_projector_cls is not None:
        _base.R2PosteriorProjector = make_initialized_projector(
            "r2_projector_state", original_r2_projector_cls
        )
    if original_memory_projector_cls is not None:
        _base.R2MemoryProjector = make_initialized_projector(
            "memory_projector_state", original_memory_projector_cls
        )
    _base.markov_rollout_rnn_losses = pow2_loss
    _base.torch.save = contract_save


def _r2_normalize(
    latent: torch.Tensor,
    entity_mask: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    fn = getattr(_base, "r2_normalize_latent", None)
    if fn is not None:
        return fn(latent, entity_mask, enabled=enabled)
    return normalize_entity_latent(latent, entity_mask) if enabled else latent * entity_mask.unsqueeze(-1)


def _merge_presence(previous: torch.Tensor, observed: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    fn = getattr(_base, "merge_observed_presence", None)
    if fn is not None:
        return fn(previous, observed, slot)
    return torch.maximum(previous * slot, observed) * slot


def _memory_update(
    memory_module: nn.Module,
    latent: torch.Tensor,
    memory: torch.Tensor,
    observed: torch.Tensor,
    *,
    action: torch.Tensor,
    action_mask: torch.Tensor,
    action_context: torch.Tensor | None,
) -> torch.Tensor:
    if action_context is not None:
        return memory_module.update(latent, memory, observed, action_context=action_context)
    try:
        return memory_module.update(
            latent, memory, observed, action=action, action_mask=action_mask
        )
    except TypeError:
        return memory_module.update(latent, memory, observed)


def _pow2_direct_losses(
    *,
    model: nn.Module,
    memory_module: nn.Module,
    target_encoder: nn.Module | None,
    batch: dict[str, torch.Tensor],
    rollout_window: int,
    r2_latent_normalize: bool,
) -> dict[str, torch.Tensor]:
    entity_seq = batch["entity_seq"]
    observation_mask_seq = batch.get("observation_mask_seq", batch.get("entity_mask_seq"))
    if observation_mask_seq is None:
        raise RuntimeError("Pow2 direct loss needs observation_mask_seq or entity_mask_seq")
    target_entity_seq = batch.get("target_entity_seq", entity_seq)
    target_entity_mask_seq = batch.get("target_entity_mask_seq", observation_mask_seq)
    slot_mask_seq = batch.get("entity_slot_mask_seq", target_entity_mask_seq)
    action_seq = batch["action_seq"]
    action_mask_seq = batch["action_mask_seq"]
    state_mask = batch["state_mask"]

    bsz = entity_seq.shape[0]
    p = int(rollout_window)
    horizons: tuple[int, ...] = _CFG["horizons"]
    max_h = int(_CFG["max_horizon"])
    if entity_seq.shape[1] < p + max_h + 1:
        raise RuntimeError(
            f"Dataset segment has {entity_seq.shape[1]} states, needs at least {p + max_h + 1}"
        )

    input_raw = model.encoder(entity_seq, observation_mask_seq)
    input_latents = _r2_normalize(
        input_raw, observation_mask_seq, enabled=r2_latent_normalize
    )
    with torch.no_grad():
        if target_encoder is not None:
            target_raw = target_encoder(target_entity_seq, target_entity_mask_seq)
        else:
            target_raw = model.encoder(target_entity_seq, target_entity_mask_seq)
        target_latents = _r2_normalize(
            target_raw, target_entity_mask_seq, enabled=r2_latent_normalize
        ).detach()

    entities = input_latents.shape[2]
    main_memory = memory_module.initial_memory(
        bsz, entities, device=entity_seq.device, dtype=input_latents.dtype
    )
    main_presence = slot_mask_seq[:, 0].to(input_latents.dtype)
    action_context_seq = None
    if getattr(memory_module, "uses_action", False) and hasattr(
        memory_module, "precompute_action_context_sequence"
    ):
        action_context_seq = memory_module.precompute_action_context_sequence(
            action_seq,
            action_mask_seq,
            entities=entities,
            dtype=input_latents.dtype,
        )

    conditioned_starts: list[torch.Tensor] = []
    start_slot_masks: list[torch.Tensor] = []
    for start in range(p):
        observed = observation_mask_seq[:, start]
        slot = slot_mask_seq[:, start]
        main_presence = _merge_presence(main_presence, observed, slot)
        conditioned_starts.append(
            memory_module.condition(input_latents[:, start], main_memory, main_presence)
        )
        start_slot_masks.append(slot)
        context = action_context_seq[:, start] if action_context_seq is not None else None
        main_memory = _memory_update(
            memory_module,
            input_latents[:, start],
            main_memory,
            observed,
            action=action_seq[:, start],
            action_mask=action_mask_seq[:, start],
            action_context=context,
        )

    conditioned = torch.stack(conditioned_starts, dim=1)  # [B,P,E,D]
    start_masks = torch.stack(start_slot_masks, dim=1)
    start_indices = torch.arange(p, device=entity_seq.device)
    action_indices = start_indices[:, None] + torch.arange(max_h, device=entity_seq.device)[None, :]
    action_blocks = action_seq[:, action_indices]
    action_mask_blocks = action_mask_seq[:, action_indices]

    bp = bsz * p
    conditioned_flat = conditioned.reshape(bp, entities, conditioned.shape[-1])
    start_mask_flat = start_masks.reshape(bp, entities)
    action_flat = action_blocks.reshape(bp, max_h, *action_blocks.shape[3:])
    action_mask_flat = action_mask_blocks.reshape(bp, max_h, action_mask_blocks.shape[-1])

    predictor: PowerOfTwoDirectPredictor = model.pow2_predictor
    all_direct_horizons = tuple(range(1, max_h + 1))
    outputs = predictor(
        conditioned_flat,
        action_flat,
        action_mask_flat,
        start_mask_flat,
        horizons=all_direct_horizons,
        include_shared_predictions=True,
    )

    direct_loss = conditioned_flat.new_zeros(())
    shared_loss = conditioned_flat.new_zeros(())
    metrics: dict[str, torch.Tensor] = {}
    target_by_h: dict[int, torch.Tensor] = {}
    mask_by_h: dict[int, torch.Tensor] = {}
    for horizon in horizons:
        target_indices = start_indices + horizon
        target = target_latents.index_select(1, target_indices).reshape(
            bp, entities, target_latents.shape[-1]
        )
        mask = (
            target_entity_mask_seq.index_select(1, target_indices)
            * state_mask.index_select(1, target_indices).unsqueeze(-1)
        ).reshape(bp, entities)
        pred = _r2_normalize(outputs[horizon], start_mask_flat, enabled=r2_latent_normalize)
        loss_h = masked_latent_mse(pred, target, mask)
        direct_loss = direct_loss + float(_CFG["horizon_weights"][horizon]) * loss_h
        metrics[f"pow2_loss_h{horizon}"] = loss_h.detach()
        target_by_h[horizon] = target
        mask_by_h[horizon] = mask

    # A shared readout receives direct supervision at a rotating arbitrary
    # horizon. Across training this covers every k in [1, max_h], removing the
    # old fixed-window restriction while the specialized power heads retain
    # high-capacity anchors for binary composition.
    dynamic_horizon = int((_LOSS_CALL_COUNT % max_h) + 1)
    dynamic_indices = start_indices + dynamic_horizon
    dynamic_target = target_latents.index_select(1, dynamic_indices).reshape(
        bp, entities, target_latents.shape[-1]
    )
    dynamic_mask = (
        target_entity_mask_seq.index_select(1, dynamic_indices)
        * state_mask.index_select(1, dynamic_indices).unsqueeze(-1)
    ).reshape(bp, entities)
    dynamic_pred = _r2_normalize(
        outputs[f"shared_{dynamic_horizon}"],
        start_mask_flat,
        enabled=r2_latent_normalize,
    )
    shared_loss = masked_latent_mse(dynamic_pred, dynamic_target, dynamic_mask)
    metrics["pow2_dynamic_shared_horizon"] = torch.as_tensor(
        float(dynamic_horizon), device=direct_loss.device
    )
    metrics["pow2_dynamic_shared_loss"] = shared_loss.detach()

    composable = [h for h in horizons if h > 1 and (h // 2) in horizons]
    composition_loss = conditioned_flat.new_zeros(())
    composition_scale = 0
    if composable and float(_CFG["composition_weight"]) > 0:
        if _CFG["composition_sample_one"]:
            selected = [composable[_LOSS_CALL_COUNT % len(composable)]]
        else:
            selected = composable
        for horizon in selected:
            half = horizon // 2
            mid = _r2_normalize(
                outputs[half], start_mask_flat, enabled=r2_latent_normalize
            )
            mid_indices = start_indices + half
            mid_mask = target_entity_mask_seq.index_select(1, mid_indices).reshape(bp, entities)
            composed = predictor.predict_block(
                mid,
                action_flat[:, half:horizon],
                action_mask_flat[:, half:horizon],
                mid_mask,
                horizon=half,
            )
            composed = _r2_normalize(composed, mid_mask, enabled=r2_latent_normalize)
            target_loss = masked_latent_mse(composed, target_by_h[horizon], mask_by_h[horizon])
            direct_consistency = masked_latent_mse(
                composed,
                _r2_normalize(outputs[horizon], start_mask_flat, enabled=r2_latent_normalize).detach(),
                mask_by_h[horizon],
            )
            composition_loss = composition_loss + target_loss + 0.25 * direct_consistency
            composition_scale = horizon
        composition_loss = composition_loss / len(selected)

    metrics.update(
        {
            "pow2_direct_loss": direct_loss,
            "pow2_shared_loss": shared_loss,
            "pow2_composition_loss": composition_loss,
            "pow2_composition_scale": torch.as_tensor(
                float(composition_scale), device=direct_loss.device
            ),
            "pow2_max_horizon": torch.as_tensor(float(max_h), device=direct_loss.device),
        }
    )
    return metrics


def main() -> None:
    _prepare_exp40_patch()
    _patch_for_pow2()
    print(
        "[EXP45] Exp40 + power-of-two direct LSTM predictor. "
        "Recursive Exp40 loss remains h=5; direct targets use 1,2,4,8,16.",
        flush=True,
    )
    _base.main()


if __name__ == "__main__":
    main()
