from __future__ import annotations

"""Evaluate Exp45's direct power-of-two and binary-composed predictions.

This evaluator is intentionally separate from the trusted Exp40 ordinary and
hidden-belief evaluators. It measures the added direct branch against real EMA
latent targets while the standard evaluators continue to score the unchanged
Exp40 recursive branch using a sanitized checkpoint.
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from smac_jepa.pow2_direct_predictor import PowerOfTwoDirectPredictor


def safe_load(path: Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("_pow2_base_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_ints(text: str) -> list[int]:
    values = [int(piece) for piece in text.replace(",", " ").split()]
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("expected positive integer horizons")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate direct power-of-two JEPA predictions and binary compositions."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--base-evaluator",
        default="eval_rnn_seqmem_dreamer_probe.py",
        help="Installed ordinary evaluator used only for dataset/model/memory builders.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=300)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--window-mode", choices=["sequential", "random"], default="sequential")
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument(
        "--power-horizons",
        type=parse_ints,
        default=None,
        help='Quoted list, e.g. --power-horizons "1 2 4 8 16". Defaults to checkpoint.',
    )
    parser.add_argument(
        "--binary-horizons",
        type=parse_ints,
        default=[3, 5, 9, 13, 15, 16],
        help='Quoted list, e.g. --binary-horizons "3 5 9 13 15 16".',
    )
    parser.add_argument(
        "--max-composed-horizon",
        type=int,
        default=64,
        help="Safety cap for binary/power-chunk evaluation; trained blocks may be reused.",
    )
    parser.add_argument(
        "--exact-horizons",
        type=parse_ints,
        default=None,
        help="Optional arbitrary one-pass horizons scored with the dynamically trained shared head.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def sanitized_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    out = copy.copy(checkpoint)
    out["model_state"] = {
        key: value
        for key, value in checkpoint["model_state"].items()
        if not str(key).startswith("pow2_predictor.")
    }
    cfg = dict(checkpoint.get("resolved_config", checkpoint.get("config", {})))
    cfg["rollout_horizon"] = int(cfg.get("pow2_base_rollout_horizon", 5))
    out["resolved_config"] = cfg
    return out


def predictor_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    explicit = checkpoint.get("pow2_predictor_state")
    if isinstance(explicit, dict) and explicit:
        return explicit
    state = {
        str(key).removeprefix("pow2_predictor."): value
        for key, value in checkpoint["model_state"].items()
        if str(key).startswith("pow2_predictor.")
    }
    if not state:
        raise RuntimeError("Checkpoint has no pow2 predictor state")
    return state


def infer_architecture(
    state: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    dataset: Any,
) -> dict[str, Any]:
    power_keys = [key for key in state if key.startswith("power_heads.")]
    horizons = sorted({int(key.split(".")[1]) for key in power_keys})
    if not horizons:
        horizons = [int(x) for x in cfg.get("pow2_horizons", [1, 2, 4, 8, 16])]

    action_weight = state["action_embedding.weight"]
    agent_weight = state["agent_embedding.weight"]
    slot_weight = state["slot_embedding.weight"]
    init_h_weight = state["init_h.weight"]
    entity_proj_weight = state["entity_context_proj.weight"]
    return {
        "latent_dim": int(entity_proj_weight.shape[1]),
        "n_actions": int(action_weight.shape[0]),
        "max_agents": int(agent_weight.shape[0]),
        "max_entities": int(slot_weight.shape[0]),
        "horizons": horizons,
        "hidden_dim": int(init_h_weight.shape[0]),
        "action_embed_dim": int(action_weight.shape[1]),
        "slot_embed_dim": int(slot_weight.shape[1]),
        "dropout": float(cfg.get("pow2_dropout", 0.0)),
        "residual_scale": float(cfg.get("pow2_residual_scale", 0.25)),
    }


def build_exp40_memory(base: Any, checkpoint: dict[str, Any], dataset: Any, device: torch.device):
    cfg = checkpoint.get("resolved_config", checkpoint.get("config", {}))
    memory_state = checkpoint.get("memory_module_state", {})
    anchored = bool(cfg.get("anchored_belief_memory", False)) or any(
        str(key).startswith("hidden_gate_net.") for key in memory_state
    )
    if not anchored:
        return base.build_memory_module(checkpoint, dataset, device)

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


def target_encoder_from_checkpoint(model: torch.nn.Module, checkpoint: dict[str, Any], device: torch.device):
    encoder = copy.deepcopy(model.encoder).to(device)
    state = checkpoint.get("target_encoder_state")
    if state is not None:
        encoder.load_state_dict(state, strict=True)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def merge_presence(previous: torch.Tensor, observed: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    return torch.maximum(previous * slot, observed) * slot


def memory_update(
    memory_module: torch.nn.Module,
    latent: torch.Tensor,
    memory: torch.Tensor,
    observed: torch.Tensor,
    action: torch.Tensor,
    action_mask: torch.Tensor,
    action_context: torch.Tensor | None,
) -> torch.Tensor:
    if action_context is not None:
        return memory_module.update(latent, memory, observed, action_context=action_context)
    try:
        return memory_module.update(latent, memory, observed, action=action, action_mask=action_mask)
    except TypeError:
        return memory_module.update(latent, memory, observed)


class MetricAccumulator:
    def __init__(self) -> None:
        self.absolute_sum = 0.0
        self.square_sum = 0.0
        self.elements = 0.0

    def add(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        expanded = mask.to(pred.dtype).unsqueeze(-1)
        difference = pred - target
        self.absolute_sum += float((difference.abs() * expanded).sum().item())
        self.square_sum += float((difference.square() * expanded).sum().item())
        self.elements += float(mask.sum().item() * pred.shape[-1])

    def result(self) -> dict[str, float | int]:
        denom = max(self.elements, 1.0)
        return {
            "mae": self.absolute_sum / denom,
            "mse": self.square_sum / denom,
            "elements": int(self.elements),
        }


def subset_masks(
    observation_mask_seq: torch.Tensor,
    target_mask_seq: torch.Tensor,
    state_mask: torch.Tensor,
    target_indices: torch.Tensor,
    max_agents: int,
) -> dict[str, torch.Tensor]:
    # [B,P,E]
    target_mask = target_mask_seq.index_select(1, target_indices)
    valid = target_mask * state_mask.index_select(1, target_indices).unsqueeze(-1)
    observed_target = observation_mask_seq.index_select(1, target_indices)
    visible = valid * observed_target

    seen_rows = []
    for target_index in target_indices.tolist():
        if target_index <= 0:
            seen_rows.append(torch.zeros_like(observation_mask_seq[:, 0]))
        else:
            seen_rows.append(observation_mask_seq[:, :target_index].amax(dim=1))
    seen_before = torch.stack(seen_rows, dim=1)
    enemy = torch.zeros_like(valid)
    enemy[..., max_agents:] = 1
    natural_hidden_enemy = valid * (1.0 - observed_target) * seen_before * enemy
    enemy_all = valid * enemy
    ally_all = valid * (1.0 - enemy)
    return {
        "all": valid,
        "visible": visible,
        "natural_hidden_enemy": natural_hidden_enemy,
        "enemy": enemy_all,
        "ally": ally_all,
    }


def normalize(base: Any, latent: torch.Tensor, mask: torch.Tensor, enabled: bool) -> torch.Tensor:
    return base.normalize_entity_latent(latent, mask, enabled=enabled)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = safe_load(checkpoint_path, "cpu")
    cfg = checkpoint.get("resolved_config", checkpoint.get("config", {}))
    if not bool(cfg.get("pow2_direct_predictor", False)):
        raise RuntimeError("Checkpoint is not marked as an Exp45 Pow2 checkpoint")

    evaluator_path = Path(args.base_evaluator).expanduser()
    if not evaluator_path.is_file():
        candidate = Path.cwd() / evaluator_path
        if candidate.is_file():
            evaluator_path = candidate
        else:
            raise RuntimeError(f"Base evaluator not found: {args.base_evaluator}")
    base = load_module(evaluator_path.resolve())

    saved_horizons = [int(x) for x in cfg.get("pow2_horizons", [1, 2, 4, 8, 16])]
    power_horizons = sorted(set(args.power_horizons or saved_horizons))
    binary_horizons = sorted(set(args.binary_horizons or []))
    exact_horizons = sorted(set(args.exact_horizons or []))
    requested_max = max(power_horizons + binary_horizons + exact_horizons)
    trained_max = max(saved_horizons)
    if max(power_horizons + exact_horizons, default=1) > trained_max:
        raise RuntimeError(
            "Specialized power and one-pass exact horizons cannot exceed the "
            f"trained maximum {trained_max}"
        )
    if max(binary_horizons, default=1) > int(args.max_composed_horizon):
        raise RuntimeError(
            f"Binary horizon exceeds --max-composed-horizon={args.max_composed_horizon}"
        )
    if any(h not in saved_horizons for h in power_horizons):
        raise RuntimeError(
            f"Power metrics must use trained horizons {saved_horizons}; requested {power_horizons}"
        )

    base_checkpoint = sanitized_checkpoint(checkpoint)
    dataset = base.build_dataset(
        args.manifest,
        args.split,
        cfg,
        args.window_mode,
        args.samples_per_epoch,
        None,
        None,
        requested_max,
    )
    model = base.build_model(base_checkpoint, dataset, device)
    memory_module = build_exp40_memory(base, base_checkpoint, dataset, device)
    target_encoder = target_encoder_from_checkpoint(model, checkpoint, device)

    state = predictor_state(checkpoint)
    architecture = infer_architecture(state, cfg, checkpoint.get("metadata", {}), dataset)
    predictor = PowerOfTwoDirectPredictor(**architecture).to(device)
    predictor.load_state_dict(state, strict=True)
    predictor.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    amp_enabled = bool(device.type == "cuda" and not args.no_amp)
    accumulators: dict[str, MetricAccumulator] = {}

    def acc(method: str, horizon: int, subset: str) -> MetricAccumulator:
        key = f"{method}/h{horizon}/{subset}"
        if key not in accumulators:
            accumulators[key] = MetricAccumulator()
        return accumulators[key]

    rollout_window = int(cfg.get("rollout_window", 20))
    r2_norm = bool(cfg.get("r2_latent_normalize", True))
    processed_batches = 0
    processed_windows = 0

    for batch_index, cpu_batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
        entity_seq = batch["entity_seq"]
        observation = batch.get("observation_mask_seq", batch.get("entity_mask_seq"))
        if observation is None:
            raise RuntimeError("Dataset lacks observation/entity mask")
        target_entity_seq = batch.get("target_entity_seq", entity_seq)
        target_mask_seq = batch.get("target_entity_mask_seq", observation)
        slot_mask_seq = batch.get("entity_slot_mask_seq", target_mask_seq)
        action_seq = batch["action_seq"]
        action_mask_seq = batch["action_mask_seq"]
        state_mask = batch["state_mask"]
        bsz = int(entity_seq.shape[0])
        p = rollout_window

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            input_latents = normalize(
                base,
                model.encoder(entity_seq, observation),
                observation,
                r2_norm,
            )
            target_latents = normalize(
                base,
                target_encoder(target_entity_seq, target_mask_seq),
                target_mask_seq,
                r2_norm,
            )

            entities = int(input_latents.shape[2])
            memory = memory_module.initial_memory(
                bsz, entities, device=device, dtype=input_latents.dtype
            )
            presence = slot_mask_seq[:, 0].to(input_latents.dtype)
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

            contexts = []
            start_masks = []
            for start in range(p):
                observed = observation[:, start]
                slot = slot_mask_seq[:, start]
                presence = merge_presence(presence, observed, slot)
                contexts.append(memory_module.condition(input_latents[:, start], memory, presence))
                start_masks.append(slot)
                action_context = (
                    action_context_seq[:, start] if action_context_seq is not None else None
                )
                memory = memory_update(
                    memory_module,
                    input_latents[:, start],
                    memory,
                    observed,
                    action_seq[:, start],
                    action_mask_seq[:, start],
                    action_context,
                )

            context = torch.stack(contexts, dim=1)
            start_mask = torch.stack(start_masks, dim=1)
            start_indices = torch.arange(p, device=device)
            action_indices = start_indices[:, None] + torch.arange(requested_max, device=device)[None, :]
            actions = action_seq[:, action_indices]
            action_masks = action_mask_seq[:, action_indices]
            bp = bsz * p
            context_flat = context.reshape(bp, entities, context.shape[-1])
            start_mask_flat = start_mask.reshape(bp, entities)
            actions_flat = actions.reshape(bp, requested_max, *actions.shape[3:])
            action_masks_flat = action_masks.reshape(bp, requested_max, action_masks.shape[-1])

            power_outputs = predictor(
                context_flat,
                actions_flat,
                action_masks_flat,
                start_mask_flat,
                horizons=power_horizons,
            )

            for horizon in power_horizons:
                target_indices = start_indices + horizon
                target = target_latents.index_select(1, target_indices).reshape(
                    bp, entities, target_latents.shape[-1]
                )
                masks = subset_masks(
                    observation,
                    target_mask_seq,
                    state_mask,
                    target_indices,
                    architecture["max_agents"],
                )
                pred = normalize(base, power_outputs[horizon], start_mask_flat, r2_norm)
                for subset, mask in masks.items():
                    acc("direct_power", horizon, subset).add(
                        pred,
                        target,
                        mask.reshape(bp, entities),
                    )

            for horizon in binary_horizons:
                target_indices = start_indices + horizon
                target = target_latents.index_select(1, target_indices).reshape(
                    bp, entities, target_latents.shape[-1]
                )
                masks = subset_masks(
                    observation,
                    target_mask_seq,
                    state_mask,
                    target_indices,
                    architecture["max_agents"],
                )
                prediction = predictor.predict_binary(
                    context_flat,
                    actions_flat[:, :horizon],
                    action_masks_flat[:, :horizon],
                    start_mask_flat,
                    horizon=horizon,
                )
                pred = normalize(base, prediction.latent, start_mask_flat, r2_norm)
                for subset, mask in masks.items():
                    acc("binary", horizon, subset).add(
                        pred,
                        target,
                        mask.reshape(bp, entities),
                    )

            for horizon in exact_horizons:
                target_indices = start_indices + horizon
                target = target_latents.index_select(1, target_indices).reshape(
                    bp, entities, target_latents.shape[-1]
                )
                masks = subset_masks(
                    observation,
                    target_mask_seq,
                    state_mask,
                    target_indices,
                    architecture["max_agents"],
                )
                pred = predictor.predict_exact(
                    context_flat,
                    actions_flat[:, :horizon],
                    action_masks_flat[:, :horizon],
                    start_mask_flat,
                    horizon=horizon,
                )
                pred = normalize(base, pred, start_mask_flat, r2_norm)
                for subset, mask in masks.items():
                    acc("direct_exact_shared", horizon, subset).add(
                        pred,
                        target,
                        mask.reshape(bp, entities),
                    )

        processed_batches += 1
        processed_windows += bsz * p

    metrics = {key: value.result() for key, value in sorted(accumulators.items())}
    def decompose(horizon: int) -> list[int]:
        remaining = int(horizon)
        blocks: list[int] = []
        descending = sorted(saved_horizons, reverse=True)
        while remaining:
            block = next((value for value in descending if value <= remaining), None)
            if block is None:
                raise RuntimeError(f"Cannot decompose horizon {horizon}")
            blocks.append(block)
            remaining -= block
        return blocks

    binary_blocks = {str(horizon): decompose(horizon) for horizon in binary_horizons}
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "eval_split": args.split,
        "processed_batches": processed_batches,
        "processed_start_windows": processed_windows,
        "rollout_window": rollout_window,
        "trained_power_horizons": saved_horizons,
        "evaluated_power_horizons": power_horizons,
        "evaluated_binary_horizons": binary_horizons,
        "evaluated_exact_diagnostic_horizons": exact_horizons,
        "binary_blocks": binary_blocks,
        "r2_latent_normalize": r2_norm,
        "architecture": architecture,
        "metric_definition": {
            "direct_power": "one real memory-conditioned context + real action prefix; no predicted latent consumed inside block",
            "binary": "greedy power-of-two chunk composition, reusing the largest block when needed; predicted latent consumed only at block boundaries",
            "natural_hidden_enemy": "full-valid enemy target, not observed at target, observed at least once before target",
            "direct_exact_shared": "one-pass shared readout; Exp45 rotates direct supervision across every horizon 1..max_horizon",
        },
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[OK] wrote {output}")


if __name__ == "__main__":
    main()
