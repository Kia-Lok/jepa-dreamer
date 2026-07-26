#!/usr/bin/env python3
from __future__ import annotations

"""Mine presentation-ready good and failure cases from the Exp40 JEPA checkpoint.

This script deliberately reuses the repository's blocker-fixed Exp31/33 evaluator
API so model construction, anchored-memory construction, visibility semantics, and
recursive rollout chronology stay identical to the official Exp40 evaluation.

It runs a 15-step autonomous rollout from many held-out windows and produces:
  * a ranked good eventful example set;
  * ranked failure examples by observed failure type;
  * GT-vs-predicted H1/H5/H15 figures for every selected example;
  * per-example CSV/Excel tables for later slide selection;
  * aggregate horizon curves (generated now, to be used only after examples);
  * a compact ZIP intended to be uploaded back to ChatGPT for analysis.

No memory ablation is performed by this script.
"""

import argparse
import csv
import heapq
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - environment-specific
    raise SystemExit(f"matplotlib is required for gallery plots: {exc}") from exc

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import eval_jepa_exp31_exp33_anchored as base
except Exception as exc:  # pragma: no cover - repository-specific
    raise SystemExit(
        "Could not import eval_jepa_exp31_exp33_anchored.py. Run this script "
        "with the SMAC-JEPA repository as the current working directory and "
        "ensure the final anchored evaluator is installed.\n"
        f"Original import error: {exc}"
    ) from exc


CATEGORY_ORDER = [
    "good_eventful",
    "late_rollout_drift",
    "position_drift",
    "health_or_damage_miss",
    "enemy_tracking_failure",
    "presence_lifecycle_failure",
    "copying_dynamic_change",
    "unstable_overshoot",
    "visibility_transition_failure",
]
CATEGORY_DESCRIPTIONS = {
    "good_eventful": (
        "Eventful rollout with low error at H15; avoids choosing a trivial static clip."
    ),
    "late_rollout_drift": (
        "Prediction is relatively accurate early but diverges substantially after H5."
    ),
    "position_drift": (
        "Entity positions drift away from the recorded trajectory by H15."
    ),
    "health_or_damage_miss": (
        "The rollout misses health/shield changes on transitions where those values change."
    ),
    "enemy_tracking_failure": (
        "Enemy-state prediction is substantially worse than allied-state prediction."
    ),
    "presence_lifecycle_failure": (
        "The presence head invents an absent entity or removes one that remains present."
    ),
    "copying_dynamic_change": (
        "The target changes, but the predicted state remains too close to the rollout start."
    ),
    "unstable_overshoot": (
        "The prediction changes more aggressively than the recorded transition or leaves plausible health bounds."
    ),
    "visibility_transition_failure": (
        "Enemy error rises around natural visibility changes or while the enemy is hidden."
    ),
}


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        real_index = int(self.indices[index])
        item = dict(self.dataset[real_index])
        item["_dataset_item_index"] = torch.tensor(real_index, dtype=torch.long)
        return item


@dataclass
class Candidate:
    category: str
    score: float
    example_id: str
    metrics: dict[str, Any]
    payload: dict[str, Any]


class CandidatePools:
    def __init__(self, keep_per_category: int):
        self.keep_per_category = int(keep_per_category)
        self.heaps: dict[str, list[tuple[float, int, Candidate]]] = {
            category: [] for category in CATEGORY_ORDER
        }
        self.counter = 0

    def consider(self, candidate: Candidate) -> None:
        if not math.isfinite(candidate.score):
            return
        heap = self.heaps[candidate.category]
        entry = (float(candidate.score), self.counter, candidate)
        self.counter += 1
        if len(heap) < self.keep_per_category:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)

    def ranked(self, category: str) -> list[Candidate]:
        return [
            item[2]
            for item in sorted(self.heaps[category], key=lambda value: value[0], reverse=True)
        ]


def parse_int_list(text: str) -> tuple[int, ...]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one integer index is required")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp40 H1/H5/H15 rollout gallery and failure-case miner"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool-multiplier", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", default="cuda", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--ally-position-indices", type=parse_int_list, default=(2, 3),
        help="Comma-separated entity-token indices. Default verified from the earlier sanity workbook: 2,3.",
    )
    parser.add_argument(
        "--enemy-position-indices", type=parse_int_list, default=(1, 2),
        help="Comma-separated entity-token indices. Default verified from the earlier sanity workbook: 1,2.",
    )
    parser.add_argument(
        "--ally-health-indices", type=parse_int_list, default=(0, 4),
        help="Comma-separated hp/shield indices for allied tokens.",
    )
    parser.add_argument(
        "--enemy-health-indices", type=parse_int_list, default=(0,),
        help="Comma-separated hp/shield indices for enemy tokens.",
    )
    parser.add_argument(
        "--good-min-change", type=float, default=0.01,
        help="Minimum H15 target dynamic change required for a good example.",
    )
    parser.add_argument(
        "--health-change-threshold", type=float, default=0.005,
        help="Minimum target hp/shield change required for the health failure category.",
    )
    parser.add_argument(
        "--max-items", type=int, default=None,
        help="Optional explicit number of dataset items; otherwise max_batches * batch_size.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return jsonable(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def safe_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def resolve_indices(length: int, count: int) -> list[int]:
    if count >= length:
        return list(range(length))
    raw = np.linspace(0, length - 1, num=count)
    indices = sorted(set(int(round(value)) for value in raw))
    if len(indices) < count:
        used = set(indices)
        for candidate in range(length):
            if candidate not in used:
                indices.append(candidate)
                used.add(candidate)
            if len(indices) >= count:
                break
    return sorted(indices[:count])


def make_feature_template(
    *,
    entities: int,
    token_dim: int,
    max_agents: int,
    ally_indices: Iterable[int],
    enemy_indices: Iterable[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    template = torch.zeros((1, 1, 1, entities, token_dim), device=device, dtype=dtype)
    for index in ally_indices:
        if 0 <= int(index) < token_dim:
            template[..., :max_agents, int(index)] = 1.0
    for index in enemy_indices:
        if 0 <= int(index) < token_dim:
            template[..., max_agents:, int(index)] = 1.0
    return template


def masked_mean_per_example(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    numerator = (values * mask).sum(dim=(-1, -2))
    denominator = mask.sum(dim=(-1, -2))
    result = numerator / denominator.clamp_min(1.0)
    return torch.where(denominator > 0, result, torch.full_like(result, float("nan")))


def flatten_metric(tensor: torch.Tensor, horizon_index: int) -> np.ndarray:
    return tensor[:, :, horizon_index].detach().float().cpu().numpy().reshape(-1)


def scalar_from_batch(batch: dict[str, torch.Tensor], key: str, b: int, default: int = -1) -> int:
    value = batch.get(key)
    if value is None:
        return int(default)
    try:
        return int(value[b].reshape(-1)[0].detach().cpu().item())
    except Exception:
        return int(default)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(row.get(key)) for key in fieldnames})


def choose_horizon_indices(horizon: int) -> list[tuple[str, int]]:
    requested = [("H1", 0), ("H5", min(4, horizon - 1)), ("H15", horizon - 1)]
    result: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, index in requested:
        if index not in seen:
            result.append((label, index))
            seen.add(index)
    return result


def entity_xy(array: np.ndarray, slot: int, max_agents: int, ally_pos: tuple[int, ...], enemy_pos: tuple[int, ...]) -> tuple[float, float]:
    indices = ally_pos if slot < max_agents else enemy_pos
    if len(indices) < 2 or max(indices[0], indices[1]) >= array.shape[-1]:
        return float("nan"), float("nan")
    return float(array[slot, indices[0]]), float(array[slot, indices[1]])


def plot_overview(candidate: Candidate, out_path: Path, args: argparse.Namespace, max_agents: int) -> None:
    payload = candidate.payload
    target = payload["target"]
    pred = payload["pred"]
    target_presence = payload["target_presence"]
    pred_presence = payload["pred_presence"]
    observed = payload["observed"]
    slot = payload["slot"]
    valid = payload["valid"]
    horizon_points = choose_horizon_indices(target.shape[0])

    all_x: list[float] = []
    all_y: list[float] = []
    for _, h in horizon_points:
        for values, presence in [(target[h], target_presence[h]), (pred[h], pred_presence[h])]:
            for entity in range(values.shape[0]):
                if not bool(slot[h, entity]) or float(presence[entity]) < args.presence_threshold:
                    continue
                x, y = entity_xy(values, entity, max_agents, args.ally_position_indices, args.enemy_position_indices)
                if math.isfinite(x) and math.isfinite(y):
                    all_x.append(x)
                    all_y.append(y)
    if all_x:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_pad = max((x_max - x_min) * 0.15, 0.08)
        y_pad = max((y_max - y_min) * 0.15, 0.08)
    else:
        x_min, x_max, y_min, y_max, x_pad, y_pad = -1.0, 1.0, -1.0, 1.0, 0.1, 0.1

    fig, axes = plt.subplots(2, len(horizon_points), figsize=(4.5 * len(horizon_points), 8.0), squeeze=False)
    for column, (label, h) in enumerate(horizon_points):
        for row, (row_name, values, presence) in enumerate([
            ("Ground truth", target[h], target_presence[h]),
            ("Prediction", pred[h], pred_presence[h]),
        ]):
            ax = axes[row, column]
            for faction, start, stop, marker in [
                ("Ally", 0, min(max_agents, values.shape[0]), "o"),
                ("Enemy", min(max_agents, values.shape[0]), values.shape[0], "^"),
            ]:
                xs: list[float] = []
                ys: list[float] = []
                labels: list[str] = []
                alphas: list[float] = []
                for entity in range(start, stop):
                    if not bool(slot[h, entity]) or float(presence[entity]) < args.presence_threshold:
                        continue
                    x, y = entity_xy(values, entity, max_agents, args.ally_position_indices, args.enemy_position_indices)
                    if not (math.isfinite(x) and math.isfinite(y)):
                        continue
                    xs.append(x)
                    ys.append(y)
                    labels.append(("A" if faction == "Ally" else "E") + str(entity if faction == "Ally" else entity - max_agents))
                    alphas.append(1.0 if faction == "Ally" or bool(observed[h, entity]) else 0.42)
                if xs:
                    # Separate scatter calls retain the default Matplotlib palette and use marker shape for faction.
                    ax.scatter(xs, ys, marker=marker, s=80, label=faction)
                    for x, y, text, alpha in zip(xs, ys, labels, alphas):
                        ax.annotate(text, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8, alpha=alpha)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.22)
            ax.set_title(f"{row_name} · {label}")
            if row == 1:
                ax.set_xlabel("normalised x")
            if column == 0:
                ax.set_ylabel("normalised y")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    metrics = candidate.metrics
    fig.suptitle(
        f"{candidate.category}: {candidate.example_id}\n"
        f"dynamic MAE H1={metrics.get('dynamic_mae_h1', float('nan')):.4f}, "
        f"H5={metrics.get('dynamic_mae_h5', float('nan')):.4f}, "
        f"H15={metrics.get('dynamic_mae_h15', float('nan')):.4f}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_overlay(candidate: Candidate, out_path: Path, args: argparse.Namespace, max_agents: int) -> None:
    payload = candidate.payload
    target = payload["target"]
    pred = payload["pred"]
    target_presence = payload["target_presence"]
    pred_presence = payload["pred_presence"]
    slot = payload["slot"]

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    for entity in range(target.shape[1]):
        prefix = "A" if entity < max_agents else "E"
        display_idx = entity if entity < max_agents else entity - max_agents
        gt_x, gt_y, pr_x, pr_y = [], [], [], []
        for h in range(target.shape[0]):
            if bool(slot[h, entity]) and float(target_presence[h, entity]) >= args.presence_threshold:
                x, y = entity_xy(target[h], entity, max_agents, args.ally_position_indices, args.enemy_position_indices)
                gt_x.append(x); gt_y.append(y)
            else:
                gt_x.append(float("nan")); gt_y.append(float("nan"))
            if bool(slot[h, entity]) and float(pred_presence[h, entity]) >= args.presence_threshold:
                x, y = entity_xy(pred[h], entity, max_agents, args.ally_position_indices, args.enemy_position_indices)
                pr_x.append(x); pr_y.append(y)
            else:
                pr_x.append(float("nan")); pr_y.append(float("nan"))
        if np.isfinite(gt_x).any() or np.isfinite(pr_x).any():
            (line,) = ax.plot(gt_x, gt_y, linewidth=1.8, label=f"{prefix}{display_idx} GT")
            ax.plot(pr_x, pr_y, linestyle="--", linewidth=1.5, color=line.get_color(), label=f"{prefix}{display_idx} pred")
    ax.set_xlabel("normalised x")
    ax.set_ylabel("normalised y")
    ax.set_title(f"Trajectory overlay H1–H{target.shape[0]} · {candidate.example_id}")
    ax.grid(True, alpha=0.22)
    ax.set_aspect("equal", adjustable="box")
    if len(ax.lines) <= 20:
        ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_error(candidate: Candidate, out_path: Path) -> None:
    payload = candidate.payload
    error = np.abs(payload["pred"] - payload["target"])
    dynamic = payload["dynamic_mask"]
    numerator = (error * dynamic).sum(axis=1)
    denominator = dynamic.sum(axis=1)
    curve = np.divide(numerator, np.maximum(denominator, 1.0), where=denominator > 0)
    curve[denominator <= 0] = np.nan
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    image = ax.imshow(curve.T, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_xlabel("rollout horizon")
    ax.set_ylabel("entity-token feature index")
    ax.set_xticks(range(0, curve.shape[0], max(1, curve.shape[0] // 8)))
    ax.set_xticklabels([str(index + 1) for index in range(0, curve.shape[0], max(1, curve.shape[0] // 8))])
    ax.set_title(f"Mean absolute feature error · {candidate.example_id}")
    fig.colorbar(image, ax=ax, label="absolute error")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_curves(rows: list[dict[str, Any]], chart_dir: Path) -> None:
    if not rows:
        return
    chart_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(row["horizon"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for key, label in [
        ("dynamic_mae", "overall"),
        ("ally_dynamic_mae", "allies"),
        ("enemy_dynamic_mae", "enemies"),
    ]:
        ax.plot(horizons, [row.get(key, float("nan")) for row in rows], marker="o", label=label)
    ax.set_xlabel("recursive rollout horizon")
    ax.set_ylabel("decoded dynamic MAE")
    ax.set_title("Exp40 decoded error across the 15-step rollout")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(chart_dir / "mae_by_horizon.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for key, label in [
        ("position_mae", "position"),
        ("health_mae", "health / shield"),
        ("visible_enemy_dynamic_mae", "visible enemies"),
        ("hidden_enemy_dynamic_mae", "hidden enemies"),
    ]:
        ax.plot(horizons, [row.get(key, float("nan")) for row in rows], marker="o", label=label)
    ax.set_xlabel("recursive rollout horizon")
    ax.set_ylabel("decoded MAE")
    ax.set_title("Error components across the rollout")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(chart_dir / "component_mae_by_horizon.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def optional_excel(out_path: Path, sheets: dict[str, list[dict[str, Any]]]) -> str | None:
    if pd is None:
        return "pandas is unavailable; CSV files were still written"
    try:
        with pd.ExcelWriter(out_path) as writer:
            for name, rows in sheets.items():
                frame = pd.DataFrame(rows)
                frame.to_excel(writer, sheet_name=name[:31], index=False)
                worksheet = writer.sheets[name[:31]]
                try:
                    worksheet.freeze_panes(1, 0)
                    worksheet.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns) - 1, 0))
                    for col_idx, column in enumerate(frame.columns):
                        sample = [str(column)] + [str(value) for value in frame[column].head(250).tolist()]
                        width = min(max(max(map(len, sample)) + 2, 10), 38)
                        worksheet.set_column(col_idx, col_idx, width)
                except Exception:
                    pass
        return None
    except Exception as exc:
        return f"Excel export failed ({exc}); CSV files were still written"


def save_selected_candidate(
    candidate: Candidate,
    *,
    category: str,
    rank: int,
    examples_dir: Path,
    args: argparse.Namespace,
    max_agents: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    safe_id = candidate.example_id.replace("/", "_")
    directory = examples_dir / category / f"{rank:02d}_{safe_id}"
    directory.mkdir(parents=True, exist_ok=True)
    payload = candidate.payload

    np.savez_compressed(
        directory / "rollout_arrays.npz",
        start=payload["start"],
        target=payload["target"],
        pred=payload["pred"],
        target_presence=payload["target_presence"],
        predicted_presence=payload["pred_presence"],
        observed=payload["observed"],
        slot=payload["slot"],
        valid=payload["valid"],
        dynamic_mask=payload["dynamic_mask"],
        position_mask=payload["position_mask"],
        health_mask=payload["health_mask"],
        actions=payload["actions"],
    )
    plot_overview(candidate, directory / "overview_h1_h5_h15.png", args, max_agents)
    plot_trajectory_overlay(candidate, directory / "trajectory_overlay_h1_h15.png", args, max_agents)
    plot_feature_error(candidate, directory / "feature_error_by_horizon.png")

    summary = dict(candidate.metrics)
    summary.update(
        {
            "category": category,
            "category_rank": rank,
            "category_score": candidate.score,
            "category_description": CATEGORY_DESCRIPTIONS[category],
            "example_dir": str(directory),
            "overview_png": str(directory / "overview_h1_h5_h15.png"),
            "trajectory_png": str(directory / "trajectory_overlay_h1_h15.png"),
            "feature_error_png": str(directory / "feature_error_by_horizon.png"),
        }
    )
    (directory / "metrics.json").write_text(json.dumps(jsonable(summary), indent=2) + "\n", encoding="utf-8")

    entity_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    horizon_points = choose_horizon_indices(payload["target"].shape[0])
    for label, h in horizon_points:
        for entity in range(payload["target"].shape[1]):
            faction = "ally" if entity < max_agents else "enemy"
            display = f"A{entity}" if faction == "ally" else f"E{entity - max_agents}"
            x_indices = args.ally_position_indices if faction == "ally" else args.enemy_position_indices
            hp_indices = args.ally_health_indices if faction == "ally" else args.enemy_health_indices
            target_x = payload["target"][h, entity, x_indices[0]] if len(x_indices) >= 2 and x_indices[0] < payload["target"].shape[-1] else np.nan
            target_y = payload["target"][h, entity, x_indices[1]] if len(x_indices) >= 2 and x_indices[1] < payload["target"].shape[-1] else np.nan
            pred_x = payload["pred"][h, entity, x_indices[0]] if len(x_indices) >= 2 and x_indices[0] < payload["pred"].shape[-1] else np.nan
            pred_y = payload["pred"][h, entity, x_indices[1]] if len(x_indices) >= 2 and x_indices[1] < payload["pred"].shape[-1] else np.nan
            hp_index = hp_indices[0] if hp_indices else 0
            target_hp = payload["target"][h, entity, hp_index] if hp_index < payload["target"].shape[-1] else np.nan
            pred_hp = payload["pred"][h, entity, hp_index] if hp_index < payload["pred"].shape[-1] else np.nan
            entity_rows.append(
                {
                    "category": category,
                    "category_rank": rank,
                    "example_id": candidate.example_id,
                    "horizon": h + 1,
                    "horizon_label": label,
                    "entity": display,
                    "slot_index": entity,
                    "faction": faction,
                    "structurally_valid": bool(payload["slot"][h, entity]),
                    "target_present": bool(payload["target_presence"][h, entity] >= 0.5),
                    "predicted_presence": float(payload["pred_presence"][h, entity]),
                    "observed": bool(payload["observed"][h, entity] >= 0.5),
                    "target_x": target_x,
                    "predicted_x": pred_x,
                    "target_y": target_y,
                    "predicted_y": pred_y,
                    "position_error_l2": float(np.hypot(pred_x - target_x, pred_y - target_y)) if np.isfinite([target_x, target_y, pred_x, pred_y]).all() else np.nan,
                    "target_hp": target_hp,
                    "predicted_hp": pred_hp,
                    "hp_abs_error": abs(float(pred_hp - target_hp)) if np.isfinite([target_hp, pred_hp]).all() else np.nan,
                }
            )
            for feature in range(payload["target"].shape[-1]):
                if payload["dynamic_mask"][h, entity, feature] <= 0:
                    continue
                if faction == "ally":
                    names = {0: "hp", 1: "cooldown_or_energy", 2: "dx", 3: "dy", 4: "shield"}
                else:
                    names = {0: "hp", 1: "dx", 2: "dy"}
                actual = float(payload["target"][h, entity, feature])
                predicted = float(payload["pred"][h, entity, feature])
                start_value = float(payload["start"][entity, feature])
                feature_rows.append(
                    {
                        "category": category,
                        "category_rank": rank,
                        "example_id": candidate.example_id,
                        "horizon": h + 1,
                        "entity": display,
                        "slot_index": entity,
                        "faction": faction,
                        "feature_index": feature,
                        "feature": names.get(feature, f"dynamic_{feature}"),
                        "actual": actual,
                        "predicted": predicted,
                        "abs_error": abs(predicted - actual),
                        "start_value": start_value,
                        "target_change_from_start": actual - start_value,
                        "predicted_change_from_start": predicted - start_value,
                    }
                )
    write_csv(directory / "entity_h1_h5_h15.csv", entity_rows)
    write_csv(directory / "feature_h1_h5_h15.csv", feature_rows)
    return summary, entity_rows, feature_rows


def create_handoff_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "UPLOAD_THIS_BACK_TO_CHAT.zip"
    include_names = {
        "run_config.json",
        "all_examples.csv",
        "horizon_metrics.csv",
        "selections.csv",
        "selected_entities.csv",
        "selected_features.csv",
        "rollout_gallery.xlsx",
        "README_RESULTS.md",
    }
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(out_dir.rglob("*")):
            if not path.is_file() or path == zip_path:
                continue
            relative = path.relative_to(out_dir)
            if relative.parts[0] in {"examples", "charts"} or relative.name in include_names:
                archive.write(path, arcname=str(relative))
    return zip_path


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if args.horizon < 15:
        raise SystemExit("Use --horizon 15 or larger; H1/H5/H15 comparisons are required")

    checkpoint = load_checkpoint(checkpoint_path)
    config = dict(base.get_config(checkpoint))
    training_horizon = int(config.get("rollout_horizon", checkpoint.get("rollout_horizon", 5)))
    rollout_window = int(config.get("rollout_window", 20))
    target_mode = "full"
    device = base.resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")

    print(f"[checkpoint] {checkpoint_path}", flush=True)
    print(f"[manifest]   {manifest_path} split={args.split}", flush=True)
    print(f"[rollout]    window={rollout_window} horizon={args.horizon} training_horizon={training_horizon}", flush=True)
    print(f"[device]     {device} amp={args.amp}", flush=True)

    dataset = base.build_dataset(
        manifest=str(manifest_path),
        split=args.split,
        resolved_config=config,
        window_mode="sequential",
        samples_per_epoch=None,
        enemy_visibility_mask=None,
        enemy_sight_range=None,
        eval_rollout_horizon=int(args.horizon),
    )
    total_items = len(dataset)
    requested_items = args.max_items or (int(args.max_batches) * int(args.batch_size))
    selected_indices = resolve_indices(total_items, min(total_items, requested_items))
    indexed_dataset = IndexedDataset(dataset, selected_indices)
    loader = DataLoader(
        indexed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    print(
        f"[dataset] items={total_items} sampled_items={len(selected_indices)} "
        f"potential_rollout_starts={len(selected_indices) * rollout_window}",
        flush=True,
    )

    model = base.build_model(checkpoint, dataset, device)
    memory_module = base.build_memory_module(checkpoint, dataset, device)
    model.eval(); memory_module.eval()
    r2_latent_normalize = bool(config.get("r2_latent_normalize", False))
    max_agents = int(dataset.metadata.max_agents)

    horizon = int(args.horizon)
    h1 = 0
    h5 = min(4, horizon - 1)
    h15 = horizon - 1
    group_names = [
        "dynamic", "ally_dynamic", "enemy_dynamic", "visible_enemy_dynamic",
        "hidden_enemy_dynamic", "position", "health",
    ]
    aggregate_num = {name: torch.zeros(horizon, dtype=torch.float64) for name in group_names}
    aggregate_den = {name: torch.zeros(horizon, dtype=torch.float64) for name in group_names}
    presence_fp = torch.zeros(horizon, dtype=torch.float64)
    presence_fn = torch.zeros(horizon, dtype=torch.float64)
    presence_valid = torch.zeros(horizon, dtype=torch.float64)

    all_rows: list[dict[str, Any]] = []
    pool_size = max(args.top_k * args.candidate_pool_multiplier, args.top_k)
    pools = CandidatePools(pool_size)
    start_time = time.time()
    examples_seen = 0

    for batch_index, batch_cpu in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        batch = base.to_device(batch_cpu, device)
        with torch.inference_mode():
            if device.type == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()
            with autocast_ctx:
                outputs = base.rollout_outputs(
                    model,
                    memory_module,
                    batch,
                    rollout_window,
                    horizon,
                    target_mode,
                    action_mode="correct",
                    zero_history_memory=False,
                    mask_mode="predicted",
                    presence_threshold=float(args.presence_threshold),
                    r2_latent_normalize=r2_latent_normalize,
                )
            masks = base.build_rollout_feature_masks(dataset, batch, outputs)

        pred = outputs["decoded"].float()
        target = outputs["target_entity"].float()
        abs_error = (pred - target).abs()
        dynamic = masks["dynamic"].float()
        ally_dynamic = masks["ally_dynamic"].float()
        enemy_dynamic = masks["enemy_dynamic"].float()

        entities = target.shape[-2]
        token_dim = target.shape[-1]
        enemy_template = torch.zeros((1, 1, 1, entities, 1), device=device, dtype=target.dtype)
        enemy_template[..., max_agents:, :] = 1.0
        visible_enemy_dynamic = masks["visible_dynamic"].float() * enemy_template
        hidden_enemy_dynamic = masks["hidden_dynamic"].float() * enemy_template
        position_template = make_feature_template(
            entities=entities, token_dim=token_dim, max_agents=max_agents,
            ally_indices=args.ally_position_indices,
            enemy_indices=args.enemy_position_indices,
            device=device, dtype=target.dtype,
        )
        health_template = make_feature_template(
            entities=entities, token_dim=token_dim, max_agents=max_agents,
            ally_indices=args.ally_health_indices,
            enemy_indices=args.enemy_health_indices,
            device=device, dtype=target.dtype,
        )
        position_mask = dynamic * position_template
        health_mask = dynamic * health_template
        metric_masks = {
            "dynamic": dynamic,
            "ally_dynamic": ally_dynamic,
            "enemy_dynamic": enemy_dynamic,
            "visible_enemy_dynamic": visible_enemy_dynamic,
            "hidden_enemy_dynamic": hidden_enemy_dynamic,
            "position": position_mask,
            "health": health_mask,
        }
        per_example = {name: masked_mean_per_example(abs_error, mask) for name, mask in metric_masks.items()}

        full_sequence = batch.get("target_entity_seq", batch["entity_seq"]).float()
        start = full_sequence[:, :rollout_window]
        start_expanded = start.unsqueeze(2).expand_as(target)
        target_delta_abs = (target - start_expanded).abs()
        pred_delta_abs = (pred - start_expanded).abs()
        target_change = masked_mean_per_example(target_delta_abs, dynamic)
        pred_change = masked_mean_per_example(pred_delta_abs, dynamic)
        health_target_change = masked_mean_per_example(target_delta_abs, health_mask)

        pred_presence_score = torch.sigmoid(outputs["presence_logits"].float())
        pred_presence = pred_presence_score >= float(args.presence_threshold)
        target_presence = outputs["target_entity_mask"] >= 0.5
        slot_valid = (outputs["entity_slot_mask"] >= 0.5) & (outputs["valid_mask"].unsqueeze(-1) >= 0.5)
        fp = (pred_presence & (~target_presence) & slot_valid).sum(dim=-1).float()
        fn = ((~pred_presence) & target_presence & slot_valid).sum(dim=-1).float()
        valid_entities = slot_valid.sum(dim=-1).float()

        health_out_of_range = (
            (((pred < -0.05) | (pred > 1.10)).float() * health_mask).sum(dim=(-1, -2))
            / health_mask.sum(dim=(-1, -2)).clamp_min(1.0)
        )

        observation_seq = batch["observation_mask_seq"] >= 0.5
        observed_start = observation_seq[:, :rollout_window]
        observed_future = outputs["observed_entity_mask"] >= 0.5
        observed_concat = torch.cat([observed_start.unsqueeze(2), observed_future], dim=2)
        changes = observed_concat[:, :, 1:] ^ observed_concat[:, :, :-1]
        enemy_entity = torch.zeros((1, 1, 1, entities), device=device, dtype=torch.bool)
        enemy_entity[..., max_agents:] = True
        visibility_change_count = (changes & enemy_entity).sum(dim=(-1, -2)).float()
        hidden_count_h15 = hidden_enemy_dynamic[:, :, h15].sum(dim=(-1, -2))
        enemy_count_h15 = enemy_dynamic[:, :, h15].sum(dim=(-1, -2))

        for name, mask in metric_masks.items():
            aggregate_num[name] += (abs_error * mask).sum(dim=(0, 1, 3, 4)).detach().double().cpu()
            aggregate_den[name] += mask.sum(dim=(0, 1, 3, 4)).detach().double().cpu()
        presence_fp += fp.sum(dim=(0, 1)).detach().double().cpu()
        presence_fn += fn.sum(dim=(0, 1)).detach().double().cpu()
        presence_valid += valid_entities.sum(dim=(0, 1)).detach().double().cpu()

        bsz, starts = target.shape[:2]
        n_examples = bsz * starts
        examples_seen += n_examples
        episode_index = batch["episode_index"].detach().cpu().numpy().reshape(-1)
        segment_start = batch["segment_start"].detach().cpu().numpy().reshape(-1)
        dataset_item_index = batch["_dataset_item_index"].detach().cpu().numpy().reshape(-1)

        metric_flat = {
            "dynamic_h1": flatten_metric(per_example["dynamic"], h1),
            "dynamic_h5": flatten_metric(per_example["dynamic"], h5),
            "dynamic_h15": flatten_metric(per_example["dynamic"], h15),
            "ally_h1": flatten_metric(per_example["ally_dynamic"], h1),
            "ally_h15": flatten_metric(per_example["ally_dynamic"], h15),
            "enemy_h1": flatten_metric(per_example["enemy_dynamic"], h1),
            "enemy_h15": flatten_metric(per_example["enemy_dynamic"], h15),
            "visible_enemy_h15": flatten_metric(per_example["visible_enemy_dynamic"], h15),
            "hidden_enemy_h15": flatten_metric(per_example["hidden_enemy_dynamic"], h15),
            "position_h1": flatten_metric(per_example["position"], h1),
            "position_h5": flatten_metric(per_example["position"], h5),
            "position_h15": flatten_metric(per_example["position"], h15),
            "health_h1": flatten_metric(per_example["health"], h1),
            "health_h15": flatten_metric(per_example["health"], h15),
            "target_change_h15": flatten_metric(target_change, h15),
            "pred_change_h15": flatten_metric(pred_change, h15),
            "health_target_change_h15": flatten_metric(health_target_change, h15),
            "fp_h15": flatten_metric(fp, h15),
            "fn_h15": flatten_metric(fn, h15),
            "valid_entities_h15": flatten_metric(valid_entities, h15),
            "health_oor_h15": flatten_metric(health_out_of_range, h15),
            "visibility_changes": visibility_change_count.detach().cpu().numpy().reshape(-1),
            "hidden_count_h15": hidden_count_h15.detach().cpu().numpy().reshape(-1),
            "enemy_count_h15": enemy_count_h15.detach().cpu().numpy().reshape(-1),
        }

        row_cache: dict[tuple[int, int], dict[str, Any]] = {}
        payload_cache: dict[tuple[int, int], dict[str, Any]] = {}

        def build_row(local_flat: int) -> dict[str, Any]:
            b = local_flat // starts
            p = local_flat % starts
            key = (b, p)
            if key in row_cache:
                return row_cache[key]
            absolute_start = int(segment_start[b]) + int(p)
            example_id = f"ep{int(episode_index[b]):05d}_t{absolute_start:04d}_item{int(dataset_item_index[b]):06d}"
            target_change_value = float(metric_flat["target_change_h15"][local_flat])
            pred_change_value = float(metric_flat["pred_change_h15"][local_flat])
            copy_ratio = pred_change_value / max(target_change_value, 1e-8)
            valid_count = max(float(metric_flat["valid_entities_h15"][local_flat]), 1.0)
            row = {
                "example_id": example_id,
                "batch_index": batch_index,
                "dataset_item_index": int(dataset_item_index[b]),
                "episode_index": int(episode_index[b]),
                "segment_start": int(segment_start[b]),
                "rollout_start_offset": int(p),
                "absolute_start_timestep": absolute_start,
                "dynamic_mae_h1": float(metric_flat["dynamic_h1"][local_flat]),
                "dynamic_mae_h5": float(metric_flat["dynamic_h5"][local_flat]),
                "dynamic_mae_h15": float(metric_flat["dynamic_h15"][local_flat]),
                "h15_minus_h5": float(metric_flat["dynamic_h15"][local_flat] - metric_flat["dynamic_h5"][local_flat]),
                "h15_minus_h1": float(metric_flat["dynamic_h15"][local_flat] - metric_flat["dynamic_h1"][local_flat]),
                "ally_dynamic_mae_h1": float(metric_flat["ally_h1"][local_flat]),
                "ally_dynamic_mae_h15": float(metric_flat["ally_h15"][local_flat]),
                "enemy_dynamic_mae_h1": float(metric_flat["enemy_h1"][local_flat]),
                "enemy_dynamic_mae_h15": float(metric_flat["enemy_h15"][local_flat]),
                "visible_enemy_dynamic_mae_h15": float(metric_flat["visible_enemy_h15"][local_flat]),
                "hidden_enemy_dynamic_mae_h15": float(metric_flat["hidden_enemy_h15"][local_flat]),
                "position_mae_h1": float(metric_flat["position_h1"][local_flat]),
                "position_mae_h5": float(metric_flat["position_h5"][local_flat]),
                "position_mae_h15": float(metric_flat["position_h15"][local_flat]),
                "health_mae_h1": float(metric_flat["health_h1"][local_flat]),
                "health_mae_h15": float(metric_flat["health_h15"][local_flat]),
                "target_dynamic_change_h15": target_change_value,
                "predicted_dynamic_change_h15": pred_change_value,
                "predicted_to_target_change_ratio_h15": copy_ratio,
                "target_health_change_h15": float(metric_flat["health_target_change_h15"][local_flat]),
                "presence_false_positive_h15": float(metric_flat["fp_h15"][local_flat]),
                "presence_false_negative_h15": float(metric_flat["fn_h15"][local_flat]),
                "presence_error_rate_h15": float((metric_flat["fp_h15"][local_flat] + metric_flat["fn_h15"][local_flat]) / valid_count),
                "health_out_of_range_fraction_h15": float(metric_flat["health_oor_h15"][local_flat]),
                "natural_enemy_visibility_changes_h1_h15": float(metric_flat["visibility_changes"][local_flat]),
                "hidden_enemy_dynamic_coordinate_count_h15": float(metric_flat["hidden_count_h15"][local_flat]),
                "enemy_dynamic_coordinate_count_h15": float(metric_flat["enemy_count_h15"][local_flat]),
            }
            row_cache[key] = row
            return row

        def build_payload(local_flat: int) -> dict[str, Any]:
            b = local_flat // starts
            p = local_flat % starts
            key = (b, p)
            if key in payload_cache:
                return payload_cache[key]
            payload = {
                "start": start[b, p].detach().float().cpu().numpy(),
                "target": target[b, p].detach().float().cpu().numpy(),
                "pred": pred[b, p].detach().float().cpu().numpy(),
                "target_presence": outputs["target_entity_mask"][b, p].detach().float().cpu().numpy(),
                "pred_presence": pred_presence_score[b, p].detach().float().cpu().numpy(),
                "observed": outputs["observed_entity_mask"][b, p].detach().float().cpu().numpy(),
                "slot": outputs["entity_slot_mask"][b, p].detach().float().cpu().numpy(),
                "valid": outputs["valid_mask"][b, p].detach().float().cpu().numpy(),
                "dynamic_mask": dynamic[b, p].detach().float().cpu().numpy(),
                "position_mask": position_mask[b, p].detach().float().cpu().numpy(),
                "health_mask": health_mask[b, p].detach().float().cpu().numpy(),
                "actions": batch["action_seq"][b, p : p + horizon].detach().cpu().numpy(),
            }
            payload_cache[key] = payload
            return payload

        for local_flat in range(n_examples):
            all_rows.append(build_row(local_flat))

        # Calculate category scores for this batch and only materialise strong local candidates.
        eps = 1e-8
        dyn1 = metric_flat["dynamic_h1"]
        dyn5 = metric_flat["dynamic_h5"]
        dyn15 = metric_flat["dynamic_h15"]
        pos1 = metric_flat["position_h1"]
        pos5 = metric_flat["position_h5"]
        pos15 = metric_flat["position_h15"]
        health15 = metric_flat["health_h15"]
        health_delta15 = metric_flat["health_target_change_h15"]
        ally15 = metric_flat["ally_h15"]
        enemy15 = metric_flat["enemy_h15"]
        hidden15 = metric_flat["hidden_enemy_h15"]
        target_change15 = metric_flat["target_change_h15"]
        pred_change15 = metric_flat["pred_change_h15"]
        presence_errors = metric_flat["fp_h15"] + metric_flat["fn_h15"]
        valid_count = np.maximum(metric_flat["valid_entities_h15"], 1.0)
        health_oor = metric_flat["health_oor_h15"]
        vis_changes = metric_flat["visibility_changes"]

        score_arrays: dict[str, np.ndarray] = {
            "good_eventful": target_change15 * 2.0 - dyn15 - np.maximum(dyn15 - dyn5, 0.0) * 0.25,
            "late_rollout_drift": np.maximum(dyn15 - dyn5, 0.0) + 0.5 * np.maximum(dyn5 - dyn1, 0.0),
            "position_drift": pos15 + np.maximum(pos15 - pos5, 0.0) + 0.25 * np.maximum(pos5 - pos1, 0.0),
            "health_or_damage_miss": health15 * (1.0 + 10.0 * health_delta15),
            "enemy_tracking_failure": np.maximum(enemy15 - ally15, 0.0) + 0.5 * np.nan_to_num(hidden15, nan=0.0) + 0.25 * enemy15,
            "presence_lifecycle_failure": presence_errors / valid_count + 0.1 * dyn15,
            "copying_dynamic_change": target_change15 * np.maximum(1.0 - pred_change15 / np.maximum(target_change15, eps), 0.0) + np.maximum(target_change15 - pred_change15, 0.0),
            "unstable_overshoot": np.maximum(pred_change15 - target_change15, 0.0) + 0.5 * health_oor + 0.2 * dyn15,
            "visibility_transition_failure": np.nan_to_num(hidden15, nan=0.0) + np.maximum(enemy15 - metric_flat["enemy_h1"], 0.0) + vis_changes * 0.005,
        }
        eligibility: dict[str, np.ndarray] = {
            "good_eventful": target_change15 >= float(args.good_min_change),
            "late_rollout_drift": dyn15 > dyn5,
            "position_drift": np.isfinite(pos15),
            "health_or_damage_miss": health_delta15 >= float(args.health_change_threshold),
            "enemy_tracking_failure": metric_flat["enemy_count_h15"] > 0,
            "presence_lifecycle_failure": presence_errors > 0,
            "copying_dynamic_change": target_change15 >= float(args.good_min_change),
            "unstable_overshoot": (pred_change15 > target_change15) | (health_oor > 0),
            "visibility_transition_failure": (vis_changes > 0) | (metric_flat["hidden_count_h15"] > 0),
        }

        local_take = min(max(args.top_k * 3, 12), n_examples)
        for category in CATEGORY_ORDER:
            score = np.asarray(score_arrays[category], dtype=np.float64)
            eligible = np.asarray(eligibility[category], dtype=bool) & np.isfinite(score)
            candidate_indices = np.flatnonzero(eligible)
            if candidate_indices.size == 0:
                continue
            if candidate_indices.size > local_take:
                local_scores = score[candidate_indices]
                partition = np.argpartition(local_scores, -local_take)[-local_take:]
                candidate_indices = candidate_indices[partition]
            for local_flat in candidate_indices:
                row = build_row(int(local_flat))
                candidate = Candidate(
                    category=category,
                    score=float(score[local_flat]),
                    example_id=str(row["example_id"]),
                    metrics=row,
                    payload=build_payload(int(local_flat)),
                )
                pools.consider(candidate)

        elapsed = time.time() - start_time
        print(
            f"[batch {batch_index + 1:03d}/{min(args.max_batches, len(loader)):03d}] "
            f"examples={examples_seen:,} elapsed={elapsed / 60:.1f} min",
            flush=True,
        )
        del outputs, masks, pred, target, abs_error
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate horizon table.
    horizon_rows: list[dict[str, Any]] = []
    for h in range(horizon):
        row: dict[str, Any] = {"horizon": h + 1}
        for name in group_names:
            value = aggregate_num[name][h] / aggregate_den[name][h].clamp_min(1.0)
            row[f"{name}_mae"] = float(value.item()) if aggregate_den[name][h] > 0 else float("nan")
        row["presence_false_positive_rate"] = float((presence_fp[h] / presence_valid[h].clamp_min(1.0)).item())
        row["presence_false_negative_rate"] = float((presence_fn[h] / presence_valid[h].clamp_min(1.0)).item())
        row["evaluated_dynamic_coordinate_count"] = float(aggregate_den["dynamic"][h].item())
        horizon_rows.append(row)

    # Select non-duplicated examples across categories where possible.
    selected: list[tuple[str, int, Candidate]] = []
    used_ids: set[str] = set()
    for category in CATEGORY_ORDER:
        ranked = pools.ranked(category)
        chosen: list[Candidate] = []
        for candidate in ranked:
            if candidate.example_id in used_ids:
                continue
            chosen.append(candidate)
            used_ids.add(candidate.example_id)
            if len(chosen) >= args.top_k:
                break
        if len(chosen) < args.top_k:
            for candidate in ranked:
                if candidate in chosen:
                    continue
                chosen.append(candidate)
                if len(chosen) >= args.top_k:
                    break
        for rank, candidate in enumerate(chosen, start=1):
            selected.append((category, rank, candidate))

    examples_dir = out_dir / "examples"
    selection_rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for category, rank, candidate in selected:
        summary, entity_part, feature_part = save_selected_candidate(
            candidate,
            category=category,
            rank=rank,
            examples_dir=examples_dir,
            args=args,
            max_agents=max_agents,
        )
        selection_rows.append(summary)
        entity_rows.extend(entity_part)
        feature_rows.extend(feature_part)

    write_csv(out_dir / "all_examples.csv", all_rows)
    write_csv(out_dir / "horizon_metrics.csv", horizon_rows)
    write_csv(out_dir / "selections.csv", selection_rows)
    write_csv(out_dir / "selected_entities.csv", entity_rows)
    write_csv(out_dir / "selected_features.csv", feature_rows)
    plot_horizon_curves(horizon_rows, out_dir / "charts")

    run_config = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch", checkpoint.get("checkpoint_saved_epoch")),
        "manifest": str(manifest_path),
        "split": args.split,
        "training_rollout_horizon": training_horizon,
        "evaluation_rollout_horizon": horizon,
        "rollout_window": rollout_window,
        "target_mode": target_mode,
        "mask_mode": "predicted",
        "teacher_forcing_after_rollout_start": False,
        "memory_ablation": False,
        "dataset_items_total": total_items,
        "dataset_items_sampled": len(selected_indices),
        "rollout_examples_evaluated": examples_seen,
        "device": str(device),
        "amp": bool(args.amp),
        "seed": args.seed,
        "elapsed_seconds": time.time() - start_time,
        "selected_categories": CATEGORY_DESCRIPTIONS,
        "position_indices": {
            "ally": args.ally_position_indices,
            "enemy": args.enemy_position_indices,
        },
        "health_indices": {
            "ally": args.ally_health_indices,
            "enemy": args.enemy_health_indices,
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(jsonable(run_config), indent=2) + "\n", encoding="utf-8")

    excel_warning = optional_excel(
        out_dir / "rollout_gallery.xlsx",
        {
            "Selections": selection_rows,
            "All_Examples": all_rows,
            "Horizon_Curves": horizon_rows,
            "Selected_Entities": entity_rows,
            "Selected_Features": feature_rows,
        },
    )

    readme_lines = [
        "# Exp40 H15 Rollout Gallery Results",
        "",
        f"Checkpoint: `{checkpoint_path}`",
        f"Held-out rollout examples evaluated: **{examples_seen:,}**",
        "",
        "## Start here",
        "",
        "1. Open `selections.csv` or the `Selections` sheet in `rollout_gallery.xlsx`.",
        "2. Inspect `examples/good_eventful/01_*/overview_h1_h5_h15.png` for the first candidate slide.",
        "3. Inspect each failure-category folder for distinct data-grounded failure slides.",
        "4. Upload `UPLOAD_THIS_BACK_TO_CHAT.zip` for final example selection and presentation analysis.",
        "",
        "## Category meanings",
        "",
    ]
    for category in CATEGORY_ORDER:
        readme_lines.append(f"- **{category}**: {CATEGORY_DESCRIPTIONS[category]}")
    readme_lines.extend([
        "",
        "## Evaluation semantics",
        "",
        "The first rollout state is grounded in the held-out trajectory. Every later state is generated recursively from the model's own prediction using recorded allied actions. Future visibility and future ground-truth presence are not supplied to the predictor.",
    ])
    if excel_warning:
        readme_lines.extend(["", f"Excel note: {excel_warning}"])
    (out_dir / "README_RESULTS.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    zip_path = create_handoff_zip(out_dir)
    print("", flush=True)
    print("=" * 78, flush=True)
    print("EXP40 ROLLOUT GALLERY COMPLETE", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"Upload this file back to ChatGPT: {zip_path}", flush=True)
    if excel_warning:
        print(f"[warning] {excel_warning}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
