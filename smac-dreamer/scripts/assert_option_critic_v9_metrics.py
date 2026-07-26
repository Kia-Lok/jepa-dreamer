#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
parser.add_argument(
    "--max-metric-age",
    type=int,
    default=50_000,
    help="Maximum environment-step age allowed for a required metric.",
)
args = parser.parse_args()
metrics_path = args.run_dir / "metrics.jsonl"
if not metrics_path.is_file():
    raise SystemExit("[FAIL] metrics.jsonl missing")

latest_step = -1
latest_by_key: dict[str, tuple[int, float]] = {}
for line in metrics_path.open(encoding="utf-8"):
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    row_step = row.get("global_step", row.get("step"))
    if not isinstance(row_step, (int, float)):
        continue
    row_step = int(row_step)
    latest_step = max(latest_step, row_step)
    for key, value in row.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            previous = latest_by_key.get(key)
            if previous is None or row_step >= previous[0]:
                latest_by_key[key] = (row_step, float(value))

if latest_step < 0:
    raise SystemExit("[FAIL] no numeric global_step/step rows in metrics.jsonl")


def number(key: str, *, fresh: bool = True) -> float:
    record = latest_by_key.get(key)
    if record is None:
        raise SystemExit(f"[FAIL] missing metric: {key}")
    metric_step, value = record
    if not math.isfinite(value):
        raise SystemExit(f"[FAIL] non-finite metric: {key}={value}")
    if fresh and latest_step - metric_step > args.max_metric_age:
        raise SystemExit(
            f"[FAIL] stale metric: {key} last_step={metric_step}, "
            f"latest_step={latest_step}, max_age={args.max_metric_age}"
        )
    return value


def near(actual: float, expected: float, tolerance: float, label: str) -> None:
    if abs(actual - expected) > tolerance:
        raise SystemExit(
            f"[FAIL] {label}: expected {expected}±{tolerance}, got {actual}"
        )


for key in (
    "train/real_post_mask_invalid_sample_rate",
    "train/imag_post_mask_invalid_sample_rate",
    "train/option/imag_min_duration_violation_rate",
    "train/option/imag_max_duration_violation_rate",
    "train/option/imag_change_without_boundary_rate",
    "train/option/real_min_duration_violation_rate",
    "train/option/real_max_duration_violation_rate",
    "train/option/real_change_without_boundary_rate",
    "train/option/source_manager_group_high_confidence_flip_rate",
):
    value = number(key)
    if abs(value) > 1.0e-8:
        raise SystemExit(f"[FAIL] invariant violation: {key}={value}")

near(number("train/option/imag_horizon"), 15.0, 0.0, "imagination horizon")
near(number("train/option/world_model_grad_scale"), 0.0, 1.0e-8, "world model scale")
near(number("train/option/termination_blend"), 0.0, 1.0e-8, "termination blend")
near(number("train/option/slot_anchor_floor"), 0.40, 1.0e-8, "slot anchor floor")

# The child worker has exactly one learning schedule. Child action scale is fixed
# at 0.10; only worker_pg_blend ramps. This catches the old accidental square of
# the warm-up fraction.
for index in range(8):
    gate = number(f"train/option/slot_gate_{index}")
    near(gate, 1.0, 1.0e-8, f"slot gate {index}")
    pg = number(f"train/option/slot_pg_blend_{index}")
    near(pg, 1.0, 1.0e-8, f"slot PG identity availability {index}")
    delta = number(f"train/option/slot_delta_scale_{index}")
    near(delta, 0.0 if index < 2 else 0.10, 1.0e-7, f"slot delta scale {index}")

worker_blend = number("train/option/worker_pg_blend")
if latest_step <= 20_000:
    expected_worker = 0.0
elif latest_step >= 150_000:
    expected_worker = 1.0
else:
    expected_worker = (latest_step - 20_000) / 130_000
near(worker_blend, expected_worker, 0.02, "worker PG schedule")

manager_blend = number("train/option/manager_pg_blend")
if latest_step <= 100_000:
    expected_manager = 0.0
elif latest_step >= 300_000:
    expected_manager = 1.0
else:
    expected_manager = (latest_step - 100_000) / 200_000
near(manager_blend, expected_manager, 0.02, "manager PG schedule")

consistency_blend = number("train/option/critic_consistency_blend")
near(consistency_blend, max(0.0, 1.0 - worker_blend), 0.02, "critic consistency schedule")
number("train/option/critic_consistency_loss")

for key in (
    "train/option/real_source_interrupt_rate",
    "train/option/imag_source_interrupt_rate",
    "train/option/source_policy_kl_mean",
    "train/option/source_policy_kl_tail",
    "train/option/real_source_policy_kl_mean",
    "train/option/real_source_policy_kl_tail",
    "train/option/worker_trainable_child_fraction",
):
    number(key)

if number("train/option/source_policy_kl_tail") > 0.10:
    raise SystemExit("[FAIL] imagined source-policy tail KL exceeded 0.10")
if number("train/option/real_source_policy_kl_tail") > 0.10:
    raise SystemExit("[FAIL] real source-policy tail KL exceeded 0.10")
if number("train/option/real_source_high_confidence_action_flip_rate") > 0.01:
    raise SystemExit("[FAIL] high-confidence source action flip rate exceeded 1%")

real_usage = []
manager_usage = []
for index in range(8):
    real = number(f"train/option/real_usage_{index}")
    manager = number(f"train/option/usage_{index}")
    if not 0.0 <= real <= 1.0:
        raise SystemExit(f"[FAIL] real_usage_{index}={real}")
    if not 0.0 <= manager <= 1.0:
        raise SystemExit(f"[FAIL] manager usage_{index}={manager}")
    real_usage.append(real)
    manager_usage.append(manager)
if abs(sum(real_usage) - 1.0) > 1.0e-3:
    raise SystemExit(f"[FAIL] real option usage does not sum to one: {sum(real_usage)}")
if abs(sum(manager_usage) - 1.0) > 1.0e-3:
    raise SystemExit(f"[FAIL] manager option usage does not sum to one: {sum(manager_usage)}")

# The conditional source-anchor probability has a mathematical minimum of
# (1-unimix)*0.40 + unimix*0.25 = 0.3985. Allow logging noise but not a broken
# probability transform.
for group, indices in enumerate(((0, 2, 4, 6), (1, 3, 5, 7))):
    group_mass = sum(manager_usage[i] for i in indices)
    if group_mass <= 1.0e-8:
        raise SystemExit(f"[FAIL] manager group {group} has zero probability mass")
    anchor_ratio = manager_usage[group] / group_mass
    if anchor_ratio < 0.39:
        raise SystemExit(
            f"[FAIL] group {group} source-anchor ratio fell below floor: {anchor_ratio}"
        )

# Before manager learning, the fixed floor yields about 45% total child identity
# coverage. A wide finite-sample band catches starvation and the unsafe old 75%
# child-dominant prior without overfitting this assertion to one replay batch.
child_fraction = number("train/option/worker_trainable_child_fraction")
if latest_step < 100_000 and not 0.25 <= child_fraction <= 0.65:
    raise SystemExit(
        "[FAIL] pre-manager child identity coverage is inconsistent with the "
        f"anchor-safe prior: {child_fraction}"
    )

print(f"[OK] Option-Critic v9 runtime invariants passed at step {latest_step}")
print(f"worker_pg_blend={worker_blend:.6g}")
print(f"manager_pg_blend={manager_blend:.6g}")
print(f"critic_consistency_blend={consistency_blend:.6g}")
print(f"worker_trainable_child_fraction={child_fraction:.6g}")
for index, value in enumerate(real_usage):
    print(f"real_usage_{index}={value:.6g}")
