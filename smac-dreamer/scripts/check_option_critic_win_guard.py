#!/usr/bin/env python3
"""One-shot macro-win regression guard for corrected Option-Critic runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

CANDIDATE_KEYS = (
    "val/macro_win_rate",
    "validation/macro_win_rate",
    "val_macro_win_rate",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--max-regression", type=float, default=0.03)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False
    )
    source = float(checkpoint.get("val_macro_win_rate", -1.0))
    if source < 0.0:
        raise SystemExit("[FAIL] source checkpoint lacks val_macro_win_rate")

    metrics = args.run / "metrics.jsonl"
    if not metrics.is_file():
        raise SystemExit(f"[FAIL] missing {metrics}")
    latest = None
    latest_step = -1
    with metrics.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = next((row[k] for k in CANDIDATE_KEYS if k in row), None)
            if isinstance(value, (int, float)):
                latest = float(value)
                latest_step = int(row.get("global_step", row.get("step", latest_step)))
    if latest is None:
        raise SystemExit("[WAIT] no macro validation win-rate metric yet")

    floor = source - float(args.max_regression)
    if latest < floor:
        raise SystemExit(
            f"[FAIL] macro win regression: latest={latest:.4f} at step={latest_step}, "
            f"source={source:.4f}, floor={floor:.4f}"
        )
    print(
        f"[OK] macro win guard: latest={latest:.4f}, source={source:.4f}, "
        f"regression={source-latest:+.4f}, step={latest_step}"
    )


if __name__ == "__main__":
    main()
