#!/usr/bin/env python3
"""Report checkpoint step and the largest step found in source-run JSONL logs.

This is diagnostic, not an automatic truth oracle. Old checkpoint step values in
this branch were derived from replay size; once replay capacity was reached they
could stop tracking actual environment progress. Prefer a trusted W&B/log value
when it is larger and clearly belongs to the same run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import torch


def collect_steps(value: Any, out: list[int], parent_key: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if key_lower in {"step", "global_step", "env_step", "train_step"}:
                try:
                    numeric = int(item)
                except (TypeError, ValueError):
                    pass
                else:
                    if numeric >= 0:
                        out.append(numeric)
            collect_steps(item, out, key_lower)
    elif isinstance(value, list):
        for item in value:
            collect_steps(item, out, parent_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir")
    args = parser.parse_args()

    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve()
    run_dir = pathlib.Path(args.run_dir).expanduser().resolve() if args.run_dir else checkpoint.parent
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_step = int(ckpt.get("step", 0))

    found: list[tuple[int, str]] = []
    for path in sorted(run_dir.rglob("*.jsonl")):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    values: list[int] = []
                    collect_steps(obj, values)
                    if values:
                        found.append((max(values), str(path)))
        except OSError:
            continue

    max_logged = max((step for step, _ in found), default=None)
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "run_dir": str(run_dir),
        "max_jsonl_step": max_logged,
        "jsonl_sources": sorted({src for _, src in found}),
        "recommended_override": (
            max_logged if max_logged is not None and max_logged > checkpoint_step else None
        ),
    }
    print(json.dumps(report, indent=2))
    if report["recommended_override"] is not None:
        print(
            "[WARN] logs contain a larger step than the checkpoint. Verify it in "
            "W&B/source logs, then export RESUME_START_STEP to that trusted value."
        )
    else:
        print(
            "[INFO] no larger JSONL step was found. The checkpoint step remains the "
            "available lower-bound unless W&B or another source shows otherwise."
        )


if __name__ == "__main__":
    main()
