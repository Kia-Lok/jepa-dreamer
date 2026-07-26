#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-run-meta", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args()

    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    cfg = OmegaConf.load(config_path)
    tactical = cfg.tactical_mixture

    expected = {
        "num_tactics": 2,
        "duration": 1,
        "residual_scale": 0.25,
        "max_abs_residual_logit": 2.0,
        "eval_confidence_threshold": 0.70,
        "min_selector_mi_normalized": 0.05,
        "base_kl_target": 0.02,
        "base_kl_scale": 0.10,
        "effect_target": 0.002,
        "effect_loss_scale": 0.10,
        "collapse_loss_scale": 0.10,
        "max_residual_to_base": 0.25,
    }
    for key, value in expected.items():
        observed = tactical.get(key)
        if float(observed) != float(value):
            fail(f"config {key}={observed!r}, expected {value!r}")
    if bool(cfg.adaptive_priority.enabled) or bool(cfg.adaptive_priority.map.enabled) or bool(cfg.adaptive_priority.sequence.enabled):
        fail("adaptive priority must be disabled")
    if str(cfg.buffer.scratch_dir) != "replay":
        fail("buffer.scratch_dir must be run-local replay")
    if bool(cfg.validation.run_at_start):
        fail("startup validation must be disabled")
    if int(cfg.validation.every) != 200000:
        fail("validation interval must be 200000")

    policy_path = repo / "external/r2dreamer/tactical_policy.py"
    dreamer_path = repo / "external/r2dreamer/dreamer.py"
    policy_text = policy_path.read_text(encoding="utf-8")
    dreamer_text = dreamer_path.read_text(encoding="utf-8")
    ast.parse(policy_text)
    ast.parse(dreamer_text)

    required_policy = [
        'ARCHITECTURE = "tactical_mixture_v1_2"',
        "raw - raw.mean(dim=-2, keepdim=True)",
        "confidence - threshold",
        '"base_kl_loss"',
        '"action_flip_rate"',
        "min_selector_mi_normalized",
    ]
    missing = [token for token in required_policy if token not in policy_text]
    if missing:
        fail(f"policy contract missing: {missing}")

    required_dreamer = [
        "TACTICAL_MIXTURE_V1_2_CENTERED_TRUST_REGION",
        '"tactical_mixture_v1_2"',
        'tactical.base_kl_scale * base_kl_loss',
        'metrics["tactic/base_kl_mean"]',
        'metrics["tactic/action_flip_rate"]',
    ]
    missing = [token for token in required_dreamer if token not in dreamer_text]
    if missing:
        fail(f"Dreamer v1.2 contract missing: {missing}")

    if args.expected_checkpoint_sha256:
        with args.checkpoint.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != args.expected_checkpoint_sha256:
            fail(f"source checkpoint SHA-256 mismatch: expected={args.expected_checkpoint_sha256}; actual={actual}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("agent_state_dict")
    if not isinstance(state, dict):
        fail("source checkpoint lacks agent_state_dict")
    if any("tactical_policy." in key for key in state):
        fail("source checkpoint already contains tactical parameters")
    source_run_meta = None
    if args.source_run_meta:
        if not args.source_run_meta.is_file():
            fail(f"source run metadata missing: {args.source_run_meta}")
        try:
            json.loads(args.source_run_meta.read_text(encoding="utf-8"))
        except Exception as exc:
            fail(f"source run metadata is not valid JSON: {exc}")
        if args.checkpoint.resolve().parent != args.source_run_meta.resolve().parent:
            fail("checkpoint and run_meta are not from the same source run")
        source_run_meta = str(args.source_run_meta.resolve())

    print(json.dumps({
        "config": str(config_path),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_run_meta": source_run_meta,
        "source_step": checkpoint.get("step"),
        "source_val_macro_win_rate": checkpoint.get("val_macro_win_rate"),
        "checks": {
            "centered_residual": "ok",
            "hard_eval_gate": "ok",
            "selector_mi_floor": "ok",
            "base_policy_kl": "ok",
            "two_tactic_config": "ok",
            "source_lineage": "ok",
        },
    }, indent=2))
    print("[OK] Tactical Mixture v1.2 audit passed")


if __name__ == "__main__":
    main()
