#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "external" / "r2dreamer", ROOT / "external" / "smaclite"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

from smacdreamer.jepa.checkpoint import sha256_file


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a pretrained SMAC-JEPA checkpoint without training.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="R2 config path for human traceability")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    path = pathlib.Path(args.checkpoint)
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=args.device)
    if not isinstance(ckpt, dict):
        raise SystemExit("checkpoint is not a dict")
    cfg = ckpt.get("resolved_config", ckpt.get("config", {}))
    meta = ckpt.get("metadata", {})
    print("checkpoint:", path)
    print("sha256:", sha256_file(path))
    print("keys:", sorted(ckpt.keys()))
    print("metadata:")
    print(json.dumps(meta, indent=2, sort_keys=True, default=str))
    print("resolved_config:")
    print(json.dumps(cfg, indent=2, sort_keys=True, default=str))
    print("memory_type:", "action_conditioned" if cfg.get("action_conditioned_memory") else "entity_gru")
    print("action_dimension:", meta.get("n_actions"))
    print("target_mode:", cfg.get("target_mode"))
    print("rollout_horizon:", cfg.get("rollout_horizon"))
    print("latent_normalization:", cfg.get("latent_normalization", cfg.get("normalize_latents")))
    print("model_parameters:", sum(v.numel() for v in ckpt.get("model_state", {}).values() if hasattr(v, "numel")))
    print("memory_parameters:", sum(v.numel() for v in ckpt.get("memory_module_state", {}).values() if hasattr(v, "numel")))


if __name__ == "__main__":
    main()
