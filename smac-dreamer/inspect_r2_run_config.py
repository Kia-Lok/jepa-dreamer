from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r2_2100_jepa_local.yaml")
    ap.add_argument("--jepa-checkpoint", default="checkpoints/jepa/model.pt")
    ap.add_argument("--run", default=None, help="Optional existing R2 run dir to inspect")
    args = ap.parse_args()

    root = Path.cwd()
    cfg_path = root / args.config
    ckpt_path = root / args.jepa_checkpoint

    print("=== requested config/checkpoint ===")
    print("config:", cfg_path, "exists=", cfg_path.exists())
    print("jepa_checkpoint:", ckpt_path, "exists=", ckpt_path.exists())

    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
        print("steps:", cfg.get("steps", None))
        print("logdir:", cfg.get("logdir", None))
        print("validation.every:", cfg.get("validation", {}).get("every", None))
        print("validation.run_at_start:", cfg.get("validation", {}).get("run_at_start", None))
        print("reward.name:", cfg.get("reward", {}).get("name", None))
        print("world_model.backend:", cfg.get("world_model", {}).get("backend", None))
        print("config world_model.jepa.checkpoint:", cfg.get("world_model", {}).get("jepa", {}).get("checkpoint", None))

    if ckpt_path.exists():
        print("checkpoint size MB:", round(ckpt_path.stat().st_size / 1024 / 1024, 2))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print("checkpoint type:", type(ckpt).__name__)
        if isinstance(ckpt, dict):
            print("checkpoint keys:", sorted(str(k) for k in ckpt.keys())[:50])
            metadata = ckpt.get("metadata", {}) or {}
            resolved = ckpt.get("resolved_config", ckpt.get("config", {})) or {}
            print("metadata max_agents:", metadata.get("max_agents"))
            print("metadata max_enemies:", metadata.get("max_enemies"))
            print("metadata max_actions:", metadata.get("max_actions"))
            print("metadata enemy_visibility_mask:", metadata.get("enemy_visibility_mask"))
            print("resolved anchored_belief_memory:", resolved.get("anchored_belief_memory"))
            print("resolved action_conditioned_memory:", resolved.get("action_conditioned_memory"))
            print("resolved presence_rollout_mode:", resolved.get("presence_rollout_mode"))

    if args.run:
        run = root / args.run
    else:
        runs = sorted((root / "logs" / "r2dreamer").glob("*"), key=lambda p: p.stat().st_mtime) if (root / "logs" / "r2dreamer").exists() else []
        run = runs[-1] if runs else None
    if run and run.exists():
        print("\n=== latest/provided R2 run ===")
        print("run:", run)
        for name in ["run_meta.json", "run_config.json"]:
            p = run / name
            print(f"{name}: exists={p.exists()}")
            if p.exists():
                try:
                    d = json.loads(p.read_text())
                    for key in ["steps", "reward_name", "logdir", "device"]:
                        if key in d:
                            print(f"  {key}: {d[key]}")
                    wm = d.get("world_model", {}) if isinstance(d, dict) else {}
                    if wm:
                        print("  world_model.backend:", wm.get("backend"))
                        print("  world_model.jepa.checkpoint:", wm.get("jepa", {}).get("checkpoint"))
                except Exception as e:
                    print("  could not parse:", e)
        log = run / "train.log"
        if log.exists():
            print("train.log tail grep hints:")
            lines = log.read_text(errors="replace").splitlines()
            needles = ("jepa", "checkpoint", "validation", "reward", "eval", "win", "nan", "error", "traceback")
            hits = [ln for ln in lines if any(n in ln.lower() for n in needles)]
            for ln in hits[-80:]:
                print(" ", ln)


if __name__ == "__main__":
    main()
