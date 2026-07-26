"""Held-out evaluation for the multimap R2-Dreamer × SMAClite pipeline.

Loads a checkpoint and evaluates deterministically on the HELD-OUT TEST maps only,
reporting per-map and per-family win rate + ORIGINAL (unshaped) return, with:
  * per-map Wilson confidence intervals on win rate (with n_episodes), and
  * an ACROSS-MAP confidence interval as the HEADLINE (each map's win rate is one sample),
    because between-map variance dominates with a small test set (~10 maps).

Asserts no TRAIN map is evaluated. Writes a JSON report.

Eval reward is the ORIGINAL SMAClite reward (smaclite_default) regardless of the training
reward, so the metric is comparable to baselines. Action selection is deterministic
(actor mode); the env's action sanitiser is the final safety net (policy-side eval masking
is a separate, later stage).

Usage (smac-r2 conda env, project root):
    python scripts\\evaluate_multimap.py --config configs\\multimap.yaml --checkpoint logs\\r2dreamer\\multimap\\latest.pt
    python scripts\\evaluate_multimap.py --config configs\\multimap.yaml --checkpoint latest.pt --episodes-per-map 16
"""

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (
    str(ROOT / "src"),
    str(ROOT / "external" / "r2dreamer"),
    str(ROOT / "external" / "smaclite"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from omegaconf import OmegaConf

import tools
from dreamer import Dreamer
from smacdreamer.envs.map_discovery import discover, SplitSpec
from smacdreamer.r2dreamer_factory import make_smaclite_multimap_env
from smacdreamer.evaluation import evaluate_heldout, DEFAULT_FIXED_SEEDS
from train_r2dreamer_smaclite_debug import make_config as _make_debug_config
# Reuse the recursive device propagation so a GPU eval sets EVERY device field (buffer,
# encoder, all heads), not just the three top-level ones — same fix as the training script.
from train_r2dreamer_smaclite_multimap import _propagate_device


def _resolve_seeds(args, cfg) -> list:
    """Fixed eval seeds: --seeds CLI > cfg.eval.fixed_seeds > DEFAULT_FIXED_SEEDS."""
    if args.seeds:
        return [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    cfg_seeds = cfg.eval.get("fixed_seeds") if cfg.get("eval") else None
    if cfg_seeds:
        return [int(s) for s in OmegaConf.to_container(cfg_seeds, resolve=True)]
    return list(DEFAULT_FIXED_SEEDS)


def main():
    ap = argparse.ArgumentParser(description="Held-out multimap evaluation (map × fixed seeds)")
    ap.add_argument("--config", default="configs/multimap.yaml")
    ap.add_argument("--checkpoint", required=True, help="path to latest.pt")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated fixed eval seeds (overrides cfg.eval/validation seeds)")
    ap.add_argument("--split", default="validation",
                    help="explicit-folder dataset split to evaluate: validation | blind_iid | "
                         "blind_compositional (ignored for legacy ratio-split configs)")
    ap.add_argument("--episodes-per-map", type=int, default=None,
                    help="DEPRECATED/ignored: eval is now map × fixed seeds, not a worker count")
    ap.add_argument("--output", default=None, help="JSON report path")
    args = ap.parse_args()

    if args.episodes_per_map is not None:
        print("[warn] --episodes-per-map is deprecated and ignored; eval iterates map × fixed "
              "seeds. Use --seeds or cfg.eval.fixed_seeds.")

    cfg_path = (ROOT / args.config) if not pathlib.Path(args.config).is_absolute() else pathlib.Path(args.config)
    cfg = OmegaConf.load(str(cfg_path))
    device = str(cfg.device)
    seeds = _resolve_seeds(args, cfg)

    # Reconstruct the EXACT training obs_mode + model dims from the checkpoint's sidecar
    # (run_meta.json), so the rebuilt model matches the checkpoint regardless of --config.
    meta_path = pathlib.Path(args.checkpoint).resolve().parent / "run_meta.json"
    run_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    obs_mode = str(run_meta.get("obs_mode",
                                cfg.observation.mode if cfg.get("observation") else "flat"))
    if obs_mode not in ("flat", "structured"):
        sys.exit(f"unsupported obs_mode {obs_mode!r}")
    _dim = lambda k: int(run_meta.get(k, cfg.get(k)))
    units, deter = _dim("units"), _dim("deter")
    batch_size, batch_length, imag_horizon = _dim("batch_size"), _dim("batch_length"), _dim("imag_horizon")
    print(f"Reconstruction: obs_mode={obs_mode} units={units} deter={deter} "
          f"(from {'run_meta.json' if run_meta else '--config'})")

    # Maps to evaluate: explicit-folder split (validation / blind_iid / blind_compositional)
    # using the EXACT training padding (from run_meta), or the legacy ratio held-out split.
    if cfg.get("maps"):
        from smacdreamer.envs.map_discovery import scan_folder_entries
        from smacdreamer.envs.padding import PaddingDims
        split_folder = cfg.maps.get(args.split)
        if not split_folder:
            sys.exit(f"--split {args.split!r} not in cfg.maps; available: {list(cfg.maps.keys())}")
        if not run_meta.get("padding"):
            sys.exit("explicit-folder eval needs run_meta.json (with padding) beside the checkpoint")
        p = run_meta["padding"]
        pad_dims = PaddingDims(max_agents=int(p["max_agents"]), max_enemies=int(p["max_enemies"]),
                               max_actions=int(p["max_actions"]), max_obs_size=int(p["max_obs_size"]))
        test_entries = scan_folder_entries(str(split_folder))
        train_names = set()   # separate folders -> inherently disjoint from train
        eval_tag = f"{run_meta.get('dataset_tag', 'dataset')}_{args.split}"
        print(f"Evaluating split '{args.split}': {split_folder} ({len(test_entries)} maps)")
    else:
        train_entries, test_entries, pad_dims = discover(
            str(cfg.maps_folder),
            SplitSpec(**OmegaConf.to_container(cfg.split, resolve=True)),
            padding_override=OmegaConf.to_container(cfg.padding, resolve=True) if cfg.get("padding") else None,
            verbose=True,
            isolate_probe=True,   # subprocess-isolated probe so large discovery doesn't OOM
            obs_mode=obs_mode,
        )
        train_names = {e.name for e in train_entries}
        eval_tag = pathlib.Path(str(cfg.maps_folder)).name
    if not test_entries:
        sys.exit("No maps to evaluate.")

    # Build the agent with the SAME obs/action shape the model was trained with: construct a
    # one-map env to read the spaces, then load the checkpoint.
    probe = make_smaclite_multimap_env(
        [test_entries[0]], pad_dims, "fixed", 0, 0, "smaclite_default", {},
        float(cfg.gamma), int(cfg.max_episode_steps), obs_mode,
    )
    obs_space, act_space = probe.observation_space, probe.action_space

    config = _make_debug_config(argparse.Namespace(
        steps=1, batch_size=batch_size, batch_length=batch_length,
        units=units, deter=deter, imag_horizon=imag_horizon,
    ))
    _propagate_device(config, device)   # set EVERY device field (buffer/encoder/heads)

    agent = Dreamer(config.model, obs_space, act_space).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Leak guard: held-out test maps must never overlap the training split.
    for entry in test_entries:
        assert entry.name not in train_names, f"LEAK: train map '{entry.name}' in eval set!"

    # ---- Dedicated map × seed evaluation (ORIGINAL return; macro/micro) --------
    print(f"\nEvaluating {len(test_entries)} held-out maps × {len(seeds)} seeds "
          f"({len(test_entries) * len(seeds)} episodes) seeds={seeds} ...")
    eval_report = evaluate_heldout(
        agent, test_entries, pad_dims,
        seeds=seeds, device=device, gamma=float(cfg.gamma),
        max_episode_steps=int(cfg.max_episode_steps), obs_mode=obs_mode, progress=True,
    )
    eval_report["obs_mode"] = obs_mode

    # Per-family macro win rate (each map = one sample within its family).
    per_family_winrates = defaultdict(list)
    for m in eval_report["per_map"].values():
        per_family_winrates[m["family"]].append(m["win_rate"])
    eval_report["per_family"] = {
        fam: {"n_maps": len(wrs), "macro_win_rate": sum(wrs) / len(wrs)}
        for fam, wrs in per_family_winrates.items()
    }
    eval_report["checkpoint"] = str(args.checkpoint)
    eval_report["split_name"] = args.split if cfg.get("maps") else "ratio_heldout"
    eval_report["eval_tag"] = eval_tag
    if cfg.get("split"):
        eval_report["split"] = OmegaConf.to_container(cfg.split, resolve=True)

    out = pathlib.Path(args.output) if args.output else (
        ROOT / "results" / f"multimap_eval_{eval_tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")

    macro, micro = eval_report["macro"], eval_report["micro"]
    mlo, mhi = macro["win_rate_ci95"]
    print(f"\n{'='*64}")
    print(f"PRIMARY (selection): MACRO held-out win rate "
          f"{macro['win_rate']:.3f}  95% CI [{mlo:.3f}, {mhi:.3f}]  "
          f"(over {eval_report['n_maps']} maps × {len(seeds)} seeds)")
    print(f"  macro original_return={macro['original_return']:.3f}  "
          f"timeout_rate={macro['timeout_rate']:.2f}  "
          f"ally_ehp={macro['final_ally_ehp_frac']:.2f}  enemy_ehp={macro['final_enemy_ehp_frac']:.2f}")
    print(f"  micro win_rate={micro['win_rate']:.3f}  original_return={micro['original_return']:.3f}")
    print(f"Report written: {out}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
