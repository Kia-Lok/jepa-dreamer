"""Build ONE multimap SMAClite env in the MAIN process (no worker) to surface the
real construction error that ParallelEnv's worker swallows as "Lost connection".

The multimap factory builds envs inside spawn workers, so a crash there only shows
up in the parent as EOFError / "Lost connection to worker". Running the identical
constructor in-process means:
  * a Python exception prints its real traceback here, and
  * a native crash prints "Segmentation fault" / an OMP/SDL abort message,
pinpointing the failing library.

Usage (smac-r2 env, from project root):
    python scripts/debug_build_one_env.py --config configs/multimap_gpu.yaml
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (str(ROOT / "src"), str(ROOT / "external" / "r2dreamer"), str(ROOT / "external" / "smaclite")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omegaconf import OmegaConf

from smacdreamer.envs.map_discovery import discover, SplitSpec
from smacdreamer.r2dreamer_factory import make_smaclite_multimap_env


def _rss_gb() -> float:
    """Current process resident memory in GB (Linux /proc; -1 if unavailable)."""
    import os
    try:
        with open(f"/proc/{os.getpid()}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)  # kB -> GB
    except Exception:
        pass
    return -1.0


def _inspect_map_terrain(map_path) -> None:
    """Print terrain grid dimensions from a map JSON — a pathological (huge) terrain is
    the prime suspect for a single-world OOM (StaticObstacle.from_terrain scales with it)."""
    import json
    try:
        raw = json.loads(pathlib.Path(map_path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f">> (could not read map JSON for terrain inspection: {e})")
        return
    terrain = raw.get("terrain")
    if isinstance(terrain, list):
        rows = len(terrain)
        cols = max((len(r) for r in terrain if isinstance(r, list)), default=0)
        print(f">> terrain grid: {rows} x {cols} = {rows * cols:,} cells")
    else:
        print(f">> terrain field type: {type(terrain).__name__} "
              f"(preset={raw.get('terrain_preset')!r}); "
              f"num_allied={raw.get('num_allied_units')} num_enemy={raw.get('num_enemy_units')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multimap_gpu.yaml")
    args = ap.parse_args()

    cfg = OmegaConf.load(str(ROOT / args.config if not pathlib.Path(args.config).is_absolute() else args.config))

    print(">> discovering maps (parent process)...")
    split = OmegaConf.to_container(cfg.split, resolve=True)
    pad_override = OmegaConf.to_container(cfg.padding, resolve=True) if cfg.get("padding") else None
    train_entries, test_entries, pad_dims = discover(
        str(cfg.maps_folder), SplitSpec(**split), padding_override=pad_override, verbose=True,
        isolate_probe=True,   # subprocess-isolated probe, same as the training factory
    )
    print(f">> train maps: {len(train_entries)}  test maps: {len(test_entries)}")
    print(f">> pad_dims: {pad_dims}")
    print(f">> RSS after discovery: {_rss_gb():.2f} GB")
    if not train_entries:
        raise SystemExit("No TRAIN maps discovered — check maps_folder / split. "
                         f"maps_folder={cfg.maps_folder!r}")

    # Identify the EXACT map the training env will open first (sampler peek), so if the
    # build OOMs we know which map's world is the culprit.
    from smacdreamer.envs.map_sampler import MapSampler
    from smacdreamer.r2dreamer_factory import _worker_seed
    sampler = MapSampler.from_entries(
        train_entries, mode=str(cfg.sampling_mode), seed=_worker_seed(int(cfg.seed), 0))
    first = sampler.peek()
    print(f">> first map to open (peek): name={first.name!r} path={first.path!r}")
    _inspect_map_terrain(ROOT / first.path)

    print(">> opening JUST the raw SMAClite world for that map (no adapter/padding)...")
    from smaclite.env.smaclite import SMACliteEnv as _SMACliteEnv
    raw = _SMACliteEnv(map_file=str(ROOT / first.path))
    print(f">> raw env constructed; RSS={_rss_gb():.2f} GB; resetting...")
    raw.reset()
    print(f">> raw reset OK; RSS={_rss_gb():.2f} GB")
    raw.close()

    print(">> building ONE full env IN-PROCESS (adapter + padding + sampler)...")
    env = make_smaclite_multimap_env(
        train_entries, pad_dims, str(cfg.sampling_mode), int(cfg.seed), 0,
        str(cfg.reward.name), OmegaConf.to_container(cfg.reward.get("params", {}), resolve=True),
        float(cfg.gamma), int(cfg.max_episode_steps),
    )
    print(f">> env constructed OK; RSS={_rss_gb():.2f} GB")
    print(">> observation_space:", env.observation_space)
    print(">> action_space:", env.action_space)

    print(">> reset()...")
    obs = env.reset()
    keys = sorted(obs[0].keys()) if isinstance(obs, tuple) else sorted(obs.keys())
    print(">> reset OK; obs keys:", keys)
    print("\nSUCCESS — single-process env construction works. The crash is worker/spawn-specific "
          "(try env_num: 1, or the threading env-vars in the docs).")


if __name__ == "__main__":
    main()
