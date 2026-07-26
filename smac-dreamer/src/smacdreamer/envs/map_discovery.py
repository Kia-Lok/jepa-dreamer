"""Folder-based map discovery + train/test split for multimap training.

Scans a folder of SMAClite map JSON files, validates each (parse, probe env dims,
check avail_actions, one valid step, known unit types), splits into train / held-out
test sets per a config spec, sets the model's padding from the **TRAIN-max only**, and
runs an **all-map safety-net** assertion so a test map never silently grows the model.

Design (per the approved multimap plan, §"Padding strategy"):
  (a) SHAPE-SETTING = TRAIN-max only. The PaddingDims the model is built against come
      from the largest dims across the TRAIN split. A config override wins if given.
      The test split NEVER influences the model shape.
  (b) SAFETY-NET = scan EVERY map (train AND test); assert none exceeds the chosen
      padding. A test map exceeding train-max FAILS FAST naming the map and its dims.
  (c) SCOPE = generalisation claim is limited to "unseen maps no larger than the
      largest training map." Acceptable for the easy/ally-advantaged gate folder.

This module is the single source of truth for folder scanning + per-map dim probing.
``scripts``-side tooling (e.g. the Phase 4 manifest builder) imports ``validate_map`` /
``sha256_file`` from here so there is no duplicate scan logic.

No edits to external/r2dreamer. Reuses MapEntry / PaddingDims from the project.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from smacdreamer.envs.map_sampler import MapEntry
from smacdreamer.envs.padding import PaddingDims


# Project root: src/smacdreamer/envs/map_discovery.py -> up 4 = repo root.
_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent

# Unit types SMAClite ships with. A map referencing anything else is rejected (the
# project constraint forbids custom units).
KNOWN_UNIT_TYPES = {
    "ZERGLING", "BANELING", "SPINE_CRAWLER",
    "MARINE", "MEDIVAC", "MARAUDER",
    "ZEALOT", "STALKER", "COLOSSUS",
}


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _extract_unit_types(groups: list) -> set:
    types = set()
    for g in groups:
        for unit_name in g.get("units", {}).keys():
            types.add(unit_name.upper())
    return types


def validate_map(
    path: pathlib.Path,
    map_dir: pathlib.Path,
    family_from_parent: bool = True,
    limits: Optional[dict] = None,
    on_exceed: str = "exclude",
    root: pathlib.Path = _ROOT,
) -> dict:
    """Validate one map JSON and probe its env dims.

    Returns a result dict identical in shape to the Phase-4 builder's contract:
      ok, reason, path, rel_path, file_hash, and (when ok) map_info with
      name/stem/family/n_agents/n_enemies/n_actions/obs_size/... . On a limit breach
      returns ok=False with limit_exceeded=True and a dimensions dict.

    The shared scanning/probing core — imported by both the multimap discovery path and
    scripts/Archive manifest tooling so there is exactly one implementation.
    """
    limits = limits or {}
    path = pathlib.Path(path)
    map_dir = pathlib.Path(map_dir)
    rel_path = str(path.relative_to(root)).replace("\\", "/")
    try:
        file_hash = sha256_file(path)
    except OSError as e:
        return {"ok": False, "reason": f"file read error: {e}", "path": path,
                "rel_path": rel_path, "file_hash": ""}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"ok": False, "reason": f"JSON parse error: {e}", "path": path,
                "rel_path": rel_path, "file_hash": file_hash}

    n_ally = raw.get("num_allied_units", 0)
    n_enemy = raw.get("num_enemy_units", 0)
    if n_ally < 1:
        return {"ok": False, "reason": f"num_allied_units={n_ally} < 1", "path": path,
                "rel_path": rel_path, "file_hash": file_hash}
    if n_enemy < 1:
        return {"ok": False, "reason": f"num_enemy_units={n_enemy} < 1", "path": path,
                "rel_path": rel_path, "file_hash": file_hash}

    groups = raw.get("groups", [])
    unit_types = _extract_unit_types(groups)
    custom_types = unit_types - KNOWN_UNIT_TYPES
    if custom_types:
        return {"ok": False, "reason": f"unknown/custom unit types: {sorted(custom_types)}",
                "path": path, "rel_path": rel_path, "file_hash": file_hash}
    if raw.get("custom_unit_path"):
        return {"ok": False, "reason": "custom_unit_path is set (custom units not allowed)",
                "path": path, "rel_path": rel_path, "file_hash": file_hash}

    # Probe env dims (import deferred so this module imports without smaclite installed).
    try:
        from smaclite.env.smaclite import SMACliteEnv as _SMACliteEnv
        env = _SMACliteEnv(map_file=str(path))
        n_agents, n_enemies = env.n_agents, env.n_enemies
        n_actions, obs_size = env.n_actions, env.obs_size
        env.reset()
        avail = env.get_avail_actions()
        avail_lens = [len(a) for a in avail]
        if len(set(avail_lens)) != 1 or avail_lens[0] != n_actions:
            env.close()
            return {"ok": False, "reason": f"non-uniform avail_actions: {set(avail_lens)}",
                    "path": path, "rel_path": rel_path, "file_hash": file_hash}
        step_acts = [next((i for i, v in enumerate(a) if v), 0) for a in avail]
        env.step(step_acts)
        env.close()
    except Exception as e:
        return {"ok": False, "reason": f"env load/step failed: {e}", "path": path,
                "rel_path": rel_path, "file_hash": file_hash}

    dim_map = {"max_agents": n_agents, "max_enemies": n_enemies,
               "max_actions": n_actions, "max_obs_size": obs_size}
    violations = [f"{k}: actual={dim_map[k]} > limit={limits[k]}"
                  for k in dim_map if limits.get(k) is not None and dim_map[k] > limits[k]]
    if violations:
        reason = "exceeds limits: " + "; ".join(violations)
        if on_exceed == "error":
            raise SystemExit(f"ERROR: {rel_path} {reason}")
        return {"ok": False, "reason": reason, "path": path, "rel_path": rel_path,
                "file_hash": file_hash, "limit_exceeded": True,
                "dimensions": {"n_agents": n_agents, "n_enemies": n_enemies,
                               "n_actions": n_actions, "obs_size": obs_size}}

    if family_from_parent:
        try:
            parts = path.relative_to(map_dir).parts
            family = parts[0] if len(parts) > 1 else "uncategorised"
        except ValueError:
            family = "uncategorised"
    else:
        family = raw.get("category", raw.get("family", "uncategorised"))

    return {
        "ok": True, "reason": "", "path": path, "rel_path": rel_path,
        "file_hash": file_hash,
        "map_info": {
            "name": raw.get("name", path.stem), "stem": path.stem, "family": family,
            "n_agents": n_agents, "n_enemies": n_enemies,
            "n_actions": n_actions, "obs_size": obs_size,
            "ally_has_shields": raw.get("ally_has_shields", False),
            "enemy_has_shields": raw.get("enemy_has_shields", False),
            "terrain_preset": raw.get("terrain_preset", ""),
            "num_unit_types": raw.get("num_unit_types", 0),
            "unit_types": sorted(unit_types),
        },
    }


def scan_folder(
    folder: str,
    recursive: bool = True,
    family_from_parent: bool = True,
    root: pathlib.Path = _ROOT,
    verbose: bool = False,
    isolate_probe: bool = False,
    probe_workers: int = 4,
    probe_maxtasks: int = 10,
) -> tuple[list, list, list]:
    """Scan a folder of *.json maps. Returns (included, excluded, invalid) result dicts.

    included entries each carry map_info + file_hash; deduped by filename and by content
    hash, sorted by rel_path for stable map_id assignment by callers.

    ``isolate_probe`` runs each map's env probe in a recycled spawn-Pool worker so that
    native memory SMAClite does not release on ``env.close()`` is reclaimed per worker
    rather than accumulating in this process. Required for large folders (e.g. 500 maps)
    under a tight pod memory cap; ``probe_workers`` parallelism and ``probe_maxtasks``
    (worker recycle interval) bound the transient footprint. Default off keeps the
    original single-process behaviour for tests / small folders.
    """
    map_dir = pathlib.Path(folder)
    if not map_dir.is_absolute():
        map_dir = (root / folder).resolve()
    if not map_dir.exists():
        raise FileNotFoundError(f"map folder does not exist: {map_dir}")

    pattern = "**/*.json" if recursive else "*.json"
    all_paths = sorted(map_dir.glob(pattern))
    if not all_paths:
        raise FileNotFoundError(f"no .json maps found under {map_dir}")

    # Dedup by filename stem (first wins).
    seen_names: dict = {}
    for p in all_paths:
        seen_names.setdefault(p.stem, p)
    unique_paths = [seen_names[s] for s in sorted(seen_names)]

    # Obtain a validate_map result per path, either in-process or via a recycled
    # subprocess pool (memory-isolated). Order matches unique_paths in both cases.
    if isolate_probe:
        import multiprocessing as _mp

        ctx = _mp.get_context("spawn")
        args = [(p, map_dir, family_from_parent, root) for p in unique_paths]
        with ctx.Pool(processes=max(1, probe_workers),
                      maxtasksperchild=max(1, probe_maxtasks)) as pool:
            results = list(pool.imap(_probe_worker, args, chunksize=1))
    else:
        results = [validate_map(p, map_dir, family_from_parent, root=root)
                   for p in unique_paths]

    included, excluded, invalid = [], [], []
    seen_hashes: dict = {}
    for path, r in zip(unique_paths, results):
        if not r["ok"]:
            (excluded if r.get("limit_exceeded") else invalid).append(
                {"path": r["rel_path"], "reason": r["reason"]})
            if verbose:
                print(f"  SKIP {path.name}: {r['reason']}")
            continue
        fh = r["file_hash"]
        if fh in seen_hashes:
            excluded.append({"path": r["rel_path"],
                             "reason": f"duplicate content of {seen_hashes[fh]}"})
            continue
        seen_hashes[fh] = r["rel_path"]
        included.append(r)

    included.sort(key=lambda r: r["rel_path"])
    for mid, r in enumerate(included):
        r["map_id"] = mid
    return included, excluded, invalid


def _probe_worker(args):
    """Module-level wrapper so a spawn Pool can pickle the probe call.

    Runs ``validate_map`` for one map inside a throwaway pool worker; the worker is
    recycled (``maxtasksperchild``) so any native memory SMAClite fails to release on
    ``env.close()`` is reclaimed by the OS instead of accumulating in the parent.
    """
    # Defensive, cross-platform path setup: spawn Pool children normally inherit the
    # parent's sys.path, but ensure smaclite/r2dreamer/src resolve even if they don't
    # (and avoid the backslash-separator bug in the factory's _ensure_paths on Linux).
    import sys
    for sub in ("src", "external/smaclite", "external/r2dreamer"):
        p = str(_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    path, map_dir, family_from_parent, root = args
    return validate_map(path, map_dir, family_from_parent, root=root)


def _to_entry(r: dict) -> MapEntry:
    mi = r["map_info"]
    return MapEntry(
        name=mi["name"], type="custom", path=r["rel_path"],
        family=mi["family"], map_id=r["map_id"],
    )


def _raise_scan_failures(label: str, excluded: list, invalid: list) -> None:
    failures = []
    for item in invalid:
        failures.append(f"INVALID {item['path']}: {item['reason']}")
    for item in excluded:
        failures.append(f"EXCLUDED {item['path']}: {item['reason']}")
    if failures:
        raise ValueError(
            f"{label} map discovery found {len(failures)} skipped map(s); refusing "
            "to start with silent map loss:\n  " + "\n  ".join(failures)
        )


@dataclass
class SplitSpec:
    """Train/test split specification (test = held-out)."""
    mode: str = "ratio"                 # "ratio" | "explicit"
    train_ratio: float = 0.8
    seed: int = 0
    train_names: Optional[list] = None  # used when mode == "explicit"
    test_names: Optional[list] = None


def split_maps(included: list, spec: SplitSpec) -> tuple[list, list]:
    """Split included result dicts into (train, test) lists. Deterministic for a seed."""
    if spec.mode == "explicit":
        train_set = set(spec.train_names or [])
        test_set = set(spec.test_names or [])
        overlap = train_set & test_set
        if overlap:
            raise ValueError(f"explicit split: maps in both train and test: {sorted(overlap)}")
        by_name = {r["map_info"]["name"]: r for r in included}
        missing = (train_set | test_set) - set(by_name)
        if missing:
            raise ValueError(f"explicit split names not found in folder: {sorted(missing)}")
        train = [by_name[n] for n in sorted(train_set)]
        test = [by_name[n] for n in sorted(test_set)]
        return train, test

    # ratio mode: shuffle deterministically, take train_ratio for train.
    if not (0.0 < spec.train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {spec.train_ratio}")
    maps = list(included)
    random.Random(spec.seed).shuffle(maps)
    n_train = max(1, round(len(maps) * spec.train_ratio))
    n_train = min(n_train, len(maps) - 1)  # guarantee a non-empty test set
    return maps[:n_train], maps[n_train:]


def compute_train_max_padding(train: list, override: Optional[dict] = None) -> PaddingDims:
    """PaddingDims from TRAIN-max only (config override wins). Test never influences shape."""
    if override and all(override.get(k) for k in
                        ("max_agents", "max_enemies", "max_actions", "max_obs_size")):
        return PaddingDims(int(override["max_agents"]), int(override["max_enemies"]),
                           int(override["max_actions"]), int(override["max_obs_size"]))
    if not train:
        raise ValueError("cannot compute train-max padding from an empty train split")
    return PaddingDims(
        max_agents=max(r["map_info"]["n_agents"] for r in train),
        max_enemies=max(r["map_info"]["n_enemies"] for r in train),
        max_actions=max(r["map_info"]["n_actions"] for r in train),
        max_obs_size=max(r["map_info"]["obs_size"] for r in train),
    )


def safety_net_check(all_maps: list, pad_dims: PaddingDims, obs_mode: str = "flat") -> None:
    """Fail fast if ANY map (train or test) exceeds the chosen padding.

    A test map exceeding train-max is the leak path: never auto-grow the model. Raises
    ValueError naming each offending map and the dimension(s) it violates.

    In ``structured`` obs mode the canonical layout is fixed by max_agents/max_enemies/
    max_actions + the global type vocab, so ``max_obs_size`` (a legacy flat-layout quantity)
    is NOT a constraint and is not checked.
    """
    check_obs_size = (obs_mode != "structured")
    offenders = []
    for r in all_maps:
        mi = r["map_info"]
        viol = []
        if mi["n_agents"] > pad_dims.max_agents:
            viol.append(f"n_agents={mi['n_agents']}>max_agents={pad_dims.max_agents}")
        if mi["n_enemies"] > pad_dims.max_enemies:
            viol.append(f"n_enemies={mi['n_enemies']}>max_enemies={pad_dims.max_enemies}")
        if mi["n_actions"] > pad_dims.max_actions:
            viol.append(f"n_actions={mi['n_actions']}>max_actions={pad_dims.max_actions}")
        if check_obs_size and mi["obs_size"] > pad_dims.max_obs_size:
            viol.append(f"obs_size={mi['obs_size']}>max_obs_size={pad_dims.max_obs_size}")
        if viol:
            offenders.append(f"  '{mi['name']}' ({r['rel_path']}): " + "; ".join(viol))
    if offenders:
        raise ValueError(
            "Padding safety-net failed — these maps exceed the model padding "
            "(set from TRAIN-max). Exclude them from the test split or raise an explicit "
            "padding cap in config. NEVER grow the model to fit a held-out test map "
            "(that leaks the test size envelope into the model shape):\n"
            + "\n".join(offenders)
        )


def discover(
    folder: str,
    split_spec: SplitSpec,
    padding_override: Optional[dict] = None,
    recursive: bool = True,
    family_from_parent: bool = True,
    verbose: bool = True,
    isolate_probe: bool = False,
    probe_workers: int = 4,
    probe_maxtasks: int = 10,
    obs_mode: str = "flat",
) -> tuple[list, list, PaddingDims]:
    """Top-level entry: scan folder -> split -> train-max padding -> safety-net.

    Returns (train_entries, test_entries, pad_dims):
      train_entries / test_entries : list[MapEntry] for the MapSampler.
      pad_dims                     : PaddingDims the model is built against (TRAIN-max
                                     or config override), validated against ALL maps.

    ``isolate_probe`` (+ ``probe_workers`` / ``probe_maxtasks``) probes each map in a
    recycled subprocess so a large folder does not blow the process memory cap; see
    ``scan_folder``.
    """
    included, excluded, invalid = scan_folder(
        folder, recursive=recursive, family_from_parent=family_from_parent, verbose=verbose,
        isolate_probe=isolate_probe, probe_workers=probe_workers, probe_maxtasks=probe_maxtasks)
    _raise_scan_failures(str(folder), excluded, invalid)
    if not included:
        raise ValueError(f"no valid maps in {folder} (excluded={len(excluded)}, invalid={len(invalid)})")

    train, test = split_maps(included, split_spec)
    pad_dims = compute_train_max_padding(train, padding_override)
    safety_net_check(included, pad_dims, obs_mode=obs_mode)  # ALL maps (train + test)

    if verbose:
        src = "config override" if padding_override else "TRAIN-max"
        print(f"[discover] folder={folder}")
        print(f"[discover] included={len(included)} excluded={len(excluded)} invalid={len(invalid)}")
        print(f"[discover] split: train={len(train)} test(held-out)={len(test)} "
              f"(mode={split_spec.mode}, seed={split_spec.seed})")
        print(f"[discover] PADDING ({src}): max_agents={pad_dims.max_agents} "
              f"max_enemies={pad_dims.max_enemies} max_actions={pad_dims.max_actions} "
              f"max_obs_size={pad_dims.max_obs_size}")
        print(f"[discover] safety-net: all {len(included)} maps fit the padding ✓")

    return [_to_entry(r) for r in train], [_to_entry(r) for r in test], pad_dims


def discover_folders(
    train_folder: str,
    validation_folder: str,
    padding_override: Optional[dict] = None,
    obs_mode: str = "flat",
    recursive: bool = True,
    family_from_parent: bool = True,
    verbose: bool = True,
    isolate_probe: bool = False,
    probe_workers: int = 4,
    probe_maxtasks: int = 10,
) -> tuple[list, list, PaddingDims]:
    """Explicit-folder discovery (no ratio split).

    The TRAIN folder alone sets the model padding (TRAIN-max, or the config override). The
    VALIDATION folder is held out for checkpoint selection and must FIT that padding (safety
    net over train+validation). ONLY these two folders are scanned — blind splits are never
    touched during training. Returns ``(train_entries, validation_entries, pad_dims)``.
    """
    tr_incl, tr_exc, tr_inv = scan_folder(
        train_folder, recursive=recursive, family_from_parent=family_from_parent,
        verbose=verbose, isolate_probe=isolate_probe,
        probe_workers=probe_workers, probe_maxtasks=probe_maxtasks)
    _raise_scan_failures("TRAIN", tr_exc, tr_inv)
    if not tr_incl:
        raise ValueError(f"no valid TRAIN maps in {train_folder} "
                         f"(excluded={len(tr_exc)}, invalid={len(tr_inv)})")
    pad_dims = compute_train_max_padding(tr_incl, padding_override)

    va_incl, va_exc, va_inv = scan_folder(
        validation_folder, recursive=recursive, family_from_parent=family_from_parent,
        verbose=verbose, isolate_probe=isolate_probe,
        probe_workers=probe_workers, probe_maxtasks=probe_maxtasks)
    _raise_scan_failures("VALIDATION", va_exc, va_inv)
    if not va_incl:
        raise ValueError(f"no valid VALIDATION maps in {validation_folder} "
                         f"(excluded={len(va_exc)}, invalid={len(va_inv)})")

    # Validation maps must fit the TRAIN-derived padding (never grow the model for validation).
    safety_net_check(tr_incl + va_incl, pad_dims, obs_mode=obs_mode)

    if verbose:
        src = "config override" if padding_override else "TRAIN-max"
        print(f"[discover_folders] train={len(tr_incl)} validation={len(va_incl)}  "
              f"PADDING ({src}): max_agents={pad_dims.max_agents} max_enemies={pad_dims.max_enemies} "
              f"max_actions={pad_dims.max_actions} max_obs_size={pad_dims.max_obs_size}  "
              f"obs_mode={obs_mode}")
        print(f"[discover_folders] safety-net: all {len(tr_incl) + len(va_incl)} train+val maps fit ✓")

    return [_to_entry(r) for r in tr_incl], [_to_entry(r) for r in va_incl], pad_dims


def scan_folder_entries(
    folder: str,
    *,
    recursive: bool = True,
    family_from_parent: bool = True,
    verbose: bool = True,
    isolate_probe: bool = True,
) -> list:
    """Scan a single folder and return its valid MapEntry list (e.g. a blind split for
    post-training standalone evaluation). Does NOT compute padding or a split."""
    incl, _exc, _inv = scan_folder(
        folder, recursive=recursive, family_from_parent=family_from_parent,
        verbose=verbose, isolate_probe=isolate_probe)
    _raise_scan_failures(str(folder), _exc, _inv)
    if not incl:
        raise ValueError(f"no valid maps in {folder}")
    return [_to_entry(r) for r in incl]


__all__ = [
    "SplitSpec", "discover", "discover_folders", "scan_folder", "scan_folder_entries",
    "split_maps", "compute_train_max_padding", "safety_net_check", "validate_map",
    "sha256_file", "KNOWN_UNIT_TYPES",
]
