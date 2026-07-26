from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

SEED = 26052026
MAP_WIDTH = 32
MAP_HEIGHT = 32
GROUP_BUFFER = 0.05
GLOBAL_UNIT_TYPE_IDS = {
    "BANELING": 0,
    "COLOSSUS": 1,
    "MARAUDER": 2,
    "MARINE": 3,
    "MEDIVAC": 4,
    "SPINE_CRAWLER": 5,
    "STALKER": 6,
    "ZEALOT": 7,
    "ZERGLING": 8,
}

@dataclass(frozen=True)
class UnitInfo:
    hp: float
    shield: float
    damage: float
    attacks: int
    cooldown: float
    speed: float
    attack_range: float
    size: float
    plane: str
    combat_type: str
    valid_targets: Tuple[str, ...]
    value: float

    @property
    def radius(self) -> float:
        return self.size / 2.0

    @property
    def has_shield(self) -> bool:
        return self.shield > 0

    @property
    def dps(self) -> float:
        if self.combat_type != "DAMAGE" or self.cooldown <= 0:
            return 0.0
        return self.damage * self.attacks / self.cooldown

UNITS: Dict[str, UnitInfo] = {
    "BANELING": UnitInfo(30, 0, 80, 1, 1.0, 3.15, 0.25, 0.75, "GROUND", "DAMAGE", ("GROUND",), 50),
    "COLOSSUS": UnitInfo(200, 150, 10, 2, 1.07, 3.15, 7.0, 2.0, "GROUND", "DAMAGE", ("GROUND",), 300),
    "MARAUDER": UnitInfo(125, 0, 10, 1, 1.07, 3.15, 6.0, 1.125, "GROUND", "DAMAGE", ("GROUND",), 100),
    "MARINE": UnitInfo(45, 0, 6, 1, 0.61, 3.15, 5.0, 0.75, "GROUND", "DAMAGE", ("GROUND", "AIR"), 50),
    "MEDIVAC": UnitInfo(150, 0, 0, 1, 0.0, 3.5, 4.0, 1.5, "AIR", "HEALING", ("GROUND",), 125),
    "SPINE_CRAWLER": UnitInfo(300, 0, 25, 1, 1.32, 0.0, 7.0, 1.0, "GROUND", "DAMAGE", ("GROUND",), 175),
    "STALKER": UnitInfo(80, 80, 13, 1, 1.34, 4.13, 6.0, 1.25, "GROUND", "DAMAGE", ("GROUND", "AIR"), 125),
    "ZEALOT": UnitInfo(100, 50, 8, 2, 0.86, 3.15, 0.25, 1.0, "GROUND", "DAMAGE", ("GROUND",), 100),
    "ZERGLING": UnitInfo(35, 0, 5, 1, 0.497, 4.13, 0.25, 0.75, "GROUND", "DAMAGE", ("GROUND",), 25),
}

ENGAGEMENT_TARGETS = {
    "immediate": 5.4,
    "near": 8.8,
    "medium": 12.8,
    "far": 17.0,
}
ENGAGEMENT_BOUNDS = {
    "immediate": (4.5, 6.2),
    "near": (7.0, 10.5),
    "medium": (10.8, 14.8),
    "far": (15.0, 19.8),
}

TERRAINS = ("SIMPLE", "NARROW", "OCTAGON")
FORMATIONS = ("compact", "split", "staggered", "type_split")

@dataclass(frozen=True)
class Family:
    family_id: str
    archetype: str
    ally: Mapping[str, int]
    enemy: Mapping[str, int]
    heldout_compositional: bool = False


def C(**kwargs: int) -> Dict[str, int]:
    return {k: int(v) for k, v in kwargs.items() if v > 0}


def build_families() -> Tuple[List[Family], List[Family]]:
    seen: List[Family] = []
    held: List[Family] = []

    def add_many(dst: List[Family], arch: str, pairs: Sequence[Tuple[Mapping[str, int], Mapping[str, int]]], heldout=False):
        for idx, (ally, enemy) in enumerate(pairs, 1):
            dst.append(Family(f"{arch}_{idx:02d}", arch, dict(ally), dict(enemy), heldout))

    add_many(seen, "marine_mirror", [
        (C(MARINE=3), C(MARINE=2)),
        (C(MARINE=4), C(MARINE=3)),
        (C(MARINE=5), C(MARINE=4)),
        (C(MARINE=7), C(MARINE=6)),
        (C(MARINE=9), C(MARINE=8)),
    ])
    add_many(seen, "stalker_mirror", [
        (C(STALKER=3), C(STALKER=2)),
        (C(STALKER=4), C(STALKER=3)),
        (C(STALKER=5), C(STALKER=4)),
        (C(STALKER=7), C(STALKER=6)),
        (C(STALKER=9), C(STALKER=8)),
    ])
    add_many(seen, "zealot_vs_zergling", [
        (C(ZEALOT=2), C(ZERGLING=6)),
        (C(ZEALOT=2), C(ZERGLING=7)),
        (C(ZEALOT=3), C(ZERGLING=9)),
        (C(ZEALOT=3), C(ZERGLING=10)),
        (C(ZEALOT=4), C(ZERGLING=10)),
    ])
    add_many(seen, "stalker_vs_zealot", [
        (C(STALKER=2), C(ZEALOT=2)),
        (C(STALKER=3), C(ZEALOT=3)),
        (C(STALKER=4), C(ZEALOT=4)),
        (C(STALKER=5), C(ZEALOT=5)),
        (C(STALKER=6), C(ZEALOT=6)),
    ])
    add_many(seen, "terran_vs_swarm", [
        (C(MARINE=2, MARAUDER=1), C(ZERGLING=4, BANELING=1)),
        (C(MARINE=2, MARAUDER=2), C(ZERGLING=6, BANELING=1)),
        (C(MARINE=3, MARAUDER=2), C(ZERGLING=6, BANELING=2)),
        (C(MARINE=4, MARAUDER=2), C(ZERGLING=8, BANELING=2)),
        (C(MARINE=5, MARAUDER=3), C(ZERGLING=7, BANELING=3)),
    ])
    add_many(seen, "mmm_mirror", [
        (C(MARINE=3, MARAUDER=1, MEDIVAC=1), C(MARINE=2, MARAUDER=1, MEDIVAC=1)),
        (C(MARINE=3, MARAUDER=2, MEDIVAC=1), C(MARINE=3, MARAUDER=1, MEDIVAC=1)),
        (C(MARINE=4, MARAUDER=2, MEDIVAC=1), C(MARINE=3, MARAUDER=2, MEDIVAC=1)),
        (C(MARINE=4, MARAUDER=3, MEDIVAC=1), C(MARINE=4, MARAUDER=2, MEDIVAC=1)),
        (C(MARINE=5, MARAUDER=3, MEDIVAC=1), C(MARINE=4, MARAUDER=3, MEDIVAC=1)),
    ])
    add_many(seen, "stalker_zealot_mirror", [
        (C(STALKER=2, ZEALOT=2), C(STALKER=2, ZEALOT=1)),
        (C(STALKER=3, ZEALOT=2), C(STALKER=2, ZEALOT=2)),
        (C(STALKER=3, ZEALOT=3), C(STALKER=3, ZEALOT=2)),
        (C(STALKER=4, ZEALOT=3), C(STALKER=3, ZEALOT=3)),
        (C(STALKER=5, ZEALOT=4), C(STALKER=4, ZEALOT=4)),
    ])
    add_many(seen, "stalker_colossus_vs_swarm", [
        (C(STALKER=1, COLOSSUS=1), C(ZERGLING=6, BANELING=4)),
        (C(STALKER=1, COLOSSUS=1), C(ZERGLING=4, BANELING=6)),
        (C(STALKER=2, COLOSSUS=1), C(ZERGLING=2, BANELING=8)),
        (C(STALKER=1, COLOSSUS=2), C(BANELING=10)),
        (C(STALKER=2, COLOSSUS=2), C(BANELING=10)),
    ])
    add_many(seen, "stalker_vs_spine", [
        (C(STALKER=2), C(SPINE_CRAWLER=1)),
        (C(STALKER=4), C(SPINE_CRAWLER=2)),
        (C(STALKER=5), C(SPINE_CRAWLER=3)),
        (C(STALKER=7), C(SPINE_CRAWLER=4)),
        (C(STALKER=9), C(SPINE_CRAWLER=5)),
    ])
    add_many(seen, "zealot_colossus_vs_terran", [
        (C(ZEALOT=2, COLOSSUS=1), C(MARINE=2, MARAUDER=2)),
        (C(ZEALOT=2, COLOSSUS=1), C(MARINE=3, MARAUDER=2)),
        (C(ZEALOT=3, COLOSSUS=1), C(MARINE=4, MARAUDER=2)),
        (C(ZEALOT=3, COLOSSUS=2), C(MARINE=5, MARAUDER=3)),
        (C(ZEALOT=4, COLOSSUS=2), C(MARINE=5, MARAUDER=4)),
    ])

    add_many(held, "full_protoss_vs_mmm", [
        (C(STALKER=2, ZEALOT=2, COLOSSUS=1), C(MARINE=3, MARAUDER=2, MEDIVAC=1)),
        (C(STALKER=3, ZEALOT=3, COLOSSUS=1), C(MARINE=4, MARAUDER=3, MEDIVAC=1)),
    ], True)
    add_many(held, "stalker_zealot_vs_swarm", [
        (C(STALKER=2, ZEALOT=2), C(ZERGLING=6, BANELING=3)),
        (C(STALKER=3, ZEALOT=3), C(ZERGLING=4, BANELING=6)),
    ], True)
    add_many(held, "marine_medivac_vs_swarm", [
        (C(MARINE=4, MEDIVAC=1), C(MARINE=1, ZERGLING=6, BANELING=2)),
        (C(MARINE=6, MEDIVAC=1), C(MARINE=1, ZERGLING=6, BANELING=3)),
    ], True)
    add_many(held, "terran_vs_stalker_colossus", [
        (C(MARINE=4, MARAUDER=3), C(STALKER=1, COLOSSUS=1)),
        (C(MARINE=5, MARAUDER=4), C(STALKER=2, COLOSSUS=1)),
    ], True)
    add_many(held, "mmm_vs_stalker_zealot", [
        (C(MARINE=4, MARAUDER=2, MEDIVAC=1), C(STALKER=2, ZEALOT=2)),
        (C(MARINE=5, MARAUDER=3, MEDIVAC=1), C(STALKER=3, ZEALOT=2)),
    ], True)
    add_many(held, "stalker_zealot_vs_mmm", [
        (C(STALKER=3, ZEALOT=2), C(MARINE=3, MARAUDER=1, MEDIVAC=1)),
        (C(STALKER=4, ZEALOT=3), C(MARINE=4, MARAUDER=2, MEDIVAC=1)),
    ], True)
    add_many(held, "colossus_zealot_vs_swarm", [
        (C(COLOSSUS=1, ZEALOT=2), C(ZERGLING=6, BANELING=4)),
        (C(COLOSSUS=2, ZEALOT=2), C(ZERGLING=2, BANELING=8)),
    ], True)
    add_many(held, "stalker_vs_swarm", [
        (C(STALKER=3), C(ZERGLING=6, BANELING=2)),
        (C(STALKER=5), C(ZERGLING=6, BANELING=4)),
    ], True)
    add_many(held, "mmm_vs_stalker_colossus", [
        (C(MARINE=4, MARAUDER=2, MEDIVAC=1), C(STALKER=1, COLOSSUS=1)),
        (C(MARINE=5, MARAUDER=3, MEDIVAC=1), C(STALKER=2, COLOSSUS=1)),
    ], True)
    add_many(held, "stalker_colossus_vs_mmm", [
        (C(STALKER=2, COLOSSUS=1), C(MARINE=3, MARAUDER=1, MEDIVAC=1)),
        (C(STALKER=3, COLOSSUS=1), C(MARINE=4, MARAUDER=2, MEDIVAC=1)),
    ], True)

    assert len(seen) == 50, len(seen)
    assert len(held) == 20, len(held)
    return seen, held


def comp_count(comp: Mapping[str, int]) -> int:
    return sum(comp.values())


def comp_value(comp: Mapping[str, int]) -> float:
    return sum(UNITS[u].value * n for u, n in comp.items())


def comp_has_shields(comp: Mapping[str, int]) -> bool:
    flags = {UNITS[u].has_shield for u in comp}
    if len(flags) != 1:
        raise ValueError(f"Faction mixes shielded and unshielded units: {comp}")
    return next(iter(flags))


def can_target_air(comp: Mapping[str, int]) -> bool:
    return any("AIR" in UNITS[u].valid_targets and n > 0 for u, n in comp.items())


def validate_family(f: Family) -> None:
    if not 2 <= comp_count(f.ally) <= 10:
        raise ValueError(f"{f.family_id}: allied team size outside [2,10]: {f.ally}")
    if not 1 <= comp_count(f.enemy) <= 10:
        raise ValueError(f"{f.family_id}: enemy team size outside [1,10]: {f.enemy}")
    for comp in (f.ally, f.enemy):
        comp_has_shields(comp)
        if not any(UNITS[u].combat_type == "DAMAGE" for u in comp):
            raise ValueError(f"{f.family_id}: all-healer composition")
    if f.ally.get("MEDIVAC", 0) and not can_target_air(f.enemy):
        raise ValueError(f"{f.family_id}: enemy cannot target allied medivac")
    if f.enemy.get("MEDIVAC", 0) and not can_target_air(f.ally):
        raise ValueError(f"{f.family_id}: allies cannot target enemy medivac")
    ratio = comp_value(f.ally) / comp_value(f.enemy)
    if not 1.04 <= ratio <= 1.85:
        raise ValueError(f"{f.family_id}: ally power ratio {ratio:.3f} outside [1.04,1.85]")


def split_composition(comp: Mapping[str, int], parts: int) -> List[Dict[str, int]]:
    parts = max(1, min(parts, comp_count(comp)))
    bins: List[Dict[str, int]] = [defaultdict(int) for _ in range(parts)]
    expanded: List[str] = []
    for unit, count in comp.items():
        expanded.extend([unit] * count)
    # Large units first for more balanced group footprints.
    expanded.sort(key=lambda u: (UNITS[u].size, UNITS[u].value), reverse=True)
    loads = [0.0] * parts
    for unit in expanded:
        idx = min(range(parts), key=lambda i: (loads[i], i))
        bins[idx][unit] += 1
        loads[idx] += UNITS[unit].size
    return [dict(b) for b in bins if b]


def make_relative_groups(comp: Mapping[str, int], formation: str, faction: str) -> List[Tuple[float, float, Dict[str, int]]]:
    n = comp_count(comp)
    if formation == "compact" or n < 4:
        return [(0.0, 0.0, dict(comp))]
    if formation == "type_split" and len(comp) > 1:
        items = [(u, c) for u, c in comp.items()]
        offsets = {
            2: [-2.5, 2.5],
            3: [-3.2, 0.0, 3.2],
        }.get(len(items), [0.0] * len(items))
        return [(0.0, offsets[i], {u: c}) for i, (u, c) in enumerate(items)]
    parts = split_composition(comp, 2)
    if len(parts) == 1:
        return [(0.0, 0.0, parts[0])]
    if formation == "split":
        ys = [-2.6, 2.6]
        xs = [0.0, 0.0]
    else:  # staggered
        ys = [-2.2, 2.2] if faction == "ALLY" else [2.2, -2.2]
        xs = [-0.4, 0.4] if faction == "ALLY" else [0.4, -0.4]
    return [(xs[i], ys[i], parts[i]) for i in range(2)]


def emulate_group_units(group: Mapping[str, object]) -> List[Dict[str, object]]:
    faction = str(group["faction"])
    units_map: Mapping[str, int] = group["units"]  # type: ignore[assignment]
    expanded: List[str] = []
    for unit_type, count in units_map.items():
        expanded.extend([unit_type] * int(count))
    size = len(expanded)
    side = int(math.ceil(math.sqrt(size)))
    grid: List[List[str | None]] = [[None for _ in range(side)] for _ in range(side)]
    a = b = 0
    for unit_type in expanded:
        grid[b][a] = unit_type
        a += 1
        if a == side:
            a = 0
            b += 1
    row_radii = [max((UNITS[u].radius if u else 0.0) for u in row) for row in grid]
    prev_row_height = 0.0
    group_height = 2 * sum(row_radii) + (side - 1) * GROUP_BUFFER
    row_widths = [sum((UNITS[u].size if u else 0.0) for u in row) for row in grid]
    group_width = max(row_widths)
    m = 1.0 if faction == "ALLY" else -1.0
    x0 = float(group["x"]) - m * group_width / 2.0
    y = float(group["y"]) - m * group_height / 2.0
    out: List[Dict[str, object]] = []
    for i, row in enumerate(grid):
        x = x0
        y += m * (prev_row_height + row_radii[i])
        prev_row_height = row_radii[i]
        prev_unit_width = 0.0
        for unit_type in row:
            if unit_type is None:
                continue
            info = UNITS[unit_type]
            x += m * (prev_unit_width + info.radius)
            prev_unit_width = info.radius
            out.append({"unit": unit_type, "faction": faction, "x": x, "y": y, "radius": info.radius, "plane": info.plane})
            x += m * GROUP_BUFFER
        y += m * GROUP_BUFFER
    return out


def walkable(terrain: str, x: float, y: float, radius: float = 0.0) -> bool:
    # Conservative analytical safe regions for the three presets used.
    pts = [(x, y), (x + radius, y), (x - radius, y), (x, y + radius), (x, y - radius)]
    for px, py in pts:
        if not (0.0 <= px < MAP_WIDTH and 0.0 <= py < MAP_HEIGHT):
            return False
        ix, iy = int(px), int(py)
        if terrain == "SIMPLE":
            ok = 8 <= iy <= 23
        elif terrain == "NARROW":
            ok = 8 <= iy <= 23 and not (ix in (14, 15) and iy not in (15, 16))
        elif terrain == "OCTAGON":
            # Exact octagon interior by row, reconstructed from the bundled preset.
            if iy < 5 or iy > 26:
                ok = False
            else:
                edge = max(5, 10 - min(iy - 5, 26 - iy))
                right = 31 - edge
                ok = edge <= ix <= right
        else:
            raise KeyError(terrain)
        if not ok:
            return False
    return True


def actual_units(groups: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for g in groups:
        out.extend(emulate_group_units(g))
    return out


def min_cross_distance(units: Sequence[Mapping[str, object]]) -> float:
    allies = [u for u in units if u["faction"] == "ALLY"]
    enemies = [u for u in units if u["faction"] == "ENEMY"]
    return min(math.dist((float(a["x"]), float(a["y"])), (float(e["x"]), float(e["y"]))) for a in allies for e in enemies)


def no_overlap(units: Sequence[Mapping[str, object]]) -> bool:
    for i, a in enumerate(units):
        for b in units[i + 1:]:
            # Different movement planes can overlap without collision.
            if a["plane"] != b["plane"]:
                continue
            dist = math.dist((float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"])))
            if dist + 1e-6 < float(a["radius"]) + float(b["radius"]) + 0.01:
                return False
    return True


def build_groups(ally: Mapping[str, int], enemy: Mapping[str, int], terrain: str, formation: str, target_distance: float, rng: random.Random) -> Tuple[List[Dict[str, object]], str]:
    base_y = 15.8 if terrain == "NARROW" else 16.0
    ally_rel = make_relative_groups(ally, formation, "ALLY")
    enemy_rel = make_relative_groups(enemy, formation, "ENEMY")

    # Small deterministic vertical jitter, kept inside conservative safe regions.
    y_jitter = round(rng.uniform(-0.45, 0.45), 3)

    def groups_for_sep(sep: float) -> List[Dict[str, object]]:
        groups: List[Dict[str, object]] = []
        if terrain == "NARROW":
            # Stay entirely left of the central wall and separate along y.
            ax = ex = 7.0
            ay = 16.0 - sep / 2.0 + y_jitter
            ey = 16.0 + sep / 2.0 - y_jitter
            for dx, dy, comp in ally_rel:
                groups.append({"x": round(ax + dx, 3), "y": round(ay + dy, 3), "faction": "ALLY", "units": comp})
            for dx, dy, comp in enemy_rel:
                groups.append({"x": round(ex + dx, 3), "y": round(ey + dy, 3), "faction": "ENEMY", "units": comp})
        else:
            ax = 16.0 - sep / 2.0
            ex = 16.0 + sep / 2.0
            for dx, dy, comp in ally_rel:
                groups.append({"x": round(ax + dx, 3), "y": round(base_y + y_jitter + dy, 3), "faction": "ALLY", "units": comp})
            for dx, dy, comp in enemy_rel:
                groups.append({"x": round(ex + dx, 3), "y": round(base_y - y_jitter + dy, 3), "faction": "ENEMY", "units": comp})
        return groups

    low, high = 2.0, 28.0
    best = None
    for _ in range(60):
        mid = (low + high) / 2.0
        gs = groups_for_sep(mid)
        units = actual_units(gs)
        d = min_cross_distance(units)
        best = (gs, units, d)
        if d < target_distance:
            low = mid
        else:
            high = mid
    assert best is not None
    groups, units, dist = best

    # If the chosen formation cannot fit safely, progressively fall back.
    if not all(walkable(terrain, float(u["x"]), float(u["y"]), float(u["radius"])) or u["plane"] == "AIR" for u in units) or not no_overlap(units):
        if formation != "compact":
            return build_groups(ally, enemy, terrain, "compact", target_distance, rng)
        raise ValueError(f"Unable to place groups safely: terrain={terrain}, target={target_distance}, d={dist}")
    return groups, formation


def difficulty_from_ratio(ratio: float) -> str:
    if ratio < 1.20:
        return "hard_winnable"
    if ratio < 1.46:
        return "moderate"
    return "easy"


def variant_plan_seen(family_idx: int) -> List[str]:
    if family_idx < 30:
        return ["immediate", "immediate", "near", "near", "near", "medium", "medium", "far"]
    if family_idx < 40:
        return ["immediate", "near", "near", "near", "medium", "medium", "medium", "far"]
    return ["immediate", "near", "near", "medium", "medium", "medium", "far", "far"]


def allocate_classes(total: int, targets: Mapping[str, int]) -> List[str]:
    arr: List[str] = []
    for k in ("immediate", "near", "medium", "far"):
        arr.extend([k] * targets[k])
    assert len(arr) == total
    return arr


def terrain_for(index: int, split: str) -> str:
    if split == "blind_compositional":
        cycle = ["SIMPLE", "NARROW", "OCTAGON", "OCTAGON", "SIMPLE", "NARROW", "OCTAGON", "SIMPLE", "NARROW", "OCTAGON"]
    else:
        cycle = ["SIMPLE", "NARROW", "SIMPLE", "OCTAGON", "SIMPLE", "NARROW", "SIMPLE", "OCTAGON", "NARROW", "SIMPLE"]
    return cycle[index % len(cycle)]


def formation_for(index: int) -> str:
    return FORMATIONS[index % len(FORMATIONS)]


def make_config(family: Family, split: str, variant_idx: int, engagement: str, global_idx: int, rng: random.Random) -> Tuple[Dict[str, object], Dict[str, object]]:
    terrain = terrain_for(global_idx + variant_idx, split)
    # The NARROW preset has a central wall with only a one-tile gate. Keep both
    # teams on the same side for combat-rich immediate/near maps; use open
    # presets for medium/far maps so large units are not accidentally trapped.
    if engagement == "far":
        terrain = "SIMPLE"
    elif terrain == "NARROW" and engagement == "medium":
        terrain = "OCTAGON"
    formation = formation_for(global_idx)
    target = ENGAGEMENT_TARGETS[engagement] + rng.uniform(-0.25, 0.25)
    groups, formation = build_groups(family.ally, family.enemy, terrain, formation, target, rng)
    units = actual_units(groups)
    observed_distance = min_cross_distance(units)
    lo, hi = ENGAGEMENT_BOUNDS[engagement]
    if not lo <= observed_distance <= hi:
        raise ValueError(f"Distance class failure {family.family_id} {split}: {engagement} -> {observed_distance:.3f}")

    ally_groups = [g for g in groups if g["faction"] == "ALLY"]
    total_ally = sum(sum(int(v) for v in g["units"].values()) for g in ally_groups)  # type: ignore[index,union-attr]
    attack_x = sum(float(g["x"]) * sum(int(v) for v in g["units"].values()) for g in ally_groups) / total_ally  # type: ignore[index,union-attr]
    attack_y = sum(float(g["y"]) * sum(int(v) for v in g["units"].values()) for g in ally_groups) / total_ally  # type: ignore[index,union-attr]

    name = f"r2_{split}_{global_idx:04d}"
    config: Dict[str, object] = {
        "name": name,
        "num_allied_units": comp_count(family.ally),
        "num_enemy_units": comp_count(family.enemy),
        "groups": groups,
        "attack_point": [round(attack_x, 3), round(attack_y, 3)],
        "terrain_preset": terrain,
        "num_unit_types": len(GLOBAL_UNIT_TYPE_IDS),
        "ally_has_shields": comp_has_shields(family.ally),
        "enemy_has_shields": comp_has_shields(family.enemy),
        "unit_type_ids": GLOBAL_UNIT_TYPE_IDS,
    }
    ratio = comp_value(family.ally) / comp_value(family.enemy)
    meta: Dict[str, object] = {
        "name": name,
        "split": split,
        "family_id": family.family_id,
        "archetype": family.archetype,
        "heldout_compositional": family.heldout_compositional,
        "variant_index": variant_idx,
        "engagement_class": engagement,
        "initial_min_cross_distance": round(observed_distance, 4),
        "terrain": terrain,
        "formation": formation,
        "num_allies": comp_count(family.ally),
        "num_enemies": comp_count(family.enemy),
        "ally_composition": dict(family.ally),
        "enemy_composition": dict(family.enemy),
        "ally_value": comp_value(family.ally),
        "enemy_value": comp_value(family.enemy),
        "ally_value_ratio": round(ratio, 4),
        "difficulty_proxy": difficulty_from_ratio(ratio),
    }
    return config, meta


def sha256_json(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def static_validate_config(config: Mapping[str, object], expected_name: str | None = None) -> List[str]:
    errors: List[str] = []
    required = {
        "name", "num_allied_units", "num_enemy_units", "groups", "attack_point",
        "terrain_preset", "num_unit_types", "ally_has_shields", "enemy_has_shields", "unit_type_ids",
    }
    if set(config) != required:
        errors.append(f"keys mismatch: missing={required-set(config)}, extra={set(config)-required}")
        return errors
    if expected_name and config["name"] != expected_name:
        errors.append("name does not match filename")
    if config["terrain_preset"] not in TERRAINS:
        errors.append("unsupported terrain")
    if config["unit_type_ids"] != GLOBAL_UNIT_TYPE_IDS:
        errors.append("unit_type_ids not equal to global vocabulary")
    if config["num_unit_types"] != len(GLOBAL_UNIT_TYPE_IDS):
        errors.append("num_unit_types mismatch")
    groups = config["groups"]
    if not isinstance(groups, list) or not groups:
        errors.append("groups missing")
        return errors
    faction_comps: Dict[str, Counter] = {"ALLY": Counter(), "ENEMY": Counter()}
    for gi, group in enumerate(groups):
        if not isinstance(group, dict):
            errors.append(f"group {gi} not object")
            continue
        if set(group) != {"x", "y", "faction", "units"}:
            errors.append(f"group {gi} keys invalid")
            continue
        faction = group["faction"]
        if faction not in faction_comps:
            errors.append(f"group {gi} invalid faction")
            continue
        if not (0 <= float(group["x"]) < MAP_WIDTH and 0 <= float(group["y"]) < MAP_HEIGHT):
            errors.append(f"group {gi} center out of bounds")
        if not isinstance(group["units"], dict) or not group["units"]:
            errors.append(f"group {gi} units invalid")
            continue
        for unit, count in group["units"].items():
            if unit not in UNITS:
                errors.append(f"unknown unit {unit}")
            if not isinstance(count, int) or count <= 0:
                errors.append(f"invalid count {unit}={count}")
            faction_comps[faction][unit] += count
    if sum(faction_comps["ALLY"].values()) != config["num_allied_units"]:
        errors.append("allied count mismatch")
    if sum(faction_comps["ENEMY"].values()) != config["num_enemy_units"]:
        errors.append("enemy count mismatch")
    for faction, flag_key in (("ALLY", "ally_has_shields"), ("ENEMY", "enemy_has_shields")):
        if faction_comps[faction]:
            flags = {UNITS[u].has_shield for u in faction_comps[faction]}
            if len(flags) != 1:
                errors.append(f"{faction} mixes shielded and unshielded units")
            elif bool(config[flag_key]) != next(iter(flags)):
                errors.append(f"{flag_key} does not match unit definitions")
    if faction_comps["ALLY"].get("MEDIVAC", 0) and not can_target_air(faction_comps["ENEMY"]):
        errors.append("enemy cannot target allied medivac")
    if faction_comps["ENEMY"].get("MEDIVAC", 0) and not can_target_air(faction_comps["ALLY"]):
        errors.append("allies cannot target enemy medivac")
    try:
        placed = actual_units(groups)
        for u in placed:
            if u["plane"] != "AIR" and not walkable(str(config["terrain_preset"]), float(u["x"]), float(u["y"]), float(u["radius"])):
                errors.append(f"spawn not walkable: {u}")
        if not no_overlap(placed):
            errors.append("initial unit overlap")
    except Exception as exc:
        errors.append(f"placement emulation failed: {exc}")
    ap = config["attack_point"]
    if not isinstance(ap, list) or len(ap) != 2 or not (0 <= float(ap[0]) < MAP_WIDTH and 0 <= float(ap[1]) < MAP_HEIGHT):
        errors.append("attack_point invalid")
    return errors


def write_dynamic_validator(root: Path) -> None:
    content = r'''#!/usr/bin/env python3
"""Dynamic SMAClite validation for the generated R2-Dreamer map set.

Run from the smac-dreamer repository root after copying `configs/maps/r2_650` in:

    PYTHONPATH=src:external/r2dreamer:external/smaclite \
      python configs/maps/r2_650/validate_in_smaclite.py \
      --root configs/maps/r2_650 --episodes 5

The script instantiates every map, checks finite observations/state/action masks,
executes a deterministic scripted policy, and writes empirical win/timeout data.
"""
from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path
import numpy as np
from smaclite.env.util.direction import Direction


def choose_actions(env):
    avail = env.get_avail_actions()
    actions = []
    for i in range(env.n_agents):
        valid = np.flatnonzero(avail[i])
        if i not in env.agents:
            actions.append(0); continue
        unit = env.agents[i]
        target_pool = env.enemies if unit.combat_type.name == "DAMAGE" else env.agents
        attack_choices = [a for a in valid if a >= 6]
        if attack_choices:
            # Lowest effective-HP valid target; for healers choose most injured target.
            scored = []
            for a in attack_choices:
                tid = a - 6
                if tid not in target_pool: continue
                t = target_pool[tid]
                if unit.combat_type.name == "HEALING":
                    score = (t.hp + t.shield) / max(t.max_hp + t.max_shield, 1e-8)
                else:
                    score = t.hp + t.shield
                scored.append((score, a))
            if scored:
                actions.append(min(scored)[1]); continue
        if unit.combat_type.name == "HEALING":
            actions.append(1 if 1 in valid else int(valid[0])); continue
        # Greedy legal movement toward nearest enemy.
        if env.enemies:
            enemy = min(env.enemies.values(), key=lambda e: float(np.linalg.norm(e.pos-unit.pos)))
            prefs = []
            # Actions 2..5 map directly to SMAClite's Direction enum.
            for a in [2, 3, 4, 5]:
                if a not in valid:
                    continue
                dest = unit.pos + Direction(a - 2).dx_dy * 2
                prefs.append((float(np.linalg.norm(dest - enemy.pos)), a))
            if prefs:
                actions.append(min(prefs)[1]); continue
        actions.append(1 if 1 in valid else int(valid[0]))
    return [int(a) for a in actions]


def run_one(path, seed, max_steps):
    from smaclite.env.smaclite import SMACliteEnv
    env = SMACliteEnv(map_file=str(path), seed=seed)
    obs, info = env.reset(seed=seed)
    assert np.isfinite(np.asarray(obs)).all(), path
    assert np.isfinite(env.get_state()).all(), path
    total = 0.0
    won = False
    done = False
    truncated = False
    for t in range(max_steps):
        actions = choose_actions(env)
        obs, reward, done, truncated, info = env.step(actions)
        assert np.isfinite(np.asarray(obs)).all(), path
        assert np.isfinite(float(reward)), path
        total += float(reward)
        if done or truncated:
            won = bool(info.get('battle_won', False)); break
    env.close()
    return won, (not (done or truncated)), total, t+1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--max-steps', type=int, default=200)
    args = p.parse_args()
    maps = sorted((args.root/'configs').glob('*/*.json'))
    rows=[]
    for idx, path in enumerate(maps,1):
        wins=timeouts=0; returns=[]; lengths=[]
        for ep in range(args.episodes):
            w,to,r,l=run_one(path, 1000+ep, args.max_steps)
            wins += int(w); timeouts += int(to); returns.append(r); lengths.append(l)
        rows.append({'path':str(path),'win_rate':wins/args.episodes,'timeout_rate':timeouts/args.episodes,
                     'mean_return':sum(returns)/len(returns),'mean_length':sum(lengths)/len(lengths)})
        print(f'[{idx:03d}/{len(maps)}] {path.name}: win={rows[-1]["win_rate"]:.2f} timeout={rows[-1]["timeout_rate"]:.2f}')
    out=args.root/'dynamic_validation.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f'Wrote {out}')

if __name__=='__main__': main()
'''
    (root / "validate_in_smaclite.py").write_text(content, encoding="utf-8")


def write_readme(root: Path, summary: Mapping[str, object]) -> None:
    text = f"""# R2-Dreamer × SMAClite 650-map benchmark

This directory contains **650 deterministic custom SMAClite map JSON files** generated for combat-rich R2-Dreamer training and blind evaluation.

## Split

- `configs/train`: 400 maps from 50 seen composition families (8 variants each)
- `configs/validation`: 50 maps from the same 50 families (1 unseen variant each)
- `configs/blind_iid`: 100 maps from the same 50 families (2 unseen variants each)
- `configs/blind_compositional`: 100 maps from 20 composition families absent from train/validation (5 variants each)

The files contain only fields accepted by SMAClite's `MapInfo`; research metadata is stored separately in `manifest.jsonl` and `manifest.csv`.

## Design choices

1. **Combat-rich distances:** train contains exactly 80 immediate, 140 near, 120 medium, and 60 far variants.
2. **Winnability proxy:** every family gives allies a static combat-value advantage between 1.04× and 1.85×. This is a generation filter, not a substitute for empirical simulation.
3. **Shield correctness:** each faction is internally either entirely shielded or entirely unshielded. The faction flags exactly match the bundled unit definitions.
4. **Medivac safety:** a medivac is only used when the opposing composition contains a unit capable of targeting air.
5. **Global type vocabulary:** every map uses the same nine-entry `unit_type_ids` mapping to prevent map-local one-hot meanings.
6. **Spawn checks:** the generator emulates SMAClite's square group placement, rejects overlaps, checks map bounds, and checks ground-unit spawn cells against the selected terrain preset.
7. **Reproducibility:** fixed seed `{SEED}`; rerun `generate_r2_smaclite_650.py` to recreate the dataset.

## Files

- `generate_r2_smaclite_650.py`: self-contained deterministic generator and static validator
- `validate_in_smaclite.py`: dynamic environment smoke test plus scripted-policy evaluation
- `manifest.jsonl` / `manifest.csv`: per-map family, composition, engagement, formation, terrain and difficulty metadata
- `split_manifest.json`: exact files in each split
- `family_catalog.json`: composition families and split status
- `validation_report.json`: static validation results and aggregate distributions
- `checksums.sha256`: content checksums

## Static validation result

- Files: {summary['total_configs']}
- Errors: {summary['validation_errors']}
- Unique content hashes: {summary['unique_hashes']}
- Seed: {SEED}

## Required dynamic validation

Static checks cannot prove game-theoretic winnability. After copying this folder into the repository, run:

```bash
PYTHONPATH=src:external/r2dreamer:external/smaclite \\
python configs/maps/r2_650/validate_in_smaclite.py \\
  --root configs/maps/r2_650 \\
  --episodes 5 \\
  --max-steps 200
```

Use the resulting `dynamic_validation.csv` to remove or rebalance maps with high timeout rate or zero scripted-policy wins before the final expensive R2-Dreamer run. A scripted baseline is deliberately only a filter; final winnability should also be checked with a trained reference policy.

## Loader recommendation

Do not randomly split one map folder at runtime. Point the trainer and evaluator at the explicit split directories or consume `split_manifest.json`. Select checkpoints using **blind/validation win rate and original SMAClite return**, never shaped return.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def generate(out_root: Path, make_zip: bool = True) -> Dict[str, object]:
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "configs").mkdir(parents=True)
    rng = random.Random(SEED)
    seen, held = build_families()
    for f in seen + held:
        validate_family(f)

    records: List[Dict[str, object]] = []
    split_files: Dict[str, List[str]] = defaultdict(list)
    configs_by_name: Dict[str, Dict[str, object]] = {}
    global_index = 1

    # Training: exact 80/140/120/60 class counts through per-family patterns.
    for fi, family in enumerate(seen):
        plan = variant_plan_seen(fi)
        for vi, engagement in enumerate(plan, 1):
            cfg, meta = make_config(family, "train", vi, engagement, global_index, rng)
            configs_by_name[cfg["name"]] = cfg
            records.append(meta)
            split_files["train"].append(f"configs/train/{cfg['name']}.json")
            global_index += 1

    # Validation target 10/18/15/7.
    val_classes = allocate_classes(50, {"immediate": 10, "near": 18, "medium": 15, "far": 7})
    rng.shuffle(val_classes)
    for fi, family in enumerate(seen):
        cfg, meta = make_config(family, "validation", 9, val_classes[fi], global_index, rng)
        configs_by_name[cfg["name"]] = cfg
        records.append(meta)
        split_files["validation"].append(f"configs/validation/{cfg['name']}.json")
        global_index += 1

    # Blind IID exact 20/35/30/15.
    iid_classes = allocate_classes(100, {"immediate": 20, "near": 35, "medium": 30, "far": 15})
    rng.shuffle(iid_classes)
    k = 0
    for family in seen:
        for vi in (10, 11):
            cfg, meta = make_config(family, "blind_iid", vi, iid_classes[k], global_index, rng)
            configs_by_name[cfg["name"]] = cfg
            records.append(meta)
            split_files["blind_iid"].append(f"configs/blind_iid/{cfg['name']}.json")
            global_index += 1
            k += 1

    # Blind compositional exact 20/35/30/15, five variants per held-out family.
    comp_classes = allocate_classes(100, {"immediate": 20, "near": 35, "medium": 30, "far": 15})
    rng.shuffle(comp_classes)
    k = 0
    for family in held:
        for vi in range(1, 6):
            cfg, meta = make_config(family, "blind_compositional", vi, comp_classes[k], global_index, rng)
            configs_by_name[cfg["name"]] = cfg
            records.append(meta)
            split_files["blind_compositional"].append(f"configs/blind_compositional/{cfg['name']}.json")
            global_index += 1
            k += 1

    assert len(records) == 650
    assert {k: len(v) for k, v in split_files.items()} == {
        "train": 400, "validation": 50, "blind_iid": 100, "blind_compositional": 100,
    }

    # Write JSON configs.
    for split, paths in split_files.items():
        d = out_root / "configs" / split
        d.mkdir(parents=True, exist_ok=True)
        for rel in paths:
            name = Path(rel).stem
            (out_root / rel).write_text(json.dumps(configs_by_name[name], indent=2) + "\n", encoding="utf-8")

    # Static validation and duplicate detection.
    validation_errors: Dict[str, List[str]] = {}
    content_hashes: Dict[str, str] = {}
    semantic_hashes: Counter = Counter()
    for split, paths in split_files.items():
        for rel in paths:
            p = out_root / rel
            cfg = json.loads(p.read_text())
            errs = static_validate_config(cfg, p.stem)
            if errs:
                validation_errors[rel] = errs
            content_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
            semantic = dict(cfg)
            semantic.pop("name")
            semantic_hashes[sha256_json(semantic)] += 1
    duplicate_semantics = [h for h, n in semantic_hashes.items() if n > 1]
    if duplicate_semantics:
        raise ValueError(f"Duplicate semantic configs: {len(duplicate_semantics)}")
    if validation_errors:
        raise ValueError(f"Static validation errors: {validation_errors}")

    # Manifests.
    with (out_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            rec = dict(rec)
            rel = next(r for r in split_files[rec["split"]] if Path(r).stem == rec["name"])
            rec["path"] = rel
            rec["sha256"] = content_hashes[rel]
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    csv_fields = [
        "name", "path", "split", "family_id", "archetype", "heldout_compositional", "variant_index",
        "engagement_class", "initial_min_cross_distance", "terrain", "formation", "num_allies", "num_enemies",
        "ally_value", "enemy_value", "ally_value_ratio", "difficulty_proxy", "ally_composition", "enemy_composition", "sha256",
    ]
    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for rec0 in records:
            rec = dict(rec0)
            rel = next(r for r in split_files[rec["split"]] if Path(r).stem == rec["name"])
            rec["path"] = rel
            rec["sha256"] = content_hashes[rel]
            rec["ally_composition"] = json.dumps(rec["ally_composition"], sort_keys=True)
            rec["enemy_composition"] = json.dumps(rec["enemy_composition"], sort_keys=True)
            w.writerow({k: rec[k] for k in csv_fields})

    (out_root / "split_manifest.json").write_text(json.dumps({
        "seed": SEED,
        "global_unit_type_ids": GLOBAL_UNIT_TYPE_IDS,
        "splits": split_files,
    }, indent=2) + "\n", encoding="utf-8")
    family_catalog = {
        "seen_families": [asdict(f) for f in seen],
        "heldout_compositional_families": [asdict(f) for f in held],
    }
    (out_root / "family_catalog.json").write_text(json.dumps(family_catalog, indent=2) + "\n", encoding="utf-8")

    split_counts = Counter(str(r["split"]) for r in records)
    engagement_counts = {split: Counter(str(r["engagement_class"]) for r in records if r["split"] == split) for split in split_counts}
    terrain_counts = {split: Counter(str(r["terrain"]) for r in records if r["split"] == split) for split in split_counts}
    formation_counts = {split: Counter(str(r["formation"]) for r in records if r["split"] == split) for split in split_counts}
    difficulty_counts = {split: Counter(str(r["difficulty_proxy"]) for r in records if r["split"] == split) for split in split_counts}
    ratios = [float(r["ally_value_ratio"]) for r in records]
    summary = {
        "seed": SEED,
        "total_configs": len(records),
        "validation_errors": len(validation_errors),
        "unique_hashes": len(semantic_hashes),
        "split_counts": dict(split_counts),
        "engagement_counts": {k: dict(v) for k, v in engagement_counts.items()},
        "terrain_counts": {k: dict(v) for k, v in terrain_counts.items()},
        "formation_counts": {k: dict(v) for k, v in formation_counts.items()},
        "difficulty_proxy_counts": {k: dict(v) for k, v in difficulty_counts.items()},
        "ally_value_ratio": {
            "min": min(ratios), "max": max(ratios), "mean": statistics.mean(ratios), "median": statistics.median(ratios),
        },
        "global_unit_type_ids": GLOBAL_UNIT_TYPE_IDS,
        "static_checks": [
            "exact JSON schema", "name/filename agreement", "unit/count validity", "team count agreement",
            "global unit-type vocabulary", "shield flag exactness", "no mixed shield capability within a faction",
            "medivac targetability", "group placement emulation", "spawn bounds", "terrain walkability",
            "same-plane overlap rejection", "attack-point bounds", "engagement bucket", "duplicate semantic map rejection",
        ],
        "dynamic_validation_required": True,
    }
    (out_root / "validation_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with (out_root / "checksums.sha256").open("w", encoding="utf-8") as f:
        for rel, digest in sorted(content_hashes.items()):
            f.write(f"{digest}  {rel}\n")

    # Copy this generator and write dynamic validator/readme.
    source_path = Path(__file__).resolve()
    if source_path != out_root / "generate_r2_smaclite_650.py":
        shutil.copy2(source_path, out_root / "generate_r2_smaclite_650.py")
    write_dynamic_validator(out_root)
    write_readme(out_root, summary)

    if make_zip:
        zip_path = out_root.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in sorted(out_root.rglob("*")):
                if p.is_file():
                    zf.write(p, Path(out_root.name) / p.relative_to(out_root))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("r2_smaclite_650_configs"))
    p.add_argument("--no-zip", action="store_true")
    args = p.parse_args()
    summary = generate(args.out, not args.no_zip)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
