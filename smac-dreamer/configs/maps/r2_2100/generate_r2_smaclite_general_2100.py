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
from typing import Dict, List, Mapping, Sequence, Tuple

SEED = 18062026
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

ENGAGEMENT_TARGETS = {"immediate": 5.4, "near": 8.8, "medium": 12.8, "far": 17.0}
ENGAGEMENT_BOUNDS = {
    "immediate": (4.45, 6.25),
    "near": (6.9, 10.6),
    "medium": (10.7, 14.9),
    "far": (14.9, 19.9),
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
    origin: str = "new"


def C(**kwargs: int) -> Dict[str, int]:
    return {k: int(v) for k, v in kwargs.items() if int(v) > 0}


def comp_count(comp: Mapping[str, int]) -> int:
    return sum(int(v) for v in comp.values())


def comp_value(comp: Mapping[str, int]) -> float:
    return sum(UNITS[u].value * int(n) for u, n in comp.items())


def comp_has_shields(comp: Mapping[str, int]) -> bool:
    flags = {UNITS[u].has_shield for u in comp}
    if len(flags) != 1:
        raise ValueError(f"Faction mixes shielded and unshielded units: {comp}")
    return next(iter(flags))


def can_target_air(comp: Mapping[str, int]) -> bool:
    return any("AIR" in UNITS[u].valid_targets and int(n) > 0 for u, n in comp.items())


def canonical_pair(ally: Mapping[str, int], enemy: Mapping[str, int]):
    return tuple(sorted(ally.items())), tuple(sorted(enemy.items()))


def validate_family(f: Family) -> None:
    # max allies deliberately kept at 9 to preserve compatibility with the existing R2-650 model.
    if not 2 <= comp_count(f.ally) <= 9:
        raise ValueError(f"{f.family_id}: allied team size outside [2,9]: {f.ally}")
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
    if not 0.92 <= ratio <= 1.86:
        raise ValueError(f"{f.family_id}: ally value ratio {ratio:.3f} outside [0.92,1.86]")


def load_legacy_families(source_root: Path) -> Tuple[List[Family], List[Family]]:
    catalog = json.loads((source_root / "family_catalog.json").read_text(encoding="utf-8"))
    seen = [
        Family(
            family_id=str(x["family_id"]), archetype=str(x["archetype"]),
            ally=dict(x["ally"]), enemy=dict(x["enemy"]),
            heldout_compositional=False, origin="legacy_r2_1300",
        )
        for x in catalog["seen_families"]
    ]
    held = [
        Family(
            family_id=str(x["family_id"]), archetype=str(x["archetype"]),
            ally=dict(x["ally"]), enemy=dict(x["enemy"]),
            heldout_compositional=True, origin="legacy_r2_1300",
        )
        for x in catalog["heldout_compositional_families"]
    ]
    return seen, held


def style_compositions() -> Dict[str, List[Dict[str, int]]]:
    styles: Dict[str, List[Dict[str, int]]] = {}
    styles["marauder"] = [C(MARAUDER=n) for n in range(2, 10)]
    styles["zealot"] = [C(ZEALOT=n) for n in range(2, 10)]
    styles["zergling"] = [C(ZERGLING=n) for n in range(3, 11)]
    styles["marine"] = [C(MARINE=n) for n in range(2, 10)]
    styles["stalker"] = [C(STALKER=n) for n in range(2, 10)]
    styles["bio"] = [C(MARINE=m, MARAUDER=r) for m in range(1, 7) for r in range(1, 5) if 3 <= m + r <= 9]
    styles["mmm"] = [C(MARINE=m, MARAUDER=r, MEDIVAC=1) for m in range(2, 7) for r in range(1, 4) if m + r + 1 <= 9]
    styles["marine_medivac"] = [C(MARINE=m, MEDIVAC=1) for m in range(2, 8) if m + 1 <= 9]
    styles["swarm"] = [C(ZERGLING=z, BANELING=b) for z in range(2, 9) for b in range(1, 5) if 4 <= z + b <= 10]
    styles["swarm_aa"] = [C(MARINE=1, ZERGLING=z, BANELING=b) for z in range(2, 7) for b in range(1, 4) if 4 <= 1 + z + b <= 10]
    styles["stalker_zealot"] = [C(STALKER=s, ZEALOT=z) for s in range(1, 6) for z in range(1, 6) if 3 <= s + z <= 9]
    styles["stalker_colossus"] = [C(STALKER=s, COLOSSUS=c) for s in range(1, 7) for c in (1, 2) if 2 <= s + c <= 9]
    styles["zealot_colossus"] = [C(ZEALOT=z, COLOSSUS=c) for z in range(1, 7) for c in (1, 2) if 2 <= z + c <= 9]
    styles["protoss_triad"] = [
        C(STALKER=s, ZEALOT=z, COLOSSUS=c)
        for s in range(1, 5) for z in range(1, 5) for c in (1, 2)
        if 4 <= s + z + c <= 9
    ]
    styles["spine"] = [C(SPINE_CRAWLER=n) for n in range(2, 6)]
    styles["spine_swarm"] = [C(SPINE_CRAWLER=p, ZERGLING=z) for p in range(1, 4) for z in range(2, 7) if 3 <= p + z <= 9]
    return styles


def _pair_is_safe(ally: Mapping[str, int], enemy: Mapping[str, int]) -> bool:
    if ally.get("MEDIVAC", 0) and not can_target_air(enemy):
        return False
    if enemy.get("MEDIVAC", 0) and not can_target_air(ally):
        return False
    return True


def select_pairs(
    ally_style: str,
    enemy_style: str,
    n: int,
    target_ratios: Sequence[float],
    styles: Mapping[str, Sequence[Mapping[str, int]]],
    forbidden: set,
) -> List[Tuple[Dict[str, int], Dict[str, int]]]:
    candidates = []
    for ally0 in styles[ally_style]:
        for enemy0 in styles[enemy_style]:
            ally, enemy = dict(ally0), dict(enemy0)
            key = canonical_pair(ally, enemy)
            if key in forbidden or not _pair_is_safe(ally, enemy):
                continue
            if not (2 <= comp_count(ally) <= 9 and 1 <= comp_count(enemy) <= 10):
                continue
            ratio = comp_value(ally) / comp_value(enemy)
            if 0.92 <= ratio <= 1.65:
                candidates.append((ally, enemy, ratio, comp_count(ally) + comp_count(enemy)))
    if len(candidates) < n:
        raise ValueError(f"Not enough candidates for {ally_style} vs {enemy_style}: {len(candidates)}")
    totals = sorted(x[3] for x in candidates)
    qidx = [round(i * (len(totals) - 1) / max(n - 1, 1)) for i in range(n)]
    target_totals = [totals[i] for i in qidx]
    selected: List[Tuple[Dict[str, int], Dict[str, int]]] = []
    used = set()
    for target_ratio, target_total in zip(target_ratios, target_totals):
        ranked = []
        for ally, enemy, ratio, total in candidates:
            key = canonical_pair(ally, enemy)
            if key in used:
                continue
            score = abs(math.log(ratio / target_ratio)) + 0.025 * abs(total - target_total)
            ranked.append((score, abs(ratio - target_ratio), total, ally, enemy))
        ranked.sort(key=lambda x: (x[0], x[1], x[2], canonical_pair(x[3], x[4])))
        if not ranked:
            raise ValueError(f"Selection exhausted for {ally_style} vs {enemy_style}")
        _, _, _, ally, enemy = ranked[0]
        used.add(canonical_pair(ally, enemy))
        selected.append((ally, enemy))
    return selected


def build_new_families(legacy_seen: Sequence[Family], legacy_held: Sequence[Family]) -> Tuple[List[Family], List[Family]]:
    styles = style_compositions()
    forbidden = {canonical_pair(f.ally, f.enemy) for f in list(legacy_seen) + list(legacy_held)}
    seen_specs = [
        ("marauder_mirror", "marauder", "marauder"),
        ("zealot_mirror", "zealot", "zealot"),
        ("zergling_mirror", "zergling", "zergling"),
        ("marine_vs_zergling", "marine", "zergling"),
        ("stalker_vs_marine", "stalker", "marine"),
        ("marauder_vs_zealot", "marauder", "zealot"),
        ("swarm_vs_bio", "swarm", "bio"),
        ("bio_vs_stalker_zealot", "bio", "stalker_zealot"),
        ("protoss_triad_vs_bio", "protoss_triad", "bio"),
        ("bio_vs_spine_swarm", "bio", "spine_swarm"),
    ]
    held_specs = [
        ("marine_medivac_vs_swarm_aa", "marine_medivac", "swarm_aa"),
        ("mmm_vs_stalker_colossus", "mmm", "stalker_colossus"),
        ("stalker_zealot_vs_mmm", "stalker_zealot", "mmm"),
        ("protoss_triad_vs_mmm", "protoss_triad", "mmm"),
        ("spine_swarm_vs_stalker_zealot", "spine_swarm", "stalker_zealot"),
        ("swarm_vs_stalker_zealot", "swarm", "stalker_zealot"),
        ("bio_vs_stalker_colossus", "bio", "stalker_colossus"),
        ("stalker_colossus_vs_bio", "stalker_colossus", "bio"),
        ("zealot_colossus_vs_swarm", "zealot_colossus", "swarm"),
        ("mmm_vs_protoss_triad", "mmm", "protoss_triad"),
    ]
    new_seen: List[Family] = []
    for archetype, ally_style, enemy_style in seen_specs:
        pairs = select_pairs(ally_style, enemy_style, 5, [0.96, 1.04, 1.14, 1.30, 1.50], styles, forbidden)
        for idx, (ally, enemy) in enumerate(pairs, 1):
            f = Family(f"gen_{archetype}_{idx:02d}", archetype, ally, enemy, False, "general_v2")
            forbidden.add(canonical_pair(ally, enemy))
            new_seen.append(f)
    new_held: List[Family] = []
    for archetype, ally_style, enemy_style in held_specs:
        pairs = select_pairs(ally_style, enemy_style, 2, [0.98, 1.35], styles, forbidden)
        for idx, (ally, enemy) in enumerate(pairs, 1):
            f = Family(f"gen_{archetype}_{idx:02d}", archetype, ally, enemy, True, "general_v2")
            forbidden.add(canonical_pair(ally, enemy))
            new_held.append(f)
    assert len(new_seen) == 50
    assert len(new_held) == 20
    return new_seen, new_held


def split_composition(comp: Mapping[str, int], parts: int) -> List[Dict[str, int]]:
    parts = max(1, min(parts, comp_count(comp)))
    bins: List[Dict[str, int]] = [defaultdict(int) for _ in range(parts)]
    expanded: List[str] = []
    for unit, count in comp.items():
        expanded.extend([unit] * int(count))
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
        items = list(comp.items())
        offsets = {2: [-2.5, 2.5], 3: [-3.2, 0.0, 3.2]}.get(len(items), [0.0] * len(items))
        return [(0.0, offsets[i], {u: int(c)}) for i, (u, c) in enumerate(items)]
    parts = split_composition(comp, 2)
    if len(parts) == 1:
        return [(0.0, 0.0, parts[0])]
    if formation == "split":
        laterals, forwards = [-2.6, 2.6], [0.0, 0.0]
    else:
        laterals = [-2.2, 2.2] if faction == "ALLY" else [2.2, -2.2]
        forwards = [-0.4, 0.4] if faction == "ALLY" else [0.4, -0.4]
    return [(forwards[i], laterals[i], parts[i]) for i in range(2)]


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
    for group in groups:
        out.extend(emulate_group_units(group))
    return out


def min_cross_distance(units: Sequence[Mapping[str, object]]) -> float:
    allies = [u for u in units if u["faction"] == "ALLY"]
    enemies = [u for u in units if u["faction"] == "ENEMY"]
    return min(math.dist((float(a["x"]), float(a["y"])), (float(e["x"]), float(e["y"]))) for a in allies for e in enemies)


def no_overlap(units: Sequence[Mapping[str, object]]) -> bool:
    for i, a in enumerate(units):
        for b in units[i + 1:]:
            if a["plane"] != b["plane"]:
                continue
            dist = math.dist((float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"])))
            if dist + 1e-6 < float(a["radius"]) + float(b["radius"]) + 0.01:
                return False
    return True


def orientation_for(terrain: str, engagement: str, layout: str, variant_seed: int) -> Tuple[Tuple[float, float], Tuple[float, float], str]:
    # Returns centre, direction from allies to enemies, and resolved layout name.
    side = -1 if variant_seed % 2 == 0 else 1
    if terrain == "NARROW":
        if engagement in ("immediate", "near"):
            # Same-side vertical encounters exercise choke-adjacent movement without requiring
            # a wall crossing before first contact.
            centre = (7.0 if side < 0 else 25.0, 16.0)
            return centre, (0.0, 1.0), "narrow_same_side"
        # Medium/far encounters begin on opposite sides of the central gate and explicitly test
        # navigation through the one-tile passage.
        return (16.0, 15.5 if side < 0 else 16.5), (1.0, 0.0), "narrow_cross_gate"
    directions = {
        "horizontal": (1.0, 0.0),
        "vertical": (0.0, 1.0),
        "diag_up": (0.9239, 0.3827),
        "diag_down": (0.9239, -0.3827),
    }
    d = directions.get(layout, directions["horizontal"])
    # SIMPLE has a horizontal walkable strip; avoid far vertical placements near its edges.
    if terrain == "SIMPLE" and engagement == "far" and layout == "vertical":
        d, layout = directions["horizontal"], "horizontal"
    return (16.0, 16.0), d, layout


def build_groups(
    ally: Mapping[str, int], enemy: Mapping[str, int], terrain: str, formation: str,
    target_distance: float, layout: str, rng: random.Random, variant_seed: int,
) -> Tuple[List[Dict[str, object]], str, str]:
    centre, direction, resolved_layout = orientation_for(terrain, "far" if target_distance >= 14.9 else "medium" if target_distance >= 10.7 else "near" if target_distance >= 6.9 else "immediate", layout, variant_seed)
    dx, dy = direction
    norm = math.hypot(dx, dy)
    dx, dy = dx / norm, dy / norm
    px, py = -dy, dx
    ally_rel = make_relative_groups(ally, formation, "ALLY")
    enemy_rel = make_relative_groups(enemy, formation, "ENEMY")
    # Small centre jitter broadens layouts while preserving class bounds.
    jitter_long = rng.uniform(-0.25, 0.25)
    jitter_lat = rng.uniform(-0.35, 0.35)
    cx = centre[0] + dx * jitter_long + px * jitter_lat
    cy = centre[1] + dy * jitter_long + py * jitter_lat

    def groups_for_sep(sep: float, chosen_formation: str) -> List[Dict[str, object]]:
        a_rel = make_relative_groups(ally, chosen_formation, "ALLY")
        e_rel = make_relative_groups(enemy, chosen_formation, "ENEMY")
        ax, ay = cx - dx * sep / 2.0, cy - dy * sep / 2.0
        ex, ey = cx + dx * sep / 2.0, cy + dy * sep / 2.0
        groups: List[Dict[str, object]] = []
        for longitudinal, lateral, comp in a_rel:
            groups.append({
                "x": round(ax + dx * longitudinal + px * lateral, 3),
                "y": round(ay + dy * longitudinal + py * lateral, 3),
                "faction": "ALLY", "units": comp,
            })
        for longitudinal, lateral, comp in e_rel:
            groups.append({
                "x": round(ex + dx * longitudinal + px * lateral, 3),
                "y": round(ey + dy * longitudinal + py * lateral, 3),
                "faction": "ENEMY", "units": comp,
            })
        return groups

    def solve(chosen_formation: str):
        low, high = 2.0, 28.0
        best = None
        for _ in range(64):
            mid = (low + high) / 2.0
            groups = groups_for_sep(mid, chosen_formation)
            units = actual_units(groups)
            dist = min_cross_distance(units)
            best = (groups, units, dist)
            if dist < target_distance:
                low = mid
            else:
                high = mid
        return best

    for chosen_formation in (formation, "compact") if formation != "compact" else ("compact",):
        solved = solve(chosen_formation)
        assert solved is not None
        groups, units, dist = solved
        placement_ok = all(
            u["plane"] == "AIR" or walkable(terrain, float(u["x"]), float(u["y"]), float(u["radius"]))
            for u in units
        ) and no_overlap(units)
        if placement_ok:
            return groups, chosen_formation, resolved_layout
    raise ValueError(
        f"Unable to place groups safely: terrain={terrain}, formation={formation}, "
        f"layout={resolved_layout}, target={target_distance:.3f}"
    )


def difficulty_bucket(ratio: float) -> str:
    if ratio < 0.99:
        return "slight_disadvantage"
    if ratio < 1.10:
        return "balanced"
    if ratio < 1.35:
        return "moderate_advantage"
    return "strong_advantage"


# 12 training variants per seen family: exactly 4 per terrain, 3 per formation.
TRAIN_SPECS = [
    ("immediate", "NARROW", "compact", "vertical"),
    ("immediate", "OCTAGON", "split", "diag_up"),
    ("immediate", "SIMPLE", "staggered", "horizontal"),
    ("near", "NARROW", "type_split", "vertical"),
    ("near", "OCTAGON", "compact", "vertical"),
    ("near", "OCTAGON", "split", "diag_down"),
    ("near", "NARROW", "staggered", "vertical"),
    ("medium", "SIMPLE", "type_split", "horizontal"),
    ("medium", "OCTAGON", "compact", "diag_up"),
    ("medium", "NARROW", "split", "horizontal"),
    ("far", "SIMPLE", "staggered", "horizontal"),
    ("far", "SIMPLE", "type_split", "horizontal"),
]

# Balanced reusable schedule for validation/blind variants.
EVAL_SPECS = [
    ("immediate", "OCTAGON", "compact", "horizontal"),
    ("near", "NARROW", "split", "vertical"),
    ("medium", "OCTAGON", "staggered", "diag_up"),
    ("far", "SIMPLE", "type_split", "horizontal"),
    ("near", "OCTAGON", "compact", "vertical"),
    ("medium", "NARROW", "split", "horizontal"),
    ("immediate", "OCTAGON", "staggered", "diag_down"),
    ("near", "SIMPLE", "type_split", "diag_up"),
    ("far", "SIMPLE", "compact", "horizontal"),
    ("medium", "SIMPLE", "split", "diag_down"),
    ("immediate", "NARROW", "staggered", "vertical"),
    ("near", "NARROW", "compact", "vertical"),
]


def make_config(
    family: Family, split: str, variant_idx: int, global_idx: int, spec: Tuple[str, str, str, str], rng: random.Random,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    engagement, terrain, formation, layout = spec
    target_base = ENGAGEMENT_TARGETS[engagement]
    if engagement == "far" and terrain == "OCTAGON":
        target_base = 15.6
    elif engagement == "far" and terrain == "NARROW":
        target_base = 16.0
    target = target_base + rng.uniform(-0.20, 0.20)
    groups, formation_used, layout_used = build_groups(
        family.ally, family.enemy, terrain, formation, target, layout, rng, global_idx + variant_idx,
    )
    units = actual_units(groups)
    observed_distance = min_cross_distance(units)
    lo, hi = ENGAGEMENT_BOUNDS[engagement]
    if not lo <= observed_distance <= hi:
        raise ValueError(f"Distance class failure {family.family_id} {split}: {engagement} -> {observed_distance:.3f}")

    ally_groups = [g for g in groups if g["faction"] == "ALLY"]
    total_ally = sum(sum(int(v) for v in g["units"].values()) for g in ally_groups)  # type: ignore[index,union-attr]
    attack_x = sum(float(g["x"]) * sum(int(v) for v in g["units"].values()) for g in ally_groups) / total_ally  # type: ignore[index,union-attr]
    attack_y = sum(float(g["y"]) * sum(int(v) for v in g["units"].values()) for g in ally_groups) / total_ally  # type: ignore[index,union-attr]

    name = f"r2g_{split}_{global_idx:04d}"
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
    # Static seed score for a future adaptive task sampler; runtime empirical metrics should replace it.
    static_difficulty_score = max(0.0, min(1.0, 0.50 + (1.10 - ratio) / 0.55))
    meta: Dict[str, object] = {
        "name": name,
        "split": split,
        "family_id": family.family_id,
        "archetype": family.archetype,
        "family_origin": family.origin,
        "heldout_compositional": family.heldout_compositional,
        "variant_index": variant_idx,
        "engagement_class": engagement,
        "initial_min_cross_distance": round(observed_distance, 4),
        "terrain": terrain,
        "formation": formation_used,
        "layout": layout_used,
        "num_allies": comp_count(family.ally),
        "num_enemies": comp_count(family.enemy),
        "ally_composition": dict(family.ally),
        "enemy_composition": dict(family.enemy),
        "ally_value": comp_value(family.ally),
        "enemy_value": comp_value(family.enemy),
        "ally_value_ratio": round(ratio, 4),
        "difficulty_proxy": difficulty_bucket(ratio),
        "static_difficulty_score": round(static_difficulty_score, 6),
    }
    return config, meta


def sha256_json(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def static_validate_config(config: Mapping[str, object], expected_name: str | None = None) -> List[str]:
    errors: List[str] = []
    required = {
        "name", "num_allied_units", "num_enemy_units", "groups", "attack_point", "terrain_preset",
        "num_unit_types", "ally_has_shields", "enemy_has_shields", "unit_type_ids",
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
        if not isinstance(group, dict) or set(group) != {"x", "y", "faction", "units"}:
            errors.append(f"group {gi} invalid")
            continue
        faction = group["faction"]
        if faction not in faction_comps:
            errors.append(f"group {gi} invalid faction")
            continue
        if not (0 <= float(group["x"]) < MAP_WIDTH and 0 <= float(group["y"]) < MAP_HEIGHT):
            errors.append(f"group {gi} centre out of bounds")
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
                errors.append(f"{flag_key} mismatch")
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
    source = r'''#!/usr/bin/env python3
"""Dynamic SMAClite validation for r2_smaclite_general_2100_configs.

Run from the smac-dreamer repository root after copying the dataset to configs/maps/:

PYTHONPATH=src:external/r2dreamer:external/smaclite \
python configs/maps/r2_smaclite_general_2100_configs/validate_in_smaclite.py \
  --root configs/maps/r2_smaclite_general_2100_configs --episodes 3 --max-steps 200
"""
from __future__ import annotations
import argparse, csv
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
            scored = []
            for a in attack_choices:
                tid = a - 6
                if tid not in target_pool: continue
                target = target_pool[tid]
                if unit.combat_type.name == "HEALING":
                    score = (target.hp + target.shield) / max(target.max_hp + target.max_shield, 1e-8)
                else:
                    score = target.hp + target.shield
                scored.append((score, a))
            if scored:
                actions.append(min(scored)[1]); continue
        if unit.combat_type.name == "HEALING":
            actions.append(1 if 1 in valid else int(valid[0])); continue
        if env.enemies:
            enemy = min(env.enemies.values(), key=lambda e: float(np.linalg.norm(e.pos - unit.pos)))
            prefs = []
            for a in (2, 3, 4, 5):
                if a not in valid: continue
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
    total = 0.0; won = False; done = False; truncated = False
    for t in range(max_steps):
        obs, reward, done, truncated, info = env.step(choose_actions(env))
        assert np.isfinite(np.asarray(obs)).all(), path
        assert np.isfinite(float(reward)), path
        total += float(reward)
        if done or truncated:
            won = bool(info.get("battle_won", False)); break
    env.close()
    return won, not (done or truncated), total, t + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--split", choices=("all", "train", "validation", "blind_iid", "blind_compositional"), default="all")
    args = p.parse_args()
    pattern = "*/*.json" if args.split == "all" else f"{args.split}/*.json"
    maps = sorted((args.root / "configs").glob(pattern))
    rows = []
    for idx, path in enumerate(maps, 1):
        wins = timeouts = 0; returns = []; lengths = []
        for ep in range(args.episodes):
            win, timeout, ret, length = run_one(path, 1000 + ep, args.max_steps)
            wins += int(win); timeouts += int(timeout); returns.append(ret); lengths.append(length)
        row = {"path": str(path), "win_rate": wins / args.episodes, "timeout_rate": timeouts / args.episodes,
               "mean_return": sum(returns) / len(returns), "mean_length": sum(lengths) / len(lengths)}
        rows.append(row)
        print(f"[{idx:04d}/{len(maps)}] {path.name}: win={row['win_rate']:.2f} timeout={row['timeout_rate']:.2f}")
    out = args.root / f"dynamic_validation_{args.split}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
'''
    (root / "validate_in_smaclite.py").write_text(source, encoding="utf-8")


def write_readme(root: Path, summary: Mapping[str, object]) -> None:
    text = f"""# R2-Dreamer × SMAClite General Policy 2100-map dataset

This package contains **{summary['total_configs']} deterministic SMAClite maps** designed as a broader successor to the 1300-map pool while remaining shape-compatible with the existing R2-650/R2-1300 model.

## Split

- `configs/train`: 1,200 maps from 100 seen composition families, 12 variants per family
- `configs/validation`: 200 maps from the same 100 families, 2 unseen variants per family
- `configs/blind_iid`: 300 maps from the same 100 families, 3 unseen variants per family
- `configs/blind_compositional`: 400 maps from 40 composition families absent from train/validation, 10 variants per family

The 100 seen families retain all **50 original R2-1300 families** and add **50 new families**. The compositional blind split retains all **20 original held-out families** and adds **20 new held-out families**.

## Main improvements

1. **Broader compositions:** 100 train families instead of 50, including balanced and slightly ally-disadvantaged matchups.
2. **Balanced training terrain:** exactly 400 SIMPLE, 400 NARROW and 400 OCTAGON training maps.
3. **Choke traversal:** medium NARROW maps and some evaluation maps start teams on opposite sides of the central gate.
4. **More layout orientations:** horizontal, vertical and diagonal deployments, alongside four formation modes.
5. **Transfer compatibility:** maximum allies=9, enemies=10, actions=16 and the same nine-entry global unit vocabulary.
6. **Curriculum metadata:** every map has a stable family ID, family origin and a static seed difficulty score. Replace the seed score with empirical win/timeout/EHP statistics during training.
7. **Strict split discipline:** validation is for checkpoint selection; blind splits are for post-training evaluation only.

## Static validation

- Total maps: {summary['total_configs']}
- Validation errors: {summary['validation_errors']}
- Unique semantic configurations: {summary['unique_hashes']}
- Seed: {SEED}

Static checks cover schema, counts, unit vocabulary, shield consistency, medivac targetability, placement emulation, terrain walkability, overlap, engagement bucket and semantic duplicates.

## Required dynamic validation

Static validation cannot prove practical winnability or navigation quality. Run at least one scripted episode over all maps before an expensive training run:

```bash
PYTHONPATH=src:external/r2dreamer:external/smaclite \\
python configs/maps/r2_smaclite_general_2100_configs/validate_in_smaclite.py \\
  --root configs/maps/r2_smaclite_general_2100_configs \\
  --episodes 1 --max-steps 200
```

Then rerun suspicious or hard configurations with 3–5 episodes.

## Suggested continuation setup

- Resume the existing checkpoint with the same observation/action dimensions.
- Use shuffled round-robin for the first 2–3 full passes.
- Then use a mixture such as 75% family-balanced uniform + 25% empirical hard-map sampling.
- Keep blind splits untouched until final evaluation.
- Select checkpoints using macro validation win rate, with original return as tie-breaker.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def generate(out_root: Path, source_root: Path, make_zip: bool = True) -> Dict[str, object]:
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "configs").mkdir(parents=True)
    rng = random.Random(SEED)

    legacy_seen, legacy_held = load_legacy_families(source_root)
    new_seen, new_held = build_new_families(legacy_seen, legacy_held)
    seen = legacy_seen + new_seen
    held = legacy_held + new_held
    assert len(seen) == 100
    assert len(held) == 40
    for family in seen + held:
        validate_family(family)

    records: List[Dict[str, object]] = []
    split_files: Dict[str, List[str]] = defaultdict(list)
    configs_by_name: Dict[str, Dict[str, object]] = {}
    global_idx = 1

    # Train: 12 variants per family, exact terrain and formation balance.
    for family in seen:
        for vi, spec in enumerate(TRAIN_SPECS, 1):
            cfg, meta = make_config(family, "train", vi, global_idx, spec, rng)
            configs_by_name[str(cfg["name"])] = cfg
            records.append(meta)
            split_files["train"].append(f"configs/train/{cfg['name']}.json")
            global_idx += 1

    # Validation: 2 variants per seen family.
    for fi, family in enumerate(seen):
        for j in range(2):
            spec = EVAL_SPECS[(fi * 2 + j) % len(EVAL_SPECS)]
            cfg, meta = make_config(family, "validation", 13 + j, global_idx, spec, rng)
            configs_by_name[str(cfg["name"])] = cfg
            records.append(meta)
            split_files["validation"].append(f"configs/validation/{cfg['name']}.json")
            global_idx += 1

    # Blind IID: 3 variants per seen family.
    for fi, family in enumerate(seen):
        for j in range(3):
            spec = EVAL_SPECS[(fi * 3 + j + 4) % len(EVAL_SPECS)]
            cfg, meta = make_config(family, "blind_iid", 15 + j, global_idx, spec, rng)
            configs_by_name[str(cfg["name"])] = cfg
            records.append(meta)
            split_files["blind_iid"].append(f"configs/blind_iid/{cfg['name']}.json")
            global_idx += 1

    # Blind compositional: 10 variants per unseen family.
    for fi, family in enumerate(held):
        for j in range(10):
            spec = EVAL_SPECS[(fi * 10 + j + 7) % len(EVAL_SPECS)]
            cfg, meta = make_config(family, "blind_compositional", 1 + j, global_idx, spec, rng)
            configs_by_name[str(cfg["name"])] = cfg
            records.append(meta)
            split_files["blind_compositional"].append(f"configs/blind_compositional/{cfg['name']}.json")
            global_idx += 1

    expected = {"train": 1200, "validation": 200, "blind_iid": 300, "blind_compositional": 400}
    assert len(records) == 2100
    assert {k: len(v) for k, v in split_files.items()} == expected

    for split, paths in split_files.items():
        directory = out_root / "configs" / split
        directory.mkdir(parents=True, exist_ok=True)
        for rel in paths:
            name = Path(rel).stem
            (out_root / rel).write_text(json.dumps(configs_by_name[name], indent=2) + "\n", encoding="utf-8")

    validation_errors: Dict[str, List[str]] = {}
    content_hashes: Dict[str, str] = {}
    semantic_hashes: Counter = Counter()
    for paths in split_files.values():
        for rel in paths:
            path = out_root / rel
            cfg = json.loads(path.read_text(encoding="utf-8"))
            errors = static_validate_config(cfg, path.stem)
            if errors:
                validation_errors[rel] = errors
            content_hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            semantic = dict(cfg); semantic.pop("name")
            semantic_hashes[sha256_json(semantic)] += 1
    duplicates = [key for key, count in semantic_hashes.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate semantic configurations: {len(duplicates)}")
    if validation_errors:
        sample = list(validation_errors.items())[:5]
        raise ValueError(f"Static validation errors ({len(validation_errors)}), sample={sample}")

    path_lookup = {Path(rel).stem: rel for paths in split_files.values() for rel in paths}
    with (out_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for record0 in records:
            record = dict(record0)
            rel = path_lookup[str(record["name"])]
            record["path"] = rel; record["sha256"] = content_hashes[rel]
            f.write(json.dumps(record, sort_keys=True) + "\n")

    csv_fields = [
        "name", "path", "split", "family_id", "archetype", "family_origin", "heldout_compositional",
        "variant_index", "engagement_class", "initial_min_cross_distance", "terrain", "formation", "layout",
        "num_allies", "num_enemies", "ally_value", "enemy_value", "ally_value_ratio", "difficulty_proxy",
        "static_difficulty_score", "ally_composition", "enemy_composition", "sha256",
    ]
    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields); writer.writeheader()
        for record0 in records:
            record = dict(record0); rel = path_lookup[str(record["name"])]
            record["path"] = rel; record["sha256"] = content_hashes[rel]
            record["ally_composition"] = json.dumps(record["ally_composition"], sort_keys=True)
            record["enemy_composition"] = json.dumps(record["enemy_composition"], sort_keys=True)
            writer.writerow({key: record[key] for key in csv_fields})

    # Curriculum seed file is intentionally mutable by downstream training code.
    with (out_root / "curriculum_seed_scores.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["name", "family_id", "archetype", "static_difficulty_score", "episodes", "wins", "timeouts", "ema_difficulty"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for record in records:
            if record["split"] == "train":
                writer.writerow({
                    "name": record["name"], "family_id": record["family_id"], "archetype": record["archetype"],
                    "static_difficulty_score": record["static_difficulty_score"], "episodes": 0, "wins": 0,
                    "timeouts": 0, "ema_difficulty": record["static_difficulty_score"],
                })

    (out_root / "split_manifest.json").write_text(json.dumps({
        "seed": SEED,
        "global_unit_type_ids": GLOBAL_UNIT_TYPE_IDS,
        "max_compatibility": {"max_agents": 9, "max_enemies": 10, "max_actions": 16},
        "splits": split_files,
    }, indent=2) + "\n", encoding="utf-8")
    family_catalog = {
        "seen_families": [asdict(f) for f in seen],
        "heldout_compositional_families": [asdict(f) for f in held],
        "counts": {
            "seen_legacy": len(legacy_seen), "seen_new": len(new_seen),
            "heldout_legacy": len(legacy_held), "heldout_new": len(new_held),
        },
    }
    (out_root / "family_catalog.json").write_text(json.dumps(family_catalog, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(source_root / "family_catalog.json", out_root / "source_legacy_family_catalog.json")

    split_counts = Counter(str(r["split"]) for r in records)
    def grouped(field):
        return {split: dict(Counter(str(r[field]) for r in records if r["split"] == split)) for split in split_counts}
    ratios = [float(r["ally_value_ratio"]) for r in records]
    summary = {
        "seed": SEED,
        "total_configs": len(records),
        "validation_errors": len(validation_errors),
        "unique_hashes": len(semantic_hashes),
        "family_counts": {
            "seen_total": len(seen), "seen_legacy": len(legacy_seen), "seen_new": len(new_seen),
            "heldout_total": len(held), "heldout_legacy": len(legacy_held), "heldout_new": len(new_held),
        },
        "split_counts": dict(split_counts),
        "engagement_counts": grouped("engagement_class"),
        "terrain_counts": grouped("terrain"),
        "formation_counts": grouped("formation"),
        "layout_counts": grouped("layout"),
        "difficulty_proxy_counts": grouped("difficulty_proxy"),
        "ally_value_ratio": {
            "min": min(ratios), "max": max(ratios), "mean": statistics.mean(ratios), "median": statistics.median(ratios),
        },
        "compatibility": {"max_agents": 9, "max_enemies": 10, "max_actions": 16, "unit_type_ids": GLOBAL_UNIT_TYPE_IDS},
        "static_checks": [
            "exact JSON schema", "name/filename agreement", "unit/count validity", "team count agreement",
            "global unit-type vocabulary", "shield flag exactness", "medivac targetability", "group placement emulation",
            "spawn bounds", "terrain walkability", "same-plane overlap rejection", "attack-point bounds",
            "engagement bucket", "duplicate semantic map rejection",
        ],
        "dynamic_validation_required": True,
    }
    (out_root / "validation_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with (out_root / "checksums.sha256").open("w", encoding="utf-8") as f:
        for rel, digest in sorted(content_hashes.items()):
            f.write(f"{digest}  {rel}\n")

    source_path = Path(__file__).resolve()
    shutil.copy2(source_path, out_root / "generate_r2_smaclite_general_2100.py")
    write_dynamic_validator(out_root)
    write_readme(out_root, summary)

    if make_zip:
        zip_path = out_root.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(out_root.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(out_root.name) / path.relative_to(out_root))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("r2_smaclite_general_2100_configs"))
    parser.add_argument("--source-root", type=Path, required=True, help="Extracted r2_smaclite_1300_configs root")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(args.out, args.source_root, not args.no_zip), indent=2))


if __name__ == "__main__":
    main()
