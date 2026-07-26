"""Canonical structured observation for centralised multimap SMAClite control (P0.3).

Replaces whole-vector right-padding with a representation whose every semantic feature sits
at a FIXED location regardless of the map's ally count, enemy count, shield support, or
unit-type composition. The blocks (per centralised controller step) are:

    self_features    [max_agents, F_self]              F_self  = 2 + V   (hp, shield, type[V])
    ally_features    [max_agents, max_agents, F_ally]  F_ally  = 6 + V   (visible,dist,dx,dy,hp,shield,type[V])
    enemy_features   [max_agents, max_enemies, F_enemy]F_enemy = 6 + V   (attackable,dist,dx,dy,hp,shield,type[V])
    movement_features[max_agents, 4]
    avail_actions    [max_agents, max_actions]
    agent_slot_mask  [max_agents]                       real (non-padded) agent slots
    agent_alive_mask [max_agents]                       alive agents
    ally_entity_mask [max_agents, max_agents]           valid ally slots (b != a, both real)
    enemy_entity_mask[max_agents, max_enemies]          valid enemy slots

Key invariants vs the old flat obs:
  * HP and shields occupy SEPARATE fixed dimensions (shield = 0 when a unit/map has no shield).
  * Unit type uses a GLOBAL vocabulary (``GLOBAL_UNIT_VOCAB``) shared across all maps, NOT the
    per-map local one-hot — so "STALKER" maps to the same index on every map.

It is built by PARSING SMAClite's own per-agent obs (reusing its exact visibility +
normalisation) and reshaping/remapping into the canonical blocks. ``flatten_for_model`` then
produces a flat MLP-friendly obs dict whose flat indices ARE the canonical layout (so the
fixed-location guarantee survives flattening); ``unflatten`` recovers the blocks for a future
entity-attention encoder.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

# Global unit-type vocabulary: the fixed set of SMAClite unit types, in a stable order. Every
# map remaps its local type ids into THIS vocabulary so feature meaning is map-independent.
GLOBAL_UNIT_VOCAB = (
    "BANELING", "COLOSSUS", "MARAUDER", "MARINE", "MEDIVAC",
    "SPINE_CRAWLER", "STALKER", "ZEALOT", "ZERGLING",
)
UNIT_TYPE_TO_GLOBAL: Dict[str, int] = {name: i for i, name in enumerate(GLOBAL_UNIT_VOCAB)}
V = len(GLOBAL_UNIT_VOCAB)

# Fixed per-entity feature widths (independent of any map).
F_SELF = 2 + V    # hp, shield, type[V]
F_ALLY = 6 + V    # visible, dist, dx, dy, hp, shield, type[V]
F_ENEMY = 6 + V   # attackable, dist, dx, dy, hp, shield, type[V]

# Block key order used to build the flat `state` vector (canonical, fixed).
_STATE_BLOCKS = ("movement_features", "self_features", "ally_features", "enemy_features")


def build_local_to_global(unit_type_ids: Optional[dict]) -> np.ndarray:
    """Map a map's local type index -> global vocab index.

    ``unit_type_ids`` is the map's ``{unit_type_or_name: local_index}`` (SMAClite ``map_info``).
    Returns an int array ``local_to_global`` of length ``num_local`` where
    ``local_to_global[local_index] = GLOBAL index``.

    An unknown unit-type name RAISES (rather than silently mapping to index 0): a type outside
    the global vocabulary would corrupt the canonical type encoding and must be caught.
    """
    if not unit_type_ids:
        return np.zeros(0, dtype=np.int64)
    num_local = max(int(i) for i in unit_type_ids.values()) + 1
    arr = np.zeros(num_local, dtype=np.int64)
    for key, local_idx in unit_type_ids.items():
        name = getattr(key, "name", str(key)).upper()
        if name not in UNIT_TYPE_TO_GLOBAL:
            raise ValueError(
                f"unit type {name!r} is not in the global vocabulary {GLOBAL_UNIT_VOCAB}; "
                "add it to GLOBAL_UNIT_VOCAB (no silent fallback to index 0)"
            )
        arr[int(local_idx)] = UNIT_TYPE_TO_GLOBAL[name]
    return arr


def feature_sizes() -> dict:
    """Fixed feature widths (handy for tests / encoders)."""
    return {"V": V, "F_self": F_SELF, "F_ally": F_ALLY, "F_enemy": F_ENEMY}


def _caps(pad_dims, n_agents: int, n_enemies: int, n_actions: int) -> tuple:
    """Resolve (max_agents, max_enemies, max_actions); fall back to real dims when unpadded."""
    if pad_dims is None:
        return int(n_agents), int(n_enemies), int(n_actions)
    return int(pad_dims.max_agents), int(pad_dims.max_enemies), int(pad_dims.max_actions)


def build_structured_obs(
    obs_tuple,
    avail,
    *,
    n_agents: int,
    n_enemies: int,
    enemy_feat_size: int,
    ally_feat_size: int,
    enemy_has_shields: bool,
    ally_has_shields: bool,
    num_unit_types: int,
    n_actions: int,
    alive_ids: Iterable[int],
    local_to_global: np.ndarray,
    pad_dims=None,
    agent_type_g=None,
    enemy_type_g=None,
) -> Dict[str, np.ndarray]:
    """Parse SMAClite's per-agent flat obs into the canonical padded blocks + masks.

    ``obs_tuple`` : sequence of n_agents flat obs vectors (SMAClite layout; dead agents are
                    all-zero). ``avail`` : per-agent available-action vectors (length n_actions).
    Returns a dict of the canonical blocks, every array shaped to the (padded) caps.

    Unit type: when ``agent_type_g`` / ``enemy_type_g`` (arrays of GLOBAL type indices, -1 for
    dead/unknown slots) are provided, the type one-hot is set from them — a single GLOBAL
    vocabulary independent of the map's local encoding (SMAClite drops the local one-hot
    entirely for single-type maps, so parsing it cannot give a consistent global type). Type is
    set for self (always, when alive) and for allies/enemies only where they are VISIBLE in the
    obs (no leaking hidden enemies). When the arrays are omitted, the local one-hot is parsed
    via ``local_to_global`` (used by isolated feature-parsing unit tests).
    """
    A, E, C = _caps(pad_dims, n_agents, n_enemies, n_actions)
    alive = set(int(i) for i in alive_ids)

    self_features = np.zeros((A, F_SELF), dtype=np.float32)
    ally_features = np.zeros((A, A, F_ALLY), dtype=np.float32)
    enemy_features = np.zeros((A, E, F_ENEMY), dtype=np.float32)
    movement = np.zeros((A, 4), dtype=np.float32)
    avail_actions = np.zeros((A, C), dtype=np.float32)
    agent_slot_mask = np.zeros((A,), dtype=np.float32)
    agent_alive_mask = np.zeros((A,), dtype=np.float32)
    ally_entity_mask = np.zeros((A, A), dtype=np.float32)
    enemy_entity_mask = np.zeros((A, E), dtype=np.float32)

    def _global_type(local_onehot: np.ndarray) -> np.ndarray:
        out = np.zeros(V, dtype=np.float32)
        if local_onehot.size and float(local_onehot.max()) > 0.0:
            out[int(local_to_global[int(np.argmax(local_onehot))])] = 1.0
        return out

    ally_off = 4 + n_enemies * enemy_feat_size
    self_off = ally_off + (n_agents - 1) * ally_feat_size

    for a in range(min(n_agents, A)):
        o = np.asarray(obs_tuple[a], dtype=np.float32)
        agent_slot_mask[a] = 1.0
        if a in alive:
            agent_alive_mask[a] = 1.0

        movement[a, :] = o[0:4]
        av = np.asarray(avail[a], dtype=np.float32).reshape(-1)
        avail_actions[a, : min(len(av), C)] = av[: min(len(av), C)]

        # --- Enemies (slot e = enemy id_in_faction) ---
        for e in range(min(n_enemies, E)):
            base = 4 + e * enemy_feat_size
            idx = base + 5
            shield = float(o[idx]) if enemy_has_shields else 0.0
            idx += int(enemy_has_shields)
            tl = o[idx: idx + num_unit_types] if num_unit_types else np.zeros(0, np.float32)
            enemy_features[a, e, 0] = o[base + 0]   # attackable
            enemy_features[a, e, 1] = o[base + 1]   # distance / sight
            enemy_features[a, e, 2] = o[base + 2]   # dx / sight
            enemy_features[a, e, 3] = o[base + 3]   # dy / sight
            enemy_features[a, e, 4] = o[base + 4]   # hp / max_hp
            enemy_features[a, e, 5] = shield        # shield / max_shield (0 if none)
            if enemy_type_g is not None:
                g = int(enemy_type_g[e])
                if g >= 0 and enemy_features[a, e, 4] > 0.0:   # visible alive enemy only
                    enemy_features[a, e, 6 + g] = 1.0
            else:
                enemy_features[a, e, 6:6 + V] = _global_type(tl)
            enemy_entity_mask[a, e] = 1.0

        # --- Allies (canonical slot = ally global id b; SMAClite block skips self) ---
        for b in range(n_agents):
            if b == a or b >= A:
                continue
            k = b if b < a else b - 1   # invert SMAClite's self-skip reindex
            base = ally_off + k * ally_feat_size
            idx = base + 5
            shield = float(o[idx]) if ally_has_shields else 0.0
            idx += int(ally_has_shields)
            tl = o[idx: idx + num_unit_types] if num_unit_types else np.zeros(0, np.float32)
            ally_features[a, b, 0] = o[base + 0]   # visible
            ally_features[a, b, 1] = o[base + 1]   # distance / sight
            ally_features[a, b, 2] = o[base + 2]   # dx / sight
            ally_features[a, b, 3] = o[base + 3]   # dy / sight
            ally_features[a, b, 4] = o[base + 4]   # hp / max_hp
            ally_features[a, b, 5] = shield
            if agent_type_g is not None:
                g = int(agent_type_g[b])
                if g >= 0 and ally_features[a, b, 0] > 0.0:    # visible ally only
                    ally_features[a, b, 6 + g] = 1.0
            else:
                ally_features[a, b, 6:6 + V] = _global_type(tl)
            ally_entity_mask[a, b] = 1.0

        # --- Self ---
        hp = float(o[self_off])
        idx = self_off + 1
        shield = float(o[idx]) if ally_has_shields else 0.0
        idx += int(ally_has_shields)
        tl = o[idx: idx + num_unit_types] if num_unit_types else np.zeros(0, np.float32)
        self_features[a, 0] = hp
        self_features[a, 1] = shield
        if agent_type_g is not None:
            g = int(agent_type_g[a])
            if g >= 0:   # self is always observed
                self_features[a, 2 + g] = 1.0
        else:
            self_features[a, 2:2 + V] = _global_type(tl)

    return {
        "self_features": self_features,
        "ally_features": ally_features,
        "enemy_features": enemy_features,
        "movement_features": movement,
        "avail_actions": avail_actions,
        "agent_slot_mask": agent_slot_mask,
        "agent_alive_mask": agent_alive_mask,
        "ally_entity_mask": ally_entity_mask,
        "enemy_entity_mask": enemy_entity_mask,
    }


def flatten_for_model(blocks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Flatten the canonical blocks into an MLP-friendly obs dict.

    ``state`` concatenates the feature blocks in the FIXED order ``_STATE_BLOCKS``; with fixed
    caps + global vocab, every semantic feature lands at a map-independent flat index. Masks
    and avail_actions are flattened separately so the model/loss can use them.
    """
    state = np.concatenate([blocks[k].reshape(-1) for k in _STATE_BLOCKS]).astype(np.float32)
    return {
        "state": state,
        "avail_actions": blocks["avail_actions"].reshape(-1).astype(np.float32),
        "agent_slot_mask": blocks["agent_slot_mask"].reshape(-1).astype(np.float32),
        "agent_alive_mask": blocks["agent_alive_mask"].reshape(-1).astype(np.float32),
        "ally_entity_mask": blocks["ally_entity_mask"].reshape(-1).astype(np.float32),
        "enemy_entity_mask": blocks["enemy_entity_mask"].reshape(-1).astype(np.float32),
    }


def unflatten(state: np.ndarray, pad_dims) -> Dict[str, np.ndarray]:
    """Recover the feature blocks from a flat ``state`` (for a future entity-attention encoder)."""
    A, E = int(pad_dims.max_agents), int(pad_dims.max_enemies)
    sizes = [A * 4, A * F_SELF, A * A * F_ALLY, A * E * F_ENEMY]
    parts = np.split(np.asarray(state).reshape(-1), np.cumsum(sizes)[:-1])
    return {
        "movement_features": parts[0].reshape(A, 4),
        "self_features": parts[1].reshape(A, F_SELF),
        "ally_features": parts[2].reshape(A, A, F_ALLY),
        "enemy_features": parts[3].reshape(A, E, F_ENEMY),
    }


def state_dim(pad_dims) -> int:
    """Length of the flat ``state`` vector for the given caps."""
    A, E = int(pad_dims.max_agents), int(pad_dims.max_enemies)
    return A * 4 + A * F_SELF + A * A * F_ALLY + A * E * F_ENEMY


def observation_space(pad_dims):
    """Gymnasium Dict space for the flattened structured model obs (MLP-compatible)."""
    import gymnasium as gym
    from gymnasium import spaces

    A, E, C = int(pad_dims.max_agents), int(pad_dims.max_enemies), int(pad_dims.max_actions)
    d = {
        "state":             spaces.Box(-np.inf, np.inf, shape=(state_dim(pad_dims),), dtype=np.float32),
        "avail_actions":     spaces.Box(0.0, 1.0, shape=(A * C,), dtype=np.float32),
        "agent_slot_mask":   spaces.Box(0.0, 1.0, shape=(A,), dtype=np.float32),
        "agent_alive_mask":  spaces.Box(0.0, 1.0, shape=(A,), dtype=np.float32),
        "ally_entity_mask":  spaces.Box(0.0, 1.0, shape=(A * A,), dtype=np.float32),
        "enemy_entity_mask": spaces.Box(0.0, 1.0, shape=(A * E,), dtype=np.float32),
        "is_first":          spaces.Box(0, 1, shape=(), dtype=bool),
        "is_last":           spaces.Box(0, 1, shape=(), dtype=bool),
        "is_terminal":       spaces.Box(0, 1, shape=(), dtype=bool),
    }
    return spaces.Dict(d)


__all__ = [
    "GLOBAL_UNIT_VOCAB", "UNIT_TYPE_TO_GLOBAL", "V", "F_SELF", "F_ALLY", "F_ENEMY",
    "build_local_to_global", "build_structured_obs", "flatten_for_model", "unflatten",
    "state_dim", "observation_space", "feature_sizes",
]
