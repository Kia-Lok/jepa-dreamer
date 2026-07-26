"""Tests for the canonical structured observation (P0.3). Pure NumPy — no smaclite/torch.

Strategy: synthesise SMAClite-layout per-agent obs (matching smaclite.py __get_agent_obs)
for heterogeneous configs, parse them with build_structured_obs, and prove:
  * features land in fixed canonical slots (HP and shield separate),
  * unit type uses the GLOBAL vocab regardless of map-local ordering,
  * masks (slot / alive / ally-entity / enemy-entity) are correct,
  * the SAME semantic feature occupies the SAME flat location across different maps.
"""

import numpy as np
import pytest

from smacdreamer.envs.padding import PaddingDims
from smacdreamer.envs import structured_obs as so


# ----------------------------------------------------------------------
# SMAClite-layout obs synthesiser (mirrors external/smaclite __get_agent_obs)
# ----------------------------------------------------------------------

def _encode_agent_obs(a, n_agents, n_enemies, enemy_has_shields, ally_has_shields,
                      num_unit_types, movement, enemies, allies, self_feat):
    efs = 5 + int(enemy_has_shields) + num_unit_types
    afs = 5 + int(ally_has_shields) + num_unit_types
    size = 4 + n_enemies * efs + (n_agents - 1) * afs + (1 + int(ally_has_shields) + num_unit_types)
    o = np.zeros(size, dtype=np.float32)
    o[0:4] = movement
    for e, ed in enumerate(enemies):
        base = 4 + e * efs
        o[base + 0] = ed["attackable"]; o[base + 1] = ed["dist"]
        o[base + 2] = ed["dx"]; o[base + 3] = ed["dy"]; o[base + 4] = ed["hp"]
        idx = base + 5
        if enemy_has_shields:
            o[idx] = ed.get("shield", 0.0); idx += 1
        if num_unit_types:
            o[idx:idx + num_unit_types] = ed["type_local"]
    aoff = 4 + n_enemies * efs
    for b, ad in allies.items():
        k = b if b < a else b - 1
        base = aoff + k * afs
        o[base + 0] = ad["visible"]; o[base + 1] = ad["dist"]
        o[base + 2] = ad["dx"]; o[base + 3] = ad["dy"]; o[base + 4] = ad["hp"]
        idx = base + 5
        if ally_has_shields:
            o[idx] = ad.get("shield", 0.0); idx += 1
        if num_unit_types:
            o[idx:idx + num_unit_types] = ad["type_local"]
    soff = aoff + (n_agents - 1) * afs
    o[soff] = self_feat["hp"]
    idx = soff + 1
    if ally_has_shields:
        o[idx] = self_feat.get("shield", 0.0); idx += 1
    if num_unit_types:
        o[idx:idx + num_unit_types] = self_feat["type_local"]
    return o


def _pad(max_agents=4, max_enemies=5, max_actions=16):
    return PaddingDims(max_agents=max_agents, max_enemies=max_enemies,
                       max_actions=max_actions, max_obs_size=999)


# ----------------------------------------------------------------------
# Basic parsing: features in canonical slots, separate HP/shield, global type
# ----------------------------------------------------------------------

def test_parse_features_masks_and_global_type():
    # 2 allies, 2 enemies; allies have shields, enemies do not; types {STALKER:0, ZEALOT:1}.
    n_agents, n_enemies = 2, 2
    local_ids = {"STALKER": 0, "ZEALOT": 1}
    l2g = so.build_local_to_global(local_ids)

    obs0 = _encode_agent_obs(
        a=0, n_agents=n_agents, n_enemies=n_enemies,
        enemy_has_shields=False, ally_has_shields=True, num_unit_types=2,
        movement=[1, 0, 1, 0],
        enemies=[
            {"attackable": 1, "dist": 0.5, "dx": 0.1, "dy": 0.2, "hp": 0.9, "type_local": [1, 0]},  # STALKER
            {"attackable": 0, "dist": 0.3, "dx": -0.1, "dy": 0.0, "hp": 0.4, "type_local": [0, 1]},  # ZEALOT
        ],
        allies={1: {"visible": 1, "dist": 0.2, "dx": 0.05, "dy": -0.05, "hp": 0.8,
                    "shield": 0.5, "type_local": [0, 1]}},  # ally 1 is ZEALOT
        self_feat={"hp": 1.0, "shield": 0.7, "type_local": [1, 0]},   # self is STALKER
    )
    obs1 = _encode_agent_obs(
        a=1, n_agents=n_agents, n_enemies=n_enemies,
        enemy_has_shields=False, ally_has_shields=True, num_unit_types=2,
        movement=[0, 1, 0, 1],
        enemies=[{"attackable": 0, "dist": 0, "dx": 0, "dy": 0, "hp": 0, "type_local": [0, 0]},
                 {"attackable": 0, "dist": 0, "dx": 0, "dy": 0, "hp": 0, "type_local": [0, 0]}],
        allies={0: {"visible": 1, "dist": 0.2, "dx": -0.05, "dy": 0.05, "hp": 1.0,
                    "shield": 0.7, "type_local": [1, 0]}},  # ally 0 is STALKER
        self_feat={"hp": 0.8, "shield": 0.5, "type_local": [0, 1]},   # self is ZEALOT
    )

    blocks = so.build_structured_obs(
        [obs0, obs1], avail=[np.ones(8), np.ones(8)],
        n_agents=n_agents, n_enemies=n_enemies,
        enemy_feat_size=5 + 0 + 2, ally_feat_size=5 + 1 + 2,
        enemy_has_shields=False, ally_has_shields=True, num_unit_types=2,
        n_actions=8, alive_ids=[0, 1], local_to_global=l2g, pad_dims=_pad(),
    )

    g_stalker = so.UNIT_TYPE_TO_GLOBAL["STALKER"]
    g_zealot = so.UNIT_TYPE_TO_GLOBAL["ZEALOT"]

    # Movement
    assert np.array_equal(blocks["movement_features"][0], [1, 0, 1, 0])
    # Enemy 0 (as seen by agent 0): canonical [attackable,dist,dx,dy,hp,shield, type...]
    ef = blocks["enemy_features"][0, 0]
    assert ef[0] == pytest.approx(1.0)       # attackable
    assert ef[4] == pytest.approx(0.9)       # hp
    assert ef[5] == pytest.approx(0.0)       # shield is a SEPARATE slot, 0 (enemies shield-less)
    assert ef[6 + g_stalker] == pytest.approx(1.0)
    assert ef[6:6 + so.V].sum() == pytest.approx(1.0)   # exactly one global type bit
    # Enemy 1 is a ZEALOT
    assert blocks["enemy_features"][0, 1][6 + g_zealot] == pytest.approx(1.0)
    # Ally slot 1 (as seen by agent 0): shield occupies its own fixed dim
    al = blocks["ally_features"][0, 1]
    assert al[0] == pytest.approx(1.0)       # visible
    assert al[4] == pytest.approx(0.8)       # hp
    assert al[5] == pytest.approx(0.5)       # shield
    assert al[6 + g_zealot] == pytest.approx(1.0)
    # Self (agent 0) is a STALKER with hp/shield in separate slots
    sf = blocks["self_features"][0]
    assert sf[0] == pytest.approx(1.0)       # hp
    assert sf[1] == pytest.approx(0.7)       # shield
    assert sf[2 + g_stalker] == pytest.approx(1.0)

    # Masks
    assert np.array_equal(blocks["agent_slot_mask"], [1, 1, 0, 0])
    assert np.array_equal(blocks["agent_alive_mask"], [1, 1, 0, 0])
    assert blocks["ally_entity_mask"][0, 1] == 1.0 and blocks["ally_entity_mask"][0, 0] == 0.0  # self excluded
    assert blocks["ally_entity_mask"][1, 0] == 1.0 and blocks["ally_entity_mask"][1, 1] == 0.0
    assert np.array_equal(blocks["enemy_entity_mask"][0], [1, 1, 0, 0, 0])
    assert np.array_equal(blocks["enemy_entity_mask"][2], [0, 0, 0, 0, 0])  # padded agent


# ----------------------------------------------------------------------
# Shield is a separate fixed dim that is 0 on shield-less maps
# ----------------------------------------------------------------------

def test_shield_dim_zero_when_map_has_no_shields():
    obs0 = _encode_agent_obs(
        a=0, n_agents=1, n_enemies=1, enemy_has_shields=False, ally_has_shields=False,
        num_unit_types=1, movement=[1, 1, 0, 0],
        enemies=[{"attackable": 1, "dist": 0.4, "dx": 0.1, "dy": 0.1, "hp": 0.6, "type_local": [1]}],
        allies={}, self_feat={"hp": 0.5, "type_local": [1]},
    )
    blocks = so.build_structured_obs(
        [obs0], avail=[np.ones(7)], n_agents=1, n_enemies=1,
        enemy_feat_size=5 + 0 + 1, ally_feat_size=5 + 0 + 1,
        enemy_has_shields=False, ally_has_shields=False, num_unit_types=1,
        n_actions=7, alive_ids=[0], local_to_global=so.build_local_to_global({"MARINE": 0}),
        pad_dims=_pad(),
    )
    assert blocks["enemy_features"][0, 0, 4] == pytest.approx(0.6)   # hp present
    assert blocks["enemy_features"][0, 0, 5] == pytest.approx(0.0)   # shield slot exists and is 0
    assert blocks["self_features"][0, 0] == pytest.approx(0.5)       # hp
    assert blocks["self_features"][0, 1] == pytest.approx(0.0)       # shield slot 0
    assert blocks["self_features"][0, 2 + so.UNIT_TYPE_TO_GLOBAL["MARINE"]] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Global vocab independence: same physical type -> same global index
# ----------------------------------------------------------------------

def test_global_type_independent_of_local_ordering():
    # Map A: STALKER local 0. Map B: STALKER local 1. The global STALKER bit must match.
    def enemy_obs(local_ids, stalker_local, n_types):
        type_local = [0] * n_types
        type_local[stalker_local] = 1
        o = _encode_agent_obs(
            a=0, n_agents=1, n_enemies=1, enemy_has_shields=False, ally_has_shields=False,
            num_unit_types=n_types, movement=[0, 0, 0, 0],
            enemies=[{"attackable": 1, "dist": 0.1, "dx": 0, "dy": 0, "hp": 1.0, "type_local": type_local}],
            allies={}, self_feat={"hp": 1.0, "type_local": [0] * n_types},
        )
        return so.build_structured_obs(
            [o], avail=[np.ones(7)], n_agents=1, n_enemies=1,
            enemy_feat_size=5 + 0 + n_types, ally_feat_size=5 + 0 + n_types,
            enemy_has_shields=False, ally_has_shields=False, num_unit_types=n_types,
            n_actions=7, alive_ids=[0], local_to_global=so.build_local_to_global(local_ids),
            pad_dims=_pad(),
        )

    a = enemy_obs({"STALKER": 0, "ZEALOT": 1}, stalker_local=0, n_types=2)
    b = enemy_obs({"ZEALOT": 0, "STALKER": 1}, stalker_local=1, n_types=2)
    g = so.UNIT_TYPE_TO_GLOBAL["STALKER"]
    assert a["enemy_features"][0, 0, 6 + g] == 1.0
    assert b["enemy_features"][0, 0, 6 + g] == 1.0
    # and identical full type sub-vectors
    assert np.array_equal(a["enemy_features"][0, 0, 6:6 + so.V],
                          b["enemy_features"][0, 0, 6:6 + so.V])


# ----------------------------------------------------------------------
# Fixed semantic location across maps of different sizes
# ----------------------------------------------------------------------

def test_same_feature_same_flat_index_across_maps():
    pad = _pad(max_agents=4, max_enemies=5, max_actions=16)

    def build(n_agents, n_enemies):
        obs = []
        for a in range(n_agents):
            enemies = [{"attackable": 1, "dist": 0.5, "dx": 0.0, "dy": 0.0, "hp": 0.5,
                        "type_local": [1]} for _ in range(n_enemies)]
            allies = {b: {"visible": 1, "dist": 0.1, "dx": 0.0, "dy": 0.0, "hp": 1.0,
                          "type_local": [1]} for b in range(n_agents) if b != a}
            obs.append(_encode_agent_obs(
                a, n_agents, n_enemies, False, False, 1, [0, 0, 0, 0], enemies, allies,
                {"hp": 1.0, "type_local": [1]}))
        blocks = so.build_structured_obs(
            obs, avail=[np.ones(7)] * n_agents, n_agents=n_agents, n_enemies=n_enemies,
            enemy_feat_size=6, ally_feat_size=6, enemy_has_shields=False,
            ally_has_shields=False, num_unit_types=1, n_actions=7,
            alive_ids=list(range(n_agents)),
            local_to_global=so.build_local_to_global({"MARINE": 0}), pad_dims=pad)
        return so.flatten_for_model(blocks)["state"]

    s_small = build(2, 2)
    s_big = build(3, 4)
    assert s_small.shape == s_big.shape == (so.state_dim(pad),)

    # Enemy-0 HP for agent 1 must be at the same flat index on both maps.
    A, E = pad.max_agents, pad.max_enemies
    enemy_start = A * 4 + A * so.F_SELF + A * A * so.F_ALLY
    idx = enemy_start + 1 * (E * so.F_ENEMY) + 0 * so.F_ENEMY + 4   # agent 1, enemy 0, hp
    assert s_small[idx] == pytest.approx(0.5)
    assert s_big[idx] == pytest.approx(0.5)
    # unflatten recovers the same slot
    assert so.unflatten(s_small, pad)["enemy_features"][1, 0, 4] == pytest.approx(0.5)
    assert so.unflatten(s_big, pad)["enemy_features"][1, 0, 4] == pytest.approx(0.5)


def test_flatten_unflatten_roundtrip():
    pad = _pad()
    obs0 = _encode_agent_obs(0, 1, 1, False, False, 1, [1, 0, 1, 0],
                             [{"attackable": 1, "dist": 0.2, "dx": 0.3, "dy": 0.4, "hp": 0.7,
                               "type_local": [1]}], {}, {"hp": 0.9, "type_local": [1]})
    blocks = so.build_structured_obs(
        [obs0], avail=[np.ones(7)], n_agents=1, n_enemies=1, enemy_feat_size=6,
        ally_feat_size=6, enemy_has_shields=False, ally_has_shields=False, num_unit_types=1,
        n_actions=7, alive_ids=[0], local_to_global=so.build_local_to_global({"MARINE": 0}),
        pad_dims=pad)
    state = so.flatten_for_model(blocks)["state"]
    rec = so.unflatten(state, pad)
    for k in ("movement_features", "self_features", "ally_features", "enemy_features"):
        assert np.allclose(rec[k], blocks[k]), k


def test_global_type_arrays_set_types_even_without_local_onehot():
    # Single-type maps carry NO local type one-hot (SMAClite sets num_unit_types=0). The global
    # type arrays must still set a consistent global type for self + visible entities.
    g_stalker = so.UNIT_TYPE_TO_GLOBAL["STALKER"]
    g_marine = so.UNIT_TYPE_TO_GLOBAL["MARINE"]
    obs0 = _encode_agent_obs(
        a=0, n_agents=2, n_enemies=1, enemy_has_shields=False, ally_has_shields=False,
        num_unit_types=0, movement=[0, 0, 0, 0],
        enemies=[{"attackable": 1, "dist": 0.2, "dx": 0, "dy": 0, "hp": 0.5, "type_local": []}],
        allies={1: {"visible": 1, "dist": 0.1, "dx": 0, "dy": 0, "hp": 1.0, "type_local": []}},
        self_feat={"hp": 1.0, "type_local": []})
    obs1 = _encode_agent_obs(
        a=1, n_agents=2, n_enemies=1, enemy_has_shields=False, ally_has_shields=False,
        num_unit_types=0, movement=[0, 0, 0, 0],
        enemies=[{"attackable": 0, "dist": 0, "dx": 0, "dy": 0, "hp": 0, "type_local": []}],
        allies={0: {"visible": 1, "dist": 0.1, "dx": 0, "dy": 0, "hp": 1.0, "type_local": []}},
        self_feat={"hp": 1.0, "type_local": []})
    blocks = so.build_structured_obs(
        [obs0, obs1], avail=[np.ones(7), np.ones(7)], n_agents=2, n_enemies=1,
        enemy_feat_size=5, ally_feat_size=5, enemy_has_shields=False, ally_has_shields=False,
        num_unit_types=0, n_actions=7, alive_ids=[0, 1],
        local_to_global=np.zeros(0, dtype=np.int64), pad_dims=_pad(),
        agent_type_g=[g_stalker, g_stalker], enemy_type_g=[g_marine])
    assert blocks["self_features"][0, 2 + g_stalker] == 1.0
    assert blocks["self_features"][1, 2 + g_stalker] == 1.0
    assert blocks["self_features"][0, 2:2 + so.V].sum() == pytest.approx(1.0)
    assert blocks["enemy_features"][0, 0, 6 + g_marine] == 1.0   # visible enemy typed
    assert blocks["ally_features"][0, 1, 6 + g_stalker] == 1.0   # visible ally typed


def test_global_type_array_not_leaked_for_invisible_enemy():
    g_marine = so.UNIT_TYPE_TO_GLOBAL["MARINE"]
    # enemy not visible (hp feature 0) -> type must NOT be set (no hidden-enemy leakage).
    obs0 = _encode_agent_obs(
        a=0, n_agents=1, n_enemies=1, enemy_has_shields=False, ally_has_shields=False,
        num_unit_types=0, movement=[0, 0, 0, 0],
        enemies=[{"attackable": 0, "dist": 0, "dx": 0, "dy": 0, "hp": 0, "type_local": []}],
        allies={}, self_feat={"hp": 1.0, "type_local": []})
    blocks = so.build_structured_obs(
        [obs0], avail=[np.ones(7)], n_agents=1, n_enemies=1, enemy_feat_size=5, ally_feat_size=5,
        enemy_has_shields=False, ally_has_shields=False, num_unit_types=0, n_actions=7,
        alive_ids=[0], local_to_global=np.zeros(0, dtype=np.int64), pad_dims=_pad(),
        agent_type_g=[g_marine], enemy_type_g=[g_marine])
    assert blocks["enemy_features"][0, 0, 6:6 + so.V].sum() == pytest.approx(0.0)


def test_build_local_to_global_names_and_enums():
    import enum
    by_name = so.build_local_to_global({"STALKER": 0, "ZEALOT": 1})
    assert by_name[0] == so.UNIT_TYPE_TO_GLOBAL["STALKER"]
    assert by_name[1] == so.UNIT_TYPE_TO_GLOBAL["ZEALOT"]

    # Real enum keys (hashable, with .name) — mirrors SMAClite's UnitType enum map keys.
    class _UT(enum.Enum):
        MARINE = 0
        MEDIVAC = 1
    by_enum = so.build_local_to_global({_UT.MARINE: 0, _UT.MEDIVAC: 1})
    assert by_enum[0] == so.UNIT_TYPE_TO_GLOBAL["MARINE"]
    assert by_enum[1] == so.UNIT_TYPE_TO_GLOBAL["MEDIVAC"]


def test_unknown_unit_type_raises():
    with pytest.raises(ValueError):
        so.build_local_to_global({"DRAGON": 0})   # not in the global vocabulary


def test_dead_agent_slot_is_zero_but_marked_not_alive():
    # agent 1 dead -> obs all zeros, alive_ids excludes it.
    obs0 = _encode_agent_obs(0, 2, 1, False, False, 1, [1, 1, 1, 1],
                             [{"attackable": 1, "dist": 0.1, "dx": 0, "dy": 0, "hp": 1.0,
                               "type_local": [1]}],
                             {1: {"visible": 0, "dist": 0, "dx": 0, "dy": 0, "hp": 0,
                                  "type_local": [0]}},
                             {"hp": 1.0, "type_local": [1]})
    obs1 = np.zeros_like(obs0)  # dead agent
    blocks = so.build_structured_obs(
        [obs0, obs1], avail=[np.ones(7), np.zeros(7)], n_agents=2, n_enemies=1,
        enemy_feat_size=6, ally_feat_size=6, enemy_has_shields=False, ally_has_shields=False,
        num_unit_types=1, n_actions=7, alive_ids=[0],  # agent 1 not alive
        local_to_global=so.build_local_to_global({"MARINE": 0}), pad_dims=_pad())
    assert blocks["agent_slot_mask"][1] == 1.0     # slot is real
    assert blocks["agent_alive_mask"][1] == 0.0    # but not alive
    assert blocks["self_features"][1].sum() == pytest.approx(0.0)  # dead -> zero features


# ======================================================================
# Env-integration: parse REAL SMAClite obs (the ultimate correctness check)
# ======================================================================

from conftest import requires_smaclite, FIXED_SCENARIO  # noqa: E402


@requires_smaclite
def test_env_structured_mode_obs_matches_space():
    from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
    env = SMACliteDreamerEnv(scenario=FIXED_SCENARIO, max_episode_steps=20, seed=0,
                             obs_mode="structured")
    try:
        obs, _ = env.reset(seed=0)
        # Exactly the structured keys, all within the declared observation space.
        assert set(obs) == set(env.observation_space.spaces)
        assert env.observation_space.contains(obs)
        # Masks reflect the real scenario: every real agent slot marked, alive at reset.
        A = env._structured_pad_dims().max_agents
        assert obs["agent_slot_mask"].sum() == pytest.approx(float(env.n_agents))
        assert obs["agent_alive_mask"].sum() == pytest.approx(float(env.n_agents))
        # Step once and stay consistent with the space.
        flat = env.codec.encode([1] * env.n_agents, num_real_agents=env.n_agents)
        obs2, _, _, _, _ = env.step(flat)
        assert env.observation_space.contains(obs2)
    finally:
        env.close()


@requires_smaclite
def test_structured_heterogeneous_map_switch_no_leakage():
    # Cycle two built-in scenarios with DIFFERENT ally/enemy counts, shield support and unit
    # types; confirm fixed output shape, consistent semantics, correct masks, and no leakage.
    from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
    from smacdreamer.envs.map_sampler import MapSampler, MapEntry
    from smacdreamer.envs.padding import PaddingDims

    pad = PaddingDims(max_agents=6, max_enemies=6, max_actions=12, max_obs_size=999)
    entries = [MapEntry(name="2s3z", type="builtin"),       # 5v5, stalker/zealot, both shielded
               MapEntry(name="2s_vs_1sc", type="builtin")]  # 2v1, stalkers vs spine-crawler (enemy shield-less)
    sampler = MapSampler.from_entries(entries, mode="round_robin", seed=0)
    env = SMACliteDreamerEnv(scenario="2s3z", max_episode_steps=20, seed=0,
                             map_sampler=sampler, pad_dims=pad, obs_mode="structured")
    try:
        shapes = set()
        seen = {}
        for _ in range(4):   # cycle through both maps twice
            obs, _ = env.reset(seed=0)
            assert set(obs) == set(env.observation_space.spaces)
            assert env.observation_space.contains(obs)
            shapes.add(tuple(obs["state"].shape))
            # Mask reflects THIS map's real agent count (correct masks).
            assert obs["agent_slot_mask"].sum() == pytest.approx(float(env.n_agents))
            assert obs["agent_alive_mask"].sum() == pytest.approx(float(env.n_agents))
            blocks = so.unflatten(obs["state"], pad)
            # Consistent semantics: every live agent carries exactly one global type bit.
            for a in range(env.n_agents):
                assert blocks["self_features"][a, 2:2 + so.V].sum() == pytest.approx(1.0)
            # No leakage: padded slots beyond this map's agents are fully zero, even after a
            # previous (larger) map populated them.
            for a in range(env.n_agents, pad.max_agents):
                assert blocks["self_features"][a].sum() == pytest.approx(0.0)
                assert obs["agent_slot_mask"][a] == pytest.approx(0.0)
            seen[env._current_map_name] = env.n_agents
        assert len(shapes) == 1                          # fixed output shape across maps
        assert set(seen) == {"2s3z", "2s_vs_1sc"}        # visited both heterogeneous maps
        assert len(set(seen.values())) >= 2              # different ally counts (5 vs 2)
    finally:
        env.close()
