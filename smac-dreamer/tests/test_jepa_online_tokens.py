import pathlib
import sys
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smacdreamer.jepa.online_tokens import (
    JEPAVisibilityConfig,
    JEPATokenSpec,
    apply_enemy_visibility_mask,
    encode_state_vector,
)


def test_encode_state_vector_matches_dataset_layout():
    spec = JEPATokenSpec(
        n_agents=2,
        n_enemies=1,
        max_agents=3,
        max_enemies=2,
        max_actions=4,
        ally_state_feat_size=3,
        enemy_state_feat_size=2,
        dynamic_token_dim=3,
        entity_static_feat_size=2,
        static_dim=5,
        token_dim=5,
        ally_has_shields=False,
        enemy_has_shields=False,
        num_unit_types=0,
    )
    state = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    entity_static = np.arange(spec.entities * 2, dtype=np.float32).reshape(spec.entities, 2)
    tokens, mask, slot = encode_state_vector(state, spec, entity_static)
    assert tokens.shape == (5, 5)
    np.testing.assert_allclose(tokens[0, :3], [1, 2, 3])
    np.testing.assert_allclose(tokens[1, :3], [4, 5, 6])
    np.testing.assert_allclose(tokens[3, :2], [7, 8])
    np.testing.assert_allclose(tokens[:, 3:5], entity_static)
    assert mask.tolist() == [1.0, 1.0, 0.0, 1.0, 0.0]
    assert slot.tolist() == [1.0, 1.0, 0.0, 1.0, 0.0]


def test_structural_slot_mask_only_marks_real_map_slots():
    spec = JEPATokenSpec(
        n_agents=3,
        n_enemies=4,
        max_agents=10,
        max_enemies=10,
        max_actions=5,
        ally_state_feat_size=2,
        enemy_state_feat_size=2,
        dynamic_token_dim=2,
        entity_static_feat_size=1,
        static_dim=4,
        token_dim=3,
        ally_has_shields=False,
        enemy_has_shields=False,
        num_unit_types=0,
    )
    state = np.ones(spec.n_agents * 2 + spec.n_enemies * 2, dtype=np.float32)
    static = np.zeros((spec.entities, 1), dtype=np.float32)
    _, _, slot = encode_state_vector(state, spec, static)
    assert int(slot.sum()) == 7
    assert slot[:3].tolist() == [1.0, 1.0, 1.0]
    assert slot[3:10].sum() == 0.0
    assert slot[10:14].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert slot[14:].sum() == 0.0


def test_visibility_mask_zeros_enemy_outside_any_live_ally_sight():
    allies = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 100.0, 100.0],
    ], dtype=np.float32)
    enemies = np.asarray([
        [1.0, 0.0, 3.0, 4.0],
        [1.0, 0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    out = apply_enemy_visibility_mask(
        allies,
        enemies,
        static_condition=np.asarray([1.0, 1.0], dtype=np.float32),
        visibility=JEPAVisibilityConfig(enemy_visibility_mask=True, enemy_sight_range=5.0, xy_indices=(2, 3)),
    )
    np.testing.assert_allclose(out[0], enemies[0])
    np.testing.assert_allclose(out[1], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(out[2], np.zeros(4, dtype=np.float32))


def test_visibility_boundary_and_normalized_coordinate_scaling():
    allies = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    enemies = np.asarray([
        [1.0, 0.0, 0.5, 0.0],
        [1.0, 0.0, 0.51, 0.0],
    ], dtype=np.float32)
    out = apply_enemy_visibility_mask(
        allies,
        enemies,
        static_condition=np.asarray([10.0, 10.0], dtype=np.float32),
        visibility=JEPAVisibilityConfig(enemy_visibility_mask=True, enemy_sight_range=5.0, xy_indices=(2, 3)),
    )
    np.testing.assert_allclose(out[0], enemies[0])
    np.testing.assert_allclose(out[1], np.zeros(4, dtype=np.float32))
