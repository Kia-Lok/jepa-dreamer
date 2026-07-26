import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))

import preflight_jepa_training as preflight


def _episode_npz(path):
    np.savez(
        path,
        states=np.zeros((1, 3, 12), dtype=np.float32),
        actions=np.zeros((1, 2, 2), dtype=np.int64),
        static_condition=np.zeros((4,), dtype=np.float32),
        entity_static=np.zeros((3, 2), dtype=np.float32),
        n_agents=2,
        n_enemies=1,
        n_actions=3,
        ally_state_feat_size=4,
        enemy_state_feat_size=4,
        ally_has_shields=0,
        enemy_has_shields=0,
        num_unit_types=0,
        static_dim=4,
        entity_static_feat_size=2,
    )


def test_preflight_fails_without_padding_or_existing_map_source(tmp_path):
    ep = tmp_path / "episode.npz"
    cfg = tmp_path / "config.yaml"
    _episode_npz(ep)
    cfg.write_text(
        """
imag_horizon: 5
observation:
  mode: structured
maps:
  train: missing/train
""",
        encoding="utf-8",
    )
    vis = type("V", (), {"metadata": lambda self: {
        "enemy_visibility_mask": False,
        "enemy_sight_range": 9.0,
        "visibility_xy_indices": (2, 3),
    }})()
    with pytest.raises(FileNotFoundError, match="cannot derive JEPA runtime padding"):
        preflight.derive_runtime_metadata(cfg, ep, {"latent_dim": 6, "rollout_memory_dim": 7}, vis)


def test_preflight_uses_explicit_padding_without_map_source(tmp_path):
    ep = tmp_path / "episode.npz"
    cfg = tmp_path / "config.yaml"
    _episode_npz(ep)
    cfg.write_text(
        """
imag_horizon: 5
observation:
  mode: structured
padding:
  max_agents: 4
  max_enemies: 5
  max_actions: 6
  max_obs_size: 9
""",
        encoding="utf-8",
    )
    vis = type("V", (), {"metadata": lambda self: {
        "enemy_visibility_mask": False,
        "enemy_sight_range": 9.0,
        "visibility_xy_indices": (2, 3),
    }})()
    live, _ = preflight.derive_runtime_metadata(cfg, ep, {"latent_dim": 6, "rollout_memory_dim": 7}, vis)
    assert live["max_agents"] == 4
    assert live["max_enemies"] == 5
    assert live["max_actions"] == 6
