"""Tests for the dedicated held-out map×seed evaluator (P0.4).

Pure-Python tests inject a fake env_factory + episode_fn, so they need neither torch nor the
SMAClite simulator. One env-integration test (requires_smaclite) proves the reset-seed
propagation fix makes the multimap path deterministic per seed.
"""

import types

import numpy as np
import pytest

from smacdreamer.evaluation import (
    evaluate_heldout, is_validation_improvement, DEFAULT_FIXED_SEEDS,
)

from conftest import requires_smaclite


def _entries(names):
    return [types.SimpleNamespace(name=n, family="fam") for n in names]


class _FakeEnv:
    def __init__(self, name):
        self.map_name = name
        self.closed = False

    def close(self):
        self.closed = True


def _fake_factory_recording(closed_box):
    def factory(entries, pad_dims, mode, *args):
        env = _FakeEnv(entries[0].name)
        closed_box.append(env)
        return env
    return factory


# ----------------------------------------------------------------------
# Coverage: every held-out map is visited under every configured seed
# ----------------------------------------------------------------------

def test_visits_every_map_for_every_seed():
    maps = ["m0", "m1", "m2"]
    seeds = [7, 8]
    visited = []

    def episode_fn(agent, env, seed, device, max_steps):
        visited.append((env.map_name, seed))
        return dict(win=True, original_return=1.0, length=10, timeout=False,
                    final_ally_ehp_frac=1.0, final_enemy_ehp_frac=0.0)

    envs = []
    report = evaluate_heldout(
        None, _entries(maps), pad_dims=None, seeds=seeds,
        env_factory=_fake_factory_recording(envs), episode_fn=episode_fn,
    )
    assert set(visited) == {(m, s) for m in maps for s in seeds}
    assert len(visited) == len(maps) * len(seeds)
    assert report["n_maps"] == 3
    assert report["episodes_per_map"] == 2
    assert report["n_episodes_total"] == 6
    assert all(e.closed for e in envs)  # every per-map env is closed


def test_seeds_are_used_in_order_and_deduped_per_map():
    seeds = [0, 1, 2]
    counts = {}

    def episode_fn(agent, env, seed, device, max_steps):
        counts[env.map_name] = counts.get(env.map_name, 0) + 1
        return dict(win=False, original_return=0.0, length=1, timeout=True,
                    final_ally_ehp_frac=0.0, final_enemy_ehp_frac=1.0)

    evaluate_heldout(None, _entries(["a", "b"]), None, seeds=seeds,
                     env_factory=_fake_factory_recording([]), episode_fn=episode_fn)
    assert counts == {"a": 3, "b": 3}


# ----------------------------------------------------------------------
# Aggregation: per-map, macro and micro
# ----------------------------------------------------------------------

def test_per_map_and_macro_micro_aggregation():
    # win_all wins every seed; lose_all loses every seed; returns encode the seed.
    seeds = [0, 1]

    def episode_fn(agent, env, seed, device, max_steps):
        win = env.map_name.startswith("win")
        return dict(win=win, original_return=float(seed), length=5 + seed,
                    timeout=(not win), final_ally_ehp_frac=0.5, final_enemy_ehp_frac=0.25)

    report = evaluate_heldout(
        None, _entries(["win_a", "win_b", "lose_c"]), None, seeds=seeds,
        env_factory=_fake_factory_recording([]), episode_fn=episode_fn,
    )
    pm = report["per_map"]
    assert pm["win_a"]["win_rate"] == pytest.approx(1.0)
    assert pm["lose_c"]["win_rate"] == pytest.approx(0.0)
    assert pm["win_a"]["original_return"] == pytest.approx(0.5)  # mean of seeds 0,1
    assert pm["lose_c"]["timeout_rate"] == pytest.approx(1.0)
    assert pm["win_a"]["family"] == "fam"

    # macro = each map one sample: mean over (1, 1, 0)
    assert report["macro"]["win_rate"] == pytest.approx(2.0 / 3.0)
    # micro = each episode one sample: 4 wins / 6 episodes
    assert report["micro"]["win_rate"] == pytest.approx(4.0 / 6.0)
    assert report["macro"]["final_ally_ehp_frac"] == pytest.approx(0.5)


def test_primary_metric_is_macro_winrate():
    def episode_fn(agent, env, seed, device, max_steps):
        return dict(win=True, original_return=2.0, length=3, timeout=False,
                    final_ally_ehp_frac=1.0, final_enemy_ehp_frac=0.0)

    report = evaluate_heldout(None, _entries(["x"]), None, seeds=[0],
                              env_factory=_fake_factory_recording([]), episode_fn=episode_fn)
    assert report["primary_metric"] == "macro_heldout_win_rate"
    assert report["primary_value"] == pytest.approx(report["macro"]["win_rate"])
    assert "NEVER" in report["selection_note"]  # shaped-return guard documented


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------

def test_empty_entries_raises():
    with pytest.raises(ValueError):
        evaluate_heldout(None, [], None, seeds=[0],
                         env_factory=_fake_factory_recording([]), episode_fn=lambda *a: {})


def test_empty_seeds_raises():
    with pytest.raises(ValueError):
        evaluate_heldout(None, _entries(["a"]), None, seeds=[],
                         env_factory=_fake_factory_recording([]), episode_fn=lambda *a: {})


def test_default_fixed_seeds_present():
    assert len(DEFAULT_FIXED_SEEDS) >= 1


# ----------------------------------------------------------------------
# P0.4: best-checkpoint selection rule (macro win rate; tie-break macro return)
# ----------------------------------------------------------------------

def test_validation_improvement_win_rate_dominates():
    # Higher macro win rate wins regardless of return.
    assert is_validation_improvement(0.6, 0.0, best_win_rate=0.5, best_original_return=100.0)
    # Lower win rate never improves, even with a much higher return.
    assert not is_validation_improvement(0.4, 999.0, best_win_rate=0.5, best_original_return=0.0)


def test_validation_improvement_tie_broken_by_return():
    # Equal win rate -> higher ORIGINAL return wins.
    assert is_validation_improvement(0.5, 12.0, best_win_rate=0.5, best_original_return=10.0)
    # Equal win rate + lower/equal return -> no improvement.
    assert not is_validation_improvement(0.5, 10.0, best_win_rate=0.5, best_original_return=10.0)
    assert not is_validation_improvement(0.5, 9.0, best_win_rate=0.5, best_original_return=10.0)


def test_validation_improvement_first_eval_always_improves():
    # Initial best is (-1.0, -inf) in ValidationTrainer, so the first validation always saves.
    assert is_validation_improvement(0.0, 0.0, best_win_rate=-1.0, best_original_return=float("-inf"))


# ----------------------------------------------------------------------
# P0.4: explicit-folder discovery guards (no smaclite needed for missing folders)
# ----------------------------------------------------------------------

def test_discover_folders_missing_folder_raises():
    from smacdreamer.envs.map_discovery import discover_folders
    with pytest.raises(FileNotFoundError):
        discover_folders("does/not/exist/train", "does/not/exist/validation", verbose=False)


# ======================================================================
# Env-integration: reset-seed determinism on the multimap path
# ======================================================================

@requires_smaclite
def test_reset_seed_determinism_on_multimap_path():
    from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
    from smacdreamer.envs.map_sampler import MapSampler, MapEntry

    sampler = MapSampler.from_entries([MapEntry(name="2s3z", type="builtin")], mode="fixed", seed=0)
    env = SMACliteDreamerEnv(scenario="2s3z", max_episode_steps=20, seed=0, map_sampler=sampler)

    def run(seed):
        obs, _ = env.reset(seed=seed)
        states = [np.asarray(obs["state"]).copy()]
        for _ in range(5):
            flat = env.codec.encode([1] * env.n_agents, num_real_agents=env.n_agents)  # all "stop"
            obs, _, terminated, truncated, _ = env.step(flat)
            states.append(np.asarray(obs["state"]).copy())
            if terminated or truncated:
                break
        return states

    try:
        a = run(0)
        b = run(0)
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert np.allclose(x, y), "same seed must reproduce an identical trajectory"
    finally:
        env.close()
