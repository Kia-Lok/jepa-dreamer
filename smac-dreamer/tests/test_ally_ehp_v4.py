"""Tests for the ally_ehp_v4 effective-HP preservation reward ablation.

Two layers:
  * Pure-Python reward-function + _unit_ehp tests (no smaclite/torch) — always run.
  * Env-integration tests (require the SMAClite simulator) — skipped when unavailable.

These must not disturb smaclite_default or dense_v3 (covered unchanged in
test_reward_registry.py).
"""

import types

import numpy as np
import pytest

from smacdreamer.envs.reward_registry import resolve, resolved_params, available, RewardContext
from smacdreamer.envs.smaclite_dreamer_env import _unit_ehp

from conftest import requires_smaclite, FIXED_SCENARIO


W = 0.5  # default w_ally_ehp


def _ctx(**kw):
    """RewardContext with neutral defaults; override only what a test cares about."""
    base = dict(
        base_reward=0.0, gamma=1.0,
        ally_ehp_frac=1.0, prev_ally_ehp_frac=1.0,
        enemy_ehp_frac=1.0, prev_enemy_ehp_frac=1.0,
        terminated=False, truncated=False, battle_won=False,
    )
    base.update(kw)
    return RewardContext(**base)


# ----------------------------------------------------------------------
# Registry plumbing (run metadata / hashing)
# ----------------------------------------------------------------------

def test_registered_and_does_not_shadow_others():
    names = available()
    assert "ally_ehp_v4" in names
    # existing rewards remain available and untouched
    assert {"smaclite_default", "dense_v3"}.issubset(names)


def test_resolved_params_fills_defaults_for_hashing():
    rp = resolved_params("ally_ehp_v4", {"w_ally_ehp": 0.5, "w_win": 1.0})
    assert rp["w_ally_ehp"] == 0.5
    assert rp["w_win"] == 1.0
    # defaults filled so identical effective configs hash identically
    assert rp["w_enemy_ehp"] == 0.0
    assert rp["w_loss"] == 0.0
    assert rp["w_timeout"] == 0.0
    assert resolved_params("ally_ehp_v4", {"w_ally_ehp": 0.5, "w_win": 1.0}) == rp


# ----------------------------------------------------------------------
# _unit_ehp: effective HP includes shields
# ----------------------------------------------------------------------

def test_unit_ehp_includes_shield():
    assert _unit_ehp(types.SimpleNamespace(hp=10.0, shield=5.0)) == pytest.approx(15.0)


def test_unit_ehp_without_shield_attr_defaults_zero():
    assert _unit_ehp(types.SimpleNamespace(hp=7.0)) == pytest.approx(7.0)


def test_unit_ehp_dead_unit_is_zero():
    assert _unit_ehp(types.SimpleNamespace(hp=0.0, shield=0.0)) == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Allied-EHP shaping term
# ----------------------------------------------------------------------

def test_no_change_at_full_health_gives_zero_shaping():
    fn = resolve("ally_ehp_v4")
    r, t = fn(_ctx(ally_ehp_frac=1.0, prev_ally_ehp_frac=1.0))
    assert t["ally_ehp"] == pytest.approx(0.0)
    assert t["shaping_total"] == pytest.approx(0.0)
    assert r == pytest.approx(0.0)


def test_ally_hp_damage_is_negative_immediate_term():
    fn = resolve("ally_ehp_v4")
    _, t = fn(_ctx(prev_ally_ehp_frac=1.0, ally_ehp_frac=0.8))
    # W*(gamma*(0.8-1) - (1-1)) = 0.5*(-0.2) = -0.1
    assert t["ally_ehp"] == pytest.approx(-0.1)
    assert t["ally_ehp"] < 0.0


def test_shield_only_damage_is_negative_term():
    # Shields count toward EHP, so a shield-only hit drops ally_ehp_frac and must register.
    full = _unit_ehp(types.SimpleNamespace(hp=10.0, shield=6.0))   # 16
    hit = _unit_ehp(types.SimpleNamespace(hp=10.0, shield=2.0))    # 12 (only shield lost)
    frac = hit / full
    assert frac < 1.0
    fn = resolve("ally_ehp_v4")
    _, t = fn(_ctx(prev_ally_ehp_frac=1.0, ally_ehp_frac=frac))
    assert t["ally_ehp"] < 0.0


def test_healing_is_positive_relative_to_damaged_state():
    fn = resolve("ally_ehp_v4")
    _, t = fn(_ctx(prev_ally_ehp_frac=0.6, ally_ehp_frac=0.8))
    # W*(gamma*(0.8-1) - (0.6-1)) = 0.5*(-0.2 + 0.4) = +0.1
    assert t["ally_ehp"] == pytest.approx(0.1)
    assert t["ally_ehp"] > 0.0


def test_ally_death_captured_through_ehp_decrease():
    fn = resolve("ally_ehp_v4")
    _, t = fn(_ctx(prev_ally_ehp_frac=1.0, ally_ehp_frac=0.5))  # one of two allies wiped out
    assert t["ally_ehp"] < 0.0


# ----------------------------------------------------------------------
# Termination vs truncation semantics
# ----------------------------------------------------------------------

def test_true_terminal_uses_terminal_potential_zero():
    fn = resolve("ally_ehp_v4")
    # On a true terminal, phi_next is forced to 0: term = W*(gamma*0 - (prev-1)).
    _, t_term = fn(_ctx(prev_ally_ehp_frac=0.5, ally_ehp_frac=0.2, terminated=True))
    expected = W * (0.0 - (0.5 - 1.0))  # = 0.25
    assert t_term["ally_ehp"] == pytest.approx(expected)
    # ... and differs from the non-terminal computation with the same fractions.
    _, t_nonterm = fn(_ctx(prev_ally_ehp_frac=0.5, ally_ehp_frac=0.2, terminated=False))
    assert t_nonterm["ally_ehp"] != pytest.approx(expected)


def test_truncation_does_not_use_true_terminal_semantics():
    fn = resolve("ally_ehp_v4")
    ctx = _ctx(prev_ally_ehp_frac=0.5, ally_ehp_frac=0.2, terminated=False, truncated=True)
    _, t_trunc = fn(ctx)
    # Uses the RAW next potential (not zeroed): W*(gamma*(0.2-1) - (0.5-1)) = 0.5*(-0.8+0.5) = -0.15
    non_terminal_expected = W * (1.0 * (0.2 - 1.0) - (0.5 - 1.0))
    assert t_trunc["ally_ehp"] == pytest.approx(non_terminal_expected)
    terminal_zeroed = W * (0.0 - (0.5 - 1.0))
    assert t_trunc["ally_ehp"] != pytest.approx(terminal_zeroed)


# ----------------------------------------------------------------------
# Terminal / timeout anchors — each applied at most once
# ----------------------------------------------------------------------

def test_win_anchor_once():
    fn = resolve("ally_ehp_v4", {"w_win": 1.0, "w_loss": 1.0, "w_timeout": 1.0})
    _, t = fn(_ctx(terminated=True, truncated=False, battle_won=True))
    assert t["terminal"] == pytest.approx(1.0)
    assert t["timeout"] == pytest.approx(0.0)


def test_loss_anchor_once():
    fn = resolve("ally_ehp_v4", {"w_win": 1.0, "w_loss": 1.0, "w_timeout": 1.0})
    _, t = fn(_ctx(terminated=True, truncated=False, battle_won=False))
    assert t["terminal"] == pytest.approx(-1.0)
    assert t["timeout"] == pytest.approx(0.0)


def test_timeout_anchor_once_and_not_terminal():
    fn = resolve("ally_ehp_v4", {"w_win": 1.0, "w_loss": 1.0, "w_timeout": 1.0})
    _, t = fn(_ctx(terminated=False, truncated=True, battle_won=False))
    assert t["timeout"] == pytest.approx(-1.0)
    assert t["terminal"] == pytest.approx(0.0)


def test_no_anchor_on_intermediate_step():
    fn = resolve("ally_ehp_v4", {"w_win": 1.0, "w_loss": 1.0, "w_timeout": 1.0})
    _, t = fn(_ctx(terminated=False, truncated=False))
    assert t["terminal"] == pytest.approx(0.0)
    assert t["timeout"] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Potential-based telescoping identity (anchors disabled)
# ----------------------------------------------------------------------

def test_potential_telescopes_to_zero_over_full_episode():
    # With anchors off, starting at full health, and the terminal potential forced to 0, the
    # DISCOUNTED sum of the allied-EHP shaping over an episode telescopes to -W*Phi(s0) = 0.
    fn = resolve("ally_ehp_v4")  # anchors default to 0
    gamma = 0.9
    fracs = [1.0, 0.9, 0.7, 0.4]  # fracs[0] = initial (full); transitions fracs[i]->fracs[i+1]
    discounted = 0.0
    for i in range(len(fracs) - 1):
        terminated = (i == len(fracs) - 2)  # last transition is terminal
        _, t = fn(_ctx(prev_ally_ehp_frac=fracs[i], ally_ehp_frac=fracs[i + 1],
                       terminated=terminated, gamma=gamma))
        discounted += (gamma ** i) * t["ally_ehp"]
    assert discounted == pytest.approx(0.0, abs=1e-9)


def test_enemy_ehp_optional_off_by_default():
    fn = resolve("ally_ehp_v4")
    _, t = fn(_ctx(prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.5))  # enemies damaged
    assert t["enemy_ehp"] == pytest.approx(0.0)  # w_enemy_ehp default 0


def test_enemy_ehp_when_enabled_rewards_destruction():
    fn = resolve("ally_ehp_v4", {"w_enemy_ehp": 0.5})
    _, t = fn(_ctx(prev_enemy_ehp_frac=1.0, enemy_ehp_frac=0.5))
    # enemy_phi_prev = 1-1 = 0; enemy_phi_next = 1-0.5 = 0.5; term = 0.5*(1.0*0.5 - 0) = 0.25
    assert t["enemy_ehp"] == pytest.approx(0.25)
    assert t["enemy_ehp"] > 0.0


# ======================================================================
# Env-integration tests (require the SMAClite simulator)
# ======================================================================

def _make_env(reward_name, params=None, max_steps=50):
    from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
    reward_fn = resolve(reward_name, params or {})
    return SMACliteDreamerEnv(
        scenario=FIXED_SCENARIO, max_episode_steps=max_steps, seed=0,
        reward_fn=reward_fn, gamma=0.997,
    )


def _random_valid_onehot(env):
    avail = env._env.unwrapped.get_avail_actions()
    ints = []
    for i in range(env.n_agents):
        valid = [j for j, v in enumerate(avail[i]) if v]
        ints.append(int(np.random.choice(valid)) if valid else 0)
    return env.codec.encode(ints, num_real_agents=env.n_agents)


@requires_smaclite
def test_env_emits_new_ehp_log_keys_on_done():
    env = _make_env("ally_ehp_v4", {"w_ally_ehp": 0.5, "w_win": 1.0, "w_loss": 1.0, "w_timeout": 1.0})
    try:
        env.reset(seed=0)
        info = {}
        for _ in range(env._max_episode_steps):
            _, _, terminated, truncated, info = env.step(_random_valid_onehot(env))
            if terminated or truncated:
                break
        for key in (
            "log_episode_original_env_return", "log_episode_shaped_return",
            "log_episode_total_shaping", "log_episode_ally_ehp_shaping",
            "log_episode_terminal_anchor", "log_episode_timeout_penalty",
            "log_final_ally_ehp_frac", "log_final_enemy_ehp_frac",
            "log_episode_ally_ehp_lost", "log_episode_enemy_ehp_lost",
            "log_shaping_to_original_ratio",
        ):
            assert key in info, f"missing {key}"
        assert 0.0 <= float(info["log_final_ally_ehp_frac"]) <= 1.0
        assert 0.0 <= float(info["log_final_enemy_ehp_frac"]) <= 1.0
    finally:
        env.close()


@requires_smaclite
def test_env_ally_ehp_frac_starts_full_and_is_clamped():
    # 2s3z has Stalkers (shields) -> EHP includes shields; fractions stay within [0,1].
    env = _make_env("ally_ehp_v4")
    try:
        env.reset(seed=0)
        assert env._init_ally_ehp_total > 0.0
        assert env._cur_ally_ehp_frac == pytest.approx(1.0)
        for _ in range(10):
            env.step(_random_valid_onehot(env))
            assert 0.0 <= env._cur_ally_ehp_frac <= 1.0
            assert 0.0 <= env._cur_enemy_ehp_frac <= 1.0
    finally:
        env.close()


@requires_smaclite
def test_env_default_and_dense_v3_still_emit_ehp_diagnostics():
    # The EHP diagnostics are reward-independent, so they must appear for every reward.
    for name in ("smaclite_default", "dense_v3"):
        env = _make_env(name)
        try:
            env.reset(seed=0)
            info = {}
            for _ in range(env._max_episode_steps):
                _, _, terminated, truncated, info = env.step(_random_valid_onehot(env))
                if terminated or truncated:
                    break
            assert "log_final_ally_ehp_frac" in info
            # No EHP shaping is produced by these rewards.
            assert float(info["log_episode_ally_ehp_shaping"]) == pytest.approx(0.0)
        finally:
            env.close()
