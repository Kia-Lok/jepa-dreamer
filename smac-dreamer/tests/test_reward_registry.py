"""Tests for the swappable reward registry. Pure Python — no smaclite/torch required."""

import pytest

from smacdreamer.envs.reward_registry import (
    resolve, resolved_params, available, RewardContext,
)


def test_available_lists_builtins():
    names = available()
    assert {"smaclite_default", "v2_shaping", "dense_v3"}.issubset(names)


def test_smaclite_default_returns_base():
    fn = resolve("smaclite_default")
    r, terms = fn(RewardContext(base_reward=3.5))
    assert r == pytest.approx(3.5)
    assert terms["original"] == pytest.approx(3.5)


def test_unknown_reward_raises():
    with pytest.raises(ValueError):
        resolve("does_not_exist")


def test_v2_shaping_terminal_win_and_kill():
    fn = resolve("v2_shaping", {"win_bonus": 10.0, "enemy_kill_bonus": 1.0})
    r, t = fn(RewardContext(base_reward=1.0, kill_delta=2, is_last=True,
                            battle_won=True, allies_alive=3))
    assert t["win"] == pytest.approx(10.0)
    assert t["kill"] == pytest.approx(2.0)
    assert r == pytest.approx(1.0 + 10.0 + 2.0)


def test_dense_v3_noop_telescopes_to_zero():
    # SANITY CHECK ONLY (not proof of policy-invariance): with nothing changing and
    # gamma=1, the potential differences telescope to 0 and reward == base.
    fn = resolve("dense_v3")
    r, t = fn(RewardContext(base_reward=0.0, enemy_hp_frac=1.0, prev_enemy_hp_frac=1.0,
                            ally_alive_frac=1.0, prev_ally_alive_frac=1.0, gamma=1.0))
    assert t["hp"] == pytest.approx(0.0)
    assert t["ally"] == pytest.approx(0.0)
    assert t["shaping_total"] == pytest.approx(0.0)
    assert r == pytest.approx(0.0)


def test_dense_v3_hp_destroyed_positive():
    fn = resolve("dense_v3")  # w_hp default 0.1
    r, t = fn(RewardContext(base_reward=0.0, prev_enemy_hp_frac=1.0, enemy_hp_frac=0.8,
                            ally_alive_frac=1.0, prev_ally_alive_frac=1.0, gamma=1.0))
    assert t["hp"] == pytest.approx(0.1 * 0.2)  # +0.02


def test_dense_v3_ally_death_negative():
    fn = resolve("dense_v3")  # w_ally default 0.1
    r, t = fn(RewardContext(base_reward=0.0, prev_enemy_hp_frac=1.0, enemy_hp_frac=1.0,
                            prev_ally_alive_frac=1.0, ally_alive_frac=0.5, gamma=1.0))
    assert t["ally"] == pytest.approx(0.1 * (0.5 - 1.0))  # -0.05


def test_dense_v3_terminal_win_loss():
    fn = resolve("dense_v3")
    _, tw = fn(RewardContext(base_reward=0.0, is_last=True, battle_won=True,
                             enemy_hp_frac=0.0, prev_enemy_hp_frac=0.0, gamma=1.0))
    assert tw["win"] == pytest.approx(1.0)
    _, tl = fn(RewardContext(base_reward=0.0, is_last=True, battle_won=False,
                             enemy_hp_frac=0.5, prev_enemy_hp_frac=0.5,
                             ally_alive_frac=1.0, prev_ally_alive_frac=1.0, gamma=1.0))
    assert tl["win"] == pytest.approx(-1.0)  # combined terminal term carries the loss


def test_dense_v3_positioning_off_by_default():
    fn = resolve("dense_v3")
    _, t = fn(RewardContext(base_reward=0.0, allies_alive=3, ally_deaths=0,
                            enemy_hp_frac=1.0, prev_enemy_hp_frac=1.0,
                            ally_alive_frac=1.0, prev_ally_alive_frac=1.0, gamma=1.0))
    assert t["positioning"] == pytest.approx(0.0)


def test_dense_v3_positioning_on_when_weighted():
    fn = resolve("dense_v3", {"nonpotential": {"positioning_weight": 0.5}})
    _, t = fn(RewardContext(base_reward=0.0, allies_alive=4, ally_deaths=0,
                            enemy_hp_frac=1.0, prev_enemy_hp_frac=1.0,
                            ally_alive_frac=1.0, prev_ally_alive_frac=1.0, gamma=1.0))
    assert t["positioning"] > 0.0


def test_resolved_params_fills_defaults_for_hashing():
    rp = resolved_params("dense_v3", {"w_hp": 0.2})
    assert rp["w_hp"] == 0.2          # override kept
    assert rp["w_win"] == 1.0         # default filled
    assert rp["w_ally"] == 0.1        # default filled
    assert rp["nonpotential"] == {"positioning_weight": 0.0}
    # Identical effective configs resolve identically (hash stability precondition).
    assert resolved_params("dense_v3", {"w_hp": 0.2}) == rp
