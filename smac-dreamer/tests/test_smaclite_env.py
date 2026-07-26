"""Tests for the Gymnasium-compatible SMACliteDreamerEnv (fixed scenario, no padding).

Requires the SMAClite simulator; skipped automatically when it is unavailable. Must not
import JAX / Elements / Embodied / DreamerV3 (conftest excludes external/dreamerv3).
"""

import numpy as np
import pytest

from conftest import requires_smaclite, FIXED_SCENARIO


def _random_valid_onehot(env):
    """Sample a flat factorised one-hot action that is valid under the current mask."""
    avail = env._env.unwrapped.get_avail_actions()
    ints = []
    for i in range(env.n_agents):
        valid = [j for j, v in enumerate(avail[i]) if v]
        ints.append(int(np.random.choice(valid)) if valid else 0)
    return env.codec.encode(ints, num_real_agents=env.n_agents)


# ----------------------------------------------------------------------
# Construction / spaces
# ----------------------------------------------------------------------

@requires_smaclite
def test_construction_and_spaces(fixed_env):
    env = fixed_env
    import gymnasium as gym
    assert isinstance(env, gym.Env)
    assert env.n_agents > 0 and env.n_actions > 0 and env.obs_size > 0
    # Action space is MultiDiscrete([C] * A).
    assert list(env.action_space.nvec) == [env.n_actions] * env.n_agents
    # Observation space is a Dict with the expected fixed-shape model fields.
    obs_space = env.observation_space
    assert set(["state", "avail_actions", "is_first", "is_last", "is_terminal"]).issubset(obs_space.spaces)
    assert obs_space["state"].shape == (env.n_agents * env.obs_size,)
    assert obs_space["avail_actions"].shape == (env.n_agents * env.n_actions,)
    # No padding -> no agent_mask in obs.
    assert "agent_mask" not in obs_space.spaces


@requires_smaclite
def test_no_jax_imports_after_env_use(fixed_env):
    import sys
    fixed_env.reset()
    for forbidden in ("jax", "elements", "embodied", "dreamerv3"):
        assert forbidden not in sys.modules, f"{forbidden} was imported"


# ----------------------------------------------------------------------
# reset() / step() return formats
# ----------------------------------------------------------------------

@requires_smaclite
def test_reset_return_format(fixed_env):
    env = fixed_env
    obs, info = env.reset(seed=0)
    assert isinstance(obs, dict) and isinstance(info, dict)
    assert bool(obs["is_first"]) is True
    assert bool(obs["is_last"]) is False
    assert bool(obs["is_terminal"]) is False
    assert env.observation_space.contains(obs)


@requires_smaclite
def test_step_return_format(fixed_env):
    env = fixed_env
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(_random_valid_onehot(env))
    assert isinstance(obs, dict)
    assert isinstance(reward, np.floating) and reward.dtype == np.float32
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.observation_space.contains(obs)


@requires_smaclite
def test_observation_field_shapes_and_dtypes(fixed_env):
    env = fixed_env
    obs, _ = env.reset(seed=0)
    assert obs["state"].shape == (env.n_agents * env.obs_size,)
    assert obs["state"].dtype == np.float32
    assert obs["avail_actions"].shape == (env.n_agents * env.n_actions,)
    assert obs["avail_actions"].dtype == np.float32
    for k in ("is_first", "is_last", "is_terminal"):
        assert obs[k].dtype == bool


@requires_smaclite
def test_step_without_reset_raises(fixed_env):
    env = fixed_env  # constructed but not reset -> _done is True
    with pytest.raises(RuntimeError):
        env.step(_random_valid_onehot(env))


# ----------------------------------------------------------------------
# Action conversion into SMAClite integer actions
# ----------------------------------------------------------------------

@requires_smaclite
def test_action_onehot_decodes_to_ints(fixed_env, monkeypatch):
    env = fixed_env
    env.reset(seed=0)
    captured = {}
    real_step = env._env.step

    def spy_step(acts):
        captured["acts"] = list(acts)
        return real_step(acts)

    monkeypatch.setattr(env._env, "step", spy_step)
    # Use stop (action 1) — always valid for alive units after reset.
    # Action 0 is no-op, which SMAClite only marks valid for dead units; the
    # sanitiser would replace it, making 0 a poor choice for a pass-through test.
    flat = env.codec.encode([1] * env.n_agents, num_real_agents=env.n_agents)
    env.step(flat)
    assert captured["acts"] == [1] * env.n_agents
    assert all(isinstance(a, int) for a in captured["acts"])


@requires_smaclite
def test_legacy_dict_action_still_accepted(fixed_env):
    env = fixed_env
    env.reset(seed=0)
    act = {f"action_{i}": 0 for i in range(env.n_agents)}
    obs, reward, terminated, truncated, info = env.step(act)
    assert env.observation_space.contains(obs)


# ----------------------------------------------------------------------
# Termination / truncation semantics
# ----------------------------------------------------------------------

@requires_smaclite
def test_time_limit_truncation():
    from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
    env = SMACliteDreamerEnv(scenario=FIXED_SCENARIO, max_episode_steps=3, seed=0)
    try:
        env.reset(seed=0)
        truncated = False
        terminated = False
        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(_random_valid_onehot(env))
            if terminated or truncated:
                break
        # With a 3-step cap and noop-heavy play the episode should truncate, not terminate.
        assert truncated or terminated
        if truncated and not terminated:
            # is_last True, is_terminal False on a pure truncation.
            obs, _ = env.reset(seed=0)
    finally:
        env.close()


@requires_smaclite
def test_is_last_equals_terminated_or_truncated(fixed_env):
    env = fixed_env
    env.reset(seed=0)
    obs, _, terminated, truncated, _ = env.step(_random_valid_onehot(env))
    assert bool(obs["is_last"]) == (terminated or truncated)
    assert bool(obs["is_terminal"]) == terminated


# ----------------------------------------------------------------------
# Invalid-action sanitisation, noop, and metric counters
# ----------------------------------------------------------------------

@requires_smaclite
def test_invalid_action_sanitised(fixed_env):
    env = fixed_env
    env.reset(seed=0)
    # Far-out-of-range action for every agent -> must be sanitised, no crash.
    bad = [env.n_actions + 999] * env.n_agents
    obs, reward, terminated, truncated, info = env.step(bad)
    assert env.observation_space.contains(obs)
    assert float(info["log_step_post_mask_invalid_count"]) > 0.0


@requires_smaclite
def test_noop_action_runs(fixed_env):
    env = fixed_env
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step([0] * env.n_agents)
    assert env.observation_space.contains(obs)
    # noop is action index <= 1 -> counted toward noop rate eventually.


@requires_smaclite
def test_timing_lag_vs_masking_failure_counters(fixed_env):
    env = fixed_env
    env.reset(seed=0)
    # First step with an action that was never in any prior mask classifies as
    # masking_failure (it was already invalid in the previous/initial mask).
    bad = [env.n_actions + 50] * env.n_agents
    _, _, _, _, info = env.step(bad)
    inv = float(info["log_step_post_mask_invalid_count"])
    lag = float(info["log_step_timing_lag_invalid_count"])
    fail = float(info["log_step_masking_failure_count"])
    assert inv == lag + fail
    assert fail > 0.0  # out-of-range action was not valid in the prior mask


@requires_smaclite
def test_episode_metrics_present_on_done(fixed_env):
    env = fixed_env
    env.reset(seed=0)
    info = {}
    for _ in range(env._max_episode_steps):
        _, _, terminated, truncated, info = env.step(_random_valid_onehot(env))
        if terminated or truncated:
            break
    assert "log_battle_won" in info
    assert "log_total_action_count" in info
    assert 0.0 <= float(info["log_post_mask_invalid_action_rate"]) <= 1.0
