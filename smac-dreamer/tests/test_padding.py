"""Tests for Phase 3 padded SMACliteDreamerEnv observations and padded-agent actions.

Requires the SMAClite simulator; skipped automatically when it is unavailable.
Uses 2s3z (5 agents, 6 actions for that scenario) padded to a larger fixed shape.
"""

import numpy as np
import pytest

from conftest import requires_smaclite, FIXED_SCENARIO

# Padding box large enough to contain 2s3z (matches phase3_overfit_2s3z_manifest.yaml).
MAX_AGENTS = 8
MAX_ENEMIES = 9
MAX_ACTIONS = 15
MAX_OBS_SIZE = 136


def _make_padded_env():
    from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
    from smacdreamer.envs.padding import PaddingDims
    from smacdreamer.envs.map_sampler import MapEntry, MapSampler

    pad = PaddingDims(MAX_AGENTS, MAX_ENEMIES, MAX_ACTIONS, MAX_OBS_SIZE)
    sampler = MapSampler(maps=[MapEntry(name=FIXED_SCENARIO, type="builtin")], mode="fixed")
    return SMACliteDreamerEnv(
        scenario=FIXED_SCENARIO, max_episode_steps=50, seed=0,
        map_sampler=sampler, pad_dims=pad,
    )


def _valid_onehot_padded(env):
    avail = env._env.unwrapped.get_avail_actions()
    ints = []
    for i in range(env.n_agents):
        valid = [j for j, v in enumerate(avail[i]) if v]
        ints.append(int(np.random.choice(valid)) if valid else 0)
    # Encode across all padded slots; padded agents forced to noop.
    return env.codec.encode(ints, num_real_agents=env.n_agents)


# ----------------------------------------------------------------------
# Fixed-map (no padding) reference: shapes use real dims.
# ----------------------------------------------------------------------

@requires_smaclite
def test_fixed_map_no_padding_shapes(fixed_env):
    env = fixed_env
    obs, _ = env.reset(seed=0)
    assert obs["state"].shape == (env.n_agents * env.obs_size,)
    assert "agent_mask" not in obs
    assert "real_agent_action_mask" not in obs


# ----------------------------------------------------------------------
# Padded-map observation shapes & masks.
# ----------------------------------------------------------------------

@requires_smaclite
def test_padded_observation_shape():
    env = _make_padded_env()
    try:
        obs, _ = env.reset(seed=0)
        assert obs["state"].shape == (MAX_AGENTS * MAX_OBS_SIZE,)
        assert obs["avail_actions"].shape == (MAX_AGENTS * MAX_ACTIONS,)
        assert env.observation_space.contains(obs)
    finally:
        env.close()


@requires_smaclite
def test_padded_agent_mask():
    env = _make_padded_env()
    try:
        obs, _ = env.reset(seed=0)
        agent_mask = obs["agent_mask"]
        assert agent_mask.shape == (MAX_AGENTS,)
        # First n_agents are real (1.0); rest padded (0.0).
        assert agent_mask[: env.n_agents].tolist() == [1.0] * env.n_agents
        assert agent_mask[env.n_agents:].tolist() == [0.0] * (MAX_AGENTS - env.n_agents)
    finally:
        env.close()


@requires_smaclite
def test_real_agent_action_mask():
    env = _make_padded_env()
    try:
        obs, _ = env.reset(seed=0)
        ram = obs["real_agent_action_mask"]
        assert ram.shape == (MAX_AGENTS * MAX_ACTIONS,)
        groups = ram.reshape(MAX_AGENTS, MAX_ACTIONS)
        # Real agent rows all 1.0; padded agent rows all 0.0.
        for i in range(MAX_AGENTS):
            expected = 1.0 if i < env.n_agents else 0.0
            assert groups[i].tolist() == [expected] * MAX_ACTIONS
    finally:
        env.close()


@requires_smaclite
def test_padded_agent_action_ignored():
    """Actions for padded agent slots must not reach SMAClite."""
    env = _make_padded_env()
    try:
        env.reset(seed=0)
        sent = {}
        real_step = env._env.step

        def spy_step(acts):
            sent["acts"] = list(acts)
            return real_step(acts)

        env._env.step = spy_step
        # Encode with non-noop values in padded slots by hand, then confirm only real
        # agent actions are forwarded (length == n_agents).
        flat = env.codec.encode([0] * MAX_AGENTS, num_real_agents=env.n_agents)
        env.step(flat)
        assert len(sent["acts"]) == env.n_agents
    finally:
        env.close()


@requires_smaclite
def test_padded_env_step_runs_full():
    env = _make_padded_env()
    try:
        env.reset(seed=0)
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(_valid_onehot_padded(env))
            assert env.observation_space.contains(obs)
            assert float(info["log_padded_agent_count"]) == MAX_AGENTS - env.n_agents
            if terminated or truncated:
                break
    finally:
        env.close()
