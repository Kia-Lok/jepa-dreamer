"""Env-level action-masking tests (P0.1): strict mode + requested-vs-executed alignment.

Requires the SMAClite simulator (skipped otherwise).
"""

import numpy as np
import pytest

from conftest import requires_smaclite, FIXED_SCENARIO


def _make(strict):
    from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
    return SMACliteDreamerEnv(scenario=FIXED_SCENARIO, max_episode_steps=20, seed=0,
                              strict_actions=strict)


@requires_smaclite
def test_strict_mode_raises_on_invalid_action():
    env = _make(strict=True)
    try:
        env.reset(seed=0)
        bad = [env.n_actions + 5] * env.n_agents      # every action out of range -> invalid
        with pytest.raises(ValueError):
            env.step(bad)
    finally:
        env.close()


@requires_smaclite
def test_requested_equals_executed_for_valid_actions_under_strict():
    env = _make(strict=True)
    try:
        env.reset(seed=0)
        # action 1 ("stop") is valid for every alive unit right after reset.
        flat = env.codec.encode([1] * env.n_agents, num_real_agents=env.n_agents)
        obs, _, _, _, info = env.step(flat)
        assert env._last_requested_action == env._last_executed_action == [1] * env.n_agents
        assert float(info["log_step_post_mask_invalid_count"]) == 0.0
        assert float(info["log_step_sanitisation_occurred"]) == 0.0
    finally:
        env.close()


@requires_smaclite
def test_non_strict_sanitises_and_flags_occurred():
    env = _make(strict=False)
    try:
        env.reset(seed=0)
        bad = [env.n_actions + 5] * env.n_agents
        obs, _, _, _, info = env.step(bad)
        assert env._last_requested_action != env._last_executed_action   # sanitiser changed it
        assert float(info["log_step_post_mask_invalid_count"]) > 0.0
        assert float(info["log_step_sanitisation_occurred"]) == 1.0
        # every EXECUTED action is valid under the current mask
        avail = env._env.unwrapped.get_avail_actions()
        for i, a in enumerate(env._last_executed_action):
            valid = [j for j, v in enumerate(avail[i]) if v]
            assert (a in valid) or not valid
    finally:
        env.close()
