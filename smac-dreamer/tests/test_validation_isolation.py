import os
import time
import types

import pytest

from smacdreamer.evaluation import evaluate_heldout
from smacdreamer.isolated_env import EnvFactorySpec, IsolatedEnvProxy


CHILD_PIDS = []


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _wait_dead(pid, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


class FakeValidationEnv:
    def __init__(self, entries, *args, **kwargs):
        self.map_name = entries[0].name
        self.pid = os.getpid()
        self._blob = bytearray(2_000_000)
        self._reset_count = 0

    def reset(self, seed=None):
        if int(seed) == 13:
            raise RuntimeError(f"injected reset failure for {self.map_name}")
        self._reset_count += 1
        return {"pid": self.pid, "seed": int(seed), "map": self.map_name}

    def step(self, action):
        return {"pid": self.pid, "map": self.map_name}, 1.0, True, {"battle_won": True}

    def close(self):
        self._blob = None


def fake_validation_factory(entries, pad_dims, mode, base_seed, worker_idx, reward_name,
                            reward_params, gamma, max_episode_steps, obs_mode="flat"):
    return FakeValidationEnv(entries)


def isolated_factory(entries, pad_dims, mode, base_seed, worker_idx, reward_name,
                     reward_params, gamma, max_episode_steps, obs_mode="flat",
                     shutdown_timeout_seconds=5.0):
    env = IsolatedEnvProxy(
        EnvFactorySpec(__name__, "fake_validation_factory"),
        (entries, pad_dims, mode, base_seed, worker_idx, reward_name,
         reward_params, gamma, max_episode_steps, obs_mode),
        {},
        map_name=entries[0].name,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
    )
    CHILD_PIDS.append(env.pid)
    return env


def _entries(names):
    return [types.SimpleNamespace(name=n, family="fam") for n in names]


def _episode_fn(agent, env, seed, device, max_steps):
    obs = env.reset(seed=seed)
    step_obs, reward, done, info = env.step([0])
    assert done
    assert obs["pid"] == step_obs["pid"]
    agent.setdefault("pids", []).append(obs["pid"])
    agent.setdefault("maps", []).append(obs["map"])
    return dict(win=bool(info["battle_won"]), original_return=reward, length=1,
                timeout=False, final_ally_ehp_frac=1.0, final_enemy_ehp_frac=0.0)


def test_validation_envs_are_child_per_map_and_reused_per_seed():
    parent = os.getpid()
    agent = {}
    report = evaluate_heldout(
        agent, _entries(["m0", "m1"]), None, seeds=[0, 1, 2],
        env_factory=isolated_factory, episode_fn=_episode_fn,
    )
    assert report["macro"]["win_rate"] == pytest.approx(1.0)
    assert len(set(agent["pids"][:3])) == 1
    assert len(set(agent["pids"][3:])) == 1
    assert agent["pids"][0] != agent["pids"][3]
    assert all(pid != parent for pid in agent["pids"])
    assert agent["maps"] == ["m0"] * 3 + ["m1"] * 3
    assert all(_wait_dead(pid) for pid in set(agent["pids"]))


def test_validation_children_cleanup_after_episode_exception():
    before = {p.pid for p in __import__("multiprocessing").active_children()}
    with pytest.raises(Exception, match="injected reset failure"):
        evaluate_heldout(
            {}, _entries(["bad"]), None, seeds=[13],
            env_factory=isolated_factory, episode_fn=_episode_fn,
        )
    after = {p.pid for p in __import__("multiprocessing").active_children()}
    assert after <= before
    assert all(_wait_dead(pid) for pid in CHILD_PIDS)


def test_isolated_metrics_match_in_process_fake():
    isolated_agent = {}
    direct_agent = {}
    isolated = evaluate_heldout(
        isolated_agent, _entries(["m0", "m1"]), None, seeds=[0, 1],
        env_factory=isolated_factory, episode_fn=_episode_fn,
    )
    direct = evaluate_heldout(
        direct_agent, _entries(["m0", "m1"]), None, seeds=[0, 1],
        env_factory=fake_validation_factory, episode_fn=_episode_fn,
    )
    assert isolated["macro"] == direct["macro"]
    assert isolated["micro"] == direct["micro"]
