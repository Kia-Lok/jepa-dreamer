import pathlib
import sys
import os
import signal

import numpy as np
import pytest
import torch
from gymnasium import spaces


ROOT = pathlib.Path(__file__).resolve().parent.parent
R2 = ROOT / "external" / "r2dreamer"
if str(R2) not in sys.path:
    sys.path.insert(0, str(R2))

from envs.parallel import ParallelEnv
from smacdreamer.envs.map_sampler import MapEntry, MapSampler
from smacdreamer.r2dreamer_factory import _worker_seed


class TinyEnv:
    def __init__(self, slot, generation, episode_len=2):
        self.slot = int(slot)
        self.generation = int(generation)
        self.pid = __import__("os").getpid()
        self.episode_len = int(episode_len)
        self.t = 0
        self.reset_count = 0
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-1, 1, shape=(2,), dtype=np.float32),
            "is_first": spaces.Box(0, 1, shape=(), dtype=bool),
            "is_last": spaces.Box(0, 1, shape=(), dtype=bool),
            "is_terminal": spaces.Box(0, 1, shape=(), dtype=bool),
        })
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def _obs(self, first=False, last=False):
        return {
            "state": np.asarray([self.slot, self.generation], dtype=np.float32),
            "is_first": np.asarray(first),
            "is_last": np.asarray(last),
            "is_terminal": np.asarray(last),
        }

    def reset(self):
        self.t = 0
        self.reset_count += 1
        return self._obs(first=True)

    def step(self, action):
        self.t += 1
        done = self.t >= self.episode_len
        return self._obs(last=done), float(self.generation), done, {}

    def close(self):
        pass

    def die(self):
        __import__("os")._exit(7)


def make_tiny(slot, generation=0):
    return lambda: TinyEnv(slot, generation)


def make_tiny_one_arg(slot):
    return lambda: TinyEnv(slot, 0)


def bad_constructor(slot, generation=0):
    raise RuntimeError("constructor exploded")


class SamplerEnv:
    def __init__(self, slot, generation, completed_episode_offset, mode, names, base_seed=123):
        self.slot = int(slot)
        self.generation = int(generation)
        self.completed_episode_offset = int(completed_episode_offset)
        self.sampler_seed = _worker_seed(base_seed, self.slot, 0)
        self.simulator_seed = _worker_seed(base_seed, self.slot, self.generation)
        entries = [MapEntry(name=n, type="custom", map_id=i) for i, n in enumerate(names)]
        self.sampler = MapSampler.from_entries(entries, mode=mode, seed=self.sampler_seed)
        self.sampler.advance(self.completed_episode_offset)
        self.observation_space = spaces.Dict({
            "map_id": spaces.Box(0, 10_000, shape=(), dtype=np.int64),
            "generation": spaces.Box(0, 10_000, shape=(), dtype=np.int64),
            "completed_episode_offset": spaces.Box(0, 1_000_000, shape=(), dtype=np.int64),
            "sampler_seed": spaces.Box(0, 2**32 - 1, shape=(), dtype=np.int64),
            "simulator_seed": spaces.Box(0, 2**32 - 1, shape=(), dtype=np.int64),
            "sampling_cycle": spaces.Box(0, 1_000_000, shape=(), dtype=np.int64),
            "maps_seen_this_cycle": spaces.Box(0, 1_000_000, shape=(), dtype=np.int64),
            "total_unique_maps_seen": spaces.Box(0, 1_000_000, shape=(), dtype=np.int64),
            "is_first": spaces.Box(0, 1, shape=(), dtype=bool),
            "is_last": spaces.Box(0, 1, shape=(), dtype=bool),
            "is_terminal": spaces.Box(0, 1, shape=(), dtype=bool),
        })
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def _obs(self, entry, first=False, last=False):
        cov = self.sampler.coverage_metrics()
        return {
            "map_id": np.asarray(entry.map_id, dtype=np.int64),
            "generation": np.asarray(self.generation, dtype=np.int64),
            "completed_episode_offset": np.asarray(self.completed_episode_offset, dtype=np.int64),
            "sampler_seed": np.asarray(self.sampler_seed, dtype=np.int64),
            "simulator_seed": np.asarray(self.simulator_seed, dtype=np.int64),
            "sampling_cycle": np.asarray(cov["sampling_cycle"], dtype=np.int64),
            "maps_seen_this_cycle": np.asarray(cov["maps_seen_this_cycle"], dtype=np.int64),
            "total_unique_maps_seen": np.asarray(cov["total_unique_maps_seen"], dtype=np.int64),
            "is_first": np.asarray(first),
            "is_last": np.asarray(last),
            "is_terminal": np.asarray(last),
        }

    def reset(self):
        return self._obs(self.sampler.next(), first=True)

    def step(self, action):
        entry = self.sampler.peek()
        return self._obs(entry, last=True), 0.0, True, {}

    def close(self):
        pass


def make_sampler_ctor(mode, names, base_seed=123):
    def ctor(slot, generation=0, completed_episode_offset=0):
        return lambda: SamplerEnv(slot, generation, completed_episode_offset, mode, names, base_seed)
    return ctor


def _step(envs, done):
    action = torch.zeros(envs.env_num, 1)
    return envs.step(action, torch.as_tensor(done, dtype=torch.bool))


def _collect_sampler_run(mode, recycle_every, episodes=100, names=None):
    names = names or [f"m{i}" for i in range(10)]
    envs = ParallelEnv(
        make_sampler_ctor(mode, names),
        1,
        "cpu",
        max_episodes_per_worker=recycle_every,
        shutdown_timeout_seconds=1,
    )
    rows = []
    try:
        done = torch.tensor([True])
        for _ in range(episodes):
            trans, done = _step(envs, done)
            rows.append({
                "map_id": int(trans["map_id"][0]),
                "generation": int(trans["generation"][0]),
                "offset": int(trans["completed_episode_offset"][0]),
                "sampler_seed": int(trans["sampler_seed"][0]),
                "simulator_seed": int(trans["simulator_seed"][0]),
                "sampling_cycle": int(trans["sampling_cycle"][0]),
                "maps_seen_this_cycle": int(trans["maps_seen_this_cycle"][0]),
                "total_unique_maps_seen": int(trans["total_unique_maps_seen"][0]),
            })
            trans, done = _step(envs, done)
            assert bool(done[0])
        return rows, envs.worker_infos()
    finally:
        envs.close()


def test_worker_restarts_at_episode_boundary_and_other_slots_unchanged():
    envs = ParallelEnv(make_tiny, 2, "cpu", max_episodes_per_worker=1, shutdown_timeout_seconds=1)
    try:
        _, done = _step(envs, [True, True])
        pids0 = [i["pid"] for i in envs.worker_infos()]
        _, done = _step(envs, [False, False])
        assert not done.any()
        assert [i["pid"] for i in envs.worker_infos()] == pids0
        trans, done = _step(envs, [False, False])
        assert done.all()
        assert trans["state"].shape == (2, 2)
        _, done = _step(envs, done)
        pids1 = [i["pid"] for i in envs.worker_infos()]
        assert pids1[0] != pids0[0]
        assert pids1[1] != pids0[1]
        assert [i["generation"] for i in envs.worker_infos()] == [1, 1]
    finally:
        envs.close()


def test_one_argument_constructor_still_supported():
    envs = ParallelEnv(make_tiny_one_arg, 1, "cpu", shutdown_timeout_seconds=1)
    try:
        trans, done = _step(envs, [True])
        assert int(trans["state"][0, 0]) == 0
    finally:
        envs.close()


def test_constructor_typeerror_fallback_does_not_hide_real_exceptions():
    with pytest.raises(RuntimeError, match="constructor exploded"):
        ParallelEnv(bad_constructor, 1, "cpu", shutdown_timeout_seconds=1).observation_space


def test_single_slot_restart_does_not_restart_other_slot():
    envs = ParallelEnv(make_tiny, 2, "cpu", max_episodes_per_worker=1, shutdown_timeout_seconds=1)
    try:
        _step(envs, [True, True])
        pids0 = [i["pid"] for i in envs.worker_infos()]
        _step(envs, [False, False])
        _, done = _step(envs, [False, False])
        assert done.all()
        _step(envs, [True, False])
        pids1 = [i["pid"] for i in envs.worker_infos()]
        assert pids1[0] != pids0[0]
        assert pids1[1] == pids0[1]
    finally:
        envs.close()


def test_generation_sequence_is_deterministic():
    def run_once():
        envs = ParallelEnv(make_tiny, 1, "cpu", max_episodes_per_worker=1, shutdown_timeout_seconds=1)
        states = []
        try:
            done = torch.tensor([True])
            for _ in range(5):
                trans, done = _step(envs, done)
                states.append(tuple(trans["state"][0].tolist()))
                if not bool(done[0]):
                    trans, done = _step(envs, done)
                    states.append(tuple(trans["state"][0].tolist()))
            return states
        finally:
            envs.close()

    assert run_once() == run_once()


def test_unexpected_worker_death_reports_context():
    envs = ParallelEnv(make_tiny, 1, "cpu", shutdown_timeout_seconds=1)
    try:
        _step(envs, [True])
        os.kill(envs.worker_infos()[0]["pid"], signal.SIGKILL)
        with pytest.raises(RuntimeError, match="worker slot=0.*pid=.*phase=step"):
            _step(envs, [False])
    finally:
        envs.close()


@pytest.mark.parametrize("mode", ["shuffled_round_robin", "round_robin", "uniform_map"])
def test_recycled_worker_map_sequence_matches_unrecycled(mode):
    unrecycled, _ = _collect_sampler_run(mode, recycle_every=0, episodes=100)
    recycled, _ = _collect_sampler_run(mode, recycle_every=25, episodes=100)
    assert [r["map_id"] for r in recycled] == [r["map_id"] for r in unrecycled]


def test_first_shuffled_round_robin_cycle_contains_every_map_once():
    names = [f"m{i}" for i in range(10)]
    rows, _ = _collect_sampler_run("shuffled_round_robin", recycle_every=25, episodes=10, names=names)
    assert sorted(r["map_id"] for r in rows) == list(range(10))
    assert rows[-1]["sampling_cycle"] == 1
    assert rows[-1]["maps_seen_this_cycle"] == 0
    assert rows[-1]["total_unique_maps_seen"] == 10


def test_recycling_changes_simulator_seed_not_sampler_seed_or_cursor():
    rows, infos = _collect_sampler_run("shuffled_round_robin", recycle_every=25, episodes=30)
    assert infos[0]["generation"] == 1
    assert infos[0]["completed_episode_offset"] == 30
    assert rows[0]["generation"] == 0
    assert rows[25]["generation"] == 1
    assert rows[0]["sampler_seed"] == rows[25]["sampler_seed"]
    assert rows[0]["simulator_seed"] != rows[25]["simulator_seed"]
    unrecycled, _ = _collect_sampler_run("shuffled_round_robin", recycle_every=0, episodes=30)
    assert rows[25]["map_id"] == unrecycled[25]["map_id"]
    assert rows[25]["total_unique_maps_seen"] == unrecycled[25]["total_unique_maps_seen"]


def test_max_episodes_zero_does_not_recycle():
    rows, infos = _collect_sampler_run("round_robin", recycle_every=0, episodes=30)
    assert infos[0]["generation"] == 0
    assert len({r["generation"] for r in rows}) == 1
