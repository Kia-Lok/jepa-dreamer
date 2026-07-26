import torch

from smacdreamer.envs.map_sampler import MapEntry, MapSampler


def test_adaptive_sampler_reads_shared_probabilities():
    entries = [
        MapEntry(name="a", type="builtin", map_id=1),
        MapEntry(name="b", type="builtin", map_id=2),
        MapEntry(name="c", type="builtin", map_id=3),
    ]
    probabilities = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64).share_memory_()
    version = torch.zeros((), dtype=torch.int64).share_memory_()
    sampler = MapSampler.from_entries(
        entries,
        mode="adaptive_priority",
        seed=123,
        shared_probabilities=probabilities,
        shared_version=version,
    )
    assert all(sampler.next().map_id == 2 for _ in range(20))


def test_invalid_shared_probabilities_fall_back_safely():
    entries = [
        MapEntry(name="a", type="builtin", map_id=1),
        MapEntry(name="b", type="builtin", map_id=2),
    ]
    probabilities = torch.tensor([float("nan"), 0.0], dtype=torch.float64).share_memory_()
    sampler = MapSampler.from_entries(
        entries,
        mode="adaptive_priority",
        seed=4,
        shared_probabilities=probabilities,
    )
    sampled = {sampler.next().map_id for _ in range(20)}
    assert sampled.issubset({1, 2})
    assert sampled
