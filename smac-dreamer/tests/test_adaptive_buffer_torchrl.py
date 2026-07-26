from types import SimpleNamespace

import pytest
import torch

torchrl = pytest.importorskip("torchrl")
pytest.importorskip("tensordict")
from tensordict import TensorDict

from adaptive_buffer import AdaptiveBuffer


def make_config():
    return SimpleNamespace(
        device="cpu",
        storage_device="cpu",
        batch_size=2,
        batch_length=4,
        max_size=256,
        storage_backend="tensor",
        scratch_dir=None,
        adaptive_priority={
            "sequence": {
                "enabled": True,
                "candidate_multiplier": 2,
                "alpha": 0.6,
                "beta_start": 0.4,
                "beta_end": 1.0,
                "beta_anneal_env_steps": 1000,
                "eps": 1e-6,
                "min_priority": 1e-3,
                "max_priority": 100.0,
                "cache_max_entries": 256,
                "seed": 7,
            }
        },
    )


def transition(step: int) -> TensorDict:
    batch = 2
    return TensorDict(
        {
            "action": torch.full((batch, 3), float(step)),
            "stoch": torch.full((batch, 2), float(step)),
            "deter": torch.full((batch, 3), float(step)),
            "episode": torch.arange(batch, dtype=torch.int32),
            "is_first": torch.zeros(batch, 1),
            "is_last": torch.zeros(batch, 1),
            "is_terminal": torch.zeros(batch, 1),
            "reward": torch.zeros(batch, 1),
            "log_map_id": torch.tensor([[10.0], [20.0]]),
        },
        batch_size=[batch],
    )


def test_candidate_sequence_per_preserves_shapes_and_updates():
    buffer = AdaptiveBuffer(make_config())
    for step in range(12):
        buffer.add_transition(transition(step))

    data, info, initial = buffer.sample()
    assert tuple(data.batch_size) == (2, 4)
    assert initial[0].shape == (2, 2)
    assert initial[1].shape == (2, 3)
    assert info.sequence_uids.shape == (2,)
    assert info.importance_weights.shape == (2,)
    assert torch.isfinite(info.importance_weights).all()
    assert torch.all(info.importance_weights > 0)
    assert float(info.importance_weights.max()) <= 1.0 + 1e-6

    buffer.update_priorities(info.sequence_uids, torch.tensor([1.0, 9.0]))
    assert len(buffer._priority_by_uid) >= 1

    buffer.update(
        info.transition_indices,
        torch.zeros(2, 4, 2),
        torch.zeros(2, 4, 3),
    )
    buffer.close()
