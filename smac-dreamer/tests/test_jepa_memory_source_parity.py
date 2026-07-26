import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smacdreamer.jepa.memory import (
    ActionConditionedEntityRolloutGRUMemory as CompatActionMemory,
    EntityRolloutGRUMemory as CompatEntityMemory,
)


def _copy_state(src, dst):
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    assert not missing
    assert not unexpected


def _assert_update_sequence_close(src, compat, *, uses_action):
    torch.manual_seed(0)
    z = torch.randn(2, 3, 4)
    prev = torch.randn(2, 3, 5)
    masks = [
        torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=torch.float32),
        torch.tensor([[1, 1, 1], [0, 1, 0]], dtype=torch.float32),
    ]
    action = torch.zeros(2, 2, 3)
    action[..., 0] = 1.0
    action[:, 1, 1] = 1.0
    action_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
    src_mem = prev.clone()
    compat_mem = prev.clone()
    for mask in masks:
        if uses_action:
            src_mem = src.update(z, src_mem, mask, action=action, action_mask=action_mask)
            compat_mem = compat.update(z, compat_mem, mask, action=action, action_mask=action_mask)
        else:
            src_mem = src.update(z, src_mem, mask)
            compat_mem = compat.update(z, compat_mem, mask)
        torch.testing.assert_close(compat_mem, src_mem, rtol=1e-6, atol=1e-6)


def test_entity_rollout_memory_source_parity_when_installed():
    source_module = pytest.importorskip("smac_jepa.modules.rollout_memory")
    Source = getattr(source_module, "EntityRolloutGRUMemory")
    torch.manual_seed(123)
    src = Source(latent_dim=4, memory_dim=5, hidden_dim=7, residual=True)
    compat = CompatEntityMemory(latent_dim=4, memory_dim=5, hidden_dim=7, residual=True)
    _copy_state(src, compat)
    _assert_update_sequence_close(src, compat, uses_action=False)


def test_action_conditioned_rollout_memory_source_parity_when_installed():
    source_module = pytest.importorskip("smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments")
    Source = getattr(source_module, "ActionConditionedEntityRolloutGRUMemory")
    torch.manual_seed(123)
    src = Source(latent_dim=4, memory_dim=5, n_actions=3, hidden_dim=7, residual=True)
    compat = CompatActionMemory(latent_dim=4, memory_dim=5, n_actions=3, hidden_dim=7, residual=True)
    _copy_state(src, compat)
    _assert_update_sequence_close(src, compat, uses_action=True)

    prev = torch.randn(1, 2, 5)
    z = torch.randn(1, 2, 4)
    action = torch.zeros(1, 1, 3)
    action[..., 0] = 1.0
    mask = torch.tensor([[1, 0]], dtype=torch.float32)
    out = compat.update(z, prev, mask, action=action, action_mask=torch.ones(1, 1))
    torch.testing.assert_close(out[:, 1], prev[:, 1])
