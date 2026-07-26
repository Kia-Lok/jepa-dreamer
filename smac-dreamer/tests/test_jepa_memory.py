import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smacdreamer.jepa.memory import ActionConditionedEntityRolloutGRUMemory


def test_action_conditioned_memory_preserves_masked_entity_history():
    torch.manual_seed(0)
    memory = ActionConditionedEntityRolloutGRUMemory(latent_dim=4, memory_dim=5, n_actions=3)
    z = torch.randn(2, 3, 4)
    prev = torch.randn(2, 3, 5)
    action = torch.zeros(2, 2, 3)
    action[..., 0] = 1.0
    action_mask = torch.ones(2, 2)
    entity_mask = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
    out = memory.update(z, prev, entity_mask, action=action, action_mask=action_mask)
    torch.testing.assert_close(out[entity_mask == 0], prev[entity_mask == 0])
    assert not torch.allclose(out[entity_mask == 1], prev[entity_mask == 1])


def test_action_conditioned_memory_invisible_then_visible_can_update_again():
    torch.manual_seed(1)
    memory = ActionConditionedEntityRolloutGRUMemory(latent_dim=4, memory_dim=5, n_actions=3)
    z = torch.randn(1, 2, 4)
    prev = torch.randn(1, 2, 5)
    action = torch.zeros(1, 1, 3)
    action[..., 1] = 1.0
    hidden = torch.tensor([[1, 0]], dtype=torch.float32)
    out_hidden = memory.update(z, prev, hidden, action=action, action_mask=torch.ones(1, 1))
    torch.testing.assert_close(out_hidden[:, 1], prev[:, 1])
    visible = torch.tensor([[1, 1]], dtype=torch.float32)
    out_visible = memory.update(z, out_hidden, visible, action=action, action_mask=torch.ones(1, 1))
    assert not torch.allclose(out_visible[:, 1], out_hidden[:, 1])
