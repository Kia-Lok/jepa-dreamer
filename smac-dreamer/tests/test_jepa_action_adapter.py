import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smacdreamer.jepa.action_adapter import JEPAActionAdapter


def test_flat_action_to_jepa_masks_dead_agent_to_noop():
    adapter = JEPAActionAdapter(max_agents=2, max_actions=3, checkpoint_n_actions=3)
    flat = torch.tensor([[0, 1, 0, 0, 0, 1]], dtype=torch.float32)
    action, mask = adapter.flat_to_jepa(flat, torch.tensor([[1, 0]], dtype=torch.float32))
    assert action.shape == (1, 2, 3)
    assert mask.tolist() == [[1.0, 0.0]]
    assert action[0, 0].tolist() == [0.0, 1.0, 0.0]
    assert action[0, 1].tolist() == [1.0, 0.0, 0.0]


def test_action_width_mismatch_fails():
    with pytest.raises(ValueError, match="action width mismatch"):
        JEPAActionAdapter(max_agents=2, max_actions=4, checkpoint_n_actions=3)
