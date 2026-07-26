import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))

from validate_jepa_r2_integration import previous_actions_for_states, rollout_actions_from_start
from validate_jepa_token_parity import _first_mismatch


def test_raw_transition_actions_get_initial_zero_for_three_states():
    actions = torch.zeros(2, 2, 3)
    actions[0, :, 1] = 1.0
    actions[1, :, 2] = 1.0
    prev = previous_actions_for_states(actions)
    assert prev.shape == (3, 6)
    assert prev[0].sum().item() == 0.0
    torch.testing.assert_close(prev[1], actions[0].reshape(-1))
    torch.testing.assert_close(prev[2], actions[1].reshape(-1))


def test_deliberate_off_by_one_previous_action_shift_is_detected():
    actions = torch.zeros(2, 1, 3)
    actions[0, 0, 1] = 1.0
    actions[1, 0, 2] = 1.0
    correct = previous_actions_for_states(actions)
    off_by_one = torch.cat([actions.reshape(2, -1), torch.zeros(1, 3)], dim=0)
    with pytest.raises(AssertionError, match="previous_action_shift"):
        _first_mismatch("previous_action_shift", off_by_one, correct)


def test_imagined_rollout_from_state_i_starts_with_action_i():
    transition_actions = torch.arange(1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3)
    sliced = rollout_actions_from_start(transition_actions, start_idx=2, rollout_horizon=10)
    torch.testing.assert_close(sliced[:, 0], transition_actions[:, 2])
    torch.testing.assert_close(sliced[:, 1], transition_actions[:, 3])
    assert sliced.shape[1] == 2


def test_rollout_action_slice_rejects_invalid_start():
    transition_actions = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="outside transition action range"):
        rollout_actions_from_start(transition_actions, start_idx=2, rollout_horizon=1)
