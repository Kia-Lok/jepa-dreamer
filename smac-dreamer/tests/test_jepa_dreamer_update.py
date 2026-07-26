import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "external" / "r2dreamer", ROOT / "tests"):
    sys.path.insert(0, str(p))

from test_jepa_dreamer import _has_nonzero_grad, _make_agent, _synthetic_batch


class TinyReplay:
    def __init__(self, agent):
        self.data = _synthetic_batch()
        initial_td = agent.get_initial_state(self.data.shape[0])
        self.initial = (initial_td["stoch"], initial_td["deter"])
        self.updated = None

    def sample(self):
        return self.data, torch.arange(self.data.shape[0]), self.initial

    def update(self, index, stoch, deter):
        assert stoch.shape[:2] == self.data.shape
        assert deter.shape[:2] == self.data.shape
        assert torch.isfinite(stoch).all()
        assert torch.isfinite(deter).all()
        self.updated = (index, stoch.detach().clone(), deter.detach().clone())


def _clone_params(module):
    return [p.detach().clone() for p in module.parameters()]


def _changed(before, module):
    return any(not torch.equal(a, b) for a, b in zip(before, module.parameters()))


def _unchanged(before, params):
    return all(torch.equal(a, b) for a, b in zip(before, params))


def test_jepa_dreamer_update_trains_adapter_and_downstream_only(monkeypatch, tmp_path):
    agent = _make_agent(monkeypatch, tmp_path)
    with torch.no_grad():
        agent._frozen_avail_head.last.weight.zero_()
        agent._frozen_avail_head.last.bias.fill_(10.0)
        agent._frozen_alive_head.last.weight.zero_()
        agent._frozen_alive_head.last.bias.fill_(10.0)
    replay = TinyReplay(agent)
    frozen_before = [p.detach().clone() for p in agent.jepa_world_model.parameters_frozen()]
    adapter_before = _clone_params(agent.jepa_world_model.feature_adapter)
    reward_before = _clone_params(agent.reward)
    cont_before = _clone_params(agent.cont)
    avail_before = _clone_params(agent.avail_head)
    alive_before = _clone_params(agent.alive_head)
    actor_before = _clone_params(agent.actor)
    value_before = _clone_params(agent.value)

    # Keep gradients visible after the real update path for this test only.
    agent._optimizer.zero_grad = lambda *args, **kwargs: None
    metrics = agent.update(replay)

    assert replay.updated is not None
    assert all(torch.isfinite(v).all() for v in metrics.values() if torch.is_tensor(v))
    forbidden = ("loss/dyn", "loss/rep", "loss/barlow", "loss/kl", "prior_ent", "post_ent")
    assert not any(key in metrics for key in forbidden)
    assert _has_nonzero_grad(agent.jepa_world_model.feature_adapter)
    assert _has_nonzero_grad(agent.reward)
    assert _has_nonzero_grad(agent.cont)
    assert _has_nonzero_grad(agent.avail_head)
    assert _has_nonzero_grad(agent.alive_head)
    assert _has_nonzero_grad(agent.actor)
    assert _has_nonzero_grad(agent.value)
    assert all(p.grad is None for p in agent.jepa_world_model.parameters_frozen())

    assert _changed(adapter_before, agent.jepa_world_model.feature_adapter)
    assert _changed(reward_before, agent.reward)
    assert _changed(cont_before, agent.cont)
    assert _changed(avail_before, agent.avail_head)
    assert _changed(alive_before, agent.alive_head)
    assert _changed(actor_before, agent.actor)
    assert _changed(value_before, agent.value)
    assert _unchanged(frozen_before, list(agent.jepa_world_model.parameters_frozen()))


def test_jepa_cal_grad_uses_replay_provided_previous_actions_without_second_shift(monkeypatch, tmp_path):
    agent = _make_agent(monkeypatch, tmp_path)
    data = _synthetic_batch()
    initial_td = agent.get_initial_state(data.shape[0])
    initial = (initial_td["stoch"], initial_td["deter"])
    captured = {}
    original_observe = agent.jepa_world_model.observe

    def wrapped_observe(encoded_sequence, action_sequence, initial_state, reset_sequence):
        captured["actions"] = action_sequence.detach().clone()
        return original_observe(encoded_sequence, action_sequence, initial_state, reset_sequence)

    monkeypatch.setattr(agent.jepa_world_model, "observe", wrapped_observe)
    agent._cal_grad(data, initial)
    torch.testing.assert_close(captured["actions"], data["action"])
    assert captured["actions"][:, 0].sum() > 0
