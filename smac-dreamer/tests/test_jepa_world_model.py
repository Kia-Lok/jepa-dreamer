import pathlib
import sys

import pytest
import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smacdreamer.jepa.checkpoint import JEPACheckpointInfo
from smacdreamer.jepa.memory import ActionConditionedEntityRolloutGRUMemory, EntityRolloutGRUMemory
from smacdreamer.jepa.state import pack_state, unpack_state
from smacdreamer.jepa.world_model import FrozenJEPAWorldModel


class TinyEncoder(nn.Module):
    def __init__(self, token_dim=5, latent_dim=6):
        super().__init__()
        self.proj = nn.Linear(token_dim, latent_dim)

    def forward(self, entity, mask):
        return self.proj(entity) * mask.unsqueeze(-1)


class TinyPredictor(nn.Module):
    def __init__(self, latent_dim=6, n_actions=3):
        super().__init__()
        self.proj = nn.Linear(latent_dim + n_actions, latent_dim)

    def forward(self, conditioned, action, action_mask, timestep_mask, entity_mask, static):
        ctx = action.mean(dim=2).unsqueeze(2).expand(-1, -1, conditioned.shape[2], -1)
        return self.proj(torch.cat([conditioned, ctx], dim=-1))


class TinyJEPA(nn.Module):
    def __init__(self, token_dim=5, latent_dim=6, n_actions=3):
        super().__init__()
        self.encoder = TinyEncoder(token_dim, latent_dim)
        self.predictor = TinyPredictor(latent_dim, n_actions)
        self.presence = nn.Linear(latent_dim, 1)

    def predict_presence(self, latents):
        return self.presence(latents).squeeze(-1)


def _info(action_conditioned=False, max_agents=2, max_enemies=1):
    meta = {
        "state_dim": 8,
        "n_agents": 2,
        "n_enemies": 1,
        "n_actions": 3,
        "ally_state_feat_size": 3,
        "enemy_state_feat_size": 2,
        "ally_has_shields": False,
        "enemy_has_shields": False,
        "num_unit_types": 0,
        "max_agents": max_agents,
        "max_enemies": max_enemies,
        "max_actions": 3,
        "token_dim": 5,
        "dynamic_token_dim": 3,
        "static_dim": 4,
        "entity_static_feat_size": 2,
        "mode": "entity",
        "latent_dim": 6,
        "memory_dim": 7,
        "action_conditioned_memory": action_conditioned,
        "enemy_visibility_mask": False,
        "enemy_sight_range": 9.0,
        "visibility_xy_indices": (2, 3),
        "latent_normalization": "none",
    }
    return JEPACheckpointInfo("synthetic", "0" * 64, meta, {}, "synthetic", action_conditioned, 6, 7, 3)


def _model(action_conditioned=False, max_agents=2, max_enemies=1):
    memory_cls = ActionConditionedEntityRolloutGRUMemory if action_conditioned else EntityRolloutGRUMemory
    kwargs = {"latent_dim": 6, "memory_dim": 7}
    if action_conditioned:
        kwargs["n_actions"] = 3
    return FrozenJEPAWorldModel(
        core=TinyJEPA(n_actions=3),
        memory_module=memory_cls(**kwargs),
        info=_info(action_conditioned, max_agents=max_agents, max_enemies=max_enemies),
        feature_dim=16,
    )


def _obs(batch=2, time=None, entities=3, token_dim=5):
    shape = (batch, entities, token_dim) if time is None else (batch, time, entities, token_dim)
    prefix = (batch,) if time is None else (batch, time)
    return {
        "jepa_entity": torch.randn(*shape),
        "jepa_entity_mask": torch.ones(*prefix, entities),
        "jepa_entity_slot_mask": torch.ones(*prefix, entities),
        "jepa_static_condition": torch.randn(*prefix, 4),
    }


def test_initial_observe_and_imagine_shapes():
    wm = _model()
    z0, d0 = wm.initial(2)
    assert z0.shape == (2, 3, 6)
    assert d0.shape == (2, wm.state_spec.deter_dim)
    enc = wm.encode_obs(_obs(batch=2, time=4))
    actions = torch.zeros(2, 4, 6)
    actions[..., 0] = 1
    z, d = wm.observe(enc, actions, (z0, d0), torch.zeros(2, 4, dtype=torch.bool))
    assert z.shape == (2, 4, 3, 6)
    assert d.shape == (2, 4, wm.state_spec.deter_dim)
    zi, di = wm.imagine_with_action(z[:, 0], d[:, 0], actions[:, :2])
    assert zi.shape == (2, 2, 3, 6)
    assert di.shape == (2, 2, wm.state_spec.deter_dim)
    feat = wm.get_feat(z[:, 0], d[:, 0])
    assert feat.shape == (2, 16)


def test_repeated_obs_step_equals_observe():
    wm = _model()
    z0, d0 = wm.initial(1)
    enc = wm.encode_obs(_obs(batch=1, time=3))
    actions = torch.zeros(1, 3, 6)
    actions[..., 0] = 1
    resets = torch.zeros(1, 3, dtype=torch.bool)
    z_seq, d_seq = wm.observe(enc, actions, (z0, d0), resets)
    z, d = z0, d0
    zs, ds = [], []
    for t in range(3):
        z, d = wm.obs_step(z, d, actions[:, t], {k: v[:, t] for k, v in enc.items()}, resets[:, t])
        zs.append(z)
        ds.append(d)
    torch.testing.assert_close(z_seq, torch.stack(zs, 1))
    torch.testing.assert_close(d_seq, torch.stack(ds, 1))


def test_observe_requires_shifted_previous_actions_for_all_states():
    wm = _model()
    z0, d0 = wm.initial(1)
    enc = wm.encode_obs(_obs(batch=1, time=3))
    transition_actions = torch.zeros(1, 2, 6)
    transition_actions[..., 0] = 1
    with pytest.raises(ValueError, match="one previous action per observation"):
        wm.observe(enc, transition_actions, (z0, d0), torch.zeros(1, 3, dtype=torch.bool))
    previous_actions = torch.cat([torch.zeros(1, 1, 6), transition_actions], dim=1)
    resets = torch.tensor([[True, False, True]], dtype=torch.bool)
    z_seq, d_seq = wm.observe(enc, previous_actions, (z0, d0), resets)
    z, d = z0, d0
    zs, ds = [], []
    for t in range(3):
        z, d = wm.obs_step(z, d, previous_actions[:, t], {k: v[:, t] for k, v in enc.items()}, resets[:, t])
        zs.append(z)
        ds.append(d)
    torch.testing.assert_close(z_seq, torch.stack(zs, 1))
    torch.testing.assert_close(d_seq, torch.stack(ds, 1))


def test_repeated_img_step_equals_imagine_with_action():
    wm = _model(action_conditioned=True)
    z0, d0 = wm.initial(1)
    enc = wm.encode_obs(_obs(batch=1))
    actions = torch.zeros(1, 4, 6)
    actions[..., 0] = 1
    z, d = wm.obs_step(z0, d0, actions[:, 0], enc, torch.zeros(1, dtype=torch.bool))
    z_seq, d_seq = wm.imagine_with_action(z, d, actions[:, 1:])
    zs, ds = [], []
    for t in range(1, 4):
        z, d = wm.img_step(z, d, actions[:, t])
        zs.append(z)
        ds.append(d)
    torch.testing.assert_close(z_seq, torch.stack(zs, 1))
    torch.testing.assert_close(d_seq, torch.stack(ds, 1))


def test_feature_adapter_gets_gradients_and_frozen_core_stays_bitwise_unchanged():
    torch.manual_seed(0)
    wm = _model(action_conditioned=True)
    frozen_before = [p.detach().clone() for p in wm.parameters_frozen()]
    adapter_before = [p.detach().clone() for p in wm.feature_adapter.parameters()]
    opt = torch.optim.Adam(wm.feature_adapter.parameters(), lr=1e-2)
    z0, d0 = wm.initial(2)
    enc = wm.encode_obs(_obs(batch=2))
    actions = torch.zeros(2, 6)
    actions[:, 0] = 1
    z, d = wm.obs_step(z0, d0, actions, enc, torch.zeros(2, dtype=torch.bool))
    feature = wm.get_feat(z, d)
    assert feature.requires_grad
    loss = feature.square().mean()
    loss.backward()
    assert any(
        p.grad is not None
        and torch.isfinite(p.grad).all()
        and p.grad.abs().sum() > 0
        for p in wm.feature_adapter.parameters()
    )
    assert all(p.grad is None for p in wm.parameters_frozen())
    opt.step()
    assert any(not torch.equal(before, after) for before, after in zip(adapter_before, wm.feature_adapter.parameters()))
    assert all(torch.equal(before, after) for before, after in zip(frozen_before, wm.parameters_frozen()))


def test_core_remains_eval_after_train():
    wm = _model()
    wm.train()
    assert not wm.core.training
    assert not wm.memory_module.training
    assert sum(p.numel() for p in wm.parameters_frozen() if p.requires_grad) == 0


def test_presence_prediction_cannot_reactivate_padded_slots():
    wm = _model(max_agents=10, max_enemies=10)
    with torch.no_grad():
        wm.core.presence.weight.zero_()
        wm.core.presence.bias.fill_(10.0)
    batch = 1
    z = torch.randn(batch, 20, 6)
    memory = torch.zeros(batch, 20, 7)
    entity_mask = torch.zeros(batch, 20)
    slot_mask = torch.zeros(batch, 20)
    slot_mask[:, :3] = 1
    slot_mask[:, 10:14] = 1
    static = torch.zeros(batch, 4)
    deter = pack_state(memory, entity_mask, slot_mask, static)
    action = torch.zeros(batch, 30)
    action[:, 0] = 1
    _, next_deter = wm.img_step(z, deter, action)
    _, next_mask, _, _ = unpack_state(next_deter, wm.state_spec)
    assert next_mask.sum().item() == 7
    assert next_mask[:, 3:10].sum().item() == 0
    assert next_mask[:, 14:].sum().item() == 0
