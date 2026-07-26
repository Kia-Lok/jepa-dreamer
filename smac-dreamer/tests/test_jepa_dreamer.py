import argparse
import pathlib
import sys

import torch
from gymnasium import spaces
from tensordict import TensorDict
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "external" / "r2dreamer", ROOT / "scripts"):
    sys.path.insert(0, str(p))

from dreamer import Dreamer
from smacdreamer.jepa.checkpoint import JEPACheckpointInfo
from smacdreamer.jepa.memory import EntityRolloutGRUMemory
from train_r2dreamer_smaclite_debug import make_config


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(5, 6)

            def forward(self, entity, mask):
                return self.proj(entity) * mask.unsqueeze(-1)

        self.encoder = Encoder()
        class Predictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(9, 6)

            def forward(self, conditioned, action, action_mask, timestep_mask, entity_mask, static):
                ctx = action.mean(dim=2).unsqueeze(2).expand(-1, -1, conditioned.shape[2], -1)
                return self.proj(torch.cat([conditioned, ctx], dim=-1))

        self.predictor = Predictor()
        self.presence = nn.Linear(6, 1)

    def predict_presence(self, latents):
        return self.presence(latents).squeeze(-1)


def _metadata():
    return {
        "state_dim": 8,
        "n_agents": 2,
        "n_enemies": 1,
        "n_actions": 3,
        "ally_state_feat_size": 3,
        "enemy_state_feat_size": 2,
        "ally_has_shields": False,
        "enemy_has_shields": False,
        "num_unit_types": 0,
        "max_agents": 2,
        "max_enemies": 1,
        "max_actions": 3,
        "token_dim": 5,
        "dynamic_token_dim": 3,
        "static_dim": 4,
        "entity_static_feat_size": 2,
        "mode": "entity",
        "latent_dim": 6,
        "memory_dim": 7,
        "action_conditioned_memory": False,
        "enemy_visibility_mask": False,
        "enemy_sight_range": 9.0,
        "visibility_xy_indices": (2, 3),
        "latent_normalization": "none",
    }


def test_jepa_dreamer_constructs_and_keeps_core_eval(monkeypatch, tmp_path):
    import smacdreamer.jepa.checkpoint as checkpoint_mod

    meta = _metadata()

    def fake_loader(*args, **kwargs):
        return (
            TinyCore(),
            EntityRolloutGRUMemory(latent_dim=6, memory_dim=7),
            JEPACheckpointInfo("synthetic", "0" * 64, meta, {}, "synthetic", False, 6, 7, 3),
        )

    monkeypatch.setattr(checkpoint_mod, "load_frozen_jepa_checkpoint", fake_loader)
    cfg = make_config(argparse.Namespace(steps=10, batch_size=1, batch_length=2, units=16, deter=32, imag_horizon=2))
    cfg.model.action_masking = True
    cfg.model.world_model = {
        "backend": "jepa",
        "jepa": {
            "checkpoint": str(tmp_path / "unused.pt"),
            "strict_checkpoint": True,
            "freeze_core": True,
            "presence_threshold": 0.5,
            "feature_dim": 64,
            "live_metadata": meta,
        },
    }
    obs_space = spaces.Dict({
        "jepa_entity": spaces.Box(-10, 10, shape=(3, 5), dtype=float),
        "jepa_entity_mask": spaces.Box(0, 1, shape=(3,), dtype=float),
        "jepa_entity_slot_mask": spaces.Box(0, 1, shape=(3,), dtype=float),
        "jepa_static_condition": spaces.Box(-10, 10, shape=(4,), dtype=float),
        "avail_actions": spaces.Box(0, 1, shape=(6,), dtype=float),
        "agent_slot_mask": spaces.Box(0, 1, shape=(2,), dtype=float),
        "agent_alive_mask": spaces.Box(0, 1, shape=(2,), dtype=float),
        "is_first": spaces.Box(0, 1, shape=(), dtype=bool),
        "is_last": spaces.Box(0, 1, shape=(), dtype=bool),
        "is_terminal": spaces.Box(0, 1, shape=(), dtype=bool),
    })
    act_space = spaces.Box(0.0, 1.0, shape=(3, 3), dtype=float)
    act_space.multi_discrete = True
    agent = Dreamer(cfg.model, obs_space, act_space)
    assert agent.world_model_backend == "jepa"
    agent.train()
    assert not agent.jepa_world_model.core.training
    assert sum(p.numel() for p in agent.jepa_world_model.parameters_frozen() if p.requires_grad) == 0


def _make_agent(monkeypatch, tmp_path):
    import smacdreamer.jepa.checkpoint as checkpoint_mod

    meta = _metadata()

    def fake_loader(*args, **kwargs):
        return (
            TinyCore(),
            EntityRolloutGRUMemory(latent_dim=6, memory_dim=7),
            JEPACheckpointInfo("synthetic", "0" * 64, meta, {}, "synthetic", False, 6, 7, 3),
        )

    monkeypatch.setattr(checkpoint_mod, "load_frozen_jepa_checkpoint", fake_loader)
    cfg = make_config(argparse.Namespace(steps=10, batch_size=2, batch_length=4, units=16, deter=32, imag_horizon=2))
    cfg.model.action_masking = True
    cfg.model.amp_dtype = "float32"
    cfg.model.act_entropy = 0.01
    cfg.model.world_model = {
        "backend": "jepa",
        "jepa": {
            "checkpoint": str(tmp_path / "unused.pt"),
            "strict_checkpoint": True,
            "freeze_core": True,
            "presence_threshold": 0.5,
            "feature_dim": 64,
            "live_metadata": meta,
        },
    }
    obs_space = spaces.Dict({
        "jepa_entity": spaces.Box(-10, 10, shape=(3, 5), dtype=float),
        "jepa_entity_mask": spaces.Box(0, 1, shape=(3,), dtype=float),
        "jepa_entity_slot_mask": spaces.Box(0, 1, shape=(3,), dtype=float),
        "jepa_static_condition": spaces.Box(-10, 10, shape=(4,), dtype=float),
        "avail_actions": spaces.Box(0, 1, shape=(6,), dtype=float),
        "agent_slot_mask": spaces.Box(0, 1, shape=(2,), dtype=float),
        "agent_alive_mask": spaces.Box(0, 1, shape=(2,), dtype=float),
        "is_first": spaces.Box(0, 1, shape=(), dtype=bool),
        "is_last": spaces.Box(0, 1, shape=(), dtype=bool),
        "is_terminal": spaces.Box(0, 1, shape=(), dtype=bool),
    })
    act_space = spaces.Box(0.0, 1.0, shape=(3, 3), dtype=float)
    act_space.multi_discrete = True
    return Dreamer(cfg.model, obs_space, act_space)


def _synthetic_batch(batch=2, time=4):
    torch.manual_seed(0)
    action = torch.zeros(batch, time, 6)
    action[..., 0] = 1.0
    action[..., 3] = 1.0
    return TensorDict({
        "jepa_entity": torch.randn(batch, time, 3, 5),
        "jepa_entity_mask": torch.ones(batch, time, 3),
        "jepa_entity_slot_mask": torch.ones(batch, time, 3),
        "jepa_static_condition": torch.randn(batch, time, 4),
        "action": action,
        "reward": torch.randn(batch, time, 1).clamp(-1, 1),
        "is_first": torch.zeros(batch, time, 1, dtype=torch.bool),
        "is_last": torch.zeros(batch, time, 1, dtype=torch.bool),
        "is_terminal": torch.zeros(batch, time, 1, dtype=torch.bool),
        "avail_actions": torch.ones(batch, time, 6),
        "agent_slot_mask": torch.ones(batch, time, 2),
        "agent_alive_mask": torch.ones(batch, time, 2),
    }, batch_size=(batch, time))


def _has_nonzero_grad(module):
    return any(
        p.grad is not None
        and torch.isfinite(p.grad).all()
        and p.grad.detach().abs().sum() > 0
        for p in module.parameters()
    )


def test_jepa_dreamer_training_update_gradients_and_optimizer_contents(monkeypatch, tmp_path):
    agent = _make_agent(monkeypatch, tmp_path)
    data = _synthetic_batch()
    initial_td = agent.get_initial_state(2)
    initial = (initial_td["stoch"], initial_td["deter"])
    frozen_before = [p.detach().clone() for p in agent.jepa_world_model.parameters_frozen()]
    adapter_before = [p.detach().clone() for p in agent.jepa_world_model.feature_adapter.parameters()]
    with torch.no_grad():
        agent._frozen_avail_head.last.weight.zero_()
        agent._frozen_avail_head.last.bias.fill_(10.0)
        agent._frozen_alive_head.last.weight.zero_()
        agent._frozen_alive_head.last.bias.fill_(10.0)

    named_params = list(agent._named_params.items())
    adapter_ids = {id(p) for p in agent.jepa_world_model.feature_adapter.parameters()}
    frozen_ids = {id(p) for p in agent.jepa_world_model.parameters_frozen()}
    assert sum(id(p) in adapter_ids for _, p in named_params) == len(adapter_ids)
    assert all(id(p) not in frozen_ids for _, p in named_params)

    (_, _), metrics = agent._cal_grad(data, initial)
    assert "loss/dyn" not in metrics
    assert "loss/rep" not in metrics
    assert "loss/barlow" not in metrics
    assert _has_nonzero_grad(agent.jepa_world_model.feature_adapter)
    assert _has_nonzero_grad(agent.reward)
    assert _has_nonzero_grad(agent.cont)
    assert _has_nonzero_grad(agent.avail_head)
    assert _has_nonzero_grad(agent.alive_head)
    assert _has_nonzero_grad(agent.actor)
    assert _has_nonzero_grad(agent.value)
    assert all(p.grad is None for p in agent.jepa_world_model.parameters_frozen())

    agent._scaler.unscale_(agent._optimizer)
    agent._agc(agent._named_params.values())
    agent._scaler.step(agent._optimizer)
    agent._scaler.update()
    agent._scheduler.step()
    assert any(
        not torch.equal(before, after)
        for before, after in zip(adapter_before, agent.jepa_world_model.feature_adapter.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(frozen_before, agent.jepa_world_model.parameters_frozen())
    )
    agent._optimizer.zero_grad(set_to_none=True)
