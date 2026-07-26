import pathlib
import sys

import pytest
import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import smacdreamer.jepa.checkpoint as checkpoint_mod
from smacdreamer.jepa.checkpoint import JEPACompatibilityError, load_frozen_jepa_checkpoint
from smacdreamer.jepa.memory import EntityRolloutGRUMemory


class TinySMACJEPA(nn.Module):
    def __init__(
        self,
        *,
        state_dim,
        n_agents,
        n_actions,
        latent_dim,
        hidden_dim,
        action_dim,
        num_heads,
        mode,
        max_agents,
        max_enemies,
        max_actions,
        token_dim,
        decoder_weight=1.0,
        encoder_layers=1,
        action_layers=1,
        predictor_layers=1,
        max_context_len=32,
        static_dim=0,
    ):
        super().__init__()
        self.encoder = nn.Linear(token_dim, latent_dim)
        self.predictor = nn.Linear(latent_dim + n_actions, latent_dim)
        self.presence = nn.Linear(latent_dim, 1)

    def predict_presence(self, latents):
        return self.presence(latents).squeeze(-1)


def _meta():
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


def _cfg():
    return {
        "latent_dim": 6,
        "hidden_dim": 8,
        "action_dim": 4,
        "num_heads": 2,
        "rollout_memory_dim": 7,
        "training_regime": "synthetic",
        "enemy_visibility_mask": False,
        "enemy_sight_range": 9.0,
        "xy_indices": (2, 3),
        "latent_normalization": "none",
    }


def _make_checkpoint(path):
    meta = _meta()
    cfg = _cfg()
    model = TinySMACJEPA(
        state_dim=meta["state_dim"],
        n_agents=meta["n_agents"],
        n_actions=meta["n_actions"],
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
        action_dim=cfg["action_dim"],
        num_heads=cfg["num_heads"],
        mode="entity",
        max_agents=meta["max_agents"],
        max_enemies=meta["max_enemies"],
        max_actions=meta["max_actions"],
        token_dim=meta["token_dim"],
        static_dim=meta["static_dim"],
    )
    memory = EntityRolloutGRUMemory(latent_dim=6, memory_dim=7, hidden_dim=None, residual=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "memory_module_state": memory.state_dict(),
            "metadata": meta,
            "resolved_config": cfg,
        },
        path,
    )
    return meta


@pytest.fixture(autouse=True)
def _fake_jepa_import(monkeypatch):
    monkeypatch.setattr(checkpoint_mod, "_import_jepa", lambda: (TinySMACJEPA, EntityRolloutGRUMemory))


def test_synthetic_checkpoint_loads_and_freezes(tmp_path):
    path = tmp_path / "jepa.pt"
    meta = _make_checkpoint(path)
    model, memory, info = load_frozen_jepa_checkpoint(path, map_location="cpu", live_metadata=meta)
    assert info.sha256
    assert all(not p.requires_grad for p in model.parameters())
    assert all(not p.requires_grad for p in memory.parameters())


@pytest.mark.parametrize("field,value", [
    ("max_actions", 9),
    ("n_actions", 9),
    ("enemy_visibility_mask", True),
    ("enemy_sight_range", 3.0),
    ("latent_dim", 12),
    ("memory_dim", 11),
])
def test_checkpoint_metadata_mismatch_fails(tmp_path, field, value):
    path = tmp_path / "jepa.pt"
    meta = _make_checkpoint(path)
    live = dict(meta)
    live[field] = value
    with pytest.raises(JEPACompatibilityError, match=field):
        load_frozen_jepa_checkpoint(path, map_location="cpu", live_metadata=live)


def test_checkpoint_missing_live_field_fails(tmp_path):
    path = tmp_path / "jepa.pt"
    meta = _make_checkpoint(path)
    live = dict(meta)
    live.pop("enemy_visibility_mask")
    with pytest.raises(JEPACompatibilityError, match="enemy_visibility_mask"):
        load_frozen_jepa_checkpoint(path, map_location="cpu", live_metadata=live)
