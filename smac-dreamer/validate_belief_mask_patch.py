"""Validate the JEPA belief-mask API patch.

Run from the repository root that contains:
    smac-dreamer/src/smacdreamer/jepa/world_model.py

This is a lightweight synthetic check. It does not need a SMAC env or JEPA
checkpoint. It verifies the exact Exp33 failure mode:

    raw current visibility says hidden enemy mask = 0
    anchored memory says hidden enemy has been seen = 1
    patched world_model builds belief_mask = 1
    condition(..., raw_visibility) would zero the hidden belief
    condition(..., belief_mask) preserves the hidden belief
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import torch
from torch import nn


ROOT = pathlib.Path.cwd()
SRC = ROOT / "smac-dreamer" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smacdreamer.jepa.world_model import FrozenJEPAWorldModel  # noqa: E402
from smacdreamer.jepa.state import pack_state, unpack_state  # noqa: E402


class DummyCore(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.dummy = nn.Parameter(torch.zeros(()))

    def encoder(self, entity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return entity[..., : self.latent_dim] * mask.unsqueeze(-1)

    def predictor(
        self,
        conditioned: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor,
        seq_mask: torch.Tensor,
        entity_mask: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        return conditioned

    def predict_presence(self, pred: torch.Tensor) -> torch.Tensor:
        return torch.full(pred.shape[:-1], 8.0, device=pred.device, dtype=pred.dtype)


class DummyAnchoredMemory(nn.Module):
    uses_action = True
    anchored_belief_memory = True

    def __init__(self, latent_dim: int, recurrent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.recurrent_dim = recurrent_dim
        self.memory_dim = recurrent_dim + latent_dim + 2
        self.dummy = nn.Parameter(torch.zeros(()))

    def _split(self, memory: torch.Tensor):
        r = self.recurrent_dim
        l = self.latent_dim
        recurrent = memory[..., :r]
        anchor = memory[..., r : r + l]
        seen = memory[..., r + l : r + l + 1]
        age = memory[..., r + l + 1 : r + l + 2]
        return recurrent, anchor, seen, age

    def _join(
        self,
        recurrent: torch.Tensor,
        anchor: torch.Tensor,
        seen: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([recurrent, anchor, seen, age], dim=-1)

    def initial_memory(
        self,
        batch_size: int,
        entities: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, entities, self.memory_dim, device=device, dtype=dtype)

    def condition(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        belief_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, anchor, seen, _ = self._split(memory)
        visible = z.detach().abs().amax(dim=-1, keepdim=True) > 1.0e-8
        out = torch.where(visible, z, torch.where(seen > 0.5, anchor, z))
        if belief_gate is not None:
            out = out * belief_gate.clamp(0.0, 1.0).unsqueeze(-1)
        return out

    def update(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        update_gate: torch.Tensor | None = None,
        *,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The validation focuses on condition/exposure semantics, so this can
        # simply return memory.
        return memory


def main() -> None:
    torch.manual_seed(0)

    max_agents = 2
    max_enemies = 1
    entities = max_agents + max_enemies
    latent_dim = 4
    recurrent_dim = 3
    memory_dim = recurrent_dim + latent_dim + 2
    hidden_enemy_idx = 2

    info = SimpleNamespace(
        metadata={
            "max_agents": max_agents,
            "max_enemies": max_enemies,
            "max_actions": 3,
            "n_actions": 3,
            "static_dim": 0,
            "enemy_visibility_mask": True,
            "enemy_sight_range": 9.0,
            "visibility_xy_indices": (2, 3),
        },
        latent_dim=latent_dim,
        memory_dim=memory_dim,
        resolved_config={"anchored_belief_memory": True},
    )

    wm = FrozenJEPAWorldModel(
        core=DummyCore(latent_dim),
        memory_module=DummyAnchoredMemory(latent_dim, recurrent_dim),
        info=info,
        feature_dim=16,
    )

    memory = wm.memory_module.initial_memory(
        1, entities, device=torch.device("cpu"), dtype=torch.float32
    )
    recurrent, anchor, seen, age = wm.memory_module._split(memory)
    anchor[:, hidden_enemy_idx] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    seen[:, hidden_enemy_idx] = 1.0
    age[:, hidden_enemy_idx] = 3.0
    memory = wm.memory_module._join(recurrent, anchor, seen, age)

    z = torch.zeros(1, entities, latent_dim)
    z[:, 0, 0] = 0.5  # visible agent token
    z[:, 1, 0] = 0.7  # visible agent token
    raw_visibility = torch.tensor([[1.0, 1.0, 0.0]])
    slot_mask = torch.tensor([[1.0, 1.0, 1.0]])
    static = torch.zeros(1, 0)

    belief_mask = wm._belief_mask(raw_visibility, slot_mask, memory)
    print("raw_visibility:", raw_visibility.tolist())
    print("seen:", seen.squeeze(-1).tolist())
    print("belief_mask:", belief_mask.tolist())

    assert raw_visibility[0, hidden_enemy_idx].item() == 0.0
    assert seen[0, hidden_enemy_idx].item() == 1.0
    assert belief_mask[0, hidden_enemy_idx].item() == 1.0

    old_conditioned = wm.memory_module.condition(z, memory, raw_visibility)
    new_conditioned = wm.memory_module.condition(z, memory, belief_mask)

    old_norm = old_conditioned[0, hidden_enemy_idx].norm().item()
    new_norm = new_conditioned[0, hidden_enemy_idx].norm().item()
    print("old hidden conditioned norm:", old_norm)
    print("new hidden conditioned norm:", new_norm)

    assert old_norm == 0.0
    assert new_norm > 0.0

    deter = pack_state(memory, raw_visibility, slot_mask, static)
    feat = wm.get_feat(z, deter)
    print("get_feat shape:", tuple(feat.shape))
    assert feat.shape == (1, 16)
    assert torch.isfinite(feat).all()

    _, packed_mask, _, _ = unpack_state(deter, wm.state_spec)
    assert packed_mask[0, hidden_enemy_idx].item() == 0.0
    # get_feat should still recover belief exposure internally from memory.seen.

    print("PASS: hidden-but-seen entity is exposed through belief_mask.")


if __name__ == "__main__":
    main()
