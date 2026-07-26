from __future__ import annotations

import torch
from torch import nn


def pooled_action_context(
    action: torch.Tensor,
    action_mask: torch.Tensor | None,
    *,
    n_actions: int,
) -> torch.Tensor:
    """Pool per-agent one-hot actions into the JEPA training representation.

    The restored JEPA training script uses a mean pooled action context over live
    allied slots. Shape is ``[B, A, C]`` plus optional ``[B, A]`` mask.
    """
    if action.ndim != 3:
        raise ValueError(f"action must have shape [B, A, C], got {tuple(action.shape)}")
    if action.shape[-1] != int(n_actions):
        raise ValueError(f"action width {action.shape[-1]} != checkpoint n_actions {n_actions}")
    if action_mask is None:
        return action.mean(dim=1)
    mask = action_mask.to(dtype=action.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (action * mask).sum(dim=1) / denom


class ActionConditionedEntityRolloutGRUMemory(nn.Module):
    """Runtime-compatible copy of the JEPA training-script memory module.

    The intended checkpoint defines this class inside
    ``train_markov_rollout_rnn_visibility_seqmem_experiments.py``. Importing a
    training entry point at R2 runtime is intentionally avoided, so this class
    preserves constructor names, tensor semantics and state-dict key layout.
    """

    uses_action = True

    def __init__(
        self,
        *,
        latent_dim: int,
        memory_dim: int,
        n_actions: int,
        hidden_dim: int | None = None,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.memory_dim = int(memory_dim)
        self.n_actions = int(n_actions)
        self.residual = bool(residual)
        hidden = int(hidden_dim or max(latent_dim, memory_dim))
        self.action_proj = nn.Sequential(
            nn.Linear(self.n_actions, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.gru = nn.GRUCell(latent_dim + memory_dim, memory_dim)
        self.condition_net = nn.Sequential(
            nn.Linear(latent_dim + memory_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def initial_memory(self, batch_size: int, entities: int, *, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_size, entities, self.memory_dim, device=device, dtype=dtype)

    def condition(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        correction = self.condition_net(torch.cat([z, memory], dim=-1))
        out = z + correction if self.residual else correction
        if entity_mask is not None:
            out = out * entity_mask.unsqueeze(-1)
        return out

    def update(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
        *,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, entities, _ = z.shape
        if action is None:
            action_ctx = torch.zeros(bsz, self.n_actions, device=z.device, dtype=z.dtype)
        else:
            action_ctx = pooled_action_context(action, action_mask, n_actions=self.n_actions).to(dtype=z.dtype)
        action_emb = self.action_proj(action_ctx).unsqueeze(1).expand(-1, entities, -1)
        gru_in = torch.cat([z, action_emb], dim=-1)
        new_mem = self.gru(
            gru_in.reshape(bsz * entities, -1),
            memory.reshape(bsz * entities, self.memory_dim),
        ).reshape(bsz, entities, self.memory_dim)
        if entity_mask is not None:
            keep = entity_mask.unsqueeze(-1).bool()
            new_mem = torch.where(keep, new_mem, memory)
        return new_mem


class EntityRolloutGRUMemory(nn.Module):
    """Runtime-compatible copy of ``smac_jepa.modules.rollout_memory``."""

    uses_action = False

    def __init__(
        self,
        latent_dim: int,
        memory_dim: int = 128,
        hidden_dim: int | None = None,
        residual: bool = True,
    ):
        super().__init__()
        if memory_dim < 1:
            raise ValueError("memory_dim must be >= 1")
        self.latent_dim = int(latent_dim)
        self.memory_dim = int(memory_dim)
        self.hidden_dim = int(hidden_dim or max(latent_dim, memory_dim))
        self.residual = bool(residual)
        self.gru = nn.GRUCell(self.latent_dim, self.memory_dim)
        self.fuse = nn.Sequential(
            nn.Linear(self.latent_dim + self.memory_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.norm = nn.LayerNorm(self.latent_dim)

    def initial_memory(self, batch_entities: int, entity_slots: int, *, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_entities, entity_slots, self.memory_dim, device=device, dtype=dtype)

    def condition(
        self,
        z: torch.Tensor,
        memory: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        correction = self.fuse(torch.cat([z, memory], dim=-1))
        out = self.norm(z + correction) if self.residual else self.norm(correction)
        if entity_mask is not None:
            out = out * entity_mask.unsqueeze(-1)
        return out

    def update(
        self,
        z_next: torch.Tensor,
        memory: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n, e, d = z_next.shape
        new_mem = self.gru(
            z_next.reshape(n * e, d),
            memory.reshape(n * e, self.memory_dim),
        ).reshape(n, e, self.memory_dim)
        if entity_mask is not None:
            new_mem = new_mem * entity_mask.unsqueeze(-1)
        return new_mem
