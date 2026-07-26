from __future__ import annotations

import torch
from torch import nn


class JEPAFeatureAdapter(nn.Module):
    """Convert frozen JEPA per-entity state into one R2-Dreamer feature.

    If max_agents is provided, preserve ordered ally slots explicitly, then append enemy and
    global summaries. This avoids collapsing all agent/enemy slots with a single masked mean.

    The output remains one global R2-Dreamer feature vector, so the existing actor/value/reward
    heads stay API-compatible.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        memory_dim: int,
        static_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        max_agents: int | None = None,
        slot_dim: int | None = None,
    ):
        super().__init__()
        hidden = int(hidden_dim or max(out_dim, latent_dim + memory_dim))
        self.max_agents = None if max_agents is None else int(max_agents)
        if self.max_agents is not None and self.max_agents <= 0:
            raise ValueError(f"max_agents must be positive, got {self.max_agents}")

        self.entity_mlp = nn.Sequential(
            nn.Linear(latent_dim + memory_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        if self.max_agents is None:
            # Legacy fallback only. The R2 JEPA world model should pass max_agents.
            self.slot_mlp = None
            self.slot_dim = None
            self.proj = nn.Sequential(
                nn.Linear(hidden + static_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )
            return

        self.slot_dim = int(slot_dim or min(256, hidden))
        self.slot_mlp = nn.Sequential(
            nn.Linear(hidden, self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, self.slot_dim),
            nn.SiLU(),
        )
        # ally slots + enemy summary + global summary + static condition
        in_dim = self.slot_dim * (self.max_agents + 2) + static_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=-2).clamp_min(1.0)
        return (x * mask).sum(dim=-2) / denom

    def forward(
        self,
        conditioned_z: torch.Tensor,
        memory: torch.Tensor,
        entity_mask: torch.Tensor,
        static_condition: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([conditioned_z, memory], dim=-1)
        x = self.entity_mlp(x)
        mask = entity_mask.to(dtype=x.dtype).unsqueeze(-1)
        static = static_condition.to(dtype=x.dtype)

        if self.max_agents is None:
            pooled = self._masked_mean(x, mask)
            feat = self.proj(torch.cat([pooled, static], dim=-1))
            if not torch.isfinite(feat).all():
                raise FloatingPointError("non-finite JEPA feature adapter output")
            return feat

        slot = self.slot_mlp(x)

        # First max_agents entity slots are ally slots in this backend.
        allies = slot[:, : self.max_agents]
        ally_mask = mask[:, : self.max_agents]
        if allies.shape[-2] < self.max_agents:
            pad_n = self.max_agents - allies.shape[-2]
            allies = torch.cat(
                [
                    allies,
                    torch.zeros(
                        *allies.shape[:-2],
                        pad_n,
                        allies.shape[-1],
                        device=allies.device,
                        dtype=allies.dtype,
                    ),
                ],
                dim=-2,
            )
            ally_mask = torch.cat(
                [
                    ally_mask,
                    torch.zeros(
                        *ally_mask.shape[:-2],
                        pad_n,
                        ally_mask.shape[-1],
                        device=ally_mask.device,
                        dtype=ally_mask.dtype,
                    ),
                ],
                dim=-2,
            )
        allies_flat = (allies * ally_mask).flatten(start_dim=-2)

        enemies = slot[:, self.max_agents :]
        enemy_mask = mask[:, self.max_agents :]
        if enemies.shape[-2] == 0:
            enemy_summary = torch.zeros_like(slot[:, 0])
        else:
            enemy_summary = self._masked_mean(enemies, enemy_mask)

        global_summary = self._masked_mean(slot, mask)
        feat_in = torch.cat([allies_flat, enemy_summary, global_summary, static], dim=-1)
        feat = self.proj(feat_in)
        if not torch.isfinite(feat).all():
            raise FloatingPointError("non-finite JEPA feature adapter output")
        return feat
