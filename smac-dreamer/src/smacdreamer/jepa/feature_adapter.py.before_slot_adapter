from __future__ import annotations

import torch
from torch import nn


class JEPAFeatureAdapter(nn.Module):
    """Pool per-entity JEPA state into one R2-Dreamer global feature."""

    def __init__(
        self,
        *,
        latent_dim: int,
        memory_dim: int,
        static_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        hidden = int(hidden_dim or max(out_dim, latent_dim + memory_dim))
        self.entity_mlp = nn.Sequential(
            nn.Linear(latent_dim + memory_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden + static_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

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
        pooled = (x * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1.0)
        feat = self.proj(torch.cat([pooled, static_condition.to(dtype=x.dtype)], dim=-1))
        if not torch.isfinite(feat).all():
            raise FloatingPointError("non-finite JEPA feature adapter output")
        return feat
