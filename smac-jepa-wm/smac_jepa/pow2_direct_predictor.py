from __future__ import annotations

"""Power-of-two direct latent trajectory predictor.

The module consumes one real JEPA context latent and a *real/planned* sequence of
joint actions. It never feeds an intermediate predicted latent back while making
one direct block prediction. Separate block calls can be composed with binary
horizon decomposition, e.g. 13 = 8 + 4 + 1.
"""

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def canonical_pow2_horizons(values: Iterable[int]) -> tuple[int, ...]:
    horizons = tuple(sorted({int(v) for v in values}))
    if not horizons:
        raise ValueError("At least one power-of-two horizon is required")
    for horizon in horizons:
        if horizon < 1 or horizon & (horizon - 1):
            raise ValueError(f"Horizon {horizon} is not a positive power of two")
    return horizons


@dataclass(frozen=True)
class BinaryPrediction:
    latent: torch.Tensor
    blocks: tuple[int, ...]


class PowerOfTwoDirectPredictor(nn.Module):
    """Causal LSTM decoder with direct readouts at 1, 2, 4, ... steps.

    Parameters
    ----------
    latent_dim:
        JEPA entity-latent width.
    n_actions:
        Number of discrete action classes.
    max_agents:
        Maximum number of allied action slots in a joint action.
    max_entities:
        Total ally + enemy entity slots.
    horizons:
        Directly supervised power-of-two horizons.

    Notes
    -----
    * The decoder's recurrent input is the real/planned joint-action sequence,
      not a predicted world state.
    * Entity identity is preserved by fixed slot embeddings.
    * Every trained power has its own residual readout head. A shared head is
      also trained at those powers and permits diagnostic exact predictions at
      non-power-of-two horizons.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        n_actions: int,
        max_agents: int,
        max_entities: int,
        horizons: Iterable[int] = (1, 2, 4, 8, 16),
        hidden_dim: int = 384,
        action_embed_dim: int = 48,
        slot_embed_dim: int = 32,
        dropout: float = 0.0,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_actions = int(n_actions)
        self.max_agents = int(max_agents)
        self.max_entities = int(max_entities)
        self.horizons = canonical_pow2_horizons(horizons)
        self.max_horizon = max(self.horizons)
        self.hidden_dim = int(hidden_dim)
        self.action_embed_dim = int(action_embed_dim)
        self.slot_embed_dim = int(slot_embed_dim)
        self.residual_scale = float(residual_scale)

        if self.max_agents < 1 or self.max_entities < 1:
            raise ValueError("max_agents and max_entities must be positive")
        if self.n_actions < 2:
            raise ValueError("n_actions must be at least two")

        self.action_embedding = nn.Embedding(self.n_actions, self.action_embed_dim)
        self.agent_embedding = nn.Embedding(self.max_agents, self.action_embed_dim)
        self.joint_action_proj = nn.Sequential(
            nn.Linear(self.max_agents * self.action_embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.context_pool_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.init_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.init_c = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.step_embedding = nn.Embedding(self.max_horizon + 1, self.hidden_dim)
        self.decoder = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

        self.entity_context_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.slot_embedding = nn.Embedding(self.max_entities, self.slot_embed_dim)
        head_in = self.hidden_dim * 2 + self.slot_embed_dim

        def make_head() -> nn.Module:
            return nn.Sequential(
                nn.Linear(head_in, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.latent_dim),
            )

        self.power_heads = nn.ModuleDict({str(h): make_head() for h in self.horizons})
        self.shared_head = make_head()

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, n_actions={self.n_actions}, "
            f"max_agents={self.max_agents}, max_entities={self.max_entities}, "
            f"horizons={self.horizons}, hidden_dim={self.hidden_dim}"
        )

    def _normalize_action_shape(
        self,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad/truncate the ally action axis to max_agents."""
        if action_mask.ndim != 2:
            raise ValueError(f"Expected action_mask [N,A], got {tuple(action_mask.shape)}")
        if actions.ndim not in (2, 3):
            raise ValueError(
                "Expected discrete actions [N,A] or one-hot/probabilities [N,A,C], "
                f"got {tuple(actions.shape)}"
            )
        n, agents = action_mask.shape
        if actions.shape[0] != n or actions.shape[1] != agents:
            raise ValueError("Action and action-mask leading dimensions disagree")

        keep = min(agents, self.max_agents)
        actions = actions[:, :keep]
        action_mask = action_mask[:, :keep]
        if keep < self.max_agents:
            pad_agents = self.max_agents - keep
            mask_pad = torch.zeros(n, pad_agents, device=action_mask.device, dtype=action_mask.dtype)
            action_mask = torch.cat([action_mask, mask_pad], dim=1)
            if actions.ndim == 2:
                action_pad = torch.zeros(n, pad_agents, device=actions.device, dtype=actions.dtype)
            else:
                action_pad = torch.zeros(
                    n,
                    pad_agents,
                    actions.shape[-1],
                    device=actions.device,
                    dtype=actions.dtype,
                )
            actions = torch.cat([actions, action_pad], dim=1)
        return actions, action_mask

    def encode_joint_action(
        self,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        actions, action_mask = self._normalize_action_shape(actions, action_mask)
        if actions.ndim == 3:
            if actions.shape[-1] != self.n_actions:
                raise ValueError(
                    f"One-hot action width {actions.shape[-1]} != n_actions {self.n_actions}"
                )
            embedded = actions.to(self.action_embedding.weight.dtype) @ self.action_embedding.weight
        else:
            ids = actions.long().clamp_(0, self.n_actions - 1)
            embedded = self.action_embedding(ids)

        agent_ids = torch.arange(self.max_agents, device=embedded.device)
        embedded = embedded + self.agent_embedding(agent_ids).unsqueeze(0)
        embedded = embedded * action_mask.to(embedded.dtype).unsqueeze(-1)
        return self.joint_action_proj(embedded.reshape(embedded.shape[0], -1))

    def _initial_state(
        self,
        context_latent: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context_latent.ndim != 3:
            raise ValueError(
                f"Expected context_latent [N,E,D], got {tuple(context_latent.shape)}"
            )
        if entity_mask.shape != context_latent.shape[:2]:
            raise ValueError("entity_mask must match context [N,E]")
        denom = entity_mask.to(context_latent.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (
            context_latent * entity_mask.to(context_latent.dtype).unsqueeze(-1)
        ).sum(dim=1) / denom
        pooled = self.context_pool_proj(pooled)
        return torch.tanh(self.init_h(pooled)), torch.tanh(self.init_c(pooled))

    def _readout(
        self,
        context_latent: torch.Tensor,
        decoder_hidden: torch.Tensor,
        entity_mask: torch.Tensor,
        *,
        horizon: int,
        use_power_head: bool,
    ) -> torch.Tensor:
        n, entities, _ = context_latent.shape
        if entities > self.max_entities:
            raise ValueError(
                f"Context has {entities} entities but max_entities={self.max_entities}"
            )
        slot_ids = torch.arange(entities, device=context_latent.device)
        slot = self.slot_embedding(slot_ids).unsqueeze(0).expand(n, -1, -1)
        entity_context = self.entity_context_proj(context_latent)
        global_context = decoder_hidden.unsqueeze(1).expand(-1, entities, -1)
        features = torch.cat([entity_context, global_context, slot], dim=-1)
        if use_power_head and str(horizon) in self.power_heads:
            delta = self.power_heads[str(horizon)](features)
        else:
            delta = self.shared_head(features)
        pred = context_latent + self.residual_scale * torch.tanh(delta)
        return pred * entity_mask.to(pred.dtype).unsqueeze(-1)

    def forward(
        self,
        context_latent: torch.Tensor,
        action_seq: torch.Tensor,
        action_mask_seq: torch.Tensor,
        entity_mask: torch.Tensor,
        *,
        horizons: Iterable[int] | None = None,
        include_shared_predictions: bool = False,
    ) -> dict[int | str, torch.Tensor]:
        """Predict requested horizons directly from one real context.

        action_seq has shape [N,T,A] for integer actions or [N,T,A,C] for
        one-hot/probability actions. No predicted latent is consumed inside this
        method.
        """
        if action_seq.ndim not in (3, 4):
            raise ValueError(f"Unexpected action_seq shape {tuple(action_seq.shape)}")
        if action_mask_seq.ndim != 3:
            raise ValueError(f"Unexpected action_mask_seq shape {tuple(action_mask_seq.shape)}")
        if action_seq.shape[:3] != action_mask_seq.shape:
            raise ValueError("action_seq and action_mask_seq dimensions disagree")

        requested = tuple(sorted({int(h) for h in (horizons or self.horizons)}))
        if not requested:
            raise ValueError("No requested horizons")
        if min(requested) < 1:
            raise ValueError("Horizons must be positive")
        max_requested = max(requested)
        if max_requested > action_seq.shape[1]:
            raise ValueError(
                f"Need {max_requested} actions but received {action_seq.shape[1]}"
            )
        if max_requested > self.max_horizon:
            raise ValueError(
                f"Requested horizon {max_requested} exceeds configured max {self.max_horizon}"
            )

        h_state, c_state = self._initial_state(context_latent, entity_mask)
        outputs: dict[int | str, torch.Tensor] = {}
        for step in range(1, max_requested + 1):
            action_context = self.encode_joint_action(
                action_seq[:, step - 1], action_mask_seq[:, step - 1]
            )
            step_ids = torch.full(
                (context_latent.shape[0],), step, device=context_latent.device, dtype=torch.long
            )
            decoder_input = action_context + self.step_embedding(step_ids)
            h_state, c_state = self.decoder(self.dropout(decoder_input), (h_state, c_state))
            if step in requested:
                outputs[step] = self._readout(
                    context_latent,
                    h_state,
                    entity_mask,
                    horizon=step,
                    use_power_head=(step in self.horizons),
                )
                if include_shared_predictions:
                    outputs[f"shared_{step}"] = self._readout(
                        context_latent,
                        h_state,
                        entity_mask,
                        horizon=step,
                        use_power_head=False,
                    )
        return outputs

    def predict_block(
        self,
        context_latent: torch.Tensor,
        action_seq: torch.Tensor,
        action_mask_seq: torch.Tensor,
        entity_mask: torch.Tensor,
        *,
        horizon: int,
    ) -> torch.Tensor:
        horizon = int(horizon)
        if horizon not in self.horizons:
            raise ValueError(
                f"Binary block horizon {horizon} is not trained; available={self.horizons}"
            )
        return self(
            context_latent,
            action_seq[:, :horizon],
            action_mask_seq[:, :horizon],
            entity_mask,
            horizons=(horizon,),
        )[horizon]

    def predict_exact(
        self,
        context_latent: torch.Tensor,
        action_seq: torch.Tensor,
        action_mask_seq: torch.Tensor,
        entity_mask: torch.Tensor,
        *,
        horizon: int,
    ) -> torch.Tensor:
        """One-pass prediction for any horizon up to max_horizon.

        At powers of two the specialized power head is used. At other horizons
        the shared head is used. Exp45 trains that shared head on a rotating
        arbitrary horizon, so every step from 1 through max_horizon receives
        direct supervision over a sufficiently long run.
        """
        return self(
            context_latent,
            action_seq[:, :horizon],
            action_mask_seq[:, :horizon],
            entity_mask,
            horizons=(horizon,),
        )[int(horizon)]

    def predict_binary(
        self,
        context_latent: torch.Tensor,
        action_seq: torch.Tensor,
        action_mask_seq: torch.Tensor,
        entity_mask: torch.Tensor,
        *,
        horizon: int,
    ) -> BinaryPrediction:
        """Compose trained power-of-two jumps for any supplied action horizon.

        Horizons above the largest trained block reuse that largest block, e.g.
        25=16+8+1 and 32=16+16.
        """
        remaining = int(horizon)
        if remaining < 1 or remaining > action_seq.shape[1]:
            raise ValueError("Invalid binary prediction horizon")
        available = sorted((h for h in self.horizons if h <= remaining), reverse=True)
        blocks: list[int] = []
        offset = 0
        latent = context_latent
        while remaining:
            block = next((h for h in available if h <= remaining), None)
            if block is None:
                raise ValueError(
                    f"Cannot decompose horizon {horizon} with available blocks {self.horizons}"
                )
            latent = self.predict_block(
                latent,
                action_seq[:, offset : offset + block],
                action_mask_seq[:, offset : offset + block],
                entity_mask,
                horizon=block,
            )
            blocks.append(block)
            offset += block
            remaining -= block
        return BinaryPrediction(latent=latent, blocks=tuple(blocks))


def masked_latent_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.ndim == pred.ndim - 1:
        mask = mask.unsqueeze(-1)
    mask = mask.to(pred.dtype)
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return ((pred - target).pow(2) * mask).sum() / denom


def normalize_entity_latent(latent: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
    latent = F.layer_norm(
        latent.float(),
        (latent.shape[-1],),
        weight=None,
        bias=None,
        eps=1e-5,
    ).to(latent.dtype)
    return latent * entity_mask.to(latent.dtype).unsqueeze(-1)
