from __future__ import annotations

import torch


class JEPAActionAdapter:
    """Convert R2 flattened factorised actions to JEPA per-agent one-hot actions."""

    def __init__(self, *, max_agents: int, max_actions: int, checkpoint_n_actions: int):
        self.max_agents = int(max_agents)
        self.max_actions = int(max_actions)
        self.checkpoint_n_actions = int(checkpoint_n_actions)
        if self.max_actions != self.checkpoint_n_actions:
            raise ValueError(
                "JEPA action width mismatch: "
                f"R2 max_actions={self.max_actions}, checkpoint n_actions={self.checkpoint_n_actions}"
            )

    def flat_to_jepa(
        self,
        flat_action: torch.Tensor,
        agent_active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if flat_action.shape[-1] != self.max_agents * self.max_actions:
            raise ValueError(
                f"flat action width {flat_action.shape[-1]} != {self.max_agents * self.max_actions}"
            )
        action = flat_action.reshape(*flat_action.shape[:-1], self.max_agents, self.max_actions)
        if agent_active_mask is None:
            mask = torch.ones(*flat_action.shape[:-1], self.max_agents, device=flat_action.device, dtype=flat_action.dtype)
        else:
            mask = agent_active_mask.to(device=flat_action.device, dtype=flat_action.dtype)
        # Padded/dead agents are forced to NOOP one-hot in the representation but masked out.
        noop = torch.zeros_like(action)
        noop[..., 0] = 1.0
        action = torch.where(mask.unsqueeze(-1) > 0, action, noop)
        return action, mask
