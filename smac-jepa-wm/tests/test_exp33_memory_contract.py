from __future__ import annotations

import torch

from smac_jepa.anchored_belief_memory import (
    AnchoredActionConditionedEntityRolloutGRUMemory,
)


def main() -> None:
    latent_dim = 192
    memory_dim = 322
    max_agents = 10
    entities = 20
    n_actions = 198
    module = AnchoredActionConditionedEntityRolloutGRUMemory(
        latent_dim=latent_dim,
        memory_dim=memory_dim,
        n_actions=n_actions,
        max_agents=max_agents,
    )
    assert module.recurrent_dim == 128
    assert module.action_identity_preserved
    memory = module.initial_memory(
        2, entities, device=torch.device("cpu"), dtype=torch.float32
    )
    z = torch.randn(2, entities, latent_dim)
    z[:, 12:] = 0.0
    mask = torch.ones(2, entities)
    action = torch.randint(0, n_actions, (2, max_agents))
    action_mask = torch.ones(2, max_agents)
    conditioned = module.condition(z, memory, mask)
    updated = module.update(
        z,
        memory,
        mask,
        action=action,
        action_mask=action_mask,
    )
    assert conditioned.shape == (2, entities, latent_dim)
    assert updated.shape == (2, entities, memory_dim)
    assert torch.isfinite(conditioned).all()
    assert torch.isfinite(updated).all()
    print("Exp33 anchored memory contract self-test: PASS")


if __name__ == "__main__":
    main()
