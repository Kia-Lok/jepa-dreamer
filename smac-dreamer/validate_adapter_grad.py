from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smacdreamer.jepa.feature_adapter import JEPAFeatureAdapter  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    batch = 4
    entities = 7
    latent_dim = 11
    memory_dim = 13
    static_dim = 5
    out_dim = 32

    adapter = JEPAFeatureAdapter(
        latent_dim=latent_dim,
        memory_dim=memory_dim,
        static_dim=static_dim,
        out_dim=out_dim,
        num_entities=entities,
    )
    conditioned = torch.randn(batch, entities, latent_dim)
    memory = torch.randn(batch, entities, memory_dim)
    # Slot 2 is hidden/absent for all samples; others are exposed.
    mask = torch.ones(batch, entities)
    mask[:, 2] = 0.0
    static = torch.randn(batch, static_dim)

    feat = adapter(conditioned, memory, mask, static)
    loss = feat.square().mean()
    loss.backward()

    grads = {
        name: p.grad.detach().abs().sum().item()
        for name, p in adapter.named_parameters()
        if p.requires_grad
    }
    print("feature shape:", tuple(feat.shape))
    print("grad sums:")
    for name, value in grads.items():
        print(f"  {name}: {value:.6f}")

    if tuple(feat.shape) != (batch, out_dim):
        raise SystemExit("FAIL: wrong feature shape")
    if not grads or not all(v > 0 for v in grads.values()):
        raise SystemExit("FAIL: at least one adapter parameter did not receive gradient")
    print("PASS: feature adapter is trainable and receives gradients")


if __name__ == "__main__":
    main()
