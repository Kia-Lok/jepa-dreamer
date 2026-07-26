from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JEPAStateSpec:
    entities: int
    latent_dim: int
    memory_dim: int
    static_dim: int

    @property
    def deter_dim(self) -> int:
        return self.entities * self.memory_dim + self.entities + self.entities + self.static_dim


def pack_state(
    memory: torch.Tensor,
    entity_mask: torch.Tensor,
    slot_mask: torch.Tensor,
    static_condition: torch.Tensor,
) -> torch.Tensor:
    if memory.ndim != 3:
        raise ValueError(f"memory must have [B,E,M], got {tuple(memory.shape)}")
    b, e, _ = memory.shape
    if entity_mask.shape != (b, e):
        raise ValueError("entity_mask shape must match memory [B,E]")
    if slot_mask.shape != (b, e):
        raise ValueError("slot_mask shape must match memory [B,E]")
    if static_condition.ndim != 2 or static_condition.shape[0] != b:
        raise ValueError("static_condition must have [B,S]")
    return torch.cat(
        [
            memory.reshape(b, -1),
            entity_mask.to(dtype=memory.dtype),
            slot_mask.to(dtype=memory.dtype),
            static_condition.to(dtype=memory.dtype),
        ],
        dim=-1,
    )


def unpack_state(deter: torch.Tensor, spec: JEPAStateSpec):
    if deter.shape[-1] != spec.deter_dim:
        raise ValueError(f"deter width {deter.shape[-1]} != JEPA state width {spec.deter_dim}")
    b = deter.shape[0]
    e, m, s = spec.entities, spec.memory_dim, spec.static_dim
    pos = 0
    memory = deter[:, pos : pos + e * m].reshape(b, e, m)
    pos += e * m
    entity_mask = deter[:, pos : pos + e]
    pos += e
    slot_mask = deter[:, pos : pos + e]
    pos += e
    static_condition = deter[:, pos : pos + s]
    return memory, entity_mask, slot_mask, static_condition
