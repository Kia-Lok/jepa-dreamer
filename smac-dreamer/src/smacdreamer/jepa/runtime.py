"""Stable runtime imports for the frozen JEPA backend."""

from .checkpoint import load_frozen_jepa_checkpoint
from .world_model import FrozenJEPAWorldModel

__all__ = ["FrozenJEPAWorldModel", "load_frozen_jepa_checkpoint"]
