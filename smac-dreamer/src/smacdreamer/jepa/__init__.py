"""Frozen JEPA world-model support for R2-Dreamer.

The JEPA package is optional. Importing :mod:`smacdreamer.jepa` itself must not
require the external ``smac_jepa`` repository; loader/runtime construction raises
an actionable error only when the JEPA backend is selected.
"""

from .action_adapter import JEPAActionAdapter
from .feature_adapter import JEPAFeatureAdapter
from .state import JEPAStateSpec, pack_state, unpack_state

__all__ = [
    "JEPAActionAdapter",
    "JEPAFeatureAdapter",
    "JEPAStateSpec",
    "pack_state",
    "unpack_state",
]
