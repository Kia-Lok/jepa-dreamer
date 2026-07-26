"""Shared pytest fixtures and path setup for the smacdreamer test suite.

These tests must run WITHOUT importing JAX, Elements, Embodied, Portal, or DreamerV3.
Only ``src``, ``external/r2dreamer``, and ``external/smaclite`` are placed on ``sys.path``.
Tests that require the SMAClite simulator are skipped automatically when it is not importable,
so the pure-NumPy action-codec tests still run in any environment that has numpy.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Deliberately exclude external/dreamerv3 so no JAX/Elements/Embodied is importable here.
for _p in (ROOT / "src", ROOT / "external" / "r2dreamer", ROOT / "external" / "smaclite"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Some isolated hierarchy tests install lightweight modules with setdefault at collection time.
# Load the real pure-PyTorch implementation first so the complete suite is order-independent.
import smacdreamer.masked_actions  # noqa: E402,F401


def _smaclite_available() -> bool:
    try:
        import smaclite  # noqa: F401
        return True
    except Exception:
        return False


SMACLITE_AVAILABLE = _smaclite_available()
requires_smaclite = pytest.mark.skipif(
    not SMACLITE_AVAILABLE,
    reason="smaclite simulator not importable in this environment",
)

# A small fixed scenario used by env tests. 2s3z: 5 allied units, shared n_actions.
FIXED_SCENARIO = "2s3z"


@pytest.fixture
def fixed_env():
    """A fresh fixed-scenario SMACliteDreamerEnv (Phase 1, no padding)."""
    from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
    env = SMACliteDreamerEnv(scenario=FIXED_SCENARIO, max_episode_steps=50, seed=0)
    yield env
    env.close()
