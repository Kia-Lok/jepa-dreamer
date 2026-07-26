"""Padding utilities for Phase 3 variable-size map support.

Observation padding convention (left-aligned):
    Each agent's obs vector (length obs_size) is stored left-aligned starting at
    offset i * max_obs_size within the flat state tensor. The trailing
    max_obs_size - obs_size features are zero. The world model sees these trailing
    zeros as "no signal", not as a dedicated padding token.

    Maps with the same n_agents but different obs_size will present different-length
    meaningful prefixes in the same slot positions. The world model must learn to
    interpret the zero-suffix as absent features.

    real_agent_action_mask distinguishes "agent slot is padded" from "action is
    unavailable for a real agent". avail_actions handles the latter and is the sole
    mask passed to policy and imagination masking.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PaddingDims:
    """Maximum dimensions across all maps in a Phase 3 manifest.

    max_enemies is used for validation and logging only. It does NOT pad enemy
    observations — enemies are embedded inside each agent's per-agent obs vector
    (part of obs_size). If entity-wise obs padding is added in a later phase,
    max_enemies would be used there.
    """
    max_agents: int
    max_enemies: int   # validation + logging only; no enemy-obs padding
    max_actions: int
    max_obs_size: int


def pad_state(obs_tuple, real_obs_size: int, max_agents: int, max_obs_size: int) -> np.ndarray:
    """Pad per-agent observations to (max_agents * max_obs_size,). Left-aligned; trailing zeros.

    Raises ValueError if:
      len(obs_tuple) > max_agents
      real_obs_size > max_obs_size
      any agent_obs.shape != (real_obs_size,)  [full ndarray shape check]
    """
    if len(obs_tuple) > max_agents:
        raise ValueError(
            f"pad_state: len(obs_tuple)={len(obs_tuple)} > max_agents={max_agents}")
    if real_obs_size > max_obs_size:
        raise ValueError(
            f"pad_state: real_obs_size={real_obs_size} > max_obs_size={max_obs_size}")
    expected_shape = (real_obs_size,)
    for i, agent_obs in enumerate(obs_tuple):
        arr = np.asarray(agent_obs)
        if arr.shape != expected_shape:
            raise ValueError(
                f"pad_state: agent {i} obs shape {arr.shape} != expected {expected_shape}")
    result = np.zeros(max_agents * max_obs_size, dtype=np.float32)
    for i, agent_obs in enumerate(obs_tuple):
        result[i * max_obs_size : i * max_obs_size + real_obs_size] = agent_obs
    return result


def pad_avail(avail, real_n_actions: int, max_agents: int, max_actions: int) -> np.ndarray:
    """Pad per-agent avail masks to (max_agents * max_actions,). Extra action slots = 0.

    Raises ValueError if:
      len(avail) > max_agents
      real_n_actions > max_actions
      any avail[i].shape != (real_n_actions,)  [full ndarray shape check]
    """
    if len(avail) > max_agents:
        raise ValueError(
            f"pad_avail: len(avail)={len(avail)} > max_agents={max_agents}")
    if real_n_actions > max_actions:
        raise ValueError(
            f"pad_avail: real_n_actions={real_n_actions} > max_actions={max_actions}")
    expected_shape = (real_n_actions,)
    for i, agent_avail in enumerate(avail):
        arr = np.asarray(agent_avail)
        if arr.shape != expected_shape:
            raise ValueError(
                f"pad_avail: agent {i} avail shape {arr.shape} != expected {expected_shape}")
    result = np.zeros(max_agents * max_actions, dtype=np.float32)
    for i, agent_avail in enumerate(avail):
        result[i * max_actions : i * max_actions + real_n_actions] = agent_avail
    return result


def make_agent_mask(real_n_agents: int, max_agents: int) -> np.ndarray:
    """Create agent_mask: 1.0 for real agent slots, 0.0 for padded. Shape: (max_agents,)."""
    mask = np.zeros(max_agents, dtype=np.float32)
    mask[:real_n_agents] = 1.0
    return mask


def make_real_agent_action_mask(agent_mask: np.ndarray, max_actions: int) -> np.ndarray:
    """Per-(agent, action) indicator that the agent slot is real.

    Shape: (max_agents * max_actions,) = np.repeat(agent_mask, max_actions).
    Value is 1.0 iff the agent slot is real, regardless of action availability.

    NOT the same as avail_actions. avail_actions reflects whether each action is
    currently available for a real agent; real_agent_action_mask reflects agent-slot
    reality only. A real agent with no available actions has
    real_agent_action_mask[...] = 1.0 but avail_actions[...] = 0.0.
    real_agent_action_mask is NOT used for policy or imagination masking — that is
    done exclusively via avail_actions.
    """
    return np.repeat(agent_mask, max_actions)


def validate_padding_dims(maps, pad_dims: PaddingDims) -> None:
    """Load each map's env and verify all dims fit within pad_dims.

    Raises ValueError listing every violating map and dimension.
    Called once at training startup before the first training step.
    Cost: one gym env init per map (acceptable at startup).
    """
    errors = []
    for entry in maps:
        try:
            if entry.type == 'builtin':
                import smaclite  # noqa: registers smaclite/* gym IDs
                import gymnasium as gym
                env = gym.make(f'smaclite/{entry.name}-v0')
                uw = env.unwrapped
                n_a, n_e, n_act, obs_sz = (
                    uw.n_agents, uw.n_enemies, uw.n_actions, uw.obs_size)
                env.close()
            else:
                import pathlib
                from smaclite.env.smaclite import SMACliteEnv as _SMACliteEnv
                root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
                abs_path = root / entry.path
                e = _SMACliteEnv(map_file=str(abs_path))
                n_a, n_e, n_act, obs_sz = (
                    e.n_agents, e.n_enemies, e.n_actions, e.obs_size)
                e.close()
        except Exception as exc:
            errors.append(f"  map='{entry.name}': failed to load: {exc}")
            continue

        for attr, actual, limit in [
            ('max_agents',   n_a,    pad_dims.max_agents),
            ('max_enemies',  n_e,    pad_dims.max_enemies),
            ('max_actions',  n_act,  pad_dims.max_actions),
            ('max_obs_size', obs_sz, pad_dims.max_obs_size),
        ]:
            if actual > limit:
                errors.append(
                    f"  map='{entry.name}' {attr}: actual={actual} > limit={limit}")

    if errors:
        raise ValueError(
            "Phase 3 padding dims are too small for one or more maps:\n"
            + "\n".join(errors)
            + "\nUpdate the 'padding' block in the manifest."
        )
