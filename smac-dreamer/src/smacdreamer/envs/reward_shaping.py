"""Reward shaping configuration for DreamerV3 × SMAClite.

Provides RewardShapingConfig (a simple dataclass) and from_dict() to parse
a YAML block or elements.Config-like object into the config.

When reward_shaping_config is None or enabled=False in SMACliteDreamerEnv,
the env behaves identically to the original pipeline (zero behaviour change).
"""
from dataclasses import dataclass


@dataclass
class RewardShapingConfig:
    enabled: bool = False

    # Terminal outcome shaping (applied exactly once on the terminal step)
    win_bonus: float = 0.0       # added on terminal win
    loss_penalty: float = 0.0   # added on terminal loss (expected to be negative)

    # Per-step combat progress shaping
    enemy_kill_bonus: float = 0.0      # per enemy killed this step
    ally_death_penalty: float = 0.0    # per ally that died this step (expected negative)
    ally_survival_bonus: float = 0.0   # per alive ally every step

    # Time pressure
    step_penalty: float = 0.0         # subtracted every step

    # Reserved for future use
    damage_delta_scale: float = 0.0   # always 0.0 for now


def from_dict(d) -> RewardShapingConfig:
    """Parse a dict or Config-like object into RewardShapingConfig.

    Accepts plain dicts, elements.Config objects, and any mapping that
    supports [] or attribute access. Returns a disabled config if d is
    falsy or enabled is missing/False.
    """
    if not d:
        return RewardShapingConfig(enabled=False)

    def _get(key, default):
        try:
            if hasattr(d, '__getitem__'):
                v = d[key]
            else:
                v = getattr(d, key, default)
            return default if v is None else v
        except (KeyError, AttributeError, TypeError):
            return default

    if not _get("enabled", False):
        return RewardShapingConfig(enabled=False)

    return RewardShapingConfig(
        enabled=True,
        win_bonus=float(_get("win_bonus", 0.0)),
        loss_penalty=float(_get("loss_penalty", 0.0)),
        enemy_kill_bonus=float(_get("enemy_kill_bonus", 0.0)),
        ally_death_penalty=float(_get("ally_death_penalty", 0.0)),
        ally_survival_bonus=float(_get("ally_survival_bonus", 0.0)),
        step_penalty=float(_get("step_penalty", 0.0)),
        damage_delta_scale=float(_get("damage_delta_scale", 0.0)),
    )
