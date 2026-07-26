"""Dedicated held-out multimap evaluator (P0.4).

Explicitly iterates::

    for map in held_out_maps:
        for seed in fixed_eval_seeds:
            evaluate(map, seed)

This REPLACES the previous behaviour where ``episodes_per_map`` was used as the number of
parallel eval workers (a single unstructured pass that visited neither every map nor every
seed). Here every held-out map is evaluated under every configured seed.

Reported metrics, per-map and aggregate:
  * win rate
  * ORIGINAL SMAClite return (never the shaped reward)
  * episode length
  * timeout rate
  * final allied effective-HP fraction
  * final enemy effective-HP fraction

Aggregates are reported BOTH ways:
  * macro = each MAP contributes one sample (mean of per-map means) — the headline,
  * micro = each EPISODE contributes one sample (pooled over all map×seed episodes).

The PRIMARY checkpoint-selection metric is the **macro held-out win rate**. Shaped return is
never used for selection. Eval envs are constructed with the ``smaclite_default`` reward so the
measured return is the true (unshaped) environment return regardless of the training reward.
"""

from __future__ import annotations

import math
import inspect
from typing import Callable, Optional, Sequence

import numpy as np


# Per-episode metric keys produced by ``evaluate_episode``.
_METRICS = ("win", "original_return", "length", "timeout",
            "final_ally_ehp_frac", "final_enemy_ehp_frac")

# Sensible default fixed seeds when a config does not specify them.
DEFAULT_FIXED_SEEDS = (0, 1, 2, 3, 4)


def _call_env_factory(env_factory, args, *, include_jepa_obs, jepa_visibility_config, shutdown_timeout_seconds):
    sig = inspect.signature(env_factory)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    kwargs = {}
    if accepts_kwargs or "include_jepa_obs" in sig.parameters:
        kwargs["include_jepa_obs"] = include_jepa_obs
    if accepts_kwargs or "jepa_visibility_config" in sig.parameters:
        kwargs["jepa_visibility_config"] = jepa_visibility_config
    if accepts_kwargs or "shutdown_timeout_seconds" in sig.parameters:
        kwargs["shutdown_timeout_seconds"] = shutdown_timeout_seconds
    return env_factory(*args, **kwargs)


def _mean(xs) -> float:
    xs = list(xs)
    return sum(float(x) for x in xs) / len(xs) if xs else 0.0


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion (per-map win-rate CI)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _normal_ci(values, z: float = 1.96) -> tuple:
    """Mean ± normal CI across samples (each map's win rate is one sample) — macro CI."""
    values = [float(v) for v in values]
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return (mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return (max(0.0, mean - z * se), min(1.0, mean + z * se))


def evaluate_episode(agent, env, seed, device, max_episode_steps) -> dict:
    """Drive ONE deterministic greedy episode on a single-map env; return a metric dict.

    Uses the agent's evaluation (mode) action. Termination *type* is read from the obs
    is_last / is_terminal flags (the adapter's 4-tuple step collapses them into one ``done``),
    and the original return + final EHP fractions come from the env's ``log_`` info.
    """
    import torch
    from tensordict import TensorDict

    with torch.no_grad():
        obs = env.reset(seed=int(seed))
        state = agent.get_initial_state(1)
        act = state["prev_action"].clone()
        done = False
        ep_orig = 0.0
        length = 0
        last_obs, last_info = obs, {}
        # Hard guard: env truncates at max_episode_steps, but never loop unbounded.
        while not done and length <= int(max_episode_steps) + 1:
            td = TensorDict(
                {k: torch.as_tensor(v).unsqueeze(0).to(device) for k, v in obs.items()},
                batch_size=(1,),
            )
            td["action"] = act
            act, state = agent.act(td, state, eval=True)
            a = act.detach().cpu().numpy().reshape(-1)
            obs, reward, done, info = env.step(a)
            ep_orig += float(info.get("log_reward_original", reward))
            length += 1
            last_obs, last_info = obs, info

    is_last = bool(np.asarray(last_obs["is_last"]))
    is_terminal = bool(np.asarray(last_obs["is_terminal"]))
    return {
        "win": bool(last_info.get("battle_won", False)),
        "original_return": float(ep_orig),
        "length": int(length),
        "timeout": bool(is_last and not is_terminal),
        "final_ally_ehp_frac": float(last_info.get("log_final_ally_ehp_frac", 0.0)),
        "final_enemy_ehp_frac": float(last_info.get("log_final_enemy_ehp_frac", 0.0)),
    }


def evaluate_heldout(
    agent,
    test_entries,
    pad_dims,
    *,
    seeds: Sequence[int],
    device: str = "cpu",
    gamma: float = 0.997,
    max_episode_steps: int = 200,
    obs_mode: str = "flat",
    include_jepa_obs: bool = False,
    jepa_visibility_config=None,
    env_factory: Optional[Callable] = None,
    episode_fn: Optional[Callable] = None,
    shutdown_timeout_seconds: float = 5.0,
    progress: bool = False,
) -> dict:
    """Evaluate ``agent`` on every held-out map under every seed; return a structured report.

    Parameters
    ----------
    test_entries : list of MapEntry (need ``.name``; ``.family`` optional).
    seeds        : the fixed eval seeds — every map is run once per seed.
    env_factory  : isolated SMAClite subprocess factory by default; injectable for testing.
    episode_fn   : ``evaluate_episode`` by default; injectable for testing (no torch needed).

    The env for each map uses a ``fixed`` sampler over the single entry and the
    ``smaclite_default`` reward, so per-(map,seed) runs are deterministic (given the seed-
    propagation fix) and the return measured is the ORIGINAL env return.
    """
    if not test_entries:
        raise ValueError("evaluate_heldout: no held-out maps to evaluate")
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("evaluate_heldout: fixed_eval_seeds is empty")

    if env_factory is None:
        from smacdreamer.isolated_env import make_isolated_smaclite_env
        env_factory = make_isolated_smaclite_env
    if episode_fn is None:
        episode_fn = evaluate_episode

    if hasattr(agent, "eval"):
        agent.eval()

    per_map: dict = {}
    pooled: list = []   # every episode's metric dict (for micro averaging)
    validation_children: list = []

    for entry in test_entries:
        env = _call_env_factory(
            env_factory,
            ([entry], pad_dims, "fixed", 0, 0, "smaclite_default", {},
            gamma, max_episode_steps, obs_mode),
            include_jepa_obs=include_jepa_obs,
            jepa_visibility_config=jepa_visibility_config,
            shutdown_timeout_seconds=shutdown_timeout_seconds,
        )
        try:
            child_pid = getattr(env, "pid", None)
            if child_pid is not None:
                validation_children.append({"map": entry.name, "pid": int(child_pid)})
            ep_metrics = []
            for seed in seeds:
                m = episode_fn(agent, env, seed, device, max_episode_steps)
                ep_metrics.append(m)
                pooled.append(m)
        finally:
            try:
                env.close()
            except Exception:
                pass
            if getattr(env, "is_alive", False):
                raise RuntimeError(
                    f"validation child still alive after map {entry.name!r}: "
                    f"pid={getattr(env, 'pid', None)}"
                )

        agg = {k: _mean([m[k] for m in ep_metrics]) for k in _METRICS}
        wins = int(sum(1 for m in ep_metrics if m["win"]))
        lo, hi = _wilson(wins, len(ep_metrics))
        per_map[entry.name] = {
            "family": getattr(entry, "family", "uncategorised"),
            "n_episodes": len(ep_metrics),
            "win_rate": agg["win"],
            "win_rate_ci95": [lo, hi],
            "original_return": agg["original_return"],
            "length": agg["length"],
            "timeout_rate": agg["timeout"],
            "final_ally_ehp_frac": agg["final_ally_ehp_frac"],
            "final_enemy_ehp_frac": agg["final_enemy_ehp_frac"],
        }
        if progress:
            pm = per_map[entry.name]
            print(f"  {entry.name:<32} win={pm['win_rate']:.2f} "
                  f"return={pm['original_return']:.3f} len={pm['length']:.1f} "
                  f"timeout={pm['timeout_rate']:.2f} allyEHP={pm['final_ally_ehp_frac']:.2f} "
                  f"enemyEHP={pm['final_enemy_ehp_frac']:.2f}")

    # Macro: each MAP one sample (mean of per-map means).
    macro = {
        "win_rate":             _mean([m["win_rate"] for m in per_map.values()]),
        "original_return":      _mean([m["original_return"] for m in per_map.values()]),
        "length":               _mean([m["length"] for m in per_map.values()]),
        "timeout_rate":         _mean([m["timeout_rate"] for m in per_map.values()]),
        "final_ally_ehp_frac":  _mean([m["final_ally_ehp_frac"] for m in per_map.values()]),
        "final_enemy_ehp_frac": _mean([m["final_enemy_ehp_frac"] for m in per_map.values()]),
    }
    macro["win_rate_ci95"] = list(_normal_ci([m["win_rate"] for m in per_map.values()]))

    # Micro: each EPISODE one sample (pooled).
    micro = {
        "win_rate":             _mean([1.0 if m["win"] else 0.0 for m in pooled]),
        "original_return":      _mean([m["original_return"] for m in pooled]),
        "length":               _mean([m["length"] for m in pooled]),
        "timeout_rate":         _mean([1.0 if m["timeout"] else 0.0 for m in pooled]),
        "final_ally_ehp_frac":  _mean([m["final_ally_ehp_frac"] for m in pooled]),
        "final_enemy_ehp_frac": _mean([m["final_enemy_ehp_frac"] for m in pooled]),
    }

    return {
        "primary_metric": "macro_heldout_win_rate",
        "primary_value": macro["win_rate"],
        "selection_note": "Select checkpoints by macro held-out win rate; NEVER by shaped return.",
        "n_maps": len(per_map),
        "seeds": list(seeds),
        "episodes_per_map": len(seeds),
        "n_episodes_total": len(pooled),
        "macro": macro,
        "micro": micro,
        "per_map": per_map,
        "validation_children": validation_children,
    }


def is_validation_improvement(win_rate, original_return,
                              best_win_rate, best_original_return, eps: float = 1e-9) -> bool:
    """Checkpoint-selection rule: MACRO validation win rate is primary, MACRO original return
    is the tie-breaker. Never uses shaped return. Returns True if (win_rate, original_return)
    improves over the current best."""
    if win_rate > best_win_rate + eps:
        return True
    if abs(win_rate - best_win_rate) <= eps and original_return > best_original_return + eps:
        return True
    return False


__all__ = [
    "evaluate_episode", "evaluate_heldout", "is_validation_improvement", "DEFAULT_FIXED_SEEDS",
]
