"""Standalone smoke test for the Gymnasium-compatible SMACliteDreamerEnv (Phase 1A).

Exercises the migrated PyTorch-side interface only:
  * gymnasium reset() / step() returning (obs, reward, terminated, truncated, info)
  * factorised one-hot actions via FactorisedActionCodec
  * one full episode with valid sampled actions

This script must NOT import JAX, Elements, Embodied, Portal, or DreamerV3. It only puts
``src`` and ``external/smaclite`` on sys.path (no external/dreamerv3).

Usage (PowerShell):
    python scripts\\smoke_test_gym_smaclite_env.py --scenario 2s3z
Usage (bash):
    python scripts/smoke_test_gym_smaclite_env.py --scenario 2s3z
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Deliberately exclude external/dreamerv3 to guarantee a JAX-free import path.
for p in (ROOT / "src", ROOT / "external" / "smaclite"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv


def assert_no_forbidden_imports():
    forbidden = [m for m in ("jax", "elements", "embodied", "portal", "dreamerv3")
                 if m in sys.modules]
    if forbidden:
        raise AssertionError(f"Forbidden modules imported: {forbidden}")


def sample_valid_action(env: SMACliteDreamerEnv) -> np.ndarray:
    """Sample a flat factorised one-hot action valid under the current avail mask."""
    avail = env._env.unwrapped.get_avail_actions()
    ints = []
    for i in range(env.n_agents):
        valid = [j for j, v in enumerate(avail[i]) if v]
        ints.append(int(np.random.choice(valid)) if valid else 0)
    return env.codec.encode(ints, num_real_agents=env.n_agents)


def run(scenario: str):
    print(f"\n{'=' * 60}")
    print(f"Gymnasium SMACliteDreamerEnv smoke test — scenario: {scenario}")
    print(f"{'=' * 60}\n")

    env = SMACliteDreamerEnv(scenario=scenario, max_episode_steps=200, seed=0)
    assert_no_forbidden_imports()

    print(f"  n_agents        : {env.n_agents}")
    print(f"  n_enemies       : {env.n_enemies}")
    print(f"  n_actions       : {env.n_actions}")
    print(f"  obs_size        : {env.obs_size}")
    print(f"  action_space    : {env.action_space}")
    print(f"  obs fields      : {sorted(env.observation_space.spaces)}")
    print(f"  flat action dim : {env.codec.flat_dim}  (groups {env.codec.group_sizes})")
    print()

    obs, info = env.reset(seed=0)
    assert bool(obs["is_first"]) is True, "is_first must be True after reset"
    assert env.observation_space.contains(obs), "reset obs not in observation_space"

    episode_return = 0.0
    episode_length = 0
    step_invalid_total = 0
    masking_failure_total = 0
    terminated = truncated = False
    last_info = info

    while not (terminated or truncated):
        obs, reward, terminated, truncated, last_info = env.step(sample_valid_action(env))
        assert env.observation_space.contains(obs), "step obs not in observation_space"
        episode_return += float(reward)
        episode_length += 1
        step_invalid_total += int(float(last_info["log_step_post_mask_invalid_count"]))
        masking_failure_total += int(float(last_info["log_step_masking_failure_count"]))

    battle_won = bool(float(last_info["log_battle_won"]))

    print(f"  total reward        : {episode_return:.4f}")
    print(f"  episode length      : {episode_length}")
    print(f"  battle outcome      : {'WIN' if battle_won else 'loss/draw'}")
    print(f"  invalid-action count: {step_invalid_total}")
    print(f"  masking-failure cnt : {masking_failure_total}")
    print(f"  final state shape   : {obs['state'].shape}")
    print(f"  final avail shape   : {obs['avail_actions'].shape}")
    print(f"  is_last / is_terminal: {bool(obs['is_last'])} / {bool(obs['is_terminal'])}")
    print()

    assert episode_length > 0, "episode must have at least one step"

    env.close()
    print(f"{'=' * 60}")
    print("Gymnasium smoke test PASSED.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="2s3z")
    args = parser.parse_args()
    run(args.scenario)
