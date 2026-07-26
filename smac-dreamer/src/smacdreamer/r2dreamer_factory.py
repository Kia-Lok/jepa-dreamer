"""Factory that creates R2-Dreamer-compatible parallel env pools for SMAClite.

Returns (train_envs, eval_envs, obs_space, act_space) — the same 4-tuple that
R2-Dreamer's train.py expects from make_envs() — so the rest of the training
pipeline is unchanged.

Does NOT modify any file inside external/r2dreamer.
"""

import pathlib
import sys

import numpy as np


def _worker_seed(base_seed: int, idx: int, generation: int = 0) -> int:
    """Robust per-worker seed: hash (base_seed, idx, generation) via SeedSequence.

    Avoids correlating adjacent RNG streams (base+idx). Validated by the §0 spike to give
    de-correlated per-worker map sequences.
    """
    return int(
        np.random.SeedSequence([int(base_seed), int(idx), int(generation)])
        .generate_state(1, dtype=np.uint32)[0]
    )


def _ensure_paths():
    """Add r2dreamer and smaclite to sys.path in the calling process.

    Called before any r2dreamer import so that r2dreamer's own unqualified
    imports (tools, rssm, networks, envs.parallel, …) resolve correctly.
    Also called inside worker lambdas so spawned subprocesses get the paths too.
    """
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    for sub in ("external/r2dreamer", "external/smaclite"):
        p = str(root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def make_smaclite_env(scenario, max_episode_steps=200, seed=0, worker_idx=0):
    """Construct a single R2-Dreamer-compatible SMAClite env instance.

    Called inside ParallelEnv worker processes via a cloudpickle-serialised
    lambda; _ensure_paths() sets up sys.path before any import.
    """
    _ensure_paths()
    from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
    from smacdreamer.envs.r2dreamer_adapter import SMACliteR2DreamerAdapter

    env = SMACliteDreamerEnv(
        scenario=scenario,
        max_episode_steps=max_episode_steps,
        seed=seed + worker_idx,
    )
    return SMACliteR2DreamerAdapter(env)


def make_smaclite_envs(
    scenario,
    env_num,
    eval_episode_num,
    device,
    max_episode_steps=200,
    seed=0,
):
    """Create train and (optionally) eval ParallelEnv pools.

    Parameters
    ----------
    scenario        : SMAClite scenario ID, e.g. "2s3z"
    env_num         : number of parallel training environments
    eval_episode_num: number of parallel evaluation environments (0 = disabled)
    device          : torch device string, e.g. "cpu" or "cuda:0"
    max_episode_steps: per-episode step limit passed to SMACliteDreamerEnv
    seed            : base RNG seed; each worker gets seed + worker_idx

    Returns
    -------
    (train_envs, eval_envs, obs_space, act_space)
        train_envs : ParallelEnv
        eval_envs  : ParallelEnv | None
        obs_space  : gymnasium.spaces.Dict
        act_space  : gymnasium.spaces.Box  (multi_discrete=True)
    """
    _ensure_paths()
    from envs.parallel import ParallelEnv

    def constructor(idx):
        # Returns a zero-argument callable for ParallelEnv; captured vars are
        # cloudpickle-serialised so the worker subprocess can reconstruct the env.
        return lambda: make_smaclite_env(scenario, max_episode_steps, seed, idx)

    train_envs = ParallelEnv(constructor, env_num, device)
    eval_envs = (
        ParallelEnv(constructor, eval_episode_num, device)
        if eval_episode_num > 0
        else None
    )
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space


# ---------------------------------------------------------------------------
# Multimap factory
# ---------------------------------------------------------------------------

def make_smaclite_multimap_env(
    entries, pad_dims, sampling_mode, base_seed, worker_idx,
    reward_name, reward_params, gamma, max_episode_steps, obs_mode="flat",
    worker_generation=0, completed_episode_offset=0, include_jepa_obs=False,
    jepa_visibility_config=None,
    shared_map_probabilities=None, shared_map_version=None,
):
    # UNIFIED_PRIORITY_V1
    """Construct one R2-Dreamer-compatible multimap SMAClite env (worker-side).

    Reconstructs the MapSampler + resolved reward callable inside the worker from picklable
    primitives (entries, mode, names/params) — never pickles a live sampler or callable.
    All workers share the SAME pad_dims so every map presents the identical padded shape.

    ``obs_mode``: "flat" (legacy right-padded) or "structured" (canonical per-entity layout).
    Default "flat" so existing callers (eval, training) are unchanged until masking consumes
    the structured masks.
    """
    _ensure_paths()
    from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
    from smacdreamer.envs.r2dreamer_adapter import SMACliteR2DreamerAdapter
    from smacdreamer.envs.map_sampler import MapSampler
    from smacdreamer.envs.reward_registry import resolve

    sampler_seed = _worker_seed(
        base_seed, worker_idx,
        worker_generation if sampling_mode == 'adaptive_priority' else 0,
    )
    simulator_seed = _worker_seed(base_seed, worker_idx, worker_generation)
    sampler = MapSampler.from_entries(
        entries, mode=sampling_mode, seed=sampler_seed,
        shared_probabilities=shared_map_probabilities,
        shared_version=shared_map_version,
    )
    if sampling_mode != 'adaptive_priority':
        sampler.advance(int(completed_episode_offset))
    reward_fn = resolve(reward_name, reward_params)
    print(
        f"[env_lifecycle] constructing smaclite worker idx={worker_idx} "
        f"generation={worker_generation} sampler_seed={sampler_seed} "
        f"simulator_seed={simulator_seed} completed_episode_offset={int(completed_episode_offset)} "
        f"next_map={sampler.peek().name}",
        flush=True,
    )
    env = SMACliteDreamerEnv(
        scenario=entries[0].name,                 # placeholder; sampler drives map selection
        max_episode_steps=max_episode_steps,
        seed=simulator_seed,
        map_sampler=sampler,
        pad_dims=pad_dims,
        reward_fn=reward_fn,
        gamma=gamma,
        obs_mode=obs_mode,
        include_jepa_obs=include_jepa_obs,
        jepa_visibility_config=jepa_visibility_config,
    )
    return SMACliteR2DreamerAdapter(env)


def make_smaclite_multimap_envs(
    maps_folder,
    split_spec,                 # SplitSpec | dict
    env_num,
    eval_episode_num,
    device,
    sampling_mode="shuffled_round_robin",
    reward_name="dense_v3",
    reward_params=None,
    gamma=0.997,
    max_episode_steps=200,
    seed=0,
    padding_override=None,
    obs_mode="flat",
    train_entries=None,         # if given, skip discovery (explicit-folder datasets)
    test_entries=None,
    pad_dims=None,
    env_lifecycle=None,
    include_jepa_obs=False,
    jepa_visibility_config=None,
    shared_map_probabilities=None,
    shared_map_version=None,
):
    """Create multimap train + held-out eval ParallelEnv pools.

    Discovery runs ONCE in this (parent) process: scan folder -> split -> TRAIN-max padding
    (or override) -> all-map safety-net. Train workers sample the TRAIN split with the
    configured reward; eval workers sample the held-out TEST split with the SAME pad_dims and
    the ORIGINAL (unshaped) reward so eval reports the true generalisation metric.

    Returns
    -------
    (train_envs, eval_envs, obs_space, act_space, discovery_info)
        discovery_info: dict with train/test entry names, pad_dims, counts — for logging.
    """
    _ensure_paths()
    from envs.parallel import ParallelEnv
    from smacdreamer.envs.map_discovery import discover, SplitSpec

    if obs_mode not in ("flat", "structured"):
        raise ValueError(f"unsupported obs_mode {obs_mode!r}; expected 'flat' or 'structured'")

    # Explicit-folder datasets (P0.4) pre-discover train/validation entries and pass them in;
    # otherwise fall back to ratio/explicit split discovery on a single folder.
    if train_entries is None:
        if isinstance(split_spec, dict):
            split_spec = SplitSpec(**split_spec)
        # isolate_probe: probe each map in a recycled subprocess so discovery of a large
        # folder does not accumulate SMAClite native memory past the pod cap. In structured
        # mode the canonical layout is fixed by max_agents/enemies/actions (not max_obs_size).
        train_entries, test_entries, pad_dims = discover(
            maps_folder, split_spec, padding_override=padding_override, verbose=True,
            isolate_probe=True, obs_mode=obs_mode,
        )

    reward_params = reward_params or {}

    lifecycle = env_lifecycle or {}

    def train_ctor(idx, generation=0, completed_episode_offset=0):
        return lambda: make_smaclite_multimap_env(
            train_entries, pad_dims, sampling_mode, seed, idx,
            reward_name, reward_params, gamma, max_episode_steps, obs_mode=obs_mode,
            worker_generation=generation,
            completed_episode_offset=completed_episode_offset,
            include_jepa_obs=include_jepa_obs,
            jepa_visibility_config=jepa_visibility_config,
            shared_map_probabilities=shared_map_probabilities,
            shared_map_version=shared_map_version,
        )

    # Eval pool: held-out TEST maps, SAME padding, ORIGINAL reward (smaclite_default).
    def eval_ctor(idx, generation=0, completed_episode_offset=0):
        return lambda: make_smaclite_multimap_env(
            test_entries, pad_dims,
            ('shuffled_round_robin' if sampling_mode == 'adaptive_priority' else sampling_mode),
            seed + 10_000, idx,
            "smaclite_default", {}, gamma, max_episode_steps, obs_mode=obs_mode,
            worker_generation=generation,
            completed_episode_offset=completed_episode_offset,
            include_jepa_obs=include_jepa_obs,
            jepa_visibility_config=jepa_visibility_config,
        )

    train_envs = ParallelEnv(
        train_ctor,
        env_num,
        device,
        max_episodes_per_worker=int(lifecycle.get("max_episodes_per_worker", 0) or 0),
        shutdown_timeout_seconds=float(lifecycle.get("shutdown_timeout_seconds", 5.0)),
        log_worker_memory=bool(lifecycle.get("log_worker_memory", False)),
    )
    eval_envs = (
        ParallelEnv(
            eval_ctor,
            eval_episode_num,
            device,
            max_episodes_per_worker=0,
            shutdown_timeout_seconds=float(lifecycle.get("shutdown_timeout_seconds", 5.0)),
            log_worker_memory=bool(lifecycle.get("log_worker_memory", False)),
        )
        if eval_episode_num > 0 and test_entries
        else None
    )
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    discovery_info = {
        "train_maps": [e.name for e in train_entries],
        "test_maps": [e.name for e in test_entries],
        "n_train": len(train_entries),
        "n_test": len(test_entries),
        "padding": {
            "max_agents": pad_dims.max_agents, "max_enemies": pad_dims.max_enemies,
            "max_actions": pad_dims.max_actions, "max_obs_size": pad_dims.max_obs_size,
        },
    }
    return train_envs, eval_envs, obs_space, act_space, discovery_info
