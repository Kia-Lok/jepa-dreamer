"""API compatibility adapter between SMACliteDreamerEnv and R2-Dreamer.

R2-Dreamer's wrappers and ParallelEnv use the old Gymnasium 4-tuple API:
    env.reset()  -> obs_dict          (no info)
    env.step(a)  -> (obs, rew, done, info)

SMACliteDreamerEnv uses the modern Gymnasium 5-tuple API:
    env.reset()  -> (obs_dict, info)
    env.step(a)  -> (obs, rew, terminated, truncated, info)

This adapter bridges the gap.

log_* metric forwarding
-----------------------
SMACliteDreamerEnv returns all diagnostic metrics (``log_map_id``, the invalid-action
counters, reward breakdown, …) in the Gymnasium ``info`` dict. R2-Dreamer's ParallelEnv
*discards* ``info`` on step (``o, r, d, _ = p()``), so those keys would never reach the
TensorDict the trainer reads/aggregates. To keep the invalid-action signal (and map id)
flowing — without editing external/r2dreamer — this adapter merges every ``log_``-prefixed
``info`` key into the returned ``obs`` dict, which ParallelEnv *does* propagate. The encoder
(networks.MultiEncoder) already excludes any ``log_`` key, so these never become model
inputs; they are picked up only by OnlineTrainer's ``log_``-prefixed aggregation.

Action space: SMACliteDreamerEnv exposes MultiDiscrete([C]*A). Dreamer.__init__
detects the action space type via hasattr checks:
    hasattr(space, "n")              -> OneHotDist (single Discrete)
    hasattr(space, "multi_discrete") -> MultiOneHotDist (factorised)
    else                             -> continuous Normal

We convert MultiDiscrete([C]*A) to Box(0,1, shape=(C,)*A, multi_discrete=True).
Dreamer then computes:
    act_dim       = sum(shape) = A*C   (e.g. 5*11=55)
    actor.shape   = (C,)*A             (e.g. (11,11,11,11,11))
    actor dist    = MultiOneHotDist  with 5 groups of 11

The actor emits flat one-hot of shape (B, A*C) which SMACliteDreamerEnv.step()
already handles natively via its FactorisedActionCodec.
"""

import numpy as np
import gymnasium as gym


class FactorisedMultiOneHotSpace(gym.spaces.Space):
    """Action space for factorised multi-one-hot actions, WITHOUT materialising a
    ``(C,)*A`` array.

    Dreamer reads only ``.shape`` (=``(C,)*A`` → ``act_dim = sum(shape)`` and the actor's
    per-agent groups) and ``.multi_discrete``. A real ``gym.spaces.Box(shape=(C,)*A)``
    allocates ``low``/``high`` arrays of ``C ** A`` elements — for padded multimap dims
    (e.g. 10 agents × 16 actions → 16**10 ≈ 1.1e12 floats ≈ 4 TB) that OOMs the process at
    construction. The base ``gym.spaces.Space`` stores ``shape`` as a plain tuple, so this
    space costs nothing. ``sample()`` returns a flat one-hot ``(A*C,)`` (one random action
    per agent group), matching what SMACliteDreamerEnv.step()/the codec consume.
    """

    def __init__(self, num_agents: int, num_actions: int, dtype=np.float32):
        self.num_agents = int(num_agents)
        self.num_actions = int(num_actions)
        super().__init__(shape=(self.num_actions,) * self.num_agents, dtype=dtype)
        self.multi_discrete = True

    def sample(self, mask=None):
        flat = np.zeros(self.num_agents * self.num_actions, dtype=self.dtype)
        picks = self.np_random.integers(self.num_actions, size=self.num_agents)
        flat[np.arange(self.num_agents) * self.num_actions + picks] = 1.0
        return flat

    def contains(self, x) -> bool:
        x = np.asarray(x)
        return x.shape == (self.num_agents * self.num_actions,)


class SMACliteR2DreamerAdapter(gym.Wrapper):
    """Adapts SMACliteDreamerEnv to the old-style 4-tuple API that R2-Dreamer expects."""

    def __init__(self, env):
        super().__init__(env)
        # MultiDiscrete([C]*A) -> factorised multi-one-hot space (shape=(C,)*A,
        # multi_discrete=True). NOT a gym Box: a Box of shape (C,)*A would allocate C**A
        # elements and OOM under padding (see FactorisedMultiOneHotSpace).
        nvec = env.action_space.nvec                        # numpy array [C, C, ..., C]
        self.action_space = FactorisedMultiOneHotSpace(
            num_agents=len(nvec), num_actions=int(nvec[0]),
        )

    @staticmethod
    def _merge_log_keys(obs, info):
        """Copy every log_* key from info into obs so ParallelEnv propagates them.

        ParallelEnv drops info but keeps obs; the encoder excludes log_* keys, so these
        reach the trainer's log_ aggregation without ever becoming model inputs.
        """
        merged = dict(obs)
        for k, v in info.items():
            if k.startswith("log_"):
                merged[k] = np.asarray(v, dtype=np.float32)
        return merged

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._merge_log_keys(obs, info)              # ParallelEnv expects bare obs dict

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        obs = self._merge_log_keys(obs, info)
        return obs, float(reward), done, info               # 4-tuple expected by ParallelEnv

    def get_debug_context(self):
        fn = getattr(self.env, "get_debug_context", None)
        return fn() if callable(fn) else {}
