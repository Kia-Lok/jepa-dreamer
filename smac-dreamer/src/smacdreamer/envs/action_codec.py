"""Factorised multi-agent action codec for centralised SMAClite control.

The centralised controller chooses *one categorical action per allied unit*. We do
**not** build a joint action space of size ``C ** A``. Instead the action is represented
internally as ``A`` independent categorical groups, concatenated into a single flat
one-hot vector of length ``A * C``:

    A = number of allied-agent slots (n_agents, or pad_dims.max_agents in Phase 3)
    C = number of actions per agent  (n_actions, or pad_dims.max_actions in Phase 3)

    flat one-hot shape : (A * C,)
    categorical groups : [C, C, ..., C]   (A groups)
    group i occupies   : flat[i * C : (i + 1) * C]

This factorised layout maps cleanly onto R2-Dreamer's ``MultiOneHotDist`` (which splits
its logits with ``torch.split(logits, [C] * A, dim=-1)``) while SMAClite still receives a
plain list of per-unit integer actions.

This module is framework-agnostic and pure NumPy: it must import without JAX, Elements,
Embodied, Portal, or DreamerV3. It accepts torch tensors opportunistically (anything with
``.detach().cpu().numpy()``) and always returns NumPy.

Padded agents (Phase 3): for slot indices ``>= n_real_agents`` the encoder forces noop
(action 0) and the decoder ignores the slot when producing the integer-action list sent
to SMAClite. Padded-slot reality is tracked by ``agent_mask`` upstream; this codec only
needs to know how many leading slots are real.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # gymnasium is a code dependency of the migrated env; degrade gracefully if absent.
    import gymnasium as _gym
except Exception:  # pragma: no cover - exercised only in JAX-only legacy environments
    _gym = None

# Index of the noop action within each agent's action group. SMAClite action ordering is
# [noop, stop, move(4), attack(...)]; index 0 is noop. Used to fill padded agent slots.
NOOP_ACTION: int = 0


def _to_numpy(x) -> np.ndarray:
    """Convert torch tensors / array-likes to a NumPy array without importing torch."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


@dataclass(frozen=True)
class FactorisedActionCodec:
    """Bidirectional codec between per-agent integer actions and a flat one-hot tensor.

    Parameters
    ----------
    num_agents : int
        ``A`` — number of agent slots (max_agents under padding). Determines how many
        categorical groups the flat action contains.
    num_actions : int
        ``C`` — number of discrete actions per agent (max_actions under padding).
    one_hot_dtype : np.dtype
        dtype of the produced flat one-hot array. float32 by default, matching the
        R2-Dreamer ``MultiOneHotAction`` Box space.
    """

    num_agents: int
    num_actions: int
    one_hot_dtype: np.dtype = np.float32

    def __post_init__(self):
        if self.num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {self.num_agents}")
        if self.num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {self.num_actions}")

    # ------------------------------------------------------------------
    # Shapes / spaces
    # ------------------------------------------------------------------

    @property
    def flat_dim(self) -> int:
        """Length of the concatenated one-hot vector: ``A * C``."""
        return self.num_agents * self.num_actions

    @property
    def group_sizes(self) -> list[int]:
        """Per-group cardinalities ``[C] * A`` (the split passed to MultiOneHotDist)."""
        return [self.num_actions] * self.num_agents

    @property
    def nvec(self) -> np.ndarray:
        """MultiDiscrete nvec: one categorical of size ``C`` per agent."""
        return np.full(self.num_agents, self.num_actions, dtype=np.int64)

    def action_space(self):
        """Return the Gymnasium action space for the factorised action.

        ``MultiDiscrete([C] * A)`` is the logical space (one categorical per agent). The
        downstream R2-Dreamer ``MultiOneHotAction`` wrapper converts this into a flat
        one-hot Box and tags it ``multi_discrete=True`` so the actor selects
        ``MultiOneHotDist``. Returning the logical MultiDiscrete here keeps this codec
        independent of R2-Dreamer internals.
        """
        if _gym is None:
            raise RuntimeError(
                "gymnasium is not importable; action_space() requires gymnasium."
            )
        return _gym.spaces.MultiDiscrete(self.nvec)

    # ------------------------------------------------------------------
    # Encode: per-agent integers -> flat one-hot
    # ------------------------------------------------------------------

    def encode(self, int_actions, num_real_agents: int | None = None) -> np.ndarray:
        """Encode per-agent integer actions into a flat one-hot vector ``(A * C,)``.

        Parameters
        ----------
        int_actions : sequence of int, length A (or num_real_agents)
            One action index per agent. Entries must be in ``[0, C)``.
        num_real_agents : int, optional
            If given, agent slots ``>= num_real_agents`` are *padded*: their group is set
            to noop (action 0) regardless of any supplied value. If ``int_actions`` has
            length ``num_real_agents`` it is treated as covering only the real agents and
            padded slots are appended automatically.

        Raises
        ------
        ValueError
            On wrong length or out-of-range action index.
        """
        arr = np.asarray(_to_numpy(int_actions)).reshape(-1).astype(np.int64)
        n_real = self.num_agents if num_real_agents is None else int(num_real_agents)
        if not (0 <= n_real <= self.num_agents):
            raise ValueError(
                f"num_real_agents={n_real} out of range [0, {self.num_agents}]"
            )

        if arr.shape[0] == self.num_agents:
            real_actions = arr[:n_real]
        elif arr.shape[0] == n_real:
            real_actions = arr
        else:
            raise ValueError(
                f"encode: expected {self.num_agents} (or {n_real}) actions, "
                f"got {arr.shape[0]}"
            )

        if real_actions.size and (real_actions.min() < 0 or real_actions.max() >= self.num_actions):
            bad = real_actions[(real_actions < 0) | (real_actions >= self.num_actions)]
            raise ValueError(
                f"encode: action index out of range [0, {self.num_actions}): {bad.tolist()}"
            )

        flat = np.zeros(self.flat_dim, dtype=self.one_hot_dtype)
        # Real agent groups.
        for i in range(n_real):
            flat[i * self.num_actions + int(real_actions[i])] = 1.0
        # Padded agent groups -> noop.
        for i in range(n_real, self.num_agents):
            flat[i * self.num_actions + NOOP_ACTION] = 1.0
        return flat

    # ------------------------------------------------------------------
    # Decode: flat one-hot -> per-agent integers
    # ------------------------------------------------------------------

    def decode(self, one_hot, num_real_agents: int | None = None, *, validate: bool = True) -> list[int]:
        """Decode a flat one-hot vector into a list of per-agent integer actions.

        Parameters
        ----------
        one_hot : array-like, shape (A * C,)
            Concatenated factorised one-hot (or, more leniently, per-group logits — the
            argmax of each group is taken). With ``validate=True`` each group must be a
            valid one-hot.
        num_real_agents : int, optional
            If given, only the first ``num_real_agents`` integer actions are returned —
            i.e. only actions for real agents are passed on to SMAClite; padded-agent
            slots are dropped.
        validate : bool
            If True, enforce that each group is exactly one-hot via ``validate_one_hot``.

        Returns
        -------
        list[int]
            ``num_real_agents`` (or ``A``) integer action indices in ``[0, C)``.
        """
        arr = self._reshape_to_groups(one_hot)
        if validate:
            self.validate_one_hot(arr)
        ints = arr.argmax(axis=-1).astype(int).tolist()
        if num_real_agents is not None:
            n_real = int(num_real_agents)
            if not (0 <= n_real <= self.num_agents):
                raise ValueError(
                    f"num_real_agents={n_real} out of range [0, {self.num_agents}]"
                )
            return ints[:n_real]
        return ints

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _reshape_to_groups(self, one_hot) -> np.ndarray:
        """Validate flat shape/dtype and reshape ``(A * C,)`` -> ``(A, C)``."""
        arr = np.asarray(_to_numpy(one_hot))
        if arr.dtype == bool or not np.issubdtype(arr.dtype, np.number):
            raise TypeError(
                f"one_hot must be a numeric array, got dtype {arr.dtype}"
            )
        flat = arr.reshape(-1)
        if flat.shape[0] != self.flat_dim:
            raise ValueError(
                f"one_hot flat length {flat.shape[0]} != A*C = "
                f"{self.num_agents}*{self.num_actions} = {self.flat_dim}"
            )
        return flat.reshape(self.num_agents, self.num_actions)

    def validate_one_hot(self, one_hot) -> None:
        """Raise ValueError unless every agent group is exactly one-hot.

        Accepts either a flat ``(A*C,)`` vector or a pre-reshaped ``(A, C)`` array.
        Values must be in {0, 1} and each group must sum to exactly 1.
        """
        arr = np.asarray(_to_numpy(one_hot))
        if arr.ndim == 1:
            arr = self._reshape_to_groups(arr)
        elif arr.shape != (self.num_agents, self.num_actions):
            raise ValueError(
                f"validate_one_hot: array shape {arr.shape} != "
                f"({self.num_agents}, {self.num_actions})"
            )
        if not np.all(np.isin(arr, (0.0, 1.0))):
            raise ValueError("validate_one_hot: values must be 0 or 1")
        sums = arr.sum(axis=-1)
        if not np.all(sums == 1.0):
            bad = np.where(sums != 1.0)[0].tolist()
            raise ValueError(
                f"validate_one_hot: groups {bad} are not exactly one-hot (sums={sums[bad].tolist()})"
            )


__all__ = ["FactorisedActionCodec", "NOOP_ACTION"]
