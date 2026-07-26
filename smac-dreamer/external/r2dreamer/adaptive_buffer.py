"""Candidate-pool sequence PER that preserves TorchRL SliceSampler semantics.

This is intentionally not a replacement for TorchRL's internal slice-index
machinery. It asks the existing SliceSampler for a larger pool of valid,
contiguous recurrent windows and then priority-resamples complete windows.

Advantages for this repository:
  * exact preservation of trajectory boundaries and strict sequence length,
  * no dependency on private TorchRL 0.9.x internals,
  * overwrite-safe persistent identities through monotonically increasing UIDs,
  * importance weights returned per selected sequence.

The probability correction is exact for the candidate distribution and is an
approximation to global sequence PER. The candidate multiplier controls that
approximation/throughput trade-off.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping
import pathlib

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _load_memmap_storage():
    try:
        from torchrl.data.replay_buffers import LazyMemmapStorage

        return LazyMemmapStorage
    except Exception:
        try:
            from torchrl.data import LazyMemmapStorage

            return LazyMemmapStorage
        except Exception as exc:
            raise RuntimeError(
                "buffer.storage_backend='memmap' was requested, but this "
                "TorchRL installation does not expose LazyMemmapStorage."
            ) from exc


@dataclass
class ReplaySampleInfo:
    transition_indices: list[torch.Tensor]
    sequence_uids: torch.Tensor
    importance_weights: torch.Tensor
    sampling_probabilities: torch.Tensor
    beta: float


class AdaptiveBuffer:
    """Drop-in R2-Dreamer buffer with candidate-pool sequence PER."""

    UID_KEY = "_priority_uid"

    def __init__(self, config, priority_controller=None):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.num_eps = 0
        self.storage_backend = str(getattr(config, "storage_backend", "tensor"))
        self.scratch_dir = getattr(config, "scratch_dir", None)
        self.priority_controller = priority_controller

        adaptive = _cfg_get(config, "adaptive_priority", None)
        seq_cfg = _cfg_get(adaptive, "sequence", adaptive)
        self.sequence_enabled = bool(_cfg_get(seq_cfg, "enabled", True))
        self.candidate_multiplier = max(
            1, int(_cfg_get(seq_cfg, "candidate_multiplier", 4))
        )
        self.alpha = max(0.0, float(_cfg_get(seq_cfg, "alpha", 0.6)))
        self.beta_start = float(_cfg_get(seq_cfg, "beta_start", 0.4))
        self.beta_end = float(_cfg_get(seq_cfg, "beta_end", 1.0))
        self.beta_anneal_env_steps = max(
            1, int(_cfg_get(seq_cfg, "beta_anneal_env_steps", 2_000_000))
        )
        self.priority_eps = max(0.0, float(_cfg_get(seq_cfg, "eps", 1e-6)))
        self.min_priority = max(
            1e-12, float(_cfg_get(seq_cfg, "min_priority", 1e-3))
        )
        self.max_priority_limit = max(
            self.min_priority,
            float(_cfg_get(seq_cfg, "max_priority", 100.0)),
        )
        self.cache_max_entries = max(
            self.batch_size,
            int(_cfg_get(seq_cfg, "cache_max_entries", 500_000)),
        )
        self._candidate_batch_size = (
            self.batch_size * self.candidate_multiplier
            if self.sequence_enabled
            else self.batch_size
        )
        self._env_step = 0
        self._next_uid = 0
        self._current_max_priority = 1.0
        self._priority_by_uid: OrderedDict[int, float] = OrderedDict()
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(_cfg_get(seq_cfg, "seed", 0)))
        self._last_metrics: dict[str, float] = {}

        storage = self._make_storage(config)
        self._buffer = ReplayBuffer(
            storage=storage,
            sampler=SliceSampler(
                num_slices=self._candidate_batch_size,
                end_key=None,
                traj_key="episode",
                truncated_key=None,
                strict_length=True,
            ),
            prefetch=0,
            batch_size=self._candidate_batch_size * (self.batch_length + 1),
        )
        print(
            "[replay] adaptive candidate_sequence_per="
            f"{self.sequence_enabled} target_batch={self.batch_size} "
            f"candidate_batch={self._candidate_batch_size} "
            f"alpha={self.alpha:g} beta={self.beta_start:g}->{self.beta_end:g}",
            flush=True,
        )

    def _make_storage(self, config):
        if self.storage_backend == "tensor":
            return LazyTensorStorage(
                max_size=config.max_size,
                device=self.storage_device,
                ndim=2,
            )
        if self.storage_backend == "memmap":
            if not self.scratch_dir:
                raise ValueError(
                    "buffer.storage_backend='memmap' requires buffer.scratch_dir"
                )
            scratch = pathlib.Path(str(self.scratch_dir))
            scratch.mkdir(parents=True, exist_ok=True)
            LazyMemmapStorage = _load_memmap_storage()
            kwargs = {
                "max_size": config.max_size,
                "scratch_dir": str(scratch),
                "device": self.storage_device,
                "ndim": 2,
            }
            try:
                return LazyMemmapStorage(**kwargs)
            except TypeError as exc:
                if "ndim" not in str(exc):
                    raise
                kwargs.pop("ndim")
                return LazyMemmapStorage(**kwargs)
        raise ValueError(
            f"Unknown buffer.storage_backend {self.storage_backend!r}; "
            "expected 'tensor' or 'memmap'."
        )

    def set_env_step(self, step: int) -> None:
        self._env_step = max(0, int(step))
        if self.priority_controller is not None:
            self.priority_controller.set_env_step(self._env_step)

    def current_env_step(self) -> int:
        return int(self._env_step)

    def add_transition(self, data):
        # The TensorDict passed by the trainer has batch shape (num_envs,).
        batch = int(data.batch_size[0])
        uids = torch.arange(
            self._next_uid,
            self._next_uid + batch,
            dtype=torch.int64,
            device=data.device,
        )
        self._next_uid += batch
        data = data.clone(recurse=False)
        data.set(self.UID_KEY, uids)
        self._buffer.extend(data.unsqueeze(1))

    def record_collection(self, data, *, env_step: int) -> None:
        if self.priority_controller is None or "log_map_id" not in data:
            return
        self.priority_controller.record_collection(
            data["log_map_id"],
            data.get("is_first", None),
            env_step=env_step,
        )

    def _beta(self) -> float:
        fraction = min(1.0, max(0.0, self._env_step / self.beta_anneal_env_steps))
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def _lookup_priorities(self, uids: torch.Tensor) -> torch.Tensor:
        values = []
        default = self._current_max_priority
        for uid in uids.tolist():
            value = self._priority_by_uid.get(int(uid), default)
            values.append(value)
        return torch.tensor(values, dtype=torch.float64)

    def _remember_priority(self, uid: int, priority: float) -> None:
        uid = int(uid)
        priority = float(priority)
        self._priority_by_uid[uid] = priority
        self._priority_by_uid.move_to_end(uid)
        while len(self._priority_by_uid) > self.cache_max_entries:
            self._priority_by_uid.popitem(last=False)

    def sample(self):
        sample_td, info = self._buffer.sample(return_info=True)
        C = self._candidate_batch_size
        L = self.batch_length + 1
        sample_td = sample_td.view(C, L)

        candidate_uids = (
            sample_td[self.UID_KEY][:, 0].detach().to("cpu").reshape(C).to(torch.int64)
        )
        raw_priorities = self._lookup_priorities(candidate_uids)
        raw_priorities = torch.nan_to_num(
            raw_priorities, nan=self._current_max_priority,
            posinf=self.max_priority_limit,
            neginf=self.min_priority,
        ).clamp(self.min_priority, self.max_priority_limit)

        if not self.sequence_enabled or self.alpha == 0.0:
            probabilities = torch.full((C,), 1.0 / C, dtype=torch.float64)
        else:
            scaled = (raw_priorities + self.priority_eps).pow(self.alpha)
            probabilities = scaled / scaled.sum().clamp_min(1e-12)

        # Replacement keeps the draw probability explicit and allows standard
        # per-draw IS weights. Duplicate windows are valid PER samples.
        selected = torch.multinomial(
            probabilities,
            self.batch_size,
            replacement=True,
            generator=self._rng,
        )
        selected_p = probabilities[selected]
        beta = self._beta()
        weights = (C * selected_p).clamp_min(1e-12).pow(-beta)
        weights /= weights.max().clamp_min(1e-12)

        sample_td = sample_td[selected]
        selected_uids = candidate_uids[selected]
        sample_td.pop(self.UID_KEY)

        raw_index = info["index"]
        if isinstance(raw_index, torch.Tensor):
            raw_index = [raw_index]
        transition_indices = [
            ind.view(C, L)[selected, 1:].clone() for ind in list(raw_index)
        ]

        src_dev = sample_td.device or self.storage_device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)

        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])

        ess = float(
            (weights.sum().square() / weights.square().sum().clamp_min(1e-12))
        )
        self._last_metrics = {
            "priority/sequence_candidate_multiplier": float(self.candidate_multiplier),
            "priority/sequence_raw_mean": float(raw_priorities.mean()),
            "priority/sequence_raw_max": float(raw_priorities.max()),
            "priority/sequence_probability_max": float(probabilities.max()),
            "priority/is_weight_mean": float(weights.mean()),
            "priority/is_weight_min": float(weights.min()),
            "priority/effective_sample_size": ess,
            "priority/beta": float(beta),
            "priority/cache_size": float(len(self._priority_by_uid)),
        }
        return data, ReplaySampleInfo(
            transition_indices=transition_indices,
            sequence_uids=selected_uids,
            importance_weights=weights.to(torch.float32),
            sampling_probabilities=selected_p.to(torch.float32),
            beta=float(beta),
        ), initial

    def update(self, index, stoch, deter):
        index = [ind.reshape(-1) for ind in index]
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        deter = deter.reshape(-1, *deter.shape[2:])
        n = index[0].shape[0]
        self._buffer[index[1], index[0]] = TensorDict(
            {"stoch": stoch, "deter": deter}, batch_size=(n,)
        )

    def update_priorities(
        self, sequence_uids: torch.Tensor, priorities: torch.Tensor
    ) -> None:
        uids = sequence_uids.detach().to("cpu").reshape(-1).to(torch.int64)
        vals = priorities.detach().to("cpu", dtype=torch.float64).reshape(-1)
        if uids.numel() != vals.numel():
            raise ValueError("sequence_uids and priorities must have same length")
        vals = torch.nan_to_num(
            vals,
            nan=self._current_max_priority,
            posinf=self.max_priority_limit,
            neginf=self.min_priority,
        )
        vals = (vals + self.priority_eps).clamp(
            self.min_priority, self.max_priority_limit
        )
        for uid, value in zip(uids.tolist(), vals.tolist()):
            self._remember_priority(int(uid), float(value))
        if vals.numel():
            self._current_max_priority = max(
                self._current_max_priority, float(vals.max())
            )

    def metrics(self) -> dict[str, float]:
        metrics = dict(self._last_metrics)
        if self.priority_controller is not None:
            metrics.update(self.priority_controller.metrics())
        return metrics

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()

    def close(self):
        storage = getattr(self._buffer, "storage", None)
        close = getattr(storage, "close", None)
        if callable(close):
            close()
