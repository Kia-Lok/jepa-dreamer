"""Automatic map-priority controller for R2-Dreamer × SMAClite.

The controller receives critic-error feedback from training replay only and
publishes a shared CPU probability vector that environment workers read on
episode reset. Validation maps never feed this controller.

The prioritisation is deliberately task-agnostic:
  * no human difficulty labels,
  * no hand-authored event categories,
  * no reward thresholds.

Map probabilities combine rank-normalised critic error, rank-normalised
staleness, and an explicit uniform coverage floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import math

import torch


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _rank01(values: torch.Tensor) -> torch.Tensor:
    """Tie-aware ranks in [0, 1], with larger input receiving larger rank.

    Equal statistics must produce equal sampling scores. A plain argsort rank
    would arbitrarily privilege map order when all maps are initially unseen.
    """
    values = values.to(dtype=torch.float64, device="cpu").reshape(-1)
    if values.numel() <= 1:
        return torch.ones_like(values)
    unique, inverse = torch.unique(values, sorted=True, return_inverse=True)
    if unique.numel() <= 1:
        return torch.ones_like(values)
    return inverse.to(torch.float64) / float(unique.numel() - 1)


@dataclass(frozen=True)
class MapPrioritySettings:
    enabled: bool = True
    error_ema_decay: float = 0.99
    uniform_floor: float = 0.10
    staleness_mix: float = 0.20
    update_every_feedbacks: int = 256
    minimum_feedback: int = 4
    initial_error: float = 1.0

    @classmethod
    def from_config(cls, cfg: Any) -> "MapPrioritySettings":
        return cls(
            enabled=bool(_cfg_get(cfg, "enabled", True)),
            error_ema_decay=float(_cfg_get(cfg, "error_ema_decay", 0.99)),
            uniform_floor=float(_cfg_get(cfg, "uniform_floor", 0.10)),
            staleness_mix=float(_cfg_get(cfg, "staleness_mix", 0.20)),
            update_every_feedbacks=max(
                1, int(_cfg_get(cfg, "update_every_feedbacks", 256))
            ),
            minimum_feedback=max(0, int(_cfg_get(cfg, "minimum_feedback", 4))),
            initial_error=max(0.0, float(_cfg_get(cfg, "initial_error", 1.0))),
        )

    def validate(self) -> None:
        if not 0.0 <= self.error_ema_decay < 1.0:
            raise ValueError("map.error_ema_decay must be in [0, 1)")
        if not 0.0 <= self.uniform_floor <= 1.0:
            raise ValueError("map.uniform_floor must be in [0, 1]")
        if not 0.0 <= self.staleness_mix <= 1.0:
            raise ValueError("map.staleness_mix must be in [0, 1]")


class AdaptivePriorityController:
    """Learner-side map statistics plus worker-visible shared probabilities."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        map_ids: Sequence[int],
        config: Any,
        *,
        map_names: Sequence[str] | None = None,
    ) -> None:
        if not map_ids:
            raise ValueError("AdaptivePriorityController requires at least one map")
        self.map_ids = tuple(int(x) for x in map_ids)
        if len(set(self.map_ids)) != len(self.map_ids):
            raise ValueError(
                "Training map_id values must be unique for adaptive map replay"
            )
        self.map_names = tuple(map_names or [str(x) for x in self.map_ids])
        if len(self.map_names) != len(self.map_ids):
            raise ValueError("map_names length must match map_ids")

        map_cfg = _cfg_get(config, "map", config)
        self.settings = MapPrioritySettings.from_config(map_cfg)
        self.settings.validate()

        self._id_to_index = {mid: i for i, mid in enumerate(self.map_ids)}
        n = len(self.map_ids)
        self.error_ema = torch.full(
            (n,), self.settings.initial_error, dtype=torch.float64
        )
        self.feedback_count = torch.zeros(n, dtype=torch.int64)
        self.collection_count = torch.zeros(n, dtype=torch.int64)
        self.last_collection_step = torch.zeros(n, dtype=torch.int64)
        self.feedback_events = 0
        self.last_recompute_feedback = 0
        self.current_env_step = 0

        self.shared_probabilities = torch.full(
            (n,), 1.0 / n, dtype=torch.float64
        ).share_memory_()
        self.shared_version = torch.zeros((), dtype=torch.int64).share_memory_()

    @classmethod
    def from_entries(cls, entries: Sequence[Any], config: Any):
        return cls(
            [int(e.map_id) for e in entries],
            config,
            map_names=[str(e.name) for e in entries],
        )

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def set_env_step(self, step: int) -> None:
        self.current_env_step = max(self.current_env_step, int(step))

    @torch.no_grad()
    def record_collection(
        self,
        map_ids: torch.Tensor,
        is_first: torch.Tensor | None,
        *,
        env_step: int,
    ) -> None:
        """Record newly collected episodes using training transitions only."""
        self.set_env_step(env_step)
        ids = map_ids.detach().to("cpu")
        if ids.ndim > 1 and ids.shape[-1] == 1:
            ids = ids.squeeze(-1)
        ids = ids.reshape(-1).round().to(torch.int64)

        if is_first is None:
            active = torch.ones_like(ids, dtype=torch.bool)
        else:
            first = is_first.detach().to("cpu")
            if first.ndim > 1 and first.shape[-1] == 1:
                first = first.squeeze(-1)
            active = first.reshape(-1) > 0.5

        for mid in ids[active].tolist():
            idx = self._id_to_index.get(int(mid))
            if idx is None:
                continue
            self.collection_count[idx] += 1
            self.last_collection_step[idx] = int(env_step)

    @torch.no_grad()
    def record_critic_feedback(
        self,
        map_ids: torch.Tensor,
        errors: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        learner_step: int | None = None,
        env_step: int | None = None,
    ) -> None:
        """Aggregate absolute critic errors by per-timestep training map_id."""
        if not self.enabled:
            return
        if env_step is not None:
            self.set_env_step(env_step)

        ids = map_ids.detach().to("cpu")
        err = errors.detach().to(dtype=torch.float64, device="cpu")
        valid = valid_mask.detach().to(dtype=torch.float64, device="cpu")

        while ids.ndim > err.ndim and ids.shape[-1] == 1:
            ids = ids.squeeze(-1)
        while err.ndim > ids.ndim and err.shape[-1] == 1:
            err = err.squeeze(-1)
        while valid.ndim > ids.ndim and valid.shape[-1] == 1:
            valid = valid.squeeze(-1)

        ids = ids.round().to(torch.int64)
        ids, err, valid = torch.broadcast_tensors(ids, err, valid)
        finite = torch.isfinite(err) & torch.isfinite(valid) & (valid > 0)
        if not finite.any():
            return

        decay = self.settings.error_ema_decay
        updated_maps = 0
        for mid in torch.unique(ids[finite]).tolist():
            idx = self._id_to_index.get(int(mid))
            if idx is None:
                continue
            mask = finite & (ids == int(mid))
            denom = valid[mask].sum()
            if not torch.isfinite(denom) or float(denom) <= 0:
                continue
            mean_error = (err[mask] * valid[mask]).sum() / denom
            if not torch.isfinite(mean_error):
                continue
            if int(self.feedback_count[idx]) == 0:
                self.error_ema[idx] = mean_error
            else:
                self.error_ema[idx] = (
                    decay * self.error_ema[idx] + (1.0 - decay) * mean_error
                )
            self.feedback_count[idx] += int(mask.sum())
            updated_maps += 1

        if updated_maps:
            self.feedback_events += 1
        if (
            self.feedback_events - self.last_recompute_feedback
            >= self.settings.update_every_feedbacks
        ):
            self.recompute_probabilities()

    @torch.no_grad()
    def recompute_probabilities(self) -> torch.Tensor:
        n = len(self.map_ids)
        if not self.enabled:
            probs = torch.full((n,), 1.0 / n, dtype=torch.float64)
        else:
            error_score = _rank01(self.error_ema)

            unseen = self.feedback_count < self.settings.minimum_feedback
            if unseen.any():
                # Unseen/under-observed maps receive the highest learning score
                # until enough agent-derived evidence exists.
                error_score[unseen] = 1.0

            age = (
                torch.full_like(self.last_collection_step, self.current_env_step)
                - self.last_collection_step
            ).clamp_min(0)
            stale_score = _rank01(age.to(torch.float64))

            mix = self.settings.staleness_mix
            score = (1.0 - mix) * error_score + mix * stale_score
            score = torch.nan_to_num(score, nan=1.0, posinf=1.0, neginf=0.0)
            score = score.clamp_min(1e-12)
            adaptive = score / score.sum()

            floor = self.settings.uniform_floor
            probs = floor * torch.full_like(adaptive, 1.0 / n)
            probs += (1.0 - floor) * adaptive
            probs /= probs.sum()

        self.shared_probabilities.copy_(probs)
        self.shared_version.add_(1)
        self.last_recompute_feedback = self.feedback_events
        return probs.clone()

    def metrics(self) -> dict[str, float]:
        p = self.shared_probabilities.detach().to(torch.float64)
        entropy = float(-(p * p.clamp_min(1e-12).log()).sum())
        uniform_entropy = math.log(max(1, len(self.map_ids)))
        return {
            "priority/map_entropy": entropy,
            "priority/map_entropy_fraction": (
                entropy / uniform_entropy if uniform_entropy > 0 else 1.0
            ),
            "priority/map_probability_min": float(p.min()),
            "priority/map_probability_max": float(p.max()),
            "priority/maps_with_feedback": float((self.feedback_count > 0).sum()),
            "priority/map_count": float(len(self.map_ids)),
            "priority/map_feedback_events": float(self.feedback_events),
            "priority/map_probability_version": float(self.shared_version),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "map_ids": list(self.map_ids),
            "map_names": list(self.map_names),
            "error_ema": self.error_ema.clone(),
            "feedback_count": self.feedback_count.clone(),
            "collection_count": self.collection_count.clone(),
            "last_collection_step": self.last_collection_step.clone(),
            "feedback_events": int(self.feedback_events),
            "last_recompute_feedback": int(self.last_recompute_feedback),
            "current_env_step": int(self.current_env_step),
            "probabilities": self.shared_probabilities.clone(),
            "version": int(self.shared_version),
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ValueError(
                "Unsupported adaptive-priority checkpoint schema: "
                f"{state.get('schema_version')!r}"
            )
        saved_ids = tuple(int(x) for x in state["map_ids"])
        if saved_ids != self.map_ids:
            message = (
                "Adaptive map state map_id order does not match this training set. "
                "Refusing to attach statistics to the wrong maps."
            )
            if strict:
                raise ValueError(message)
            print(f"[adaptive_priority] WARN: {message} Starting uniform.", flush=True)
            return

        for name in (
            "error_ema",
            "feedback_count",
            "collection_count",
            "last_collection_step",
        ):
            target = getattr(self, name)
            source = torch.as_tensor(state[name], dtype=target.dtype, device="cpu")
            if source.shape != target.shape:
                raise ValueError(f"{name} shape mismatch: {source.shape} != {target.shape}")
            target.copy_(source)

        self.feedback_events = int(state.get("feedback_events", 0))
        self.last_recompute_feedback = int(state.get("last_recompute_feedback", 0))
        self.current_env_step = int(state.get("current_env_step", 0))
        probabilities = torch.as_tensor(
            state.get("probabilities", self.shared_probabilities),
            dtype=torch.float64,
            device="cpu",
        )
        if probabilities.shape != self.shared_probabilities.shape:
            raise ValueError("Saved adaptive map probability shape mismatch")
        probabilities = torch.nan_to_num(
            probabilities, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0)
        if float(probabilities.sum()) <= 0:
            probabilities.fill_(1.0 / probabilities.numel())
        else:
            probabilities /= probabilities.sum()
        self.shared_probabilities.copy_(probabilities)
        self.shared_version.fill_(int(state.get("version", 0)) + 1)

    def snapshot(self) -> list[dict[str, Any]]:
        p = self.shared_probabilities.detach().cpu()
        rows = []
        for i, mid in enumerate(self.map_ids):
            rows.append(
                {
                    "map_id": int(mid),
                    "map_name": self.map_names[i],
                    "probability": float(p[i]),
                    "critic_error_ema": float(self.error_ema[i]),
                    "feedback_count": int(self.feedback_count[i]),
                    "collection_count": int(self.collection_count[i]),
                    "last_collection_step": int(self.last_collection_step[i]),
                }
            )
        return rows
