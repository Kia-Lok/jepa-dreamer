"""Hardened shared tactical latent policy for centralized R2-Dreamer.

Version 1.2 keeps the original primitive actor and action-mask contract intact,
but fixes several weaknesses from the initial tactical-mixture integration:

* a tiny configurable symmetry break avoids the exact zero-gradient diversity
  fixed point while keeping the inherited policy essentially unchanged;
* the anti-collapse loss only activates near collapse instead of forcing a
  globally uniform tactic marginal;
* diversity diagnostics never backpropagate into the inherited base actor;
* metrics distinguish marginal usage, sampled usage, argmax usage, and true
  state-dependent selector specialization (mutual information);
* empty action masks are repaired defensively before probability operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import math
import torch
from torch import nn
from torch.distributions import Categorical


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


@dataclass(frozen=True)
class TacticalSettings:
    enabled: bool = False
    num_tactics: int = 4
    embedding_dim: int = 16
    hidden_dim: int = 128
    tactic_pg_scale: float = 1.0
    tactic_entropy_scale: float = 1.0e-4
    collapse_loss_scale: float = 1.0e-3
    effect_loss_scale: float = 1.0e-3
    effect_target: float = 0.02
    residual_guard_scale: float = 1.0e-3
    max_residual_to_base: float = 1.0
    max_abs_residual_logit: float = 4.0
    max_effect_states: int = 256
    duration: int = 1
    symmetry_break_std: float = 1.0e-2
    selector_symmetry_break_std: float = 1.0e-3
    residual_scale: float = 0.25
    min_selector_mi_normalized: float = 0.05
    base_kl_target: float = 0.02
    base_kl_scale: float = 0.10
    max_usage_target: float = 0.80
    min_effective_tactics: float = 2.0
    eval_confidence_threshold: float = 0.55
    freeze_base_actor: bool = True
    freeze_feature_adapter: bool = True

    @classmethod
    def from_config(cls, cfg: Any) -> "TacticalSettings":
        # Backward compatibility: old configs used balance_loss_scale.
        collapse_default = float(_cfg_get(cfg, "balance_loss_scale", 1.0e-3))
        return cls(
            enabled=bool(_cfg_get(cfg, "enabled", False)),
            num_tactics=int(_cfg_get(cfg, "num_tactics", 4)),
            embedding_dim=int(_cfg_get(cfg, "embedding_dim", 16)),
            hidden_dim=int(_cfg_get(cfg, "hidden_dim", 128)),
            tactic_pg_scale=float(_cfg_get(cfg, "tactic_pg_scale", 1.0)),
            tactic_entropy_scale=float(
                _cfg_get(cfg, "tactic_entropy_scale", 1.0e-4)
            ),
            collapse_loss_scale=float(
                _cfg_get(cfg, "collapse_loss_scale", collapse_default)
            ),
            effect_loss_scale=float(_cfg_get(cfg, "effect_loss_scale", 1.0e-3)),
            effect_target=float(_cfg_get(cfg, "effect_target", 0.02)),
            residual_guard_scale=float(
                _cfg_get(cfg, "residual_guard_scale", 1.0e-3)
            ),
            max_residual_to_base=float(
                _cfg_get(cfg, "max_residual_to_base", 1.0)
            ),
            max_abs_residual_logit=float(
                _cfg_get(cfg, "max_abs_residual_logit", 4.0)
            ),
            max_effect_states=int(_cfg_get(cfg, "max_effect_states", 256)),
            duration=int(_cfg_get(cfg, "duration", 1)),
            symmetry_break_std=float(_cfg_get(cfg, "symmetry_break_std", 1.0e-2)),
            selector_symmetry_break_std=float(
                _cfg_get(cfg, "selector_symmetry_break_std", 1.0e-3)
            ),
            residual_scale=float(_cfg_get(cfg, "residual_scale", 0.25)),
            min_selector_mi_normalized=float(
                _cfg_get(cfg, "min_selector_mi_normalized", 0.05)
            ),
            base_kl_target=float(_cfg_get(cfg, "base_kl_target", 0.02)),
            base_kl_scale=float(_cfg_get(cfg, "base_kl_scale", 0.10)),
            max_usage_target=float(_cfg_get(cfg, "max_usage_target", 0.80)),
            min_effective_tactics=float(
                _cfg_get(cfg, "min_effective_tactics", 2.0)
            ),
            eval_confidence_threshold=float(
                _cfg_get(cfg, "eval_confidence_threshold", 0.55)
            ),
            freeze_base_actor=bool(
                _cfg_get(cfg, "freeze_base_actor", True)
            ),
            freeze_feature_adapter=bool(
                _cfg_get(cfg, "freeze_feature_adapter", True)
            ),
        )

    def validate(self) -> None:
        if self.num_tactics < 2:
            raise ValueError("tactical_mixture.num_tactics must be >= 2")
        if self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("tactical embedding/hidden dimensions must be positive")
        if self.duration != 1:
            raise ValueError(
                "Tactical Mixture v1.1 supports duration=1 only. Persistent "
                "tactics belong to the later hierarchical extension."
            )
        for name in (
            "tactic_pg_scale",
            "tactic_entropy_scale",
            "collapse_loss_scale",
            "effect_loss_scale",
            "effect_target",
            "residual_guard_scale",
            "max_residual_to_base",
            "max_abs_residual_logit",
            "symmetry_break_std",
            "selector_symmetry_break_std",
            "residual_scale",
            "min_selector_mi_normalized",
            "base_kl_target",
            "base_kl_scale",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"tactical_mixture.{name} must be finite and non-negative")
        if not 0.0 < self.residual_scale <= 1.0:
            raise ValueError("tactical_mixture.residual_scale must be in (0, 1]")
        if not 0.0 <= self.min_selector_mi_normalized <= 1.0:
            raise ValueError(
                "tactical_mixture.min_selector_mi_normalized must be in [0, 1]"
            )
        if self.base_kl_target <= 0:
            raise ValueError("tactical_mixture.base_kl_target must be positive")
        if self.max_effect_states <= 0:
            raise ValueError("tactical_mixture.max_effect_states must be positive")
        if self.max_residual_to_base <= 0:
            raise ValueError(
                "tactical_mixture.max_residual_to_base must be positive"
            )
        if self.max_abs_residual_logit <= 0:
            raise ValueError(
                "tactical_mixture.max_abs_residual_logit must be positive"
            )
        if not 1.0 / self.num_tactics <= self.max_usage_target <= 1.0:
            raise ValueError(
                "tactical_mixture.max_usage_target must be between uniform usage and 1"
            )
        if not 1.0 <= self.min_effective_tactics <= float(self.num_tactics):
            raise ValueError(
                "tactical_mixture.min_effective_tactics must be in [1, num_tactics]"
            )
        if not (
            1.0 / self.num_tactics
            <= self.eval_confidence_threshold
            <= 1.0
        ):
            raise ValueError(
                "tactical_mixture.eval_confidence_threshold must be between "
                "uniform tactic probability and 1"
            )


class TacticalMixturePolicy(nn.Module):
    """Team-level categorical selector and tactic-conditioned logit residual."""

    SCHEMA_VERSION = 3
    ARCHITECTURE = "tactical_mixture_v1_2"

    def __init__(self, feature_dim: int, action_logit_dim: int, config: Any) -> None:
        super().__init__()
        self.settings = TacticalSettings.from_config(config)
        self.settings.validate()
        self.feature_dim = int(feature_dim)
        self.action_logit_dim = int(action_logit_dim)
        if self.feature_dim <= 0 or self.action_logit_dim <= 0:
            raise ValueError("feature_dim and action_logit_dim must be positive")

        k = self.settings.num_tactics
        hidden = self.settings.hidden_dim
        emb = self.settings.embedding_dim
        self.selector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, k),
        )
        self.embedding = nn.Embedding(k, emb)
        self.residual = nn.Sequential(
            nn.Linear(self.feature_dim + emb, hidden),
            nn.ELU(),
            nn.Linear(hidden, self.action_logit_dim),
        )
        self.reset_parameters()

    @property
    def num_tactics(self) -> int:
        return self.settings.num_tactics

    def reset_parameters(self) -> None:
        # Start very close to uniform while avoiding the exact zero-gradient
        # mutual-information fixed point. Deterministic evaluation remains the
        # inherited policy because the confidence gate is zero below threshold.
        selector_std = self.settings.selector_symmetry_break_std / math.sqrt(
            float(self.settings.hidden_dim)
        )
        if selector_std > 0:
            nn.init.normal_(
                self.selector[-1].weight,
                mean=0.0,
                std=selector_std,
            )
        else:
            nn.init.zeros_(self.selector[-1].weight)
        nn.init.zeros_(self.selector[-1].bias)

        std = self.settings.symmetry_break_std / math.sqrt(
            float(self.settings.hidden_dim)
        )
        if std > 0:
            nn.init.normal_(self.residual[-1].weight, mean=0.0, std=std)
        else:
            nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def selector_logits(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[-1] != self.feature_dim:
            raise ValueError(
                f"tactical feature dim {feat.shape[-1]} != expected {self.feature_dim}"
            )
        return self.selector(feat.float())

    def selector_dist(self, feat: torch.Tensor) -> Categorical:
        return Categorical(logits=self.selector_logits(feat))

    def select_tactic(
        self,
        feat: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        logits = self.selector_logits(feat)
        return logits.argmax(dim=-1) if deterministic else Categorical(logits=logits).sample()

    def _uncentered_all_residual_logits(
        self,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        """Return bounded raw residuals with shape (..., K, action_dim)."""
        leading = feat.shape[:-1]
        k = self.num_tactics
        feat_all = feat.unsqueeze(-2).expand(*leading, k, self.feature_dim)
        tactic = torch.arange(k, device=feat.device, dtype=torch.long)
        tactic = tactic.view(*([1] * len(leading)), k).expand(*leading, k)
        emb = self.embedding(tactic)
        raw = self.residual(torch.cat([feat_all.float(), emb], dim=-1))
        cap = self.settings.max_abs_residual_logit
        return cap * torch.tanh(raw / cap)

    def residual_logits(self, feat: torch.Tensor, tactic: torch.Tensor) -> torch.Tensor:
        tactic = tactic.to(device=feat.device, dtype=torch.long)
        if tactic.shape != feat.shape[:-1]:
            raise ValueError(
                f"tactic shape {tuple(tactic.shape)} must equal feature leading "
                f"shape {tuple(feat.shape[:-1])}"
            )
        all_residual = self.all_residual_logits(feat)
        gather_index = tactic.unsqueeze(-1).unsqueeze(-1).expand(
            *tactic.shape,
            1,
            self.action_logit_dim,
        )
        return all_residual.gather(-2, gather_index).squeeze(-2)

    def combine_logits(
        self,
        base_logits: torch.Tensor,
        feat: torch.Tensor,
        tactic: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.shape[:-1] != feat.shape[:-1]:
            raise ValueError("base-logit and feature leading shapes differ")
        if base_logits.shape[-1] != self.action_logit_dim:
            raise ValueError("base-logit action dimension is incompatible")
        residual = self.residual_logits(feat, tactic).to(base_logits.dtype)
        return base_logits + residual

    def eval_combined_logits(
        self,
        base_logits: torch.Tensor,
        feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Conservative deterministic tactic selection for validation.

        A nearly uniform selector has not learned which tactic applies to the
        current state. Applying the arbitrary argmax category (ties resolve to
        index zero) can therefore make validation worse. The deterministic
        policy uses a continuous confidence gate: zero residual at uniform
        confidence and the full residual only once ``eval_confidence_threshold``
        is reached. This avoids both arbitrary tie-breaking and a discontinuous
        validation-policy switch.
        """
        tactic_logits = self.selector_logits(feat)
        tactic_probs = tactic_logits.float().softmax(dim=-1)
        confidence, tactic = tactic_probs.max(dim=-1)
        conditioned = self.combine_logits(base_logits, feat, tactic)
        # No tactical deviation is applied until the selector is actually
        # confident. This preserves the inherited deterministic policy during
        # the uncertain early phase instead of applying a large partial gate
        # merely because confidence exceeds the uniform baseline slightly.
        threshold = self.settings.eval_confidence_threshold
        denominator = max(1.0 - threshold, 1.0e-8)
        gate = ((confidence - threshold) / denominator).clamp(0.0, 1.0)
        final_logits = base_logits + gate.unsqueeze(-1) * (
            conditioned - base_logits
        )
        return final_logits, tactic, confidence, gate

    def all_residual_logits(self, feat: torch.Tensor) -> torch.Tensor:
        """Return zero-mean tactic residuals with shape (..., K, action_dim).

        Centering across tactics removes the common-mode failure observed in
        v1.1: the tactical branch cannot become a second generic actor that
        applies nearly the same large residual for every tactic. Only relative
        tactic-specific deviations survive.
        """
        raw = self._uncentered_all_residual_logits(feat)
        centered = raw - raw.mean(dim=-2, keepdim=True)
        return self.settings.residual_scale * centered

    def all_combined_logits(
        self,
        base_logits: torch.Tensor,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        """Return tactic-conditioned logits with shape (..., K, action_dim)."""
        return base_logits.unsqueeze(-2) + self.all_residual_logits(feat).to(
            base_logits.dtype
        )

    @staticmethod
    def _weights(
        weights: torch.Tensor | None,
        shape: Sequence[int],
        device: torch.device,
    ) -> torch.Tensor:
        if weights is None:
            out = torch.ones(tuple(shape), device=device, dtype=torch.float32)
        else:
            out = weights.detach().to(device=device, dtype=torch.float32)
            out = torch.broadcast_to(out, tuple(shape))
        return torch.nan_to_num(
            out,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0)

    def usage_statistics(
        self,
        tactic_logits: torch.Tensor,
        sampled_tactic: torch.Tensor | None = None,
        state_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Weighted selector statistics that separate balance from specialization."""
        probs = tactic_logits.float().softmax(dim=-1)
        weights = self._weights(state_weights, probs.shape[:-1], probs.device)
        denominator = weights.sum().clamp_min(1.0)

        marginal = (
            probs * weights.unsqueeze(-1)
        ).reshape(-1, self.num_tactics).sum(0) / denominator
        marginal = marginal / marginal.sum().clamp_min(1e-8)

        marginal_entropy = -(
            marginal.clamp_min(1e-8) * marginal.clamp_min(1e-8).log()
        ).sum()
        conditional_entropy_per_state = -(
            probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()
        ).sum(-1)
        conditional_entropy = (
            conditional_entropy_per_state * weights
        ).sum() / denominator
        mutual_information = (marginal_entropy - conditional_entropy).clamp_min(0)
        log_k = math.log(self.num_tactics)
        effective_count = marginal_entropy.exp()
        usage_max = marginal.max()

        max_excess = torch.relu(
            usage_max
            - torch.as_tensor(
                self.settings.max_usage_target,
                device=probs.device,
                dtype=probs.dtype,
            )
        )
        effective_shortfall = torch.relu(
            torch.as_tensor(
                self.settings.min_effective_tactics,
                device=probs.device,
                dtype=probs.dtype,
            )
            - effective_count
        ) / float(self.num_tactics)
        mi_normalized = mutual_information / log_k
        mi_shortfall = torch.relu(
            torch.as_tensor(
                self.settings.min_selector_mi_normalized,
                device=probs.device,
                dtype=probs.dtype,
            )
            - mi_normalized
        )
        collapse_loss = (
            max_excess.square()
            + effective_shortfall.square()
            + mi_shortfall.square()
        )

        argmax = probs.argmax(dim=-1)
        argmax_onehot = torch.nn.functional.one_hot(
            argmax,
            num_classes=self.num_tactics,
        ).float()
        argmax_usage = (
            argmax_onehot * weights.unsqueeze(-1)
        ).reshape(-1, self.num_tactics).sum(0) / denominator

        if sampled_tactic is None:
            sampled_usage = torch.zeros_like(marginal)
        else:
            sampled = sampled_tactic.to(device=probs.device, dtype=torch.long)
            if sampled.shape != probs.shape[:-1]:
                raise ValueError(
                    "sampled tactic leading shape does not match tactic logits"
                )
            sampled_onehot = torch.nn.functional.one_hot(
                sampled,
                num_classes=self.num_tactics,
            ).float()
            sampled_usage = (
                sampled_onehot * weights.unsqueeze(-1)
            ).reshape(-1, self.num_tactics).sum(0) / denominator

        return {
            "marginal": marginal,
            "sampled_usage": sampled_usage,
            "argmax_usage": argmax_usage,
            "marginal_entropy": marginal_entropy,
            "conditional_entropy": conditional_entropy,
            "mutual_information": mutual_information,
            "mutual_information_normalized": mi_normalized,
            "mi_shortfall": mi_shortfall,
            "effective_count": effective_count,
            "usage_max": usage_max,
            "selector_max_probability": (
                probs.max(-1).values * weights
            ).sum()
            / denominator,
            "selector_logit_std": tactic_logits.float().std(unbiased=False),
            "collapse_loss": collapse_loss,
        }

    def balance_loss(
        self,
        tactic_logits: torch.Tensor,
        state_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility shim: now returns collapse-only loss, not KL-to-uniform."""
        stats = self.usage_statistics(tactic_logits, None, state_weights)
        return stats["collapse_loss"], stats["marginal"]

    @staticmethod
    def _repair_empty_masks(mask: torch.Tensor) -> torch.Tensor:
        """Guarantee one valid action per agent for safe probability operations."""
        mask = mask.bool().clone()
        empty = ~mask.any(dim=-1)
        mask[..., 0] = mask[..., 0] | empty
        return mask

    def effect_statistics(
        self,
        feat: torch.Tensor,
        base_logits: torch.Tensor,
        action_mask: torch.Tensor,
        agent_active: torch.Tensor,
        actor_shape: Sequence[int],
        state_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Masked pairwise JS and residual metrics on a bounded state sample.

        ``base_logits`` is detached internally so the diversity auxiliary cannot
        change the inherited primitive actor merely to increase tactic distance.
        """
        actor_shape = tuple(int(value) for value in actor_shape)
        num_agents = len(actor_shape)
        if num_agents == 0 or len(set(actor_shape)) != 1:
            raise ValueError(
                "tactical effect metric requires equal action sizes per agent"
            )
        num_actions = actor_shape[0]
        if num_agents * num_actions != self.action_logit_dim:
            raise ValueError("actor shape does not match action-logit dimension")

        flat_feat = feat.detach().reshape(-1, self.feature_dim)
        flat_base = base_logits.detach().reshape(-1, self.action_logit_dim)
        flat_mask = self._repair_empty_masks(
            action_mask.reshape(-1, num_agents, num_actions)
        )
        flat_active = agent_active.reshape(-1, num_agents).bool()
        flat_weights = self._weights(
            state_weights,
            feat.shape[:-1],
            feat.device,
        ).reshape(-1)

        # Avoid data-dependent ``nonzero`` and Tensor-to-Python branches inside
        # the optionally torch-compiled gradient function. Replay batches are
        # already randomized, so a deterministic bounded prefix is sufficient
        # for this diagnostic/auxiliary objective. Invalid states retain zero
        # weight instead of changing tensor shapes.
        max_states = min(self.settings.max_effect_states, flat_feat.shape[0])
        feat_sel = flat_feat[:max_states]
        base_sel = flat_base[:max_states]
        mask_sel = flat_mask[:max_states]
        active_sel = flat_active[:max_states]
        weight_sel = flat_weights[:max_states]

        residual = self.all_residual_logits(feat_sel)
        logits = base_sel.unsqueeze(1) + residual
        logits = logits.reshape(-1, self.num_tactics, num_agents, num_actions)
        mask = mask_sel.unsqueeze(1).expand_as(logits)
        probs = logits.float().masked_fill(~mask, -1.0e9).softmax(dim=-1)
        base_logits_agent = base_sel.reshape(-1, num_agents, num_actions)
        base_probs = (
            base_logits_agent.float()
            .masked_fill(~mask_sel, -1.0e9)
            .softmax(dim=-1)
        )

        active_weight = active_sel.float() * weight_sel.unsqueeze(-1)
        denominator = active_weight.sum().clamp_min(1.0)
        state_weight = active_sel.any(-1).float() * weight_sel
        state_denominator = state_weight.sum().clamp_min(1.0)
        pair_values: list[torch.Tensor] = []
        eps = 1e-8
        for left in range(self.num_tactics):
            for right in range(left + 1, self.num_tactics):
                p = probs[:, left].clamp_min(eps)
                q = probs[:, right].clamp_min(eps)
                middle = 0.5 * (p + q)
                js = 0.5 * (
                    (p * (p.log() - middle.log())).sum(-1)
                    + (q * (q.log() - middle.log())).sum(-1)
                )
                pair_values.append((js * active_weight).sum() / denominator)

        pair_tensor = torch.stack(pair_values)
        residual_float = residual.float()
        residual_per_state = residual_float.square().mean(dim=(-1, -2))
        base_per_state = base_sel.float().square().mean(dim=-1)
        base_probs_expanded = base_probs.unsqueeze(1).expand_as(probs)
        tactic_kl = (
            probs.clamp_min(eps)
            * (
                probs.clamp_min(eps).log()
                - base_probs_expanded.clamp_min(eps).log()
            )
        ).sum(-1)
        tactic_active_weight = active_weight.unsqueeze(1).expand_as(tactic_kl)
        tactic_denominator = tactic_active_weight.sum().clamp_min(1.0)
        base_kl_mean = (
            tactic_kl * tactic_active_weight
        ).sum() / tactic_denominator
        base_kl_excess = torch.relu(
            tactic_kl
            - torch.as_tensor(
                self.settings.base_kl_target,
                device=tactic_kl.device,
                dtype=tactic_kl.dtype,
            )
        )
        base_kl_loss = (
            base_kl_excess.square() * tactic_active_weight
        ).sum() / tactic_denominator
        tactic_argmax = probs.argmax(dim=-1)
        base_argmax = base_probs.argmax(dim=-1).unsqueeze(1)
        action_flip_rate = (
            (tactic_argmax != base_argmax).float() * tactic_active_weight
        ).sum() / tactic_denominator

        result = {
            "js_mean": pair_tensor.mean(),
            "js_min": pair_tensor.min(),
            "js_max": pair_tensor.max(),
            "base_kl_mean": base_kl_mean,
            "base_kl_max": tactic_kl.masked_fill(
                tactic_active_weight <= 0,
                0.0,
            ).max(),
            "base_kl_loss": base_kl_loss,
            "action_flip_rate": action_flip_rate,
            "residual_rms": (
                (residual_per_state * state_weight).sum()
                / state_denominator
            ).sqrt(),
            "base_rms": (
                (base_per_state * state_weight).sum()
                / state_denominator
            ).sqrt(),
        }
        for index in range(self.num_tactics):
            per_state = residual_float[:, index].square().mean(dim=-1)
            result[f"residual_rms_{index}"] = (
                (per_state * state_weight).sum() / state_denominator
            ).sqrt()
        return result

    def effect_js(
        self,
        feat: torch.Tensor,
        base_logits: torch.Tensor,
        action_mask: torch.Tensor,
        agent_active: torch.Tensor,
        actor_shape: Sequence[int],
        state_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Backward-compatible mean JS helper."""
        return self.effect_statistics(
            feat,
            base_logits,
            action_mask,
            agent_active,
            actor_shape,
            state_weights,
        )["js_mean"]

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "architecture": self.ARCHITECTURE,
            "feature_dim": self.feature_dim,
            "action_logit_dim": self.action_logit_dim,
            "num_tactics": self.settings.num_tactics,
            "embedding_dim": self.settings.embedding_dim,
            "hidden_dim": self.settings.hidden_dim,
            "duration": self.settings.duration,
            "symmetry_break_std": self.settings.symmetry_break_std,
            "selector_symmetry_break_std": (
                self.settings.selector_symmetry_break_std
            ),
            "residual_scale": self.settings.residual_scale,
            "min_selector_mi_normalized": (
                self.settings.min_selector_mi_normalized
            ),
            "base_kl_target": self.settings.base_kl_target,
            "base_kl_scale": self.settings.base_kl_scale,
            "max_residual_to_base": self.settings.max_residual_to_base,
            "max_abs_residual_logit": self.settings.max_abs_residual_logit,
            "eval_confidence_threshold": (
                self.settings.eval_confidence_threshold
            ),
            "freeze_base_actor": self.settings.freeze_base_actor,
            "freeze_feature_adapter": (
                self.settings.freeze_feature_adapter
            ),
        }

    def assert_legacy_equivalence_ready(self) -> None:
        """Validate the conservative non-tactical warm start.

        The selector and residual output layers use tiny independent
        perturbations to avoid exact symmetry. Deterministic evaluation is
        nevertheless exactly the inherited actor because selector confidence
        begins below the hard application threshold.
        """
        if torch.count_nonzero(self.selector[-1].bias).item() != 0:
            raise RuntimeError("selector final bias is not zero-initialized")
        if torch.count_nonzero(self.residual[-1].bias).item() != 0:
            raise RuntimeError("residual final bias is not zero-initialized")
        for name, parameter, configured_std in (
            (
                "selector",
                self.selector[-1].weight,
                self.settings.selector_symmetry_break_std,
            ),
            (
                "residual",
                self.residual[-1].weight,
                self.settings.symmetry_break_std,
            ),
        ):
            if not torch.isfinite(parameter).all():
                raise RuntimeError(f"{name} final weight contains non-finite values")
            observed = float(parameter.detach().std(unbiased=False))
            expected = configured_std / math.sqrt(
                float(self.settings.hidden_dim)
            )
            upper = max(1.0e-10, expected * 4.0)
            if observed > upper:
                raise RuntimeError(
                    f"{name} symmetry-break std {observed:.6g} exceeds safe "
                    f"initialization bound {upper:.6g}"
                )
