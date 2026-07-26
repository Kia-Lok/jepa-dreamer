"""Call-and-return hierarchical option policy for JEPA-backed R2-Dreamer.

The module is deliberately independent from JEPA and replay. It owns only:

* a high-level manager over discrete options;
* option-conditioned, zero-mean residuals around an inherited primitive actor;
* fixed-duration and source-interrupt execution guards;
* the real/imagination option state machine;
* option-collapse and behaviour-diversity diagnostics.

The option index is never an environment action. Primitive action masking remains
outside this module and must be applied after option-conditioned logits are built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class HierarchicalOptionSettings:
    enabled: bool = False
    num_options: int = 8
    option_embedding_dim: int = 16
    age_embedding_dim: int = 8
    hidden_dim: int = 128

    # Short fixed persistence with immediate source-group safety interruption.
    min_duration: int = 1
    max_duration: int = 4
    commitment_warmup_steps: int = 0
    commitment_full_steps: int = 1
    commitment_reselect_initial: float = 0.0
    commitment_reselect_final: float = 0.0

    initial_termination_probability: float = 0.10
    termination_warmup_steps: int = 800_000
    termination_full_steps: int = 800_001
    termination_max_probability_during_ramp: float = 0.30
    termination_max_probability_final: float = 0.30
    termination_cap_full_steps: int = 800_002
    # Temperature for a smooth approximation to min(raw_beta, cap). Unlike
    # interval rescaling, probabilities well below the cap remain unchanged as
    # the cap schedule relaxes.
    termination_soft_cap_temperature: float = 0.03
    termination_margin_normalized: float = 0.02
    termination_loss_scale: float = 0.0
    termination_entropy_scale: float = 0.0
    termination_collapse_scale: float = 0.0
    termination_mean_min: float = 0.02
    termination_mean_max: float = 0.60
    termination_advantage_clip: float = 1.0
    termination_min_advantage_magnitude: float = 0.01
    termination_max_target_disagreement: float = 0.25
    termination_unimix: float = 0.02
    eval_sample_termination: bool = False
    eval_termination_hazard_threshold: float = 1.0

    # Preserve the migrated source-group routing exactly; only within-group slot
    # identity is explored and learned.
    # The top-level two-way source routing remains separate from slot routing.
    # This prevents the six new slots from changing the inherited tactical
    # decision merely by becoming available.
    manager_unimix_initial: float = 0.0
    manager_unimix_final: float = 0.0
    manager_unimix_decay_steps: int = 600_000
    slot_manager_unimix: float = 0.01
    # A fixed probability floor on the immutable source anchor inside each
    # source group. This keeps all six children active from step zero while
    # preventing random child identities from dominating collection before
    # their state-dependent advantages are learned. The remaining mass is
    # allocated by the trainable slot manager.
    slot_anchor_floor: float = 0.40
    slot_pair_unlock_initial_steps: int = 0
    slot_pair_unlock_interval_steps: int = 1
    slot_unlock_ramp_steps: int = 1
    slot_pg_ramp_steps: int = 180_000
    manager_pg_scale: float = 1.0
    manager_pg_warmup_steps: int = 100_000
    manager_pg_full_steps: int = 300_000
    manager_entropy_scale: float = 0.0
    manager_collapse_scale: float = 0.0
    manager_mi_target_normalized: float = 0.10
    manager_mi_scale: float = 0.0
    max_usage_target: float = 0.95
    min_effective_options: float = 1.0

    worker_pg_scale: float = 1.0
    worker_pg_warmup_steps: int = 20_000
    worker_pg_full_steps: int = 150_000
    worker_entropy_scale: float = 0.0
    # The migrated worker is already trained. Do not zero it at step zero.
    worker_scale_initial: float = 0.25
    worker_scale_warmup_steps: int = 0
    worker_scale_full_steps: int = 1
    worker_scale_max: float = 0.25
    # Child slots learn only a bounded delta around their source tactic. Anchor
    # options 0/1 have exactly zero specialization delta for the full run.
    slot_delta_scale_max: float = 0.10
    max_abs_residual_logit: float = 2.0
    max_residual_to_base: float = 0.25
    residual_guard_scale: float = 0.05
    # These trust-region terms are evaluated against the frozen full Tactical
    # Mixture source policy, not merely the primitive base actor.
    base_kl_target: float = 0.002
    base_kl_tail_target: float = 0.01
    base_kl_tail_fraction: float = 0.10
    base_kl_tail_relative_scale: float = 1.0
    base_kl_scale: float = 0.50
    action_preservation_confidence: float = 0.80
    action_preservation_margin: float = 0.05
    action_preservation_scale: float = 0.50

    # Preserve the source Tactical Mixture's two-way routing distribution while
    # allowing four interruptible slots inside each source group.
    source_manager_group_count: int = 2
    manager_group_kl_target: float = 0.001
    manager_group_kl_tail_target: float = 0.005
    manager_group_kl_tail_fraction: float = 0.10
    manager_group_kl_tail_relative_scale: float = 1.0
    manager_group_kl_scale: float = 0.50
    manager_group_preservation_confidence: float = 0.80
    manager_group_preservation_margin: float = 0.05
    manager_group_preservation_scale: float = 0.50

    action_diversity_target: float = 0.002
    action_diversity_scale: float = 0.0
    residual_cosine_target: float = 0.95
    residual_cosine_scale: float = 0.0
    max_diversity_states: int = 2048
    max_diversity_pairs: int = 12

    option_critic_scale: float = 1.0
    # While child policies are still close to their source anchor, their option
    # values should not split merely because random option identities received
    # different Monte-Carlo noise. This consistency term decays exactly with
    # the worker PG warm-up and is zero once child policies are fully active.
    option_critic_consistency_scale: float = 1.0
    hierarchy_value_scale: float = 0.5
    slow_target_update: int = 1
    slow_target_fraction: float = 0.005

    freeze_base_actor: bool = True
    freeze_feature_adapter: bool = True
    # Keep the source JEPA representation fixed for this controlled 800k H=15
    # comparison. A later experiment may explicitly opt into adaptation.
    world_model_grad_scale_initial: float = 0.0
    world_model_grad_scale_final: float = 0.0
    world_model_grad_warmup_steps: int = 200_000
    world_model_grad_full_steps: int = 500_000

    # Exact H=15 for direct comparison with the requested RSSM R2-Dreamer run.
    imag_horizon_initial_max: int = 15
    imag_horizon_final_max: int = 15
    imag_horizon_window: int = 1
    imag_horizon_ramp_steps: int = 1

    @classmethod
    def from_config(cls, cfg: Any) -> "HierarchicalOptionSettings":
        return cls(**{
            field: _cfg_get(cfg, field, getattr(cls(), field))
            for field in cls.__dataclass_fields__
        })

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.num_options < 2:
            raise ValueError("hierarchical_options.num_options must be >= 2")
        if self.option_embedding_dim <= 0 or self.age_embedding_dim <= 0:
            raise ValueError("option and age embedding dimensions must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hierarchical_options.hidden_dim must be positive")
        if not 1 <= self.min_duration < self.max_duration:
            raise ValueError("require 1 <= min_duration < max_duration")
        if self.max_duration > 255:
            raise ValueError("max_duration > 255 is intentionally unsupported")
        if not 0.0 < self.initial_termination_probability < 1.0:
            raise ValueError("initial_termination_probability must be in (0, 1)")
        if not 0 <= self.termination_warmup_steps < self.termination_full_steps:
            raise ValueError("termination warmup/full steps are inconsistent")
        if not 0.0 < self.termination_max_probability_during_ramp <= 1.0:
            raise ValueError("termination ramp probability cap must be in (0, 1]")
        if not (
            self.termination_max_probability_during_ramp
            <= self.termination_max_probability_final
            <= 1.0
        ):
            raise ValueError("termination probability caps are inconsistent")
        if self.termination_cap_full_steps <= self.termination_full_steps:
            raise ValueError(
                "termination_cap_full_steps must exceed termination_full_steps"
            )
        if not math.isfinite(self.termination_soft_cap_temperature) or self.termination_soft_cap_temperature <= 0.0:
            raise ValueError("termination_soft_cap_temperature must be finite and positive")
        if not (
            0.0 <= self.manager_unimix_initial < 1.0
            and 0.0 <= self.manager_unimix_final < 1.0
        ):
            raise ValueError("manager unimix schedule is invalid")
        if self.manager_unimix_decay_steps <= 0:
            raise ValueError("manager_unimix_decay_steps must be positive")
        if not 0.0 <= self.slot_manager_unimix < 1.0:
            raise ValueError("slot_manager_unimix must be in [0, 1)")
        if not 0.0 <= self.slot_anchor_floor < 1.0:
            raise ValueError("slot_anchor_floor must be in [0, 1)")
        if self.slot_pair_unlock_initial_steps < 0:
            raise ValueError("slot_pair_unlock_initial_steps must be non-negative")
        if self.slot_pair_unlock_interval_steps <= 0:
            raise ValueError("slot_pair_unlock_interval_steps must be positive")
        if self.slot_unlock_ramp_steps <= 0 or self.slot_pg_ramp_steps <= 0:
            raise ValueError("slot unlock/PG ramps must be positive")
        if not 0 <= self.manager_pg_warmup_steps < self.manager_pg_full_steps:
            raise ValueError("manager PG warmup/full steps are inconsistent")
        if not 0 <= self.worker_pg_warmup_steps < self.worker_pg_full_steps:
            raise ValueError("worker PG warmup/full steps are inconsistent")
        if not 0 <= self.commitment_warmup_steps < self.commitment_full_steps:
            raise ValueError("commitment warmup/full steps are inconsistent")
        if not (
            0.0 <= self.commitment_reselect_final
            <= self.commitment_reselect_initial <= 1.0
        ):
            raise ValueError("commitment reselection schedule is invalid")
        if not 1.0 / self.num_options <= self.max_usage_target <= 1.0:
            raise ValueError("max_usage_target must be between uniform usage and 1")
        if not 1.0 <= self.min_effective_options <= self.num_options:
            raise ValueError("min_effective_options must be in [1, num_options]")
        if not 0.0 < self.worker_scale_max <= 1.0:
            raise ValueError("worker_scale_max must be in (0, 1]")
        if not 0.0 < self.slot_delta_scale_max <= self.worker_scale_max:
            raise ValueError("slot_delta_scale_max must be in (0, worker_scale_max]")
        if not 0.0 <= self.worker_scale_initial <= self.worker_scale_max:
            raise ValueError("worker_scale_initial must be in [0, worker_scale_max]")
        if not 0 < self.max_abs_residual_logit:
            raise ValueError("max_abs_residual_logit must be positive")
        if not 0 < self.max_residual_to_base:
            raise ValueError("max_residual_to_base must be positive")
        if not 0.0 <= self.termination_unimix < 1.0:
            raise ValueError("termination_unimix must be in [0, 1)")
        termination_floor = 0.5 * self.termination_unimix
        termination_ceiling = 1.0 - termination_floor
        if not termination_floor < self.initial_termination_probability < termination_ceiling:
            raise ValueError(
                "initial_termination_probability must lie inside the termination "
                "unimix support"
            )
        if not math.isfinite(self.eval_termination_hazard_threshold) or self.eval_termination_hazard_threshold <= 0.0:
            raise ValueError("eval_termination_hazard_threshold must be finite and positive")
        if not 0.0 <= self.termination_mean_min < self.termination_mean_max <= 1.0:
            raise ValueError("termination mean bounds are invalid")
        if not self.termination_advantage_clip > 0:
            raise ValueError("termination_advantage_clip must be positive")
        if not 0.0 <= self.termination_min_advantage_magnitude <= self.termination_advantage_clip:
            raise ValueError("termination_min_advantage_magnitude is invalid")
        if not self.termination_max_target_disagreement > 0:
            raise ValueError("termination_max_target_disagreement must be positive")
        if not 0.0 <= self.manager_mi_target_normalized <= 1.0:
            raise ValueError("manager_mi_target_normalized must be in [0, 1]")
        if not 0.0 <= self.residual_cosine_target <= 1.0:
            raise ValueError("residual_cosine_target must be in [0, 1]")
        if not 0 < self.base_kl_target:
            raise ValueError("base_kl_target must be positive")
        if not self.base_kl_tail_target >= self.base_kl_target:
            raise ValueError("base_kl_tail_target must be >= base_kl_target")
        if not 0.0 < self.base_kl_tail_fraction <= 1.0:
            raise ValueError("base_kl_tail_fraction must be in (0, 1]")
        if not 0.0 <= self.action_preservation_confidence <= 1.0:
            raise ValueError("action_preservation_confidence must be in [0, 1]")
        if not self.action_preservation_margin >= 0.0:
            raise ValueError("action_preservation_margin must be non-negative")
        if not 2 <= self.source_manager_group_count <= self.num_options:
            raise ValueError("source_manager_group_count must be in [2, num_options]")
        if self.num_options % self.source_manager_group_count != 0:
            raise ValueError("num_options must be divisible by source_manager_group_count")
        if not 0.0 < self.manager_group_kl_target:
            raise ValueError("manager_group_kl_target must be positive")
        if not self.manager_group_kl_tail_target >= self.manager_group_kl_target:
            raise ValueError("manager_group_kl_tail_target must be >= manager_group_kl_target")
        if not 0.0 < self.manager_group_kl_tail_fraction <= 1.0:
            raise ValueError("manager_group_kl_tail_fraction must be in (0, 1]")
        if not 0.0 <= self.manager_group_preservation_confidence <= 1.0:
            raise ValueError("manager_group_preservation_confidence must be in [0, 1]")
        if not self.manager_group_preservation_margin >= 0.0:
            raise ValueError("manager_group_preservation_margin must be non-negative")
        if not (
            0.0 <= self.world_model_grad_scale_initial <= 1.0
            and 0.0 <= self.world_model_grad_scale_final <= 1.0
        ):
            raise ValueError("world-model gradient scales must be in [0, 1]")
        if not 0 <= self.world_model_grad_warmup_steps < self.world_model_grad_full_steps:
            raise ValueError("world-model gradient schedule is inconsistent")
        if not 1 <= self.imag_horizon_initial_max <= self.imag_horizon_final_max:
            raise ValueError("imagination horizon maxima are inconsistent")
        if not 1 <= self.imag_horizon_window <= self.imag_horizon_initial_max:
            raise ValueError("imag_horizon_window is invalid")
        if self.imag_horizon_ramp_steps <= 0:
            raise ValueError("imag_horizon_ramp_steps must be positive")
        if self.max_diversity_states <= 0 or self.max_diversity_pairs <= 0:
            raise ValueError("diversity subsampling limits must be positive")
        if not 0 < self.slow_target_fraction <= 1.0:
            raise ValueError("slow_target_fraction must be in (0, 1]")
        for name in (
            "termination_margin_normalized",
            "termination_loss_scale",
            "termination_entropy_scale",
            "termination_collapse_scale",
            "manager_pg_scale",
            "manager_entropy_scale",
            "manager_collapse_scale",
            "manager_mi_target_normalized",
            "manager_mi_scale",
            "worker_pg_scale",
            "worker_scale_initial",
            "worker_scale_max",
            "slot_delta_scale_max",
            "worker_entropy_scale",
            "residual_guard_scale",
            "base_kl_tail_relative_scale",
            "base_kl_scale",
            "action_preservation_scale",
            "manager_group_kl_tail_relative_scale",
            "manager_group_kl_scale",
            "manager_group_preservation_scale",
            "action_diversity_target",
            "action_diversity_scale",
            "residual_cosine_target",
            "residual_cosine_scale",
            "option_critic_scale",
            "option_critic_consistency_scale",
            "hierarchy_value_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"hierarchical_options.{name} must be finite and non-negative")


class OptionStep(NamedTuple):
    option: torch.Tensor
    action_age: torch.Tensor
    carry_age: torch.Tensor
    has_option: torch.Tensor
    option_started: torch.Tensor
    option_terminated: torch.Tensor
    termination_eligible: torch.Tensor
    termination_probability: torch.Tensor
    previous_option: torch.Tensor
    previous_age: torch.Tensor
    manager_log_prob: torch.Tensor
    manager_entropy: torch.Tensor
    carry_termination_hazard: torch.Tensor


class HierarchicalOptionsPolicy(nn.Module):
    """Manager, option worker residuals, and learned termination head."""

    ARCHITECTURE = "dreamer_option_critic_v9_anchor_safe_8slot"
    SCHEMA_VERSION = 9

    def __init__(self, feature_dim: int, action_logit_dim: int, config: Any) -> None:
        super().__init__()
        self.settings = HierarchicalOptionSettings.from_config(config)
        self.settings.validate()
        self.feature_dim = int(feature_dim)
        self.action_logit_dim = int(action_logit_dim)
        if self.feature_dim <= 0 or self.action_logit_dim <= 0:
            raise ValueError("feature_dim and action_logit_dim must be positive")

        s = self.settings
        if s.num_options % s.source_manager_group_count != 0:
            raise ValueError("num_options must be divisible by source groups")
        self.slots_per_group = s.num_options // s.source_manager_group_count
        # Factorized manager: frozen source tactical group first, trainable
        # within-group specialization slot second. All slot identities are
        # available at migration, but every child initially executes its exact
        # source policy, preserving primitive behavior.
        self.manager_group = nn.Sequential(
            nn.Linear(self.feature_dim, s.hidden_dim),
            nn.ELU(),
            nn.Linear(s.hidden_dim, s.source_manager_group_count),
        )
        self.manager_slot = nn.Sequential(
            nn.Linear(self.feature_dim, s.hidden_dim),
            nn.ELU(),
            nn.Linear(s.hidden_dim, s.num_options),
        )
        # The inherited worker operates over the two source groups. Child slots
        # add bounded deltas through a separate zero-initialized head.
        self.option_embedding = nn.Embedding(
            s.source_manager_group_count, s.option_embedding_dim
        )
        self.slot_embedding = nn.Embedding(s.num_options, s.option_embedding_dim)
        # Termination has its own option embedding so termination gradients cannot
        # silently rewrite the worker policy through a shared representation.
        self.termination_option_embedding = nn.Embedding(
            s.num_options, s.option_embedding_dim
        )
        self.age_embedding = nn.Embedding(s.max_duration + 1, s.age_embedding_dim)
        self.worker_residual = nn.Sequential(
            nn.Linear(self.feature_dim + s.option_embedding_dim, s.hidden_dim),
            nn.ELU(),
            nn.Linear(s.hidden_dim, self.action_logit_dim),
        )
        self.slot_delta = nn.Sequential(
            nn.Linear(self.feature_dim + s.option_embedding_dim, s.hidden_dim),
            nn.ELU(),
            nn.Linear(s.hidden_dim, self.action_logit_dim),
        )
        self.termination = nn.Sequential(
            nn.Linear(
                self.feature_dim + s.option_embedding_dim + s.age_embedding_dim,
                s.hidden_dim,
            ),
            nn.ELU(),
            nn.Linear(s.hidden_dim, 1),
        )
        self.register_buffer("training_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("diversity_calls", torch.zeros((), dtype=torch.long))
        self.register_buffer("horizon_calls", torch.zeros((), dtype=torch.long))
        # Schedules and pair rotation are host-side control decisions. Mirror
        # their checkpointed buffers with Python integers to avoid repeated
        # accelerator synchronizations from Tensor.item() in the hot path.
        self._training_step_int = 0
        self._diversity_calls_int = 0
        self._horizon_calls_int = 0
        self.reset_parameters()

    @property
    def num_options(self) -> int:
        return self.settings.num_options

    def reset_parameters(self) -> None:
        # The inherited primitive actor remains the initial policy. Option
        # residuals start tiny but non-identical so diversity gradients do not
        # sit at an exact symmetric fixed point.
        nn.init.normal_(
            self.manager_group[-1].weight,
            mean=0.0,
            std=1.0e-3 / math.sqrt(float(self.settings.hidden_dim)),
        )
        nn.init.zeros_(self.manager_group[-1].bias)
        # Keep the manager output exact and state-independent at migration while
        # preserving a live gradient path. Zeroing *both* linear layers creates a
        # dead MLP that can only learn global output biases. The hidden layer is
        # therefore non-degenerate and the final layer alone starts at zero.
        nn.init.orthogonal_(self.manager_slot[0].weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.manager_slot[0].bias)
        nn.init.zeros_(self.manager_slot[-1].weight)
        nn.init.zeros_(self.manager_slot[-1].bias)
        nn.init.normal_(
            self.worker_residual[-1].weight,
            mean=0.0,
            std=1.0e-2 / math.sqrt(float(self.settings.hidden_dim)),
        )
        nn.init.zeros_(self.worker_residual[-1].bias)
        # The child policy must have exactly zero output at migration but must not
        # be a dead network. A live hidden layer plus a zero output layer gives
        # exact source actions and non-zero output-weight gradients immediately.
        nn.init.orthogonal_(self.slot_delta[0].weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.slot_delta[0].bias)
        nn.init.zeros_(self.slot_delta[-1].weight)
        nn.init.zeros_(self.slot_delta[-1].bias)
        nn.init.zeros_(self.termination[-1].weight)
        # Initialise the learned branch so the *executed* probability equals the
        # fixed hazard under the initial smooth cap. Bisection avoids coupling
        # this initialisation to an approximate closed-form inverse.
        target = float(self.settings.initial_termination_probability)
        lo, hi = 1.0e-6, 1.0 - 1.0e-6
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            value = self._bounded_probability_from_raw_scalar(
                mid,
                float(self.settings.termination_max_probability_during_ramp),
            )
            if value < target:
                lo = mid
            else:
                hi = mid
        initial_raw = 0.5 * (lo + hi)
        nn.init.constant_(self.termination[-1].bias, _logit(initial_raw))

    def set_training_step(self, step: int | torch.Tensor) -> None:
        value = int(step.detach().cpu().item()) if torch.is_tensor(step) else int(step)
        value = max(value, 0)
        self._training_step_int = value
        self.training_step.fill_(value)

    def set_diversity_calls(self, calls: int) -> None:
        value = max(int(calls), 0)
        self._diversity_calls_int = value
        self.diversity_calls.fill_(value)

    def set_horizon_calls(self, calls: int) -> None:
        value = max(int(calls), 0)
        self._horizon_calls_int = value
        self.horizon_calls.fill_(value)

    def _step_float(self, step: int | torch.Tensor | None = None) -> float:
        if step is None or step is self.training_step:
            return float(self._training_step_int)
        if torch.is_tensor(step):
            # Internal callers pass the registered training_step buffer. External
            # tensor steps are rare control-plane inputs and may synchronize.
            return float(step.detach().cpu().item())
        return float(step)

    def manager_unimix(self, step: int | torch.Tensor | None = None) -> float:
        s = self.settings
        fraction = min(max(self._step_float(step) / s.manager_unimix_decay_steps, 0.0), 1.0)
        return s.manager_unimix_initial + fraction * (
            s.manager_unimix_final - s.manager_unimix_initial
        )

    def worker_scale(self, step: int | torch.Tensor | None = None) -> float:
        s = self.settings
        x = self._step_float(step)
        if x <= s.worker_scale_warmup_steps:
            return s.worker_scale_initial
        if x >= s.worker_scale_full_steps:
            return s.worker_scale_max
        fraction = (x - s.worker_scale_warmup_steps) / max(
            float(s.worker_scale_full_steps - s.worker_scale_warmup_steps), 1.0
        )
        return s.worker_scale_initial + fraction * (
            s.worker_scale_max - s.worker_scale_initial
        )

    def worker_pg_blend(self, step: int | torch.Tensor | None = None) -> float:
        s = self.settings
        x = self._step_float(step)
        if x <= s.worker_pg_warmup_steps:
            return 0.0
        if x >= s.worker_pg_full_steps:
            return 1.0
        return (x - s.worker_pg_warmup_steps) / max(
            float(s.worker_pg_full_steps - s.worker_pg_warmup_steps), 1.0
        )

    def commitment_reselect_probability(
        self, step: int | torch.Tensor | None = None
    ) -> float:
        s = self.settings
        x = self._step_float(step)
        if x <= s.commitment_warmup_steps:
            return s.commitment_reselect_initial
        if x >= s.commitment_full_steps:
            return s.commitment_reselect_final
        fraction = (x - s.commitment_warmup_steps) / max(
            float(s.commitment_full_steps - s.commitment_warmup_steps), 1.0
        )
        return s.commitment_reselect_initial + fraction * (
            s.commitment_reselect_final - s.commitment_reselect_initial
        )

    def world_model_grad_scale(
        self, step: int | torch.Tensor | None = None
    ) -> float:
        s = self.settings
        x = self._step_float(step)
        if x <= s.world_model_grad_warmup_steps:
            return s.world_model_grad_scale_initial
        if x >= s.world_model_grad_full_steps:
            return s.world_model_grad_scale_final
        fraction = (x - s.world_model_grad_warmup_steps) / max(
            float(s.world_model_grad_full_steps - s.world_model_grad_warmup_steps),
            1.0,
        )
        return s.world_model_grad_scale_initial + fraction * (
            s.world_model_grad_scale_final - s.world_model_grad_scale_initial
        )

    def active_imagination_horizon_range(
        self, step: int | torch.Tensor | None = None
    ) -> tuple[int, int]:
        """Return the inclusive horizon range active at ``step``.

        Keeping this calculation separate from the round-robin sampler makes
        the schedule directly auditable and prevents tests/metrics from
        advancing the checkpointed horizon counter.
        """
        s = self.settings
        fraction = min(
            max(self._step_float(step) / float(s.imag_horizon_ramp_steps), 0.0),
            1.0,
        )
        active_max = int(round(
            s.imag_horizon_initial_max
            + fraction * (s.imag_horizon_final_max - s.imag_horizon_initial_max)
        ))
        active_min = max(1, active_max - s.imag_horizon_window + 1)
        return int(active_min), int(active_max)

    def next_imagination_horizon(self) -> int:
        active_min, active_max = self.active_imagination_horizon_range()
        span = active_max - active_min + 1
        horizon = active_min + (self._horizon_calls_int % span)
        self.set_horizon_calls(self._horizon_calls_int + 1)
        return int(horizon)

    def manager_pg_blend(self, step: int | torch.Tensor | None = None) -> float:
        """Keep manager task-PG off until option workers have causal effect."""
        s = self.settings
        x = self._step_float(step)
        if x <= s.manager_pg_warmup_steps:
            return 0.0
        if x >= s.manager_pg_full_steps:
            return 1.0
        return (x - s.manager_pg_warmup_steps) / max(
            float(s.manager_pg_full_steps - s.manager_pg_warmup_steps), 1.0
        )

    def termination_blend(self, step: int | torch.Tensor | None = None) -> float:
        s = self.settings
        x = self._step_float(step)
        if x <= s.termination_warmup_steps:
            return 0.0
        if x >= s.termination_full_steps:
            return 1.0
        return (x - s.termination_warmup_steps) / max(
            float(s.termination_full_steps - s.termination_warmup_steps), 1.0
        )

    def termination_probability_cap(
        self, step: int | torch.Tensor | None = None
    ) -> float:
        """Smoothly relax the execution cap after learned beta is active."""
        s = self.settings
        x = self._step_float(step)
        if x <= s.termination_full_steps:
            return s.termination_max_probability_during_ramp
        if x >= s.termination_cap_full_steps:
            return s.termination_max_probability_final
        fraction = (x - s.termination_full_steps) / max(
            float(s.termination_cap_full_steps - s.termination_full_steps), 1.0
        )
        return s.termination_max_probability_during_ramp + fraction * (
            s.termination_max_probability_final
            - s.termination_max_probability_during_ramp
        )

    def option_group(self, option: torch.Tensor) -> torch.Tensor:
        return option.to(dtype=torch.long) % int(self.settings.source_manager_group_count)

    def option_slot(self, option: torch.Tensor) -> torch.Tensor:
        return option.to(dtype=torch.long) // int(self.settings.source_manager_group_count)

    def source_group(self, feat: torch.Tensor, step: int | torch.Tensor | None = None) -> torch.Tensor:
        """Deterministic frozen Tactical-v1.2 group used by real and imagined execution."""
        return self.manager_group_probs(feat, step).argmax(dim=-1)

    def interruption_mask(
        self,
        feat: torch.Tensor,
        option: torch.Tensor,
        age: torch.Tensor,
        has_option: torch.Tensor | None = None,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exact deterministic interruption event used by the v9 SMDP target.

        An option must return control when its carried source group disagrees with
        the current frozen Tactical selector or when its action count reaches the
        fixed maximum duration. This function is shared by execution and critic
        bootstrapping so the Bellman target cannot value an unreachable continue
        transition.
        """
        option = option.to(device=feat.device, dtype=torch.long)
        age = age.to(device=feat.device, dtype=torch.long)
        mask = (self.option_group(option) != self.source_group(feat, step)) | (
            age >= int(self.settings.max_duration)
        )
        if has_option is not None:
            mask = mask & has_option.to(device=feat.device, dtype=torch.bool)
        return mask

    def q_by_group(self, q_all: torch.Tensor) -> torch.Tensor:
        """Convert option order [g0s0,g1s0,g0s1,...] to [...,group,slot]."""
        if q_all.shape[-1] != self.num_options:
            raise ValueError("option-value dimension mismatch")
        groups = int(self.settings.source_manager_group_count)
        return q_all.reshape(*q_all.shape[:-1], self.slots_per_group, groups).transpose(-1, -2)

    def slot_probs_for_group(
        self,
        feat: torch.Tensor,
        group: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = self.manager_slot_probs(feat, step)
        group = group.to(device=feat.device, dtype=torch.long)
        return probs.gather(
            -2,
            group.unsqueeze(-1).unsqueeze(-1).expand(*group.shape, 1, self.slots_per_group),
        ).squeeze(-2)

    def switch_value_for_source_group(
        self,
        feat: torch.Tensor,
        q_all_age0: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expected option value over only the slots executable in the current group."""
        group = self.source_group(feat, step)
        slot_probs = self.slot_probs_for_group(feat, group, step)
        q_group = self.q_by_group(q_all_age0).gather(
            -2,
            group.unsqueeze(-1).unsqueeze(-1).expand(*group.shape, 1, self.slots_per_group),
        ).squeeze(-2)
        return (slot_probs.float() * q_group.float()).sum(dim=-1)

    def slot_unlock_step(self, slot_index: int) -> int:
        if slot_index <= 0:
            return 0
        s = self.settings
        return int(
            s.slot_pair_unlock_initial_steps
            + (slot_index - 1) * s.slot_pair_unlock_interval_steps
        )

    def slot_gate_by_slot(
        self, step: int | torch.Tensor | None = None
    ) -> torch.Tensor:
        """All eight capacity slots are available from step zero.

        Safety comes from immutable source anchors, source-group interruption,
        and zero-initialized bounded child deltas—not from hiding capacity.
        """
        return torch.ones(
            self.slots_per_group,
            dtype=torch.float32,
            device=self.training_step.device,
        )

    def slot_gate_by_option(
        self, step: int | torch.Tensor | None = None
    ) -> torch.Tensor:
        gates = self.slot_gate_by_slot(step)
        slot = torch.arange(
            self.num_options, device=gates.device, dtype=torch.long
        ) // int(self.settings.source_manager_group_count)
        return gates.index_select(0, slot)

    def slot_pg_blend_for_option(
        self,
        option: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Identity availability for slot-manager gradients.

        Worker learning has its own single PG schedule. Returning one here avoids
        the previous accidental double-gating where both the policy output and
        the score-function loss were multiplied by the same warm-up fraction.
        """
        del step
        return torch.ones_like(option, dtype=torch.float32)

    def slot_delta_scale_by_option(
        self,
        option: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        del step
        slot = self.option_slot(option)
        # Anchor slots are exact source policies for the full run. Child output
        # layers initialize at exact zero, so a fixed bound preserves migration
        # while leaving the worker PG warm-up as the sole learning schedule.
        return torch.where(
            slot == 0,
            torch.zeros_like(slot, dtype=torch.float32),
            torch.full_like(
                slot, float(self.settings.slot_delta_scale_max), dtype=torch.float32
            ),
        )

    def manager_group_logits(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[-1] != self.feature_dim:
            raise ValueError("manager feature dimension mismatch")
        return self.manager_group(feat.float())

    def manager_group_probs(
        self,
        feat: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = self.manager_group_logits(feat).softmax(dim=-1)
        unimix = self.manager_unimix(step)
        groups = int(self.settings.source_manager_group_count)
        return (1.0 - unimix) * probs + unimix / groups

    def manager_slot_logits(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[-1] != self.feature_dim:
            raise ValueError("manager feature dimension mismatch")
        raw = self.manager_slot(feat.float())
        # Option layout is [g0s0, g1s0, g0s1, g1s1, ...].
        return raw.reshape(
            *raw.shape[:-1], self.slots_per_group,
            int(self.settings.source_manager_group_count),
        ).transpose(-1, -2)

    def manager_slot_probs(
        self,
        feat: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.manager_slot_logits(feat)
        gates = self.slot_gate_by_slot(step).to(
            device=logits.device, dtype=logits.dtype
        )
        view = gates.view(*([1] * (logits.ndim - 1)), self.slots_per_group)
        available = view > 0.0
        gated_logits = logits + view.clamp_min(1.0e-12).log()
        gated_logits = gated_logits.masked_fill(~available, -1.0e9)
        probs = gated_logits.softmax(dim=-1)
        # Keep a fixed floor on slot zero, the immutable source anchor. The
        # transformation remains differentiable through the remaining mass and
        # permits a useful child to become the deterministic argmax (up to 1-floor).
        anchor_floor = float(self.settings.slot_anchor_floor)
        if anchor_floor:
            anchor = torch.zeros_like(probs)
            anchor[..., 0] = 1.0
            probs = anchor_floor * anchor + (1.0 - anchor_floor) * probs
        unimix = float(self.settings.slot_manager_unimix)
        if unimix:
            uniform = available.to(logits.dtype)
            uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
            probs = (1.0 - unimix) * probs + unimix * uniform
        return probs

    def manager_logits(self, feat: torch.Tensor) -> torch.Tensor:
        # Compatibility helper: full option log-probabilities are valid logits.
        return self.manager_probs(feat).clamp_min(1.0e-12).log()

    def manager_probs(
        self,
        feat: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        group = self.manager_group_probs(feat, step)
        slot = self.manager_slot_probs(feat, step)
        full = group.unsqueeze(-1) * slot
        # [.., group, slot] -> [.., slot, group] -> option index order.
        return full.transpose(-1, -2).reshape(*feat.shape[:-1], self.num_options)

    def manager_log_prob_components(
        self,
        feat: torch.Tensor,
        option: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        group_probs = self.manager_group_probs(feat, step)
        slot_probs = self.manager_slot_probs(feat, step)
        group = self.option_group(option).to(device=feat.device)
        slot = self.option_slot(option).to(device=feat.device)
        group_dist = Categorical(probs=group_probs)
        selected_group_slot_probs = slot_probs.gather(
            -2,
            group.unsqueeze(-1).unsqueeze(-1).expand(
                *group.shape, 1, self.slots_per_group
            ),
        ).squeeze(-2)
        slot_dist = Categorical(probs=selected_group_slot_probs)
        return (
            group_dist.log_prob(group),
            slot_dist.log_prob(slot),
            group_dist.entropy(),
            slot_dist.entropy(),
            self.slot_pg_blend_for_option(option, step).to(feat.device),
        )

    def manager_dist(
        self,
        feat: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> Categorical:
        return Categorical(probs=self.manager_probs(feat, step))

    def _all_group_uncentered_residuals(self, feat: torch.Tensor) -> torch.Tensor:
        leading = feat.shape[:-1]
        groups = int(self.settings.source_manager_group_count)
        feat_all = feat.unsqueeze(-2).expand(*leading, groups, self.feature_dim)
        ids = torch.arange(groups, device=feat.device, dtype=torch.long)
        ids = ids.view(*([1] * len(leading)), groups).expand(*leading, groups)
        emb = self.option_embedding(ids)
        raw = self.worker_residual(torch.cat([feat_all.float(), emb], dim=-1))
        cap = self.settings.max_abs_residual_logit
        return cap * torch.tanh(raw / cap)

    def _all_group_residual_logits(
        self,
        feat: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw = self._all_group_uncentered_residuals(feat)
        centered = raw - raw.mean(dim=-2, keepdim=True)
        cap = float(self.settings.max_abs_residual_logit)
        max_abs = centered.abs().amax(dim=-2, keepdim=True).clamp_min(1.0e-8)
        projection = torch.clamp(cap / max_abs, max=1.0)
        return self.worker_scale(step) * centered * projection

    def _all_slot_delta_logits(
        self,
        feat: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        leading = feat.shape[:-1]
        k = self.num_options
        feat_all = feat.unsqueeze(-2).expand(*leading, k, self.feature_dim)
        ids = torch.arange(k, device=feat.device, dtype=torch.long)
        ids = ids.view(*([1] * len(leading)), k).expand(*leading, k)
        emb = self.slot_embedding(ids)
        raw = self.slot_delta(torch.cat([feat_all.float(), emb], dim=-1))
        cap = float(self.settings.max_abs_residual_logit)
        bounded = cap * torch.tanh(raw / cap)
        scales = self.slot_delta_scale_by_option(ids, step).unsqueeze(-1)
        return bounded * scales

    def all_residual_logits(
        self,
        feat: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        group_residual = self._all_group_residual_logits(feat, step)
        group_index = (
            torch.arange(self.num_options, device=feat.device, dtype=torch.long)
            % int(self.settings.source_manager_group_count)
        )
        expanded_group = group_residual.index_select(-2, group_index)
        combined = expanded_group + self._all_slot_delta_logits(feat, step)
        cap = float(self.settings.max_abs_residual_logit)
        max_abs = combined.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
        projection = torch.clamp(cap / max_abs, max=1.0)
        return combined * projection

    def residual_logits(
        self,
        feat: torch.Tensor,
        option: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        option = option.to(device=feat.device, dtype=torch.long)
        if option.shape != feat.shape[:-1]:
            raise ValueError("option shape must match feature leading shape")
        all_residual = self.all_residual_logits(feat, step)
        index = option.unsqueeze(-1).unsqueeze(-1).expand(
            *option.shape, 1, self.action_logit_dim
        )
        return all_residual.gather(-2, index).squeeze(-2)

    def combine_logits(
        self,
        base_logits: torch.Tensor,
        feat: torch.Tensor,
        option: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return base_logits + self.residual_logits(feat, option, step).to(base_logits.dtype)

    def learned_termination_probability(
        self,
        feat: torch.Tensor,
        option: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
        option = option.to(device=feat.device, dtype=torch.long)
        age = age.to(device=feat.device, dtype=torch.long).clamp(
            0, self.settings.max_duration
        )
        oemb = self.termination_option_embedding(option)
        aemb = self.age_embedding(age)
        logits = self.termination(torch.cat([feat.float(), oemb, aemb], dim=-1))
        return logits.squeeze(-1).sigmoid()

    @staticmethod
    def _softplus_scalar(value: float) -> float:
        return max(value, 0.0) + math.log1p(math.exp(-abs(value)))

    def _bounded_probability_from_raw_scalar(
        self, raw_probability: float, cap: float
    ) -> float:
        unimix = float(self.settings.termination_unimix)
        probability = (1.0 - unimix) * float(raw_probability) + 0.5 * unimix
        temperature = float(self.settings.termination_soft_cap_temperature)
        return probability - temperature * self._softplus_scalar(
            (probability - float(cap)) / temperature
        )

    def bounded_learned_termination_probability(
        self,
        feat: torch.Tensor,
        option: torch.Tensor,
        age: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply unimix and a smooth approximation to ``min(beta, cap)``.

        Probabilities well below the cap remain effectively unchanged when the
        cap schedule relaxes. This avoids the old interval-rescaling bug where
        merely raising the cap increased every executed termination probability.
        """
        raw = self.learned_termination_probability(feat, option, age)
        unimix = float(self.settings.termination_unimix)
        probability = (1.0 - unimix) * raw + 0.5 * unimix
        cap = float(self.termination_probability_cap(step))
        temperature = float(self.settings.termination_soft_cap_temperature)
        if not 0.0 < temperature < cap:
            raise RuntimeError("termination soft-cap temperature must lie below cap")
        return probability - temperature * F.softplus(
            (probability - cap) / temperature
        )

    def effective_termination_probability(
        self,
        feat: torch.Tensor,
        option: torch.Tensor,
        age: torch.Tensor,
        step: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic fixed-duration execution for the v9 controlled comparison.

        Learned beta is deliberately disabled. Options are interrupted when the
        frozen source selector changes group, or at ``max_duration``.
        """
        age = age.to(device=feat.device, dtype=torch.long)
        forced_continue = age < self.settings.min_duration
        forced_terminate = age >= self.settings.max_duration
        eligible = (~forced_continue) & (~forced_terminate)
        effective = torch.where(
            forced_terminate,
            torch.ones_like(age, dtype=torch.float32),
            torch.zeros_like(age, dtype=torch.float32),
        )
        return effective, eligible, forced_continue, forced_terminate

    def step_option(
        self,
        feat: torch.Tensor,
        option: torch.Tensor,
        age: torch.Tensor,
        has_option: torch.Tensor,
        is_first: torch.Tensor,
        *,
        deterministic: bool,
        step: int | torch.Tensor | None = None,
        termination_uniform: torch.Tensor | None = None,
        manager_uniform: torch.Tensor | None = None,
        termination_hazard: torch.Tensor | None = None,
    ) -> OptionStep:
        """Interruptible call-and-return with a frozen reactive source group.

        The inherited two-way Tactical Mixture selector is consulted at every
        state. A carried option is interrupted immediately when that selector
        changes group. Within the current source group, the learned slot manager
        may carry one of four experts for at most ``max_duration`` actions.
        """
        leading = feat.shape[:-1]
        option = option.to(device=feat.device, dtype=torch.long).reshape(leading)
        age = age.to(device=feat.device, dtype=torch.long).reshape(leading)
        has_option = has_option.to(device=feat.device, dtype=torch.bool).reshape(leading)
        is_first = is_first.to(device=feat.device, dtype=torch.bool).reshape(leading)
        safe_option = option.clamp(0, self.num_options - 1)
        safe_age = age.clamp(0, self.settings.max_duration)

        # The source-group head is frozen after migration. Argmax routing keeps
        # collection/evaluation semantics identical and avoids a stochastic
        # group-switch mismatch.
        source_group = self.source_group(feat, step)
        carried_group = self.option_group(safe_option)
        reset = is_first | (~has_option)
        source_interrupt = (~reset) & (source_group != carried_group)
        forced_terminate = (~reset) & (safe_age >= self.settings.max_duration)
        terminated = source_interrupt | forced_terminate
        boundary = reset | terminated

        slot_probs_all = self.manager_slot_probs(feat, step)
        selected_slot_probs = slot_probs_all.gather(
            -2,
            source_group.unsqueeze(-1).unsqueeze(-1).expand(
                *source_group.shape, 1, self.slots_per_group
            ),
        ).squeeze(-2)
        slot_dist = Categorical(probs=selected_slot_probs)
        if deterministic:
            proposed_slot = selected_slot_probs.argmax(dim=-1)
        elif manager_uniform is None:
            proposed_slot = slot_dist.sample()
        else:
            cdf = selected_slot_probs.cumsum(dim=-1)
            proposed_slot = (manager_uniform.unsqueeze(-1) > cdf).sum(dim=-1)
            proposed_slot = proposed_slot.clamp_max(self.slots_per_group - 1)
        proposed = proposed_slot * int(self.settings.source_manager_group_count) + source_group

        selected = torch.where(boundary, proposed, safe_option)
        action_age = torch.where(boundary, torch.zeros_like(safe_age), safe_age)
        carry_age = action_age + 1
        selected_slot = self.option_slot(selected)
        selected_slot_log_prob = slot_dist.log_prob(selected_slot)
        manager_log_prob = torch.where(
            boundary, selected_slot_log_prob, torch.zeros_like(selected_slot_log_prob)
        )
        manager_entropy = torch.where(
            boundary, slot_dist.entropy(), torch.zeros_like(selected_slot_log_prob)
        )
        beta = forced_terminate.to(dtype=torch.float32)
        eligible = (~reset) & (~forced_terminate)
        return OptionStep(
            option=selected,
            action_age=action_age,
            carry_age=carry_age,
            has_option=torch.ones_like(has_option),
            option_started=boundary,
            option_terminated=terminated,
            termination_eligible=eligible,
            termination_probability=beta,
            previous_option=safe_option,
            previous_age=safe_age,
            manager_log_prob=manager_log_prob,
            manager_entropy=manager_entropy,
            carry_termination_hazard=torch.zeros(leading, device=feat.device),
        )

    @staticmethod
    def _masked_probs(
        logits: torch.Tensor,
        action_mask: torch.Tensor,
        active_mask: torch.Tensor,
        actor_shape: Sequence[int],
        unimix_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = len(tuple(actor_shape))
        c = int(tuple(actor_shape)[0])
        shaped = logits.float().reshape(*logits.shape[:-1], a, c)
        mask = action_mask.to(dtype=torch.bool, device=logits.device).reshape(shaped.shape)
        active = active_mask.to(dtype=torch.bool, device=logits.device).reshape(
            *shaped.shape[:-1], 1
        )
        # NOOP must be legal whenever the predicted mask is empty.
        empty = ~mask.any(dim=-1, keepdim=True)
        noop = torch.zeros_like(mask)
        noop[..., 0] = True
        mask = torch.where(empty, noop, mask)
        masked_logits = shaped.masked_fill(~mask, -1.0e9)
        probs = masked_logits.softmax(dim=-1)
        unimix = float(unimix_ratio)
        if unimix:
            uniform = mask.float() / mask.float().sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
            probs = (1.0 - unimix) * probs + unimix * uniform
        return probs, active

    def behaviour_statistics(
        self,
        feat: torch.Tensor,
        base_logits: torch.Tensor,
        reference_logits: torch.Tensor,
        selected_option: torch.Tensor,
        action_mask: torch.Tensor,
        active_mask: torch.Tensor,
        actor_shape: Sequence[int],
        state_weights: torch.Tensor | None = None,
        step: int | torch.Tensor | None = None,
        *,
        unimix_ratio: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        # Subsample deterministically across the complete flattened sequence.
        # This auxiliary path must not allocate all K pairs for the full B*T
        # batch, but it must also not bias diversity checks toward a prefix.
        flat_feat = feat.reshape(-1, feat.shape[-1])
        flat_base = base_logits.reshape(-1, base_logits.shape[-1])
        flat_reference = reference_logits.reshape(-1, reference_logits.shape[-1])
        flat_option = selected_option.reshape(-1)
        flat_mask = action_mask.reshape(-1, action_mask.shape[-2], action_mask.shape[-1])
        flat_active = active_mask.reshape(-1, active_mask.shape[-1])
        if state_weights is None:
            flat_state_weight = torch.ones(
                flat_feat.shape[0], device=feat.device, dtype=torch.float32
            )
        else:
            flat_state_weight = torch.broadcast_to(
                state_weights.float(), feat.shape[:-1]
            ).reshape(-1)
        total_states = flat_feat.shape[0]
        n = min(total_states, self.settings.max_diversity_states)
        if n < total_states:
            sample_index = torch.linspace(
                0, total_states - 1, n, device=feat.device
            ).round().long()
            flat_feat = flat_feat.index_select(0, sample_index)
            flat_base = flat_base.index_select(0, sample_index)
            flat_reference = flat_reference.index_select(0, sample_index)
            flat_option = flat_option.index_select(0, sample_index)
            flat_mask = flat_mask.index_select(0, sample_index)
            flat_active = flat_active.index_select(0, sample_index)
            flat_state_weight = flat_state_weight.index_select(0, sample_index)
        flat_state_weight = flat_state_weight.clamp_min(0.0)

        all_logits = flat_base.unsqueeze(-2) + self.all_residual_logits(flat_feat, step)
        selected_logits = all_logits.gather(
            -2,
            flat_option[:, None, None].expand(n, 1, self.action_logit_dim),
        ).squeeze(-2)
        reference_probs, active = self._masked_probs(
            flat_reference, flat_mask, flat_active, actor_shape, unimix_ratio
        )
        selected_probs, _ = self._masked_probs(
            selected_logits, flat_mask, flat_active, actor_shape, unimix_ratio
        )
        eps = 1.0e-8
        # Forward KL KL(source || live) is the correct behaviour-preserving
        # direction: dropping a source-supported action is strongly penalized,
        # whereas reverse KL can be mode-seeking and miss this failure.
        base_kl_per_agent = (
            reference_probs.clamp_min(eps)
            * (
                reference_probs.clamp_min(eps).log()
                - selected_probs.clamp_min(eps).log()
            )
        ).sum(-1).clamp_min(0.0)
        active_float = active.squeeze(-1).float()
        active_weight = active_float * flat_state_weight.unsqueeze(-1)
        denominator = active_weight.sum().clamp_min(1.0)
        base_kl_mean = (base_kl_per_agent * active_weight).sum() / denominator
        valid_kl = base_kl_per_agent[active_weight > 0.0]
        if valid_kl.numel() == 0:
            base_kl_max = base_kl_per_agent.sum() * 0.0
            base_kl_tail = base_kl_max
        else:
            base_kl_max = valid_kl.max()
            tail_count = max(
                1,
                int(math.ceil(
                    valid_kl.numel()
                    * float(self.settings.base_kl_tail_fraction)
                )),
            )
            base_kl_tail = torch.topk(
                valid_kl, k=min(tail_count, valid_kl.numel())
            ).values.mean()

        base_mode = reference_probs.argmax(dim=-1)
        selected_mode = selected_probs.argmax(dim=-1)
        action_flip_rate = (
            (base_mode != selected_mode).float() * active_weight
        ).sum() / denominator
        reference_confidence = reference_probs.max(dim=-1).values
        high_confidence = (
            reference_confidence
            >= float(self.settings.action_preservation_confidence)
        ).float() * active_weight
        confidence_den = high_confidence.sum().clamp_min(1.0)
        selected_reference_prob = selected_probs.gather(
            -1, base_mode.unsqueeze(-1)
        ).squeeze(-1)
        selected_other = selected_probs.masked_fill(
            torch.nn.functional.one_hot(
                base_mode, selected_probs.shape[-1]
            ).bool(),
            -1.0,
        ).max(dim=-1).values
        action_preservation_loss = (
            torch.relu(
                selected_other
                - selected_reference_prob
                + float(self.settings.action_preservation_margin)
            ).square()
            * high_confidence
        ).sum() / confidence_den
        high_confidence_flip_rate = (
            (base_mode != selected_mode).float() * high_confidence
        ).sum() / confidence_den

        all_probs = []
        for option_index in range(self.num_options):
            option_probs, _ = self._masked_probs(
                all_logits[:, option_index], flat_mask, flat_active, actor_shape,
                unimix_ratio,
            )
            all_probs.append(option_probs)
        pair_js = []
        pairs = []
        for i in range(self.num_options):
            for j in range(i + 1, self.num_options):
                pairs.append((i, j))
        # Rotate the sampled subset by training step so all pairs receive
        # diagnostics over time without evaluating an excessive number.
        if len(pairs) > self.settings.max_diversity_pairs:
            # Replay count caps at buffer capacity, so using environment step
            # alone would permanently freeze the pair subset late in training.
            # Rotate by a checkpointed per-update counter instead.
            offset = self._diversity_calls_int % len(pairs)
            self.set_diversity_calls(self._diversity_calls_int + 1)
            pairs = (pairs[offset:] + pairs[:offset])[: self.settings.max_diversity_pairs]
        for i, j in pairs:
            p = all_probs[i].clamp_min(eps)
            q = all_probs[j].clamp_min(eps)
            m = 0.5 * (p + q)
            js = 0.5 * (
                (p * (p.log() - m.log())).sum(-1)
                + (q * (q.log() - m.log())).sum(-1)
            )
            pair_js.append((js * active_weight).sum() / denominator)
        js_tensor = torch.stack(pair_js) if pair_js else torch.zeros(1, device=feat.device)
        js_mean = js_tensor.mean()
        js_target = torch.as_tensor(
            self.settings.action_diversity_target,
            device=js_mean.device,
            dtype=js_mean.dtype,
        )
        # Penalize every sampled duplicate pair rather than only the average.
        # A mean-only hinge can be satisfied by a few highly distinct pairs while
        # other options remain exact duplicates. Pair rotation gives all K choose
        # 2 pairs coverage over successive updates.
        pairwise_js_shortfall = torch.relu(js_target - js_tensor)
        diversity_loss = pairwise_js_shortfall.mean()

        # Magnitude and direction safeguards are computed only on actions that
        # can actually affect the environment. Invalid actions and padded/dead
        # agents must not inflate the residual guard or provide fake diversity.
        actor_count = len(tuple(actor_shape))
        action_count = int(tuple(actor_shape)[0])
        valid_mask = flat_mask.bool().reshape(n, actor_count, action_count)
        active_agents = flat_active.bool().reshape(n, actor_count, 1)
        empty = ~valid_mask.any(dim=-1, keepdim=True)
        noop = torch.zeros_like(valid_mask)
        noop[..., 0] = True
        valid_mask = torch.where(empty, noop, valid_mask)
        effective_action_mask = valid_mask & active_agents
        entry_weight = (
            effective_action_mask.float() * flat_state_weight[:, None, None]
        )
        entry_denominator = entry_weight.sum().clamp_min(1.0)

        residual = (selected_logits - flat_reference).float().reshape(
            n, actor_count, action_count
        )
        base_shaped = flat_reference.float().reshape(n, actor_count, action_count)
        # sqrt has an infinite derivative at exactly zero. Source-preserving
        # migration intentionally starts with zero policy deviation, so an
        # unregularized RMS would inject NaN gradients on the first update even
        # though the scalar metric is finite. Add epsilon *inside* the root.
        residual_mse = (residual.square() * entry_weight).sum() / entry_denominator
        base_mse = (base_shaped.square() * entry_weight).sum() / entry_denominator
        residual_rms = residual_mse.clamp_min(1.0e-12).sqrt() - 1.0e-6
        base_rms = base_mse.clamp_min(1.0e-12).sqrt().clamp_min(1.0e-6)
        residual_ratio = residual_rms / base_rms

        # Collapse-only directional guard. Positive cosine near one means two
        # options are learning the same valid-action residual direction. Opposite
        # directions are intentionally not penalized because they are distinct.
        residual_all = self.all_residual_logits(flat_feat, step).float().reshape(
            n, self.num_options, actor_count, action_count
        )
        residual_all = (
            residual_all * effective_action_mask[:, None].float()
        ).reshape(n, self.num_options, -1)
        normalized = F.normalize(residual_all, dim=-1, eps=1.0e-8)
        cosine = torch.einsum("nkd,njd->nkj", normalized, normalized)
        upper = torch.triu(
            torch.ones(
                self.num_options, self.num_options,
                device=cosine.device, dtype=torch.bool
            ), diagonal=1
        )
        cosine_pairs = cosine[:, upper]
        cosine_weight = flat_state_weight[:, None]
        cosine_excess = torch.relu(
            cosine_pairs - float(self.settings.residual_cosine_target)
        )
        residual_cosine_loss = (
            cosine_excess.square() * cosine_weight
        ).sum() / (cosine_weight.sum() * max(cosine_pairs.shape[-1], 1)).clamp_min(1.0)
        weighted_cosine_mean = (
            cosine_pairs * cosine_weight
        ).sum() / (cosine_weight.sum() * max(cosine_pairs.shape[-1], 1)).clamp_min(1.0)
        residual_duplicate_fraction = (
            (cosine_pairs > float(self.settings.residual_cosine_target)).float()
            * cosine_weight
        ).sum() / (cosine_weight.sum() * max(cosine_pairs.shape[-1], 1)).clamp_min(1.0)
        residual_guard = torch.relu(
            residual_ratio
            - torch.as_tensor(
                self.settings.max_residual_to_base,
                device=residual_ratio.device,
                dtype=residual_ratio.dtype,
            )
        ).square()
        mean_kl_excess = torch.relu(
            base_kl_mean
            - torch.as_tensor(
                self.settings.base_kl_target,
                device=base_kl_mean.device,
                dtype=base_kl_mean.dtype,
            )
        ).square()
        tail_kl_excess = torch.relu(
            base_kl_tail
            - torch.as_tensor(
                self.settings.base_kl_tail_target,
                device=base_kl_tail.device,
                dtype=base_kl_tail.dtype,
            )
        ).square()
        # Distillation is always active. A pure hinge is zero throughout the
        # nominal trust region and can allow many small critical-state changes
        # to accumulate before any restoring gradient appears.
        base_kl_distillation = base_kl_mean + (
            float(self.settings.base_kl_tail_relative_scale) * base_kl_tail
        )
        base_kl_guard = mean_kl_excess + (
            float(self.settings.base_kl_tail_relative_scale) * tail_kl_excess
        )
        base_kl_loss = base_kl_distillation + base_kl_guard
        return {
            "base_kl_mean": base_kl_mean,
            "base_kl_tail": base_kl_tail,
            "base_kl_max": base_kl_max,
            "base_kl_loss": base_kl_loss,
            "action_flip_rate": action_flip_rate,
            "high_confidence_flip_rate": high_confidence_flip_rate,
            "action_preservation_loss": action_preservation_loss,
            "js_mean": js_mean,
            "js_min": js_tensor.min(),
            "js_max": js_tensor.max(),
            "diversity_loss": diversity_loss,
            "js_shortfall_fraction": (js_tensor < js_target).float().mean(),
            "residual_rms": residual_rms,
            "base_rms": base_rms,
            "residual_ratio": residual_ratio,
            "residual_guard_loss": residual_guard,
            "residual_cosine_loss": residual_cosine_loss,
            "residual_cosine_mean": weighted_cosine_mean,
            "residual_duplicate_fraction": residual_duplicate_fraction,
            "duplicate_pair_fraction": (js_tensor < 1.0e-4).float().mean(),
        }

    def grouped_manager_probs(self, manager_probs: torch.Tensor) -> torch.Tensor:
        """Aggregate option probabilities by their migrated Tactical v1.2 mode."""
        if manager_probs.shape[-1] != self.num_options:
            raise ValueError("manager probability dimension mismatch")
        groups = int(self.settings.source_manager_group_count)
        group_index = (
            torch.arange(self.num_options, device=manager_probs.device) % groups
        )
        index = group_index.view(*([1] * (manager_probs.ndim - 1)), self.num_options)
        index = index.expand(*manager_probs.shape[:-1], self.num_options)
        grouped = torch.zeros(
            *manager_probs.shape[:-1], groups,
            device=manager_probs.device, dtype=manager_probs.dtype,
        )
        grouped.scatter_add_(-1, index, manager_probs)
        return grouped

    def manager_source_statistics(
        self,
        manager_probs: torch.Tensor,
        source_manager_probs: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Trust region for the source two-way tactical routing decision."""
        live = self.grouped_manager_probs(manager_probs.float())
        source = self.grouped_manager_probs(source_manager_probs.float()).detach()
        if live.shape != source.shape:
            raise ValueError("live/source grouped manager shapes differ")
        if weights is None:
            w = torch.ones(live.shape[:-1], device=live.device, dtype=live.dtype)
        else:
            w = torch.broadcast_to(weights.float(), live.shape[:-1]).clamp_min(0.0)
        denominator = w.sum().clamp_min(1.0)
        eps = 1.0e-8
        per_state_kl = (
            source.clamp_min(eps)
            * (source.clamp_min(eps).log() - live.clamp_min(eps).log())
        ).sum(dim=-1).clamp_min(0.0)
        kl_mean = (per_state_kl * w).sum() / denominator
        valid_kl = per_state_kl[w > 0.0]
        if valid_kl.numel() == 0:
            kl_tail = per_state_kl.sum() * 0.0
            kl_max = kl_tail
        else:
            tail_count = max(
                1,
                int(math.ceil(
                    valid_kl.numel()
                    * float(self.settings.manager_group_kl_tail_fraction)
                )),
            )
            kl_tail = torch.topk(
                valid_kl, k=min(tail_count, valid_kl.numel())
            ).values.mean()
            kl_max = valid_kl.max()
        mean_excess = torch.relu(
            kl_mean - torch.as_tensor(
                self.settings.manager_group_kl_target,
                device=live.device, dtype=live.dtype,
            )
        ).square()
        tail_excess = torch.relu(
            kl_tail - torch.as_tensor(
                self.settings.manager_group_kl_tail_target,
                device=live.device, dtype=live.dtype,
            )
        ).square()
        # Always distill the original two-way routing distribution, then add
        # a squared guard outside the tighter trust-region thresholds.
        kl_distillation = kl_mean + (
            float(self.settings.manager_group_kl_tail_relative_scale) * kl_tail
        )
        kl_guard = mean_excess + (
            float(self.settings.manager_group_kl_tail_relative_scale)
            * tail_excess
        )
        kl_loss = kl_distillation + kl_guard

        source_mode = source.argmax(dim=-1)
        live_mode = live.argmax(dim=-1)
        source_confidence = source.max(dim=-1).values
        high_confidence = (
            source_confidence
            >= float(self.settings.manager_group_preservation_confidence)
        ).float() * w
        high_confidence_den = high_confidence.sum().clamp_min(1.0)
        live_source_prob = live.gather(
            -1, source_mode.unsqueeze(-1)
        ).squeeze(-1)
        live_other_prob = live.masked_fill(
            F.one_hot(source_mode, live.shape[-1]).bool(), -1.0
        ).max(dim=-1).values
        preservation_loss = (
            torch.relu(
                live_other_prob
                - live_source_prob
                + float(self.settings.manager_group_preservation_margin)
            ).square()
            * high_confidence
        ).sum() / high_confidence_den
        flip_rate = ((live_mode != source_mode).float() * w).sum() / denominator
        high_confidence_flip_rate = (
            (live_mode != source_mode).float() * high_confidence
        ).sum() / high_confidence_den
        return {
            "kl_mean": kl_mean,
            "kl_tail": kl_tail,
            "kl_max": kl_max,
            "kl_loss": kl_loss,
            "flip_rate": flip_rate,
            "high_confidence_flip_rate": high_confidence_flip_rate,
            "preservation_loss": preservation_loss,
            "live_group_probs": live,
            "source_group_probs": source,
        }

    def manager_statistics(
        self,
        manager_probs: torch.Tensor,
        sampled_option: torch.Tensor,
        boundary_mask: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        probs = manager_probs.float()
        boundary = boundary_mask.float()
        if weights is None:
            w = boundary
        else:
            w = boundary * torch.broadcast_to(weights.float(), boundary.shape)
        denominator = w.sum().clamp_min(1.0)
        marginal = (probs * w.unsqueeze(-1)).reshape(-1, self.num_options).sum(0)
        marginal = marginal / marginal.sum().clamp_min(1.0e-8)
        marginal_entropy = -(
            marginal.clamp_min(1.0e-8) * marginal.clamp_min(1.0e-8).log()
        ).sum()
        conditional_per_state = -(
            probs.clamp_min(1.0e-8) * probs.clamp_min(1.0e-8).log()
        ).sum(dim=-1)
        conditional_entropy = (
            conditional_per_state * w
        ).sum() / denominator
        mutual_information = (marginal_entropy - conditional_entropy).clamp_min(0.0)
        normalized_mutual_information = mutual_information / max(
            math.log(float(self.num_options)), 1.0e-8
        )
        effective = marginal_entropy.exp()
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
                self.settings.min_effective_options,
                device=probs.device,
                dtype=probs.dtype,
            )
            - effective
        )
        collapse_loss = max_excess.square() + (
            effective_shortfall / max(float(self.num_options), 1.0)
        ).square()
        mi_shortfall_loss = torch.relu(
            torch.as_tensor(
                self.settings.manager_mi_target_normalized,
                device=probs.device, dtype=probs.dtype,
            ) - normalized_mutual_information
        ).square()
        sampled = F.one_hot(
            sampled_option.long().clamp(0, self.num_options - 1),
            self.num_options,
        ).float()
        sampled_usage = (sampled * w.unsqueeze(-1)).reshape(-1, self.num_options).sum(0)
        sampled_usage = sampled_usage / sampled_usage.sum().clamp_min(1.0)
        return {
            "marginal": marginal,
            "sampled_usage": sampled_usage,
            "effective_count": effective,
            "marginal_entropy": marginal_entropy,
            "conditional_entropy": conditional_entropy,
            "mutual_information": mutual_information,
            "mutual_information_normalized": normalized_mutual_information,
            "usage_max": usage_max,
            "collapse_loss": collapse_loss,
            "mi_shortfall_loss": mi_shortfall_loss,
            "boundary_count": boundary.sum(),
        }

    def metadata(self) -> dict[str, Any]:
        s = self.settings
        return {
            "schema_version": self.SCHEMA_VERSION,
            "architecture": self.ARCHITECTURE,
            "enabled": bool(s.enabled),
            "num_options": s.num_options,
            "option_embedding_dim": s.option_embedding_dim,
            "age_embedding_dim": s.age_embedding_dim,
            "hidden_dim": s.hidden_dim,
            "min_duration": s.min_duration,
            "max_duration": s.max_duration,
            "feature_dim": self.feature_dim,
            "action_logit_dim": self.action_logit_dim,
            "freeze_base_actor": s.freeze_base_actor,
            "freeze_feature_adapter": s.freeze_feature_adapter,
            "worker_scale_initial": s.worker_scale_initial,
            "worker_scale_max": s.worker_scale_max,
            "slot_delta_scale_max": s.slot_delta_scale_max,
            "worker_pg_warmup_steps": s.worker_pg_warmup_steps,
            "worker_pg_full_steps": s.worker_pg_full_steps,
            "manager_unimix_initial": s.manager_unimix_initial,
            "manager_unimix_final": s.manager_unimix_final,
            "slot_manager_unimix": s.slot_manager_unimix,
            "slot_anchor_floor": s.slot_anchor_floor,
            "slot_pair_unlock_initial_steps": s.slot_pair_unlock_initial_steps,
            "slot_pair_unlock_interval_steps": s.slot_pair_unlock_interval_steps,
            "slot_unlock_ramp_steps": s.slot_unlock_ramp_steps,
            "slot_pg_ramp_steps": s.slot_pg_ramp_steps,
            "manager_pg_warmup_steps": s.manager_pg_warmup_steps,
            "manager_pg_full_steps": s.manager_pg_full_steps,
            "commitment_warmup_steps": s.commitment_warmup_steps,
            "commitment_full_steps": s.commitment_full_steps,
            "termination_warmup_steps": s.termination_warmup_steps,
            "termination_full_steps": s.termination_full_steps,
            "termination_cap_full_steps": s.termination_cap_full_steps,
            "termination_soft_cap_temperature": s.termination_soft_cap_temperature,
            "world_model_grad_scale_initial": s.world_model_grad_scale_initial,
            "world_model_grad_scale_final": s.world_model_grad_scale_final,
            "base_kl_target": s.base_kl_target,
            "base_kl_tail_target": s.base_kl_tail_target,
            "action_preservation_scale": s.action_preservation_scale,
            "source_manager_group_count": s.source_manager_group_count,
            "manager_group_kl_target": s.manager_group_kl_target,
            "manager_group_kl_tail_target": s.manager_group_kl_tail_target,
            "manager_group_kl_scale": s.manager_group_kl_scale,
            "manager_group_preservation_scale": s.manager_group_preservation_scale,
            "manager_collapse_scale": s.manager_collapse_scale,
            "manager_mi_scale": s.manager_mi_scale,
            "action_diversity_scale": s.action_diversity_scale,
            "residual_cosine_scale": s.residual_cosine_scale,
            "option_critic_consistency_scale": s.option_critic_consistency_scale,
            "imag_horizon_initial_max": s.imag_horizon_initial_max,
            "imag_horizon_final_max": s.imag_horizon_final_max,
            "imag_horizon_window": s.imag_horizon_window,
            "imag_horizon_ramp_steps": s.imag_horizon_ramp_steps,
            "eval_sample_termination": s.eval_sample_termination,
            "eval_termination_hazard_threshold": s.eval_termination_hazard_threshold,
        }
