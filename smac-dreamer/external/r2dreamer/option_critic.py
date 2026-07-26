"""Option-conditioned critic and mathematically explicit Option-Critic losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from hierarchical_options import HierarchicalOptionSettings


class OptionCritic(nn.Module):
    """Q_Omega(h, option, age) with a zero-initialized residual head.

    The critic can be initialized from an inherited state-value estimate by
    passing ``base_value`` to :meth:`q_all`. The learned residual is initialized
    to zero, so every option initially shares the inherited value estimate.
    """

    def __init__(self, feature_dim: int, config: Any) -> None:
        super().__init__()
        self.settings = HierarchicalOptionSettings.from_config(config)
        self.settings.validate()
        s = self.settings
        self.feature_dim = int(feature_dim)
        self.option_embedding = nn.Embedding(s.num_options, s.option_embedding_dim)
        self.age_embedding = nn.Embedding(s.max_duration + 1, s.age_embedding_dim)
        self.trunk = nn.Sequential(
            nn.Linear(
                self.feature_dim + s.option_embedding_dim + s.age_embedding_dim,
                s.hidden_dim,
            ),
            nn.ELU(),
            nn.Linear(s.hidden_dim, 1),
        )
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

    def q_all(
        self,
        feat: torch.Tensor,
        age: torch.Tensor,
        base_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        leading = feat.shape[:-1]
        k = self.settings.num_options
        age = age.to(device=feat.device, dtype=torch.long).clamp(
            0, self.settings.max_duration
        )
        feat_all = feat.unsqueeze(-2).expand(*leading, k, self.feature_dim)
        option = torch.arange(k, device=feat.device, dtype=torch.long)
        option = option.view(*([1] * len(leading)), k).expand(*leading, k)
        age_all = age.unsqueeze(-1).expand(*leading, k)
        residual = self.trunk(
            torch.cat(
                [
                    feat_all.float(),
                    self.option_embedding(option),
                    self.age_embedding(age_all),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        if base_value is None:
            return residual
        base = base_value.float()
        if base.shape == (*leading, 1):
            base = base.squeeze(-1)
        if base.shape != leading:
            raise ValueError(
                f"base value shape {tuple(base.shape)} != feature leading shape {leading}"
            )
        return base.unsqueeze(-1) + residual

    def q_selected(
        self,
        feat: torch.Tensor,
        option: torch.Tensor,
        age: torch.Tensor,
        base_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_all(feat, age, base_value)
        option = option.to(device=feat.device, dtype=torch.long)
        return q.gather(-1, option.unsqueeze(-1)).squeeze(-1)



def call_and_return_bootstrap(
    continue_value: torch.Tensor,
    switch_value: torch.Tensor,
    termination_probability: torch.Tensor,
) -> torch.Tensor:
    """Expected next-state value under call-and-return execution.

    With probability ``1-beta`` the current option continues; with probability
    ``beta`` it terminates and control returns to the manager.
    """
    if not (
        continue_value.shape
        == switch_value.shape
        == termination_probability.shape
    ):
        raise ValueError("call-and-return bootstrap tensors must have equal shape")
    # The hierarchy constructs beta through sigmoid/unimix/clamping, so it is
    # already in [0, 1]. Avoid Python truth tests on accelerator tensors here:
    # they would introduce a device synchronization in every learner update.
    beta = termination_probability.float()
    return torch.lerp(continue_value.float(), switch_value.float(), beta)


def option_lambda_return(
    reward_next: torch.Tensor,
    continuation_next: torch.Tensor,
    bootstrap_next: torch.Tensor,
    *,
    discount: float,
    lambda_: float,
) -> torch.Tensor:
    """Call-and-return lambda return for temporally extended options.

    ``reward_next[:, t]`` and ``continuation_next[:, t]`` correspond to the
    state reached after primitive action ``t``. ``bootstrap_next[:, t]`` is
    the expected call-and-return value at that next state: continue the current
    option with probability ``1-beta`` or switch through the manager with
    probability ``beta``.
    """
    if not (reward_next.shape == continuation_next.shape == bootstrap_next.shape):
        raise ValueError("option lambda-return tensors must have equal shape")
    if reward_next.ndim < 2:
        raise ValueError("option lambda-return expects a time dimension")
    live = continuation_next.float() * float(discount)
    bootstrap = bootstrap_next.detach().float()
    reward = reward_next.float()
    out = torch.empty_like(reward)
    next_return = bootstrap[:, -1]
    for index in reversed(range(reward.shape[1])):
        next_return = reward[:, index] + live[:, index] * (
            (1.0 - float(lambda_)) * bootstrap[:, index]
            + float(lambda_) * next_return
        )
        out[:, index] = next_return
    return out


def normalized_advantage(
    target: torch.Tensor,
    baseline: torch.Tensor,
    return_scale: torch.Tensor,
) -> torch.Tensor:
    """Scale-invariant detached-baseline advantage used by both actors."""
    scale = return_scale.detach().float().clamp_min(1.0)
    while scale.ndim < target.ndim:
        scale = scale.unsqueeze(-1)
    return (target.float() - baseline.detach().float()) / scale

def manager_value(manager_probs: torch.Tensor, q_all_age0: torch.Tensor) -> torch.Tensor:
    if manager_probs.shape != q_all_age0.shape:
        raise ValueError("manager probabilities and option values must have equal shape")
    return (manager_probs.float() * q_all_age0.float()).sum(dim=-1)


def option_critic_loss(
    q_selected: torch.Tensor,
    target_return: torch.Tensor,
    return_scale: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Robust option-value regression in detached return-scale units."""
    scale = return_scale.detach().float().clamp_min(1.0)
    while scale.ndim < q_selected.ndim:
        scale = scale.unsqueeze(-1)
    error = (q_selected.float() - target_return.detach().float()) / scale
    per_item = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none")
    w = torch.broadcast_to(weights.detach().float(), per_item.shape)
    return (per_item * w).sum() / w.sum().clamp_min(1.0)




def within_group_option_consistency_loss(
    q_all_age0: torch.Tensor,
    *,
    source_group_count: int,
    return_scale: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Keep child option values tied to their anchor while policies are identical.

    Option order is ``[g0s0, g1s0, g0s1, g1s1, ...]``. Random identity
    assignment gives statistically identical children different sampled returns;
    without this guard, the slot manager can later exploit critic noise before a
    child has acquired a causal action difference. The anchor target is detached,
    so this regularizer cannot move the source-backed anchor value.
    """
    if q_all_age0.ndim < 2:
        raise ValueError("q_all_age0 must contain an option dimension")
    groups = int(source_group_count)
    if groups < 2 or q_all_age0.shape[-1] % groups != 0:
        raise ValueError("option count must be divisible by source_group_count")
    slots = q_all_age0.shape[-1] // groups
    if slots < 2:
        return q_all_age0.sum() * 0.0
    grouped = q_all_age0.reshape(
        *q_all_age0.shape[:-1], slots, groups
    ).transpose(-1, -2)
    anchor = grouped[..., :1].detach()
    children = grouped[..., 1:]
    scale = return_scale.detach().float().clamp_min(1.0)
    while scale.ndim < children.ndim:
        scale = scale.unsqueeze(-1)
    error = (children.float() - anchor.float()) / scale
    per_item = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none")
    w = weights.detach().float()
    while w.ndim < per_item.ndim:
        w = w.unsqueeze(-1)
    w = torch.broadcast_to(w, per_item.shape)
    return (per_item * w).sum() / w.sum().clamp_min(1.0)


def worker_policy_loss(
    log_prob: torch.Tensor,
    entropy: torch.Tensor,
    advantage: torch.Tensor,
    weights: torch.Tensor,
    *,
    pg_scale: float,
    entropy_scale: float,
) -> torch.Tensor:
    objective = pg_scale * log_prob * advantage.detach() + entropy_scale * entropy
    w = torch.broadcast_to(weights.detach().float(), objective.shape)
    return -(objective.float() * w).sum() / w.sum().clamp_min(1.0)


def manager_policy_loss(
    manager_log_prob: torch.Tensor,
    manager_entropy: torch.Tensor,
    manager_advantage: torch.Tensor,
    boundary_mask: torch.Tensor,
    weights: torch.Tensor,
    *,
    pg_scale: float,
    entropy_scale: float,
) -> torch.Tensor:
    """Manager gradients are applied only at actual option boundaries."""
    boundary = boundary_mask.detach().float()
    w = torch.broadcast_to(weights.detach().float(), boundary.shape) * boundary
    objective = (
        pg_scale * manager_log_prob * manager_advantage.detach()
        + entropy_scale * manager_entropy
    )
    return -(objective.float() * w).sum() / w.sum().clamp_min(1.0)


def termination_loss(
    termination_probability: torch.Tensor,
    continue_value: torch.Tensor,
    switch_value: torch.Tensor,
    eligible_mask: torch.Tensor,
    weights: torch.Tensor,
    return_scale: torch.Tensor,
    *,
    normalized_margin: float,
    advantage_clip: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Option-Critic termination objective with a continuation margin.

    Minimizing ``beta_exec * (Q_continue - V_switch + margin)`` decreases the
    executed termination probability when the current option is better and
    increases it when switching is better. ``termination_probability`` must be
    the exact differentiable probability used by the call-and-return state
    machine (including warm-up blend, unimix, and execution cap). This preserves
    the correct chain rule and prevents the raw head from saturating invisibly
    behind a hard execution cap. The value advantage is detached so termination
    cannot manipulate its critic targets. Forced continuation and forced
    maximum-duration termination are excluded through ``eligible_mask``.
    """
    scale = return_scale.detach().float().clamp_min(1.0)
    while scale.ndim < continue_value.ndim:
        scale = scale.unsqueeze(-1)
    normalized_advantage = (
        continue_value.detach().float() - switch_value.detach().float()
    ) / scale
    signal = (normalized_advantage + float(normalized_margin)).clamp(
        -float(advantage_clip), float(advantage_clip)
    )
    eligible = eligible_mask.detach().float()
    w = torch.broadcast_to(weights.detach().float(), eligible.shape) * eligible
    loss = (
        termination_probability.float() * signal * w
    ).sum() / w.sum().clamp_min(1.0)
    return loss, normalized_advantage


def termination_gradient_sign_check() -> dict[str, float]:
    """Small executable proof of the intended termination gradient sign."""
    beta_logit = torch.tensor(0.0, requires_grad=True)
    beta = beta_logit.sigmoid()
    continue_better = beta * torch.tensor(1.0)
    continue_better.backward()
    grad_continue = float(beta_logit.grad)

    beta_logit2 = torch.tensor(0.0, requires_grad=True)
    beta2 = beta_logit2.sigmoid()
    switch_better = beta2 * torch.tensor(-1.0)
    switch_better.backward()
    grad_switch = float(beta_logit2.grad)
    return {
        "continue_better_grad": grad_continue,
        "switch_better_grad": grad_switch,
    }
