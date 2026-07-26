"""Integration helpers for call-and-return options inside R2-Dreamer.

These helpers keep the invasive edits to ``dreamer.py`` small. They are called
from patched construction, acting, and update sites while preserving JEPA's
primitive-action transition contract.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from tensordict import TensorDict

from hierarchical_options import HierarchicalOptionsPolicy
from option_critic import (
    OptionCritic,
    call_and_return_bootstrap,
    manager_policy_loss,
    manager_value,
    normalized_advantage,
    option_critic_loss,
    option_lambda_return,
    within_group_option_consistency_loss,
    termination_loss,
    worker_policy_loss,
)


def build_hierarchical_modules(agent: Any, config: Any) -> None:
    cfg = getattr(config, "hierarchical_options", None)
    agent.hierarchical_enabled = bool(
        getattr(cfg, "enabled", False) if cfg is not None else False
    )
    agent.hierarchical_options = None
    agent.option_critic = None
    agent._slow_option_critic = None
    agent._source_hierarchical_options = None
    agent._hierarchy_world_model_grad_hooks = []
    agent._option_slow_updates = 0
    if not agent.hierarchical_enabled:
        return
    if agent.world_model_backend != "jepa":
        raise NotImplementedError("hierarchical options are validated only for JEPA")
    if not agent.action_masking or not agent.act_discrete:
        raise ValueError("hierarchical options require masked discrete actions")
    if bool(getattr(config, "compile", False)):
        raise ValueError(
            "hierarchical options v9 requires config.compile=false; the auxiliary "
            "option state machine contains stochastic control flow"
        )

    agent.hierarchical_options = HierarchicalOptionsPolicy(
        agent.feat_size,
        sum(agent._actor_shape),
        cfg,
    )
    agent.option_critic = OptionCritic(agent.feat_size, cfg)
    # v9 safety contract: the inherited Tactical Mixture group selector and
    # source worker residual are immutable anchors. Only within-group slot
    # routing, child deltas, and critics are trainable.
    for parameter in agent.hierarchical_options.manager_group.parameters():
        parameter.requires_grad_(False)
    for parameter in agent.hierarchical_options.worker_residual.parameters():
        parameter.requires_grad_(False)
    for parameter in agent.hierarchical_options.option_embedding.parameters():
        parameter.requires_grad_(False)
    # Learned termination is deliberately absent from the controlled v9 run.
    # Freeze both the head and its private embedding so a zero configured loss
    # cannot be undermined by an accidental auxiliary path.
    for parameter in agent.hierarchical_options.termination.parameters():
        parameter.requires_grad_(False)
    for parameter in agent.hierarchical_options.termination_option_embedding.parameters():
        parameter.requires_grad_(False)
    agent._slow_option_critic = copy.deepcopy(agent.option_critic)
    for parameter in agent._slow_option_critic.parameters():
        parameter.requires_grad_(False)
    agent._slow_option_critic.train(False)

    # A permanent source-policy reference is registered in the agent state dict.
    # It is synchronized after Tactical v1.2 migration and never refreshed by
    # online updates. Trust-region losses therefore protect the actual inherited
    # tactical policy rather than the weaker primitive actor.
    agent._source_hierarchical_options = copy.deepcopy(agent.hierarchical_options)
    for parameter in agent._source_hierarchical_options.parameters():
        parameter.requires_grad_(False)
    agent._source_hierarchical_options.train(False)

    settings = agent.hierarchical_options.settings
    if settings.freeze_base_actor:
        for parameter in agent.actor.parameters():
            parameter.requires_grad_(False)
    if settings.freeze_feature_adapter:
        for parameter in agent.jepa_world_model.feature_adapter.parameters():
            parameter.requires_grad_(False)

    # Scale all live JEPA gradients without removing parameters from the graph.
    # This keeps backward valid while freezing the actor-facing representation
    # during migration preservation and allowing only slow adaptation later.
    for parameter in agent.jepa_world_model.parameters():
        if parameter.requires_grad:
            handle = parameter.register_hook(
                lambda grad, controller=agent.hierarchical_options: (
                    grad * controller.world_model_grad_scale()
                )
            )
            agent._hierarchy_world_model_grad_hooks.append(handle)



def apply_hierarchy_gradient_guards(agent: Any) -> None:
    """Fail-safe optimizer guard for the source JEPA representation.

    Gradient hooks scale the live world-model gradient, but a zero tensor can
    still be changed by optimizer momentum or decoupled weight decay. During the
    conservative frozen phase, clearing ``grad`` makes PyTorch optimizers skip
    those parameters entirely, preserving the source representation exactly.
    """
    if not getattr(agent, "hierarchical_enabled", False):
        return
    scale = agent.hierarchical_options.world_model_grad_scale()
    if scale <= 0.0:
        for parameter in agent.jepa_world_model.parameters():
            parameter.grad = None


def sync_source_hierarchy(agent: Any) -> None:
    if not getattr(agent, "hierarchical_enabled", False):
        return
    agent._source_hierarchical_options.load_state_dict(
        agent.hierarchical_options.state_dict(), strict=True
    )
    agent._source_hierarchical_options.set_training_step(0)
    agent._source_hierarchical_options.set_diversity_calls(0)
    agent._source_hierarchical_options.set_horizon_calls(0)
    for parameter in agent._source_hierarchical_options.parameters():
        parameter.requires_grad_(False)
    agent._source_hierarchical_options.train(False)


def clone_and_freeze_hierarchy(agent: Any) -> None:
    if not getattr(agent, "hierarchical_enabled", False):
        agent._frozen_hierarchical_options = None
        return
    frozen = copy.deepcopy(agent.hierarchical_options)
    for (source_name, source), (target_name, target) in zip(
        agent.hierarchical_options.named_parameters(),
        frozen.named_parameters(),
    ):
        if source_name != target_name:
            raise RuntimeError("hierarchical frozen-view parameter order mismatch")
        # Match R2-Dreamer's existing frozen actor/world-model semantics: this
        # is a no-grad *view of the current online policy*, not a target-network
        # snapshot. Sharing parameter storage lets real acting and imagined
        # collection immediately observe optimizer updates while the distinct
        # Parameter objects remain requires_grad=False. The permanent source
        # controller below is a true deep copy and must never share storage.
        target.data = source.data
        target.requires_grad_(False)
    for (source_name, source), (target_name, target) in zip(
        agent.hierarchical_options.named_buffers(),
        frozen.named_buffers(),
    ):
        if source_name != target_name:
            raise RuntimeError("hierarchical frozen-copy buffer order mismatch")
        target.data.copy_(source.data)
    frozen.train(False)
    agent._frozen_hierarchical_options = frozen
    if getattr(agent, "_source_hierarchical_options", None) is not None:
        agent._source_hierarchical_options.train(False)


def hierarchy_state_dict_fields(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    shape = (int(batch_size), 1)
    return {
        # Persistent call-and-return state carried to the next observation.
        "option_id": torch.zeros(shape, dtype=torch.long, device=device),
        "option_age": torch.zeros(shape, dtype=torch.long, device=device),
        "option_has": torch.zeros(shape, dtype=torch.bool, device=device),
        "option_termination_hazard": torch.zeros(
            shape, dtype=torch.float32, device=device
        ),
        # Pre-decision state aligned with the current posterior h_t. Replay
        # imagination must start from these fields, not from carry_age, or it
        # would evaluate termination twice at the same state.
        "option_before_id": torch.zeros(shape, dtype=torch.long, device=device),
        "option_before_age": torch.zeros(shape, dtype=torch.long, device=device),
        "option_before_has": torch.zeros(shape, dtype=torch.bool, device=device),
        # Age under which the primitive action selected at h_t executes.
        "option_action_age": torch.zeros(shape, dtype=torch.long, device=device),
        "option_started": torch.zeros(shape, dtype=torch.bool, device=device),
        "option_terminated": torch.zeros(shape, dtype=torch.bool, device=device),
        "option_termination_eligible": torch.zeros(
            shape, dtype=torch.bool, device=device
        ),
        "option_termination_prob": torch.zeros(
            shape, dtype=torch.float32, device=device
        ),
    }


def hierarchical_act_logits(
    agent: Any,
    feat: torch.Tensor,
    base_logits: torch.Tensor,
    state: TensorDict,
    obs: TensorDict,
    *,
    deterministic: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not agent.hierarchical_enabled:
        return base_logits, {}
    controller = agent._frozen_hierarchical_options
    is_first = obs["is_first"].reshape(feat.shape[:-1]) > 0.5
    before_option = state["option_id"].reshape(feat.shape[:-1])
    before_age = state["option_age"].reshape(feat.shape[:-1])
    before_has = state["option_has"].reshape(feat.shape[:-1])
    before_hazard = state["option_termination_hazard"].reshape(feat.shape[:-1])
    decision = controller.step_option(
        feat,
        before_option,
        before_age,
        before_has,
        is_first,
        deterministic=deterministic,
        step=controller.training_step,
        termination_hazard=before_hazard,
    )
    logits = controller.combine_logits(
        base_logits,
        feat,
        decision.option,
        controller.training_step,
    )
    fields = {
        "option_id": decision.option.unsqueeze(-1),
        "option_age": decision.carry_age.unsqueeze(-1),
        "option_has": decision.has_option.unsqueeze(-1),
        "option_termination_hazard": (
            decision.carry_termination_hazard.unsqueeze(-1)
        ),
        "option_before_id": before_option.unsqueeze(-1),
        "option_before_age": before_age.unsqueeze(-1),
        "option_before_has": before_has.unsqueeze(-1),
        "option_action_age": decision.action_age.unsqueeze(-1),
        "option_started": decision.option_started.unsqueeze(-1),
        "option_terminated": decision.option_terminated.unsqueeze(-1),
        "option_termination_eligible": decision.termination_eligible.unsqueeze(-1),
        "option_termination_prob": decision.termination_probability.unsqueeze(-1),
    }
    return logits, fields


@torch.no_grad()
def imagine_hierarchy(
    agent: Any,
    start: tuple[torch.Tensor, torch.Tensor],
    option_start: dict[str, torch.Tensor],
    horizon: int,
) -> dict[str, torch.Tensor]:
    from smacdreamer.masked_actions import MaskedMultiOneHotDist

    controller = agent._frozen_hierarchical_options
    stoch, deter = start
    option = option_start["option_id"].long()
    age = option_start["option_age"].long()
    has_option = option_start["option_has"].bool()
    is_first = option_start["is_first"].bool()

    output: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "feat",
            "action",
            "option",
            "action_age",
            "carry_age",
            "option_started",
            "option_terminated",
            "termination_eligible",
            "termination_probability",
            "previous_option",
            "previous_age",
            "manager_log_prob",
            "manager_entropy",
        )
    }

    for _ in range(int(horizon)):
        feat = agent._frozen_jepa_world_model.get_feat(stoch, deter)
        decision = controller.step_option(
            feat,
            option,
            age,
            has_option,
            is_first,
            deterministic=False,
            step=controller.training_step,
        )
        base_logits = agent._frozen_actor.last(agent._frozen_actor.mlp(feat))
        policy_logits = controller.combine_logits(
            base_logits,
            feat,
            decision.option,
            controller.training_step,
        )
        action_mask, active_mask = agent._predicted_action_mask(feat)
        action = MaskedMultiOneHotDist(
            policy_logits,
            action_mask,
            active_mask,
            agent._actor_shape,
            agent._actor_unimix,
        ).rsample()

        output["feat"].append(feat)
        output["action"].append(action)
        output["option"].append(decision.option)
        output["action_age"].append(decision.action_age)
        output["carry_age"].append(decision.carry_age)
        output["option_started"].append(decision.option_started)
        output["option_terminated"].append(decision.option_terminated)
        output["termination_eligible"].append(decision.termination_eligible)
        output["termination_probability"].append(decision.termination_probability)
        output["previous_option"].append(decision.previous_option)
        output["previous_age"].append(decision.previous_age)
        output["manager_log_prob"].append(decision.manager_log_prob)
        output["manager_entropy"].append(decision.manager_entropy)

        stoch, deter = agent._frozen_jepa_world_model.img_step(
            stoch, deter, action
        )
        option = decision.option
        age = decision.carry_age
        has_option = decision.has_option
        is_first = torch.zeros_like(is_first)

    return {key: torch.stack(values, dim=1) for key, values in output.items()}


def interruptible_option_bootstrap(
    controller: Any,
    critic: Any,
    feat: torch.Tensor,
    carried_option: torch.Tensor,
    next_age: torch.Tensor,
    base_value: torch.Tensor,
    step: int | torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact v9 SMDP bootstrap matching the deterministic execution state machine.

    Continuation is legal only while the frozen Tactical source group still agrees
    with the carried option and the fixed maximum duration has not been reached.
    At an interruption, control returns only to the four slots in the current
    source group; options in the other group are unreachable and must not enter the
    Bellman target or manager baseline.
    """
    continue_value = critic.q_selected(feat, carried_option, next_age, base_value)
    q_all_age0 = critic.q_all(feat, torch.zeros_like(next_age), base_value)
    switch_value = controller.switch_value_for_source_group(feat, q_all_age0, step)
    must_switch = controller.interruption_mask(feat, carried_option, next_age, step=step)
    bootstrap = torch.where(must_switch, switch_value, continue_value)
    return bootstrap, continue_value, switch_value, must_switch


def _return_scale(ret: torch.Tensor) -> torch.Tensor:
    flat = ret.detach().float().reshape(-1)
    if flat.numel() < 4:
        return flat.std(unbiased=False).clamp_min(1.0)
    low = torch.quantile(flat, 0.05)
    high = torch.quantile(flat, 0.95)
    return (high - low).clamp_min(1.0)


def hierarchical_auxiliary_loss(
    agent: Any,
    raw_data: TensorDict,
    post_stoch: torch.Tensor,
    post_deter: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute worker, manager, critic, and termination losses.

    This is called after the existing Dreamer gradient computation but before
    optimizer unscaling/stepping, so it augments rather than duplicates the
    world-model losses. The inherited actor and JEPA adapter are frozen.
    """
    if not agent.hierarchical_enabled:
        zero = post_deter.sum() * 0.0
        return zero, {}
    required = (
        "option_id",
        "option_age",
        "option_has",
        "option_before_id",
        "option_before_age",
        "option_before_has",
        "option_action_age",
        "option_started",
        "option_terminated",
        "option_termination_eligible",
        "option_termination_prob",
        "is_first",
        "is_last",
    )
    missing = [key for key in required if key not in raw_data.keys()]
    if missing:
        raise RuntimeError(f"hierarchical replay fields missing: {missing}")

    b, t = raw_data.shape
    n = b * t
    start = (
        post_stoch.reshape(n, *post_stoch.shape[2:]).detach(),
        post_deter.reshape(n, post_deter.shape[-1]).detach(),
    )
    option_start = {
        # These are the call-and-return fields that entered act() at h_t.
        # Using post-action carry fields here would make imagination decide
        # termination a second time at the same posterior state.
        "option_id": raw_data["option_before_id"].reshape(n, -1)[:, 0],
        "option_age": raw_data["option_before_age"].reshape(n, -1)[:, 0],
        "option_has": raw_data["option_before_has"].reshape(n, -1)[:, 0],
        "is_first": raw_data["is_first"].reshape(n, -1)[:, 0] > 0.5,
    }
    imag_horizon = agent.hierarchical_options.next_imagination_horizon()
    imagined = imagine_hierarchy(
        agent,
        start,
        option_start,
        imag_horizon + 1,
    )
    feat = imagined["feat"].detach()
    action = imagined["action"].detach()
    option = imagined["option"].detach()
    age = imagined["action_age"].detach()

    imag_reward = agent._frozen_reward(feat).mode()
    imag_cont = agent._frozen_cont(feat).mean
    current_value = agent.value(feat).mode()
    disc = 1.0 - 1.0 / agent.horizon

    # Option-aware call-and-return bootstrap at h_(t+1). This is the key
    # mathematical correction over treating a long-lived option as an ordinary
    # one-step actor. The next-state value explicitly mixes continuing the
    # current option and terminating into the manager value.
    next_feat = feat[:, 1:]
    current_option = option[:, :-1]
    next_age = imagined["carry_age"][:, :-1]
    slow_base_next = agent._frozen_slow_value(next_feat).mode().detach()
    exact_bootstrap_next, slow_continue_next, slow_switch_next, switch_next = (
        interruptible_option_bootstrap(
            agent._frozen_hierarchical_options,
            agent._slow_option_critic,
            next_feat,
            current_option,
            next_age,
            slow_base_next,
            agent._frozen_hierarchical_options.training_step,
        )
    )
    call_return_bootstrap = exact_bootstrap_next.unsqueeze(-1)
    ret = option_lambda_return(
        imag_reward[:, 1:],
        imag_cont[:, 1:],
        call_return_bootstrap,
        discount=disc,
        lambda_=agent.lamb,
    )
    scale = _return_scale(ret)

    # Weight action t by survival up to, but not including, action t.
    transition_discount = imag_cont[:, 1:] * disc
    weight = torch.cat(
        [
            torch.ones_like(transition_discount[:, :1]),
            torch.cumprod(transition_discount[:, :-1], dim=1),
        ],
        dim=1,
    )
    # Episode starts are valid manager boundaries and must receive task credit.
    # Only terminal/post-terminal replay starts are excluded.
    start_valid = (1.0 - raw_data["is_last"].float()).reshape(n, 1, 1)
    weight = weight * start_valid

    from smacdreamer.masked_actions import MaskedMultiOneHotDist

    policy_feat = feat
    base_logits = agent.actor.last(agent.actor.mlp(policy_feat))
    policy_logits = agent.hierarchical_options.combine_logits(
        base_logits,
        policy_feat,
        option,
        agent.hierarchical_options.training_step,
    )
    action_mask, active_mask = agent._predicted_action_mask(policy_feat)
    policy_dist = MaskedMultiOneHotDist(
        policy_logits,
        action_mask,
        active_mask,
        agent._actor_shape,
        agent._actor_unimix,
    )
    q_selected = agent.option_critic.q_selected(
        feat[:, :-1],
        option[:, :-1],
        age[:, :-1],
        current_value[:, :-1].detach(),
    )
    advantage = normalized_advantage(ret.squeeze(-1), q_selected, scale)
    if advantage.ndim < 3:
        advantage = advantage.unsqueeze(-1)

    log_prob = policy_dist.log_prob(action)[:, :-1]
    entropy = policy_dist.entropy()[:, :-1]
    if log_prob.ndim < advantage.ndim:
        log_prob = log_prob.unsqueeze(-1)
    if entropy.ndim < advantage.ndim:
        entropy = entropy.unsqueeze(-1)
    worker_pg_blend = agent.hierarchical_options.worker_pg_blend(
        agent.hierarchical_options.training_step
    )
    # Options 0/1 are immutable source anchors. Their log probabilities have
    # no trainable path, so including anchor samples in the normalization would
    # silently dilute the child-policy gradient. Optimize the worker objective
    # only on child-selected transitions; critic/value targets still use all
    # transitions.
    trainable_child = (
        agent.hierarchical_options.option_slot(option[:, :-1]) > 0
    ).to(dtype=weight.dtype).unsqueeze(-1)
    worker_weight = weight * trainable_child
    worker_loss = worker_policy_loss(
        log_prob,
        entropy,
        advantage,
        worker_weight,
        pg_scale=(
            worker_pg_blend
            * agent.hierarchical_options.settings.worker_pg_scale
        ),
        entropy_scale=(
            worker_pg_blend
            * agent.hierarchical_options.settings.worker_entropy_scale
        ),
    )

    hierarchy_value_log_prob = agent.value(feat[:, :-1]).log_prob(ret.detach())
    if hierarchy_value_log_prob.ndim < weight.ndim:
        hierarchy_value_log_prob = hierarchy_value_log_prob.unsqueeze(-1)
    hierarchy_value_per_item = (
        -hierarchy_value_log_prob * weight.detach()
    )
    hierarchy_value_loss = hierarchy_value_per_item.sum() / (
        weight.detach().sum().clamp_min(1.0)
    )

    q_loss = option_critic_loss(
        q_selected,
        ret.squeeze(-1),
        scale,
        weight[:, :, 0],
    )

    manager_probs = agent.hierarchical_options.manager_probs(
        feat[:, :-1], agent.hierarchical_options.training_step
    )
    q_all_age0 = agent.option_critic.q_all(
        feat[:, :-1],
        torch.zeros_like(age[:, :-1]),
        current_value[:, :-1].detach(),
    )
    critic_consistency_blend = max(0.0, 1.0 - float(worker_pg_blend))
    critic_consistency_loss = within_group_option_consistency_loss(
        q_all_age0,
        source_group_count=(
            agent.hierarchical_options.settings.source_manager_group_count
        ),
        return_scale=scale,
        weights=weight[:, :, 0],
    )
    # Use the slow option critic as the manager baseline. The baseline is
    # action-independent and detached; normalising it by the same return scale
    # as the worker prevents raw-return magnitude from dominating manager PG.
    slow_manager_q_age0 = agent._slow_option_critic.q_all(
        feat[:, :-1],
        torch.zeros_like(age[:, :-1]),
        agent._frozen_slow_value(feat[:, :-1]).mode().detach(),
    )
    # Only the within-group slot manager is trainable. Its baseline must exclude
    # the other Tactical source group, which execution cannot select at this
    # boundary because group routing is frozen and deterministic.
    manager_v = agent._frozen_hierarchical_options.switch_value_for_source_group(
        feat[:, :-1],
        slow_manager_q_age0,
        agent._frozen_hierarchical_options.training_step,
    )
    manager_adv = normalized_advantage(ret.squeeze(-1), manager_v, scale)
    (
        group_manager_log_prob,
        slot_manager_log_prob,
        group_manager_entropy,
        slot_manager_entropy,
        selected_slot_pg_blend,
    ) = agent.hierarchical_options.manager_log_prob_components(
        feat[:, :-1],
        option[:, :-1],
        agent.hierarchical_options.training_step,
    )
    boundary = imagined["option_started"][:, :-1]
    manager_pg_blend = agent.hierarchical_options.manager_pg_blend(
        agent.hierarchical_options.training_step
    )
    # The Tactical-v1.2 group router is immutable. Optimising a group policy
    # gradient here would be mathematically meaningless and previously obscured
    # whether the trainable slot manager received the intended gradient.
    group_manager_loss = group_manager_log_prob.sum() * 0.0
    # Slot routing starts only after child policies have received a substantial
    # worker-learning warm-up, avoiding the manager suppressing zero-output
    # children before they can develop a causal advantage.
    slot_manager_loss = manager_policy_loss(
        slot_manager_log_prob,
        slot_manager_entropy,
        manager_adv,
        boundary,
        weight[:, :, 0] * selected_slot_pg_blend.detach(),
        pg_scale=(
            manager_pg_blend
            * agent.hierarchical_options.settings.manager_pg_scale
        ),
        entropy_scale=agent.hierarchical_options.settings.manager_entropy_scale,
    )
    manager_loss = slot_manager_loss
    manager_entropy = slot_manager_entropy

    slow_base = agent._frozen_slow_value(feat[:, :-1]).mode().detach()
    previous_option = imagined["previous_option"][:, :-1]
    previous_age = imagined["previous_age"][:, :-1]
    continue_q = agent._slow_option_critic.q_selected(
        feat[:, :-1], previous_option, previous_age, slow_base
    )
    slow_q_age0 = agent._slow_option_critic.q_all(
        feat[:, :-1], torch.zeros_like(previous_age), slow_base
    )
    switch_v = agent._frozen_hierarchical_options.switch_value_for_source_group(
        feat[:, :-1], slow_q_age0,
        agent._frozen_hierarchical_options.training_step,
    )
    learned_beta = agent.hierarchical_options.learned_termination_probability(
        feat[:, :-1], previous_option, previous_age
    )
    bounded_learned_beta = (
        agent.hierarchical_options.bounded_learned_termination_probability(
            feat[:, :-1], previous_option, previous_age,
            agent.hierarchical_options.training_step,
        )
    )
    # Optimize the exact executed probability. The learned branch uses a smooth
    # bounded sigmoid, so gradients remain available near the probability cap.
    # Preservation reselection and fixed-hazard warm-up intentionally gate this
    # gradient until temporal commitment becomes active.
    execution_beta, online_termination_eligible, _, _ = (
        agent.hierarchical_options.effective_termination_probability(
            feat[:, :-1], previous_option, previous_age,
            agent.hierarchical_options.training_step,
        )
    )
    blend = agent.hierarchical_options.termination_blend(
        agent.hierarchical_options.training_step
    )
    online_continue_q = agent.option_critic.q_selected(
        feat[:, :-1], previous_option, previous_age,
        current_value[:, :-1].detach(),
    ).detach()
    online_q_age0 = agent.option_critic.q_all(
        feat[:, :-1], torch.zeros_like(previous_age),
        current_value[:, :-1].detach(),
    ).detach()
    online_switch_v = agent.hierarchical_options.switch_value_for_source_group(
        feat[:, :-1], online_q_age0,
        agent.hierarchical_options.training_step,
    )
    slow_norm_adv = (continue_q - switch_v) / scale
    online_norm_adv = (online_continue_q - online_switch_v) / scale
    target_disagreement = (slow_norm_adv - online_norm_adv).abs()
    sign_agreement = (slow_norm_adv * online_norm_adv) >= 0.0
    reliable_termination = (
        imagined["termination_eligible"][:, :-1]
        & online_termination_eligible
        & sign_agreement
        & (slow_norm_adv.abs() >= agent.hierarchical_options.settings.termination_min_advantage_magnitude)
        & (target_disagreement <= agent.hierarchical_options.settings.termination_max_target_disagreement)
    )
    term_loss, term_adv = termination_loss(
        execution_beta,
        continue_q,
        switch_v,
        reliable_termination,
        weight[:, :, 0],
        scale,
        normalized_margin=(
            agent.hierarchical_options.settings.termination_margin_normalized
        ),
        advantage_clip=(
            agent.hierarchical_options.settings.termination_advantage_clip
        ),
    )

    eligible_float = imagined["termination_eligible"][:, :-1].float()
    eligible_weight = weight[:, :, 0].detach() * eligible_float
    eligible_denominator = eligible_weight.sum().clamp_min(1.0)
    beta_safe = learned_beta.float().clamp(1.0e-6, 1.0 - 1.0e-6)
    termination_entropy = (
        -(beta_safe * beta_safe.log() + (1.0 - beta_safe) * (1.0 - beta_safe).log())
        * eligible_weight
    ).sum() / eligible_denominator
    eligible_beta_mean = (beta_safe * eligible_weight).sum() / eligible_denominator
    term_low = torch.relu(
        torch.as_tensor(
            agent.hierarchical_options.settings.termination_mean_min,
            device=feat.device, dtype=eligible_beta_mean.dtype,
        ) - eligible_beta_mean
    )
    active_beta_upper = min(
        agent.hierarchical_options.settings.termination_mean_max,
        agent.hierarchical_options.termination_probability_cap(
            agent.hierarchical_options.training_step
        ),
    )
    term_high = torch.relu(
        eligible_beta_mean - torch.as_tensor(
            active_beta_upper,
            device=feat.device, dtype=eligible_beta_mean.dtype,
        )
    )
    termination_collapse_loss = term_low.square() + term_high.square()

    source_reference_logits = (
        agent._source_hierarchical_options.combine_logits(
            base_logits[:, :-1].detach(),
            feat[:, :-1].detach(),
            option[:, :-1],
            agent._source_hierarchical_options.training_step,
        )
    )
    behavior = agent.hierarchical_options.behaviour_statistics(
        feat[:, :-1].detach(),
        base_logits[:, :-1].detach(),
        source_reference_logits.detach(),
        option[:, :-1],
        action_mask[:, :-1].detach(),
        active_mask[:, :-1].detach(),
        agent._actor_shape,
        weight[:, :, 0].detach(),
        agent.hierarchical_options.training_step,
        unimix_ratio=agent._actor_unimix,
    )
    manager_stats = agent.hierarchical_options.manager_statistics(
        manager_probs,
        option[:, :-1],
        boundary,
        weight[:, :, 0],
    )
    source_manager_probs = (
        agent._source_hierarchical_options.manager_probs(
            feat[:, :-1].detach(),
            agent._source_hierarchical_options.training_step,
        )
    )
    manager_source = agent.hierarchical_options.manager_source_statistics(
        manager_probs,
        source_manager_probs,
        weight[:, :, 0].detach(),
    )

    # Protect the source policy on real posterior states as well as imagined
    # states. Imagination-only trust regions can miss rare but decisive states
    # when the frozen model does not revisit them accurately.
    real_feat = agent._frozen_jepa_world_model.get_feat(
        post_stoch.detach(), post_deter.detach()
    ).detach()
    real_option = raw_data["option_id"].long().squeeze(-1).clamp(
        0, agent.hierarchical_options.num_options - 1
    )
    real_weight = (1.0 - raw_data["is_last"].float()).squeeze(-1).detach()
    real_base_logits = agent.actor.last(agent.actor.mlp(real_feat))
    real_action_mask, real_active_mask = agent._predicted_action_mask(real_feat)
    real_source_logits = agent._source_hierarchical_options.combine_logits(
        real_base_logits.detach(),
        real_feat,
        real_option,
        agent._source_hierarchical_options.training_step,
    )
    real_behavior = agent.hierarchical_options.behaviour_statistics(
        real_feat,
        real_base_logits.detach(),
        real_source_logits.detach(),
        real_option,
        real_action_mask.detach(),
        real_active_mask.detach(),
        agent._actor_shape,
        real_weight,
        agent.hierarchical_options.training_step,
        unimix_ratio=agent._actor_unimix,
    )
    real_manager_probs = agent.hierarchical_options.manager_probs(
        real_feat, agent.hierarchical_options.training_step
    )
    real_source_manager_probs = (
        agent._source_hierarchical_options.manager_probs(
            real_feat, agent._source_hierarchical_options.training_step
        )
    )
    real_manager_source = agent.hierarchical_options.manager_source_statistics(
        real_manager_probs, real_source_manager_probs, real_weight
    )

    # Average real and imagined source-preservation objectives so neither
    # distribution can dominate only because it contributes more samples.
    source_policy_kl_loss = 0.5 * (
        behavior["base_kl_loss"] + real_behavior["base_kl_loss"]
    )
    source_action_preservation_loss = 0.5 * (
        behavior["action_preservation_loss"]
        + real_behavior["action_preservation_loss"]
    )
    source_residual_guard_loss = 0.5 * (
        behavior["residual_guard_loss"]
        + real_behavior["residual_guard_loss"]
    )
    source_manager_kl_loss = 0.5 * (
        manager_source["kl_loss"] + real_manager_source["kl_loss"]
    )
    source_manager_preservation_loss = 0.5 * (
        manager_source["preservation_loss"]
        + real_manager_source["preservation_loss"]
    )
    s = agent.hierarchical_options.settings
    # Do not attach disabled objectives to the autograd graph. Multiplying a
    # numerically ill-conditioned diagnostic by literal zero is not sufficient:
    # backward can still encounter 0 * NaN in its local Jacobian and poison an
    # otherwise unrelated parameter. This was exposed by an anchor-only rollout
    # where the disabled all-option cosine diagnostic had undefined gradients at
    # exact zero child deltas.
    total = (
        worker_loss
        + manager_loss
        + s.option_critic_scale * q_loss
        + s.hierarchy_value_scale * hierarchy_value_loss
        + s.manager_group_kl_scale * source_manager_kl_loss
        + s.manager_group_preservation_scale * source_manager_preservation_loss
    )
    # During the critic-only warm-up, child policies must be *exactly* frozen.
    # Even a mathematically zero KL at identical distributions can leave tiny
    # floating-point gradients on an exact-zero output layer. AdamW can then
    # turn that numerical residue into behavior drift before worker learning is
    # intended to begin. Attach worker safety losses only when the worker PG
    # schedule is active, and ramp them through the same single schedule.
    if worker_pg_blend:
        total = total + worker_pg_blend * (
            s.base_kl_scale * source_policy_kl_loss
            + s.action_preservation_scale * source_action_preservation_loss
            + s.residual_guard_scale * source_residual_guard_loss
        )
    if s.option_critic_consistency_scale and critic_consistency_blend:
        total = total + (
            s.option_critic_consistency_scale
            * critic_consistency_blend
            * critic_consistency_loss
        )
    if s.termination_loss_scale:
        total = total + s.termination_loss_scale * term_loss
    if s.termination_entropy_scale and blend:
        total = total - blend * s.termination_entropy_scale * termination_entropy
    if s.termination_collapse_scale and blend:
        total = total + blend * s.termination_collapse_scale * termination_collapse_loss
    if s.manager_collapse_scale:
        total = total + s.manager_collapse_scale * manager_stats["collapse_loss"]
    if s.manager_mi_scale and manager_pg_blend:
        total = total + (
            manager_pg_blend * s.manager_mi_scale * manager_stats["mi_shortfall_loss"]
        )
    if s.action_diversity_scale:
        total = total + s.action_diversity_scale * behavior["diversity_loss"]
    if s.residual_cosine_scale:
        total = total + s.residual_cosine_scale * behavior["residual_cosine_loss"]

    eligible = imagined["termination_eligible"][:, :-1].float()
    terminated = imagined["option_terminated"][:, :-1].float()
    eligible_den = eligible.sum().clamp_min(1.0)

    # Replay-grounded option-state diagnostics. These fields come from real
    # environment execution and are separate from imagined option statistics.
    replay_valid = (
        (1.0 - raw_data["is_first"].float())
        * (1.0 - raw_data["is_last"].float())
    ).squeeze(-1)
    replay_den = replay_valid.sum().clamp_min(1.0)
    replay_started = raw_data["option_started"].float().squeeze(-1)
    replay_terminated = raw_data["option_terminated"].float().squeeze(-1)
    replay_eligible = raw_data["option_termination_eligible"].float().squeeze(-1)
    replay_option = raw_data["option_id"].long().squeeze(-1).clamp(
        0, agent.hierarchical_options.num_options - 1
    )
    replay_before_option = raw_data["option_before_id"].long().squeeze(-1).clamp(
        0, agent.hierarchical_options.num_options - 1
    )
    replay_before_age = raw_data["option_before_age"].float().squeeze(-1)
    replay_age = raw_data["option_action_age"].float().squeeze(-1)
    replay_usage = torch.nn.functional.one_hot(
        replay_option, agent.hierarchical_options.num_options
    ).float()
    replay_usage = (replay_usage * replay_valid.unsqueeze(-1)).sum((0, 1))
    replay_usage = replay_usage / replay_usage.sum().clamp_min(1.0)
    replay_eligible_den = (replay_eligible * replay_valid).sum().clamp_min(1.0)

    imag_prev_age = imagined["previous_age"][:, :-1]
    imag_started = imagined["option_started"][:, :-1]
    imag_terminated_bool = imagined["option_terminated"][:, :-1]
    imag_option = imagined["option"][:, :-1]
    imag_previous_option = imagined["previous_option"][:, :-1]
    imag_source_interrupt = (
        agent._frozen_hierarchical_options.option_group(imag_previous_option)
        != agent._frozen_hierarchical_options.source_group(
            feat[:, :-1], agent._frozen_hierarchical_options.training_step
        )
    )
    # Source-group safety interrupts intentionally override the ordinary minimum
    # duration. They are not state-machine violations.
    imag_min_violation = (
        imag_terminated_bool
        & (imag_prev_age < s.min_duration)
        & (~imag_source_interrupt)
    )
    imag_max_violation = (imag_prev_age >= s.max_duration) & (~imag_terminated_bool)
    imag_change_without_boundary = (
        (imag_option != imag_previous_option) & (~imag_started)
    )
    real_before_has = raw_data["option_before_has"].bool().squeeze(-1)
    real_is_first = raw_data["is_first"].bool().squeeze(-1)
    real_structural = replay_valid.bool() & real_before_has & (~real_is_first)
    real_source_interrupt = (
        agent._frozen_hierarchical_options.option_group(replay_before_option)
        != agent._frozen_hierarchical_options.source_group(
            real_feat, agent._frozen_hierarchical_options.training_step
        )
    )
    real_min_violation = (
        replay_terminated.bool()
        & (replay_before_age < s.min_duration)
        & real_structural
        & (~real_source_interrupt)
    )
    real_max_violation = (
        (~replay_terminated.bool())
        & (replay_before_age >= s.max_duration)
        & real_structural
    )
    real_change_without_boundary = (
        (replay_option != replay_before_option)
        & (~replay_started.bool())
        & real_structural
    )
    structural_den = real_structural.float().sum().clamp_min(1.0)

    metrics: dict[str, torch.Tensor] = {
        "option/worker_policy_loss": worker_loss.detach(),
        "option/manager_policy_loss": manager_loss.detach(),
        "option/group_manager_policy_loss": group_manager_loss.detach(),
        "option/slot_manager_policy_loss": slot_manager_loss.detach(),
        "option/selected_slot_pg_blend": (
            selected_slot_pg_blend.detach() * weight[:, :, 0].detach()
        ).sum() / weight[:, :, 0].detach().sum().clamp_min(1.0),
        "option/termination_loss": term_loss.detach(),
        "option/critic_loss": q_loss.detach(),
        "option/critic_consistency_loss": critic_consistency_loss.detach(),
        "option/critic_consistency_blend": torch.as_tensor(
            critic_consistency_blend, device=feat.device
        ),
        "option/hierarchy_value_loss": hierarchy_value_loss.detach(),
        "option/termination_blend": torch.as_tensor(blend, device=feat.device),
        "option/termination_probability": imagined[
            "termination_probability"
        ][:, :-1].mean(),
        "option/eligible_learned_beta_mean": eligible_beta_mean.detach(),
        "option/bounded_learned_beta_mean": (
            bounded_learned_beta.float() * eligible_weight
        ).sum() / eligible_denominator,
        "option/raw_to_bounded_beta_gap": (
            (learned_beta.float() - bounded_learned_beta.float()).abs()
            * eligible_weight
        ).sum() / eligible_denominator,
        "option/termination_entropy": termination_entropy.detach(),
        "option/termination_collapse_loss": termination_collapse_loss.detach(),
        "option/eligible_termination_rate": (
            terminated * eligible
        ).sum() / eligible_den,
        "option/forced_max_termination_rate": (
            imagined["option_terminated"][:, :-1]
            & (~imagined["termination_eligible"][:, :-1])
            & (imagined["previous_age"][:, :-1] >= s.max_duration)
        ).float().mean(),
        "option/mean_age": age[:, :-1].float().mean(),
        "option/boundary_rate": boundary.float().mean(),
        "option/same_option_reselection_rate": (
            (imagined["option_started"][:, :-1])
            & (imagined["option"][:, :-1] == imagined["previous_option"][:, :-1])
            & imagined["option_terminated"][:, :-1]
        ).float().sum() / imagined["option_terminated"][:, :-1].float().sum().clamp_min(1.0),
        "option/manager_boundary_count": manager_stats["boundary_count"],
        "option/manager_pg_blend": torch.as_tensor(
            manager_pg_blend, device=feat.device
        ),
        "option/worker_pg_blend": torch.as_tensor(
            worker_pg_blend, device=feat.device
        ),
        "option/worker_trainable_child_fraction": (
            trainable_child * weight.detach()
        ).sum() / weight.detach().sum().clamp_min(1.0),
        "option/commitment_reselect_probability": torch.as_tensor(
            agent.hierarchical_options.commitment_reselect_probability(),
            device=feat.device,
        ),
        "option/world_model_grad_scale": torch.as_tensor(
            agent.hierarchical_options.world_model_grad_scale(),
            device=feat.device,
        ),
        "option/imag_horizon": torch.as_tensor(
            imag_horizon, device=feat.device
        ),
        "option/termination_active_upper_bound": torch.as_tensor(
            active_beta_upper, device=feat.device
        ),
        "option/termination_execution_cap": torch.as_tensor(
            agent.hierarchical_options.termination_probability_cap(
                agent.hierarchical_options.training_step
            ),
            device=feat.device,
        ),
        "option/worker_advantage_rms": advantage.detach().float().square().mean().sqrt(),
        "option/manager_advantage_rms": manager_adv.detach().float().square().mean().sqrt(),
        "option/imag_min_duration_violation_rate": imag_min_violation.float().mean(),
        "option/imag_max_duration_violation_rate": imag_max_violation.float().mean(),
        "option/imag_change_without_boundary_rate": (
            imag_change_without_boundary.float().mean()
        ),
        "option/imag_source_interrupt_rate": imag_source_interrupt.float().mean(),
        "option/worker_entropy": (
            entropy.float() * weight
        ).sum() / weight.sum().clamp_min(1.0),
        "option/manager_entropy": manager_stats["conditional_entropy"],
        "option/manager_marginal_entropy": manager_stats["marginal_entropy"],
        "option/manager_mutual_information": manager_stats["mutual_information"],
        "option/manager_mutual_information_normalized": manager_stats[
            "mutual_information_normalized"
        ],
        "option/effective_count": manager_stats["effective_count"],
        "option/usage_max": manager_stats["usage_max"],
        "option/collapse_loss": manager_stats["collapse_loss"],
        "option/manager_mi_shortfall_loss": manager_stats["mi_shortfall_loss"],
        "option/source_manager_group_kl_mean": manager_source["kl_mean"],
        "option/source_manager_group_kl_tail": manager_source["kl_tail"],
        "option/real_source_manager_group_kl_mean": real_manager_source["kl_mean"],
        "option/real_source_manager_group_kl_tail": real_manager_source["kl_tail"],
        "option/real_source_manager_group_flip_rate": real_manager_source[
            "flip_rate"
        ],
        "option/source_manager_group_kl_max": manager_source["kl_max"],
        "option/source_manager_group_flip_rate": manager_source["flip_rate"],
        "option/source_manager_group_high_confidence_flip_rate": manager_source[
            "high_confidence_flip_rate"
        ],
        "option/source_manager_group_preservation_loss": manager_source[
            "preservation_loss"
        ],
        "option/action_js_mean": behavior["js_mean"],
        "option/action_js_min": behavior["js_min"],
        "option/action_js_max": behavior["js_max"],
        "option/duplicate_pair_fraction": behavior["duplicate_pair_fraction"],
        "option/js_shortfall_fraction": behavior["js_shortfall_fraction"],
        "option/base_kl_mean": behavior["base_kl_mean"],
        "option/base_kl_tail": behavior["base_kl_tail"],
        "option/base_kl_max": behavior["base_kl_max"],
        "option/source_policy_kl_mean": behavior["base_kl_mean"],
        "option/source_policy_kl_tail": behavior["base_kl_tail"],
        "option/real_source_policy_kl_mean": real_behavior["base_kl_mean"],
        "option/real_source_policy_kl_tail": real_behavior["base_kl_tail"],
        "option/real_source_action_flip_rate": real_behavior["action_flip_rate"],
        "option/real_source_high_confidence_action_flip_rate": real_behavior[
            "high_confidence_flip_rate"
        ],
        "option/action_flip_rate": behavior["action_flip_rate"],
        "option/high_confidence_action_flip_rate": behavior[
            "high_confidence_flip_rate"
        ],
        "option/action_preservation_loss": behavior[
            "action_preservation_loss"
        ],
        "option/residual_rms": behavior["residual_rms"],
        "option/residual_to_base_ratio": behavior["residual_ratio"],
        "option/residual_cosine_mean": behavior["residual_cosine_mean"],
        "option/residual_duplicate_fraction": behavior["residual_duplicate_fraction"],
        "option/worker_scale": torch.as_tensor(
            agent.hierarchical_options.worker_scale(), device=feat.device
        ),
        "option/manager_unimix": torch.as_tensor(
            agent.hierarchical_options.manager_unimix(), device=feat.device
        ),
        "option/slot_anchor_floor": torch.as_tensor(
            agent.hierarchical_options.settings.slot_anchor_floor, device=feat.device
        ),
        "option/termination_advantage_mean": term_adv.mean(),
        "option/termination_reliable_fraction": (
            reliable_termination.float() * eligible_float
        ).sum() / eligible_denominator,
        "option/termination_target_sign_agreement": (
            sign_agreement.float() * eligible_float
        ).sum() / eligible_denominator,
        "option/termination_target_disagreement": (
            target_disagreement * eligible_float
        ).sum() / eligible_denominator,
        "option/continue_value_mean": continue_q.mean(),
        "option/switch_value_mean": switch_v.mean(),
        "option/real_boundary_rate": (
            replay_started * replay_valid
        ).sum() / replay_den,
        "option/real_termination_rate": (
            replay_terminated * replay_valid
        ).sum() / replay_den,
        "option/real_eligible_termination_rate": (
            replay_terminated * replay_eligible * replay_valid
        ).sum() / replay_eligible_den,
        "option/real_eligible_fraction": (
            replay_eligible * replay_valid
        ).sum() / replay_den,
        "option/real_mean_action_age": (
            replay_age * replay_valid
        ).sum() / replay_den,
        "option/real_forced_max_termination_rate": (
            replay_terminated
            * (replay_before_age >= s.max_duration).float()
            * replay_valid
        ).sum() / replay_den,
        "option/real_same_option_reselection_rate": (
            replay_started
            * replay_terminated
            * (replay_option == replay_before_option).float()
            * replay_valid
        ).sum() / (
            (replay_started * replay_terminated * replay_valid).sum().clamp_min(1.0)
        ),
        "option/real_termination_probability": (
            raw_data["option_termination_prob"].float().squeeze(-1)
            * replay_eligible * replay_valid
        ).sum() / replay_eligible_den,
        "option/real_min_duration_violation_rate": (
            real_min_violation.float().sum() / structural_den
        ),
        "option/real_max_duration_violation_rate": (
            real_max_violation.float().sum() / structural_den
        ),
        "option/real_change_without_boundary_rate": (
            real_change_without_boundary.float().sum() / structural_den
        ),
        "option/real_source_interrupt_rate": (
            (real_source_interrupt & real_structural).float().sum() / structural_den
        ),
        "option/real_mean_completed_dwell": (
            replay_before_age
            * replay_terminated
            * real_structural.float()
        ).sum() / (
            (replay_terminated * real_structural.float()).sum().clamp_min(1.0)
        ),
    }
    for group_index in range(
        agent.hierarchical_options.settings.source_manager_group_count
    ):
        metrics[f"option/source_manager_group_live_{group_index}"] = (
            manager_source["live_group_probs"][..., group_index]
            * weight[:, :, 0].detach()
        ).sum() / weight[:, :, 0].detach().sum().clamp_min(1.0)
        metrics[f"option/source_manager_group_reference_{group_index}"] = (
            manager_source["source_group_probs"][..., group_index]
            * weight[:, :, 0].detach()
        ).sum() / weight[:, :, 0].detach().sum().clamp_min(1.0)

    slot_gates = agent.hierarchical_options.slot_gate_by_option(
        agent.hierarchical_options.training_step
    ).to(feat.device)
    option_ids_for_scale = torch.arange(
        agent.hierarchical_options.num_options, device=feat.device
    )
    slot_delta_scales = agent.hierarchical_options.slot_delta_scale_by_option(
        option_ids_for_scale, agent.hierarchical_options.training_step
    )
    slot_pg_scales = agent.hierarchical_options.slot_pg_blend_for_option(
        option_ids_for_scale, agent.hierarchical_options.training_step
    )
    for index in range(agent.hierarchical_options.num_options):
        metrics[f"option/slot_gate_{index}"] = slot_gates[index]
        metrics[f"option/slot_delta_scale_{index}"] = slot_delta_scales[index]
        metrics[f"option/slot_pg_blend_{index}"] = slot_pg_scales[index]
        metrics[f"option/usage_{index}"] = manager_stats["marginal"][index]
        metrics[f"option/sampled_usage_{index}"] = manager_stats[
            "sampled_usage"
        ][index]
        metrics[f"option/real_usage_{index}"] = replay_usage[index]
        option_weight = (
            (option[:, :-1] == index).float() * weight[:, :, 0].detach()
        )
        option_den = option_weight.sum().clamp_min(1.0)
        metrics[f"option/imag_weight_{index}"] = option_weight.sum()
        metrics[f"option/imag_return_{index}"] = (
            ret.squeeze(-1).detach() * option_weight
        ).sum() / option_den
    return total, metrics


def update_slow_option_critic(agent: Any) -> None:
    if not getattr(agent, "hierarchical_enabled", False):
        return
    settings = agent.hierarchical_options.settings
    if agent._option_slow_updates % settings.slow_target_update == 0:
        mix = settings.slow_target_fraction
        with torch.no_grad():
            for online, slow in zip(
                agent.option_critic.parameters(),
                agent._slow_option_critic.parameters(),
            ):
                slow.data.copy_(mix * online.data + (1.0 - mix) * slow.data)
    agent._option_slow_updates += 1


def hierarchy_training_state(agent: Any) -> dict[str, Any]:
    if not getattr(agent, "hierarchical_enabled", False):
        return {}
    return {
        "slow_option_critic": agent._slow_option_critic.state_dict(),
        "option_slow_updates": int(agent._option_slow_updates),
        "hierarchy_training_step": int(
            agent.hierarchical_options._training_step_int
        ),
        "hierarchy_diversity_calls": int(
            agent.hierarchical_options._diversity_calls_int
        ),
        "hierarchy_horizon_calls": int(
            agent.hierarchical_options._horizon_calls_int
        ),
    }


def load_hierarchy_training_state(agent: Any, state: dict[str, Any]) -> None:
    if not getattr(agent, "hierarchical_enabled", False):
        return
    payload = state.get("hierarchical_options") or {}
    if payload.get("slow_option_critic") is not None:
        agent._slow_option_critic.load_state_dict(
            payload["slow_option_critic"], strict=True
        )
    agent._option_slow_updates = int(
        payload.get("option_slow_updates", agent._option_slow_updates)
    )
    agent.hierarchical_options.set_training_step(
        int(payload.get("hierarchy_training_step", 0))
    )
    agent.hierarchical_options.set_diversity_calls(
        int(payload.get("hierarchy_diversity_calls", 0))
    )
    agent.hierarchical_options.set_horizon_calls(
        int(payload.get("hierarchy_horizon_calls", 0))
    )
    clone_and_freeze_hierarchy(agent)


def load_hierarchical_compatible_state(
    agent: Any,
    state_dict: dict[str, torch.Tensor],
    *,
    checkpoint_metadata: dict[str, Any] | None = None,
    tactical_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strict hierarchy resume or controlled migration from Tactical v1.2."""
    if not getattr(agent, "hierarchical_enabled", False):
        agent.load_state_dict(state_dict, strict=True)
        agent.clone_and_freeze()
        return {"migrated": False, "strict": True}

    has_hierarchy = any(
        key.startswith("hierarchical_options.") for key in state_dict
    )
    if has_hierarchy:
        expected = agent.hierarchical_metadata()
        if checkpoint_metadata is None:
            raise RuntimeError("hierarchical checkpoint lacks metadata")
        for key in (
            "architecture",
            "num_options",
            "option_embedding_dim",
            "age_embedding_dim",
            "hidden_dim",
            "min_duration",
            "max_duration",
            "feature_dim",
            "action_logit_dim",
            "worker_scale_initial",
            "worker_scale_max",
            "slot_delta_scale_max",
            "worker_pg_warmup_steps",
            "worker_pg_full_steps",
            "manager_unimix_initial",
            "manager_unimix_final",
            "slot_manager_unimix",
            "slot_anchor_floor",
            "slot_pair_unlock_initial_steps",
            "slot_pair_unlock_interval_steps",
            "slot_unlock_ramp_steps",
            "slot_pg_ramp_steps",
            "manager_pg_warmup_steps",
            "manager_pg_full_steps",
            "commitment_warmup_steps",
            "commitment_full_steps",
            "termination_warmup_steps",
            "termination_full_steps",
            "termination_cap_full_steps",
            "termination_soft_cap_temperature",
            "world_model_grad_scale_initial",
            "world_model_grad_scale_final",
            "base_kl_target",
            "base_kl_tail_target",
            "action_preservation_scale",
            "source_manager_group_count",
            "manager_group_kl_target",
            "manager_group_kl_tail_target",
            "manager_group_kl_scale",
            "manager_group_preservation_scale",
            "manager_collapse_scale",
            "manager_mi_scale",
            "action_diversity_scale",
            "residual_cosine_scale",
            "option_critic_consistency_scale",
            "imag_horizon_initial_max",
            "imag_horizon_final_max",
            "imag_horizon_window",
            "imag_horizon_ramp_steps",
            "eval_sample_termination",
            "eval_termination_hazard_threshold",
        ):
            if checkpoint_metadata.get(key) != expected.get(key):
                raise RuntimeError(
                    f"hierarchical metadata mismatch for {key}: "
                    f"{checkpoint_metadata.get(key)!r} != {expected.get(key)!r}"
                )
        agent.load_state_dict(state_dict, strict=True)
        agent.clone_and_freeze()
        return {"migrated": False, "strict": True}

    architecture = (tactical_metadata or {}).get("architecture")
    if architecture != "tactical_mixture_v1_2":
        raise RuntimeError(
            "Option-Critic migration requires a Tactical Mixture v1.2 "
            f"checkpoint, got {architecture!r}"
        )
    source_num = int((tactical_metadata or {}).get("num_tactics", -1))
    if source_num != 2:
        raise RuntimeError(
            "Option-Critic migration requires exactly two v1.2 modes, "
            f"got {source_num}"
        )

    tactical = {
        key: value
        for key, value in state_dict.items()
        if key.startswith("tactical_policy.")
    }
    required = (
        "tactical_policy.selector.0.weight",
        "tactical_policy.selector.0.bias",
        "tactical_policy.selector.2.weight",
        "tactical_policy.selector.2.bias",
        "tactical_policy.embedding.weight",
        "tactical_policy.residual.0.weight",
        "tactical_policy.residual.0.bias",
        "tactical_policy.residual.2.weight",
        "tactical_policy.residual.2.bias",
    )
    missing_source = [key for key in required if key not in tactical]
    if missing_source:
        raise RuntimeError(
            f"v1.2 migration source is missing tactical keys: {missing_source}"
        )

    base_state = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith((
            "tactical_policy.",
            "_frozen_tactical_policy.",
        ))
    }
    incompatible = agent.load_state_dict(base_state, strict=False)
    allowed_missing = (
        "hierarchical_options.",
        "_frozen_hierarchical_options.",
        "_source_hierarchical_options.",
        "option_critic.",
        "_slow_option_critic.",
    )
    illegal_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing)
    ]
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "v1.2 base-state migration failed: "
            f"illegal_missing={illegal_missing}, "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )

    with torch.no_grad():
        target = agent.hierarchical_options
        if target.num_options != 8:
            raise RuntimeError(
                f"v9 anchor-safe migration requires eight slots, got {target.num_options}"
            )
        if target.settings.source_manager_group_count != 2:
            raise RuntimeError("v9 anchor-safe migration requires two source groups")

        # Exact source-group selector. Slot routing is a separate zero-output
        # head; all identities are available, while every child action remains
        # exactly source-equivalent at step zero. A fixed anchor floor in the
        # probability transform protects collection until child advantages are
        # learned without hiding any of the eight option identities.
        target.manager_group[0].weight.copy_(
            tactical["tactical_policy.selector.0.weight"]
        )
        target.manager_group[0].bias.copy_(
            tactical["tactical_policy.selector.0.bias"]
        )
        target.manager_group[2].weight.copy_(
            tactical["tactical_policy.selector.2.weight"]
        )
        target.manager_group[2].bias.copy_(
            tactical["tactical_policy.selector.2.bias"]
        )
        # Preserve the non-degenerate hidden layer created by reset_parameters.
        # Only the output layer is zero, so the initial routing prior is exact but
        # state-dependent gradients can reach the output weights immediately.
        target.manager_slot[0].bias.zero_()
        target.manager_slot[2].weight.zero_()
        # Output order is [g0s0, g1s0, g0s1, g1s1, ...]. Raw logits remain
        # uniform at migration; manager_slot_probs applies the fixed anchor floor.
        # Every child initially executes the exact parent policy, so identity
        # exploration changes no primitive action. Deterministic evaluation
        # selects slot zero, exactly reproducing the source.
        target.manager_slot[2].bias.zero_()

        # Exact inherited source worker over the two tactical groups. Child slot
        # deltas are identically zero and acquire causal effect only through the
        # per-slot specialization schedule.
        target.option_embedding.weight.copy_(
            tactical["tactical_policy.embedding.weight"]
        )
        source_embedding = tactical["tactical_policy.embedding.weight"]
        for row in range(target.num_options):
            group = row % 2
            slot = row // 2
            base = source_embedding[group]
            if slot == 0:
                target.slot_embedding.weight[row].copy_(base)
            else:
                # Deterministic zero-mean offsets break child symmetry without
                # changing actions because the child output layer is exactly zero.
                idx = torch.arange(base.numel(), device=base.device, dtype=base.dtype)
                offset = torch.sin(idx + float(17 * row + 1))
                offset = offset - offset.mean()
                offset = 0.01 * offset / offset.norm().clamp_min(1.0e-8)
                target.slot_embedding.weight[row].copy_(base + offset)
        target.worker_residual[0].weight.copy_(
            tactical["tactical_policy.residual.0.weight"]
        )
        target.worker_residual[0].bias.copy_(
            tactical["tactical_policy.residual.0.bias"]
        )
        target.worker_residual[2].weight.copy_(
            tactical["tactical_policy.residual.2.weight"]
        )
        target.worker_residual[2].bias.copy_(
            tactical["tactical_policy.residual.2.bias"]
        )
        # Keep the non-degenerate child hidden layer. The zero output layer gives
        # exact source actions at migration while retaining immediate non-zero
        # gradients for state-dependent child policies.
        target.slot_delta[0].bias.zero_()
        target.slot_delta[2].weight.zero_()
        target.slot_delta[2].bias.zero_()

        agent.option_critic.trunk[-1].weight.zero_()
        agent.option_critic.trunk[-1].bias.zero_()
        agent._slow_option_critic.load_state_dict(
            agent.option_critic.state_dict(), strict=True
        )

    sync_source_hierarchy(agent)
    agent.clone_and_freeze()
    return {
        "migrated": True,
        "strict": False,
        "source_architecture": architecture,
        "source_options": 2,
        "target_options": target.num_options,
        "migration_layout": "two_frozen_source_anchors_plus_six_anchor_floor_interruptible_children",
        "trajectory_preservation": (
            "exact_interruptible_smdp_with_group_restricted_slot_manager"
        ),
    }
