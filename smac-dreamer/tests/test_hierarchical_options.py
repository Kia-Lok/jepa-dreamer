from __future__ import annotations

import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/r2dreamer"))

from hierarchical_options import HierarchicalOptionsPolicy  # noqa: E402


def cfg(**overrides):
    base = {
        "enabled": True,
        "num_options": 8,
        "option_embedding_dim": 16,
        "age_embedding_dim": 8,
        "hidden_dim": 32,
        "min_duration": 1,
        "max_duration": 20,
        "commitment_warmup_steps": 50,
        "commitment_full_steps": 150,
        "commitment_reselect_initial": 1.0,
        "commitment_reselect_final": 0.0,
        "initial_termination_probability": 0.10,
        "termination_warmup_steps": 50,
        "termination_full_steps": 150,
        "termination_max_probability_during_ramp": 0.30,
        "termination_max_probability_final": 0.80,
        "termination_cap_full_steps": 250,
        "termination_soft_cap_temperature": 0.03,
        "termination_margin_normalized": 0.02,
        "termination_loss_scale": 0.05,
        "termination_entropy_scale": 0.0,
        "termination_collapse_scale": 0.05,
        "termination_mean_min": 0.02,
        "termination_mean_max": 0.60,
        "termination_advantage_clip": 1.0,
        "termination_min_advantage_magnitude": 0.01,
        "termination_max_target_disagreement": 0.25,
        "termination_unimix": 0.02,
        "eval_sample_termination": False,
        "eval_termination_hazard_threshold": 1.0,
        "manager_unimix_initial": 0.0,
        "manager_unimix_final": 0.02,
        "manager_unimix_decay_steps": 200,
        "slot_manager_unimix": 0.01,
        "slot_pair_unlock_initial_steps": 50,
        "slot_pair_unlock_interval_steps": 50,
        "slot_unlock_ramp_steps": 50,
        "slot_pg_ramp_steps": 50,
        "manager_pg_scale": 1.0,
        "manager_pg_warmup_steps": 50,
        "manager_pg_full_steps": 150,
        "manager_entropy_scale": 0.0,
        "manager_collapse_scale": 0.0,
        "manager_mi_target_normalized": 0.10,
        "manager_mi_scale": 0.0,
        "max_usage_target": 0.75,
        "min_effective_options": 3.0,
        "worker_pg_scale": 1.0,
        "worker_pg_warmup_steps": 25,
        "worker_pg_full_steps": 100,
        "worker_entropy_scale": 0.0,
        "worker_scale_initial": 0.25,
        "worker_scale_warmup_steps": 0,
        "worker_scale_full_steps": 1,
        "worker_scale_max": 0.25,
        "slot_delta_scale_max": 0.10,
        "max_abs_residual_logit": 2.0,
        "max_residual_to_base": 0.25,
        "residual_guard_scale": 0.05,
        "base_kl_target": 0.01,
        "base_kl_tail_target": 0.03,
        "base_kl_tail_fraction": 0.10,
        "base_kl_tail_relative_scale": 1.0,
        "base_kl_scale": 0.25,
        "action_preservation_confidence": 0.80,
        "action_preservation_margin": 0.05,
        "action_preservation_scale": 0.25,
        "source_manager_group_count": 2,
        "manager_group_kl_target": 0.005,
        "manager_group_kl_tail_target": 0.02,
        "manager_group_kl_tail_fraction": 0.10,
        "manager_group_kl_tail_relative_scale": 1.0,
        "manager_group_kl_scale": 0.25,
        "manager_group_preservation_confidence": 0.80,
        "manager_group_preservation_margin": 0.05,
        "manager_group_preservation_scale": 0.25,
        "action_diversity_target": 0.002,
        "action_diversity_scale": 0.0,
        "residual_cosine_target": 0.95,
        "residual_cosine_scale": 0.0,
        "max_diversity_states": 64,
        "max_diversity_pairs": 8,
        "option_critic_scale": 1.0,
        "hierarchy_value_scale": 0.5,
        "slow_target_update": 1,
        "slow_target_fraction": 0.005,
        "freeze_base_actor": True,
        "freeze_feature_adapter": True,
        "world_model_grad_scale_initial": 0.0,
        "world_model_grad_scale_final": 0.0,
        "world_model_grad_warmup_steps": 100,
        "world_model_grad_full_steps": 200,
        "imag_horizon_initial_max": 8,
        "imag_horizon_final_max": 10,
        "imag_horizon_window": 4,
        "imag_horizon_ramp_steps": 200,
    }
    base.update(overrides)
    return base


def policy(**overrides):
    torch.manual_seed(0)
    return HierarchicalOptionsPolicy(12, 15, cfg(**overrides))


def test_source_group_residuals_are_centered_and_anchor_slots_are_exact():
    model = policy()
    model.set_training_step(0)
    feat = torch.randn(7, 12)
    group_residual = model._all_group_residual_logits(feat, 0)
    residual = model.all_residual_logits(feat, 0)
    assert residual.shape == (7, 8, 15)
    assert torch.allclose(
        group_residual.sum(dim=-2), torch.zeros(7, 15), atol=2e-6, rtol=0
    )
    assert torch.allclose(residual[:, 0], group_residual[:, 0], atol=1e-7)
    assert torch.allclose(residual[:, 1], group_residual[:, 1], atol=1e-7)
    # Every child has zero specialization delta at migration.
    for index in range(2, 8):
        assert torch.allclose(
            residual[:, index], group_residual[:, index % 2], atol=1e-7
        )


def test_migrated_worker_scale_is_preserved_from_step_zero():
    model = policy()
    assert model.worker_scale(0) == 0.25
    assert model.worker_scale(25) == 0.25
    assert model.worker_scale(1000) == 0.25


def test_worker_policy_gradient_activates_gradually():
    model = policy()
    assert model.worker_pg_blend(0) == 0.0
    assert model.worker_pg_blend(25) == 0.0
    assert math.isclose(model.worker_pg_blend(62.5), 0.5)
    assert model.worker_pg_blend(100) == 1.0


def test_termination_blend_schedule():
    model = policy()
    assert model.termination_blend(0) == 0.0
    assert model.termination_blend(50) == 0.0
    assert math.isclose(model.termination_blend(100), 0.5)
    assert model.termination_blend(150) == 1.0


def test_minimum_duration_is_never_violated():
    model = policy()
    model.set_training_step(1_000)
    feat = torch.randn(4, 12)
    option = torch.tensor([0, 1, 2, 3])
    has = torch.ones(4, dtype=torch.bool)
    first = torch.zeros(4, dtype=torch.bool)
    for age_value in (0,):
        step = model.step_option(
            feat,
            option,
            torch.full((4,), age_value),
            has,
            first,
            deterministic=False,
            termination_uniform=torch.zeros(4),
        )
        assert not step.option_terminated.any()
        assert torch.equal(step.option, option)


def test_maximum_duration_always_terminates():
    model = policy()
    model.set_training_step(1_000)
    feat = torch.randn(4, 12)
    previous = torch.tensor([0, 1, 2, 3])
    step = model.step_option(
        feat,
        previous,
        torch.full((4,), 20),
        torch.ones(4, dtype=torch.bool),
        torch.zeros(4, dtype=torch.bool),
        deterministic=False,
        termination_uniform=torch.ones(4),
        manager_uniform=torch.tensor([0.01, 0.2, 0.4, 0.8]),
    )
    assert step.option_terminated.all()
    assert step.option_started.all()
    assert torch.equal(step.action_age, torch.zeros(4, dtype=torch.long))
    assert torch.equal(step.carry_age, torch.ones(4, dtype=torch.long))


def test_same_option_reselection_is_allowed():
    model = policy()
    model.set_training_step(1_000)
    with torch.no_grad():
        model.manager_group[-1].weight.zero_()
        model.manager_group[-1].bias.copy_(torch.tensor([-10.0, 10.0]))
        model.manager_slot[-1].weight.zero_()
        model.manager_slot[-1].bias.fill_(-10.0)
        model.manager_slot[-1].bias[3] = 10.0
    feat = torch.randn(1, 12)
    step = model.step_option(
        feat,
        torch.tensor([3]),
        torch.tensor([20]),
        torch.ones(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
        deterministic=True,
    )
    assert step.option_terminated.item()
    assert step.option.item() == 3
    assert step.option_started.item()


def test_new_episode_selects_fresh_option_and_resets_age():
    model = policy()
    feat = torch.randn(3, 12)
    step = model.step_option(
        feat,
        torch.tensor([4, 4, 4]),
        torch.tensor([9, 9, 9]),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        deterministic=True,
    )
    assert step.option_started.all()
    assert not step.option_terminated.any()
    assert torch.equal(step.action_age, torch.zeros(3, dtype=torch.long))


def test_preservation_phase_reselects_at_every_eligible_state():
    model = policy()
    feat = torch.randn(5, 12)
    option = torch.zeros(5, dtype=torch.long)
    age = torch.full((5,), 1)
    beta, eligible, _, _ = model.effective_termination_probability(
        feat, option, age, step=0
    )
    assert eligible.all()
    assert torch.equal(beta, torch.ones_like(beta))


def test_eligible_probability_uses_fixed_hazard_without_preservation_override():
    model = policy(
        commitment_reselect_initial=0.0,
        commitment_reselect_final=0.0,
    )
    feat = torch.randn(5, 12)
    option = torch.zeros(5, dtype=torch.long)
    age = torch.full((5,), 5)
    beta, eligible, _, _ = model.effective_termination_probability(
        feat, option, age, step=0
    )
    assert eligible.all()
    assert torch.allclose(beta, torch.full_like(beta, 0.1), atol=1e-6)


def test_action_statistics_are_finite_and_masked():
    model = policy()
    model.set_training_step(100)
    feat = torch.randn(16, 12)
    base = torch.randn(16, 15)
    option = torch.arange(16) % 8
    mask = torch.ones(16, 3, 5, dtype=torch.bool)
    mask[:, :, -1] = False
    active = torch.ones(16, 3, dtype=torch.bool)
    stats = model.behaviour_statistics(
        feat, base, base.clone(), option, mask, active, (5, 5, 5), None, 100
    )
    for value in stats.values():
        assert torch.isfinite(value).all()
    assert stats["base_kl_mean"] >= 0
    assert 0 <= stats["action_flip_rate"] <= 1


def test_bfloat16_path_is_finite():
    model = policy().eval()
    model.set_training_step(100)
    feat = torch.randn(8, 12)
    base = torch.randn(8, 15, dtype=torch.bfloat16)
    option = torch.arange(8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model.combine_logits(base, feat, option)
        beta = model.learned_termination_probability(
            feat, option, torch.arange(8) % 5
        )
    assert torch.isfinite(logits.float()).all()
    assert torch.isfinite(beta.float()).all()


def test_group_residuals_and_combined_residuals_are_hard_bounded():
    model = policy(max_abs_residual_logit=0.5, worker_scale_max=0.25)
    model.set_training_step(1000)
    with torch.no_grad():
        model.worker_residual[-1].weight.mul_(1000)
        model.slot_delta[-1].weight.mul_(1000)
    feat = torch.randn(9, 12)
    group = model._all_group_residual_logits(feat)
    residual = model.all_residual_logits(feat)
    assert group.abs().max() <= 0.1250001
    assert residual.abs().max() <= 0.5000001
    assert torch.allclose(group.sum(dim=-2), torch.zeros(9, 15), atol=2e-6)


def test_eval_uses_deterministic_cumulative_hazard():
    model = policy(
        eval_sample_termination=False,
        commitment_reselect_initial=0.0,
        commitment_reselect_final=0.0,
    )
    model.set_training_step(0)
    feat = torch.randn(1, 12)
    option = torch.tensor([1])
    age = torch.tensor([3])
    has = torch.ones(1, dtype=torch.bool)
    first = torch.zeros(1, dtype=torch.bool)
    hazard = torch.zeros(1)
    # Fixed beta=0.1 should not terminate immediately and should terminate
    # deterministically after roughly ten eligible decisions.
    seen = None
    for index in range(12):
        step = model.step_option(
            feat, option, age, has, first, deterministic=True,
            termination_hazard=hazard,
        )
        if step.option_terminated.item():
            seen = index + 1
            break
        hazard = step.carry_termination_hazard
        age = step.carry_age
    assert seen == 10


def test_behavior_statistics_honors_zero_state_weights():
    model = policy()
    model.set_training_step(100)
    feat = torch.randn(4, 12)
    base = torch.randn(4, 15)
    option = torch.arange(4)
    mask = torch.ones(4, 3, 5, dtype=torch.bool)
    active = torch.ones(4, 3, dtype=torch.bool)
    weighted = model.behaviour_statistics(
        feat, base, base.clone(), option, mask, active, (5, 5, 5),
        torch.tensor([1.0, 0.0, 0.0, 0.0]), 100
    )
    single = model.behaviour_statistics(
        feat[:1], base[:1], base[:1].clone(), option[:1], mask[:1], active[:1],
        (5, 5, 5), torch.ones(1), 100
    )
    assert torch.allclose(weighted["base_kl_mean"], single["base_kl_mean"], atol=1e-6)


def test_replay_predecision_state_reproduces_the_current_option_decision():
    """Replay must store the state entering act(), not post-action carry age."""
    model = policy()
    model.set_training_step(1000)
    feat = torch.randn(3, 12)
    before_option = torch.tensor([1, 2, 3])
    before_age = torch.tensor([3, 7, 20])
    before_has = torch.ones(3, dtype=torch.bool)
    first = torch.zeros(3, dtype=torch.bool)
    termination_u = torch.tensor([0.0, 1.0, 1.0])
    manager_u = torch.tensor([0.1, 0.5, 0.9])

    real = model.step_option(
        feat,
        before_option,
        before_age,
        before_has,
        first,
        deterministic=False,
        termination_uniform=termination_u,
        manager_uniform=manager_u,
    )
    replay_reconstructed = model.step_option(
        feat,
        before_option,
        before_age,
        before_has,
        first,
        deterministic=False,
        termination_uniform=termination_u,
        manager_uniform=manager_u,
    )
    for left, right in zip(real, replay_reconstructed):
        assert torch.equal(left, right)

    # Starting from carry_age would incorrectly make the same posterior state
    # one primitive action older, and can trigger an early boundary.
    wrong = model.step_option(
        feat,
        real.option,
        real.carry_age,
        real.has_option,
        first,
        deterministic=False,
        termination_uniform=termination_u,
        manager_uniform=manager_u,
    )
    assert not torch.equal(real.previous_age, wrong.previous_age)


def test_manager_mi_guard_detects_state_independent_uniform_routing():
    model = policy()
    probs = torch.full((16, 8), 1.0 / 8.0)
    sampled = torch.arange(16) % 8
    boundary = torch.ones(16, dtype=torch.bool)
    stats = model.manager_statistics(probs, sampled, boundary)
    assert stats["mutual_information_normalized"] < 1.0e-6
    assert stats["mi_shortfall_loss"] > 0


def test_manager_mi_guard_is_inactive_for_state_dependent_routing():
    model = policy()
    probs = torch.nn.functional.one_hot(torch.arange(16) % 8, 8).float()
    sampled = torch.arange(16) % 8
    boundary = torch.ones(16, dtype=torch.bool)
    stats = model.manager_statistics(probs, sampled, boundary)
    assert stats["mutual_information_normalized"] > 0.99
    assert stats["mi_shortfall_loss"] == 0


def test_manager_pg_blend_starts_after_preservation_warmup():
    model = policy()
    assert model.manager_pg_blend(0) == 0.0
    assert model.manager_pg_blend(50) == 0.0
    assert math.isclose(model.manager_pg_blend(100), 0.5)
    assert model.manager_pg_blend(150) == 1.0


def test_manager_unimix_starts_source_exact_and_ramps_up():
    model = policy()
    assert model.manager_unimix(0) == 0.0
    assert math.isclose(model.manager_unimix(100), 0.01)
    assert model.manager_unimix(200) == 0.02


def test_commitment_reselection_decays_continuously():
    model = policy()
    assert model.commitment_reselect_probability(0) == 1.0
    assert model.commitment_reselect_probability(50) == 1.0
    assert math.isclose(model.commitment_reselect_probability(100), 0.5)
    assert model.commitment_reselect_probability(150) == 0.0


def test_variable_imagination_horizon_is_bounded_and_checkpointable():
    model = policy()
    model.set_training_step(0)
    assert [model.next_imagination_horizon() for _ in range(4)] == [5, 6, 7, 8]
    model.set_training_step(200)
    model.set_horizon_calls(0)
    assert [model.next_imagination_horizon() for _ in range(4)] == [7, 8, 9, 10]
    assert int(model.horizon_calls.item()) == 4


def test_world_model_gradient_scale_is_frozen_for_conservative_phase():
    model = policy()
    assert model.world_model_grad_scale(0) == 0.0
    assert model.world_model_grad_scale(100) == 0.0
    assert model.world_model_grad_scale(150) == 0.0
    assert model.world_model_grad_scale(200) == 0.0


def test_termination_embedding_is_gradient_isolated_from_worker():
    model = policy()
    model.set_training_step(1000)
    feat = torch.randn(6, 12)
    option = torch.arange(6) % 8
    worker_loss = model.residual_logits(feat, option).square().sum()
    worker_loss.backward()
    worker_grad = model.option_embedding.weight.grad
    term_grad = model.termination_option_embedding.weight.grad
    assert worker_grad is not None and torch.count_nonzero(worker_grad) > 0
    assert term_grad is None or torch.count_nonzero(term_grad) == 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.termination[-1].weight.normal_(0.0, 0.1)
    beta = model.learned_termination_probability(
        feat, option, torch.full((6,), 5)
    )
    beta.sum().backward()
    worker_grad = model.option_embedding.weight.grad
    term_grad = model.termination_option_embedding.weight.grad
    assert term_grad is not None and torch.count_nonzero(term_grad) > 0
    assert worker_grad is None or torch.count_nonzero(worker_grad) == 0


def test_invalid_action_logits_cannot_fake_residual_magnitude_or_diversity():
    import types

    model = policy()
    model.set_training_step(1000)
    n = 4
    feat = torch.randn(n, 12)
    base = torch.zeros(n, 15)
    option = torch.arange(n) % 8
    # Only action 0 is valid for each of three agents.
    mask = torch.zeros(n, 3, 5, dtype=torch.bool)
    mask[..., 0] = True
    active = torch.ones(n, 3, dtype=torch.bool)

    def fake_all(self, local_feat, step=None):
        out = torch.zeros(local_feat.shape[0], 8, 15)
        # Large option-specific changes exist only in invalid action slots.
        for k in range(8):
            shaped = out[:, k].reshape(local_feat.shape[0], 3, 5)
            shaped[..., 1:] = float(k + 1)
        return out.to(local_feat.device)

    model.all_residual_logits = types.MethodType(fake_all, model)
    stats = model.behaviour_statistics(
        feat, base, base.clone(), option, mask, active, (5, 5, 5), torch.ones(n), 1000
    )
    assert stats["base_kl_mean"] == 0
    assert stats["js_mean"] == 0
    assert stats["residual_rms"] == 0
    assert stats["residual_cosine_mean"] == 0


def test_diversity_hinge_penalizes_duplicate_pairs_even_when_mean_js_is_high():
    import types

    model = policy(
        action_diversity_target=0.001,
        max_diversity_pairs=28,
    )
    model.set_training_step(1000)
    n = 6
    feat = torch.randn(n, 12)
    base = torch.zeros(n, 15)
    selected = torch.zeros(n, dtype=torch.long)
    mask = torch.ones(n, 3, 5, dtype=torch.bool)
    active = torch.ones(n, 3, dtype=torch.bool)

    def fake_all(self, local_feat, step=None):
        out = torch.zeros(local_feat.shape[0], 8, 15, device=local_feat.device)
        # Options 0 and 1 are exact duplicates. The remaining options are made
        # strongly distinct, so a hinge on only mean JS would miss the duplicate.
        for k in range(2, 8):
            shaped = out[:, k].reshape(local_feat.shape[0], 3, 5)
            shaped[..., k % 5] = 4.0
        return out

    model.all_residual_logits = types.MethodType(fake_all, model)
    stats = model.behaviour_statistics(
        feat, base, base.clone(), selected, mask, active, (5, 5, 5), torch.ones(n), 1000
    )
    assert stats["js_mean"] > model.settings.action_diversity_target
    assert stats["js_shortfall_fraction"] > 0
    assert stats["diversity_loss"] > 0


def test_tail_kl_and_high_confidence_action_guard_catch_rare_damage():
    import types

    model = policy(
        max_diversity_states=128,
        base_kl_tail_fraction=0.25,
        action_preservation_confidence=0.7,
    )
    model.set_training_step(1000)
    n = 8
    feat = torch.randn(n, 12)
    base = torch.zeros(n, 15)
    reference = torch.zeros(n, 15)
    reference[:, 0::5] = 8.0
    option = torch.zeros(n, dtype=torch.long)
    mask = torch.ones(n, 3, 5, dtype=torch.bool)
    active = torch.ones(n, 3, dtype=torch.bool)

    def damaging_residuals(self, local_feat, step=None):
        out = torch.zeros(local_feat.shape[0], 8, 15, device=local_feat.device)
        shaped = out[:, 0].reshape(local_feat.shape[0], 3, 5)
        shaped[0, ..., 1] = 10.0
        return out

    model.all_residual_logits = types.MethodType(damaging_residuals, model)
    stats = model.behaviour_statistics(
        feat, base, reference, option, mask, active, (5, 5, 5), torch.ones(n), 1000
    )
    assert stats["base_kl_tail"] > stats["base_kl_mean"]
    assert stats["high_confidence_flip_rate"] > 0
    assert stats["action_preservation_loss"] > 0
    assert stats["base_kl_loss"] > 0


def test_termination_execution_cap_relaxes_continuously_without_full_blend_jump():
    model = policy(
        commitment_reselect_initial=0.0,
        commitment_reselect_final=0.0,
    )
    # Learned-beta blend becomes full at 150, but the execution cap remains at
    # 0.30 there and only reaches 0.80 at cap_full=250.
    assert math.isclose(model.termination_probability_cap(149), 0.30)
    assert math.isclose(model.termination_probability_cap(150), 0.30)
    assert math.isclose(model.termination_probability_cap(200), 0.55)
    assert math.isclose(model.termination_probability_cap(250), 0.80)

    with torch.no_grad():
        model.termination[-1].weight.zero_()
        model.termination[-1].bias.fill_(20.0)
    feat = torch.randn(4, 12)
    option = torch.zeros(4, dtype=torch.long)
    age = torch.full((4,), 5)
    before, *_ = model.effective_termination_probability(feat, option, age, 149)
    at_full, *_ = model.effective_termination_probability(feat, option, age, 150)
    later, *_ = model.effective_termination_probability(feat, option, age, 200)
    assert before.max() <= 0.300001
    assert at_full.max() <= 0.300001
    assert later.max() <= 0.550001


def test_initialized_bounded_termination_matches_fixed_hazard():
    model = policy(
        commitment_reselect_initial=0.0,
        commitment_reselect_final=0.0,
    )
    feat = torch.randn(5, 12)
    option = torch.arange(5) % 8
    age = torch.full((5,), 5)
    beta, eligible, _, _ = model.effective_termination_probability(
        feat, option, age, step=150
    )
    assert eligible.all()
    assert torch.allclose(beta, torch.full_like(beta, 0.10), atol=1e-6)


def test_executed_termination_probability_has_warmup_gate_and_smooth_cap_gradients():
    model = policy(
        commitment_reselect_initial=0.0,
        commitment_reselect_final=0.0,
    )
    feat = torch.zeros(1, 12)
    option = torch.zeros(1, dtype=torch.long)
    age = torch.full((1,), 5, dtype=torch.long)
    with torch.no_grad():
        model.termination[-1].weight.zero_()
        model.termination[-1].bias.fill_(0.0)  # raw sigmoid = 0.5

    model.set_training_step(0)
    beta_warm, eligible, _, _ = model.effective_termination_probability(
        feat, option, age
    )
    assert eligible.all()
    beta_warm.sum().backward()
    warm_grad = model.termination[-1].bias.grad
    assert warm_grad is not None and warm_grad.item() == 0.0

    model.zero_grad(set_to_none=True)
    model.set_training_step(100)  # blend = 0.5, cap = 0.30
    # raw beta=0.2 stays below the cap after 2% unimix.
    raw_beta = 0.2
    model.termination[-1].bias.data.fill_(math.log(raw_beta / (1.0 - raw_beta)))
    beta_mid, _, _, _ = model.effective_termination_probability(
        feat, option, age
    )
    beta_mid.sum().backward()
    cap = model.termination_probability_cap(100)
    unimix = model.settings.termination_unimix
    probability = (1.0 - unimix) * raw_beta + 0.5 * unimix
    temperature = model.settings.termination_soft_cap_temperature
    soft_cap_derivative = 1.0 - torch.sigmoid(
        torch.tensor((probability - cap) / temperature)
    ).item()
    expected = (
        0.5
        * (1.0 - unimix)
        * soft_cap_derivative
        * raw_beta
        * (1.0 - raw_beta)
    )
    assert torch.allclose(
        model.termination[-1].bias.grad, torch.tensor(expected), atol=1e-7
    )

    model.zero_grad(set_to_none=True)
    model.set_training_step(200)  # full blend, execution cap=0.55
    model.termination[-1].bias.data.fill_(3.0)
    beta_capped, _, _, _ = model.effective_termination_probability(
        feat, option, age
    )
    beta_capped.sum().backward()
    capped_grad = model.termination[-1].bias.grad
    assert capped_grad is not None and capped_grad.item() > 0.0
    assert beta_capped.item() < model.termination_probability_cap(200)


def test_smooth_cap_relaxation_does_not_raise_low_termination_probability():
    model = policy(
        commitment_reselect_initial=0.0,
        commitment_reselect_final=0.0,
    )
    feat = torch.zeros(3, 12)
    option = torch.zeros(3, dtype=torch.long)
    age = torch.full((3,), 5, dtype=torch.long)
    raw_beta = 0.10
    with torch.no_grad():
        model.termination[-1].weight.zero_()
        model.termination[-1].bias.fill_(
            math.log(raw_beta / (1.0 - raw_beta))
        )
    low_cap = model.bounded_learned_termination_probability(
        feat, option, age, step=150
    )
    high_cap = model.bounded_learned_termination_probability(
        feat, option, age, step=250
    )
    # Raising the cap must not scale a low beta upward. The tiny difference is
    # only the smooth-min approximation error.
    assert (high_cap - low_cap).abs().max().item() < 1.0e-4


def test_manager_grouping_matches_migrated_even_odd_layout():
    model = policy()
    probs = torch.tensor([[0.10, 0.05, 0.20, 0.10, 0.15, 0.15, 0.05, 0.20]])
    grouped = model.grouped_manager_probs(probs)
    assert torch.allclose(grouped, torch.tensor([[0.50, 0.50]]))


def test_manager_source_trust_region_detects_group_routing_swap_and_backpropagates():
    model = policy()
    source = torch.tensor([[0.2375, 0.0125] * 4], dtype=torch.float32)
    live_logits = torch.tensor(
        [[-2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0]],
        requires_grad=True,
    )
    live = live_logits.softmax(dim=-1)
    stats = model.manager_source_statistics(live, source)
    assert stats["kl_mean"].item() > 1.0
    assert stats["flip_rate"].item() == 1.0
    total = stats["kl_loss"] + stats["preservation_loss"]
    total.backward()
    assert live_logits.grad is not None
    assert torch.isfinite(live_logits.grad).all()
    assert live_logits.grad.abs().sum().item() > 0.0


def test_manager_source_trust_region_is_zero_for_identical_group_routing():
    model = policy()
    source = torch.tensor([[0.15, 0.10, 0.20, 0.05, 0.10, 0.15, 0.05, 0.20]])
    stats = model.manager_source_statistics(source, source)
    assert torch.allclose(stats["kl_mean"], torch.zeros_like(stats["kl_mean"]), atol=1e-7)
    assert stats["flip_rate"].item() == 0.0
    assert stats["preservation_loss"].item() == 0.0


def test_source_forward_kl_penalizes_dropping_source_supported_actions():
    model = policy(base_kl_target=10.0, base_kl_tail_target=10.0)
    n = 8
    feat = torch.zeros(n, 12)
    base = torch.zeros(n, 15)
    reference = torch.zeros(n, 15)
    option = torch.zeros(n, dtype=torch.long)
    mask = torch.ones(n, 3, 5, dtype=torch.bool)
    active = torch.ones(n, 3, dtype=torch.bool)

    # Source gives meaningful mass to actions 0 and 1, while the live policy
    # collapses almost entirely onto action 0. Forward KL must remain positive.
    reference_shaped = reference.reshape(n, 3, 5)
    reference_shaped[..., 0] = 1.0
    reference_shaped[..., 1] = 1.0

    def collapsed_residuals(self, local_feat, step=None):
        out = torch.zeros(local_feat.shape[0], 8, 15, device=local_feat.device)
        out[:, 0].reshape(local_feat.shape[0], 3, 5)[..., 0] = 10.0
        return out

    import types
    model.all_residual_logits = types.MethodType(collapsed_residuals, model)
    stats = model.behaviour_statistics(
        feat, base, reference, option, mask, active, (5, 5, 5), torch.ones(n), 0
    )
    assert stats["base_kl_mean"] > 0.1
    # Targets are deliberately huge, so this proves the always-on distillation
    # component remains active even when the squared hinge is inactive.
    assert stats["base_kl_loss"] > 0.1


def test_manager_source_distillation_is_always_on_inside_hinge_threshold():
    model = policy(
        manager_group_kl_target=1.0,
        manager_group_kl_tail_target=1.0,
    )
    source = torch.tensor([[0.26, 0.24, 0.26, 0.24, 0.0, 0.0, 0.0, 0.0]])
    live = torch.tensor([[0.255, 0.245, 0.255, 0.245, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    stats = model.manager_source_statistics(live, source, torch.ones(1))
    assert stats["kl_mean"] > 0
    assert stats["kl_loss"] > 0
    stats["kl_loss"].backward()
    assert live.grad is not None
    assert torch.isfinite(live.grad).all()


def test_v5_stability_schedule_retains_reactive_reselection_floor():
    model = policy(
        num_options=2,
        min_effective_options=1.0,
        max_duration=8,
        commitment_warmup_steps=100,
        commitment_full_steps=600,
        commitment_reselect_final=0.25,
        termination_warmup_steps=350,
        termination_full_steps=800,
        termination_max_probability_final=0.30,
        termination_cap_full_steps=900,
        manager_pg_warmup_steps=100,
        manager_pg_full_steps=500,
        worker_pg_warmup_steps=20,
        worker_pg_full_steps=150,
        source_manager_group_count=2,
        imag_horizon_initial_max=10,
        imag_horizon_final_max=12,
        imag_horizon_window=4,
        imag_horizon_ramp_steps=600,
    )
    assert model.commitment_reselect_probability(0) == 1.0
    assert math.isclose(model.commitment_reselect_probability(600), 0.25)
    assert math.isclose(model.commitment_reselect_probability(1_000_000), 0.25)
    assert model.termination_probability_cap(1_000_000) == 0.30
    assert model.settings.max_duration == 8
    lo, hi = model.active_imagination_horizon_range(1_000_000)
    assert (lo, hi) == (9, 12)


def test_randomized_call_and_return_state_machine_invariants():
    model = policy(
        num_options=2,
        min_effective_options=1.0,
        max_duration=8,
        source_manager_group_count=2,
        commitment_reselect_initial=0.0,
        commitment_reselect_final=0.0,
    )
    torch.manual_seed(123)
    batch = 64
    option = torch.zeros(batch, dtype=torch.long)
    age = torch.zeros(batch, dtype=torch.long)
    has = torch.zeros(batch, dtype=torch.bool)
    first = torch.ones(batch, dtype=torch.bool)
    hazard = torch.zeros(batch)
    for _ in range(200):
        step = model.step_option(
            torch.randn(batch, 12), option, age, has, first,
            deterministic=False, termination_hazard=hazard,
        )
        assert (step.option >= 0).all() and (step.option < 2).all()
        assert (step.action_age >= 0).all()
        assert (step.action_age < model.settings.max_duration).all()
        assert (step.carry_age >= 1).all()
        assert (step.carry_age <= model.settings.max_duration).all()
        assert not (
            step.option_terminated & (step.previous_age < model.settings.min_duration)
        ).any()
        assert not (
            (step.previous_age >= model.settings.max_duration)
            & (~step.option_terminated)
            & (~first)
            & has
        ).any()
        changed = step.option != step.previous_option
        assert not (changed & (~step.option_started)).any()
        option, age, has = step.option, step.carry_age, step.has_option
        hazard = step.carry_termination_hazard
        first = torch.rand(batch) < 0.02


def test_locked_slots_have_zero_probability_and_zero_pg_maturity():
    model = policy()
    feat = torch.randn(6, 12)
    probs = model.manager_probs(feat, step=0)
    assert torch.count_nonzero(probs[:, 2:]) == 0
    maturity = model.slot_pg_blend_for_option(torch.arange(8), step=0)
    assert torch.equal(maturity[:2], torch.ones(2))
    assert torch.count_nonzero(maturity[2:]) == 0


def test_factorized_manager_matches_group_times_conditional_slot_probability():
    model = policy()
    feat = torch.randn(9, 12)
    step = 125
    group = model.manager_group_probs(feat, step)
    slot = model.manager_slot_probs(feat, step)
    full = model.manager_probs(feat, step)
    expected = (group.unsqueeze(-1) * slot).transpose(-1, -2).reshape(9, 8)
    assert torch.allclose(full, expected, atol=1e-7, rtol=1e-6)
    assert torch.allclose(model.grouped_manager_probs(full), group, atol=1e-7)


def test_locked_child_delta_cannot_change_source_policy_before_unlock():
    model = policy()
    with torch.no_grad():
        model.slot_delta[-1].weight.normal_(0.0, 100.0)
        model.slot_delta[-1].bias.normal_(0.0, 100.0)
    feat = torch.randn(7, 12)
    residual = model.all_residual_logits(feat, step=0)
    group = model._all_group_residual_logits(feat, step=0)
    for option in range(8):
        assert torch.allclose(residual[:, option], group[:, option % 2], atol=1e-7)


def test_child_learned_termination_waits_for_slot_maturity():
    model = policy(
        commitment_reselect_initial=0.0,
        commitment_reselect_final=0.0,
        termination_warmup_steps=0,
        termination_full_steps=1,
        termination_cap_full_steps=2,
    )
    with torch.no_grad():
        model.termination[-1].bias.fill_(5.0)
    feat = torch.randn(2, 12)
    option = torch.tensor([0, 2])
    age = torch.full((2,), 3)
    beta, eligible, _, _ = model.effective_termination_probability(
        feat, option, age, step=50
    )
    assert eligible.all()
    assert beta[0] > 0.2
    # Child slot 1 has just unlocked at step 50, so learned beta has no causal
    # effect yet and the fixed 0.1 hazard is retained.
    assert torch.allclose(beta[1], torch.tensor(0.1), atol=1e-6)
