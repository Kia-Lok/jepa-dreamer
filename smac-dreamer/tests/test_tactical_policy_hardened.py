import math

import torch

from tactical_policy import TacticalMixturePolicy


CFG = {
    "enabled": True,
    "num_tactics": 4,
    "embedding_dim": 8,
    "hidden_dim": 16,
    "max_effect_states": 64,
    "duration": 1,
    "symmetry_break_std": 1.0e-2,
    "max_usage_target": 0.8,
    "min_effective_tactics": 2.0,
    "eval_confidence_threshold": 0.55,
    "freeze_base_actor": True,
    "freeze_feature_adapter": True,
    "residual_guard_scale": 1.0e-3,
    "max_residual_to_base": 1.0,
    "max_abs_residual_logit": 4.0,
}


def make_policy():
    torch.manual_seed(7)
    return TacticalMixturePolicy(12, 15, CFG)


def test_selector_uniform_but_residual_has_tiny_symmetry_break():
    policy = make_policy()
    feat = torch.randn(32, 12)
    probs = policy.selector_dist(feat).probs
    assert torch.allclose(probs, torch.full_like(probs, 0.25), atol=1e-7)

    base = torch.randn(32, 15)
    logits = policy.all_combined_logits(base, feat)
    residual = logits - base.unsqueeze(1)
    assert torch.isfinite(residual).all()
    assert float(residual.detach().abs().max()) < 0.05
    assert float(residual.detach().std()) > 0.0


def test_first_backward_reaches_embedding_and_residual():
    policy = make_policy()
    feat = torch.randn(64, 12)
    base = torch.randn(64, 15)
    dist = policy.selector_dist(feat)
    tactic = dist.sample()
    advantage = torch.linspace(-1, 1, 64)
    combined = policy.combine_logits(base, feat, tactic)
    loss = -(dist.log_prob(tactic) * advantage).mean() + combined.square().mean()
    loss.backward()
    assert policy.selector[-1].weight.grad is not None
    assert policy.residual[-1].weight.grad is not None
    assert policy.embedding.weight.grad is not None
    assert torch.isfinite(policy.embedding.weight.grad).all()


def test_usage_statistics_distinguish_uniform_from_specialized_balanced():
    policy = make_policy()

    uniform_logits = torch.zeros(100, 4)
    uniform = policy.usage_statistics(uniform_logits)
    assert torch.allclose(
        uniform["marginal"], torch.full((4,), 0.25), atol=1e-7
    )
    assert abs(float(uniform["effective_count"]) - 4.0) < 1e-6
    assert float(uniform["mutual_information"]) < 1e-6
    assert float(uniform["collapse_loss"]) < 1e-7

    specialized_logits = torch.full((100, 4), -20.0)
    for index in range(100):
        specialized_logits[index, index % 4] = 20.0
    specialized = policy.usage_statistics(specialized_logits)
    assert abs(float(specialized["effective_count"]) - 4.0) < 1e-4
    assert float(specialized["conditional_entropy"]) < 1e-4
    assert abs(float(specialized["mutual_information"]) - math.log(4)) < 1e-4
    assert float(specialized["collapse_loss"]) < 1e-7


def test_collapse_loss_is_hinge_not_uniformity_penalty():
    policy = make_policy()

    moderate = torch.tensor([[2.0, 1.0, 0.0, -1.0]]).expand(128, -1)
    moderate_stats = policy.usage_statistics(moderate)
    assert float(moderate_stats["usage_max"]) < 0.8
    assert float(moderate_stats["collapse_loss"]) < 1e-7

    collapsed = torch.full((128, 4), -20.0)
    collapsed[:, 0] = 20.0
    collapsed_stats = policy.usage_statistics(collapsed)
    assert float(collapsed_stats["usage_max"]) > 0.99
    assert float(collapsed_stats["effective_count"]) < 1.01
    assert float(collapsed_stats["collapse_loss"]) > 0.0


def test_effect_auxiliary_does_not_backprop_into_base_actor_logits():
    policy = make_policy()
    feat = torch.randn(16, 12)
    base = torch.randn(16, 15, requires_grad=True)
    mask = torch.ones(16, 3, 5, dtype=torch.bool)
    active = torch.ones(16, 3, dtype=torch.bool)

    stats = policy.effect_statistics(feat, base, mask, active, (5, 5, 5))
    loss = -stats["js_mean"]
    loss.backward()

    assert base.grad is None or torch.count_nonzero(base.grad) == 0
    assert policy.residual[-1].weight.grad is not None
    assert torch.isfinite(policy.residual[-1].weight.grad).all()


def test_effect_statistics_repair_empty_masks_and_are_finite():
    policy = make_policy()
    feat = torch.randn(10, 12)
    base = torch.randn(10, 15)
    mask = torch.zeros(10, 3, 5, dtype=torch.bool)
    active = torch.ones(10, 3, dtype=torch.bool)
    stats = policy.effect_statistics(feat, base, mask, active, (5, 5, 5))
    for value in stats.values():
        assert torch.isfinite(value).all()
    assert float(stats["js_mean"].detach()) >= 0.0


def test_sampled_and_argmax_usage_are_reported_separately():
    policy = make_policy()
    logits = torch.zeros(8, 4)
    sampled = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    stats = policy.usage_statistics(logits, sampled_tactic=sampled)
    assert torch.allclose(stats["sampled_usage"], torch.full((4,), 0.25))
    # Argmax ties resolve to index zero; this is exactly why the metric matters.
    assert torch.allclose(
        stats["argmax_usage"], torch.tensor([1.0, 0.0, 0.0, 0.0])
    )


def test_metadata_and_initialization_guard():
    policy = make_policy()
    policy.assert_legacy_equivalence_ready()
    meta = policy.metadata()
    assert meta["schema_version"] == 2
    assert meta["architecture"] == "tactical_mixture_v1_1"
    assert meta["feature_dim"] == 12
    assert meta["action_logit_dim"] == 15
    assert meta["freeze_base_actor"] is True
    assert meta["freeze_feature_adapter"] is True
    assert meta["max_residual_to_base"] == 1.0
    assert meta["max_abs_residual_logit"] == 4.0


def test_effect_auxiliary_detaches_feature_tensor_too():
    policy = make_policy()
    feat = torch.randn(12, 12, requires_grad=True)
    base = torch.randn(12, 15, requires_grad=True)
    mask = torch.ones(12, 3, 5, dtype=torch.bool)
    active = torch.ones(12, 3, dtype=torch.bool)
    stats = policy.effect_statistics(feat, base, mask, active, (5, 5, 5))
    (-stats["js_mean"]).backward()
    assert feat.grad is None or torch.count_nonzero(feat.grad) == 0
    assert base.grad is None or torch.count_nonzero(base.grad) == 0
    assert policy.residual[-1].weight.grad is not None


def test_zero_state_weights_are_finite_and_report_collapse_not_nan():
    policy = make_policy()
    logits = torch.randn(7, 4)
    stats = policy.usage_statistics(logits, state_weights=torch.zeros(7))
    for value in stats.values():
        assert torch.isfinite(value).all()
    assert abs(float(stats["effective_count"].detach()) - 1.0) < 1e-5
    assert float(stats["collapse_loss"].detach()) > 0.0


def test_bfloat16_autocast_paths_are_finite():
    policy = make_policy()
    feat = torch.randn(16, 12)
    base = torch.randn(16, 15)
    mask = torch.ones(16, 3, 5, dtype=torch.bool)
    active = torch.ones(16, 3, dtype=torch.bool)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        dist = policy.selector_dist(feat)
        tactic = dist.sample()
        combined = policy.combine_logits(base, feat, tactic)
        stats = policy.effect_statistics(feat, base, mask, active, (5, 5, 5))
        loss = combined.float().square().mean() - stats["js_mean"]
    loss.backward()
    assert torch.isfinite(combined.float()).all()
    for value in stats.values():
        assert torch.isfinite(value).all()
    for parameter in policy.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_policy_parameter_registry_has_no_internal_aliases():
    policy = make_policy()
    identifiers = [id(parameter) for parameter in policy.parameters()]
    assert len(identifiers) == len(set(identifiers))


def test_deterministic_eval_uses_smooth_confidence_gate():
    policy = make_policy()
    feat = torch.randn(20, 12)
    base = torch.randn(20, 15)
    final, tactic, confidence, gate = policy.eval_combined_logits(base, feat)
    assert torch.equal(tactic, torch.zeros_like(tactic))
    assert torch.allclose(confidence, torch.full_like(confidence, 0.25))
    assert torch.equal(gate, torch.zeros_like(gate))
    assert torch.equal(final, base)

    with torch.no_grad():
        # Moderate confidence blends only part of the tactical residual.
        policy.selector[-1].bias.copy_(torch.tensor([0.6, 0.0, 0.0, 0.0]))
    partial, tactic, confidence, gate = policy.eval_combined_logits(base, feat)
    conditioned = policy.combine_logits(base, feat, tactic)
    assert ((gate > 0) & (gate < 1)).all()
    expected = base + gate.unsqueeze(-1) * (conditioned - base)
    assert torch.allclose(partial, expected)

    with torch.no_grad():
        policy.selector[-1].bias.copy_(torch.tensor([8.0, -8.0, -8.0, -8.0]))
    final, tactic, confidence, gate = policy.eval_combined_logits(base, feat)
    assert torch.equal(gate, torch.ones_like(gate))
    assert (tactic == 0).all()
    assert (confidence > 0.99).all()
    assert not torch.equal(final, base)


def test_effect_statistics_all_zero_weights_return_zero_without_nan():
    policy = make_policy()
    feat = torch.randn(10, 12)
    base = torch.randn(10, 15)
    mask = torch.ones(10, 3, 5, dtype=torch.bool)
    active = torch.ones(10, 3, dtype=torch.bool)
    stats = policy.effect_statistics(
        feat, base, mask, active, (5, 5, 5), state_weights=torch.zeros(10)
    )
    for value in stats.values():
        assert torch.isfinite(value).all()
    assert float(stats["js_mean"].detach()) == 0.0
    assert float(stats["residual_rms"].detach()) == 0.0
    assert float(stats["base_rms"].detach()) == 0.0


def test_effect_statistics_has_no_data_dependent_nonzero_path():
    import inspect

    source = inspect.getsource(TacticalMixturePolicy.effect_statistics)
    assert "torch.nonzero" not in source
    assert "empty.any()" not in source


def test_effect_statistics_fullgraph_compiles_without_data_dependent_branch():
    policy = make_policy()

    def objective(feat, base, mask, active, weights):
        stats = policy.effect_statistics(
            feat, base, mask, active, (5, 5, 5), weights
        )
        return stats["js_mean"] + stats["residual_rms"] + stats["base_rms"]

    compiled = torch.compile(objective, backend="eager", fullgraph=True)
    feat = torch.randn(16, 12)
    base = torch.randn(16, 15)
    mask = torch.ones(16, 3, 5, dtype=torch.bool)
    active = torch.ones(16, 3, dtype=torch.bool)
    weights = torch.ones(16)
    value = compiled(feat, base, mask, active, weights)
    assert torch.isfinite(value)


def test_residual_logits_are_hard_bounded():
    policy = make_policy()
    feat = torch.randn(32, 12) * 1000
    tactic = torch.arange(32) % 4
    residual = policy.residual_logits(feat, tactic)
    assert torch.isfinite(residual).all()
    assert float(residual.detach().abs().max()) <= 4.0 + 1e-6
