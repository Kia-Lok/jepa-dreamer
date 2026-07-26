from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

POLICY_PATH = Path(__file__).parents[1] / "external/r2dreamer/tactical_policy.py"
spec = importlib.util.spec_from_file_location("tactical_policy_v1_2", POLICY_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
TacticalMixturePolicy = module.TacticalMixturePolicy


def make_policy(**overrides):
    cfg = dict(
        enabled=True,
        num_tactics=2,
        embedding_dim=8,
        hidden_dim=16,
        duration=1,
        residual_scale=0.25,
        max_abs_residual_logit=2.0,
        eval_confidence_threshold=0.70,
        min_selector_mi_normalized=0.05,
        base_kl_target=0.02,
    )
    cfg.update(overrides)
    return TacticalMixturePolicy(12, 18, cfg)


def test_residual_is_zero_mean_across_tactics():
    policy = make_policy()
    feat = torch.randn(17, 12)
    residual = policy.all_residual_logits(feat)
    assert residual.shape == (17, 2, 18)
    torch.testing.assert_close(
        residual.mean(dim=-2),
        torch.zeros(17, 18),
        atol=1e-7,
        rtol=0,
    )


def test_deterministic_eval_is_exact_base_below_confidence_threshold():
    policy = make_policy()
    feat = torch.randn(11, 12)
    base = torch.randn(11, 18)
    final, _, confidence, gate = policy.eval_combined_logits(base, feat)
    assert float(confidence.max()) < 0.70
    assert torch.count_nonzero(gate) == 0
    torch.testing.assert_close(final, base, atol=0, rtol=0)


def test_common_mode_cannot_survive_centering():
    policy = make_policy()
    feat = torch.randn(7, 12)
    raw = policy._uncentered_all_residual_logits(feat)
    common = torch.randn(7, 1, 18)
    expected = policy.settings.residual_scale * (
        raw - raw.mean(dim=-2, keepdim=True)
    )
    shifted = policy.settings.residual_scale * (
        (raw + common) - (raw + common).mean(dim=-2, keepdim=True)
    )
    torch.testing.assert_close(expected, shifted, atol=1e-7, rtol=1e-6)


def test_mi_shortfall_penalizes_state_invariant_selector():
    policy = make_policy()
    invariant = torch.zeros(64, 2)
    stats_invariant = policy.usage_statistics(invariant)
    assert float(stats_invariant["mi_shortfall"]) > 0

    specialized = torch.cat(
        [
            torch.tensor([[8.0, -8.0]]).expand(32, -1),
            torch.tensor([[-8.0, 8.0]]).expand(32, -1),
        ],
        dim=0,
    )
    stats_specialized = policy.usage_statistics(specialized)
    assert float(stats_specialized["mi_shortfall"]) == 0
    assert float(stats_specialized["collapse_loss"]) < float(
        stats_invariant["collapse_loss"]
    )


def test_effect_statistics_expose_trust_region_metrics():
    policy = make_policy()
    feat = torch.randn(13, 12)
    base = torch.randn(13, 18)
    mask = torch.ones(13, 3, 6, dtype=torch.bool)
    active = torch.ones(13, 3, dtype=torch.bool)
    stats = policy.effect_statistics(
        feat,
        base,
        mask,
        active,
        [6, 6, 6],
    )
    for key in (
        "base_kl_mean",
        "base_kl_max",
        "base_kl_loss",
        "action_flip_rate",
        "js_mean",
    ):
        assert key in stats
        assert torch.isfinite(stats[key])
    assert 0 <= float(stats["action_flip_rate"]) <= 1


def test_effect_loss_has_residual_gradient_at_initialization():
    policy = make_policy()
    feat = torch.randn(31, 12)
    base = torch.randn(31, 18)
    mask = torch.ones(31, 3, 6, dtype=torch.bool)
    active = torch.ones(31, 3, dtype=torch.bool)
    stats = policy.effect_statistics(feat, base, mask, active, [6, 6, 6])
    loss = torch.relu(torch.tensor(0.002) - stats["js_mean"])
    loss = loss + 0.1 * stats["base_kl_loss"]
    loss.backward()
    total = sum(
        parameter.grad.abs().sum()
        for parameter in policy.residual.parameters()
        if parameter.grad is not None
    )
    assert float(total) > 0


def test_metadata_identifies_v1_2():
    policy = make_policy()
    metadata = policy.metadata()
    assert metadata["architecture"] == "tactical_mixture_v1_2"
    assert metadata["num_tactics"] == 2
    assert metadata["residual_scale"] == 0.25
    assert metadata["base_kl_target"] == 0.02
