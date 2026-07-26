import torch

from tactical_policy import TacticalMixturePolicy


CFG = {
    "enabled": True,
    "num_tactics": 4,
    "embedding_dim": 8,
    "hidden_dim": 16,
    "max_effect_states": 64,
    "duration": 1,
}


def make_policy():
    torch.manual_seed(7)
    return TacticalMixturePolicy(12, 15, CFG)


def test_selector_is_uniform_at_initialization():
    policy = make_policy()
    feat = torch.randn(11, 12)
    probs = policy.selector_dist(feat).probs
    assert torch.allclose(probs, torch.full_like(probs, 0.25), atol=1e-7)


def test_zero_residual_exactly_preserves_base_logits_for_every_tactic():
    policy = make_policy()
    feat = torch.randn(5, 12)
    base = torch.randn(5, 15)
    for tactic_id in range(4):
        tactic = torch.full((5,), tactic_id)
        assert torch.equal(policy.combine_logits(base, feat, tactic), base)


def test_gradient_reaches_selector_residual_and_embedding():
    policy = make_policy()
    feat = torch.randn(32, 12)
    dist = policy.selector_dist(feat)
    tactic = dist.sample()
    base = torch.randn(32, 15)
    advantage = torch.linspace(-1, 1, 32)
    loss = -(dist.log_prob(tactic) * advantage).mean()
    loss = loss + policy.combine_logits(base, feat, tactic).square().mean()
    loss.backward()
    assert policy.selector[-1].weight.grad is not None
    assert policy.residual[-1].weight.grad is not None

    # The zero final residual layer blocks first-step hidden/embedding gradients.
    # Once that output layer has moved, the embedding must receive gradients.
    with torch.no_grad():
        policy.residual[-1].weight.add_(0.01 * policy.residual[-1].weight.grad)
    policy.zero_grad(set_to_none=True)
    policy.combine_logits(base, feat, tactic).square().mean().backward()
    assert policy.embedding.weight.grad is not None
    assert torch.isfinite(policy.embedding.weight.grad).all()


def test_balance_loss_uniform_and_collapsed():
    policy = make_policy()
    loss_uniform, marginal = policy.balance_loss(torch.zeros(20, 4))
    assert float(loss_uniform) < 1e-7
    assert torch.allclose(marginal, torch.full_like(marginal, 0.25), atol=1e-7)

    collapsed = torch.full((20, 4), -20.0)
    collapsed[:, 0] = 20.0
    loss_collapsed, _ = policy.balance_loss(collapsed)
    assert float(loss_collapsed) > 1.0


def test_effect_js_respects_mask_and_inactive_agents():
    policy = make_policy()
    feat = torch.randn(6, 12)
    base = torch.randn(6, 15)
    mask = torch.zeros(6, 3, 5, dtype=torch.bool)
    mask[:, 0, 0] = True
    mask[:, 1, :2] = True
    mask[:, 2, 0] = True
    active = torch.tensor([[1, 1, 0]], dtype=torch.bool).expand(6, -1)
    assert float(policy.effect_js(feat, base, mask, active, (5, 5, 5)).detach()) == 0.0

    with torch.no_grad():
        policy.residual[-1].weight.normal_(0, 0.05)
    js = policy.effect_js(feat, base, mask, active, (5, 5, 5))
    assert torch.isfinite(js)
    assert float(js.detach()) >= 0.0


def test_metadata_and_duration_guard():
    policy = make_policy()
    meta = policy.metadata()
    assert meta["architecture"] == "tactical_mixture_v1"
    assert meta["num_tactics"] == 4
    bad = dict(CFG)
    bad["duration"] = 3
    try:
        TacticalMixturePolicy(12, 15, bad)
    except ValueError as exc:
        assert "duration=1" in str(exc)
    else:
        raise AssertionError("duration > 1 must be rejected in v1")
