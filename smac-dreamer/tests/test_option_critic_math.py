from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/r2dreamer"))

from option_critic import (  # noqa: E402
    OptionCritic,
    call_and_return_bootstrap,
    manager_policy_loss,
    manager_value,
    normalized_advantage,
    option_critic_loss,
    option_lambda_return,
    termination_gradient_sign_check,
    termination_loss,
)
from test_hierarchical_options import cfg  # noqa: E402


def test_option_critic_zero_residual_matches_base_value():
    critic = OptionCritic(10, cfg())
    feat = torch.randn(6, 10)
    age = torch.arange(6)
    base = torch.randn(6)
    q = critic.q_all(feat, age, base)
    assert q.shape == (6, 8)
    assert torch.allclose(q, base[:, None].expand_as(q), atol=1e-6)


def test_manager_value_is_probability_weighted_q():
    probs = torch.tensor([[0.25, 0.75]])
    q = torch.tensor([[2.0, 6.0]])
    assert torch.allclose(manager_value(probs, q), torch.tensor([5.0]))


def test_manager_gradient_only_occurs_at_boundaries():
    log_prob = torch.tensor([1.0, 2.0], requires_grad=True)
    entropy = torch.zeros(2)
    advantage = torch.ones(2)
    boundary = torch.tensor([1.0, 0.0])
    loss = manager_policy_loss(
        log_prob,
        entropy,
        advantage,
        boundary,
        torch.ones(2),
        pg_scale=1.0,
        entropy_scale=0.0,
    )
    loss.backward()
    assert log_prob.grad[0] != 0
    assert log_prob.grad[1] == 0


def test_termination_gradient_sign_is_correct():
    signs = termination_gradient_sign_check()
    # Gradient descent with positive gradient lowers beta when continuation is better.
    assert signs["continue_better_grad"] > 0
    # Negative gradient raises beta when switching is better.
    assert signs["switch_better_grad"] < 0


def test_termination_loss_excludes_ineligible_decisions():
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    beta = logits.sigmoid()
    loss, _ = termination_loss(
        beta,
        torch.tensor([2.0, 2.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.ones(2),
        torch.tensor(1.0),
        normalized_margin=0.0,
    )
    loss.backward()
    assert logits.grad[0] != 0
    assert logits.grad[1] == 0


def test_option_critic_loss_is_finite_and_scaled():
    q = torch.tensor([0.0, 2.0], requires_grad=True)
    target = torch.tensor([1.0, 1.0])
    loss = option_critic_loss(q, target, torch.tensor(2.0), torch.ones(2))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(q.grad).all()


def test_termination_loss_uses_probability_chain_rule_without_extra_blend():
    # The loss itself must not apply a second blend factor. The probability
    # passed in is already the exact executed beta and owns the full chain rule.
    beta_logit = torch.tensor(0.2, requires_grad=True)
    beta = beta_logit.sigmoid().expand(4)
    loss, _ = termination_loss(
        beta,
        torch.ones(4),
        torch.zeros(4),
        torch.ones(4, dtype=torch.bool),
        torch.ones(4),
        torch.tensor(1.0),
        normalized_margin=0.02,
        advantage_clip=2.0,
    )
    loss.backward()
    expected = beta_logit.sigmoid() * (1.0 - beta_logit.sigmoid()) * 1.02
    assert torch.allclose(beta_logit.grad, expected, atol=1e-7)




def test_call_and_return_bootstrap_matches_boundary_cases_and_mixture():
    continue_value = torch.tensor([8.0, 8.0, 8.0])
    switch_value = torch.tensor([2.0, 2.0, 2.0])
    beta = torch.tensor([0.0, 1.0, 0.25])
    out = call_and_return_bootstrap(continue_value, switch_value, beta)
    assert torch.allclose(out, torch.tensor([8.0, 2.0, 6.5]))


def test_option_lambda_return_matches_hand_calculation():
    reward = torch.tensor([[[1.0], [2.0], [3.0]]])
    cont = torch.ones_like(reward)
    boot = torch.tensor([[[10.0], [20.0], [30.0]]])
    out = option_lambda_return(
        reward, cont, boot, discount=0.5, lambda_=0.0
    )
    expected = reward + 0.5 * boot
    assert torch.allclose(out, expected)

    out_full = option_lambda_return(
        reward, cont, boot, discount=0.5, lambda_=1.0
    )
    # G2 = 3 + .5*30 = 18; G1 = 2 + .5*18 = 11; G0 = 1 + .5*11 = 6.5
    assert torch.allclose(
        out_full, torch.tensor([[[6.5], [11.0], [18.0]]])
    )



def test_normalized_advantage_is_invariant_to_common_return_scaling():
    target = torch.tensor([2.0, 6.0])
    baseline = torch.tensor([1.0, 2.0])
    a = normalized_advantage(target, baseline, torch.tensor(2.0))
    b = normalized_advantage(10.0 * target, 10.0 * baseline, torch.tensor(20.0))
    assert torch.allclose(a, b)
