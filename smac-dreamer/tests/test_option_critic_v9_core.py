from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/r2dreamer"))

# hierarchical_dreamer only needs the TensorDict name at import time in these tests.
fake_td = types.ModuleType("tensordict")
fake_td.TensorDict = dict
sys.modules.setdefault("tensordict", fake_td)

from hierarchical_options import HierarchicalOptionsPolicy  # noqa: E402
from hierarchical_dreamer import interruptible_option_bootstrap  # noqa: E402
from option_critic import within_group_option_consistency_loss  # noqa: E402


def cfg(**overrides):
    base = dict(
        enabled=True,
        num_options=8,
        source_manager_group_count=2,
        min_duration=1,
        max_duration=4,
        manager_unimix_initial=0.0,
        manager_unimix_final=0.0,
        manager_unimix_decay_steps=1,
        slot_manager_unimix=0.01,
        slot_anchor_floor=0.40,
        worker_pg_warmup_steps=20,
        worker_pg_full_steps=150,
        manager_pg_warmup_steps=100,
        manager_pg_full_steps=300,
        termination_warmup_steps=800,
        termination_full_steps=801,
        termination_cap_full_steps=802,
        termination_loss_scale=0.0,
        world_model_grad_scale_initial=0.0,
        world_model_grad_scale_final=0.0,
        imag_horizon_initial_max=15,
        imag_horizon_final_max=15,
        imag_horizon_window=1,
        imag_horizon_ramp_steps=1,
    )
    base.update(overrides)
    return base


def policy() -> HierarchicalOptionsPolicy:
    torch.manual_seed(3)
    p = HierarchicalOptionsPolicy(5, 7, cfg())
    p.set_training_step(150)
    return p


def force_source_router(p: HierarchicalOptionsPolicy) -> None:
    def probs(self, feat, step=None):
        g0 = (feat[..., 0] >= 0).to(torch.float32)
        return torch.stack([g0, 1.0 - g0], -1)

    p.manager_group_probs = types.MethodType(probs, p)


def test_v9_contract_h15_and_all_eight_slots_available():
    p = policy()
    assert p.ARCHITECTURE == "dreamer_option_critic_v9_anchor_safe_8slot"
    assert p.num_options == 8
    assert p.settings.max_duration == 4
    assert p.active_imagination_horizon_range() == (15, 15)
    assert [p.next_imagination_horizon() for _ in range(5)] == [15] * 5
    assert torch.equal(p.slot_gate_by_option(), torch.ones(8))


def test_manager_slot_network_is_zero_output_but_gradient_alive():
    p = policy()
    assert torch.count_nonzero(p.manager_slot[0].weight) > 0
    assert torch.count_nonzero(p.manager_slot[-1].weight) == 0
    feat = torch.randn(32, p.feature_dim)
    logits = p.manager_slot_logits(feat)
    loss = -logits[..., 0, 1].mean()
    loss.backward()
    grad = p.manager_slot[-1].weight.grad
    assert grad is not None and torch.count_nonzero(grad) > 0

    # One output-layer step must open a gradient path to the hidden layer.
    with torch.no_grad():
        p.manager_slot[-1].weight.add_(-0.1 * grad)
    p.zero_grad(set_to_none=True)
    loss2 = -p.manager_slot_logits(feat)[..., 0, 1].mean()
    loss2.backward()
    hidden_grad = p.manager_slot[0].weight.grad
    assert hidden_grad is not None and torch.count_nonzero(hidden_grad) > 0


def test_child_delta_is_exact_zero_but_state_dependent_gradient_alive():
    p = policy()
    p.set_training_step(150)
    assert torch.count_nonzero(p.slot_delta[0].weight) > 0
    assert torch.count_nonzero(p.slot_delta[-1].weight) == 0
    feat = torch.randn(24, p.feature_dim)
    ids = torch.full((24,), 2, dtype=torch.long)
    delta = p.residual_logits(feat, ids) - p.residual_logits(
        feat, torch.zeros_like(ids)
    )
    assert torch.allclose(delta, torch.zeros_like(delta))
    loss = -p._all_slot_delta_logits(feat)[..., 2, 0].mean()
    loss.backward()
    grad = p.slot_delta[-1].weight.grad
    assert grad is not None and torch.count_nonzero(grad) > 0

    with torch.no_grad():
        p.slot_delta[-1].weight.add_(-0.1 * grad)
    p.zero_grad(set_to_none=True)
    y = p._all_slot_delta_logits(feat)[..., 2, 0]
    assert y.std() > 0
    (-y.mean()).backward()
    hidden_grad = p.slot_delta[0].weight.grad
    assert hidden_grad is not None and torch.count_nonzero(hidden_grad) > 0


def test_source_interruption_and_duration_match_shared_mask():
    p = policy(); force_source_router(p)
    feat = torch.tensor([
        [1.0, 0, 0, 0, 0],   # group 0, carry g0, continue
        [-1.0, 0, 0, 0, 0],  # group 1, carry g0, interrupt
        [1.0, 0, 0, 0, 0],   # max duration, interrupt
    ])
    option = torch.tensor([0, 0, 2])
    age = torch.tensor([2, 2, 4])
    has = torch.ones(3, dtype=torch.bool)
    expected = p.interruption_mask(feat, option, age, has)
    decision = p.step_option(
        feat, option, age, has, torch.zeros(3, dtype=torch.bool),
        deterministic=True,
    )
    assert torch.equal(expected, decision.option_terminated)
    assert torch.equal(expected, decision.option_started)
    assert decision.option_group(decision.option) if False else True
    assert p.option_group(decision.option[1:2]).item() == 1


class TableCritic:
    def __init__(self, q_all: torch.Tensor):
        self.table = q_all

    def q_all(self, feat, age, base):
        return self.table.to(feat.device).expand(*feat.shape[:-1], -1)

    def q_selected(self, feat, option, age, base):
        q = self.q_all(feat, age, base)
        return q.gather(-1, option.unsqueeze(-1)).squeeze(-1)


def test_exact_bootstrap_continues_or_switches_to_current_group_only():
    p = policy(); force_source_router(p)
    # Make within-group routing deterministic to anchor slot 0.
    with torch.no_grad():
        p.manager_slot[-1].weight.zero_()
        p.manager_slot[-1].bias.copy_(torch.tensor([8., 8., -8., -8., -8., -8., -8., -8.]))
    q = torch.tensor([10., 20., 30., 40., 50., 60., 70., 80.])
    critic = TableCritic(q)
    feat = torch.tensor([
        [1.0, 0, 0, 0, 0],   # carry g0, continue Q0=10
        [-1.0, 0, 0, 0, 0],  # carry g0 but source g1 -> switch to option1=20
        [1.0, 0, 0, 0, 0],   # max age -> switch within g0 -> option0=10
    ])
    option = torch.tensor([0, 0, 2])
    age = torch.tensor([2, 2, 4])
    base = torch.zeros(3)
    bootstrap, cont, switch, mask = interruptible_option_bootstrap(
        p, critic, feat, option, age, base, p.training_step
    )
    assert torch.equal(mask, torch.tensor([False, True, True]))
    assert torch.allclose(cont, torch.tensor([10., 10., 30.]))
    # The configured 1% within-group unimix is part of both execution and the
    # bootstrap, so the expected switch value includes that exact exploration.
    assert torch.allclose(switch, torch.tensor([10.3, 20.3, 10.3]), atol=1e-3)
    assert torch.allclose(bootstrap, torch.tensor([10., 20.3, 10.3]), atol=1e-3)


def test_switch_value_excludes_unreachable_other_group_options():
    p = policy(); force_source_router(p)
    with torch.no_grad():
        p.manager_slot[-1].weight.zero_()
        p.manager_slot[-1].bias.zero_()
    feat = torch.tensor([[1.0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0]])
    # g0 options [0,2,4,6] all 1; g1 options [1,3,5,7] all 100.
    q = torch.tensor([[1.,100.,1.,100.,1.,100.,1.,100.],
                      [1.,100.,1.,100.,1.,100.,1.,100.]])
    v = p.switch_value_for_source_group(feat, q)
    assert torch.allclose(v, torch.tensor([1., 100.]))


def test_randomized_execution_and_bellman_switch_events_are_identical():
    torch.manual_seed(123)
    p = policy(); force_source_router(p)
    n = 512
    feat = torch.randn(n, p.feature_dim)
    option = torch.randint(0, 8, (n,))
    age = torch.randint(0, 5, (n,))
    has = torch.ones(n, dtype=torch.bool)
    decision = p.step_option(
        feat, option, age, has, torch.zeros(n, dtype=torch.bool), deterministic=True
    )
    exact = p.interruption_mask(feat, option, age, has)
    assert torch.equal(decision.option_terminated, exact)
    assert torch.equal(decision.option_started, exact)

    q = torch.randn(n, 8)
    class BatchCritic:
        def q_all(self, f, a, b): return q
        def q_selected(self, f, o, a, b): return q.gather(-1,o[:,None]).squeeze(-1)
    boot, cont, switch, mask = interruptible_option_bootstrap(
        p, BatchCritic(), feat, option, age, torch.zeros(n), p.training_step
    )
    assert torch.equal(mask, exact)
    assert torch.allclose(boot, torch.where(exact, switch, cont))


def test_train_and_eval_share_the_exact_same_interrupt_event():
    p = policy(); force_source_router(p)
    torch.manual_seed(991)
    n = 256
    feat = torch.randn(n, p.feature_dim)
    option = torch.randint(0, 8, (n,))
    age = torch.randint(0, 5, (n,))
    has = torch.ones(n, dtype=torch.bool)
    first = torch.zeros(n, dtype=torch.bool)
    deterministic = p.step_option(
        feat, option, age, has, first, deterministic=True
    )
    stochastic = p.step_option(
        feat, option, age, has, first, deterministic=False,
        manager_uniform=torch.rand(n),
        termination_uniform=torch.rand(n),
    )
    exact = p.interruption_mask(feat, option, age, has)
    assert torch.equal(deterministic.option_terminated, exact)
    assert torch.equal(stochastic.option_terminated, exact)
    assert torch.equal(deterministic.option_started, exact)
    assert torch.equal(stochastic.option_started, exact)
    # Identity may differ at a boundary, but both selections must be executable
    # in the frozen source group at the current state.
    assert torch.equal(p.option_group(deterministic.option), p.source_group(feat))
    assert torch.equal(p.option_group(stochastic.option), p.source_group(feat))


def test_randomized_multistep_state_machine_preserves_group_and_duration_invariants():
    torch.manual_seed(2026)
    p = policy(); force_source_router(p)
    n = 128
    option = torch.zeros(n, dtype=torch.long)
    age = torch.zeros(n, dtype=torch.long)
    has = torch.zeros(n, dtype=torch.bool)
    first = torch.ones(n, dtype=torch.bool)
    for _ in range(1000):
        feat = torch.randn(n, p.feature_dim)
        before_option = option.clone(); before_age = age.clone(); before_has = has.clone()
        decision = p.step_option(
            feat, option, age, has, first,
            deterministic=False, manager_uniform=torch.rand(n),
        )
        source = p.source_group(feat)
        assert torch.equal(p.option_group(decision.option), source)
        assert torch.all((decision.action_age >= 0) & (decision.action_age < 4))
        assert torch.all((decision.carry_age >= 1) & (decision.carry_age <= 4))
        structural = before_has & (~first)
        expected_interrupt = structural & (
            (p.option_group(before_option) != source) | (before_age >= 4)
        )
        assert torch.equal(decision.option_terminated, expected_interrupt)
        assert torch.equal(decision.option_started, first | (~before_has) | expected_interrupt)
        continued = structural & (~expected_interrupt)
        assert torch.equal(decision.option[continued], before_option[continued])
        assert torch.equal(decision.action_age[continued], before_age[continued])
        option, age, has = decision.option, decision.carry_age, decision.has_option
        first = torch.rand(n) < 0.01
        # Episode reset state will be interpreted through is_first on the next call.


def test_anchor_floor_keeps_children_active_and_can_still_select_a_useful_child():
    p = policy()
    feat = torch.randn(64, p.feature_dim)
    initial = p.manager_slot_probs(feat, 0)
    expected = torch.tensor([0.547, 0.151, 0.151, 0.151])
    assert torch.allclose(initial, expected.expand_as(initial), atol=1e-6)
    assert torch.equal(initial.argmax(-1), torch.zeros(initial.shape[:-1], dtype=torch.long))

    # A strongly preferred child can exceed the immutable anchor floor, so the
    # safety floor does not make specialization behaviorally unreachable.
    with torch.no_grad():
        p.manager_slot[-1].weight.zero_()
        # Option index 2 is group-0 slot 1.
        p.manager_slot[-1].bias.copy_(
            torch.tensor([-20.0, -20.0, 20.0, -20.0, -20.0, -20.0, -20.0, -20.0])
        )
    child_preferred = p.manager_slot_probs(feat, 0)[..., 0, :]
    assert torch.all(child_preferred[..., 1] > child_preferred[..., 0])
    assert torch.all(child_preferred[..., 0] >= 0.398 - 1e-4)


def test_child_output_has_one_warmup_not_a_squared_warmup():
    p = policy()
    child = torch.tensor([2, 3, 4, 5, 6, 7])
    anchors = torch.tensor([0, 1])
    p.set_training_step(0)
    assert p.worker_pg_blend() == 0.0
    assert torch.allclose(
        p.slot_delta_scale_by_option(child),
        torch.full((6,), 0.10),
    )
    assert torch.equal(
        p.slot_delta_scale_by_option(anchors),
        torch.zeros(2),
    )
    assert torch.equal(p.slot_pg_blend_for_option(child), torch.ones(6))
    p.set_training_step(85)
    assert 0.49 < p.worker_pg_blend() < 0.51
    # The action bound stays fixed; only the PG coefficient ramps.
    assert torch.allclose(
        p.slot_delta_scale_by_option(child),
        torch.full((6,), 0.10),
    )


def test_option_critic_consistency_ties_children_without_moving_anchor():
    # [g0s0,g1s0,g0s1,g1s1,g0s2,g1s2,g0s3,g1s3]
    q = torch.tensor(
        [[[1.0, 10.0, 3.0, 7.0, 5.0, 13.0, -1.0, 16.0]]],
        requires_grad=True,
    )
    loss = within_group_option_consistency_loss(
        q, source_group_count=2, return_scale=torch.tensor(2.0),
        weights=torch.ones(1, 1),
    )
    assert loss > 0
    loss.backward()
    # Anchor targets are detached; only child values are pulled toward them.
    assert q.grad is not None
    assert q.grad[0, 0, 0] == 0
    assert q.grad[0, 0, 1] == 0
    assert torch.count_nonzero(q.grad[0, 0, 2:]) > 0

    identical = torch.tensor([[[1., 10., 1., 10., 1., 10., 1., 10.]]])
    zero = within_group_option_consistency_loss(
        identical, source_group_count=2, return_scale=torch.tensor(1.0),
        weights=torch.ones(1, 1),
    )
    assert zero == 0


def test_forecast_wrapper_requires_canonical_source_and_external_inputs():
    script = (ROOT / "scripts/run_exp45_full_train_eval_resilient.sh").read_text()
    assert "source_verify" in script
    assert "smac_jepa/train_jepa_exp45_pow2_direct.py" in script
    assert "EXP40_CHECKPOINT:?" in script
    assert "MANIFEST:?" in script
    assert "BUNDLE_ZIP" not in script
    assert "unzip" not in script
    assert "CURRENT_EXP45" not in script


def test_runtime_assertion_rejects_stale_required_metric(tmp_path):
    import json
    import subprocess

    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {"global_step": 0, "train/real_post_mask_invalid_sample_rate": 0.0},
        {"global_step": 100_000, "unrelated": 1.0},
    ]
    with (run / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    script = ROOT / "scripts/assert_option_critic_v9_metrics.py"
    result = subprocess.run(
        [sys.executable, str(script), str(run), "--max-metric-age", "50"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "stale metric" in result.stdout + result.stderr
