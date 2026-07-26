from __future__ import annotations

import copy
import pathlib
import sys
import types

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/r2dreamer"))

fake_td = types.ModuleType("tensordict")
fake_td.TensorDict = dict
sys.modules.setdefault("tensordict", fake_td)

from hierarchical_dreamer import load_hierarchical_compatible_state  # noqa: E402
from hierarchical_options import HierarchicalOptionsPolicy  # noqa: E402
from option_critic import OptionCritic  # noqa: E402
from test_hierarchical_options import cfg  # noqa: E402


class DummyAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        settings = cfg(hidden_dim=32, num_options=8, source_manager_group_count=2)
        self.hierarchical_enabled = True
        self.base = nn.Linear(5, 7)
        self.hierarchical_options = HierarchicalOptionsPolicy(12, 15, settings)
        self.option_critic = OptionCritic(12, settings)
        self._slow_option_critic = copy.deepcopy(self.option_critic)
        self._frozen_hierarchical_options = copy.deepcopy(self.hierarchical_options)
        self._source_hierarchical_options = copy.deepcopy(self.hierarchical_options)
        for parameter in self._source_hierarchical_options.parameters():
            parameter.requires_grad_(False)

    def hierarchical_metadata(self):
        metadata = self.hierarchical_options.metadata()
        metadata["enabled"] = True
        return metadata

    def clone_and_freeze(self):
        self._frozen_hierarchical_options = copy.deepcopy(self.hierarchical_options)
        for parameter in self._frozen_hierarchical_options.parameters():
            parameter.requires_grad_(False)


def source_tactical_state(agent: DummyAgent) -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    target = agent.hierarchical_options
    state = {
        "base.weight": torch.randn_like(agent.base.weight),
        "base.bias": torch.randn_like(agent.base.bias),
        "tactical_policy.selector.0.weight": torch.randn_like(
            target.manager_group[0].weight
        ),
        "tactical_policy.selector.0.bias": torch.randn_like(
            target.manager_group[0].bias
        ),
        "tactical_policy.selector.2.weight": torch.randn_like(
            target.manager_group[2].weight
        ),
        "tactical_policy.selector.2.bias": torch.randn_like(
            target.manager_group[2].bias
        ),
        "tactical_policy.embedding.weight": torch.randn_like(
            target.option_embedding.weight
        ),
        "tactical_policy.residual.0.weight": torch.randn_like(
            target.worker_residual[0].weight
        ),
        "tactical_policy.residual.0.bias": torch.randn_like(
            target.worker_residual[0].bias
        ),
        "tactical_policy.residual.2.weight": torch.randn_like(
            target.worker_residual[2].weight
        ),
        "tactical_policy.residual.2.bias": torch.randn_like(
            target.worker_residual[2].bias
        ),
    }
    for key, value in list(state.items()):
        if key.startswith("tactical_policy."):
            state["_frozen_" + key] = value.clone()
    return state


def migrate() -> tuple[DummyAgent, dict[str, torch.Tensor], dict]:
    agent = DummyAgent()
    source = source_tactical_state(agent)
    result = load_hierarchical_compatible_state(
        agent,
        source,
        tactical_metadata={
            "architecture": "tactical_mixture_v1_2",
            "num_tactics": 2,
        },
    )
    return agent, source, result


def source_selector_probs(
    source: dict[str, torch.Tensor], feat: torch.Tensor
) -> torch.Tensor:
    hidden = torch.nn.functional.elu(torch.nn.functional.linear(
        feat,
        source["tactical_policy.selector.0.weight"],
        source["tactical_policy.selector.0.bias"],
    ))
    logits = torch.nn.functional.linear(
        hidden,
        source["tactical_policy.selector.2.weight"],
        source["tactical_policy.selector.2.bias"],
    )
    return logits.softmax(-1)


def test_v1_2_migration_has_eight_capacity_but_only_two_active_anchors():
    agent, source, result = migrate()
    target = agent.hierarchical_options
    assert result["migrated"] is True
    assert result["target_options"] == 8
    assert result["migration_layout"] == (
        "two_source_anchors_plus_six_progressive_child_slots"
    )
    assert result["trajectory_preservation"] == (
        "exact_source_group_and_worker_at_step_zero_then_progressive_unlock"
    )

    assert torch.equal(
        target.manager_group[0].weight,
        source["tactical_policy.selector.0.weight"],
    )
    assert torch.equal(
        target.manager_group[2].weight,
        source["tactical_policy.selector.2.weight"],
    )
    assert torch.count_nonzero(target.manager_slot[-1].weight) == 0
    assert torch.count_nonzero(target.manager_slot[-1].bias) == 0

    feat = torch.randn(17, target.feature_dim)
    source_probs = source_selector_probs(source, feat)
    full = target.manager_probs(feat, step=0)
    assert torch.allclose(full[:, :2], source_probs, atol=2e-7, rtol=1e-6)
    assert torch.count_nonzero(full[:, 2:]) == 0
    assert torch.allclose(
        target.grouped_manager_probs(full), source_probs, atol=2e-7, rtol=1e-6
    )

    gates = target.slot_gate_by_option(0)
    assert torch.equal(gates[:2], torch.ones(2))
    assert torch.count_nonzero(gates[2:]) == 0


def test_migrated_worker_is_exact_source_policy_for_every_slot_at_step_zero():
    agent, source, _ = migrate()
    target = agent.hierarchical_options
    feat = torch.randn(19, target.feature_dim)
    residual = target.all_residual_logits(feat, step=0)

    source_embedding = source["tactical_policy.embedding.weight"]
    feat_all = feat.unsqueeze(-2).expand(feat.shape[0], 2, target.feature_dim)
    emb = source_embedding.view(1, 2, -1).expand(feat.shape[0], 2, -1)
    source_raw = target.worker_residual(torch.cat([feat_all.float(), emb], dim=-1))
    cap = target.settings.max_abs_residual_logit
    source_raw = cap * torch.tanh(source_raw / cap)
    source_centered = source_raw - source_raw.mean(dim=1, keepdim=True)
    expected = target.worker_scale(0) * source_centered
    for option in range(8):
        assert torch.allclose(
            residual[:, option], expected[:, option % 2], atol=2e-6, rtol=1e-6
        )
    assert torch.count_nonzero(target.slot_delta[-1].weight) == 0
    assert torch.count_nonzero(target.slot_delta[-1].bias) == 0


def test_unlocking_equal_slots_cannot_change_source_group_routing():
    agent, source, _ = migrate()
    target = agent.hierarchical_options
    feat = torch.randn(23, target.feature_dim)
    source_probs = source_selector_probs(source, feat)
    for step in (0, 50, 100, 200, 400, 800):
        full = target.manager_probs(feat, step=step)
        grouped = target.grouped_manager_probs(full)
        expected_group = target.manager_group_probs(feat, step=step)
        assert torch.allclose(grouped, expected_group, atol=2e-6, rtol=1e-6)
        if step == 0:
            assert torch.allclose(grouped, source_probs, atol=2e-6, rtol=1e-6)
        assert torch.allclose(full.sum(-1), torch.ones(feat.shape[0]), atol=1e-7)


def test_progressive_slot_unlock_and_specialization_are_continuous():
    target = DummyAgent().hierarchical_options
    # Test config: unlock child slot 1 at 50 and ramp for 50.
    assert torch.equal(target.slot_gate_by_option(49)[2:4], torch.zeros(2))
    assert torch.allclose(
        target.slot_gate_by_option(75)[2:4], torch.full((2,), 0.5)
    )
    assert torch.equal(target.slot_gate_by_option(100)[2:4], torch.ones(2))
    # Anchors never acquire child deltas. Children begin with exactly zero scale.
    ids = torch.arange(8)
    at_unlock = target.slot_delta_scale_by_option(ids, 50)
    assert torch.equal(at_unlock, torch.zeros_like(at_unlock))
    midway = target.slot_delta_scale_by_option(ids, 75)
    assert midway[0] == 0 and midway[1] == 0
    assert torch.allclose(midway[2:4], torch.full((2,), 0.05))


def test_v1_2_migration_reselects_every_state_at_step_zero():
    agent, _, _ = migrate()
    target = agent.hierarchical_options
    feat = torch.randn(32, target.feature_dim)
    option = torch.randint(0, 2, (32,))
    age = torch.ones(32, dtype=torch.long)
    beta, eligible, forced_continue, forced_terminate = (
        target.effective_termination_probability(feat, option, age, step=0)
    )
    assert eligible.all()
    assert not forced_continue.any()
    assert not forced_terminate.any()
    assert torch.equal(beta, torch.ones_like(beta))


def test_option_critic_is_neutral_for_all_eight_slots_after_migration():
    agent, _, _ = migrate()
    assert torch.count_nonzero(agent.option_critic.trunk[-1].weight) == 0
    assert torch.count_nonzero(agent.option_critic.trunk[-1].bias) == 0
    feat = torch.randn(11, 12)
    age = torch.zeros(11, dtype=torch.long)
    base = torch.randn(11)
    q = agent.option_critic.q_all(feat, age, base)
    assert q.shape == (11, 8)
    assert torch.allclose(q, base[:, None].expand_as(q))


def test_strict_hierarchical_resume_requires_matching_v6_metadata():
    agent, _, _ = migrate()
    state = agent.state_dict()
    result = load_hierarchical_compatible_state(
        agent,
        state,
        checkpoint_metadata=agent.hierarchical_metadata(),
    )
    assert result == {"migrated": False, "strict": True}

    bad = dict(agent.hierarchical_metadata())
    bad["slot_unlock_ramp_steps"] += 1
    try:
        load_hierarchical_compatible_state(
            agent,
            state,
            checkpoint_metadata=bad,
        )
    except RuntimeError as exc:
        assert "slot_unlock_ramp_steps" in str(exc)
    else:
        raise AssertionError("metadata mismatch should fail closed")
