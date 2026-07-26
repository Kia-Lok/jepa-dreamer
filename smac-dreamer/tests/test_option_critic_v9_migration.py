from __future__ import annotations

import copy
import pathlib
import sys
import types
from types import SimpleNamespace

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/r2dreamer"))
fake_td = types.ModuleType("tensordict")
fake_td.TensorDict = dict
sys.modules.setdefault("tensordict", fake_td)

from hierarchical_dreamer import (  # noqa: E402
    build_hierarchical_modules,
    clone_and_freeze_hierarchy,
    load_hierarchical_compatible_state,
)


def settings(**overrides):
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
    return SimpleNamespace(**base)


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_adapter = nn.Linear(3, 3)
        self.core = nn.Linear(3, 3)


class DummyAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.world_model_backend = "jepa"
        self.action_masking = True
        self.act_discrete = True
        self.feat_size = 12
        self._actor_shape = (5, 4, 6)
        self.actor = nn.Sequential(nn.Linear(12, 16), nn.ELU(), nn.Linear(16, 15))
        self.jepa_world_model = WorldModel()
        self.base = nn.Linear(5, 7)
        cfg = SimpleNamespace(hierarchical_options=settings(), compile=False)
        build_hierarchical_modules(self, cfg)

    def hierarchical_metadata(self):
        m = self.hierarchical_options.metadata(); m["enabled"] = True; return m

    def clone_and_freeze(self):
        clone_and_freeze_hierarchy(self)


def source_state(agent: DummyAgent):
    torch.manual_seed(11)
    h = agent.hierarchical_options
    excluded = (
        "hierarchical_options.", "_frozen_hierarchical_options.",
        "_source_hierarchical_options.", "option_critic.", "_slow_option_critic.",
    )
    state = {
        k: v.clone() for k, v in agent.state_dict().items()
        if not k.startswith(excluded)
    }
    state.update({
        "base.weight": torch.randn_like(agent.base.weight),
        "base.bias": torch.randn_like(agent.base.bias),
        "tactical_policy.selector.0.weight": torch.randn_like(h.manager_group[0].weight),
        "tactical_policy.selector.0.bias": torch.randn_like(h.manager_group[0].bias),
        "tactical_policy.selector.2.weight": torch.randn_like(h.manager_group[2].weight),
        "tactical_policy.selector.2.bias": torch.randn_like(h.manager_group[2].bias),
        "tactical_policy.embedding.weight": torch.randn_like(h.option_embedding.weight),
        "tactical_policy.residual.0.weight": torch.randn_like(h.worker_residual[0].weight),
        "tactical_policy.residual.0.bias": torch.randn_like(h.worker_residual[0].bias),
        "tactical_policy.residual.2.weight": torch.randn_like(h.worker_residual[2].weight),
        "tactical_policy.residual.2.bias": torch.randn_like(h.worker_residual[2].bias),
    })
    for k, v in list(state.items()):
        if k.startswith("tactical_policy."):
            state["_frozen_" + k] = v.clone()
    return state


def migrate():
    a = DummyAgent(); src = source_state(a)
    result = load_hierarchical_compatible_state(
        a, src,
        tactical_metadata={"architecture": "tactical_mixture_v1_2", "num_tactics": 2},
    )
    return a, src, result


def source_probs(src, feat):
    h = torch.nn.functional.elu(torch.nn.functional.linear(
        feat, src["tactical_policy.selector.0.weight"], src["tactical_policy.selector.0.bias"]
    ))
    return torch.nn.functional.linear(
        h, src["tactical_policy.selector.2.weight"], src["tactical_policy.selector.2.bias"]
    ).softmax(-1)


def test_migration_exact_source_groups_all_eight_slots_and_live_hidden_layers():
    a, src, result = migrate(); h = a.hierarchical_options
    assert result["migration_layout"] == "two_frozen_source_anchors_plus_six_anchor_floor_interruptible_children"
    assert result["trajectory_preservation"] == "exact_interruptible_smdp_with_group_restricted_slot_manager"
    feat = torch.randn(31, h.feature_dim)
    grouped = h.grouped_manager_probs(h.manager_probs(feat, 0))
    assert torch.allclose(grouped, source_probs(src, feat), atol=2e-6, rtol=1e-6)
    assert torch.equal(h.slot_gate_by_option(0), torch.ones(8))
    assert torch.count_nonzero(h.manager_slot[0].weight) > 0
    assert torch.count_nonzero(h.manager_slot[-1].weight) == 0
    # Per group: the immutable source anchor receives a fixed safety floor while
    # every child remains active and behavior-equivalent at migration.
    slot = h.manager_slot_probs(feat, 0)
    expected = torch.tensor([0.547, 0.151, 0.151, 0.151], dtype=slot.dtype)
    assert torch.allclose(slot, expected.expand_as(slot), atol=1e-6)
    deterministic = slot.argmax(dim=-1)
    assert torch.equal(deterministic, torch.zeros_like(deterministic))


def test_migration_preserves_anchor_actions_and_all_children_start_exact():
    a, _, _ = migrate(); h = a.hierarchical_options
    feat = torch.randn(23, h.feature_dim)
    residual = h.all_residual_logits(feat, 0)
    for child in (2, 4, 6):
        assert torch.allclose(residual[:, child], residual[:, 0], atol=2e-6, rtol=1e-6)
    for child in (3, 5, 7):
        assert torch.allclose(residual[:, child], residual[:, 1], atol=2e-6, rtol=1e-6)
    assert torch.count_nonzero(h.slot_delta[0].weight) > 0
    assert torch.count_nonzero(h.slot_delta[-1].weight) == 0
    assert not torch.equal(h.slot_embedding.weight[2], h.slot_embedding.weight[0])


def test_source_anchors_and_group_router_are_actually_frozen():
    a, _, _ = migrate(); h = a.hierarchical_options
    assert all(not p.requires_grad for p in h.manager_group.parameters())
    assert all(not p.requires_grad for p in h.worker_residual.parameters())
    assert all(not p.requires_grad for p in h.option_embedding.parameters())
    assert all(not p.requires_grad for p in h.termination.parameters())
    assert all(not p.requires_grad for p in h.termination_option_embedding.parameters())
    assert all(p.requires_grad for p in h.manager_slot.parameters())
    assert all(p.requires_grad for p in h.slot_delta.parameters())
    assert all(p.requires_grad for p in h.slot_embedding.parameters())
    assert all(not p.requires_grad for p in a.actor.parameters())
    assert all(not p.requires_grad for p in a.jepa_world_model.feature_adapter.parameters())


def test_online_frozen_view_shares_live_parameters_but_source_reference_does_not():
    a, _, _ = migrate()
    live = a.hierarchical_options.manager_slot[-1].bias
    frozen = a._frozen_hierarchical_options.manager_slot[-1].bias
    source = a._source_hierarchical_options.manager_slot[-1].bias
    assert live.data_ptr() == frozen.data_ptr()
    assert live.data_ptr() != source.data_ptr()
    assert live.requires_grad and not frozen.requires_grad and not source.requires_grad
    source_before = source.clone()
    with torch.no_grad():
        live.add_(1.0)
    assert torch.equal(frozen, live)
    assert torch.equal(source, source_before)


def test_optimizer_update_is_immediately_visible_to_real_and_imagined_frozen_view():
    a, _, _ = migrate(); h = a.hierarchical_options
    feat = torch.randn(64, h.feature_dim)
    child = torch.full((64,), 2, dtype=torch.long)
    frozen_before = a._frozen_hierarchical_options.residual_logits(feat, child).clone()
    optimizer = torch.optim.Adam(
        list(h.slot_delta.parameters()) + list(h.slot_embedding.parameters()), lr=1e-2
    )
    loss = -h._all_slot_delta_logits(feat)[..., 2, 0].mean()
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    live_after = h.residual_logits(feat, child)
    frozen_after = a._frozen_hierarchical_options.residual_logits(feat, child)
    assert not torch.equal(frozen_before, live_after)
    assert torch.equal(frozen_after, live_after)


def test_anchor_safe_option_identity_exploration_is_behavior_preserving():
    a, _, _ = migrate(); h = a.hierarchical_options
    torch.manual_seed(1234)
    feat = torch.randn(8192, h.feature_dim)
    # Force source group zero for a clean within-group frequency check.
    with torch.no_grad():
        h.manager_group[0].weight.zero_(); h.manager_group[0].bias.zero_()
        h.manager_group[2].weight.zero_(); h.manager_group[2].bias.copy_(torch.tensor([8.0, -8.0]))
        a._frozen_hierarchical_options = copy.deepcopy(h)
    state_option = torch.zeros(8192, dtype=torch.long)
    decision = a._frozen_hierarchical_options.step_option(
        feat,
        state_option,
        torch.ones(8192, dtype=torch.long),
        torch.zeros(8192, dtype=torch.bool),
        torch.ones(8192, dtype=torch.bool),
        deterministic=False,
    )
    slots = a._frozen_hierarchical_options.option_slot(decision.option)
    frequencies = torch.nn.functional.one_hot(slots, 4).float().mean(0)
    assert 0.52 < frequencies[0] < 0.58, frequencies
    assert torch.all((frequencies[1:] > 0.13) & (frequencies[1:] < 0.17)), frequencies

    # Every sampled identity still produces the exact same group-0 action logits.
    base = torch.randn(8192, h.action_logit_dim)
    logits = h.combine_logits(base, feat, decision.option, 0)
    anchor = h.combine_logits(base, feat, torch.zeros_like(decision.option), 0)
    assert torch.allclose(logits, anchor, atol=2e-6, rtol=1e-6)


def test_child_optimizer_step_cannot_move_frozen_anchor_actions():
    a, _, _ = migrate(); h = a.hierarchical_options
    h.set_training_step(200_000)
    feat = torch.randn(64, h.feature_dim)
    anchor_before = h.residual_logits(feat, torch.zeros(64, dtype=torch.long)).detach().clone()
    child = torch.full((64,), 2, dtype=torch.long)
    loss = h.residual_logits(feat, child).square().mean()
    optimizer = torch.optim.Adam(
        list(h.slot_delta.parameters()) + list(h.slot_embedding.parameters()), lr=1e-2
    )
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    anchor_after = h.residual_logits(feat, torch.zeros(64, dtype=torch.long)).detach()
    assert torch.equal(anchor_before, anchor_after)


def test_frozen_snapshot_buffers_do_not_alias_live_buffers():
    a, _, _ = migrate()
    frozen_step = a._frozen_hierarchical_options.training_step.clone()
    a.hierarchical_options.set_training_step(777)
    assert torch.equal(a._frozen_hierarchical_options.training_step, frozen_step)


def test_child_and_slot_manager_update_cannot_move_source_group_router():
    a, _, _ = migrate(); h = a.hierarchical_options
    h.set_training_step(200_000)
    feat = torch.randn(128, h.feature_dim)
    group_before = h.manager_group_probs(feat).detach().clone()
    anchor0_before = h.residual_logits(
        feat, torch.zeros(128, dtype=torch.long)
    ).detach().clone()
    anchor1_before = h.residual_logits(
        feat, torch.ones(128, dtype=torch.long)
    ).detach().clone()
    child = torch.full((128,), 2, dtype=torch.long)
    slot_loss = -h.manager_slot_logits(feat)[..., 0, 1].mean()
    child_loss = h.residual_logits(feat, child).square().mean()
    trainable = [p for p in h.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2, weight_decay=0.1)
    optimizer.zero_grad(set_to_none=True)
    (slot_loss + child_loss).backward()
    optimizer.step()
    assert torch.equal(h.manager_group_probs(feat), group_before)
    assert torch.equal(
        h.residual_logits(feat, torch.zeros(128, dtype=torch.long)), anchor0_before
    )
    assert torch.equal(
        h.residual_logits(feat, torch.ones(128, dtype=torch.long)), anchor1_before
    )
