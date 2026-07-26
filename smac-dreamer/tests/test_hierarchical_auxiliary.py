from __future__ import annotations

import copy
import pathlib
import sys
import types

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/r2dreamer"))


class FakeTensorDict(dict):
    def __init__(self, *args, batch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = tuple(batch_size or ())

    @property
    def shape(self):
        return self.batch_size


fake_td = types.ModuleType("tensordict")
fake_td.TensorDict = FakeTensorDict
sys.modules.setdefault("tensordict", fake_td)


class FakeMaskedMultiOneHotDist:
    def __init__(self, logits, mask, active, shape, unimix_ratio=0.0):
        self.logits = logits
        self.shape = tuple(shape)
        self.a = len(self.shape)
        self.c = self.shape[0]
        lead = logits.shape[:-1]
        raw = logits.reshape(*lead, self.a, self.c).float()
        mask = mask.reshape(*lead, self.a, self.c).bool()
        active = active.reshape(*lead, self.a).bool()
        raw = raw.masked_fill(~mask, -1.0e9)
        probs = raw.softmax(-1)
        if unimix_ratio:
            uniform = mask.float() / mask.float().sum(-1, keepdim=True).clamp_min(1)
            probs = (1 - unimix_ratio) * probs + unimix_ratio * uniform
        self.probs = probs
        self.active = active

    def rsample(self):
        index = torch.distributions.Categorical(probs=self.probs).sample()
        onehot = torch.nn.functional.one_hot(index, self.c).float()
        return onehot.reshape(*self.logits.shape[:-1], self.a * self.c)

    def log_prob(self, action):
        action = action.reshape(*self.logits.shape[:-1], self.a, self.c)
        lp = (self.probs.clamp_min(1e-8).log() * action).sum(-1)
        w = self.active.float()
        return (lp * w).sum(-1) / w.sum(-1).clamp_min(1)

    def entropy(self):
        ent = -(self.probs * self.probs.clamp_min(1e-8).log()).sum(-1)
        w = self.active.float()
        return (ent * w).sum(-1) / w.sum(-1).clamp_min(1)


fake_smac = types.ModuleType("smacdreamer")
fake_mask = types.ModuleType("smacdreamer.masked_actions")
fake_mask.MaskedMultiOneHotDist = FakeMaskedMultiOneHotDist
sys.modules.setdefault("smacdreamer", fake_smac)
sys.modules.setdefault("smacdreamer.masked_actions", fake_mask)

from hierarchical_options import HierarchicalOptionsPolicy  # noqa: E402
from option_critic import OptionCritic  # noqa: E402
from hierarchical_dreamer import (  # noqa: E402
    apply_hierarchy_gradient_guards,
    build_hierarchical_modules,
    hierarchical_auxiliary_loss,
    hierarchy_training_state,
    load_hierarchy_training_state,
)
from test_hierarchical_options import cfg  # noqa: E402


class Dist:
    def __init__(self, value):
        self._value = value
        self.mean = value.sigmoid() if value.shape[-1] == 1 else value

    def mode(self):
        return self._value

    def log_prob(self, target):
        return -(self._value - target).square().squeeze(-1)


class Head(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 24), nn.ELU())
        self.last = nn.Linear(24, out_dim)

    def forward(self, x):
        return Dist(self.last(self.mlp(x)))


class WorldModel(nn.Module):
    def __init__(self, feat_dim, action_dim):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, feat_dim, bias=False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def get_feat(self, stoch, deter):
        return deter

    def img_step(self, stoch, deter, action):
        return stoch, torch.tanh(deter + 0.01 * self.action_proj(action))


class Agent:
    pass


def lambda_return(last, term, reward, value, boot, disc, lamb):
    assert last.shape == term.shape == reward.shape == value.shape == boot.shape
    live = (1 - term.float())[:, 1:] * disc
    cont = (1 - last.float())[:, 1:] * lamb
    interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
    out = [boot[:, -1]]
    for i in reversed(range(live.shape[1])):
        out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
    return torch.stack(list(reversed(out))[:-1], 1)


def make_agent():
    torch.manual_seed(2)
    feat_dim, action_dim = 12, 15
    settings = cfg(
        commitment_warmup_steps=5,
        commitment_full_steps=10,
        termination_warmup_steps=5,
        termination_full_steps=10,
        termination_cap_full_steps=20,
        manager_pg_warmup_steps=5,
        manager_pg_full_steps=10,
        worker_pg_warmup_steps=5,
        worker_pg_full_steps=10,
        worker_scale_warmup_steps=0,
        worker_scale_full_steps=1,
        worker_scale_initial=0.25,
        worker_scale_max=0.25,
        imag_horizon_initial_max=5,
        imag_horizon_final_max=5,
        imag_horizon_window=1,
        imag_horizon_ramp_steps=20,
    )
    agent = Agent()
    agent.hierarchical_enabled = True
    agent.hierarchical_options = HierarchicalOptionsPolicy(feat_dim, action_dim, settings)
    agent.hierarchical_options.set_training_step(20)
    agent._frozen_hierarchical_options = copy.deepcopy(agent.hierarchical_options)
    agent._source_hierarchical_options = copy.deepcopy(agent.hierarchical_options)
    for module in (
        agent._frozen_hierarchical_options,
        agent._source_hierarchical_options,
    ):
        for p in module.parameters():
            p.requires_grad_(False)
    agent.option_critic = OptionCritic(feat_dim, settings)
    agent._slow_option_critic = copy.deepcopy(agent.option_critic)
    agent._option_slow_updates = 0
    for p in agent._slow_option_critic.parameters():
        p.requires_grad_(False)

    agent.actor = Head(feat_dim, action_dim)
    for p in agent.actor.parameters():
        p.requires_grad_(False)
    agent._frozen_actor = copy.deepcopy(agent.actor)
    agent._frozen_jepa_world_model = WorldModel(feat_dim, action_dim)
    agent.value = Head(feat_dim, 1)
    agent._frozen_value = copy.deepcopy(agent.value)
    agent._frozen_slow_value = copy.deepcopy(agent.value)
    agent._frozen_reward = Head(feat_dim, 1)
    agent._frozen_cont = Head(feat_dim, 1)
    for module in (
        agent._frozen_value,
        agent._frozen_slow_value,
        agent._frozen_reward,
        agent._frozen_cont,
    ):
        for p in module.parameters():
            p.requires_grad_(False)
    agent._actor_shape = (5, 5, 5)
    agent._actor_unimix = 0.01
    agent.imag_horizon = 5
    agent.horizon = 100
    agent.lamb = 0.95
    agent._lambda_return = lambda_return

    def predicted_mask(feat):
        lead = feat.shape[:-1]
        return (
            torch.ones(*lead, 3, 5, dtype=torch.bool, device=feat.device),
            torch.ones(*lead, 3, dtype=torch.bool, device=feat.device),
        )

    agent._predicted_action_mask = predicted_mask
    return agent



def test_world_model_gradients_are_frozen_then_throttled():
    class LiveWorldModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_adapter = nn.Linear(4, 4)
            self.core = nn.Linear(4, 4, bias=False)

    class BuildAgent:
        pass

    agent = BuildAgent()
    agent.world_model_backend = "jepa"
    agent.action_masking = True
    agent.act_discrete = True
    agent.feat_size = 12
    agent._actor_shape = (5, 5, 5)
    agent.actor = nn.Linear(4, 4)
    agent.jepa_world_model = LiveWorldModel()
    config = types.SimpleNamespace(
        compile=False,
        hierarchical_options=types.SimpleNamespace(**cfg(
            world_model_grad_scale_final=0.10,
        )),
    )
    build_hierarchical_modules(agent, config)

    assert all(not p.requires_grad for p in agent.actor.parameters())
    assert all(
        not p.requires_grad
        for p in agent.jepa_world_model.feature_adapter.parameters()
    )

    core = agent.jepa_world_model.core.weight
    agent.hierarchical_options.set_training_step(0)
    core.sum().backward()
    assert core.grad is not None
    assert torch.count_nonzero(core.grad) == 0

    apply_hierarchy_gradient_guards(agent)
    assert core.grad is None

    agent.hierarchical_options.set_training_step(200)
    core.sum().backward()
    assert torch.allclose(core.grad, torch.full_like(core.grad, 0.10))


def test_hierarchical_auxiliary_loss_is_finite_and_routes_gradients():
    agent = make_agent()
    b, t = 2, 4
    raw = FakeTensorDict(
        {
            "option_id": torch.tensor([[0, 0, 1, 1], [2, 2, 2, 3]]).unsqueeze(-1),
            "option_age": torch.tensor([[4, 5, 6, 7], [3, 4, 5, 6]]).unsqueeze(-1),
            "option_has": torch.ones(b, t, 1, dtype=torch.bool),
            "option_before_id": torch.tensor(
                [[0, 0, 1, 1], [2, 2, 2, 3]]
            ).unsqueeze(-1),
            "option_before_age": torch.tensor(
                [[3, 4, 5, 6], [2, 3, 4, 5]]
            ).unsqueeze(-1),
            "option_before_has": torch.ones(b, t, 1, dtype=torch.bool),
            "option_action_age": torch.tensor(
                [[3, 4, 5, 6], [2, 3, 4, 5]]
            ).unsqueeze(-1),
            "option_started": torch.zeros(b, t, 1, dtype=torch.bool),
            "option_terminated": torch.zeros(b, t, 1, dtype=torch.bool),
            "option_termination_eligible": torch.ones(
                b, t, 1, dtype=torch.bool
            ),
            "option_termination_prob": torch.full((b, t, 1), 0.1),
            "is_first": torch.zeros(b, t, 1),
            "is_last": torch.zeros(b, t, 1),
        },
        batch_size=(b, t),
    )
    post_stoch = torch.zeros(b, t, 1)
    post_deter = torch.randn(b, t, 12)
    loss, metrics = hierarchical_auxiliary_loss(agent, raw, post_stoch, post_deter)
    assert torch.isfinite(loss)
    loss.backward()
    assert agent.hierarchical_options.manager_group[0].weight.grad is not None
    assert agent.hierarchical_options.worker_residual[0].weight.grad is not None
    assert agent.hierarchical_options.termination[0].weight.grad is not None
    assert agent.option_critic.trunk[0].weight.grad is not None
    assert agent.value.mlp[0].weight.grad is not None
    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert "option/eligible_learned_beta_mean" in metrics
    assert "option/same_option_reselection_rate" in metrics
    assert "option/real_boundary_rate" in metrics
    assert "option/real_usage_0" in metrics
    assert "option/source_manager_group_kl_mean" in metrics
    assert "option/source_manager_group_kl_tail" in metrics
    assert "option/source_manager_group_high_confidence_flip_rate" in metrics
    assert "option/source_manager_group_live_0" in metrics
    assert "option/source_manager_group_reference_1" in metrics
    for key in (
        "option/imag_min_duration_violation_rate",
        "option/imag_max_duration_violation_rate",
        "option/imag_change_without_boundary_rate",
        "option/real_min_duration_violation_rate",
        "option/real_max_duration_violation_rate",
        "option/real_change_without_boundary_rate",
    ):
        assert metrics[key] == 0
    assert "option/imag_return_0" in metrics
    assert "option/imag_weight_0" in metrics
    # Zero-initialized online/slow option residuals provide no trustworthy
    # termination advantage yet, so the reliability gate must stay closed.
    assert metrics["option/termination_reliable_fraction"] == 0



def test_episode_start_manager_boundary_receives_task_gradient():
    agent = make_agent()
    b, t = 2, 3
    raw = FakeTensorDict(
        {
            "option_id": torch.zeros(b, t, 1, dtype=torch.long),
            "option_age": torch.ones(b, t, 1, dtype=torch.long),
            "option_has": torch.ones(b, t, 1, dtype=torch.bool),
            "option_before_id": torch.zeros(b, t, 1, dtype=torch.long),
            "option_before_age": torch.zeros(b, t, 1, dtype=torch.long),
            "option_before_has": torch.zeros(b, t, 1, dtype=torch.bool),
            "option_action_age": torch.ones(b, t, 1, dtype=torch.long),
            "option_started": torch.ones(b, t, 1, dtype=torch.bool),
            "option_terminated": torch.zeros(b, t, 1, dtype=torch.bool),
            "option_termination_eligible": torch.ones(b, t, 1, dtype=torch.bool),
            "option_termination_prob": torch.ones(b, t, 1),
            "is_first": torch.ones(b, t, 1),
            "is_last": torch.zeros(b, t, 1),
        },
        batch_size=(b, t),
    )
    loss, metrics = hierarchical_auxiliary_loss(
        agent,
        raw,
        torch.zeros(b, t, 1),
        torch.randn(b, t, 12),
    )
    loss.backward()
    grad = agent.hierarchical_options.manager_group[0].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
    assert metrics["option/manager_boundary_count"] > 0


def test_termination_head_is_frozen_during_fixed_hazard_warmup():
    agent = make_agent()
    agent.hierarchical_options.set_training_step(0)
    agent._frozen_hierarchical_options.set_training_step(0)
    b, t = 2, 4
    raw = FakeTensorDict(
        {
            "option_id": torch.zeros(b, t, 1, dtype=torch.long),
            "option_age": torch.full((b, t, 1), 4, dtype=torch.long),
            "option_has": torch.ones(b, t, 1, dtype=torch.bool),
            "option_before_id": torch.zeros(b, t, 1, dtype=torch.long),
            "option_before_age": torch.full((b, t, 1), 3, dtype=torch.long),
            "option_before_has": torch.ones(b, t, 1, dtype=torch.bool),
            "option_action_age": torch.full((b, t, 1), 3, dtype=torch.long),
            "option_started": torch.zeros(b, t, 1, dtype=torch.bool),
            "option_terminated": torch.zeros(b, t, 1, dtype=torch.bool),
            "option_termination_eligible": torch.ones(b, t, 1, dtype=torch.bool),
            "option_termination_prob": torch.full((b, t, 1), 0.1),
            "is_first": torch.zeros(b, t, 1),
            "is_last": torch.zeros(b, t, 1),
        },
        batch_size=(b, t),
    )
    loss, _ = hierarchical_auxiliary_loss(
        agent,
        raw,
        torch.zeros(b, t, 1),
        torch.randn(b, t, 12),
    )
    loss.backward()
    grads = [p.grad for p in agent.hierarchical_options.termination.parameters()]
    grads.append(agent.hierarchical_options.termination_option_embedding.weight.grad)
    grads.append(agent.hierarchical_options.age_embedding.weight.grad)
    assert all(g is None or torch.count_nonzero(g) == 0 for g in grads)




def test_hierarchy_control_counters_round_trip_through_training_state():
    agent = make_agent()
    agent.hierarchical_options.set_training_step(123)
    agent.hierarchical_options.set_diversity_calls(17)
    agent.hierarchical_options.set_horizon_calls(9)
    saved = hierarchy_training_state(agent)
    agent.hierarchical_options.set_training_step(0)
    agent.hierarchical_options.set_diversity_calls(0)
    agent.hierarchical_options.set_horizon_calls(0)
    load_hierarchy_training_state(agent, {"hierarchical_options": saved})
    assert agent.hierarchical_options._training_step_int == 123
    assert agent.hierarchical_options._diversity_calls_int == 17
    assert int(agent.hierarchical_options.training_step.cpu()) == 123
    assert int(agent.hierarchical_options.diversity_calls.cpu()) == 17
    assert agent.hierarchical_options._horizon_calls_int == 9
    assert int(agent.hierarchical_options.horizon_calls.cpu()) == 9


def test_real_posterior_source_trust_region_is_logged_and_backpropagates():
    agent = make_agent()
    b, t = 2, 3
    raw = FakeTensorDict(
        {
            "option_id": torch.zeros(b, t, 1, dtype=torch.long),
            "option_age": torch.ones(b, t, 1, dtype=torch.long),
            "option_has": torch.ones(b, t, 1, dtype=torch.bool),
            "option_before_id": torch.zeros(b, t, 1, dtype=torch.long),
            "option_before_age": torch.ones(b, t, 1, dtype=torch.long),
            "option_before_has": torch.ones(b, t, 1, dtype=torch.bool),
            "option_action_age": torch.ones(b, t, 1, dtype=torch.long),
            "option_started": torch.zeros(b, t, 1, dtype=torch.bool),
            "option_terminated": torch.zeros(b, t, 1, dtype=torch.bool),
            "option_termination_eligible": torch.ones(b, t, 1, dtype=torch.bool),
            "option_termination_prob": torch.full((b, t, 1), 0.1),
            "is_first": torch.zeros(b, t, 1),
            "is_last": torch.zeros(b, t, 1),
        },
        batch_size=(b, t),
    )
    # Deliberately perturb the live worker after the source copy was created.
    with torch.no_grad():
        agent.hierarchical_options.worker_residual[-1].bias.add_(0.25)
    loss, metrics = hierarchical_auxiliary_loss(
        agent, raw, torch.zeros(b, t, 1), torch.randn(b, t, 12)
    )
    assert "option/real_source_policy_kl_mean" in metrics
    assert "option/real_source_manager_group_kl_mean" in metrics
    assert metrics["option/real_source_policy_kl_mean"] >= 0
    loss.backward()
    grad = agent.hierarchical_options.worker_residual[-1].bias.grad
    assert grad is not None and torch.isfinite(grad).all()
