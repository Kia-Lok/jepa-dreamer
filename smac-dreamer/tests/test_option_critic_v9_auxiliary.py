from __future__ import annotations

import copy
from dataclasses import replace
import pathlib
import sys
import types

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/r2dreamer"))


class FakeTensorDict(dict):
    def __init__(self, *args, batch_size=None, **kwargs):
        super().__init__(*args, **kwargs); self.batch_size = tuple(batch_size or ())
    @property
    def shape(self): return self.batch_size

fake_td = types.ModuleType("tensordict"); fake_td.TensorDict = FakeTensorDict
sys.modules.setdefault("tensordict", fake_td)


class FakeMaskedMultiOneHotDist:
    def __init__(self, logits, mask, active, shape, unimix_ratio=0.0):
        self.logits = logits; self.shape = tuple(shape); a=len(shape); c=shape[0]
        lead=logits.shape[:-1]; raw=logits.reshape(*lead,a,c).float()
        mask=mask.reshape(*lead,a,c).bool(); active=active.reshape(*lead,a).bool()
        raw=raw.masked_fill(~mask,-1e9); probs=raw.softmax(-1)
        if unimix_ratio:
            uniform=mask.float()/mask.float().sum(-1,keepdim=True).clamp_min(1)
            probs=(1-unimix_ratio)*probs+unimix_ratio*uniform
        self.probs=probs; self.active=active; self.a=a; self.c=c
    def rsample(self):
        idx=torch.distributions.Categorical(probs=self.probs).sample()
        return torch.nn.functional.one_hot(idx,self.c).float().reshape(*self.logits.shape[:-1],self.a*self.c)
    def log_prob(self,action):
        action=action.reshape(*self.logits.shape[:-1],self.a,self.c)
        lp=(self.probs.clamp_min(1e-8).log()*action).sum(-1); w=self.active.float()
        return (lp*w).sum(-1)/w.sum(-1).clamp_min(1)
    def entropy(self):
        ent=-(self.probs*self.probs.clamp_min(1e-8).log()).sum(-1); w=self.active.float()
        return (ent*w).sum(-1)/w.sum(-1).clamp_min(1)

fake_smac=types.ModuleType("smacdreamer"); fake_mask=types.ModuleType("smacdreamer.masked_actions")
fake_mask.MaskedMultiOneHotDist=FakeMaskedMultiOneHotDist
sys.modules.setdefault("smacdreamer",fake_smac); sys.modules.setdefault("smacdreamer.masked_actions",fake_mask)

from hierarchical_options import HierarchicalOptionsPolicy  # noqa: E402
from option_critic import OptionCritic  # noqa: E402
from hierarchical_dreamer import hierarchical_auxiliary_loss  # noqa: E402


def cfg():
    return dict(
        enabled=True,num_options=8,source_manager_group_count=2,min_duration=1,max_duration=4,
        manager_unimix_initial=0.,manager_unimix_final=0.,manager_unimix_decay_steps=1,
        slot_manager_unimix=.01,slot_anchor_floor=.40,worker_pg_warmup_steps=0,worker_pg_full_steps=1,
        manager_pg_warmup_steps=0,manager_pg_full_steps=1,termination_warmup_steps=800,
        termination_full_steps=801,termination_cap_full_steps=802,termination_loss_scale=0.,
        world_model_grad_scale_initial=0.,world_model_grad_scale_final=0.,
        imag_horizon_initial_max=15,imag_horizon_final_max=15,imag_horizon_window=1,imag_horizon_ramp_steps=1,
    )


class Dist:
    def __init__(self,x): self.x=x; self.mean=x.sigmoid() if x.shape[-1]==1 else x
    def mode(self): return self.x
    def log_prob(self,target): return -(self.x-target).square().squeeze(-1)

class Head(nn.Module):
    def __init__(self,din,dout=1):
        super().__init__(); self.mlp=nn.Sequential(nn.Linear(din,24),nn.ELU()); self.last=nn.Linear(24,dout)
    def forward(self,x): return Dist(self.last(self.mlp(x)))

class World(nn.Module):
    def __init__(self,fd,ad):
        super().__init__(); self.proj=nn.Linear(ad,fd,bias=False)
        for p in self.parameters(): p.requires_grad_(False)
    def get_feat(self,stoch,deter): return deter
    def img_step(self,stoch,deter,action): return stoch,torch.tanh(deter+.01*self.proj(action))

class Agent: pass


def make_agent(*, step=400, actual_schedule=False):
    torch.manual_seed(9); fd=12; ad=15; a=Agent(); c=cfg()
    if actual_schedule:
        c.update(
            worker_pg_warmup_steps=20,
            worker_pg_full_steps=150,
            manager_pg_warmup_steps=100,
            manager_pg_full_steps=300,
        )
    a.hierarchical_enabled=True; a.hierarchical_options=HierarchicalOptionsPolicy(fd,ad,c)
    a.hierarchical_options.set_training_step(step)
    for module in (a.hierarchical_options.manager_group,a.hierarchical_options.worker_residual,
                   a.hierarchical_options.option_embedding,a.hierarchical_options.termination,
                   a.hierarchical_options.termination_option_embedding):
        for p in module.parameters(): p.requires_grad_(False)
    a._frozen_hierarchical_options=copy.deepcopy(a.hierarchical_options)
    a._source_hierarchical_options=copy.deepcopy(a.hierarchical_options)
    for m in (a._frozen_hierarchical_options,a._source_hierarchical_options):
        for p in m.parameters(): p.requires_grad_(False)
    a.option_critic=OptionCritic(fd,c); a._slow_option_critic=copy.deepcopy(a.option_critic)
    for p in a._slow_option_critic.parameters(): p.requires_grad_(False)
    a.actor=Head(fd,ad)
    for p in a.actor.parameters(): p.requires_grad_(False)
    a._frozen_actor=copy.deepcopy(a.actor); a._frozen_jepa_world_model=World(fd,ad)
    a.value=Head(fd,1); a._frozen_slow_value=copy.deepcopy(a.value)
    a._frozen_reward=Head(fd,1); a._frozen_cont=Head(fd,1)
    for m in (a._frozen_slow_value,a._frozen_reward,a._frozen_cont):
        for p in m.parameters(): p.requires_grad_(False)
    a._actor_shape=(5,5,5); a._actor_unimix=.01; a.horizon=100; a.lamb=.95
    def masks(feat):
        lead=feat.shape[:-1]
        return torch.ones(*lead,3,5,dtype=torch.bool),torch.ones(*lead,3,dtype=torch.bool)
    a._predicted_action_mask=masks
    return a


def test_full_auxiliary_loss_is_finite_and_routes_only_intended_gradients():
    a=make_agent(); b,t=2,3
    raw=FakeTensorDict({
        "option_id":torch.tensor([[0,2,4],[1,3,5]]).unsqueeze(-1),
        "option_age":torch.tensor([[1,2,3],[1,2,3]]).unsqueeze(-1),
        "option_has":torch.ones(b,t,1,dtype=torch.bool),
        "option_before_id":torch.tensor([[0,2,4],[1,3,5]]).unsqueeze(-1),
        "option_before_age":torch.tensor([[0,1,2],[0,1,2]]).unsqueeze(-1),
        "option_before_has":torch.ones(b,t,1,dtype=torch.bool),
        "option_action_age":torch.tensor([[0,1,2],[0,1,2]]).unsqueeze(-1),
        "option_started":torch.ones(b,t,1,dtype=torch.bool),
        "option_terminated":torch.zeros(b,t,1,dtype=torch.bool),
        "option_termination_eligible":torch.ones(b,t,1,dtype=torch.bool),
        "option_termination_prob":torch.zeros(b,t,1),
        "is_first":torch.zeros(b,t,1),"is_last":torch.zeros(b,t,1),
    },batch_size=(b,t))
    loss,metrics=hierarchical_auxiliary_loss(a,raw,torch.zeros(b,t,1),torch.randn(b,t,12))
    assert torch.isfinite(loss); loss.backward()
    assert a.hierarchical_options.manager_slot[-1].weight.grad is not None
    assert torch.count_nonzero(a.hierarchical_options.manager_slot[-1].weight.grad)>0
    assert a.hierarchical_options.slot_delta[-1].weight.grad is not None
    assert torch.count_nonzero(a.hierarchical_options.slot_delta[-1].weight.grad)>0
    assert a.option_critic.trunk[0].weight.grad is not None
    assert a.value.mlp[0].weight.grad is not None
    for module in (a.hierarchical_options.manager_group,a.hierarchical_options.worker_residual,
                   a.hierarchical_options.termination):
        assert all(p.grad is None for p in module.parameters())
    assert all(torch.isfinite(v).all() for v in metrics.values())
    for key in ("option/imag_min_duration_violation_rate","option/imag_max_duration_violation_rate",
                "option/imag_change_without_boundary_rate"):
        assert metrics[key] == 0


def test_anchor_only_imagination_does_not_create_fake_child_worker_gradient(monkeypatch):
    a=make_agent(); b,t=2,3
    # Force both source groups to choose their immutable anchor identity.
    a.hierarchical_options.settings = replace(
        a.hierarchical_options.settings, slot_manager_unimix=0.0
    )
    with torch.no_grad():
        a.hierarchical_options.manager_slot[-1].weight.zero_()
        a.hierarchical_options.manager_slot[-1].bias.copy_(
            torch.tensor([20.,20.,-20.,-20.,-20.,-20.,-20.,-20.])
        )
        a._frozen_hierarchical_options=copy.deepcopy(a.hierarchical_options)
    raw=FakeTensorDict({
        "option_id":torch.tensor([[0,0,0],[1,1,1]]).unsqueeze(-1),
        "option_age":torch.tensor([[1,2,3],[1,2,3]]).unsqueeze(-1),
        "option_has":torch.ones(b,t,1,dtype=torch.bool),
        "option_before_id":torch.tensor([[0,0,0],[1,1,1]]).unsqueeze(-1),
        "option_before_age":torch.tensor([[0,1,2],[0,1,2]]).unsqueeze(-1),
        "option_before_has":torch.ones(b,t,1,dtype=torch.bool),
        "option_action_age":torch.tensor([[0,1,2],[0,1,2]]).unsqueeze(-1),
        "option_started":torch.ones(b,t,1,dtype=torch.bool),
        "option_terminated":torch.zeros(b,t,1,dtype=torch.bool),
        "option_termination_eligible":torch.ones(b,t,1,dtype=torch.bool),
        "option_termination_prob":torch.zeros(b,t,1),
        "is_first":torch.zeros(b,t,1),"is_last":torch.zeros(b,t,1),
    },batch_size=(b,t))
    loss,metrics=hierarchical_auxiliary_loss(a,raw,torch.zeros(b,t,1),torch.randn(b,t,12))
    assert torch.isfinite(loss); loss.backward()
    grad=a.hierarchical_options.slot_delta[-1].weight.grad
    assert grad is None or torch.count_nonzero(grad)==0
    assert metrics["option/worker_trainable_child_fraction"] == 0



def _raw_batch(batch=2, time=3):
    return FakeTensorDict({
        "option_id":torch.tensor([[0,2,4],[1,3,5]]).unsqueeze(-1),
        "option_age":torch.tensor([[1,2,3],[1,2,3]]).unsqueeze(-1),
        "option_has":torch.ones(batch,time,1,dtype=torch.bool),
        "option_before_id":torch.tensor([[0,2,4],[1,3,5]]).unsqueeze(-1),
        "option_before_age":torch.tensor([[0,1,2],[0,1,2]]).unsqueeze(-1),
        "option_before_has":torch.ones(batch,time,1,dtype=torch.bool),
        "option_action_age":torch.tensor([[0,1,2],[0,1,2]]).unsqueeze(-1),
        "option_started":torch.ones(batch,time,1,dtype=torch.bool),
        "option_terminated":torch.zeros(batch,time,1,dtype=torch.bool),
        "option_termination_eligible":torch.ones(batch,time,1,dtype=torch.bool),
        "option_termination_prob":torch.zeros(batch,time,1),
        "is_first":torch.zeros(batch,time,1),"is_last":torch.zeros(batch,time,1),
    },batch_size=(batch,time))


def _nonzero_or_none(module):
    grads=[p.grad for p in module.parameters()]
    return any(g is not None and torch.count_nonzero(g)>0 for g in grads)


def _all_finite_grads(module):
    return all(g is None or torch.isfinite(g).all() for g in (p.grad for p in module.parameters()))


def test_actual_schedule_routes_gradients_stage_by_stage_under_anomaly_detection():
    # Step 0: critic/value warm-up only. Children and slot manager must not move.
    for step, expect_worker, expect_manager in (
        (0, False, False),
        (100, True, False),
        (200, True, True),
    ):
        a=make_agent(step=step, actual_schedule=True)
        raw=_raw_batch()
        with torch.autograd.detect_anomaly():
            loss,metrics=hierarchical_auxiliary_loss(
                a,raw,torch.zeros(2,3,1),torch.randn(2,3,12)
            )
            assert torch.isfinite(loss)
            loss.backward()
        assert _nonzero_or_none(a.hierarchical_options.slot_delta) is expect_worker
        assert _nonzero_or_none(a.hierarchical_options.manager_slot) is expect_manager
        assert _nonzero_or_none(a.option_critic)
        assert _nonzero_or_none(a.value)
        for module in (
            a.hierarchical_options,
            a.option_critic,
            a.value,
        ):
            assert _all_finite_grads(module)
        assert torch.isfinite(metrics["option/critic_consistency_loss"])
        expected_consistency=max(0.0,1.0-a.hierarchical_options.worker_pg_blend(step))
        assert abs(float(metrics["option/critic_consistency_blend"])-expected_consistency)<1e-6


def test_zero_worker_blend_adamw_step_cannot_change_child_or_anchor_behavior():
    a=make_agent(step=0, actual_schedule=True)
    h=a.hierarchical_options
    feat=torch.randn(128,h.feature_dim)
    ids=torch.tensor([0,1,2,3,4,5,6,7]).repeat(16)
    before=h.residual_logits(feat,ids).detach().clone()
    # A zero-blend score-function objective still creates zero gradient tensors;
    # AdamW may decay hidden parameters, but the exact-zero output layer must keep
    # primitive behavior unchanged throughout the critic-only warm-up.
    logit=h._all_slot_delta_logits(feat)
    loss=0.0*logit.square().mean()
    opt=torch.optim.AdamW(
        list(h.slot_delta.parameters())+list(h.slot_embedding.parameters()),
        lr=1e-2,weight_decay=0.1,
    )
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    after=h.residual_logits(feat,ids).detach()
    assert torch.equal(before,after)
    anchors=torch.tensor([0,1]).repeat_interleave(64)
    assert torch.equal(
        h.slot_delta_scale_by_option(anchors),torch.zeros_like(anchors,dtype=torch.float32)
    )
