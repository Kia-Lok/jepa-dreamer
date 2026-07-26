"""Tests for the mask-aware multi-one-hot action distribution (P0.1/P0.2 core). Pure torch."""

import pytest

torch = pytest.importorskip("torch")

from smacdreamer.masked_actions import (
    MaskedMultiOneHotDist, build_action_mask, hard_mask_from_logits,
    mask_quality_metrics, invalid_mass_and_greedy_rate, empty_mask_rate, NOOP_INDEX,
)


A, C = 3, 4   # 3 agent slots, 4 actions each


def _setup(active_rows):
    """Build (logits, mask, active) for a batch. active_rows: list of per-agent active flags.

    avail: agent0 valid {1,2}; agent1 valid {0,3}; agent2 valid {1}. Logits deliberately FAVOUR
    invalid actions to prove masking overrides them.
    """
    B = len(active_rows)
    avail = torch.zeros(B, A, C)
    avail[:, 0, [1, 2]] = 1.0
    avail[:, 1, [0, 3]] = 1.0
    avail[:, 2, [1]] = 1.0
    agent_active = torch.tensor(active_rows, dtype=torch.float32)
    mask, active = build_action_mask(avail.reshape(B, A * C), agent_active, A, C)
    logits = torch.zeros(B, A, C)
    logits[:, 0, [0, 3]] = 10.0   # high logit on INVALID actions for agent 0
    logits[:, 1, [1, 2]] = 10.0   # high logit on INVALID actions for agent 1
    return logits.reshape(B, A * C), mask, active


def _per_agent(flat):
    return flat.reshape(flat.shape[0], A, C)


# ----------------------------------------------------------------------
# Invalid actions: never sampled, ~zero probability
# ----------------------------------------------------------------------

def test_invalid_actions_get_zero_probability():
    logits, mask, active = _setup([[1, 1, 1]])
    d = MaskedMultiOneHotDist(logits, mask, active, shape=(C,) * A)
    p = d.probs[0]
    assert p[0, 0] < 1e-6 and p[0, 3] < 1e-6     # agent0 invalid actions
    assert p[1, 1] < 1e-6 and p[1, 2] < 1e-6     # agent1 invalid actions
    assert torch.allclose(p.sum(-1), torch.ones(A), atol=1e-5)


def test_mode_and_sample_never_pick_invalid_real_action():
    logits, mask, active = _setup([[1, 1, 1]])
    d = MaskedMultiOneHotDist(logits, mask, active, shape=(C,) * A)
    valid = mask[0]  # (A, C)
    # mode
    m = _per_agent(d.mode)[0]
    for a in range(A):
        chosen = int(torch.argmax(m[a]))
        assert valid[a, chosen], f"mode picked invalid action {chosen} for agent {a}"
    # many stochastic samples
    for _ in range(50):
        s = _per_agent(d.rsample())[0]
        for a in range(A):
            chosen = int(torch.argmax(s[a]))
            assert valid[a, chosen], f"sample picked invalid action {chosen} for agent {a}"


# ----------------------------------------------------------------------
# Padded / dead agents -> deterministic NOOP
# ----------------------------------------------------------------------

def test_padded_and_dead_agents_execute_noop():
    # agent2 padded/dead (inactive) in both batch rows.
    logits, mask, active = _setup([[1, 1, 0], [1, 0, 0]])
    d = MaskedMultiOneHotDist(logits, mask, active, shape=(C,) * A)
    for getter in (lambda: d.mode, lambda: d.rsample()):
        oh = _per_agent(getter())
        # row0 agent2 inactive, row1 agents 1 and 2 inactive -> all forced NOOP
        assert int(torch.argmax(oh[0, 2])) == NOOP_INDEX
        assert int(torch.argmax(oh[1, 1])) == NOOP_INDEX
        assert int(torch.argmax(oh[1, 2])) == NOOP_INDEX
        assert oh[0, 2, NOOP_INDEX] == 1.0 and oh[0, 2].sum() == 1.0


def test_build_action_mask_inactive_is_noop_only_and_empty_support_gets_noop():
    B = 1
    avail = torch.zeros(B, A, C)            # NO valid actions anywhere
    agent_active = torch.tensor([[1, 0, 1]], dtype=torch.float32)
    mask, active = build_action_mask(avail.reshape(B, A * C), agent_active, A, C)
    # active agents (0, 2) had empty support -> NOOP enabled
    assert mask[0, 0, NOOP_INDEX] and mask[0, 2, NOOP_INDEX]
    # inactive agent (1) -> NOOP only
    assert mask[0, 1, NOOP_INDEX]
    assert mask[0, 1].sum() == 1
    assert bool(active[0, 0]) and not bool(active[0, 1]) and bool(active[0, 2])


# ----------------------------------------------------------------------
# log-prob / entropy exclude padded/dead slots (normalised over living)
# ----------------------------------------------------------------------

def test_logprob_and_entropy_exclude_inactive_slots():
    logits, mask, active = _setup([[1, 1, 1]])      # all active
    d_all = MaskedMultiOneHotDist(logits, mask, active, shape=(C,) * A)
    logits2, mask2, active2 = _setup([[1, 1, 0]])   # agent2 inactive
    d_excl = MaskedMultiOneHotDist(logits2, mask2, active2, shape=(C,) * A)

    # entropy is a MEAN over active agents -> excluding an agent changes the count, not a sum.
    ent_all = d_all.entropy()[0]
    ent_excl = d_excl.entropy()[0]
    assert torch.isfinite(ent_all) and torch.isfinite(ent_excl)

    # build a valid one-hot action and check log_prob ignores the inactive slot entirely.
    act = d_excl.mode
    lp_excl = d_excl.log_prob(act)[0]
    # Manually: only agents 0 and 1 (active) contribute; normalised by 2.
    v = _per_agent(act)[0]
    manual = (d_excl.log_probs[0, 0] * v[0]).sum() + (d_excl.log_probs[0, 1] * v[1]).sum()
    manual = manual / 2.0
    assert torch.allclose(lp_excl, manual, atol=1e-5)


def test_unimix_spread_over_valid_only():
    logits, mask, active = _setup([[1, 1, 1]])
    d = MaskedMultiOneHotDist(logits, mask, active, shape=(C,) * A, unimix_ratio=0.5)
    p = d.probs[0]
    # Even with heavy unimix, invalid actions stay ~0 (uniform is over VALID actions only).
    assert p[0, 0] < 1e-6 and p[0, 3] < 1e-6
    assert p[1, 1] < 1e-6 and p[1, 2] < 1e-6


def test_hard_mask_from_logits_threshold():
    avail_logits = torch.tensor([[-1.0, 0.5, 2.0, -3.0]])
    m = hard_mask_from_logits(avail_logits, threshold_logit=0.0)
    assert m.tolist() == [[0.0, 1.0, 1.0, 0.0]]


# ----------------------------------------------------------------------
# Leading time dim (imagination): (B, T, A*C)
# ----------------------------------------------------------------------

def test_masking_handles_time_dimension():
    B, T = 2, 5
    avail = torch.zeros(B, T, A, C)
    avail[..., 0, [1, 2]] = 1.0
    avail[..., 1, [0, 3]] = 1.0
    avail[..., 2, [1]] = 1.0
    active = torch.ones(B, T, A)
    active[..., 2] = 0.0   # agent2 inactive across all (B,T)
    mask, act = build_action_mask(avail.reshape(B, T, A * C), active, A, C)
    assert mask.shape == (B, T, A, C)
    logits = torch.randn(B, T, A * C)
    d = MaskedMultiOneHotDist(logits, mask, act, shape=(C,) * A)
    m = d.mode
    assert m.shape == (B, T, A * C)
    assert d.log_prob(m).shape == (B, T)
    assert d.entropy().shape == (B, T)
    # invalid actions never chosen for active agents; inactive -> NOOP
    mm = m.reshape(B, T, A, C)
    for b in range(B):
        for t in range(T):
            assert mask[b, t, 0, int(torch.argmax(mm[b, t, 0]))]
            assert int(torch.argmax(mm[b, t, 2])) == NOOP_INDEX


# ----------------------------------------------------------------------
# Metrics: mask precision/recall/FPR + unmasked invalid rate
# ----------------------------------------------------------------------

def test_mask_quality_metrics():
    # pred avail at logit>=0; true availability given separately.
    pred_logits = torch.tensor([[2.0, -1.0, 3.0, -2.0]])   # predicts avail {0,2}
    true = torch.tensor([[1.0, 1.0, 0.0, 0.0]])            # truly avail {0,1}
    m = mask_quality_metrics(pred_logits, true, threshold_logit=0.0)
    # TP={0}=1, FP={2}=1, FN={1}=1, TN={3}=1
    assert m["precision"] == pytest.approx(0.5, abs=1e-4)
    assert m["recall"] == pytest.approx(0.5, abs=1e-4)
    assert m["fpr"] == pytest.approx(0.5, abs=1e-4)


def test_post_mask_invalid_sample_rate_is_zero_invariant():
    # The core invariant: a MASKED sample is never invalid, in 2D and 3D (imagination) shapes.
    logits, mask, active = _setup([[1, 1, 1], [1, 1, 0]])
    d = MaskedMultiOneHotDist(logits, mask, active, shape=(C,) * A)
    assert float(d.post_mask_invalid_sample_rate()) == pytest.approx(0.0, abs=1e-7)
    B, T = 2, 4
    avail = torch.zeros(B, T, A, C); avail[..., 0, [1, 2]] = 1; avail[..., 1, [0]] = 1; avail[..., 2, [3]] = 1
    act = torch.ones(B, T, A); act[..., 2] = 0
    m, a = build_action_mask(avail.reshape(B, T, A * C), act, A, C)
    d3 = MaskedMultiOneHotDist(torch.randn(B, T, A * C), m, a, shape=(C,) * A)
    assert float(d3.post_mask_invalid_sample_rate()) == pytest.approx(0.0, abs=1e-7)


def test_extreme_logits_no_nan():
    logits, mask, active = _setup([[1, 1, 1]])
    logits = logits.clone()
    logits[:] = 1e9            # extreme positive logits everywhere
    logits[0, :4] = -1e9       # and extreme negative for agent0
    d = MaskedMultiOneHotDist(logits, mask, active, shape=(C,) * A)
    assert torch.isfinite(d.probs).all()
    assert torch.isfinite(d.log_prob(d.mode)).all()
    assert torch.isfinite(d.entropy()).all()


def test_flatten_reshape_preserves_agent_action_ordering():
    # Make each agent's single valid action unique so the flat one-hot position is unambiguous.
    B = 1
    avail = torch.zeros(B, A, C)
    valid_idx = [1, 3, 2]                       # agent a -> its only valid action
    for a in range(A):
        avail[0, a, valid_idx[a]] = 1.0
    mask, active = build_action_mask(avail.reshape(B, A * C), torch.ones(B, A), A, C)
    d = MaskedMultiOneHotDist(torch.zeros(B, A * C), mask, active, shape=(C,) * A)
    flat = d.mode[0]                            # (A*C,)
    for a in range(A):
        block = flat[a * C:(a + 1) * C]
        assert int(torch.argmax(block)) == valid_idx[a]   # ordering preserved: agent a @ block a


def test_invalid_mass_and_empty_mask_helpers():
    logits, mask, active = _setup([[1, 1, 1]])   # raw logits favour INVALID for agents 0,1
    mass, rate = invalid_mass_and_greedy_rate(logits, mask, active)
    assert 0.0 <= mass <= 1.0 and rate > 0.0     # agents 0,1 greedily invalid -> rate>0
    # empty-mask rate on the PRE-fallback predicted avail
    pre = torch.zeros(1, A, C); pre[0, 0, 1] = 1.0   # only agent0 has a predicted-valid action
    er = empty_mask_rate(pre, torch.ones(1, A))
    assert er == pytest.approx(2.0 / 3.0, abs=1e-4)  # agents 1,2 empty out of 3 active


def test_unmasked_invalid_rate_counts_masking_interventions():
    logits, mask, active = _setup([[1, 1, 1]])   # raw logits favour INVALID for agents 0,1
    logits = logits.clone().reshape(1, A, C)
    logits[0, 2, 1] = 5.0                         # agent2 greedily picks its VALID action (1)
    logits = logits.reshape(1, A * C)
    d = MaskedMultiOneHotDist(logits, mask, active, shape=(C,) * A)
    rate = d.unmasked_invalid_rate(logits)
    # agents 0 and 1 would greedily pick invalid; agent2 valid -> 2/3.
    assert float(rate) == pytest.approx(2.0 / 3.0, abs=1e-4)
