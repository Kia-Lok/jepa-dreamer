"""Mask-aware factorised multi-one-hot action distribution (P0.1 / P0.2 core).

The centralised controller emits one categorical action per agent slot (A groups of C actions,
flattened to length A*C). This module enforces, in ONE place, the masking the policy needs:

  * invalid action logits are set to a sufficiently negative value BEFORE the categorical, so an
    invalid action can never be sampled and receives ~zero probability;
  * unimix exploration is spread over VALID actions only;
  * padded / dead agent slots are forced to a deterministic NOOP one-hot and excluded from the
    sample, the log-probability and the entropy (which are normalised over real living agents,
    not summed over every padded slot).

Pure PyTorch — no R2-Dreamer / SMAClite imports — so it is unit-testable on CPU. The real action
mask (from ``avail_actions`` in the obs) is used at environment-interaction time (P0.1); a
PREDICTED hard mask from latent heads is used during imagination (P0.2). Both call the same code.

All ops support ARBITRARY leading batch dims: ``logits`` is ``(..., A*C)``, masks are
``(..., A*C)`` or ``(..., A, C)``, ``active`` is ``(..., A)``. This covers both real acting
``(B, A*C)`` and imagination ``(B*T, T_imag, A*C)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

NOOP_INDEX = 0
NEG_LOGIT = -1e9   # "sufficiently negative" logit for invalid actions


def build_action_mask(avail_actions, agent_active, num_agents, num_actions,
                      noop_index: int = NOOP_INDEX):
    """Reshape a flat availability vector into a per-agent mask + cleaned active flags.

    ``avail_actions`` (..., A*C), ``agent_active`` (..., A). Returns ``(mask, active)`` with
    ``mask`` (..., A, C) bool and ``active`` (..., A) bool, where each ACTIVE agent is guaranteed
    >= 1 valid action (NOOP enabled if none) and each INACTIVE (padded/dead) agent's row is
    NOOP-only (so the support is never empty).
    """
    lead = avail_actions.shape[:-1]
    mask = (avail_actions.reshape(*lead, num_agents, num_actions) > 0.5).clone()
    active = (agent_active.reshape(*lead, num_agents) > 0.5)

    no_valid = active & (~mask.any(dim=-1))                 # (..., A)
    if no_valid.any():
        mask[..., noop_index] = mask[..., noop_index] | no_valid
    inactive = ~active                                      # (..., A)
    if inactive.any():
        noop_only = torch.zeros_like(mask)
        noop_only[..., noop_index] = True
        mask = torch.where(inactive.unsqueeze(-1), noop_only, mask)
    return mask, active


def hard_mask_from_logits(avail_logits, threshold_logit: float = 0.0):
    """Predicted hard availability mask from per-action availability LOGITS (P0.2).

    Returns a float {0,1} mask of the same shape. ``threshold_logit`` is the logit cut, e.g.
    ``logit(0.7) ≈ 0.847`` for a 0.7 probability threshold.
    """
    return (avail_logits >= threshold_logit).float()


def mask_quality_metrics(pred_avail_logits, true_avail, threshold_logit: float = 0.0,
                         eps: float = 1e-8) -> dict:
    """Precision / recall / false-positive-rate of a predicted availability mask vs the truth.

    Positive class = "available". Returns python floats (for logging)."""
    pred = (pred_avail_logits >= threshold_logit).float()
    true = (true_avail > 0.5).float()
    tp = (pred * true).sum()
    fp = (pred * (1.0 - true)).sum()
    fn = ((1.0 - pred) * true).sum()
    tn = ((1.0 - pred) * (1.0 - true)).sum()
    return {
        "precision": float((tp / (tp + fp + eps)).item()),
        "recall":    float((tp / (tp + fn + eps)).item()),
        "fpr":       float((fp / (fp + tn + eps)).item()),
    }


def invalid_mass_and_greedy_rate(raw_logits, mask, active):
    """Diagnostics over ACTIVE agents for UNMASKED actor logits vs a mask.

    Returns ``(pre_mask_invalid_mass, pre_mask_invalid_sample_rate)``:
      * mass  = mean over active agents of the softmax probability the UNMASKED policy puts on
                invalid actions;
      * rate  = fraction of active agents whose UNMASKED greedy (argmax) action is invalid.
    ``raw_logits`` (..., A*C); ``mask`` (..., A, C) bool; ``active`` (..., A) bool.
    """
    C = mask.shape[-1]
    r = raw_logits.reshape(*mask.shape[:-1], C)
    probs = F.softmax(r, dim=-1)
    act = (active > 0.5)
    n = act.float().sum().clamp(min=1.0)
    mass = (((probs * (~mask).float()).sum(dim=-1)) * act.float()).sum() / n
    greedy = r.argmax(dim=-1)
    chosen_valid = mask.gather(-1, greedy.unsqueeze(-1)).squeeze(-1)
    rate = (act & (~chosen_valid)).float().sum() / n
    return float(mass.item()), float(rate.item())


def empty_mask_rate(pre_fallback_avail, active):
    """Fraction of ACTIVE agents whose PREDICTED availability (before the NOOP fallback) is
    all-zero — a mask-collapse detector. ``pre_fallback_avail`` (..., A, C), ``active`` (..., A)."""
    has_valid = (pre_fallback_avail > 0.5).any(dim=-1)
    act = (active > 0.5)
    n = act.float().sum().clamp(min=1.0)
    return float(((act & (~has_valid)).float().sum() / n).item())


class MaskedMultiOneHotDist:
    """Factorised multi-one-hot distribution with per-agent action masking.

    ``logits`` (..., A*C) are the RAW actor logits (pre-distribution). ``mask`` (..., A, C) or
    (..., A*C) and ``active`` (..., A) come from :func:`build_action_mask`. ``shape`` is
    ``(C,)*A``.
    """

    def __init__(self, logits, mask, active, shape, unimix_ratio: float = 0.0,
                 noop_index: int = NOOP_INDEX):
        self.shape = tuple(int(s) for s in shape)
        self.A = len(self.shape)
        self.C = int(self.shape[0])
        self.noop_index = int(noop_index)
        self._lead = tuple(logits.shape[:-1])

        raw = logits.reshape(*self._lead, self.A, self.C).float()
        self.mask = (mask.reshape(*self._lead, self.A, self.C) > 0.5)
        self.active = (active.reshape(*self._lead, self.A) > 0.5)

        self._masked_logits = torch.where(self.mask, raw, torch.full_like(raw, NEG_LOGIT))
        probs = F.softmax(self._masked_logits, dim=-1)
        if unimix_ratio > 0.0:
            valid = self.mask.float()
            uniform = valid / valid.sum(dim=-1, keepdim=True).clamp(min=1.0)  # uniform over VALID
            probs = probs * (1.0 - unimix_ratio) + uniform * unimix_ratio
        self.probs = probs
        self.log_probs = torch.log(probs.clamp(min=1e-12))

    # ------------------------------------------------------------------
    def _force_noop_inactive(self, oh):
        """Set inactive (padded/dead) agents to a deterministic NOOP one-hot. ``oh`` (..., A, C)."""
        inactive = ~self.active
        if inactive.any():
            noop = torch.zeros_like(oh)
            noop[..., self.noop_index] = 1.0
            oh = torch.where(inactive.unsqueeze(-1), noop, oh)
        return oh

    def _flat(self, oh):
        return oh.reshape(*self._lead, self.A * self.C)

    @property
    def mode(self):
        idx = torch.argmax(self.probs, dim=-1)                 # (..., A) — always a valid action
        oh = F.one_hot(idx, self.C).float()
        return self._flat(self._force_noop_inactive(oh))

    def rsample(self, temperature: float = 1.0):
        # Gumbel-softmax (hard) over the MASKED logits -> sampled action is always valid.
        g = F.gumbel_softmax(self._masked_logits, tau=temperature, hard=True, dim=-1)
        return self._flat(self._force_noop_inactive(g))

    def sample(self, **kwargs):  # parity with the unmasked dist API
        return self.rsample()

    def log_prob(self, value):
        """Log-prob summed over ACTIVE agents and normalised by their count (excludes padding)."""
        v = value.reshape(*self._lead, self.A, self.C)
        lp = (self.log_probs * v).sum(dim=-1)                  # (..., A)
        active = self.active.float()
        n = active.sum(dim=-1).clamp(min=1.0)
        return (lp * active).sum(dim=-1) / n                   # (...,)

    def entropy(self):
        """Mean entropy over ACTIVE agents (excludes padded/dead slots)."""
        ent = -(self.probs * self.log_probs).sum(dim=-1)       # (..., A)
        active = self.active.float()
        n = active.sum(dim=-1).clamp(min=1.0)
        return (ent * active).sum(dim=-1) / n                  # (...,)

    def unmasked_invalid_rate(self, raw_logits) -> torch.Tensor:
        """Fraction of ACTIVE agents whose UNMASKED greedy action would be invalid.

        Measures how often masking intervenes (the "imagined invalid-action rate" the policy
        would otherwise incur). raw_logits (..., A*C)."""
        r = raw_logits.reshape(*self._lead, self.A, self.C)
        greedy = torch.argmax(r, dim=-1)                       # (..., A)
        chosen_valid = torch.gather(self.mask, -1, greedy.unsqueeze(-1)).squeeze(-1)  # (..., A) bool
        active = self.active
        invalid = active & (~chosen_valid)
        denom = active.float().sum().clamp(min=1.0)
        return invalid.float().sum() / denom

    def post_mask_invalid_sample_rate(self):
        """Fraction of ACTIVE agents whose MASKED sample is invalid — the invariant must be 0."""
        s = self.rsample().reshape(*self._lead, self.A, self.C)
        chosen = s.argmax(dim=-1)
        valid = self.mask.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        denom = self.active.float().sum().clamp(min=1.0)
        return (self.active & (~valid)).float().sum() / denom


__all__ = [
    "MaskedMultiOneHotDist", "build_action_mask", "hard_mask_from_logits",
    "mask_quality_metrics", "invalid_mass_and_greedy_rate", "empty_mask_rate",
    "NOOP_INDEX", "NEG_LOGIT",
]
