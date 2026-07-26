import torch


def candidate_per_probabilities(priorities, alpha):
    priorities = torch.as_tensor(priorities, dtype=torch.float64)
    scaled = priorities.pow(alpha)
    return scaled / scaled.sum()


def test_alpha_zero_is_uniform():
    p = candidate_per_probabilities([1.0, 10.0, 100.0], 0.0)
    assert torch.allclose(p, torch.full_like(p, 1.0 / 3.0))


def test_higher_priority_gets_higher_probability():
    p = candidate_per_probabilities([1.0, 2.0, 8.0], 0.6)
    assert p[2] > p[1] > p[0]


def test_importance_weights_are_finite_and_normalised():
    p = candidate_per_probabilities([1.0, 2.0, 8.0], 0.6)
    beta = 0.4
    w = (len(p) * p).pow(-beta)
    w /= w.max()
    assert torch.isfinite(w).all()
    assert torch.all(w > 0)
    assert float(w.max()) <= 1.0 + 1e-12
