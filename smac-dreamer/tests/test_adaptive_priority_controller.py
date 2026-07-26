import torch

from smacdreamer.adaptive_priority import AdaptivePriorityController


class Entry:
    def __init__(self, map_id, name):
        self.map_id = map_id
        self.name = name


def make_controller():
    cfg = {
        "map": {
            "enabled": True,
            "error_ema_decay": 0.0,
            "uniform_floor": 0.1,
            "staleness_mix": 0.0,
            "update_every_feedbacks": 1,
            "minimum_feedback": 1,
        }
    }
    return AdaptivePriorityController.from_entries(
        [Entry(1, "one"), Entry(2, "two"), Entry(3, "three")], cfg
    )


def test_probabilities_normalise_and_follow_error():
    c = make_controller()
    c.record_critic_feedback(
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[0.1, 5.0, 0.2]]),
        torch.ones(1, 3),
        env_step=100,
    )
    p = c.shared_probabilities
    assert torch.isfinite(p).all()
    assert torch.allclose(p.sum(), torch.tensor(1.0, dtype=p.dtype))
    assert p[1] > p[0]
    assert p[1] > p[2]
    assert torch.all(p >= 0.1 / 3.0 - 1e-12)


def test_map_ids_are_aggregated_per_timestep():
    c = make_controller()
    c.record_critic_feedback(
        torch.tensor([[1, 1, 2, 3]]),
        torch.tensor([[1.0, 3.0, 7.0, 0.5]]),
        torch.ones(1, 4),
        env_step=10,
    )
    assert c.feedback_count.tolist() == [2, 1, 1]
    assert abs(float(c.error_ema[0]) - 2.0) < 1e-9


def test_state_round_trip_rejects_wrong_map_order():
    c = make_controller()
    state = c.state_dict()
    clone = make_controller()
    clone.load_state_dict(state)
    assert torch.allclose(c.shared_probabilities, clone.shared_probabilities)

    wrong = AdaptivePriorityController(
        [3, 2, 1],
        {"map": {"enabled": True}},
    )
    try:
        wrong.load_state_dict(state)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong map order must be rejected")


def test_initial_recompute_is_uniform_when_all_maps_are_tied():
    c = make_controller()
    p = c.recompute_probabilities()
    assert torch.allclose(p, torch.full_like(p, 1.0 / 3.0))
