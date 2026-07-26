from __future__ import annotations

import torch

torch.set_num_threads(1)

from smac_jepa.pow2_direct_predictor import (
    PowerOfTwoDirectPredictor,
    canonical_pow2_horizons,
)


def make_model() -> PowerOfTwoDirectPredictor:
    torch.manual_seed(7)
    return PowerOfTwoDirectPredictor(
        latent_dim=16,
        n_actions=6,
        max_agents=3,
        max_entities=5,
        horizons=(1, 2, 4, 8, 16),
        hidden_dim=32,
        action_embed_dim=8,
        slot_embed_dim=4,
        dropout=0.0,
    )


def test_horizon_validation() -> None:
    assert canonical_pow2_horizons([16, 1, 4, 4, 2]) == (1, 2, 4, 16)
    try:
        canonical_pow2_horizons([1, 3])
    except ValueError:
        pass
    else:
        raise AssertionError("non-power-of-two horizon should fail")


def test_direct_output_shapes_and_finiteness() -> None:
    model = make_model()
    context = torch.randn(1, 5, 16)
    entity_mask = torch.ones(1, 5)
    actions = torch.randint(0, 6, (1, 16, 3))
    action_mask = torch.ones(1, 16, 3)
    out = model(context, actions, action_mask, entity_mask)
    assert set(out) == {1, 2, 4, 8, 16}
    for value in out.values():
        assert value.shape == (1, 5, 16)
        assert torch.isfinite(value).all()


def test_one_hot_and_integer_actions_match() -> None:
    model = make_model().eval()
    context = torch.randn(1, 5, 16)
    entity_mask = torch.ones(1, 5)
    ids = torch.randint(0, 6, (1, 16, 3))
    one_hot = torch.nn.functional.one_hot(ids, num_classes=6).float()
    mask = torch.ones(1, 16, 3)
    with torch.no_grad():
        a = model(context, ids, mask, entity_mask)[16]
        b = model(context, one_hot, mask, entity_mask)[16]
    torch.testing.assert_close(a, b)


def test_future_actions_do_not_leak_into_short_horizon() -> None:
    model = make_model().eval()
    context = torch.randn(1, 5, 16)
    entity_mask = torch.ones(1, 5)
    actions_a = torch.randint(0, 6, (1, 16, 3))
    actions_b = actions_a.clone()
    actions_b[:, 1:] = torch.randint(0, 6, actions_b[:, 1:].shape)
    mask = torch.ones(1, 16, 3)
    with torch.no_grad():
        h1_a = model(context, actions_a, mask, entity_mask, horizons=(1,))[1]
        h1_b = model(context, actions_b, mask, entity_mask, horizons=(1,))[1]
    torch.testing.assert_close(h1_a, h1_b)


def test_binary_decomposition_uses_logarithmic_number_of_blocks() -> None:
    model = make_model().eval()
    context = torch.randn(1, 5, 16)
    entity_mask = torch.ones(1, 5)
    actions = torch.randint(0, 6, (1, 16, 3))
    mask = torch.ones(1, 16, 3)
    with torch.no_grad():
        pred = model.predict_binary(
            context, actions, mask, entity_mask, horizon=13
        )
    assert pred.blocks == (8, 4, 1)
    assert pred.latent.shape == context.shape


def test_action_sequence_changes_predictions() -> None:
    model = make_model().eval()
    context = torch.randn(1, 5, 16)
    entity_mask = torch.ones(1, 5)
    zeros = torch.zeros(1, 16, 3, dtype=torch.long)
    ones = torch.ones(1, 16, 3, dtype=torch.long)
    mask = torch.ones(1, 16, 3)
    with torch.no_grad():
        pred_zero = model(context, zeros, mask, entity_mask)[16]
        pred_one = model(context, ones, mask, entity_mask)[16]
    assert (pred_zero - pred_one).abs().mean().item() > 1e-7


def test_shared_head_can_predict_arbitrary_horizon() -> None:
    model = make_model().eval()
    context = torch.randn(1, 5, 16)
    entity_mask = torch.ones(1, 5)
    actions = torch.randint(0, 6, (1, 16, 3))
    mask = torch.ones(1, 16, 3)
    with torch.no_grad():
        outputs = model(
            context,
            actions,
            mask,
            entity_mask,
            horizons=(9,),
            include_shared_predictions=True,
        )
    assert outputs[9].shape == context.shape
    assert outputs["shared_9"].shape == context.shape


def test_composition_can_reuse_largest_block_beyond_training_horizon() -> None:
    model = make_model().eval()
    context = torch.randn(1, 5, 16)
    entity_mask = torch.ones(1, 5)
    actions = torch.randint(0, 6, (1, 25, 3))
    mask = torch.ones(1, 25, 3)
    with torch.no_grad():
        prediction = model.predict_binary(
            context,
            actions,
            mask,
            entity_mask,
            horizon=25,
        )
    assert prediction.blocks == (16, 8, 1)
    assert prediction.latent.shape == context.shape
