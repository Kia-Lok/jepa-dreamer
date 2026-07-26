"""Tests for the factorised multi-agent action codec.

Pure NumPy — no JAX / Elements / Embodied / DreamerV3 / smaclite required.
"""

import numpy as np
import pytest

from smacdreamer.envs.action_codec import FactorisedActionCodec, NOOP_ACTION


# ----------------------------------------------------------------------
# Shapes / space dimensions
# ----------------------------------------------------------------------

def test_flat_dim_and_groups():
    codec = FactorisedActionCodec(num_agents=5, num_actions=11)
    assert codec.flat_dim == 55
    assert codec.group_sizes == [11, 11, 11, 11, 11]
    assert list(codec.nvec) == [11, 11, 11, 11, 11]


def test_action_space_dimensions():
    pytest.importorskip("gymnasium")
    codec = FactorisedActionCodec(num_agents=3, num_actions=7)
    space = codec.action_space()
    assert list(space.nvec) == [7, 7, 7]
    assert space.shape == (3,)


def test_rejects_nonpositive_dims():
    with pytest.raises(ValueError):
        FactorisedActionCodec(num_agents=0, num_actions=5)
    with pytest.raises(ValueError):
        FactorisedActionCodec(num_agents=3, num_actions=0)


# ----------------------------------------------------------------------
# Encode: integer -> one-hot
# ----------------------------------------------------------------------

def test_encode_basic_one_hot():
    codec = FactorisedActionCodec(num_agents=3, num_actions=4)
    flat = codec.encode([0, 2, 3])
    assert flat.shape == (12,)
    assert flat.dtype == np.float32
    expected = np.zeros(12, dtype=np.float32)
    expected[0 * 4 + 0] = 1.0
    expected[1 * 4 + 2] = 1.0
    expected[2 * 4 + 3] = 1.0
    np.testing.assert_array_equal(flat, expected)
    # Exactly one 1 per group.
    assert flat.reshape(3, 4).sum(axis=-1).tolist() == [1.0, 1.0, 1.0]


def test_encode_rejects_out_of_range():
    codec = FactorisedActionCodec(num_agents=2, num_actions=4)
    with pytest.raises(ValueError):
        codec.encode([4, 0])  # 4 >= num_actions
    with pytest.raises(ValueError):
        codec.encode([-1, 0])


def test_encode_wrong_length():
    codec = FactorisedActionCodec(num_agents=3, num_actions=4)
    with pytest.raises(ValueError):
        codec.encode([0, 1])  # only 2, expected 3


# ----------------------------------------------------------------------
# Decode + round trip
# ----------------------------------------------------------------------

def test_decode_round_trip_multiple_agents():
    codec = FactorisedActionCodec(num_agents=6, num_actions=9)
    actions = [0, 8, 3, 1, 7, 2]
    flat = codec.encode(actions)
    assert codec.decode(flat) == actions


def test_decode_from_logits_argmax():
    # decode(validate=False) should accept non-one-hot logits and take argmax per group.
    codec = FactorisedActionCodec(num_agents=2, num_actions=3)
    logits = np.array([0.1, 0.9, 0.2,   # -> 1
                       5.0, 1.0, 0.0],  # -> 0
                      dtype=np.float32)
    assert codec.decode(logits, validate=False) == [1, 0]


def test_round_trip_all_actions_each_agent():
    codec = FactorisedActionCodec(num_agents=4, num_actions=5)
    for a in range(5):
        actions = [a] * 4
        assert codec.decode(codec.encode(actions)) == actions


# ----------------------------------------------------------------------
# Padded-agent handling
# ----------------------------------------------------------------------

def test_encode_pads_with_noop():
    codec = FactorisedActionCodec(num_agents=5, num_actions=4)  # A=5 slots
    # Only 3 real agents; slots 3 and 4 should be forced to noop.
    flat = codec.encode([1, 2, 3], num_real_agents=3)
    groups = flat.reshape(5, 4)
    assert groups[0].argmax() == 1
    assert groups[1].argmax() == 2
    assert groups[2].argmax() == 3
    assert groups[3].argmax() == NOOP_ACTION
    assert groups[4].argmax() == NOOP_ACTION
    # Padded groups are still valid one-hot (so the tensor is well-formed for the model).
    assert groups.sum(axis=-1).tolist() == [1, 1, 1, 1, 1]


def test_encode_full_length_with_real_count_overrides_padded():
    codec = FactorisedActionCodec(num_agents=4, num_actions=3)
    # Supply a full-length vector but mark only 2 real; padded slots forced to noop.
    flat = codec.encode([2, 1, 2, 1], num_real_agents=2)
    groups = flat.reshape(4, 3)
    assert [g.argmax() for g in groups] == [2, 1, NOOP_ACTION, NOOP_ACTION]


def test_decode_returns_only_real_agents():
    codec = FactorisedActionCodec(num_agents=5, num_actions=4)
    flat = codec.encode([1, 2, 3], num_real_agents=3)
    # Only the first 3 (real) agent actions are sent to SMAClite.
    assert codec.decode(flat, num_real_agents=3) == [1, 2, 3]


# ----------------------------------------------------------------------
# Validation: shape / dtype / one-hot integrity
# ----------------------------------------------------------------------

def test_validate_one_hot_accepts_valid():
    codec = FactorisedActionCodec(num_agents=2, num_actions=3)
    codec.validate_one_hot(codec.encode([0, 2]))  # should not raise


def test_validate_one_hot_rejects_non_binary():
    codec = FactorisedActionCodec(num_agents=2, num_actions=3)
    bad = np.array([0.5, 0.5, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError):
        codec.validate_one_hot(bad)


def test_validate_one_hot_rejects_multi_hot_group():
    codec = FactorisedActionCodec(num_agents=2, num_actions=3)
    bad = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # group 0 sums to 2
    with pytest.raises(ValueError):
        codec.validate_one_hot(bad)


def test_decode_rejects_invalid_flat_shape():
    codec = FactorisedActionCodec(num_agents=3, num_actions=4)  # flat_dim = 12
    with pytest.raises(ValueError):
        codec.decode(np.zeros(10, dtype=np.float32))


def test_decode_rejects_bad_dtype():
    codec = FactorisedActionCodec(num_agents=2, num_actions=3)
    with pytest.raises(TypeError):
        codec.decode(np.array([True, False, False, True, False, False]))


def test_one_hot_dtype_is_configurable():
    codec = FactorisedActionCodec(num_agents=2, num_actions=2, one_hot_dtype=np.float64)
    flat = codec.encode([0, 1])
    assert flat.dtype == np.float64
