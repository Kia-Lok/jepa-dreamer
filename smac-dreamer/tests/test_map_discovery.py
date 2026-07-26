"""Tests for map discovery split/padding/safety-net logic.

The folder-scan + env-probe path (scan_folder/validate_map/discover) requires the SMAClite
simulator and is covered by the conda-env verification. These tests exercise the pure-Python
split / train-max padding / safety-net logic on synthetic 'included' result dicts.
"""

import pytest

from smacdreamer.envs.map_discovery import (
    SplitSpec, split_maps, compute_train_max_padding, safety_net_check, _raise_scan_failures,
)
from smacdreamer.envs.padding import PaddingDims


def _mk(name, a=4, e=3, act=10, obs=60, family="f"):
    return {
        "rel_path": f"maps/{name}.json", "file_hash": name, "map_id": 0,
        "map_info": {"name": name, "family": family,
                     "n_agents": a, "n_enemies": e, "n_actions": act, "obs_size": obs},
    }


def test_ratio_split_disjoint_and_nonempty_test():
    inc = [_mk(f"m{i}") for i in range(10)]
    tr, te = split_maps(inc, SplitSpec(mode="ratio", train_ratio=0.8, seed=0))
    tr_names = {r["map_info"]["name"] for r in tr}
    te_names = {r["map_info"]["name"] for r in te}
    assert tr and te                       # both non-empty
    assert not (tr_names & te_names)        # disjoint
    assert len(tr) + len(te) == 10


def test_ratio_split_reproducible_for_seed():
    inc = [_mk(f"m{i}") for i in range(10)]
    a, _ = split_maps(inc, SplitSpec("ratio", 0.8, seed=7))
    b, _ = split_maps(inc, SplitSpec("ratio", 0.8, seed=7))
    assert [r["map_info"]["name"] for r in a] == [r["map_info"]["name"] for r in b]


def test_explicit_split():
    inc = [_mk(f"m{i}") for i in range(5)]
    tr, te = split_maps(inc, SplitSpec(mode="explicit",
                                       train_names=["m0", "m1"], test_names=["m2", "m3"]))
    assert [r["map_info"]["name"] for r in tr] == ["m0", "m1"]
    assert [r["map_info"]["name"] for r in te] == ["m2", "m3"]


def test_explicit_split_overlap_raises():
    inc = [_mk(f"m{i}") for i in range(3)]
    with pytest.raises(ValueError):
        split_maps(inc, SplitSpec(mode="explicit", train_names=["m0"], test_names=["m0"]))


def test_train_max_padding_uses_train_only():
    # A big TEST map must NOT influence padding (computed from train only).
    train = [_mk("t0", a=4, e=3, act=10, obs=60), _mk("t1", a=5, e=3, act=11, obs=70)]
    pad = compute_train_max_padding(train)
    assert (pad.max_agents, pad.max_actions, pad.max_obs_size) == (5, 11, 70)


def test_padding_override_wins():
    train = [_mk("t0", a=4, e=3, act=10, obs=60)]
    pad = compute_train_max_padding(train, override={
        "max_agents": 8, "max_enemies": 9, "max_actions": 15, "max_obs_size": 136})
    assert isinstance(pad, PaddingDims)
    assert (pad.max_agents, pad.max_obs_size) == (8, 136)


def test_safety_net_passes_when_all_fit():
    inc = [_mk(f"m{i}", obs=60) for i in range(5)]
    pad = PaddingDims(8, 9, 15, 60)
    safety_net_check(inc, pad)  # must not raise


def test_safety_net_fails_and_names_oversize_test_map():
    inc = [_mk(f"m{i}", obs=60) for i in range(5)] + [_mk("huge", a=99, obs=60)]
    pad = compute_train_max_padding([_mk("m0", obs=60)])  # max_agents=4
    with pytest.raises(ValueError) as ei:
        safety_net_check(inc, pad)
    assert "huge" in str(ei.value)


def test_discovery_failure_report_consolidates_skipped_maps():
    with pytest.raises(ValueError) as ei:
        _raise_scan_failures(
            "TRAIN",
            excluded=[{"path": "dup.json", "reason": "duplicate"}],
            invalid=[{"path": "bad.json", "reason": "env load/step failed: boom"}],
        )
    msg = str(ei.value)
    assert "bad.json" in msg
    assert "dup.json" in msg


def test_r2_650_expected_split_file_counts():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    base = root / "configs" / "maps" / "r2_650" / "configs"
    assert len(list((base / "train").glob("*.json"))) == 400
    assert len(list((base / "validation").glob("*.json"))) == 50
