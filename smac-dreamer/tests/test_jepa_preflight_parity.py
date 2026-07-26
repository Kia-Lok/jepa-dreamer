import argparse
import json
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))

import preflight_jepa_training as preflight
from smacdreamer.jepa.checkpoint import JEPACompatibilityError, _arch_from, _checkpoint_contract, validate_metadata
from validate_jepa_r2_integration import previous_actions_for_states
from validate_jepa_token_parity import _first_mismatch, pad_episode_action


def _episode_npz(path, *, n_agents=2, n_enemies=1, n_actions=3):
    states = np.zeros((1, 3, n_agents * 4 + n_enemies * 4), dtype=np.float32)
    actions = np.zeros((1, 2, n_agents), dtype=np.int64)
    entity_static = np.zeros((n_agents + n_enemies, 2), dtype=np.float32)
    np.savez(
        path,
        states=states,
        actions=actions,
        valid=np.ones((1, 2), dtype=bool),
        static_condition=np.zeros((4,), dtype=np.float32),
        entity_static=entity_static,
        n_agents=n_agents,
        n_enemies=n_enemies,
        n_actions=n_actions,
        ally_state_feat_size=4,
        enemy_state_feat_size=4,
        ally_has_shields=0,
        enemy_has_shields=0,
        num_unit_types=0,
        static_dim=4,
        entity_static_feat_size=2,
        state_dim=states.shape[-1],
    )


def _config(path, *, imag_horizon=5, max_agents=2, max_enemies=1, max_actions=3):
    path.write_text(
        f"""
observation:
  mode: structured
imag_horizon: {imag_horizon}
padding:
  max_agents: {max_agents}
  max_enemies: {max_enemies}
  max_actions: {max_actions}
  max_obs_size: 8
""",
        encoding="utf-8",
    )


def test_previous_action_shift_for_three_states_two_actions():
    actions = torch.zeros(2, 2, 3)
    actions[0, :, 1] = 1
    actions[1, :, 2] = 1
    prev = previous_actions_for_states(actions)
    assert prev.shape == (3, 6)
    assert prev[0].sum() == 0
    torch.testing.assert_close(prev[1], actions[0].reshape(-1))
    torch.testing.assert_close(prev[2], actions[1].reshape(-1))


def test_first_mismatch_detects_deliberate_off_by_one_or_memory_corruption():
    actual = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    expected = torch.tensor([[0.0, 1.0], [9.0, 3.0]])
    with pytest.raises(AssertionError, match=r"index \(1, 0\).*max_error=7"):
        _first_mismatch("corrupted_recurrent_memory", actual, expected)


def test_local_action_width_pads_to_checkpoint_global_width():
    raw = np.asarray([[0, 1, 0], [1, 0, 0]], dtype=np.float32)
    padded, mask = pad_episode_action(raw, n_agents=2, n_actions=3, max_agents=3, max_actions=5)
    assert padded.shape == (3, 5)
    np.testing.assert_allclose(padded[:2, :3], raw)
    assert padded[:, 3:].sum() == 0
    assert padded[2].sum() == 0
    np.testing.assert_allclose(mask, [1, 1, 0])
    corrupted = padded.copy()
    corrupted[0, 4] = 1
    with pytest.raises(AssertionError, match="corrupted_action"):
        _first_mismatch("corrupted_action", torch.from_numpy(corrupted), torch.from_numpy(padded))


def test_runtime_metadata_padding_mismatch_fails(tmp_path):
    ep = tmp_path / "episode.npz"
    cfg_path = tmp_path / "cfg.yaml"
    _episode_npz(ep)
    _config(cfg_path, max_agents=4, max_enemies=1, max_actions=3)
    ckpt_cfg = {"latent_dim": 6, "rollout_memory_dim": 7, "action_conditioned_memory": False}
    vis = type("V", (), {"metadata": lambda self: {
        "enemy_visibility_mask": False,
        "enemy_sight_range": 9.0,
        "visibility_xy_indices": (2, 3),
    }})()
    runtime, _ = preflight.derive_runtime_metadata(cfg_path, ep, ckpt_cfg, vis)
    checkpoint_meta = dict(runtime)
    checkpoint_meta["max_agents"] = 3
    checkpoint_meta.update({"state_dim": 1, "hidden_dim": 8})
    arch = {"latent_dim": 6}
    contract = _checkpoint_contract(checkpoint_meta, ckpt_cfg, arch)
    with pytest.raises(JEPACompatibilityError, match="max_agents"):
        validate_metadata(contract, runtime)


def test_preflight_horizon_mismatch_requires_explicit_override(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _config(cfg_path, imag_horizon=15)
    config = preflight.OmegaConf.load(cfg_path)
    with pytest.raises(ValueError, match="differs from config"):
        preflight._resolve_horizon(config, {"rollout_horizon": 20}, 10, False)
    assert preflight._resolve_horizon(config, {"rollout_horizon": 20}, 10, True) == (15, 20, 10)
    with pytest.raises(ValueError, match="exceeds checkpoint"):
        preflight._resolve_horizon(config, {"rollout_horizon": 10}, None, False)


def test_preflight_json_report_success_and_failure(monkeypatch, tmp_path, capsys):
    report = tmp_path / "report.json"
    args = [
        "preflight",
        "--checkpoint", "ckpt.pt",
        "--episode-npz", "episode.npz",
        "--config", "cfg.yaml",
        "--report-json", str(report),
    ]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(preflight, "run_preflight", lambda parsed: {"result": "pass", "comparison_errors": {}})
    preflight.main()
    assert json.loads(report.read_text())["result"] == "pass"
    assert "JEPA R2-DREAMER PREFLIGHT: PASS" in capsys.readouterr().out

    report2 = tmp_path / "report_fail.json"
    args[-1] = str(report2)
    monkeypatch.setattr(sys, "argv", args)
    def fail(_):
        raise RuntimeError("boom")
    monkeypatch.setattr(preflight, "run_preflight", fail)
    with pytest.raises(SystemExit):
        preflight.main()
    assert json.loads(report2.read_text())["result"] == "fail"
    assert "JEPA R2-DREAMER PREFLIGHT: FAIL" in capsys.readouterr().err
