import pathlib
import sys
import types

import pytest
import torch


pytest.importorskip("torchrl")
pytest.importorskip("tensordict")

ROOT = pathlib.Path(__file__).resolve().parent.parent
R2 = ROOT / "external" / "r2dreamer"
if str(R2) not in sys.path:
    sys.path.insert(0, str(R2))

from buffer import Buffer
from tensordict import TensorDict


def _cfg(tmp_path, backend):
    return types.SimpleNamespace(
        device="cpu",
        storage_device="cpu",
        batch_size=1,
        batch_length=2,
        max_size=20,
        storage_backend=backend,
        scratch_dir=str(tmp_path / "replay"),
    )


def _transition(i):
    return TensorDict({
        "state": torch.full((1, 2), float(i)),
        "reward": torch.ones(1, 1),
        "action": torch.zeros(1, 1),
        "is_first": torch.zeros(1, 1, dtype=torch.bool),
        "is_last": torch.zeros(1, 1, dtype=torch.bool),
        "is_terminal": torch.zeros(1, 1, dtype=torch.bool),
        "episode": torch.zeros(1, dtype=torch.int32),
        "stoch": torch.zeros(1, 2, 3),
        "deter": torch.zeros(1, 4),
    }, batch_size=(1,))


def _fill(buf, n=6):
    for i in range(n):
        buf.add_transition(_transition(i))


@pytest.mark.parametrize("backend", ["tensor", "memmap"])
def test_replay_backend_sample_and_update(tmp_path, backend):
    buf = Buffer(_cfg(tmp_path, backend))
    try:
        _fill(buf)
        assert buf.count() >= 6
        data, index, initial = buf.sample()
        assert data["state"].shape[:2] == (1, 2)
        assert initial[0].shape[-2:] == (2, 3)
        new_stoch = torch.ones_like(data["stoch"]) * 5
        new_deter = torch.ones_like(data["deter"]) * 7
        buf.update(index, new_stoch, new_deter)
        if backend == "memmap":
            assert pathlib.Path(buf.scratch_dir).exists()
            assert any(pathlib.Path(buf.scratch_dir).iterdir())
    finally:
        buf.close()


def test_invalid_backend_fails_clearly(tmp_path):
    cfg = _cfg(tmp_path, "bogus")
    with pytest.raises(ValueError, match="storage_backend"):
        Buffer(cfg)
