import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smacdreamer.jepa.state import JEPAStateSpec, pack_state, unpack_state


def test_pack_unpack_roundtrip():
    spec = JEPAStateSpec(entities=3, latent_dim=5, memory_dim=7, static_dim=4)
    memory = torch.randn(2, 3, 7)
    entity = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
    slot = torch.ones(2, 3)
    static = torch.randn(2, 4)
    deter = pack_state(memory, entity, slot, static)
    assert deter.shape == (2, spec.deter_dim)
    got = unpack_state(deter, spec)
    torch.testing.assert_close(got[0], memory)
    torch.testing.assert_close(got[1], entity)
    torch.testing.assert_close(got[2], slot)
    torch.testing.assert_close(got[3], static)
