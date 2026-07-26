import types

import pytest

from smacdreamer.cuda_preflight import resolve_amp_dtype, run_cuda_preflight


class _FakeCuda:
    def __init__(self, *, available=True, bf16=True, capability=(8, 6), archs=None, op_error=None):
        self._available = available
        self._bf16 = bf16
        self._cap = capability
        self._archs = archs or [f"sm_{capability[0]}{capability[1]}"]
        self._op_error = op_error

    def is_available(self):
        return self._available

    def is_bf16_supported(self):
        return self._bf16

    def get_device_name(self, idx):
        return "Fake GPU"

    def get_device_capability(self, idx):
        return self._cap

    def get_arch_list(self):
        return self._archs


class _FakeTensor:
    def __add__(self, other):
        return self

    def item(self):
        return 2


class _FakeTorch:
    __version__ = "fake"
    version = types.SimpleNamespace(cuda="12.1")

    def __init__(self, cuda):
        self.cuda = cuda

    def ones(self, shape, device):
        if self.cuda._op_error:
            raise self.cuda._op_error
        return _FakeTensor()


def test_amp_resolver_exact_values():
    torch = _FakeTorch(_FakeCuda(bf16=True))
    assert resolve_amp_dtype("float32", "cuda:0", torch) == "float32"
    assert resolve_amp_dtype("float16", "cuda:0", torch) == "float16"
    assert resolve_amp_dtype("bfloat16", "cuda:0", torch) == "bfloat16"


def test_bf16_unsupported_fails_clearly():
    torch = _FakeTorch(_FakeCuda(bf16=False))
    with pytest.raises(RuntimeError, match="does not report BF16 support"):
        resolve_amp_dtype("bfloat16", "cuda:0", torch)


def test_unknown_amp_raises():
    with pytest.raises(ValueError):
        resolve_amp_dtype("auto", "cuda:0", _FakeTorch(_FakeCuda()))


def test_cuda_arch_mismatch_fails():
    torch = _FakeTorch(_FakeCuda(capability=(6, 0), archs=["sm_70", "sm_80"]))
    with pytest.raises(RuntimeError, match="does not advertise kernels"):
        run_cuda_preflight("cuda:0", "float32", torch)


def test_cuda_tensor_smoke_failure_is_clear():
    torch = _FakeTorch(_FakeCuda(op_error=RuntimeError("no kernel image")))
    with pytest.raises(RuntimeError, match="CUDA tensor smoke test failed"):
        run_cuda_preflight("cuda:0", "float32", torch)
