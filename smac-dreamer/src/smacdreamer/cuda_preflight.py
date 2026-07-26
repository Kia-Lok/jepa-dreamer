"""CUDA and AMP preflight checks for long R2-Dreamer runs."""

from __future__ import annotations


def resolve_amp_dtype(requested: str, device: str, torch_module=None) -> str:
    torch = torch_module
    if torch is None:
        import torch as torch  # type: ignore

    requested = str(requested).lower()
    if requested in ("float32", "fp32"):
        return "float32"
    if requested in ("float16", "fp16"):
        return "float16"
    if requested in ("bfloat16", "bf16"):
        if str(device).startswith("cuda") and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "amp_dtype is bfloat16 but this CUDA device does not report BF16 support. "
                "Use a BF16-capable GPU such as A40/A100/L4, or choose amp_dtype: float32 "
                "for P100/T4. This path will not silently switch BF16 to FP16."
            )
        return "bfloat16"
    raise ValueError(f"unsupported amp_dtype {requested!r}; expected float32, float16, or bfloat16")


def _arch_supported(capability: tuple[int, int], arch_list: list[str]) -> bool:
    if not arch_list:
        return True
    sm = f"sm_{capability[0]}{capability[1]}"
    compute = f"compute_{capability[0]}{capability[1]}"
    return any(sm in arch or compute in arch for arch in arch_list)


def run_cuda_preflight(device: str, amp_dtype: str, torch_module=None) -> dict:
    """Print and validate CUDA compatibility before expensive dataset discovery."""
    torch = torch_module
    if torch is None:
        import torch as torch  # type: ignore

    info = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_build": getattr(getattr(torch, "version", None), "cuda", None),
        "device": str(device),
        "amp_dtype": str(amp_dtype),
    }
    print(f"[cuda] torch={info['torch_version']} cuda_build={info['cuda_build']} device={device}")
    if not str(device).startswith("cuda"):
        print("[cuda] CPU device selected; CUDA autocast/preflight tensor test skipped.")
        return info
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but torch.cuda.is_available() is false.")

    idx = 0
    if ":" in str(device):
        idx = int(str(device).split(":", 1)[1])
    name = torch.cuda.get_device_name(idx)
    capability = tuple(torch.cuda.get_device_capability(idx))
    arch_list = list(torch.cuda.get_arch_list()) if hasattr(torch.cuda, "get_arch_list") else []
    info.update({"gpu_name": name, "gpu_capability": capability, "arch_list": arch_list})
    print(f"[cuda] gpu={name} capability=sm_{capability[0]}{capability[1]} arch_list={arch_list}")
    if not _arch_supported(capability, arch_list):
        raise RuntimeError(
            f"Installed PyTorch CUDA wheel does not advertise kernels for this GPU "
            f"(gpu capability sm_{capability[0]}{capability[1]}, arch_list={arch_list}). "
            "Install a PyTorch build compatible with this GPU before training."
        )
    try:
        x = torch.ones((1,), device=device)
        y = (x + 1).item()
    except Exception as exc:
        raise RuntimeError(
            f"CUDA tensor smoke test failed on {device}. This usually means the PyTorch "
            f"wheel is incompatible with the GPU/driver. Original error: {exc}"
        ) from exc
    if y != 2:
        raise RuntimeError(f"CUDA tensor smoke test returned unexpected value {y!r}")
    print("[cuda] tiny tensor operation succeeded.")
    return info


__all__ = ["resolve_amp_dtype", "run_cuda_preflight"]
