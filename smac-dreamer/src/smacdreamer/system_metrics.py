"""Best-effort process, cgroup, and CUDA telemetry for long SMAClite runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


GB = 1024.0 ** 3


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _rss_bytes_proc(pid: int) -> int | None:
    text = _read_text(Path("/proc") / str(pid) / "status")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    return None


def rss_bytes(pid: int | None = None) -> int | None:
    """Return RSS bytes for ``pid`` using psutil when available, else /proc."""
    pid = int(pid or os.getpid())
    try:
        import psutil  # type: ignore

        return int(psutil.Process(pid).memory_info().rss)
    except Exception:
        return _rss_bytes_proc(pid)


def _gb(value: int | float | None) -> float:
    if value is None:
        raise ValueError("cannot convert unavailable metric to GB")
    return float(value) / GB


def cgroup_metrics() -> dict[str, float]:
    """Read cgroup v2 memory counters when available."""
    base = Path("/sys/fs/cgroup")
    metrics: dict[str, float] = {}
    current = _read_text(base / "memory.current")
    if current and current.isdigit():
        metrics["system/cgroup_current_gb"] = _gb(int(current))
    maximum = _read_text(base / "memory.max")
    if maximum:
        metrics["system/cgroup_max_gb"] = 0.0 if maximum == "max" else _gb(int(maximum))
    events = _read_text(base / "memory.events")
    if events:
        for line in events.splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    metrics[f"system/cgroup_events_{parts[0]}"] = float(parts[1])
                except ValueError:
                    pass
    return metrics


def cuda_metrics() -> dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "system/cuda_allocated_gb": _gb(torch.cuda.memory_allocated()),
            "system/cuda_reserved_gb": _gb(torch.cuda.memory_reserved()),
        }
    except Exception:
        return {}


def collect_system_metrics(
    *,
    worker_infos: Iterable[dict] = (),
    replay_count: int | None = None,
    replay_backend: str | None = None,
    completed_episodes: int | None = None,
    worker_restarts: int | None = None,
) -> dict[str, float | str]:
    """Collect stable scalar metrics. Telemetry failures are intentionally ignored."""
    metrics: dict[str, float | str] = {"system/main_pid": float(os.getpid())}
    main_rss = rss_bytes()
    if main_rss is None:
        metrics["system/main_rss_available"] = 0.0
    else:
        metrics["system/main_rss_available"] = 1.0
        metrics["system/main_rss_gb"] = _gb(main_rss)
    total_worker = 0
    for info in worker_infos:
        slot = info.get("slot", 0)
        pid = info.get("pid")
        rss = rss_bytes(pid) if pid else None
        if rss is not None:
            total_worker += rss
            metrics[f"system/worker_{slot}_rss_gb"] = _gb(rss)
        if pid:
            metrics[f"system/worker_{slot}_pid"] = float(pid)
        metrics[f"system/worker_{slot}_generation"] = float(info.get("generation", 0))
    metrics["system/worker_rss_gb"] = _gb(total_worker)
    metrics.update(cgroup_metrics())
    metrics.update(cuda_metrics())
    if replay_count is not None:
        metrics["replay/count"] = float(replay_count)
    if replay_backend is not None:
        metrics["replay/storage_backend_code"] = 1.0 if replay_backend == "memmap" else 0.0
    if completed_episodes is not None:
        metrics["env/completed_episodes"] = float(completed_episodes)
    if worker_restarts is not None:
        metrics["env/worker_restarts"] = float(worker_restarts)
    return metrics


def log_system_metrics(logger, step: int, **kwargs) -> dict[str, float | str]:
    metrics = collect_system_metrics(**kwargs)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            try:
                logger.scalar(key, float(value))
            except Exception:
                pass
    return metrics
