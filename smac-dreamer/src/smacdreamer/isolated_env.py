"""Spawn-isolated Gym-style environment proxy used by held-out validation."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class EnvFactorySpec:
    module: str
    name: str

    def load(self) -> Callable:
        return getattr(importlib.import_module(self.module), self.name)


DEFAULT_SMACLITE_FACTORY = EnvFactorySpec(
    "smacdreamer.r2dreamer_factory", "make_smaclite_multimap_env"
)


class RemoteEnvError(RuntimeError):
    pass


def _child_loop(conn, factory_spec, factory_args, factory_kwargs, map_name):
    env = None
    try:
        factory = factory_spec.load() if isinstance(factory_spec, EnvFactorySpec) else factory_spec
        env = factory(*factory_args, **factory_kwargs)
        conn.send(("ready", {"pid": os.getpid(), "map_name": map_name}))
        while True:
            cmd, args, kwargs = conn.recv()
            if cmd == "reset":
                conn.send(("result", env.reset(*args, **kwargs)))
            elif cmd == "step":
                conn.send(("result", env.step(*args, **kwargs)))
            elif cmd == "close":
                try:
                    env.close()
                finally:
                    conn.send(("result", True))
                    return
            elif cmd == "getattr":
                conn.send(("result", getattr(env, args[0])))
            else:
                raise KeyError(f"unknown isolated env command {cmd!r}")
    except EOFError:
        pass
    except BaseException as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            conn.send(("error", {"map_name": map_name, "pid": os.getpid(), "traceback": tb}))
        except Exception:
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


class IsolatedEnvProxy:
    """Small reset/step RPC proxy around one spawned environment process."""

    def __init__(
        self,
        factory_spec,
        factory_args: tuple,
        factory_kwargs: dict | None = None,
        *,
        map_name: str,
        shutdown_timeout_seconds: float = 5.0,
    ):
        self.map_name = str(map_name)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self._ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = self._ctx.Pipe()
        self._process = self._ctx.Process(
            target=_child_loop,
            args=(child_conn, factory_spec, tuple(factory_args), dict(factory_kwargs or {}), self.map_name),
            daemon=False,
        )
        self._closed = False
        self._process.start()
        child_conn.close()
        kind, payload = self._recv()
        if kind != "ready":
            self.close()
            raise RemoteEnvError(f"validation env for map {self.map_name!r} failed to start: {payload}")
        self.child_pid = int(payload["pid"])

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()

    def _recv(self):
        try:
            kind, payload = self._parent_conn.recv()
        except EOFError as exc:
            raise RemoteEnvError(
                f"validation env for map {self.map_name!r} died unexpectedly "
                f"(pid={self.pid}, exitcode={self.exitcode})"
            ) from exc
        if kind == "error":
            raise RemoteEnvError(
                f"validation env error for map {payload.get('map_name', self.map_name)!r} "
                f"(pid={payload.get('pid')}):\n{payload.get('traceback')}"
            )
        return kind, payload

    def _call(self, cmd: str, *args, **kwargs):
        try:
            self._parent_conn.send((cmd, args, kwargs))
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RemoteEnvError(
                f"validation env for map {self.map_name!r} is unavailable "
                f"(pid={self.pid}, exitcode={self.exitcode})"
            ) from exc
        kind, payload = self._recv()
        if kind != "result":
            raise RemoteEnvError(f"unexpected validation env response {kind!r} for {self.map_name!r}")
        return payload

    def reset(self, *args, **kwargs):
        return self._call("reset", *args, **kwargs)

    def step(self, *args, **kwargs):
        return self._call("step", *args, **kwargs)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._process.is_alive():
                try:
                    self._parent_conn.send(("close", (), {}))
                    if self._parent_conn.poll(self.shutdown_timeout_seconds):
                        self._recv()
                except Exception:
                    pass
                self._process.join(self.shutdown_timeout_seconds)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(self.shutdown_timeout_seconds)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(1.0)
        finally:
            try:
                self._parent_conn.close()
            except Exception:
                pass
        if self._process.is_alive():
            raise RuntimeError(
                f"validation env child still alive after close: map={self.map_name!r} pid={self.pid}"
            )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._call("getattr", name)


def make_isolated_smaclite_env(
    entries,
    pad_dims,
    sampling_mode,
    base_seed,
    worker_idx,
    reward_name,
    reward_params,
    gamma,
    max_episode_steps,
    obs_mode="flat",
    *,
    include_jepa_obs: bool = False,
    jepa_visibility_config=None,
    shutdown_timeout_seconds: float = 5.0,
):
    map_name = getattr(entries[0], "name", "unknown")
    return IsolatedEnvProxy(
        DEFAULT_SMACLITE_FACTORY,
        (
            entries,
            pad_dims,
            sampling_mode,
            base_seed,
            worker_idx,
            reward_name,
            reward_params,
            gamma,
            max_episode_steps,
            obs_mode,
        ),
        {
            "include_jepa_obs": bool(include_jepa_obs),
            "jepa_visibility_config": jepa_visibility_config,
        },
        map_name=map_name,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
    )
