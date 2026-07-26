import atexit
import contextlib
import enum
import inspect
import os
import sys
import time
import traceback
from functools import partial as bind

import numpy as np
import torch
from tensordict import TensorDict

import tools


class ParallelEnv:
    def __init__(
        self,
        constructor,
        env_num,
        device,
        *,
        max_episodes_per_worker=0,
        shutdown_timeout_seconds=5.0,
        log_worker_memory=False,
    ):
        self.constructor = constructor
        self.envs = []
        self.device = device
        self.max_episodes_per_worker = int(max_episodes_per_worker or 0)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self.log_worker_memory = bool(log_worker_memory)
        self._generations = [0 for _ in range(env_num)]
        self._episodes_since_restart = [0 for _ in range(env_num)]
        self._completed_episodes = [0 for _ in range(env_num)]
        self._restart_count = 0
        for idx in range(env_num):
            self.envs.append(self._make_env(idx))

    def _constructor_for(self, idx, generation):
        offset = self._completed_episodes[idx]
        try:
            sig = inspect.signature(self.constructor)
        except (TypeError, ValueError):
            return self.constructor(idx, generation, offset)
        params = list(sig.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            return self.constructor(idx, generation, offset)
        positional = [
            p for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        required = [p for p in positional if p.default is inspect.Parameter.empty]
        if len(positional) >= 3:
            return self.constructor(idx, generation, offset)
        if len(positional) >= 2:
            return self.constructor(idx, generation)
        if len(positional) >= 1:
            return self.constructor(idx)
        raise TypeError(
            "ParallelEnv constructor must accept at least worker_idx, and may also accept "
            "worker_generation and completed_episode_offset"
        )

    def _make_env(self, idx):
        return Parallel(self._constructor_for(idx, self._generations[idx]), "process", slot=idx)

    def _worker_context(self, idx, phase):
        worker = self.envs[idx].worker.impl
        return (
            f"worker slot={idx} pid={worker.pid} exitcode={worker.exitcode} "
            f"generation={self._generations[idx]} env_step={getattr(self, '_last_step', None)} "
            f"phase={phase}"
        )

    def _restart_worker_if_due(self, idx, phase):
        if self.max_episodes_per_worker <= 0:
            return
        if self._episodes_since_restart[idx] < self.max_episodes_per_worker:
            return
        old = self.envs[idx]
        old_pid = old.worker.impl.pid
        old.close(timeout=self.shutdown_timeout_seconds)
        self._generations[idx] += 1
        self._episodes_since_restart[idx] = 0
        self._restart_count += 1
        self.envs[idx] = self._make_env(idx)
        print(
            f"[env_lifecycle] restarted worker slot={idx} old_pid={old_pid} "
            f"new_pid={self.envs[idx].worker.impl.pid} generation={self._generations[idx]} "
            f"completed_episodes={self._completed_episodes[idx]} phase={phase}",
            flush=True,
        )

    @property
    def worker_restarts(self):
        return self._restart_count

    @property
    def completed_episodes(self):
        return int(sum(self._completed_episodes))

    def worker_infos(self):
        infos = []
        for idx, env in enumerate(self.envs):
            impl = env.worker.impl
            infos.append({
                "slot": idx,
                "pid": impl.pid,
                "exitcode": impl.exitcode,
                "generation": self._generations[idx],
                "completed_episode_offset": self._completed_episodes[idx],
                "episodes_since_restart": self._episodes_since_restart[idx],
                "completed_episodes": self._completed_episodes[idx],
            })
        return infos

    @property
    def observation_space(self):
        return self.envs[0].observation_space

    @property
    def action_space(self):
        return self.envs[0].action_space

    @property
    def env_num(self):
        return len(self.envs)

    def lift_dim(self, td):
        for key in td.keys():
            if td[key].ndim == 1:
                td[key] = td[key].unsqueeze(-1)
        return td

    def step(self, action, done):
        """Step all environments.

        Notes
        -----
        This implementation intentionally steps the environment processes on CPU.
        The returned TensorDict is pinned in CPU memory so that the caller can
        transfer it to GPU asynchronously (H2D with non_blocking=True).

        ``action`` and ``done`` may arrive on any device (CPU or CUDA).
        They are moved to CPU internally before dispatching to workers.
        """
        # Ensure inputs are on CPU for worker processes.
        action_np = tools.to_np(action)  # handles any device via .detach().cpu().numpy()
        done = done.cpu() if done.is_cuda else done
        promise = []
        for idx, (e, a, d) in enumerate(zip(self.envs, action_np, done)):
            phase = "reset" if bool(d) else "step"
            if bool(d):
                self._restart_worker_if_due(idx, phase)
                e = self.envs[idx]
            try:
                promise.append(e.reset() if bool(d) else e.step(a))
            except Exception as exc:
                raise RuntimeError(f"{self._worker_context(idx, phase)} failed to submit work") from exc
        new_o, new_r, new_d = [], [], []
        for idx, (p, d) in enumerate(zip(promise, done)):
            phase = "reset" if bool(d) else "step"
            if bool(d):
                try:
                    new_o.append(p())
                except Exception as exc:
                    raise RuntimeError(f"{self._worker_context(idx, phase)} failed during reset") from exc
                new_r.append(0.0)
                new_d.append(False)
            else:
                try:
                    o, r, d, _ = p()
                except Exception as exc:
                    raise RuntimeError(f"{self._worker_context(idx, phase)} failed during step") from exc
                new_o.append(o)
                new_r.append(r)
                new_d.append(d)
                if d:
                    self._completed_episodes[idx] += 1
                    self._episodes_since_restart[idx] += 1
        obs_stacked = {k: np.stack([o[k] for o in new_o]) for k in new_o[0].keys()}

        # Build CPU tensors first to avoid implicit GPU syncs and enable async H2D in caller.
        obs_tensors = {k: torch.as_tensor(v, device="cpu") for k, v in obs_stacked.items()}
        rew_stacked = torch.as_tensor(new_r, dtype=torch.float32, device="cpu")

        # Keep data on CPU; caller will .to(device, non_blocking=True) after pinning.
        # TensorDict batch size is (B,).
        # pin_memory() requires a CUDA device; skip on CPU-only machines.
        td = TensorDict({**obs_tensors, "reward": rew_stacked}, batch_size=(self.env_num,), device="cpu")
        if torch.cuda.is_available():
            td = td.pin_memory()
        done = torch.as_tensor(new_d, device="cpu")
        return self.lift_dim(td), done

    def close(self):
        for env in self.envs:
            env.close(timeout=self.shutdown_timeout_seconds)


class Parallel:
    def __init__(self, constructor, strategy, slot=None):
        self.worker = Worker(bind(self._respond, constructor), strategy, state=True)
        self.callables = {}
        self.slot = slot

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            if name not in self.callables:
                self.callables[name] = self.worker(PMessage.CALLABLE, name)()
            if self.callables[name]:
                return bind(self.worker, PMessage.CALL, name)
            return self.worker(PMessage.READ, name)()
        except AttributeError:
            raise ValueError(name)

    def __len__(self):
        return self.worker(PMessage.CALL, "__len__")()

    def close(self, timeout=None):
        self.worker.close(timeout=timeout)

    @staticmethod
    def _respond(constructor, state, message, name, *args, **kwargs):
        state = state or constructor()  # Instantiate at first time
        try:
            if message == PMessage.CALLABLE:
                assert not args and not kwargs, (args, kwargs)
                result = callable(getattr(state, name))
            elif message == PMessage.CALL:
                result = getattr(state, name)(*args, **kwargs)
            elif message == PMessage.READ:
                assert not args and not kwargs, (args, kwargs)
                result = getattr(state, name)
        except Exception as exc:
            context = {}
            debug_context = getattr(state, "get_debug_context", None)
            if callable(debug_context):
                with contextlib.suppress(Exception):
                    context = debug_context()
            raise RuntimeError(f"worker env call failed: op={name} context={context}") from exc
        return state, result


class PMessage(enum.Enum):
    CALLABLE = 2
    CALL = 3
    READ = 4


class Worker:
    initializers = []

    def __init__(self, fn, strategy="thread", state=False):
        if not state:

            def fn_wrapper(s, *args, **kwargs):
                return (s, fn(*args, **kwargs))

            fn = fn_wrapper
        inits = self.initializers
        self.impl = {
            "process": bind(ProcessPipeWorker, initializers=inits),
            "daemon": bind(ProcessPipeWorker, initializers=inits, daemon=True),
        }[strategy](fn)
        self.promise = None

    def __call__(self, *args, **kwargs):
        self.promise and self.promise()  # Raise previous exception if any.
        self.promise = self.impl(*args, **kwargs)
        return self.promise

    def wait(self):
        return self.impl.wait()

    def close(self, timeout=None):
        self.impl.close(timeout=0.1 if timeout is None else timeout)


class ProcessPipeWorker:
    def __init__(self, fn, initializers=(), daemon=False):
        import multiprocessing

        import cloudpickle

        self._context = multiprocessing.get_context("spawn")
        self._pipe, pipe = self._context.Pipe()
        fn = cloudpickle.dumps(fn)
        initializers = cloudpickle.dumps(initializers)
        self._process = self._context.Process(target=self._loop, args=(pipe, fn, initializers), daemon=daemon)
        self._process.start()
        self._nextid = 0
        self._results = {}
        assert self._submit(Message.OK)()
        atexit.register(self.close)

    def __call__(self, *args, **kwargs):
        return self._submit(Message.RUN, (args, kwargs))

    def wait(self):
        pass

    def close(self, timeout=0.1):
        try:
            self._pipe.send((Message.STOP, self._nextid, None))
            self._pipe.close()
        except (AttributeError, OSError):
            pass  # The connection was already closed.
        try:
            self._process.join(float(timeout))
            if self._process.exitcode is None:
                try:
                    self._process.terminate()
                    self._process.join(float(timeout))
                    if self._process.exitcode is None:
                        os.kill(self._process.pid, 9)
                        time.sleep(0.1)
                except Exception:
                    pass
        except (AttributeError, AssertionError):
            pass

    @property
    def pid(self):
        return getattr(self._process, "pid", None)

    @property
    def exitcode(self):
        return getattr(self._process, "exitcode", None)

    def _submit(self, message, payload=None):
        callid = self._nextid
        self._nextid += 1
        self._pipe.send((message, callid, payload))
        return Future(self._receive, callid)

    def _receive(self, callid):
        while callid not in self._results:
            try:
                message, received_callid, payload = self._pipe.recv()
            except (OSError, EOFError):
                raise RuntimeError(
                    f"Lost connection to worker pid={self.pid} exitcode={self.exitcode}."
                )
            if message == Message.ERROR:
                raise Exception(payload)
            if message == Message.RESULT:
                self._results[received_callid] = payload
            else:
                raise RuntimeError(f"Unexpected message: {message}")
        return self._results.pop(callid)

    @staticmethod
    def _loop(pipe, function, initializers):
        try:
            callid = None
            state = None
            import cloudpickle

            initializers = cloudpickle.loads(initializers)
            function = cloudpickle.loads(function)
            [fn() for fn in initializers]
            while True:
                if not pipe.poll(0.1):
                    continue  # Wake up for keyboard interrupts.
                message, callid, payload = pipe.recv()
                if message == Message.STOP:
                    return
                if message == Message.OK:
                    pipe.send((Message.RESULT, callid, True))
                elif message == Message.RUN:
                    args, kwargs = payload
                    state, result = function(state, *args, **kwargs)
                    pipe.send((Message.RESULT, callid, result))
                else:
                    raise KeyError(f"Invalid message: {message}")
        except (EOFError, KeyboardInterrupt):
            pass
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print(f"Error inside process worker: {stacktrace}.", flush=True)
            pipe.send((Message.ERROR, callid, stacktrace))
        finally:
            with contextlib.suppress(Exception):
                pipe.close()


class Message(enum.Enum):
    OK = 1
    RUN = 2
    RESULT = 3
    STOP = 4
    ERROR = 5


class Future:
    def __init__(self, receive, callid):
        self._receive = receive
        self._callid = callid
        self._result = None
        self._complete = False

    def __call__(self):
        if not self._complete:
            self._result = self._receive(self._callid)
            self._complete = True
        return self._result
