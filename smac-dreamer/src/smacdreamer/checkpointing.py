"""Time-based periodic checkpointing for R2-Dreamer training.

R2-Dreamer's OnlineTrainer.begin() runs the whole training loop without
yielding control and never checkpoints mid-run.  Rather than fork that loop,
we hook PeriodicCheckpointer.maybe_save() into a method the loop calls every
env step (agent.act), and gate the actual save on wall-clock elapsed time.

Saves are written atomically (temp file + os.replace) so a process kill during
a write cannot corrupt the existing good checkpoint — which matters precisely
because the point of periodic checkpoints is crash resilience.

Each checkpoint contains:
    agent_state_dict   — model weights
    optims_state_dict  — optimizer state (so a run can later resume cleanly)
    step               — env-step proxy at save time (for bookkeeping)

Note: the replay buffer is NOT persisted, so a resumed run reloads weights and
optimizer state but refills its buffer from scratch.  Full buffer persistence
is a separate, larger task.
"""

import os
import random
import time
import pathlib

import numpy as np
import torch

import tools  # r2dreamer's tools (recursively_collect_optim_state_dict)


class PeriodicCheckpointer:
    """Save agent + optimizer state every ``interval_seconds`` of wall-clock time.

    Parameters
    ----------
    agent            : the Dreamer nn.Module
    logdir           : directory for checkpoint files
    interval_seconds : minimum wall-clock seconds between saves
    step_fn          : zero-arg callable returning the current env step (for
                       filenames / bookkeeping); may be None
    keep_snapshots   : if True, also write an immutable step_<N>.pt each save,
                       in addition to overwriting latest.pt
    """

    def __init__(
        self, agent, logdir, interval_seconds, step_fn=None,
        keep_snapshots=False, extra_state_fn=None,
    ):
        # UNIFIED_PRIORITY_V1
        self._agent = agent
        self._logdir = pathlib.Path(logdir)
        self._interval = float(interval_seconds)
        self._step_fn = step_fn
        self._keep_snapshots = bool(keep_snapshots)
        self._extra_state_fn = extra_state_fn
        # Start the clock now so the first save lands one interval in, not at t=0.
        self._last_save_time = time.time()
        self._save_count = 0

    def _current_step(self):
        if self._step_fn is None:
            return 0
        try:
            return int(self._step_fn())
        except Exception:
            return 0

    def maybe_save(self):
        """Save iff at least ``interval_seconds`` have elapsed since the last save."""
        if time.time() - self._last_save_time >= self._interval:
            self.save()

    def save(self, final=False):
        """Write a checkpoint now (atomically). ``final`` only affects the log line."""
        step = self._current_step()
        payload = {
            "agent_state_dict": self._agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(self._agent),
            "agent_training_state": (
                self._agent.training_state_dict()
                if hasattr(self._agent, "training_state_dict") else None
            ),
            "step": step,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

        if self._extra_state_fn is not None:
            extra = self._extra_state_fn()
            if extra:
                overlap = set(payload).intersection(extra)
                if overlap:
                    raise KeyError(f'extra checkpoint state overwrites keys: {sorted(overlap)}')
                payload.update(extra)
        latest = self._logdir / "latest.pt"
        self._atomic_torch_save(payload, latest)

        if self._keep_snapshots and not final:
            snapshot = self._logdir / f"step_{step:010d}.pt"
            # Reuse the already-written latest.pt to avoid a second serialize.
            self._atomic_torch_save(payload, snapshot)

        self._last_save_time = time.time()
        self._save_count += 1
        tag = "final checkpoint" if final else f"checkpoint #{self._save_count}"
        mins = self._interval / 60.0
        print(
            f"  [checkpoint] {tag} saved at step {step:,} -> {latest.name}"
            + ("" if final else f"  (every ~{mins:g} min)")
        )

    @staticmethod
    def _atomic_torch_save(payload, path: pathlib.Path):
        """torch.save to a temp file then os.replace — never leaves a partial file."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic on the same filesystem (incl. Windows)


def attach_checkpointing(agent, checkpointer):
    """Wrap ``agent.act`` so a time-gated save runs after each policy step.

    agent.act is called once per env step by both the train and eval loops,
    giving a reliable, frequent hook.  The save itself is time-gated, so the
    per-step overhead is just one time.time() comparison.

    Returns the original (unwrapped) act, in case the caller wants to restore it.
    """
    original_act = agent.act

    def act_with_checkpoint(*args, **kwargs):
        out = original_act(*args, **kwargs)
        checkpointer.maybe_save()
        return out

    # nn.Module.__setattr__ stores a plain function in the instance __dict__,
    # shadowing the class method; attribute lookup at call time picks this up.
    agent.act = act_with_checkpoint
    return original_act
