"""§0 DE-RISKING SPIKE — ParallelEnv per-episode map routing.

Throwaway verification script for the multimap plan. Verifies the single most
load-bearing assumption: that R2-Dreamer's ParallelEnv, when it auto-resets a done
worker by calling env.reset(), causes SMACliteDreamerEnv to advance its MapSampler and
switch maps — i.e. per-episode map routing happens automatically, with fixed obs shapes
and workers not in lockstep.

Run BEFORE building the factory / training script / reward registry / discovery module.
If the routing assumption fails, the plan's central design changes.

Checks
------
(a) map id changes across resets within a worker      (sampling advances on auto-reset)
(b) workers are not in lockstep                        (different map_id sequences)
(c) obs shapes stay fixed across all maps/resets       (model can be built once)
(d) buffer-leak guard: OnlineTrainer.eval never calls replay_buffer.add_transition
    (static code-path assertion; full eval wiring is out of spike scope)

Per-worker seeding uses a HASHED combination of (base_seed, worker_idx) via
numpy.random.SeedSequence, so the "not in lockstep" property is robust rather than a
coincidence of adjacent integer seeds.

CONTEXT — the log_* forwarding fix this spike depends on
--------------------------------------------------------
After the Phase 1A migration, SMACliteDreamerEnv puts all `log_*` metrics (including
`log_map_id` and the invalid-action diagnostics) in the Gymnasium `info` dict.
external/r2dreamer/envs/parallel.py DISCARDS `info` on step (`o, r, d, _ = p()`), so
without intervention those keys never reach the TensorDict the trainer reads. The project
adapter SMACliteR2DreamerAdapter now merges every `log_*` info key into the returned obs
dict (zero external edits; the encoder excludes `log_*` from model inputs). This spike uses
the REAL adapter and reads `log_map_id` straight from the ParallelEnv TensorDict to confirm
that forwarding works end-to-end alongside the routing checks.

Usage:
    python scripts/spike_parallelenv_map_routing.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (
    str(ROOT / "src"),
    str(ROOT / "external" / "r2dreamer"),
    str(ROOT / "external" / "smaclite"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch


# Three distinct small builtin scenarios with different unit counts. Padding keeps the
# obs/action shapes fixed across them (so check (c) is meaningful).
SPIKE_MAPS = ["2s3z", "2s_vs_1sc", "3s_vs_5z"]
PAD = dict(max_agents=8, max_enemies=9, max_actions=15, max_obs_size=136)
ENV_NUM = 2
N_RESETS = 10
BASE_SEED = 12345


def _worker_seed(base_seed: int, idx: int) -> int:
    """Robust per-worker seed: hash (base_seed, idx) via SeedSequence, not base+idx."""
    ss = np.random.SeedSequence([base_seed, idx])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _make_worker(idx: int):
    """Zero-arg constructor for ParallelEnv. Reconstructs everything inside the worker."""
    def _build():
        # Re-add paths inside the spawned worker process before any project import.
        for _p in (
            str(ROOT / "src"),
            str(ROOT / "external" / "r2dreamer"),
            str(ROOT / "external" / "smaclite"),
        ):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from smacdreamer.envs.smaclite_dreamer_env import SMACliteDreamerEnv
        from smacdreamer.envs.r2dreamer_adapter import SMACliteR2DreamerAdapter
        from smacdreamer.envs.padding import PaddingDims
        from smacdreamer.envs.map_sampler import MapSampler, MapEntry

        entries = [MapEntry(name=m, type="builtin") for m in SPIKE_MAPS]
        sampler = MapSampler(
            maps=entries,
            mode="shuffled_round_robin",
            seed=_worker_seed(BASE_SEED, idx),
        )
        env = SMACliteDreamerEnv(
            scenario=SPIKE_MAPS[0],
            max_episode_steps=200,
            seed=_worker_seed(BASE_SEED, idx),
            map_sampler=sampler,
            pad_dims=PaddingDims(**PAD),
        )
        # Real project adapter — now forwards log_* (incl. log_map_id) from info into obs.
        return SMACliteR2DreamerAdapter(env)

    return _build


def _assert_eval_has_no_buffer_write() -> bool:
    """Check (d): OnlineTrainer.eval source never calls replay_buffer.add_transition."""
    import inspect
    import trainer as _trainer

    eval_src = inspect.getsource(_trainer.OnlineTrainer.eval)
    begin_src = inspect.getsource(_trainer.OnlineTrainer.begin)
    eval_writes = "add_transition" in eval_src
    begin_writes = "add_transition" in begin_src
    print("\n[check d] buffer-leak guard (static code-path):")
    print(f"    OnlineTrainer.eval  calls add_transition : {eval_writes}  (must be False)")
    print(f"    OnlineTrainer.begin calls add_transition : {begin_writes} (expected True)")
    return (not eval_writes) and begin_writes


def main():
    from envs.parallel import ParallelEnv

    print("=" * 72)
    print("§0 SPIKE — ParallelEnv per-episode map routing")
    print(f"  maps={SPIKE_MAPS}  env_num={ENV_NUM}  resets={N_RESETS}  base_seed={BASE_SEED}")
    print(f"  per-worker seeds: " +
          ", ".join(f"w{idx}={_worker_seed(BASE_SEED, idx)}" for idx in range(ENV_NUM)))
    print("=" * 72)

    envs = ParallelEnv(lambda i: _make_worker(i), ENV_NUM, device="cpu")

    # Dummy action of the right shape: flat factorised one-hot (A*C,) per env.
    A, C = PAD["max_agents"], PAD["max_actions"]
    act = torch.zeros((ENV_NUM, A * C), dtype=torch.float32)

    # Force a reset every iteration so each step routes through env.reset() (auto-reset path).
    done = torch.ones(ENV_NUM, dtype=torch.bool)

    map_seq = [[] for _ in range(ENV_NUM)]   # per-worker map_id sequence
    shapes_seen = set()

    print(f"\n{'step':>4} | {'worker':>6} | {'map_id':>6} | state.shape")
    print("-" * 48)
    for t in range(N_RESETS):
        td, step_done = envs.step(act, done)
        map_ids = td["log_map_id"].reshape(ENV_NUM).tolist()
        states = td["state"]
        for w in range(ENV_NUM):
            mid = int(map_ids[w])
            shp = tuple(states[w].shape)
            map_seq[w].append(mid)
            shapes_seen.add(shp)
            print(f"{t:>4} | {w:>6} | {mid:>6} | {shp}")
        done = torch.ones(ENV_NUM, dtype=torch.bool)  # keep forcing resets

    # ParallelEnv has no top-level close(); close each worker env. Guard so cleanup
    # errors never mask the spike result.
    try:
        for e in getattr(envs, "envs", []):
            try:
                e.close()
            except Exception:
                pass
    except Exception:
        pass

    # ---- Assertions -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    # (a) map id changes across resets within a worker
    a_ok = all(len(set(seq)) > 1 for seq in map_seq)
    print(f"[check a] map id changes within each worker      : {a_ok}")
    for w, seq in enumerate(map_seq):
        print(f"    worker {w} map_id sequence: {seq}  (unique={sorted(set(seq))})")

    # (b) workers not in lockstep
    b_ok = ENV_NUM < 2 or any(map_seq[0] != map_seq[w] for w in range(1, ENV_NUM))
    print(f"[check b] workers not in lockstep                : {b_ok}")

    # (c) obs shapes fixed
    expected_state = (A * PAD["max_obs_size"],)
    c_ok = (len(shapes_seen) == 1) and (next(iter(shapes_seen)) == expected_state)
    print(f"[check c] obs state shape fixed                  : {c_ok}  "
          f"(seen={shapes_seen}, expected={{{expected_state}}})")

    # (d) buffer-leak guard
    d_ok = _assert_eval_has_no_buffer_write()
    print(f"[check d] eval does not write to replay buffer   : {d_ok}")

    print("\n" + "=" * 72)
    all_ok = a_ok and b_ok and c_ok and d_ok
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 72)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
