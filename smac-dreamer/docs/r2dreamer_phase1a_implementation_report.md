# R2-Dreamer Migration — Phase 1A Implementation Report

**Scope:** Environment and action-interface migration only (stages 4 + 5 of the approved
plan). No R2-Dreamer model, replay buffer, imagination masking, training loop, Hydra
configs, logging migration, checkpoint migration, or JAX removal.

**Branch:** `r2dreamer` (dedicated migration branch). Working tree was clean before edits.
No Git branches were created, deleted, renamed, tagged, committed, or pushed.

---

## 1. Scope completed
- Re-based `SMACliteDreamerEnv` from `embodied.Env` onto `gymnasium.Env` with standard
  `reset(*, seed, options)` and `step(action)` returning
  `(observation, reward, terminated, truncated, info)`.
- Replaced `elements.Space` with `gymnasium.spaces` (`Dict`, `Box`, `MultiDiscrete`).
- Moved all diagnostic / logging metrics out of the model observation and into `info`,
  renaming the `log/` prefix to `log_` (definitions and aggregation unchanged).
- Implemented a project-owned factorised multi-agent action codec
  (`FactorisedActionCodec`) using concatenated one-hot groups `[C]*A`, never a joint
  `C**A` space.
- Connected the codec to the env: `step` accepts a flat factorised one-hot (or integer
  actions, or the legacy action dict) and forwards only real-agent integer actions to
  SMAClite; padded slots are dropped / forced to noop.
- Added a `tests/` suite (codec, env, padding) and a JAX-free Gymnasium smoke-test script.
- Saved the approved migration plan to `docs/r2dreamer_migration_plan.md`.

## 2. Files created
| Path | Purpose |
|---|---|
| `src/smacdreamer/envs/action_codec.py` | `FactorisedActionCodec` (int↔one-hot, validation, padding/noop, Gym `MultiDiscrete` space). Pure NumPy, JAX-free. |
| `scripts/smoke_test_gym_smaclite_env.py` | Gymnasium smoke test: reset → sample valid factorised actions → one full episode → print reward/length/outcome/invalid/masking-failure/shapes. Imports no JAX/Elements/Embodied/Portal/DreamerV3. |
| `tests/conftest.py` | Path setup (`src`, `external/smaclite`; **excludes** `external/dreamerv3`), `requires_smaclite` skip marker, `fixed_env` fixture. |
| `tests/test_action_codec.py` | 18 codec tests (pure NumPy). |
| `tests/test_smaclite_env.py` | 14 env tests (require smaclite). |
| `tests/test_padding.py` | 6 padding tests (require smaclite). |
| `docs/r2dreamer_migration_plan.md` | The approved full migration plan. |
| `docs/r2dreamer_phase1a_implementation_report.md` | This report. |

## 3. Files modified
| Path | Change |
|---|---|
| `src/smacdreamer/envs/smaclite_dreamer_env.py` | Framework-facing rewrite: `gymnasium.Env` base, Gym spaces, `reset`/`step` semantics, `log/`→`info[log_*]`, action codec integration. **All SMAClite/NumPy logic (sanitisation, shaping, diagnostics) preserved verbatim.** |

**Not modified (preserved as required):** `src/smacdreamer/envs/padding.py`,
`src/smacdreamer/envs/map_sampler.py`, `src/smacdreamer/envs/reward_shaping.py`,
`external/r2dreamer/`, `external/smaclite/`, `external/dreamerv3/`, all config/manifest/map
files, the legacy `scripts/smoke_test_smaclite_env.py` (left intact for the JAX path).

## 4. Preserved behaviours
Centralised single controller; flattened `state`; `avail_actions`; one action per allied
unit; map loading (`_open_env`); map sampling (`MapSampler`); padding (`PaddingDims` +
helpers); `agent_mask`; `real_agent_action_mask`; reward shaping (legacy flat-param **and**
v2 config, byte-for-byte identical math); original-reward tracking; action sanitisation;
noop rescue (now via `NOOP_ACTION = 0`); timing-lag vs masking-failure classification;
invalid-action metrics; per-map metadata; coverage metrics. The per-episode accumulators,
`_sanitise_actions`, `_compute_ep_metrics`, and `_zero_ep_metrics` bodies are unchanged
except for the `log/`→`log_` key rename.

## 5. Interface changes (with rationale)
| Old (JAX/embodied) | New (Gym/PyTorch) | Why | Compatibility impact |
|---|---|---|---|
| `class SMACliteDreamerEnv(embodied.Env)` | `class SMACliteDreamerEnv(gym.Env)` | Remove embodied; R2-Dreamer expects a Gymnasium env | Base class change; ctor args unchanged |
| `obs_space` / `act_space` properties | `observation_space` / `action_space` attributes | Gymnasium convention | Callers must use Gym attribute names |
| `step(action_dict)` with `action["reset"]` | `reset(*, seed, options)` + `step(action)` | Gymnasium reset/step + ParallelEnv auto-reset | Reset is now an explicit call; legacy dict actions still accepted by `step` for convenience |
| obs returns `reward` + `log/*` | `step` returns `reward`; `log_*` go in `info` | Gym returns reward separately; logging-only fields must not be encoder inputs | Metric consumers read `info` (renamed `log_`) |
| action: dict `action_0..action_{N-1}` (int) | `MultiDiscrete([C]*A)` → flat one-hot via codec | Maps cleanly to R2-Dreamer `MultiOneHotDist` without a joint space | Action producers emit a factorised one-hot; codec converts |
| `elements.Space(...)` | `gymnasium.spaces.*` | Remove Elements | Space objects differ; shapes/dtypes preserved |

**Compatibility adapters retained:** `step` still accepts the legacy `{"action_i":...}`
dict and a plain integer-action vector, so existing call sites can migrate incrementally.

## 6. Observation schema (model observation — fixed shape within a run)
| Field | Shape (no pad / Phase 3 pad) | dtype | Destination |
|---|---|---|---|
| `state` | `(A*O,)` | float32 | encoder input |
| `avail_actions` | `(A*C,)` | float32 | replay + masking |
| `agent_mask` *(Phase 3 only)* | `(A,)` | float32 | replay + actor-loss masking |
| `real_agent_action_mask` *(Phase 3 only)* | `(A*C,)` | float32 | replay + diagnostics |
| `is_first` | `()` | bool | replay / RSSM reset |
| `is_last` | `()` | bool | replay |
| `is_terminal` | `()` | bool | replay (continuation) |

`A = n_agents` (or `pad_dims.max_agents`), `C = n_actions` (or `max_actions`),
`O = obs_size` (or `max_obs_size`). `reward` is now the `step` return value, not an obs
field. `map_id` and all `log_*` are **logging-only** and live in `info`, never encoder
inputs.

## 7. Action schema
- Logical: one categorical action per allied unit (`MultiDiscrete([C]*A)`).
- Internal/model: concatenated factorised one-hot, flat shape `(A*C,)`, groups `[C]*A`,
  group `i` occupies `flat[i*C:(i+1)*C]`. This is exactly the split R2-Dreamer's
  `MultiOneHotDist` performs (`torch.split(logits, [C]*A, -1)`).
- Fixed map (e.g. 2s3z): `A=5, C=6` → flat `(30,)`, groups `[6]*5` (illustrative; C is
  read from the live env).
- Phase 3 padded (overfit 2s3z manifest): `A=8, C=15` → flat `(120,)`, groups `[15]*8`.
- Conversions: `encode(int_actions, num_real_agents)` → one-hot (padded slots → noop);
  `decode(one_hot, num_real_agents)` → per-unit ints for real agents only (argmax/group).
- SMAClite receives only the real-agent integer list; sanitisation remains the final
  safety net.

## 8. Metric mapping (old obs key → new info key)
Definitions and aggregation level are unchanged; only the location (obs→info) and prefix
(`log/`→`log_`) changed. A machine-readable map lives in `METRIC_NAME_MAP` at the top of
`smaclite_dreamer_env.py`. Representative rows:

| Old name | New info key | Definition | Aggregation |
|---|---|---|---|
| `log/battle_won` | `log_battle_won` | episode win flag | episode |
| `log/post_mask_invalid_action_count` | `log_post_mask_invalid_action_count` | invalids at step time after policy mask | episode |
| `log/timing_lag_invalid_action_count` | `log_timing_lag_invalid_action_count` | invalid now, valid in prev mask | episode |
| `log/masking_failure_count` | `log_masking_failure_count` | invalid now, already invalid in prev mask | episode |
| `log/step_post_mask_invalid_count` | `log_step_post_mask_invalid_count` | invalids this step | step |
| `log/original_env_reward` / `log/shaped_reward` | `log_original_env_reward` / `log_shaped_reward` | raw vs shaped reward | step |
| `log/episode_*_return` | `log_episode_*_return` | per-episode return totals | episode |
| `log/map_id`, `log/num_real_agents`, `log/padded_agent_count` | `log_map_id`, … | map/padding metadata | step |
| `log/sampling_cycle`, `log/dataset_coverage_fraction` (sampler only) | `log_sampling_cycle`, … | dataset coverage | step |

(All ~55 keys are listed in `METRIC_NAME_MAP`; none were silently renamed or dropped.)

## 9. Tests added
- **Codec (18):** flat-dim/groups, action-space dims, non-positive dim rejection, encode
  one-hot, out-of-range/wrong-length rejection, decode round trip (multi-agent), decode
  from logits (argmax), all-actions round trip, padded-noop encode, real-count override,
  decode real-only, validate accept/reject (non-binary, multi-hot), invalid flat shape,
  bad dtype, configurable dtype.
- **Env (14, require smaclite):** construction/spaces, no-JAX-after-use, reset format,
  step format, obs field shapes/dtypes, step-without-reset raises, one-hot→int forwarding,
  legacy dict action, time-limit truncation, `is_last == terminated or truncated`, invalid
  sanitisation, noop, timing-lag vs masking-failure counters, episode metrics on done.
- **Padding (6, require smaclite):** fixed-map no-pad shapes, padded obs shape, agent_mask,
  real_agent_action_mask, padded-agent action ignored, padded env runs.

## 10. Test results
Command: `python -m pytest tests/` (Python 3.13.13, pytest 9.0.3, gymnasium 1.3.0).

```
tests/test_action_codec.py ..................            [18 passed]
tests/test_padding.py      ssssss                        [6 skipped]
tests/test_smaclite_env.py ssssssssssssss               [14 skipped]
======================= 18 passed, 20 skipped in 0.15s =======================
```

- **18 passed** — all pure-NumPy codec tests (incl. the gymnasium action-space test).
- **20 skipped** — env + padding tests skip because the **SMAClite simulator is not
  importable in this interpreter** (`smaclite` requires `sklearn`/`pygame`/`Rtree`/etc.,
  which are not installed in the available Python 3.13 Windows-store interpreter — this is
  not the project's real 3.11 runtime). They are written and will execute in the project
  environment with no code changes. Verified the skip reason is dependency-only via direct
  `import smaclite` → `ModuleNotFoundError: No module named 'sklearn'`.

No failures. No warnings of note.

**Static / runtime JAX-free checks (passed):**
- `grep` for `jax|elements|embodied|portal|dreamerv3` imports across all touched files →
  none found.
- `import smacdreamer.envs.action_codec` → no forbidden modules in `sys.modules`;
  encode/decode round trip correct.
- `python -m py_compile smaclite_dreamer_env.py` → OK.

## 11. Smoke-test results
Command: `python scripts/smoke_test_gym_smaclite_env.py --scenario 2s3z`.

- The script imports cleanly (no JAX/Elements/Embodied/Portal/DreamerV3) and reaches env
  construction, then fails **only** because the simulator's own dependency `sklearn` is
  missing in this interpreter (`ModuleNotFoundError: No module named 'sklearn'`) — not due
  to any forbidden import or code error.
- Exact observation/action shapes therefore could not be printed from a live rollout in
  this interpreter. Expected shapes (documented and asserted in tests): fixed 2s3z →
  `state (n_agents*obs_size,)`, `avail_actions (n_agents*n_actions,)`, action flat
  `(n_agents*n_actions,)`. The script must be run in the project's runtime to print live
  reward/length/outcome/invalid/masking-failure values.

## 12. Unresolved issues
- **Live env + padding tests and smoke test not executed here.** The available
  interpreter is Python 3.13 (Windows-store) without the SMAClite dependency chain
  (sklearn/pygame/Rtree/numba) and is not the project's 3.11/Kubeflow runtime. Installing
  that chain into a mismatched interpreter was deliberately avoided (out of scope and
  non-representative). **Action required:** run `python -m pytest tests/` and
  `python scripts/smoke_test_gym_smaclite_env.py --scenario 2s3z` in the project
  environment to confirm the 20 currently-skipped tests pass and to capture live shapes.
- No project-level dependency manifest exists; gymnasium/pytest were installed ad hoc into
  the local interpreter only to run the codec tests. Adding a manifest is a later stage.

## 13. Deviations from the approved migration plan
- None in design. The plan placed "Gymnasium wrapper" (stage 4) and "action codec"
  (stage 5) in Phase 1A; both are implemented as specified (factorised one-hot, no joint
  space, log fields moved to `info`).
- Minor additive choices (within plan intent, flagged for review):
  - `step` also accepts the **legacy action dict** and a **plain integer-action vector**
    as compatibility adapters, easing incremental call-site migration. The canonical input
    is the flat factorised one-hot.
  - `battle_won` is surfaced at the top level of `info` (in addition to `log_battle_won`)
    to match SMAClite's native `info` convention.
  - Gymnasium `reset` requires an explicit call before `step` (old code reset via an
    action flag); `step` raises `RuntimeError` if called before `reset`.

## 14. Recommended scope for Phase 1B
Phase 1B should **not** include the full R2-Dreamer model. Recommended, ordered:
1. **R2-Dreamer environment factory + registration** under `src/smacdreamer/` that wraps
   `SMACliteDreamerEnv` with R2-Dreamer's `MultiOneHotAction` (consuming the codec's
   `MultiDiscrete`) and the `Dtype`/`TimeLimit` conventions, producing a `ParallelEnv`-
   compatible factory (`make_envs`-style) without editing `external/r2dreamer`.
2. **TensorDict transition shaping** for a single env: verify the obs Dict + `action`
   one-hot + `is_*` + `reward` map into the expected `(B,1,*)` TensorDict (no replay yet).
3. **Real-rollout action masking** re-expressed against `Dreamer.act()`: apply
   `avail_actions` to per-group logits before sampling/mode (mirrors the old
   `_apply_avail_mask`), with the all-zero→noop rescue.
4. **Run the Phase 1A env/padding tests + smoke test in the real runtime** and record live
   shapes/metrics as the Phase 1B baseline.

Explicitly deferred beyond 1B: imagination masking, padded-agent actor-loss exclusion,
replay buffer, checkpointing, logging migration, Hydra training configs, JAX removal.

---

*Stop point: Phase 1A complete. Awaiting approval before beginning Phase 1B.*
