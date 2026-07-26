# R2-Dreamer Migration Plan (SMAClite pipeline: JAX DreamerV3 → PyTorch R2-Dreamer)

> **Status:** Approved. Implementation proceeds in strict stages. This document is the
> authoritative migration plan referenced by per-phase implementation reports
> (e.g. `docs/r2dreamer_phase1a_implementation_report.md`).

## 1. Executive summary
Replace the JAX DreamerV3 backend (`external/dreamerv3`, Ninjax/Optax/Elements/Embodied)
with the PyTorch R2-Dreamer backend (`external/r2dreamer`, NM512) while preserving the
tested SMAClite centralised-control integration, the project structure, and the
four-phase experiment organisation. JAX is fully removed from the migrated runtime.
R2-Dreamer (`model.rep_loss=r2dreamer`) is the only training backend. The env wrapper is
re-based from `embodied.Env` onto `gymnasium.Env`; the dict-of-ints action interface
becomes a concatenated factorised one-hot (`MultiDiscrete` / `MultiOneHotDist`). The
masking logic currently in `agent.py` is re-expressed against R2-Dreamer's
`Dreamer.act()` / imagine loop via a project-owned subclass. Validation gate before
scaling: learn-to-win on ~50 easy custom maps (allies given a large advantage).

## 2. Confirmed requirements
- Runtime target: **Kubeflow / Linux GPU only**. No native-Windows runtime requirement.
  Assume Python 3.11 + CUDA + torch 2.8. `torch.compile` / fp16 may stay enabled.
- Old JAX: tag current branch as the JAX baseline, migrate on a dedicated branch, then
  **remove** `external/dreamerv3`, the JAX code in `agent.py`, and `jax:` config blocks.
- Back-compat: preserve the *meaning* of metrics, experiments, and phases; renames are
  allowed to fit Hydra / TensorBoard / PyTorch conventions (`log/` → `log_`,
  elements `--configs` → Hydra groups).
- Logging: **WandB and TensorBoard** both.
- Imagination masking (v1): constant start-mask broadcast across the horizon. No
  availability-prediction head in v1.
- Validation gate: overfit on ~**50 easy custom maps** (large ally advantage) so winning
  is easy → win rate clearly above random before scaling to the full padded curriculum.
- Parallelism/replay: start `env_num=1` for correctness/debuggability; tune env count
  and replay device (CPU vs GPU) later once shapes/memory are known.

## 3. Hard constraints (from the migration brief)
Remove JAX from the migrated runtime; preserve the project structure; preserve the
tested SMAClite integration behaviour; preserve wrapper purpose and external behaviour;
avoid direct upstream modifications (`external/r2dreamer`, `external/smaclite`); preserve
the experiment organisation; no JAX-checkpoint compatibility assumptions.

## 4. Current codebase architecture
SMAClite → `SMACliteDreamerEnv(embodied.Env)` (flattened `state`, `avail_actions`,
`agent_mask`/`real_agent_action_mask` in Phase 3+, ~40 `log/` metrics; action dict
`reset` + `action_0..action_{N-1}` int32) → `SMACliteAgent(dreamerv3.agent.Agent)`
(masked `policy()` + `loss()`) → DreamerV3 replay/train/log/ckpt → `evaluate_phase*.py`.
Pure-Python/NumPy helpers: `padding.py`, `reward_shaping.py`, `map_sampler.py`.

## 5. Current JAX dependency inventory
| Package | Importing project files | Purpose | Replacement | Removal stage |
|---|---|---|---|---|
| jax / jax.numpy | `agent.py` | logit masking math | torch ops in new agent | 5 / 18 |
| elements | `agent.py`, `smaclite_dreamer_env.py`, train/eval scripts | Config, Space, tree | Hydra + gymnasium.spaces | 3 / 9 / 18 |
| embodied(.jax.outs) | `agent.py`, `smaclite_dreamer_env.py` | Env base, Categorical | gymnasium.Env, MultiOneHotDist | 4 / 5 |
| dreamerv3.agent | `agent.py` | Agent base + loss helpers | r2dreamer `Dreamer` subclass | 6 / 18 |
| optax / ninjax / portal | train scripts (indirect via dreamerv3) | optim / runtime | LaProp/torch (built into r2dreamer) | 18 |

## 6. Target R2-Dreamer architecture
SMAClite → project Gymnasium wrapper (fixed-shape flat obs; `log_*` scalars in `info`) →
factorised one-hot action `[A*C]` → project `SMACliteDreamer(Dreamer)` subclass with
masked `act()` / imagine → TensorDict → TorchRL replay → torch training (fp16 / compile)
→ WandB + TensorBoard → `.pt` checkpoints → project per-map evaluation (JSON).

## 7. Compatibility matrix (high level)
- Agent: replace (DreamerV3 subclass → R2-Dreamer subclass); masking re-expressed.
- Env base/spaces: rewrite internals, keep path (`embodied.Env`→`gymnasium.Env`).
- Action: convert dict-of-ints ↔ concatenated one-hot via a new codec.
- Config: elements `--configs` → Hydra groups.
- Replay/ckpt/logging: new PyTorch implementations.

## 8. Preserved SMAClite functionality
Centralised single controller; flattened `state`; `avail_actions`; one action per allied
unit; map loading/sampling; padding; `agent_mask`; `real_agent_action_mask`; reward
shaping (legacy + v2); original-reward tracking; action sanitisation; noop rescue;
timing-lag vs masking-failure classification; invalid-action metrics; termination vs
truncation; per-map metadata; per-map evaluation; reproducible sampling; fixed obs/action
shapes within a run.

## 9. Required internal rewrites
`smaclite_dreamer_env.py` (base class, spaces, step/reset semantics, log routing);
`agent.py` (PyTorch masked subclass). New modules: action codec, Gym factory/registration,
Hydra configs, WandB logger, checkpoint+resume, per-map eval JSON.

## 10. Observation & action schemas
Obs Dict (Gym): `state` (f32, `[A*Omax]`) → encoder; `avail_actions` (f32, `[A*C]`) →
replay + masking; `agent_mask` (f32, `[A]`) → replay + loss; `real_agent_action_mask`
(f32, `[A*C]`) → replay + loss; `is_first`/`is_last`/`is_terminal` (bool); `reward` (f32).
`map_id` + `log_*` are logging-only and live in `info`, never encoder inputs. Action:
per-unit integers ↔ concatenated one-hot `[A*C]`, groups `[C]*A`. Documented shapes for a
fixed map and the Phase-3 padded config (A=8/20, C=15/200).

## 11. Replay schema (TensorDict)
Keys: `state`, `avail_actions`, `agent_mask`, `real_agent_action_mask`, `action` (one-hot),
`reward`, `is_first`/`is_last`/`is_terminal`, `episode` (id), `stoch`, `deter`, `map_id`.
SliceSampler keyed by `episode`; constant episode ids so short episodes stay sampable;
initial recurrent state from `[:,0]`; start with CPU storage, revisit GPU later.

## 12. Action-masking design
Real rollout + eval: apply `avail_actions` to per-group logits before sample/mode.
Imagination: constant start-mask broadcast across horizon (v1). Actor loss: same masked
distribution for log-prob / entropy / objective; compute per-agent log-prob + entropy,
then zero padded agents via `agent_mask` before summing. Env sanitisation kept as final
safety net only (not the primary masking mechanism).

## 13. R2 representation-learning integration
`model.rep_loss=r2dreamer`: no decoder; `Projector` + Barlow invariance + redundancy loss
(`loss_scales.barlow=0.05`, `r2dreamer.lambd=5e-4`); reconstruction diagnostics are lost;
account for low-dim SMAClite state inputs.

## 14. Configuration mapping
elements `--configs name` → Hydra groups: `env/smaclite_phase{1..4}.yaml`,
`model/sizeXXX.yaml`; named Phase-3 variants become configs/overrides. Map: steps,
train_ratio, env_num, batch_size/length, imag_horizon, horizon, lamb, act_entropy,
kl_free, model size, replay capacity, device, compile, precision, map manifest, map mode,
padding, reward shaping.

## 15. Logging & evaluation mapping
WandB + TensorBoard. Map ~40 SMAClite metrics + world-model/R2 losses (dyn KL, rep KL,
barlow invariance/redundancy, rew, cont, policy, value, entropy) + throughput / GPU mem.
Per-map deterministic masked eval → JSON (aggregate + per-episode); schema preserved.

## 16. Checkpoint strategy
`.pt`: model state, optimizer, scheduler, scaler, training step, config snapshot; replay
non-persistent (v1); resume restores all. JAX checkpoints are archived baseline artifacts,
not directly loadable into PyTorch.

## 17. File-by-file modification plan
- Retain unchanged: `padding.py`, `reward_shaping.py`, `map_sampler.py`, map JSONs/manifests.
- Retain path, rewrite internals: `smaclite_dreamer_env.py`, `agent.py`.
- Create: `envs/action_codec.py`, Gym factory, `configs/{env,model}/*.yaml`, WandB logger,
  `scripts/train_r2_smaclite_phase*.py`, eval scripts, resume util, `tests/`.
- Deprecate/remove (after baseline tag): `external/dreamerv3`, JAX code in `agent.py`,
  `jax:` config blocks, old JAX train/eval scripts.

## 18. Ordered implementation stages (gated)
1. branch + baseline tag · 2. add r2dreamer deps/manifest · 3. update CLAUDE.md ·
4. Gymnasium wrapper · 5. action codec · 6. fixed-map R2 training · 7. real-rollout mask ·
8. eval mask · 9. imagination mask · 10. padded-agent loss mask · 11. replay ·
12. ckpt + resume · 13. WandB/TB logging ·
**GATE: overfit ~50 easy custom maps → win rate ≫ random** · 14. same-shape multi-map ·
15. padded multi-map · 16. reward-shaping experiments · 17. GPU/throughput validation ·
18. remove JAX runtime · 19. remove/archive JAX files · 20. docs.

> **Phase 1A = stages 4 + 5 (+ the integration glue and tests for them).** It produces a
> tested Gymnasium-compatible SMAClite env and a tested factorised action codec only.

## 19. Test & acceptance matrix
no-JAX import; deps install; wrapper reset/step; term vs trunc; fixed shapes; action
encode/decode; random valid rollout; real mask; noop rescue; timing-lag class; masking-
failure class; replay insert/sample; one finite R2 update; ckpt save/load/resume; det
eval; CPU + GPU smoke; same-shape multi-map; padded multi-map; padded-agent loss exclusion;
imagination mask; per-map JSON eval; **learn-to-win on ~50 easy maps**; baseline compare.

## 20. Risks and mitigations
Agent masking re-expression is the highest-risk item (different control flow in
R2-Dreamer); mitigate with focused unit tests + the fixed-map learning gate. fp16/compile
instability on edge shapes; mitigate with a toggle and CPU smoke fallback. Padded 500-map
transitions are large (max_obs_size up to 1600); mitigate by starting replay on CPU.

## 21. Rollback strategy
Each stage rolls back to the prior stage. The JAX baseline is always recoverable via the
baseline tag / migration branch. No destructive Git actions without explicit instruction.

## 22. Explicitly out-of-scope (v1)
Availability-prediction head; JAX↔PyTorch checkpoint/replay conversion; multi-backend
runtime; native-Windows runtime support.

## 23. Remaining open questions
Exact torch/CUDA versions on the Kubeflow image; replay capacity for the padded 500-map
dataset; whether the ~50 "easy" maps already exist or must be generated via
`build_phase4_manifest.py`.
