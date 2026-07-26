# DreamerV3 × SMAClite Implementation Plan

## 1. Objective

Implement a training pipeline that allows DreamerV3 to train on the SMAClite simulator.

SMAClite is a cooperative multi-agent simulator. For this project, SMAClite will be exposed to DreamerV3 as a **single-agent centralised-control environment**.

One DreamerV3 agent will control all allied SMAClite units.

The initial goal is not to achieve strong training performance immediately. The initial goal is to build a correct, debuggable, and extensible training pipeline.

---

## 2. Repository Structure

Expected project structure:

```text
custom-smac/
├── CLAUDE.md
├── docs/
│   ├── implementation_plan.md
│   └── running_dreamer_smaclite.md
├── external/
│   ├── dreamerv3/
│   └── smaclite/
├── src/
│   └── smacdreamer/
│       ├── __init__.py
│       └── envs/
│           ├── __init__.py
│           ├── smaclite_dreamer_env.py
│           ├── map_sampler.py
│           └── padding.py
├── configs/
│   ├── smaclite_phase1.yaml
│   ├── smaclite_phase2.yaml
│   ├── smaclite_phase3.yaml
│   └── smaclite_phase4.yaml
├── scripts/
│   ├── smoke_test_smaclite_env.py
│   ├── train_dreamer_smaclite_phase1.py
│   └── evaluate_dreamer_smaclite.py
├── logs/
├── checkpoints/
└── results/
```

The external repositories should be treated as upstream dependencies.

Avoid modifying:

```text
external/dreamerv3/
external/smaclite/
```

unless absolutely necessary.

If an external repository must be modified, document:

1. which file was changed,
2. why the change was necessary,
3. the exact minimal patch,
4. how to revert it.

---

## 3. Hard Constraints

The implementation must follow these constraints:

1. Do not create custom SMAClite units.
2. Do not use custom unit types.
3. Use only existing SMAClite units and existing/example SMAClite scenarios unless explicitly instructed otherwise.
4. Treat SMAClite as a single-agent centralised-control environment.
5. One DreamerV3 agent controls all allied SMAClite units.
6. Prioritise correctness and debuggability over training speed.
7. Initial development target is native Windows with JAX CPU only.
8. The implementation should later be portable to a cloud GPU AI stack.
9. Keep observation and action spaces fixed within a single DreamerV3 training run.
10. Do not silently hide invalid actions. Invalid actions must be counted and logged.
11. Phase 1 must work before Phase 2.
12. Phase 2 must work before Phase 3.
13. Phase 3 must work before Phase 4.
14. Do not begin Phase 3 or Phase 4 implementation before Phase 1 is working.

---

## 4. Core Modelling Decision

SMAClite is multi-agent. DreamerV3 will initially be used as a single centralised controller.

This means:

```text
SMAClite allied units
        ↓
centralised observation
        ↓
single DreamerV3 policy
        ↓
joint allied action
        ↓
SMAClite environment step
```

The DreamerV3 agent should output one action per allied unit.

The environment wrapper should convert DreamerV3 actions into the joint action list expected by SMAClite.

---

## 5. Environment Formulation

### 5.1 Observation

The DreamerV3 observation should include:

1. flattened per-agent observations from all allied units,
2. available-action mask for each allied unit,
3. optional global state if SMAClite exposes it and it is useful,
4. alive/unit mask if required for padding in later phases,
5. map/scenario metadata for logging.

For Phase 1, use fixed-shape observations only.

Recommended Phase 1 observation dictionary:

```python
{
    "state": np.ndarray,          # flattened allied observations
    "avail_actions": np.ndarray,  # shape: [n_agents, n_actions]
    "reward": np.float32,
    "is_first": bool,
    "is_last": bool,
    "is_terminal": bool,
}
```

Optional later observation fields:

```python
{
    "agent_mask": np.ndarray,
    "enemy_mask": np.ndarray,
    "global_state": np.ndarray,
    "map_id": np.ndarray,
}
```

Do not add changing string values directly into the model observation. Map names should be used for logging, not as raw tensor observations, unless encoded consistently.

---

### 5.2 Action

The DreamerV3 action should represent one discrete action per allied unit.

Preferred design:

```python
{
    "action_0": int,
    "action_1": int,
    "action_2": int,
    ...
}
```

Each `action_i` corresponds to one allied unit.

Avoid using a single flattened joint action space unless necessary.

Reason:

```text
If there are N agents and A possible actions per agent,
a flattened joint action space has A^N possible actions.
```

This becomes too large quickly.

Separate categorical action heads are preferred because they scale as:

```text
N × A
```

instead of:

```text
A^N
```

---

### 5.3 Reward

Use the shared SMAClite team reward.

The reward should be returned to DreamerV3 as a scalar float.

Recommended type:

```python
np.float32
```

---

### 5.4 Termination

Correctly handle both Gymnasium-style termination signals:

```python
terminated
truncated
```

Recommended mapping:

```python
done = terminated or truncated
is_last = done
is_terminal = terminated
```

`terminated` means the actual environment episode ended naturally.

`truncated` means the episode was cut off by a time limit or wrapper condition.

DreamerV3 should receive correct episode-boundary signals.

---

### 5.5 Invalid Actions

SMAClite exposes available actions per allied unit.

DreamerV3 may select invalid actions, especially during early training.

For Phase 1:

1. detect invalid selected actions,
2. replace invalid actions with a valid fallback action,
3. count invalid actions,
4. log invalid action count,
5. log invalid action rate.

Recommended fallback for Phase 1:

```text
first valid action
```

Do not silently replace invalid actions without logging.

Later phases should investigate cleaner action masking inside the actor/policy distribution.

---

## 6. Required Codebase Analysis

Before implementing code, inspect both repositories and document findings.

The coding agent must not immediately start editing files.

First produce a codebase analysis covering the sections below.

---

### 6.1 DreamerV3 Analysis

Inspect DreamerV3 and document:

1. training entry point,
2. config loading system,
3. environment registration system,
4. expected environment interface,
5. observation space format,
6. action space format,
7. reset/step lifecycle,
8. replay buffer assumptions,
9. logging system,
10. checkpointing system,
11. evaluation flow,
12. how to run a CPU debug training job.

Relevant files may include:

```text
external/dreamerv3/dreamerv3/main.py
external/dreamerv3/dreamerv3/configs.yaml
external/dreamerv3/embodied/envs/from_gym.py
external/dreamerv3/embodied/
```

The exact relevant files should be confirmed by inspecting the repository.

---

### 6.2 SMAClite Analysis

Inspect SMAClite and document:

1. how environments are created,
2. scenario naming convention,
3. map/scenario config format,
4. reset API,
5. step API,
6. observation format,
7. action format,
8. available-action format,
9. reward format,
10. termination/truncation behaviour,
11. whether global state is available,
12. whether battle outcome is exposed in `info`,
13. where example maps/scenarios are stored,
14. how built-in unit types are defined.

Relevant files may include:

```text
external/smaclite/example.py
external/smaclite/smaclite/
external/smaclite/smaclite/env/
external/smaclite/smaclite/maps/
external/smaclite/smaclite/scenarios/
```

The exact relevant files should be confirmed by inspecting the repository.

---

## 7. Compatibility Issues to Resolve

The implementation must explicitly resolve these expected compatibility issues:

1. SMAClite is multi-agent, while DreamerV3 expects a single-agent environment interface.
2. SMAClite uses multiple allied actions per step.
3. DreamerV3 needs stable observation/action spaces.
4. Multi-map training can break shape consistency if unit counts differ.
5. SMAClite may expose Gymnasium-style reset/step APIs.
6. DreamerV3 may expect its own `embodied.Env` interface.
7. Action masks may not be natively enforced by DreamerV3.
8. Logging SMAClite-specific metrics may require wrapper-level metric extraction.
9. Windows CPU debugging may differ from cloud GPU training.
10. External repository modification should be avoided where possible.

---

## 8. Planned Files to Create

The following files may be created over the full project.

Phase 1 should only create the files needed for a minimal working pipeline.

Full target file list:

```text
src/smacdreamer/__init__.py
src/smacdreamer/envs/__init__.py
src/smacdreamer/envs/smaclite_dreamer_env.py
src/smacdreamer/envs/map_sampler.py
src/smacdreamer/envs/padding.py

configs/smaclite_phase1.yaml
configs/smaclite_phase2.yaml
configs/smaclite_phase3.yaml
configs/smaclite_phase4.yaml

scripts/smoke_test_smaclite_env.py
scripts/train_dreamer_smaclite_phase1.py
scripts/evaluate_dreamer_smaclite.py

docs/running_dreamer_smaclite.md
```

Minimum Phase 1 files:

```text
src/smacdreamer/__init__.py
src/smacdreamer/envs/__init__.py
src/smacdreamer/envs/smaclite_dreamer_env.py
configs/smaclite_phase1.yaml
scripts/smoke_test_smaclite_env.py
docs/running_dreamer_smaclite.md
```

---

## 9. Planned Files to Modify

Avoid modifying external repositories.

Preferred approaches:

1. Add a local launcher that imports/registers the SMAClite adapter.
2. Add adapter code outside `external/`.
3. Use config-based import paths if DreamerV3 supports them.
4. Only patch DreamerV3 environment registration if no cleaner method exists.

Potential file that may require a minimal patch:

```text
external/dreamerv3/dreamerv3/main.py
```

This should only be modified if DreamerV3 does not support external environment registration cleanly.

If modified, document the patch in:

```text
docs/running_dreamer_smaclite.md
```

---

## 10. Implementation Phases

---

## Phase 1 — Single Fixed Scenario

### 10.1 Goal

Get DreamerV3 running on one fixed SMAClite scenario with fixed observation and action shapes.

This phase is for integration debugging only.

The goal is not strong performance.

---

### 10.2 Scope

Phase 1 includes:

1. one existing SMAClite scenario,
2. fixed allied unit count,
3. fixed enemy unit count,
4. fixed action dimension,
5. fixed observation shape,
6. no map sampling,
7. no padding,
8. no curriculum,
9. no custom units.

---

### 10.3 Recommended Phase 1 Scenario

Use a small existing SMAClite scenario.

Preferred starting point:

```text
2s3z
```

Alternative if `2s3z` is unavailable or unsuitable:

```text
3m
```

Avoid starting with larger or more complex maps like:

```text
MMM2
corridor
```

until the adapter has been proven.

---

### 10.4 Phase 1 Tasks

#### Task 1 — Inspect DreamerV3 Environment Interface

Document:

1. required base environment class,
2. required `obs_space`,
3. required `act_space`,
4. required `step()` behaviour,
5. reset signalling,
6. how `reward`, `is_first`, `is_last`, and `is_terminal` are expected.

Output:

```text
docs/running_dreamer_smaclite.md
```

---

#### Task 2 — Inspect SMAClite Environment Interface

Document:

1. how to create a SMAClite env,
2. how reset works,
3. how step works,
4. shape and type of observations,
5. shape and type of available actions,
6. action format expected by step,
7. reward type,
8. done/truncated behaviour,
9. useful info fields.

Output:

```text
docs/running_dreamer_smaclite.md
```

---

#### Task 3 — Implement SMAClite Dreamer Adapter

Create:

```text
src/smacdreamer/envs/smaclite_dreamer_env.py
```

The adapter should:

1. create a SMAClite environment,
2. reset the SMAClite environment,
3. convert SMAClite observations into DreamerV3 observations,
4. expose Dreamer-compatible observation spaces,
5. expose Dreamer-compatible action spaces,
6. accept one action per allied unit,
7. convert Dreamer actions to SMAClite joint actions,
8. handle invalid actions,
9. return correct reward,
10. return correct episode-boundary flags,
11. preserve useful info metrics where possible.

---

#### Task 4 — Implement Invalid Action Handling

For each allied unit:

1. check whether selected action is valid,
2. if valid, use it,
3. if invalid, increment invalid action counter,
4. replace with the first valid action,
5. continue environment step.

Track:

```text
invalid_action_count
total_action_count
invalid_action_rate
```

---

#### Task 5 — Implement Smoke Test

Create:

```text
scripts/smoke_test_smaclite_env.py
```

The smoke test should verify:

1. adapter can be imported,
2. SMAClite env can be created,
3. reset works,
4. observation keys exist,
5. observation shapes are stable,
6. action space exists,
7. random valid actions can be sampled,
8. step works,
9. one full episode can complete,
10. reward is numeric,
11. invalid action tracking works,
12. `terminated` and `truncated` are handled correctly.

The smoke test should print:

```text
scenario name
number of allied agents
number of actions
observation shape
available-action mask shape
episode return
episode length
battle_won if available
invalid action count
invalid action rate
```

---

#### Task 6 — Create Phase 1 Config

Create:

```text
configs/smaclite_phase1.yaml
```

The config should include:

```yaml
phase: 1
scenario: 2s3z
jax_platform: cpu
num_envs: 1
train_steps: 10000
logdir: logs/smaclite_phase1
invalid_action_fallback: first_valid
use_global_state: false
use_action_mask_observation: true
```

The exact config format should be adapted to DreamerV3 after inspecting its config system.

---

#### Task 7 — Integrate with DreamerV3

Preferred integration order:

1. try local launcher/import-based registration,
2. try config-based import path,
3. only patch DreamerV3 registry if needed.

If patching is required, keep it minimal.

The goal is to make a command like this possible:

```bash
python external/dreamerv3/dreamerv3/main.py \
  --configs smaclite_phase1 debug \
  --logdir logs/smaclite_phase1/debug
```

Exact command may differ depending on DreamerV3 config format.

---

#### Task 8 — Run CPU Debug Training

Run a short CPU-only DreamerV3 debug training job.

The run should verify:

1. adapter imports correctly,
2. environment creates correctly,
3. DreamerV3 collects experience,
4. replay buffer accepts observations/actions,
5. training loop starts,
6. logs are created,
7. checkpoint directory is created.

This is not expected to produce a good policy.

---

### 10.5 Phase 1 Acceptance Criteria

Phase 1 is complete only when:

1. a random rollout runs for at least one full episode,
2. reset/step work without shape errors,
3. observation shape is stable,
4. action space is stable,
5. DreamerV3 debug training starts,
6. replay buffer accepts data,
7. logs are created,
8. checkpoint directory is created,
9. reward is logged,
10. episode length is logged,
11. battle win/loss is logged if available,
12. invalid action count is logged,
13. invalid action rate is logged,
14. documentation is updated with run commands.

---

## Phase 1 Validation Results

### Status

Phase 1 has passed.

The SMACliteDreamerEnv adapter has been validated on the `2s3z` scenario, and DreamerV3 has successfully completed short CPU debug training runs.

### Validated Scenario

```text
scenario        : 2s3z
n_agents        : 5
n_enemies       : 5
n_actions       : 11
obs_size        : 80
state shape     : (400,)
avail_actions   : (55,)
act_space keys  : reset, action_0, action_1, action_2, action_3, action_4
```

### Smoke Test Results

The standalone adapter smoke test passed.

Validated:

- reset returns a valid first-step observation
- invalid actions do not crash the environment
- observation shapes remain stable
- one full episode completes
- sequential reset and second episode work correctly
- `log/` metrics are 0-dimensional `float32` scalars

### DreamerV3 Debug Training Results

The following debug training runs completed successfully:

```text
500 steps   : PASSED
5,000 steps : PASSED
10,000 steps: PASSED
```

Validated:

- DreamerV3 imports the SMAClite adapter
- DreamerV3 builds the agent using the adapter `obs_space` and `act_space`
- multi-head discrete actions work
- replay buffer accepts SMAClite observations/actions
- training loop runs
- `metrics.jsonl` is created
- `replay/` directory is created
- `ckpt/` directory is created
- no NaN/Inf values were observed in the checked metrics

### Confirmed Working Multi-Head Action Space

DreamerV3 successfully trained with separate discrete action heads:

```text
action_0
action_1
action_2
action_3
action_4
```

Training metrics confirmed action entropy was tracked for each head:

```text
train/ent/action_0
train/ent/action_1
train/ent/action_2
train/ent/action_3
train/ent/action_4
```

Therefore, the fallback to a flattened joint action space is not required for Phase 1.

### Confirmed Metric Keys

The implementation logs episode-level metrics:

```text
episode/score
episode/length
```

It also logs SMAClite-specific metrics through DreamerV3 `epstats`:

```text
epstats/log/battle_won/avg
epstats/log/battle_won/max
epstats/log/battle_won/sum

epstats/log/episode_invalid_action_count/avg
epstats/log/episode_invalid_action_count/max
epstats/log/episode_invalid_action_count/sum

epstats/log/episode_invalid_action_rate/avg
epstats/log/episode_invalid_action_rate/max
epstats/log/episode_invalid_action_rate/sum

epstats/log/episode_total_action_count/avg
epstats/log/episode_total_action_count/max
epstats/log/episode_total_action_count/sum
```

Note: the implementation uses the prefix `episode_invalid_action_*`, not `invalid_action_*`.

### Resolved Questions

| Question | Resolution |
|---|---|
| Does DreamerV3 support multiple discrete action heads? | Yes. Phase 1 training completed with `action_0` to `action_4`. |
| Is a flattened joint action fallback needed? | No for Phase 1. Multi-head actions work. |
| Does replay accept the adapter output? | Yes. Replay metrics are produced. |
| Are invalid-action metrics logged? | Yes, under `epstats/log/episode_invalid_action_*`. |
| Are battle outcome metrics logged? | Yes, under `epstats/log/battle_won/*`. |
| Can Phase 1 run on native Windows CPU? | Yes. 500, 5k, and 10k debug runs completed. |

### Known Issue Encountered

During the 5k run, the logger initially failed because the local Pillow/PIL installation was broken:

```text
ImportError: cannot import name '_imaging' from 'PIL'
```

This was a dependency/environment issue, not an adapter issue. The fix was to reinstall Pillow in the active Conda environment.

## Phase 1 Real-Rollout Masking Result

The real-rollout action masking patch was implemented using `SMACliteAgent`, a project-local subclass of DreamerV3's base `Agent`.

The patch masks unavailable actions in `policy()` by applying the SMAClite `avail_actions` observation to each discrete action head before sampling.

Files changed:

- `src/smacdreamer/agent.py`
- `scripts/train_dreamer_smaclite_phase1.py`
- `scripts/evaluate.py`
- `src/smacdreamer/envs/smaclite_dreamer_env.py`

Result:

```text
Before masking:
mean_invalid_action_rate ≈ 58%

After real-rollout masking:
mean_invalid_action_rate ≈ 2–3%
```
## Phase 1 Real-Rollout Masking Diagnostic Result

The real-rollout masking fix reduced invalid action rate from approximately 58% to 2–3%.

10-episode evaluation result:

```text
mean_episode_reward           : 3.9569
std_episode_reward            : 0.7734
mean_episode_length           : 43.0
win_rate                      : 0.000
mean_invalid_action_count     : 6.10
mean_invalid_action_rate      : 0.0294
mean_total_action_count       : 215.0
mean_was_prev_valid           : 6.10
mean_was_prev_invalid         : 0.00
mean_avail_mask_mismatch_slots: 101.60

Interpretation:

- mean_was_prev_invalid = 0.00 confirms that SMACliteAgent.policy() did not sample actions that were invalid under the mask it received.
- The remaining invalid actions were valid under the previously returned avail_actions mask but invalid when checked by the environment safety net.
- Therefore, the residual 2–3% invalid-action rate is caused by availability-mask timing/mismatch, not policy masking failure.
- _sanitise_actions() remains necessary as a hard safety net.

Conclusion:

Real-rollout action masking is successful and should remain enabled. The next unresolved issue is imagination-rollout masking inside DreamerV3 training.
```

## Phase 1B: Imagination-Rollout Action Masking

### Problem

After Phase 1A real-rollout masking, the policy no longer samples invalid actions during environment
interaction. However, DreamerV3's actor is trained through imagined rollouts inside `Agent.loss()`.
The original `policyfn` lambda (`lambda feat: sample(self.pol(..., 1))`) and the policy distribution
passed to `imag_loss()` (`self.pol(inp, 2)`) had no reference to `avail_actions`. The actor loss
gradient therefore flowed through imagined invalid actions on every training step.

### Fix

`SMACliteAgent.loss()` overrides `DreamerAgent.loss()` with exactly two targeted changes inside the
imagination section:

**Change 1 — masked policyfn (used by `dyn.imagine()` and `lastact`):**

```python
_img_avail = obs['avail_actions'][:, -K:, :].reshape((B * K, -1))  # (B*K, N*A)

def policyfn(feat):
    pol_raw = self.pol(self.feat2tensor(feat), 1)
    pol_masked = self._apply_avail_mask(pol_raw, _img_avail)
    return sample(pol_masked)
```

**Change 2 — masked policy distribution for `imag_loss()`:**

```python
_img_avail_broadc = jnp.broadcast_to(
    _img_avail[:, None, :], (B * K, H + 1, _img_avail.shape[-1]))
_pol_dist_masked = self._apply_avail_mask(self.pol(inp, 2), _img_avail_broadc)
# passed as the 4th positional argument to imag_loss (was: self.pol(inp, 2))
```

All other sections of `loss()` (world model, replay) are identical to the parent.

### Files changed

- `src/smacdreamer/agent.py`: imports extended; `loss()` method added; docstring updated

### Masking strategy

**Start-point constant masks.** For each imagined trajectory, the `avail_actions` at the real
observation that starts the trajectory (`obs["avail_actions"][:, -K:, :]`) is used as a constant
mask across all H imagination steps.

**Limitation:** Action availability changes during imagined rollouts — units die, move, change
range. The constant start-point mask is therefore an approximation. Future improvement: use the
world-model decoder to predict `avail_actions` at each imagined step and apply the predicted mask
dynamically.

### Shape invariants

| Variable | Shape |
|---|---|
| `obs["avail_actions"]` | `(B, T, 55)` |
| `_img_avail` | `(B*K, 55)` |
| policyfn input `feat` (scan step) | `(B*K, feat_dim)` |
| `inp` for imag_loss | `(B*K, H+1, feat_dim)` |
| `_img_avail_broadc` | `(B*K, H+1, 55)` |

### Sync warning

`SMACliteAgent.loss()` is a copy of `DreamerAgent.loss()`. If the upstream method changes, the
override must be updated manually. The two Phase 1B blocks are clearly marked with comments.

---

### Phase 1 Conclusion

Phase 1 is considered complete.

The adapter, smoke test, DreamerV3 training launcher, replay integration, metric logging, and multi-head action setup are working for the fixed `2s3z` scenario.

Before moving to Phase 2, complete:

1. checkpoint resume test
2. evaluation script
3. documentation update
4. final Phase 1 metric summary

---

## Phase 2 Entry Gate

Do not begin Phase 2 until the following are complete:

- Phase 1 smoke test passes
- 10k-step DreamerV3 debug run completes
- metrics are verified
- no NaN/Inf values are found
- checkpoint folder exists
- checkpoint resume is tested
- evaluation script works on the Phase 1 checkpoint

Phase 2 should only add same-shape multi-map support.

Do not add padding, variable unit counts, or large map datasets in Phase 2.

## Phase 2 — Multiple Same-Shape Maps

### 11.1 Goal

Train and evaluate across multiple SMAClite maps with the same observation and action shapes.

---

### 11.2 Scope

Phase 2 includes:

1. multiple maps,
2. same allied unit count,
3. same enemy unit count,
4. same action dimensions,
5. same observation dimensions,
6. no padding yet.

---

### 11.3 Tasks

1. Identify compatible same-shape SMAClite maps.
2. Create a map list config.
3. Implement map sampler or fixed map rotation.
4. Ensure observation/action shapes remain constant.
5. Log active map name.
6. Add train/eval map split.
7. Add per-map evaluation.

Create or extend:

```text
src/smacdreamer/envs/map_sampler.py
configs/smaclite_phase2.yaml
scripts/evaluate_dreamer_smaclite.py
```

---

### 11.4 Map Sampling Options

Supported options should include:

```text
fixed
round_robin
random
seeded_random
```

Recommended default for Phase 2:

```text
seeded_random
```

This allows reproducibility.

---

### 11.5 Phase 2 Acceptance Criteria

Phase 2 is complete when:

1. training can run across several same-shape maps,
2. map name is logged per episode,
3. train/eval map split is configurable,
4. evaluation can report metrics per map,
5. no shape mismatch occurs during replay or batching,
6. random seed controls map sampling.

---

## Phase 3 — Padded Multi-Map Curriculum

### 12.1 Goal

Support maps with different unit counts by padding observations and action masks to fixed maximum dimensions.

---

### 12.2 Scope

Phase 3 includes:

1. existing SMAClite example maps,
2. variable allied unit counts,
3. variable enemy unit counts where possible,
4. fixed maximum observation shape,
5. fixed maximum action shape,
6. valid-unit masks,
7. valid-action masks,
8. curriculum or map-family sampling.

No custom units are allowed.

---

### 12.3 Padding Design

The adapter should determine or be configured with:

```text
max_allied_units
max_enemy_units
max_actions
max_obs_dim_per_agent
max_state_dim
```

Smaller maps should be padded to these maximum values.

Recommended padded fields:

```python
{
    "state": padded_state,
    "avail_actions": padded_avail_actions,
    "agent_mask": agent_mask,
    "enemy_mask": enemy_mask,
}
```

`agent_mask` should indicate which allied unit slots are real.

Example:

```python
agent_mask = [1, 1, 1, 0, 0]
```

This means:

```text
3 real allied units
2 padded allied slots
```

Padded units must not be allowed to affect the environment.

---

### 12.4 Tasks

1. Identify variable-shape maps.
2. Compute or configure max dimensions.
3. Implement observation padding.
4. Implement available-action padding.
5. Implement allied unit mask.
6. Implement enemy unit mask if useful.
7. Ensure padded action slots are ignored.
8. Ensure replay buffer receives fixed-shape tensors.
9. Add map-family grouping.
10. Add curriculum schedule.
11. Add per-map-family evaluation.

Create or extend:

```text
src/smacdreamer/envs/padding.py
configs/smaclite_phase3.yaml
scripts/evaluate_dreamer_smaclite.py
```

---

### 12.5 Curriculum Options

Supported options may include:

```text
none
easy_to_hard
map_family_round_robin
seeded_random
performance_based
```

Recommended initial default:

```text
map_family_round_robin
```

Do not implement complex performance-based curriculum until basic padded training works.

---

### 12.6 Phase 3 Acceptance Criteria

Phase 3 is complete when:

1. smaller maps are padded correctly,
2. masks distinguish real units from padded units,
3. padded actions are ignored or replaced safely,
4. replay buffer receives fixed-shape tensors,
5. training can run across variable-size maps,
6. evaluation reports results by map family,
7. documentation explains padding assumptions and limitations.

---

## Phase 4 — Large Dataset Training

### 13.1 Goal

Train across a large dataset of SMAClite map configurations.

---

### 13.2 Scope

Phase 4 includes:

1. many generated map configs,
2. no custom units,
3. varied terrain,
4. varied unit compositions using existing units only,
5. reproducible train/eval split,
6. checkpoint resume,
7. cloud GPU training support.

---

### 13.3 Dataset Manifest

Create a dataset manifest file.

Possible location:

```text
configs/map_datasets/
```

Example manifest:

```yaml
dataset_name: smaclite_large_v1
seed: 42

train_maps:
  - map_path: maps/generated/train/map_0001.json
    family: balanced_mirror
  - map_path: maps/generated/train/map_0002.json
    family: slight_enemy_advantage

eval_maps:
  - map_path: maps/generated/eval/map_0001.json
    family: balanced_mirror
```

The manifest should support:

1. map path,
2. map family/category,
3. unit count metadata,
4. terrain metadata if available,
5. train/eval split,
6. random seed.

---

### 13.4 Tasks

1. Add dataset manifest format.
2. Add train/eval split support.
3. Add seed control.
4. Add map sampling strategy.
5. Add checkpoint resume instructions.
6. Add aggregate metrics.
7. Add per-map metrics.
8. Add per-map-family metrics.
9. Add cloud GPU training config.
10. Add long-run documentation.

Create or extend:

```text
configs/smaclite_phase4.yaml
configs/map_datasets/
docs/running_dreamer_smaclite.md
```

---

### 13.5 Phase 4 Acceptance Criteria

Phase 4 is complete when:

1. large map dataset can be loaded from config,
2. train/eval split is reproducible,
3. training can resume from checkpoint,
4. aggregate metrics are logged,
5. per-map metrics are logged,
6. per-map-family metrics are logged,
7. cloud GPU run instructions are documented.

---

## 14. Testing Plan

### 14.1 Smoke Test

Create:

```text
scripts/smoke_test_smaclite_env.py
```

The smoke test must verify:

1. adapter import,
2. environment creation,
3. reset,
4. observation keys,
5. observation shapes,
6. action space,
7. available-action mask,
8. valid random stepping,
9. full episode completion,
10. reward type,
11. termination/truncation handling,
12. invalid action count,
13. invalid action rate.

---

### 14.2 Adapter Test Checklist

Verify:

1. observation shape is stable,
2. action space matches allied unit count,
3. available-action mask shape is correct,
4. invalid actions are detected,
5. invalid actions are replaced safely,
6. invalid-action metrics are updated,
7. info metrics are preserved,
8. reset after done works,
9. multiple episodes can run sequentially.

---

### 14.3 DreamerV3 Integration Test Checklist

Verify:

1. DreamerV3 can import the adapter,
2. config loads correctly,
3. environment is created through DreamerV3,
4. DreamerV3 can reset the environment,
5. DreamerV3 can step the environment,
6. replay buffer accepts observations/actions,
7. debug training starts,
8. logs are written,
9. checkpoints are written.

---

### 14.4 Phase 2 Test Checklist

Verify:

1. map switching works,
2. same-shape assumption is enforced,
3. incompatible maps are rejected clearly,
4. map name is logged,
5. per-map evaluation works.

---

### 14.5 Phase 3 Test Checklist

Verify:

1. padding produces fixed shapes,
2. masks are correct,
3. padded units do not affect environment,
4. padded actions are ignored safely,
5. variable-size maps can be batched.

---

## 15. Logging Requirements

Log at minimum:

```text
episode reward
episode length
battle won/lost if available
map/scenario name
invalid action count
invalid action rate
number of allied units
number of enemy units
training step
checkpoint path
```

For multi-map training, also log:

```text
per-map episode reward
per-map win rate
per-map episode length
per-map invalid action rate
per-map-family reward
per-map-family win rate
```

For debugging, print:

```text
observation shape
action space
available-action mask shape
current map
current episode length
current episode return
```

---

## 16. Config Design

Create separate configs for each phase:

```text
configs/smaclite_phase1.yaml
configs/smaclite_phase2.yaml
configs/smaclite_phase3.yaml
configs/smaclite_phase4.yaml
```

The exact config format must match DreamerV3 after inspecting the codebase.

---

### 16.1 Example Phase 1 Config

```yaml
phase: 1

env:
  suite: smaclite
  scenario: 2s3z
  use_global_state: false
  use_action_mask_observation: true
  invalid_action_fallback: first_valid

jax:
  platform: cpu

run:
  num_envs: 1
  train_steps: 10000
  logdir: logs/smaclite_phase1
  seed: 42

debug:
  enabled: true
```

---

### 16.2 Example Phase 2 Config

```yaml
phase: 2

env:
  suite: smaclite
  map_sampling: seeded_random
  train_maps:
    - 2s3z
  eval_maps:
    - 2s3z
  require_same_shape: true
  use_global_state: false
  use_action_mask_observation: true
  invalid_action_fallback: first_valid

jax:
  platform: cpu

run:
  num_envs: 1
  train_steps: 50000
  logdir: logs/smaclite_phase2
  seed: 42
```

---

### 16.3 Example Phase 3 Config

```yaml
phase: 3

env:
  suite: smaclite
  map_sampling: map_family_round_robin
  use_padding: true
  max_allied_units: null
  max_enemy_units: null
  max_actions: null
  use_agent_mask: true
  use_enemy_mask: true
  use_action_mask_observation: true
  invalid_action_fallback: first_valid

jax:
  platform: cpu

run:
  num_envs: 1
  train_steps: 100000
  logdir: logs/smaclite_phase3
  seed: 42
```

---

### 16.4 Example Phase 4 Config

```yaml
phase: 4

env:
  suite: smaclite
  dataset_manifest: configs/map_datasets/smaclite_large_v1.yaml
  map_sampling: seeded_random
  use_padding: true
  use_agent_mask: true
  use_enemy_mask: true
  use_action_mask_observation: true
  invalid_action_fallback: first_valid

jax:
  platform: cuda

run:
  num_envs: 8
  train_steps: 1000000
  logdir: logs/smaclite_phase4
  seed: 42
  resume: true
```

---

## 17. Documentation Requirements

Create or update:

```text
docs/running_dreamer_smaclite.md
```

The documentation should include:

1. environment setup,
2. Windows CPU setup,
3. Python/Conda environment setup,
4. required dependencies,
5. how to run the smoke test,
6. how to run Phase 1 debug training,
7. how to run evaluation,
8. known limitations,
9. invalid action handling,
10. map compatibility rules,
11. padding rules for Phase 3,
12. cloud GPU notes for later.

---

## 18. Expected Commands

The exact commands may change after codebase inspection.

The final documentation should include working commands similar to the following.

---

### 18.1 Smoke Test

Windows CMD:

```cmd
set PYTHONPATH=%cd%\src;%cd%\external\dreamerv3;%cd%\external\smaclite
python scripts\smoke_test_smaclite_env.py --scenario 2s3z
```

PowerShell:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD\external\dreamerv3;$PWD\external\smaclite"
python scripts\smoke_test_smaclite_env.py --scenario 2s3z
```

Linux/WSL:

```bash
export PYTHONPATH=$PWD/src:$PWD/external/dreamerv3:$PWD/external/smaclite
python scripts/smoke_test_smaclite_env.py --scenario 2s3z
```

---

### 18.2 Phase 1 Debug Training

Windows CMD:

```cmd
set PYTHONPATH=%cd%\src;%cd%\external\dreamerv3;%cd%\external\smaclite
python scripts\train_dreamer_smaclite_phase1.py --config configs\smaclite_phase1.yaml
```

Linux/WSL:

```bash
export PYTHONPATH=$PWD/src:$PWD/external/dreamerv3:$PWD/external/smaclite
python scripts/train_dreamer_smaclite_phase1.py --config configs/smaclite_phase1.yaml
```

If using DreamerV3 directly, document the exact working DreamerV3 command after inspection.

---

### 18.3 Evaluation

```bash
python scripts/evaluate_dreamer_smaclite.py \
  --checkpoint checkpoints/smaclite_phase1/latest \
  --scenario 2s3z \
  --episodes 10
```

Exact path and flags should be updated after implementation.

---

## 19. Known Limitations

The initial implementation has expected limitations.

Document these clearly:

1. This is not a full MARL algorithm.
2. The DreamerV3 agent is a centralised controller.
3. Action masking is not initially enforced inside the actor distribution.
4. Invalid actions are replaced by the wrapper in Phase 1.
5. Phase 1 only supports fixed-shape training.
6. Multi-map training requires same shapes until padding is implemented.
7. Padding may introduce learning challenges.
8. Windows CPU debugging may be slow.
9. Cloud GPU training will require separate dependency validation.

---

## 20. Future Improvements

Potential future improvements:

1. proper action-mask-aware actor distribution,
2. recurrent/per-agent policy heads,
3. centralised critic with decentralised actors,
4. comparison against MAPPO/QMIX/VDN,
5. curriculum based on win rate,
6. automatic map difficulty estimation,
7. better per-map evaluation dashboard,
8. distributed environment workers,
9. cloud GPU training scripts,
10. experiment tracking with WandB or TensorBoard.

---

## 21. Open Questions

Before implementation, answer or resolve:

1. Which exact SMAClite scenario should Phase 1 use?
2. Does the installed SMAClite version expose `2s3z`?
3. Does SMAClite expose global state?
4. Should global state be included in Phase 1 observations?
5. Does DreamerV3 support dictionary action spaces cleanly?
6. Can DreamerV3 consume available-action masks without actor modification?
7. What is the cleanest way to register a new environment without modifying `external/dreamerv3`?
8. What metrics are exposed by SMAClite in `info`?
9. What is the best invalid-action fallback?
10. Should evaluation use deterministic actions, stochastic actions, or both?
11. Should Phase 2 use random map sampling or round-robin map rotation?
12. Where should checkpoints be stored?
13. Should cloud GPU training use the same repo structure or a separate deployment branch?

Recommended default answers unless overridden:

```text
Phase 1 scenario: 2s3z
Use global state in Phase 1: false
Invalid-action fallback: first_valid
Evaluation policy: deterministic if supported, otherwise stochastic
Phase 2 sampling: seeded_random
Checkpoints: checkpoints/
```

---

## 22. Coding Agent Workflow

The coding agent must follow this workflow.

---

### 22.1 Step 1 — Analysis First

Before editing files, inspect:

```text
external/dreamerv3
external/smaclite
```

Then produce:

1. relevant DreamerV3 files/functions/classes,
2. relevant SMAClite files/functions/classes,
3. compatibility issues,
4. proposed adapter design,
5. exact files to create,
6. exact files to modify,
7. unresolved questions.

Do not edit files during this step.

---

### 22.2 Step 2 — Phase 1 Implementation Only

After analysis, implement only Phase 1.

Do not implement Phase 2, Phase 3, or Phase 4 yet.

Phase 1 implementation must include:

1. adapter,
2. smoke test,
3. config,
4. debug training command,
5. documentation update.

---

### 22.3 Step 3 — Run Smoke Test

Run the smoke test.

If it fails:

1. identify root cause,
2. fix the minimal issue,
3. rerun the test,
4. document the fix.

---

### 22.4 Step 4 — Run DreamerV3 Debug Training

Run a short DreamerV3 debug training job.

If it fails:

1. identify whether the issue is from DreamerV3, SMAClite, adapter, config, dependency, or Windows environment,
2. fix the minimal issue,
3. rerun,
4. document the fix.

---

### 22.5 Step 5 — Update Documentation

After Phase 1 works, update:

```text
docs/running_dreamer_smaclite.md
docs/implementation_plan.md
```

Record:

1. working scenario,
2. working command,
3. final adapter interface,
4. known issues,
5. next steps for Phase 2.

---

## 23. Final Instruction to Coding Agent

Implement this project incrementally.

Do not jump directly to Phase 3 or Phase 4.

First complete:

1. codebase analysis,
2. Phase 1 adapter design,
3. Phase 1 implementation,
4. Phase 1 smoke test,
5. Phase 1 DreamerV3 debug training,
6. Phase 1 documentation.

After Phase 1 passes, update this plan with what was learned before proceeding to Phase 2.

---

## 24. Phase 2 — Same-Shape Multi-Map Support

### 24.1 Context

Phase 1 is complete. The adapter works on `2s3z` with real-rollout masking (Phase 1A) and
imagination-rollout masking (Phase 1B). Entropy comparison confirmed that imagination masking
is active (action head entropy dropped from ln(11)=2.3979 nats to 0.82–1.38 nats).

Phase 2 adds same-shape multi-map rotation: the agent trains across several maps that share
identical (n_agents, n_enemies, n_actions, obs_size) so the replay buffer and model weights
require no changes.

### 24.2 Key Finding: No Compatible Built-in Maps

All 12 SMAClite built-in maps have different shapes from 2s3z. Two custom map variants were
created:

| Map | Terrain | Ally start | Enemy start |
|-----|---------|-----------|-------------|
| `2s3z` | SIMPLE | (9, 16) | (23, 16) |
| `2s3z_v2` | RAVINE | (12, 16) | (20, 16) |
| `2s3z_v3` | CORRIDOR | (9, 16) | (23, 16) |

All three: STALKER:2 + ZEALOT:3 each side. n_agents=5, n_enemies=5, n_actions=11, obs_size=80.

### 24.3 Files Created

```text
configs/maps/2s3z_v2.json                        custom map — RAVINE terrain, closer starts
configs/maps/2s3z_v3.json                        custom map — CORRIDOR terrain
configs/maps/phase2_manifest.yaml                manifest listing all 3 maps
configs/smaclite_phase2.yaml                     Phase 2 training config
src/smacdreamer/envs/map_sampler.py              MapEntry + MapSampler (fixed/round_robin/seeded_random)
scripts/inspect_maps.py                          enumerate map shape profiles
scripts/smoke_test_phase2.py                     Phase 2 smoke test
scripts/train_dreamer_smaclite_phase2.py         Phase 2 training launcher
scripts/evaluate_phase2.py                       per-map evaluation
```

### 24.4 Files Modified

```text
src/smacdreamer/envs/smaclite_dreamer_env.py    optional map_sampler param; _open_env(); log/map_id
```

### 24.5 Design Decisions

- Custom maps loaded via `SMACliteEnv(map_file=path)` — no modification to external/smaclite.
- `MapSampler.peek()` used during `__init__` to configure the initial env without consuming
  an episode slot. `_reset()` always calls `sampler.next()` — first reset uses map[0].
- `log/map_id` added as scalar float32 (map index in manifest). Filtered from model inputs
  by the standard `log/` prefix convention. String map names stay outside Dreamer observations.
- Shape validation in `_reset()` checks (n_agents, n_enemies, n_actions, obs_size) and
  reports expected state/avail_actions shapes and action key range in the error message.
- Phase 1 adapter behaviour is unchanged when `map_sampler=None`.

### 24.6 Masking Preservation

- Real-rollout masking (`SMACliteAgent.policy()`) unchanged.
- Imagination-rollout masking (`SMACliteAgent.loss()`) unchanged.
- `_sanitise_actions()` unchanged.

### 24.7 Phase 2 Limitation

The imagination masking uses start-point constant masks (same approximation as Phase 1B).
Per-map mask distributions differ only in geometry/terrain, not unit composition, so the
approximation quality is identical across Phase 2 maps.


The priority is a correct, inspectable, reproducible training pipeline.
