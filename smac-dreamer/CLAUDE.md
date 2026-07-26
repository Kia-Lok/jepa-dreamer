# custom-smac Project Instructions

## Project Goal

This project implements a training pipeline for DreamerV3 on the SMAClite simulator.

The upstream repositories are located in:

- `external/dreamerv3`
- `external/smaclite`

DreamerV3 should be trained on SMAClite by treating the SMAClite multi-agent environment as a single-agent centralised-control problem.

One DreamerV3 agent controls all allied SMAClite units as a centralised controller.

## Core Constraints

- Do not create or use custom SMAClite units.
- Use only existing SMAClite unit types and example scenarios unless explicitly instructed otherwise.
- Do not directly modify files inside `external/dreamerv3` or `external/smaclite` unless absolutely necessary.
- Prefer adding project-specific code under `src/`, `adapters/`, `envs/`, `scripts/`, or `configs/`.
- If an external repository must be patched, explain why before editing and keep the change minimal.
- Prioritise correctness, debuggability, and clean interfaces over training speed.
- Initial development target is native Windows with JAX CPU only.
- Final implementation should remain portable to a cloud GPU environment.

## Environment Formulation

SMAClite should be exposed to DreamerV3 as a single centralised-control environment.

Expected formulation:

- Observation includes flattened per-agent observations.
- Observation includes available-action masks.
- Observation may include global state if available and useful.
- Action represents one discrete action per allied unit.
- Reward uses the shared SMAClite team reward.
- Episode termination must correctly handle both `terminated` and `truncated`.
- Useful info metrics such as `battle_won`, episode length, map name, and invalid action count should be preserved and logged where possible.

For Phase 1, invalid actions may be replaced with a valid fallback action, but invalid-action counts must be logged.

## Development Phases

### Phase 1 — Single Fixed Scenario

Use one existing SMAClite scenario with fixed unit composition.

Acceptance criteria:

- Random rollout works for at least one full episode.
- DreamerV3 debug training starts without crashing.
- Logs/checkpoints are created.
- Reward, episode length, win/loss, and invalid-action metrics are logged.

### Phase 2 — Multiple Same-Shape Maps

Support multiple maps with the same unit counts and action/observation shapes.

Acceptance criteria:

- Training can rotate or sample between compatible maps.
- Map name/source is logged.
- Evaluation can be run per map.

### Phase 3 — Padded Multi-Map Curriculum

Support maps with different unit counts using padding and masks.

Acceptance criteria:

- Smaller maps are padded to fixed maximum dimensions.
- Masks identify valid agents, units, and actions.
- Evaluation reports results by map family.

### Phase 4 — Large Dataset Training

Support training across a large set of SMAClite map configurations.

Acceptance criteria:

- Configurable train/eval map split.
- Reproducible map sampling with seeds.
- Aggregate and per-map metrics are logged.
- Training can resume from checkpoint.

## Coding Agent Workflow

Before major implementation:

1. Inspect the relevant DreamerV3 environment registration, config, training, logging, and evaluation code.
2. Inspect the relevant SMAClite reset, step, observation, action, reward, available-action, and scenario-loading code.
3. Produce an implementation plan.
4. List files to create and files to modify.
5. Ask for clarification if a major design choice is ambiguous.

Do not jump directly into Phase 3 or Phase 4 before Phase 1 works.