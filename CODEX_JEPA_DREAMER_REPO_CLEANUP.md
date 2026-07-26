# Codex Instructions: Clean the Combined JEPA–Dreamer Repository Without Breaking Training

## Mission

Clean the unzipped `combined-jepa-dreamer` codebase so that it no longer contains old experiment bundles, backups, generated logs, datasets, checkpoints, replay buffers, evaluation outputs, stale run-pointer files, or duplicated source trees.

The cleaned repository **must still contain working code and launchers for all of the following**:

1. Training a fresh **Exp-40 JEPA checkpoint**.
2. Training and evaluating a fresh **Exp-45 forecast checkpoint** initialized from an Exp-40 checkpoint.
3. Training the current **R2-Dreamer ordinary actor-critic baseline**, specifically the Tactical-v1.2 H=15 / 800k comparison workflow.
4. Training the current **Option-Critic implementation**, specifically **Option-Critic V9 anchor-safe, 8 slots, H=15 / 800k**.
5. Running ordinary R2-Dreamer multimap training and evaluation.
6. Validating JEPA–Dreamer checkpoint compatibility and retained Option-Critic behavior.

Do not treat filenames containing old experiment numbers as automatically obsolete. Some current workflows deliberately depend on older-named files.

---

# 1. Non-negotiable rules

## 1.1 Do not delete first and debug later

Before deleting anything:

1. Inventory the entire local tree.
2. Trace every protected launcher recursively.
3. Record Python imports, shell-script calls, config references, test references, and relative path assumptions.
4. Refactor protected launchers away from deleted runtime files.
5. Run static validation.
6. Only then delete candidates.

The final cleanup must be driven by actual references in the local unzipped tree, not by filename guessing.

## 1.2 All configs must remain

Preserve every configuration file under:

```text
smac-dreamer/configs/
```

This includes:

- active configs;
- old experiment configs;
- ablation configs;
- map configs;
- files with `tmp`, `backup`, or old experiment numbers in their names;
- YAML, YML, JSON, and map configuration files.

The request is to remove old code bundles, logs, data, and generated artifacts—not to discard experiment configurations.

Also preserve JEPA split/manifests and configuration-like metadata required to point at externally supplied datasets, including:

```text
smac-jepa-wm/splits/
smac-jepa-wm/configs/        # if present
smac-jepa-wm/pyproject.toml
smac-jepa-wm/setup.cfg       # if present
smac-jepa-wm/setup.py        # if present
```

A split manifest is configuration, not a bundled training dataset.

## 1.3 Delete all bundled runtime artifacts

The final source repository must not retain historical:

- logs;
- W&B run directories;
- checkpoints;
- replay/memmap buffers;
- collected datasets;
- generated NumPy arrays;
- generated evaluation outputs;
- generated plots;
- smoke outputs;
- archived run directories;
- copied run metadata;
- root `CURRENT_*.txt` pointers;
- audit logs;
- `.zip`, `.tar`, `.tar.gz`, or installer bundles;
- backup source trees;
- `preserve_before_*` trees;
- generated caches.

These artifacts may be created again at runtime, but none of the old copies should remain in the cleaned source tree.

## 1.4 External prerequisites are allowed and required

Because bundled datasets, checkpoints, logs, and run metadata are being deleted, some workflows cannot be completely self-contained.

The retained code must accept explicit external paths for prerequisites:

- Exp-40 requires an external JEPA dataset and manifest.
- Exp-45 requires an external Exp-40 checkpoint.
- R2-Dreamer requires an external Exp-40 JEPA checkpoint.
- The H=15 actor-critic comparison requires an external compatible Tactical-v1.2 source checkpoint.
- Option-Critic V9 requires an external compatible Tactical-v1.2 source checkpoint.

Do not preserve old checkpoints merely to make hard-coded defaults work. Refactor the launchers instead.

## 1.5 No hard-coded historical workspace paths

Remove defaults tied to historical machines or runs, such as:

```text
$HOME/workspace/dreamer/combined-upload
runs/rnn_seqmem_exp40_event_balanced_5ep_20260709_083104/checkpoint.pt
CURRENT_TACTICAL_V1_2_RUN.txt
CURRENT_UNIFIED_PRIORITY_RUN.txt
```

Every protected launcher must determine its repository root from its own script location and accept explicit prerequisite paths.

## 1.6 Do not run a long training job during cleanup

Validation should use:

- `--help`;
- import checks;
- shell syntax checks;
- config loading;
- unit tests;
- static audits;
- checkpoint-free preflight modes;
- tiny/synthetic smoke tests only where already supported.

Do not launch 800k, 2M, or full JEPA training.

---

# 2. First actions: inventory and safety snapshot

Run from the root of the unzipped repository.

```bash
set -euo pipefail

ROOT="$PWD"

find "$ROOT" -print | sort > cleanup_inventory_before.txt

find "$ROOT" -type f -printf '%s\t%p\n' 2>/dev/null \
  | sort -nr \
  > cleanup_file_sizes_before.txt || true

find "$ROOT" -type f \( \
    -name '*.py' -o \
    -name '*.sh' -o \
    -name '*.yaml' -o \
    -name '*.yml' -o \
    -name '*.json' -o \
    -name '*.toml' \
  \) -print \
  | sort \
  > cleanup_code_and_config_before.txt
```

If the ZIP retained a `.git` directory, create a branch:

```bash
git switch -c cleanup/retain-four-training-workflows
```

If there is no `.git`, do not assume Git commands are available. Work directly on the extracted copy and create:

```text
CLEANUP_REPORT.md
RETAINED_WORKFLOWS.md
```

The final report must list:

- deleted paths;
- moved files;
- modified launchers;
- protected workflow tests;
- any unresolved external prerequisites.

---

# 3. Protected workflow A: Exp-40 JEPA training

## 3.1 Required canonical entrypoint

Preserve:

```text
smac-jepa-wm/scripts/run_exp40_dreamer_event_balanced.sh
```

The launcher must continue to invoke:

```text
python -m smac_jepa.train_jepa_exp40_dreamer
```

## 3.2 Required Exp-40 source files

At minimum, preserve:

```text
smac-jepa-wm/smac_jepa/train_jepa_exp40_dreamer.py
smac-jepa-wm/smac_jepa/train_jepa_exp31_exp35.py
smac-jepa-wm/smac_jepa/anchored_belief_memory.py
smac-jepa-wm/scripts/validate_exp33_dreamer_checkpoint.py
```

`train_jepa_exp40_dreamer.py` is a thin compatibility layer over
`train_jepa_exp31_exp35.py`. The older-named base trainer is therefore a live
dependency and must not be deleted.

Preserve the core JEPA package and anything imported by the base trainer:

```text
smac-jepa-wm/smac_jepa/__init__.py
smac-jepa-wm/smac_jepa/modules/
smac-jepa-wm/smac_jepa/utils/
smac-jepa-wm/smac_jepa/config.py
smac-jepa-wm/smac_jepa/decoder.py
smac-jepa-wm/smac_jepa/jepa.py
smac-jepa-wm/smac_jepa/presets.py
smac-jepa-wm/smac_jepa/splits.py
```

Also preserve any additional package file that is imported directly or
indirectly by `train_jepa_exp31_exp35.py`.

Do not delete `smac-jepa-wm/smac_jepa/data/` if it is Python source code.
Only delete a top-level generated dataset directory such as
`smac-jepa-wm/data/`.

## 3.3 Exp-40 launcher portability changes

Update `run_exp40_dreamer_event_balanced.sh` so that:

1. It resolves `JEPA_ROOT` from the script location.
2. It uses `${PYTHON:-python}` or an explicitly supplied `PY`.
3. It accepts an explicit `MANIFEST`.
4. It accepts an explicit `OUT_DIR`.
5. It creates runtime output only under the selected `OUT_DIR`.
6. It does not rely on historical run folders.
7. It prints all resolved paths before launching.
8. It fails clearly if the manifest or referenced dataset is unavailable.
9. It retains checkpoint validation using
   `scripts/validate_exp33_dreamer_checkpoint.py`.

Recommended interface:

```bash
cd smac-jepa-wm

MANIFEST=/absolute/path/to/r2_general_2100.json \
OUT_DIR=/absolute/path/to/new_exp40_run \
PYTHON=/absolute/path/to/python \
WANDB=0 \
bash scripts/run_exp40_dreamer_event_balanced.sh
```

The launcher may retain a repository-relative manifest default, but the default
must be configuration only and must not imply that dataset shards are bundled.

## 3.4 Exp-40 acceptance checks

```bash
cd smac-jepa-wm

python -m smac_jepa.train_jepa_exp40_dreamer --help
python -m py_compile \
  smac_jepa/train_jepa_exp40_dreamer.py \
  smac_jepa/train_jepa_exp31_exp35.py \
  smac_jepa/anchored_belief_memory.py \
  scripts/validate_exp33_dreamer_checkpoint.py

bash -n scripts/run_exp40_dreamer_event_balanced.sh
```

Also verify the wrapper still contains and resolves:

```text
from . import train_jepa_exp31_exp35 as _base
AnchoredActionConditionedEntityRolloutGRUMemory
```

---

# 4. Protected workflow B: Exp-45 forecast training and evaluation

## 4.1 Required canonical launchers

Preserve:

```text
smac-dreamer/scripts/run_exp45_full_train_eval_resilient.sh

smac-jepa-wm/scripts/run_exp45_pow2_direct_train.sh
smac-jepa-wm/scripts/eval_exp45_pow2_all.sh
smac-jepa-wm/scripts/eval_exp45_pow2_ordinary.sh
smac-jepa-wm/scripts/eval_exp45_pow2_hidden.sh
smac-jepa-wm/scripts/smoke_exp45_pow2_direct.sh
smac-jepa-wm/scripts/static_audit_exp45_pow2.sh
```

## 4.2 Required forecast source and tools

Preserve:

```text
smac-jepa-wm/smac_jepa/train_jepa_exp45_pow2_direct.py
smac-jepa-wm/smac_jepa/pow2_direct_predictor.py
smac-jepa-wm/tools/audit_exp45_pow2_checkpoint.py
smac-jepa-wm/tools/eval_pow2_direct.py
smac-jepa-wm/tests/test_pow2_checkpoint_sanitizer.py
smac-jepa-wm/tests/test_pow2_direct_predictor.py
```

Preserve every additional imported Exp-40/JEPA dependency discovered from these
files.

## 4.3 Remove bundle installation behavior

`run_exp45_full_train_eval_resilient.sh` currently contains a fallback that
attempts to install Exp-45 from:

```text
exp45_pow2_direct_from_exp40_bundle.zip
```

Remove that fallback entirely.

The canonical forecast implementation already lives in the repository. The
cleaned launcher must:

1. Verify the trainer and stage scripts exist.
2. Fail with a direct source-file error if they do not.
3. Never unzip or install a bundle at runtime.
4. Never depend on `BUNDLE_ZIP`.
5. Derive `ROOT`, `JEPA_ROOT`, and `VENV` from script location or explicit
   environment overrides.
6. Require an explicit Exp-40 checkpoint path unless a valid path is supplied
   via a CLI/config override.

Remove the hard-coded default:

```text
runs/rnn_seqmem_exp40_event_balanced_5ep_20260709_083104/checkpoint.pt
```

Recommended interface:

```bash
ROOT=/absolute/path/to/cleaned-repo \
JEPA_ROOT=/absolute/path/to/cleaned-repo/smac-jepa-wm \
VENV=/absolute/path/to/cleaned-repo/.venv \
EXP40_CHECKPOINT=/absolute/path/to/exp40/checkpoint.pt \
MANIFEST=/absolute/path/to/r2_general_2100.json \
PIPE_DIR=/absolute/path/to/new_forecast_pipeline \
WANDB=0 \
bash smac-dreamer/scripts/run_exp45_full_train_eval_resilient.sh
```

## 4.4 Forecast acceptance checks

```bash
cd smac-jepa-wm

python -m smac_jepa.train_jepa_exp45_pow2_direct --help

python -m py_compile \
  smac_jepa/train_jepa_exp45_pow2_direct.py \
  smac_jepa/pow2_direct_predictor.py \
  tools/audit_exp45_pow2_checkpoint.py \
  tools/eval_pow2_direct.py

bash -n scripts/run_exp45_pow2_direct_train.sh
bash -n scripts/eval_exp45_pow2_all.sh
bash -n scripts/eval_exp45_pow2_ordinary.sh
bash -n scripts/eval_exp45_pow2_hidden.sh
bash -n scripts/smoke_exp45_pow2_direct.sh
bash -n scripts/static_audit_exp45_pow2.sh

pytest -q \
  tests/test_pow2_checkpoint_sanitizer.py \
  tests/test_pow2_direct_predictor.py
```

Then:

```bash
cd ../smac-dreamer
bash -n scripts/run_exp45_full_train_eval_resilient.sh
```

Search for stale bundle dependencies:

```bash
rg -n 'BUNDLE_ZIP|exp45_pow2_direct_from_exp40_bundle' \
  smac-dreamer smac-jepa-wm
```

The command must return no active launcher dependency.

---

# 5. Protected workflow C: ordinary R2-Dreamer actor-critic

There are two meanings that must remain supported:

1. Generic R2-Dreamer multimap training from a config.
2. The current Tactical-v1.2 ordinary actor-critic H=15 / 800k controlled
   comparison.

## 5.1 Preserve generic R2-Dreamer training and evaluation

Preserve:

```text
smac-dreamer/scripts/train_r2dreamer_smaclite_multimap.py
smac-dreamer/scripts/train_r2dreamer_smaclite_debug.py
smac-dreamer/scripts/evaluate_multimap.py
smac-dreamer/scripts/preflight_jepa_training.py
smac-dreamer/scripts/inspect_jepa_checkpoint.py
smac-dreamer/scripts/validate_jepa_r2_integration.py
smac-dreamer/scripts/validate_jepa_token_parity.py
smac-dreamer/scripts/debug_build_one_env.py
smac-dreamer/scripts/smoke_test_gym_smaclite_env.py
```

`train_r2dreamer_smaclite_multimap.py` imports `make_config` from
`train_r2dreamer_smaclite_debug.py`; both are live dependencies.

Preserve entirely:

```text
smac-dreamer/src/
smac-dreamer/external/r2dreamer/
smac-dreamer/external/smaclite/
smac-dreamer/tests/
smac-dreamer/configs/
```

Do not try to aggressively clean inside these canonical source directories in
this pass. Current Tactical and Option-Critic code is integrated into the
R2-Dreamer hot path, and broad source deletion is too risky.

Delete only generated runtime folders nested inside `external/r2dreamer`, such
as an actual `runs/` directory, while preserving source, docs, package metadata,
and configs.

## 5.2 Preserve the current actor-critic comparison files

Preserve:

```text
smac-dreamer/scripts/run_actor_critic_h15_800k.sh
smac-dreamer/scripts/static_audit_actor_critic_h15_800k.sh
smac-dreamer/scripts/audit_actor_critic_h15_800k.py

smac-dreamer/configs/r2_2100_jepa_tactical_mixture_v1_2.yaml
smac-dreamer/configs/r2_2100_jepa_tactical_mixture_v1_2_actor_critic_h15_800k.yaml
```

Also preserve the Tactical-v1.2 implementation and test:

```text
smac-dreamer/external/r2dreamer/tactical_policy.py
smac-dreamer/external/r2dreamer/dreamer.py
smac-dreamer/tests/test_tactical_policy_v1_2.py
```

## 5.3 Preserve the Tactical-v1.2 source-checkpoint workflow

The H=15 actor-critic and Option-Critic V9 both consume a Tactical-v1.2 source
checkpoint. Preserve the ability to train or continue this source architecture:

```text
smac-dreamer/scripts/run_tactical_v1_2_2m.sh
smac-dreamer/scripts/static_audit_tactical_v1_2.sh
smac-dreamer/scripts/audit_tactical_v1_2.py
smac-dreamer/scripts/assert_tactical_v1_2_metrics.py
```

However, refactor `run_tactical_v1_2_2m.sh` so it no longer depends on:

```text
CURRENT_UNIFIED_PRIORITY_RUN.txt
ADAPTIVE_RUN/best_val_macro_winrate.pt
ADAPTIVE_RUN/run_meta.json
```

Use explicit inputs:

```text
SOURCE_CHECKPOINT
SOURCE_RUN_META       # optional after semantic checkpoint validation is added
JEPA_CHECKPOINT
```

A compatible non-tactical source checkpoint is an external prerequisite.

## 5.4 Actor-critic launcher portability changes

`run_actor_critic_h15_800k.sh` currently reads:

```text
CURRENT_TACTICAL_V1_2_RUN.txt
```

and assumes:

```text
best_val_macro_winrate.pt
run_meta.json
```

inside an old log directory.

Refactor it so:

1. `SOURCE_CHECKPOINT` is required.
2. `SOURCE_RUN_META` is optional.
3. `JEPA_CHECKPOINT` can override
   `world_model.jepa.checkpoint` through the trainer's existing
   `--jepa-checkpoint` argument.
4. `EXPECTED_SOURCE_CHECKPOINT_SHA256` is optional.
5. If an expected hash is provided, enforce it.
6. If no hash is provided, semantically validate the checkpoint:
   - contains `agent_state_dict`;
   - contains Tactical-v1.2 metadata;
   - has exactly two tactics;
   - does not already contain `hierarchical_options.*`;
   - is not empty/corrupt.
7. The launcher does not require a `CURRENT_*.txt` file.
8. It writes new outputs only to a newly selected `RUN_DIR`.
9. It still enforces exactly 800,000 new environment steps for the comparison
   launcher.
10. It still uses:
    `configs/r2_2100_jepa_tactical_mixture_v1_2_actor_critic_h15_800k.yaml`.

Recommended interface:

```bash
cd smac-dreamer

SOURCE_CHECKPOINT=/absolute/path/to/tactical_v1_2_checkpoint.pt \
JEPA_CHECKPOINT=/absolute/path/to/exp40_checkpoint.pt \
RUN_DIR=/absolute/path/to/new_actor_critic_run \
FINAL_STEP=800000 \
bash scripts/run_actor_critic_h15_800k.sh
```

## 5.5 Actor-critic acceptance checks

```bash
cd smac-dreamer

python scripts/train_r2dreamer_smaclite_multimap.py --help
python scripts/evaluate_multimap.py --help
python scripts/preflight_jepa_training.py --help

python -m py_compile \
  scripts/train_r2dreamer_smaclite_multimap.py \
  scripts/train_r2dreamer_smaclite_debug.py \
  scripts/evaluate_multimap.py \
  scripts/audit_actor_critic_h15_800k.py \
  external/r2dreamer/dreamer.py \
  external/r2dreamer/tactical_policy.py

bash -n scripts/run_actor_critic_h15_800k.sh
bash -n scripts/static_audit_actor_critic_h15_800k.sh
bash -n scripts/run_tactical_v1_2_2m.sh
bash -n scripts/static_audit_tactical_v1_2.sh

pytest -q tests/test_tactical_policy_v1_2.py
```

Add a checkpoint-free static mode to both actor-critic and Tactical-v1.2 audits,
for example:

```bash
SKIP_CHECKPOINT_AUDIT=1 \
bash scripts/static_audit_actor_critic_h15_800k.sh

SKIP_CHECKPOINT_AUDIT=1 \
bash scripts/static_audit_tactical_v1_2.sh
```

The static mode must still validate:

- source syntax;
- config existence and invariants;
- required code tokens;
- relevant unit tests;
- launcher syntax.

It should skip only external checkpoint inspection.

---

# 6. Protected workflow D: current Option-Critic V9

The current retained Option-Critic is:

```text
Option-Critic V9 anchor-safe
8 slots
imagination horizon 15
800,000 new environment steps
```

Do not replace it with an older V2–V6 implementation.

## 6.1 Required Option-Critic V9 launcher and audit files

Preserve:

```text
smac-dreamer/scripts/run_option_critic_v9_anchor_safe_800k.sh
smac-dreamer/scripts/static_audit_option_critic_v9_anchor_safe.sh
smac-dreamer/scripts/audit_option_critic_v9_anchor_safe.py
smac-dreamer/scripts/assert_option_critic_v9_metrics.py
smac-dreamer/scripts/check_option_critic_win_guard.py
```

Preserve the current config:

```text
smac-dreamer/configs/r2_2100_jepa_option_critic_8_v9_anchor_safe_h15_800k.yaml
```

All other configs also remain because of the global config-preservation rule.

## 6.2 Required integrated R2-Dreamer Option-Critic source

Preserve at minimum:

```text
smac-dreamer/external/r2dreamer/dreamer.py
smac-dreamer/external/r2dreamer/trainer.py
smac-dreamer/external/r2dreamer/tools.py
smac-dreamer/external/r2dreamer/tactical_policy.py
smac-dreamer/external/r2dreamer/hierarchical_options.py
smac-dreamer/external/r2dreamer/hierarchical_dreamer.py
smac-dreamer/external/r2dreamer/option_critic.py

smac-dreamer/src/smacdreamer/validation_trainer.py
smac-dreamer/scripts/train_r2dreamer_smaclite_multimap.py
```

The current Option-Critic code is not isolated to one new module. It is
integrated into Dreamer acting, learning, target updates, checkpoint migration,
gradient guards, and validation.

## 6.3 Required Option-Critic tests

Preserve:

```text
smac-dreamer/tests/test_option_critic_v9_core.py
smac-dreamer/tests/test_option_critic_v9_migration.py
smac-dreamer/tests/test_option_critic_v9_auxiliary.py
```

Also preserve the legacy hierarchy/math tests still used by the V9 static audit:

```text
smac-dreamer/tests/test_hierarchical_options.py
smac-dreamer/tests/test_option_critic_math.py
smac-dreamer/tests/test_hierarchical_auxiliary.py
smac-dreamer/tests/test_hierarchy_migration.py
```

Do not delete these merely because their filenames do not contain `v9`.

## 6.4 Option-Critic launcher portability changes

`run_option_critic_v9_anchor_safe_800k.sh` currently depends on:

```text
CURRENT_TACTICAL_V1_2_RUN.txt
best_val_macro_winrate.pt
run_meta.json
a hard-coded source checkpoint hash
```

Refactor it so:

1. `SOURCE_CHECKPOINT` is required.
2. `SOURCE_RUN_META` is optional.
3. `JEPA_CHECKPOINT` can override the JEPA path through
   `--jepa-checkpoint`.
4. `EXPECTED_SOURCE_CHECKPOINT_SHA256` is optional.
5. An explicitly supplied hash remains enforceable for exact experiment
   reproduction.
6. Without a supplied hash, the audit semantically validates:
   - checkpoint is loadable;
   - `agent_state_dict` exists;
   - architecture metadata is Tactical-v1.2;
   - exactly two source tactics exist;
   - no `hierarchical_options.*` parameters are already present;
   - source is not itself an Option-Critic checkpoint.
7. New outputs go only to a fresh `RUN_DIR`.
8. The launcher does not read or write any root `CURRENT_*.txt`.
9. The launcher continues to enforce `FINAL_STEP=800000`.
10. The launcher continues to use:
    `r2_2100_jepa_option_critic_8_v9_anchor_safe_h15_800k.yaml`.

Recommended interface:

```bash
cd smac-dreamer

SOURCE_CHECKPOINT=/absolute/path/to/tactical_v1_2_checkpoint.pt \
JEPA_CHECKPOINT=/absolute/path/to/exp40_checkpoint.pt \
RUN_DIR=/absolute/path/to/new_option_critic_v9_run \
FINAL_STEP=800000 \
bash scripts/run_option_critic_v9_anchor_safe_800k.sh
```

## 6.5 Make source run metadata optional

Update:

```text
scripts/audit_option_critic_v9_anchor_safe.py
scripts/static_audit_option_critic_v9_anchor_safe.sh
```

Current behavior requires `SOURCE_RUN_META`.

New behavior:

- If `SOURCE_RUN_META` is provided:
  - validate that it exists;
  - validate JSON;
  - record it as lineage;
  - optionally check that it shares a directory with the source checkpoint.
- If it is omitted:
  - do not fail;
  - validate the source using checkpoint metadata/state;
  - record `"source_run_meta": null`.

Do not weaken the V9 architecture/config/source checks.

## 6.6 Preserve the exact V9 behavioral contracts

The V9 audit must continue to verify the current architecture contract,
including the current source-group anchors, six trainable child slots,
anchor-floor behavior, interruptible hierarchy, frozen world model, and current
gradient ordering.

Do not remove required tests or code merely to make the audit pass.

## 6.7 Option-Critic acceptance checks

```bash
cd smac-dreamer

python -m py_compile \
  external/r2dreamer/dreamer.py \
  external/r2dreamer/trainer.py \
  external/r2dreamer/tools.py \
  external/r2dreamer/hierarchical_options.py \
  external/r2dreamer/hierarchical_dreamer.py \
  external/r2dreamer/option_critic.py \
  scripts/train_r2dreamer_smaclite_multimap.py \
  scripts/audit_option_critic_v9_anchor_safe.py \
  scripts/assert_option_critic_v9_metrics.py \
  src/smacdreamer/validation_trainer.py

bash -n scripts/run_option_critic_v9_anchor_safe_800k.sh
bash -n scripts/static_audit_option_critic_v9_anchor_safe.sh

pytest -q \
  tests/test_option_critic_v9_core.py \
  tests/test_option_critic_v9_migration.py \
  tests/test_option_critic_v9_auxiliary.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/external/r2dreamer:$PWD/tests${PYTHONPATH:+:$PYTHONPATH}" \
python -m pytest -q \
  tests/test_hierarchical_options.py \
  tests/test_option_critic_math.py \
  tests/test_hierarchical_auxiliary.py \
  tests/test_hierarchy_migration.py
```

Add and run a checkpoint-free source audit:

```bash
SKIP_CHECKPOINT_AUDIT=1 \
bash scripts/static_audit_option_critic_v9_anchor_safe.sh
```

---

# 7. Canonical combined pipeline

Keep one master sequential launcher:

```text
smac-dreamer/scripts/run_forecast_then_actor_critic_h15_then_option_critic_v9.sh
```

This should remain the single canonical sequential workflow:

```text
Exp-45 forecast
    ->
ordinary actor-critic H=15 / 800k
    ->
Option-Critic V9 H=15 / 800k
```

Refactor it to pass explicit paths:

```text
EXP40_CHECKPOINT
TACTICAL_V12_CHECKPOINT
JEPA_CHECKPOINT
MANIFEST
```

It must not depend on:

```text
BUNDLE_ZIP
CURRENT_TACTICAL_V1_2_RUN.txt
CURRENT_OPTION_CRITIC*.txt
CURRENT_EXP45*.txt
historical logs
historical run directories
```

Delete older pipeline wrappers after the master pipeline and individual launchers
are validated.

Update `static_audit_option_critic_v9_anchor_safe.sh` so it refers to this
canonical master launcher instead of a deleted older forecast→Option-Critic
wrapper.

---

# 8. Files that must be moved before bundle deletion

## 8.1 Exp-40 rollout gallery

If these files exist only inside the root bundle, move them first:

```text
exp40_rollout_gallery_bundle/eval_exp40_rollout_gallery.py
    -> smac-jepa-wm/tools/eval_exp40_rollout_gallery.py

exp40_rollout_gallery_bundle/run_exp40_rollout_gallery.sh
    -> smac-jepa-wm/scripts/run_exp40_rollout_gallery.sh
```

Update path assumptions and validate:

```bash
python -m py_compile smac-jepa-wm/tools/eval_exp40_rollout_gallery.py
bash -n smac-jepa-wm/scripts/run_exp40_rollout_gallery.sh
```

Then delete:

```text
exp40_rollout_gallery_bundle/
exp40_rollout_gallery_bundle.zip
```

## 8.2 Map configs hidden in local assets

Before deleting `smac-jepa-wm/local_assets/`, inspect:

```text
smac-jepa-wm/local_assets/r2_compat_v1/extracted/r2_smaclite_general_2100_configs/
```

Compare it against:

```text
smac-dreamer/configs/maps/r2_2100/
```

Rules:

1. If the canonical config tree already contains identical files, do not copy
   duplicates.
2. If config files are missing, copy only the missing config files into the
   canonical config tree.
3. Preserve relative train/validation/blind split structure.
4. Do not copy dataset shards, checkpoints, logs, NumPy arrays, or generated
   outputs.
5. After config reconciliation, delete all of `smac-jepa-wm/local_assets/`.

---

# 9. Root-level deletion candidates

The following are generated outputs, bundles, backups, duplicated trees, or old
experiment staging areas. They may be deleted after the protected workflows
have been refactored and validated.

```text
.ipynb_checkpoints/

exp40_rollout_gallery_bundle/
exp45_pow2_direct_from_exp40_bundle/

forecast_ac15_ocv9_20260722_065246/
forecast_then_option_critic_v9_20260722_063325/

integration/

option_critic_1m_then_exp45_20260720_091403/
option_critic_hierarchy_bundle/
option_critic_p0p1_hotfix_bundle/
option_critic_p1_final_1m_pipeline_bundle/
option_critic_v5_1m_then_exp45_20260721_013103/
option_critic_v5_stability_hotfix_bundle/
option_critic_v6_8slot_1m_then_exp45_20260721_033236/
option_critic_v6_progressive_8slot_bundle/
option_critic_v9_anchor_safe_8slot_h15_800k_bundle/

overnight_logs/

preserve_before_option_critic_20260717_083004/
preserve_before_tactical_mixture_20260715_063403/
preserve_before_unified_priority_20260714_064543/

repaired_exp42_44_r2_first_20260713_062325/
repaired_exp42_44_r2_first_scripts/

smac-dreamer_option_critic_p0p1_hotfix_backup_20260720_020049/
smac-dreamer_option_critic_p1_final_backup_20260720_090901/
smac-dreamer_option_critic_v2_backup_20260717_084101/
smac-dreamer_option_critic_v5_stability_backup_20260721_013045/
smac-dreamer_option_critic_v6_progressive_backup_20260721_033219/
smac-dreamer_option_critic_v9_anchor_safe_backup_20260722_061818/
smac-dreamer_tactical_hardening_backup_20260716_013038/
smac-dreamer_tactical_mixture_installer_backup_20260715_072930/
smac-dreamer_tactical_v1_2_backup_20260716_101625/
smac-dreamer_unified_priority_installer_backup_20260714_064734/
smac-dreamer_unified_priority_installer_backup_20260714_065038/

smac_jepa/

tactical_mixture_bundle/
tactical_mixture_hardening_bundle/
tactical_mixture_v1_2_bundle/
unified_priority_bundle/

v9_three_stage_pipeline_patch/
```

The root `smac_jepa/` directory is a duplicate source tree. Preserve the
canonical package under:

```text
smac-jepa-wm/smac_jepa/
```

Before deleting the root duplicate, compare it and copy only a genuinely newer
file that is imported by a protected workflow. Document any such copy.

---

# 10. Root-level generated file deletion candidates

Delete old pointers, reports, archived bundles, and audit outputs, including:

```text
CURRENT_EXP45_FORECAST_PIPELINE.txt
CURRENT_FORECAST_AC15_OCV9_PIPELINE.txt
CURRENT_OPTION_CRITIC_AND_EXP45_PIPELINE.txt
CURRENT_OPTION_CRITIC_V2_RUN.txt
CURRENT_OPTION_CRITIC_V3_P0P1_RUN.txt
CURRENT_OPTION_CRITIC_V4_P1_1M_RUN.txt
CURRENT_OPTION_CRITIC_V5_AND_EXP45_PIPELINE.txt
CURRENT_OPTION_CRITIC_V5_STABILITY_1M_RUN.txt
CURRENT_OPTION_CRITIC_V6_AND_EXP45_PIPELINE.txt
CURRENT_OPTION_CRITIC_V6_PROGRESSIVE_8SLOT_1M_RUN.txt
CURRENT_OPTION_CRITIC_V9_ANCHOR_SAFE_8SLOT_800K_RUN.txt
CURRENT_TACTICAL_HARDENED_RUN.txt
CURRENT_TACTICAL_MIXTURE_RUN.txt
CURRENT_TACTICAL_V12_ACTOR_CRITIC_H15_800K_RUN.txt
CURRENT_TACTICAL_V1_2_RUN.txt
CURRENT_UNIFIED_PRIORITY_RUN.txt
TACTICAL_SOURCE_CHECKPOINT.txt

actor_critic_h15_800k_audit.txt
diagnose_tactical_run.py
diagnose_tactical_v1_2.py
option_critic_38test_audit.txt
option_critic_launcher_trace_20260717_085352.log
option_critic_p0p1_audit.txt
option_critic_p0p1_prelaunch_audit_20260720_022221.log
option_critic_p1_final_audit.txt
option_critic_p1_final_prelaunch_audit_20260720_091405.log
option_critic_prelaunch_audit_20260717_090015.log
option_critic_prelaunch_audit_20260717_090932.log
option_critic_v5_stability_audit.txt
option_critic_v5_stability_prelaunch_audit_20260721_013106.log
option_critic_v6_progressive_audit.txt
option_critic_v6_progressive_prelaunch_audit_20260721_033239.log
option_critic_v9_anchor_safe_audit.txt
smoke_slot_adapter_then_run.sh
tactical_diagnosis.txt
tactical_v1_2_diagnosis.txt

exp40_rollout_gallery_bundle.zip
exp45_pow2_direct_from_exp40_bundle.zip
option_critic_hierarchy_bundle_38tests.zip
option_critic_p0p1_hotfix_bundle.zip
option_critic_p1_final_1m_pipeline_bundle.zip
option_critic_v5_stability_1m_pipeline_bundle.zip
option_critic_v6_progressive_8slot_bundle.zip
option_critic_v9_anchor_safe_8slot_h15_800k_bundle.zip
tactical_mixture_bundle.zip
tactical_mixture_hardening_bundle.zip
tactical_mixture_v1_2_bundle.zip
unified_priority_bundle.zip
v9_three_stage_pipeline_patch.zip
v9_three_stage_pipeline_patch_v2.zip
```

Do not delete the protected source audit scripts merely because similarly named
generated `.txt` reports are being deleted.

---

# 11. Generated/cache directories to delete inside `smac-jepa-wm`

Delete if present:

```text
smac-jepa-wm/.hypothesis/
smac-jepa-wm/.pytest_cache/
smac-jepa-wm/.wandb/
smac-jepa-wm/wandb/
smac-jepa-wm/analysis_outputs/
smac-jepa-wm/eval_outputs/
smac-jepa-wm/sanity_outputs/
smac-jepa-wm/runs/
smac-jepa-wm/checkpoints/
smac-jepa-wm/local_assets/
```

Delete top-level generated dataset storage if present:

```text
smac-jepa-wm/data/
```

Do not delete:

```text
smac-jepa-wm/smac_jepa/data/
```

when that directory contains Python package source.

Delete generated archives and reports such as:

```text
smac-jepa-wm/*.zip
smac-jepa-wm/*.tar
smac-jepa-wm/*.tar.gz
smac-jepa-wm/*audit*.txt
```

except an intentionally retained human-authored source document, which must be
identified in `CLEANUP_REPORT.md`.

---

# 12. Old JEPA experiment code candidates

After tracing imports from the protected Exp-40 and Exp-45 entrypoints, remove
old experiment trainers and launchers that are not referenced.

Likely deletion candidates include:

```text
smac-jepa-wm/smac_jepa/train-lambda-prio-samp.py
smac-jepa-wm/smac_jepa/train-lambda.py
smac-jepa-wm/smac_jepa/train_enemy_visibility_mask.py
smac-jepa-wm/smac_jepa/train_jepa_blocker_fixed_experiments.py
smac-jepa-wm/smac_jepa/train_jepa_exp31_exp33.py
smac-jepa-wm/smac_jepa/train_jepa_exp33_dreamer.py
smac-jepa-wm/smac_jepa/train_jepa_exp34_dreamer.py
smac-jepa-wm/smac_jepa/train_jepa_exp35_dreamer.py
smac-jepa-wm/smac_jepa/train_jepa_exp39_dreamer.py
smac-jepa-wm/smac_jepa/train_jepa_exp41_dreamer.py
smac-jepa-wm/smac_jepa/train_lambda_prio_samp.before_one_forward_fixed.py
smac-jepa-wm/smac_jepa/train_lambda_prio_samp.py
smac-jepa-wm/smac_jepa/train_lambda_prio_samp_one_forward.py
smac-jepa-wm/smac_jepa/train_lambda_prio_samp_one_forward_visibility.py
smac-jepa-wm/smac_jepa/train_markov_rollout.py
smac-jepa-wm/smac_jepa/train_markov_rollout_deconflict.py
smac-jepa-wm/smac_jepa/train_markov_rollout_rnn_visibility_mask.py
smac-jepa-wm/smac_jepa/train_markov_rollout_rnn_visibility_seqmem.py
smac-jepa-wm/smac_jepa/train_markov_rollout_rnn_visibility_seqmem_experiments.py
smac-jepa-wm/smac_jepa/train_markov_rollout_rnn_visibility_seqmem_per(1).py
smac-jepa-wm/smac_jepa/train_markov_rollout_sample_prio.py
smac-jepa-wm/smac_jepa/train_markov_rollout_scheduled.py
smac-jepa-wm/smac_jepa/train_nstep_temporal.py
smac-jepa-wm/smac_jepa/train_prefix_rollout_temporal.py
smac-jepa-wm/smac_jepa/train_repaired_exp42_44_seqmem.py
smac-jepa-wm/smac_jepa/train_sample_prio.py
smac-jepa-wm/smac_jepa/train_weekend_exp42_51_seqmem.py
```

These are candidates, not unconditional deletions.

Before deleting each one, run:

```bash
rg -n --fixed-strings "$(basename PATH_TO_CANDIDATE)" \
  smac-jepa-wm smac-dreamer
```

Do not delete:

```text
smac-jepa-wm/smac_jepa/train_jepa_exp31_exp35.py
smac-jepa-wm/smac_jepa/train_jepa_exp40_dreamer.py
smac-jepa-wm/smac_jepa/train_jepa_exp45_pow2_direct.py
```

Likely old JEPA script candidates include:

```text
smac-jepa-wm/scripts/make_exp34_exp35_trainers.py
smac-jepa-wm/scripts/make_exp34_exp35_trainers_ast.py
smac-jepa-wm/scripts/make_exp34_exp35_trainers_ast_v2.py
smac-jepa-wm/scripts/make_exp34_exp35_trainers_v1_failed.py
smac-jepa-wm/scripts/make_exp34_exp35_trainers_v2.py
smac-jepa-wm/scripts/make_exp34_exp35_trainers_v3.py
smac-jepa-wm/scripts/make_exp34_exp35_trainers_v4.py
smac-jepa-wm/scripts/make_exp34_exp35_trainers_v5.py
smac-jepa-wm/scripts/run_exp33_dreamer_pretrain.sh
smac-jepa-wm/scripts/run_exp34_dreamer_two_mask.sh
smac-jepa-wm/scripts/run_exp34_exp35_probe.sh
smac-jepa-wm/scripts/run_exp35_dreamer_simple_loss.sh
smac-jepa-wm/scripts/run_exp39_dreamer_probe_action.sh
smac-jepa-wm/scripts/run_exp41_dreamer_hidden_change_gate.sh
```

Preserve all Exp-40 and Exp-45 scripts listed earlier.

---

# 13. Generated/cache directories to delete inside `smac-dreamer`

Delete if present:

```text
smac-dreamer/.hypothesis/
smac-dreamer/.pytest_cache/
smac-dreamer/.wandb/
smac-dreamer/wandb/
smac-dreamer/logs/
smac-dreamer/checkpoints/
smac-dreamer/results/
smac-dreamer/replay/
smac-dreamer/patch_backups/
smac-dreamer/Archive/
smac-dreamer/r2_jepa_final_fix/
smac-dreamer/r2_jepa_integration_fix/
smac-dreamer/docs/deprecated_dreamerV3/
smac-dreamer/docs/diagnostics/
```

If `external/r2dreamer/runs/` contains generated run output, delete that nested
runtime directory while preserving `external/r2dreamer` source.

Delete old fix bundles:

```text
smac-dreamer/r2_jepa_final_fix.zip
smac-dreamer/r2_jepa_integration_fix.zip
```

---

# 14. Old Dreamer launcher candidates

Delete old experiment-specific launchers only after confirming that the four
protected workflows and the canonical combined pipeline do not reference them.

Candidates include older Option-Critic versions:

```text
smac-dreamer/scripts/run_option_critic_1m_then_exp45_pipeline.sh
smac-dreamer/scripts/run_option_critic_2m.sh
smac-dreamer/scripts/run_option_critic_2m.sh.pre_audit_logging_fix
smac-dreamer/scripts/run_option_critic_p1_final_1m.sh
smac-dreamer/scripts/run_option_critic_v5_1m_then_exp45_pipeline.sh
smac-dreamer/scripts/run_option_critic_v5_stability_1m.sh
smac-dreamer/scripts/run_option_critic_v6_1m_then_exp45_pipeline.sh
smac-dreamer/scripts/run_option_critic_v6_progressive_1m.sh
```

Candidates include older Option-Critic audits/assertions:

```text
smac-dreamer/scripts/assert_option_critic_metrics.py
smac-dreamer/scripts/assert_option_critic_p1_final_metrics.py
smac-dreamer/scripts/assert_option_critic_v5_metrics.py
smac-dreamer/scripts/assert_option_critic_v6_metrics.py
smac-dreamer/scripts/audit_option_critic_hierarchy.py.pre_method_scoped_loss_audit
smac-dreamer/scripts/audit_option_critic_p1_final.py
smac-dreamer/scripts/audit_option_critic_v5_stability.py
smac-dreamer/scripts/audit_option_critic_v6_progressive.py
smac-dreamer/scripts/static_audit_option_critic_p1_final.sh
smac-dreamer/scripts/static_audit_option_critic_v5_stability.sh
smac-dreamer/scripts/static_audit_option_critic_v6_progressive.sh
```

Candidates include superseded tactical workflows:

```text
smac-dreamer/scripts/run_tactical_hardened_2m.sh
smac-dreamer/scripts/run_tactical_mixture_2m.sh
smac-dreamer/scripts/static_audit_tactical_hardening.sh
smac-dreamer/scripts/static_audit_tactical_mixture.sh
smac-dreamer/scripts/audit_tactical_hardening.py
smac-dreamer/scripts/assert_tactical_hardened_metrics.py
smac-dreamer/scripts/assert_tactical_metrics.py
```

Do not delete Tactical-v1.2 files.

Candidates include superseded priority and pipeline launchers:

```text
smac-dreamer/scripts/run_unified_priority_resume_full.sh
smac-dreamer/scripts/run_unified_priority_resume_smoke.sh
smac-dreamer/scripts/preflight_unified_priority.py
smac-dreamer/scripts/inspect_unified_priority_checkpoint.py
smac-dreamer/scripts/static_audit_unified_priority.sh
smac-dreamer/scripts/assert_unified_priority_metrics.py
smac-dreamer/scripts/run_forecast_first_then_option_critic_v9_800k.sh
```

The old forecast→Option-Critic wrapper may be deleted only after
`static_audit_option_critic_v9_anchor_safe.sh` is updated to validate the
canonical three-stage launcher instead.

Delete source backup suffixes:

```text
smac-dreamer/scripts/*.bak*
smac-dreamer/scripts/*.before_*
smac-dreamer/scripts/*.pre_*
```

only when the unsuffixed canonical file exists and compiles.

---

# 15. Global runtime artifact removal

After moving required configs and protected tools, delete runtime artifacts:

```bash
find . -type d \( \
    -name '__pycache__' -o \
    -name '.pytest_cache' -o \
    -name '.hypothesis' -o \
    -name '.ipynb_checkpoints' \
  \) -prune -exec rm -rf {} +

find . -type f \( \
    -name '*.pyc' -o \
    -name '*.pyo' -o \
    -name '*.log' \
  \) -delete
```

Delete historical model/data outputs, but exclude config trees and source test
fixtures that are genuinely required:

```bash
find . -type f \( \
    -name '*.pt' -o \
    -name '*.pth' -o \
    -name '*.ckpt' -o \
    -name '*.npy' -o \
    -name '*.npz' \
  \) -print
```

Review this list before deletion.

Do not delete tiny intentional test fixtures without checking references.
No production checkpoint should remain.

Delete archive bundles:

```bash
find . -type f \( \
    -name '*.zip' -o \
    -name '*.tar' -o \
    -name '*.tar.gz' -o \
    -name '*.tgz' \
  \) -delete
```

Run this only after all files that exist solely inside bundles have been moved
into canonical source locations.

---

# 16. Add a repository-wide ignore policy

Update the root `.gitignore` and project `.gitignore` files to ignore future:

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.hypothesis/

# Jupyter
.ipynb_checkpoints/

# Runtime logs and experiment output
logs/
runs/
results/
eval_outputs/
analysis_outputs/
sanity_outputs/
overnight_logs/
wandb/
.wandb/

# Checkpoints and replay
checkpoints/
replay/
replay_*/
*.pt
*.pth
*.ckpt

# Generated datasets and arrays
data/
local_assets/
*.npy
*.npz
*.memmap

# Generated archives
*.zip
*.tar
*.tar.gz
*.tgz

# Runtime pointers and reports
CURRENT_*.txt
*_audit.txt
*_diagnosis.txt
*.log
```

Do not ignore:

```text
smac-dreamer/configs/
smac-dreamer/configs/maps/
smac-jepa-wm/splits/
smac-jepa-wm/smac_jepa/data/
```

Use anchored ignore rules or explicit negations where necessary so source package
directories named `data` remain visible.

---

# 17. Add one canonical retained-workflow validator

Create:

```text
scripts/validate_retained_workflows.sh
```

at the repository root, or an equivalent clearly documented location.

It must perform checkpoint-free validation of:

1. Exp-40 imports and launcher syntax.
2. Exp-45 imports, tests, and launcher syntax.
3. Generic R2-Dreamer trainer imports.
4. Tactical-v1.2 source code and tests.
5. Actor-critic H=15 config and launcher.
6. Option-Critic V9 source, tests, config, and launcher.
7. Canonical combined pipeline syntax.
8. Absence of historical hard-coded paths.
9. Absence of bundle dependencies.
10. Absence of root `CURRENT_*.txt` dependencies.

Suggested skeleton:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"

cd "$ROOT/smac-jepa-wm"

"$PY" -m smac_jepa.train_jepa_exp40_dreamer --help >/dev/null
"$PY" -m smac_jepa.train_jepa_exp45_pow2_direct --help >/dev/null

bash -n scripts/run_exp40_dreamer_event_balanced.sh
bash -n scripts/run_exp45_pow2_direct_train.sh
bash -n scripts/eval_exp45_pow2_all.sh
bash -n scripts/eval_exp45_pow2_ordinary.sh
bash -n scripts/eval_exp45_pow2_hidden.sh
bash -n scripts/static_audit_exp45_pow2.sh

"$PY" -m pytest -q \
  tests/test_pow2_checkpoint_sanitizer.py \
  tests/test_pow2_direct_predictor.py

cd "$ROOT/smac-dreamer"

"$PY" scripts/train_r2dreamer_smaclite_multimap.py --help >/dev/null
"$PY" scripts/evaluate_multimap.py --help >/dev/null

bash -n scripts/run_exp45_full_train_eval_resilient.sh
bash -n scripts/run_tactical_v1_2_2m.sh
bash -n scripts/run_actor_critic_h15_800k.sh
bash -n scripts/run_option_critic_v9_anchor_safe_800k.sh
bash -n scripts/run_forecast_then_actor_critic_h15_then_option_critic_v9.sh

SKIP_CHECKPOINT_AUDIT=1 \
  bash scripts/static_audit_tactical_v1_2.sh

SKIP_CHECKPOINT_AUDIT=1 \
  bash scripts/static_audit_actor_critic_h15_800k.sh

SKIP_CHECKPOINT_AUDIT=1 \
  bash scripts/static_audit_option_critic_v9_anchor_safe.sh

"$PY" -m pytest -q \
  tests/test_tactical_policy_v1_2.py \
  tests/test_option_critic_v9_core.py \
  tests/test_option_critic_v9_migration.py \
  tests/test_option_critic_v9_auxiliary.py

cd "$ROOT"

if rg -n \
  'CURRENT_[A-Z0-9_]+\.txt|BUNDLE_ZIP|combined-upload|20260709_083104' \
  smac-jepa-wm/scripts \
  smac-dreamer/scripts; then
  echo "[FAIL] stale runtime pointer, bundle dependency, or historical path remains" >&2
  exit 1
fi

echo "[OK] all retained workflows passed checkpoint-free validation"
```

Adapt paths to the actual final repository structure.

---

# 18. Required final documentation

Create or update:

```text
RETAINED_WORKFLOWS.md
```

It must document the exact commands for the four required training workflows.

## 18.1 Exp-40

```bash
MANIFEST=/path/to/manifest.json \
OUT_DIR=/path/to/output/exp40 \
WANDB=0 \
bash smac-jepa-wm/scripts/run_exp40_dreamer_event_balanced.sh
```

## 18.2 Exp-45 forecast

```bash
EXP40_CHECKPOINT=/path/to/exp40/checkpoint.pt \
MANIFEST=/path/to/manifest.json \
PIPE_DIR=/path/to/output/exp45_pipeline \
WANDB=0 \
bash smac-dreamer/scripts/run_exp45_full_train_eval_resilient.sh
```

## 18.3 Current actor-critic comparison

```bash
SOURCE_CHECKPOINT=/path/to/tactical_v1_2_checkpoint.pt \
JEPA_CHECKPOINT=/path/to/exp40/checkpoint.pt \
RUN_DIR=/path/to/output/actor_critic_h15_800k \
FINAL_STEP=800000 \
bash smac-dreamer/scripts/run_actor_critic_h15_800k.sh
```

## 18.4 Current Option-Critic V9

```bash
SOURCE_CHECKPOINT=/path/to/tactical_v1_2_checkpoint.pt \
JEPA_CHECKPOINT=/path/to/exp40/checkpoint.pt \
RUN_DIR=/path/to/output/option_critic_v9_h15_800k \
FINAL_STEP=800000 \
bash smac-dreamer/scripts/run_option_critic_v9_anchor_safe_800k.sh
```

Document optional exact-reproduction inputs:

```text
SOURCE_RUN_META
EXPECTED_SOURCE_CHECKPOINT_SHA256
```

Also document that no checkpoints or datasets are bundled after cleanup.

---

# 19. Final repository checks

## 19.1 Compile retained Python

```bash
python -m compileall -q \
  smac-jepa-wm/smac_jepa \
  smac-jepa-wm/scripts \
  smac-jepa-wm/tools \
  smac-dreamer/src \
  smac-dreamer/scripts \
  smac-dreamer/external/r2dreamer
```

## 19.2 Run retained tests

```bash
pytest -q smac-jepa-wm/tests
pytest -q smac-dreamer/tests
```

If the complete test suite has unavailable optional dependencies, run all
protected-workflow tests and document the skipped dependency-bound tests in
`CLEANUP_REPORT.md`. Do not silently ignore failures.

## 19.3 Search for stale references

```bash
rg -n \
  'CURRENT_[A-Z0-9_]+\.txt|BUNDLE_ZIP|preserve_before_|backup_2026|combined-upload|rnn_seqmem_exp40_event_balanced_5ep_20260709_083104' \
  smac-jepa-wm \
  smac-dreamer
```

Any remaining occurrence must be either:

- an intentional historical note in documentation; or
- removed/refactored.

It must not occur in an active launcher.

## 19.4 Search for deleted-file references

Create a list of every deleted path and basename, then search for references:

```bash
while IFS= read -r deleted; do
  base="$(basename "$deleted")"
  rg -n --fixed-strings "$base" smac-jepa-wm smac-dreamer || true
done < deleted_paths.txt
```

Review every match.

## 19.5 Verify no generated artifacts remain

```bash
find . -type f \( \
    -name '*.pt' -o \
    -name '*.pth' -o \
    -name '*.ckpt' -o \
    -name '*.npy' -o \
    -name '*.npz' -o \
    -name '*.log' -o \
    -name '*.zip' -o \
    -name '*.tar' -o \
    -name '*.tar.gz' -o \
    -name '*.pyc' \
  \) -print
```

The result should be empty unless a tiny referenced test fixture is explicitly
justified in `CLEANUP_REPORT.md`.

Check generated directories:

```bash
find . -type d \( \
    -name logs -o \
    -name runs -o \
    -name results -o \
    -name replay -o \
    -name wandb -o \
    -name .wandb -o \
    -name __pycache__ -o \
    -name .pytest_cache -o \
    -name .hypothesis \
  \) -print
```

No historical runtime directory should remain.

## 19.6 Verify configs remain

```bash
find smac-dreamer/configs -type f | sort > configs_after.txt
```

Compare with the pre-cleanup config inventory. No config may disappear unless it
was an exact duplicate copied into the canonical config tree and the removal is
documented.

---

# 20. Definition of done

The cleanup is complete only when all of the following are true:

- [ ] All historical logs are deleted.
- [ ] All historical datasets are deleted.
- [ ] All historical checkpoints are deleted.
- [ ] All replay/memmap state is deleted.
- [ ] All root bundles and backup source trees are deleted.
- [ ] All root `CURRENT_*.txt` pointers are deleted.
- [ ] All configs remain.
- [ ] Exp-40 training imports and launcher syntax pass.
- [ ] Exp-45 forecast training/evaluation imports, tests, and launcher syntax
      pass.
- [ ] Generic R2-Dreamer multimap trainer imports.
- [ ] Tactical-v1.2 source code and test remain intact.
- [ ] Current ordinary actor-critic H=15 / 800k launcher remains intact.
- [ ] Current Option-Critic V9 anchor-safe 8-slot H=15 / 800k launcher,
      integrated source, audits, and tests remain intact.
- [ ] Protected launchers accept explicit external checkpoint/data paths.
- [ ] No protected launcher depends on old logs or `CURRENT_*.txt`.
- [ ] No protected launcher depends on a ZIP installer.
- [ ] No protected launcher contains a hard-coded historical workspace path.
- [ ] `scripts/validate_retained_workflows.sh` passes.
- [ ] `RETAINED_WORKFLOWS.md` documents all required commands.
- [ ] `CLEANUP_REPORT.md` lists all deletions, moves, refactors, tests, and any
      environment-bound checks that could not be run.

Do not claim completion merely because the repository is smaller. Completion
means the four required training workflows remain structurally executable after
all bundled runtime artifacts have been removed.
