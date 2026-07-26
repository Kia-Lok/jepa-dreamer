# Codex Instructions: Finish Cleaning `Kia-Lok/jepa-dreamer`

## Objective

Finish the cleanup of the repository:

```text
https://github.com/Kia-Lok/jepa-dreamer
```

The repository is already mostly cleaned and the missing JEPA dataloader source package has been restored.

This task is **not** a broad redesign. Make the smallest safe set of changes required to:

1. remove remaining historical backup and generated files;
2. ensure the restored JEPA dataloader source is tracked and validated;
3. update stale cleanup documentation;
4. strengthen the retained-workflow validator;
5. preserve all required training workflows.

---

# 1. Workflows that must remain intact

Do not break any of the following.

## 1.1 Exp-40 JEPA training

Required launcher:

```text
smac-jepa-wm/scripts/run_exp40_dreamer_event_balanced.sh
```

Required trainer chain:

```text
smac-jepa-wm/smac_jepa/train_jepa_exp40_dreamer.py
smac-jepa-wm/smac_jepa/train_jepa_exp31_exp35.py
smac-jepa-wm/smac_jepa/anchored_belief_memory.py
smac-jepa-wm/scripts/validate_exp33_dreamer_checkpoint.py
```

Do not delete `train_jepa_exp31_exp35.py` or
`validate_exp33_dreamer_checkpoint.py` merely because they contain older
experiment numbers. They are live Exp-40 dependencies.

## 1.2 Exp-45 forecast training and evaluation

Required source:

```text
smac-jepa-wm/smac_jepa/train_jepa_exp45_pow2_direct.py
smac-jepa-wm/smac_jepa/pow2_direct_predictor.py
smac-jepa-wm/tools/audit_exp45_pow2_checkpoint.py
smac-jepa-wm/tools/eval_pow2_direct.py
```

Required launchers:

```text
smac-dreamer/scripts/run_exp45_full_train_eval_resilient.sh
smac-jepa-wm/scripts/run_exp45_pow2_direct_train.sh
smac-jepa-wm/scripts/eval_exp45_pow2_all.sh
smac-jepa-wm/scripts/eval_exp45_pow2_ordinary.sh
smac-jepa-wm/scripts/eval_exp45_pow2_hidden.sh
smac-jepa-wm/scripts/smoke_exp45_pow2_direct.sh
smac-jepa-wm/scripts/static_audit_exp45_pow2.sh
```

Required tests:

```text
smac-jepa-wm/tests/test_pow2_checkpoint_sanitizer.py
smac-jepa-wm/tests/test_pow2_direct_predictor.py
```

## 1.3 R2-Dreamer ordinary actor-critic

Required generic training and evaluation source:

```text
smac-dreamer/scripts/train_r2dreamer_smaclite_multimap.py
smac-dreamer/scripts/train_r2dreamer_smaclite_debug.py
smac-dreamer/scripts/evaluate_multimap.py
smac-dreamer/scripts/preflight_jepa_training.py
smac-dreamer/scripts/inspect_jepa_checkpoint.py
smac-dreamer/scripts/validate_jepa_r2_integration.py
smac-dreamer/scripts/validate_jepa_token_parity.py
```

Required current actor-critic comparison:

```text
smac-dreamer/scripts/run_actor_critic_h15_800k.sh
smac-dreamer/scripts/static_audit_actor_critic_h15_800k.sh
smac-dreamer/scripts/audit_actor_critic_h15_800k.py
```

Required configs:

```text
smac-dreamer/configs/r2_2100_jepa_tactical_mixture_v1_2.yaml
smac-dreamer/configs/r2_2100_jepa_tactical_mixture_v1_2_actor_critic_h15_800k.yaml
```

Required Tactical-v1.2 source and tests:

```text
smac-dreamer/external/r2dreamer/tactical_policy.py
smac-dreamer/external/r2dreamer/dreamer.py
smac-dreamer/tests/test_tactical_policy_v1_2.py
```

## 1.4 Current Option-Critic V9

The retained Option-Critic implementation is:

```text
Option-Critic V9 anchor-safe
8 slots
imagination horizon 15
800,000 new environment steps
```

Required launchers and audits:

```text
smac-dreamer/scripts/run_option_critic_v9_anchor_safe_800k.sh
smac-dreamer/scripts/static_audit_option_critic_v9_anchor_safe.sh
smac-dreamer/scripts/audit_option_critic_v9_anchor_safe.py
smac-dreamer/scripts/assert_option_critic_v9_metrics.py
smac-dreamer/scripts/check_option_critic_win_guard.py
```

Required config:

```text
smac-dreamer/configs/r2_2100_jepa_option_critic_8_v9_anchor_safe_h15_800k.yaml
```

Required integrated source:

```text
smac-dreamer/external/r2dreamer/dreamer.py
smac-dreamer/external/r2dreamer/trainer.py
smac-dreamer/external/r2dreamer/tools.py
smac-dreamer/external/r2dreamer/tactical_policy.py
smac-dreamer/external/r2dreamer/hierarchical_options.py
smac-dreamer/external/r2dreamer/hierarchical_dreamer.py
smac-dreamer/external/r2dreamer/option_critic.py
smac-dreamer/src/smacdreamer/validation_trainer.py
```

Required tests:

```text
smac-dreamer/tests/test_option_critic_v9_core.py
smac-dreamer/tests/test_option_critic_v9_migration.py
smac-dreamer/tests/test_option_critic_v9_auxiliary.py
smac-dreamer/tests/test_hierarchical_options.py
smac-dreamer/tests/test_option_critic_math.py
smac-dreamer/tests/test_hierarchical_auxiliary.py
smac-dreamer/tests/test_hierarchy_migration.py
```

## 1.5 Canonical combined pipeline

Preserve:

```text
smac-dreamer/scripts/run_forecast_then_actor_critic_h15_then_option_critic_v9.sh
```

This must remain the canonical sequential workflow:

```text
Exp-45 forecast
    ->
ordinary actor-critic H=15 / 800k
    ->
Option-Critic V9 H=15 / 800k
```

---

# 2. Confirm the restored JEPA dataloader package

The following source package must exist and remain tracked:

```text
smac-jepa-wm/smac_jepa/data/
├── __init__.py
├── dataset.py
├── markov_rollout_dataset.py
└── markov_rollout_visibility_dataset.py
```

These are Python source files, not runtime dataset files.

Do not confuse:

```text
smac-jepa-wm/data/
```

with:

```text
smac-jepa-wm/smac_jepa/data/
```

The first is generated runtime data and should remain ignored.

The second is source code and must be committed.

Run:

```bash
set -euo pipefail

for file in \
  smac-jepa-wm/smac_jepa/data/__init__.py \
  smac-jepa-wm/smac_jepa/data/dataset.py \
  smac-jepa-wm/smac_jepa/data/markov_rollout_dataset.py \
  smac-jepa-wm/smac_jepa/data/markov_rollout_visibility_dataset.py
do
  [[ -s "$file" ]] || {
    echo "ERROR: missing JEPA dataloader source: $file" >&2
    exit 1
  }
done

git ls-files smac-jepa-wm/smac_jepa/data
```

Expected tracked files:

```text
smac-jepa-wm/smac_jepa/data/__init__.py
smac-jepa-wm/smac_jepa/data/dataset.py
smac-jepa-wm/smac_jepa/data/markov_rollout_dataset.py
smac-jepa-wm/smac_jepa/data/markov_rollout_visibility_dataset.py
```

---

# 3. Definite files to delete

Delete these files unconditionally.

## 3.1 macOS metadata

```text
.DS_Store
```

## 3.2 Old R2-Dreamer source snapshots

```text
smac-dreamer/external/r2dreamer/tools.py.pre_bfloat16_numpy_fix
smac-dreamer/external/r2dreamer/trainer.py.before_unified_indent_fix
```

The canonical unsuffixed files already exist.

## 3.3 Old JEPA integration source snapshots

```text
smac-dreamer/src/smacdreamer/jepa/checkpoint.py.before_capacity_fix
smac-dreamer/src/smacdreamer/jepa/checkpoint.py.pre_exp33
smac-dreamer/src/smacdreamer/jepa/feature_adapter.py.before_slot_adapter
smac-dreamer/src/smacdreamer/jepa/world_model.py.before_slot_adapter
smac-dreamer/src/smacdreamer/jepa/world_model.py.pre_exp33
```

Delete with:

```bash
git rm \
  .DS_Store \
  smac-dreamer/external/r2dreamer/tools.py.pre_bfloat16_numpy_fix \
  smac-dreamer/external/r2dreamer/trainer.py.before_unified_indent_fix \
  smac-dreamer/src/smacdreamer/jepa/checkpoint.py.before_capacity_fix \
  smac-dreamer/src/smacdreamer/jepa/checkpoint.py.pre_exp33 \
  smac-dreamer/src/smacdreamer/jepa/feature_adapter.py.before_slot_adapter \
  smac-dreamer/src/smacdreamer/jepa/world_model.py.before_slot_adapter \
  smac-dreamer/src/smacdreamer/jepa/world_model.py.pre_exp33
```

If any path no longer exists, continue with the remaining paths.

---

# 4. One-off utilities to review and probably delete

These appear to be historical debugging or patch-validation scripts:

```text
smac-dreamer/inspect_r2_run_config.py
smac-dreamer/validate_adapter_grad.py
smac-dreamer/validate_belief_mask_patch.py
smac-dreamer/validate_integration_static.py
smac-dreamer/scripts/infer_resume_step.py
smac-dreamer/scripts/spike_parallelenv_map_routing.py
```

Before deleting, check whether any protected launcher, source module, test, or
documentation file references them:

```bash
for file in \
  inspect_r2_run_config.py \
  validate_adapter_grad.py \
  validate_belief_mask_patch.py \
  validate_integration_static.py \
  infer_resume_step.py \
  spike_parallelenv_map_routing.py
do
  echo
  echo "===== $file ====="
  rg -n --fixed-strings "$file" . || true
done
```

If a file has no live reference from a protected workflow, delete it:

```bash
git rm \
  smac-dreamer/inspect_r2_run_config.py \
  smac-dreamer/validate_adapter_grad.py \
  smac-dreamer/validate_belief_mask_patch.py \
  smac-dreamer/validate_integration_static.py \
  smac-dreamer/scripts/infer_resume_step.py \
  smac-dreamer/scripts/spike_parallelenv_map_routing.py
```

Do not delete a file if an active retained launcher imports or invokes it.
Document any retained exception in `CLEANUP_REPORT.md`.

---

# 5. Old JEPA evaluator files to review

Review:

```text
smac-jepa-wm/eval_jepa_exp31_exp33.py
smac-jepa-wm/eval_jepa_exp31_exp33_anchored.py
smac-jepa-wm/eval_jepa_hidden_belief_exp31_exp33.py
smac-jepa-wm/eval_rnn_seqmem_dreamer_probe.py
```

Check references:

```bash
for file in \
  eval_jepa_exp31_exp33.py \
  eval_jepa_exp31_exp33_anchored.py \
  eval_jepa_hidden_belief_exp31_exp33.py \
  eval_rnn_seqmem_dreamer_probe.py
do
  echo
  echo "===== $file ====="
  rg -n --fixed-strings "$file" . || true
done
```

Delete files that are not referenced by:

- the Exp-40 training launcher;
- Exp-40 checkpoint validation;
- the Exp-40 rollout gallery;
- Exp-45 forecast evaluation;
- retained tests;
- current documentation.

Probable deletion command:

```bash
git rm \
  smac-jepa-wm/eval_jepa_exp31_exp33.py \
  smac-jepa-wm/eval_jepa_exp31_exp33_anchored.py \
  smac-jepa-wm/eval_jepa_hidden_belief_exp31_exp33.py \
  smac-jepa-wm/eval_rnn_seqmem_dreamer_probe.py
```

Do not delete only because the names contain old experiment numbers. Delete
only after the reference check.

---

# 6. Fix `.gitignore` rules

The current ignore rules should explicitly distinguish generated runtime data
from the Python source package named `data`.

## 6.1 Root `.gitignore`

Ensure the root `.gitignore` contains:

```gitignore
# macOS
.DS_Store

# Generated runtime datasets only
/smac-jepa-wm/data/
/smac-dreamer/data/

# Keep JEPA Python dataloader source
!smac-jepa-wm/smac_jepa/data/
!smac-jepa-wm/smac_jepa/data/**
```

Remove or replace broad rules such as:

```gitignore
**/data/
```

if they can accidentally hide nested source packages.

Keep the existing runtime ignore rules for:

```text
logs
runs
results
checkpoints
replay
wandb
NumPy outputs
model checkpoints
archives
Python caches
```

## 6.2 `smac-jepa-wm/.gitignore`

Use an anchored rule:

```gitignore
/data/
```

Do not use an unanchored:

```gitignore
data/
```

if it could match nested source paths.

## 6.3 `smac-jepa-wm/smac_jepa/.gitignore`

Remove rules such as:

```gitignore
data/
!data/
!data/**
```

The Python package should not ignore one of its own source subpackages.

## 6.4 Verify ignore behavior

Run:

```bash
set +e

git check-ignore -v \
  smac-jepa-wm/smac_jepa/data/__init__.py \
  smac-jepa-wm/smac_jepa/data/dataset.py \
  smac-jepa-wm/smac_jepa/data/markov_rollout_dataset.py \
  smac-jepa-wm/smac_jepa/data/markov_rollout_visibility_dataset.py

STATUS=$?
set -e

if [[ "$STATUS" -eq 0 ]]; then
  echo "ERROR: JEPA dataloader source is still ignored" >&2
  exit 1
fi
```

Also verify runtime data is ignored:

```bash
mkdir -p smac-jepa-wm/data
touch smac-jepa-wm/data/ignore_test.npz

git check-ignore -v smac-jepa-wm/data/ignore_test.npz

rm -f smac-jepa-wm/data/ignore_test.npz
rmdir smac-jepa-wm/data 2>/dev/null || true
```

---

# 7. Update stale `CLEANUP_REPORT.md`

The report previously stated that:

- `smac-jepa-wm/smac_jepa/data/` was absent;
- Exp-40 and Exp-45 imports were blocked;
- dataset tests could not run.

Those claims are now stale.

Replace the stale section with:

```markdown
## Restored JEPA dataset-loader source

The Python source package `smac-jepa-wm/smac_jepa/data/` has been restored from
the previous canonical JEPA repository. It contains:

- `__init__.py`
- `dataset.py`
- `markov_rollout_dataset.py`
- `markov_rollout_visibility_dataset.py`

These are tracked source files. Runtime `.npz` datasets remain excluded and
must be supplied externally.

The retained-workflow validator now compiles these loaders, imports the base and
visibility-aware datasets, runs the dataset-window tests, checks the explicit
visibility-mask contract, and validates Exp-40 and Exp-45 trainer imports.
```

Also update the report with:

- files deleted in this final cleanup;
- one-off utilities retained and why;
- older evaluators retained and why, if any;
- validation commands run;
- tests skipped because of unavailable optional dependencies;
- any unresolved failure.

Do not claim a test passed unless it was actually run successfully.

---

# 8. Strengthen `scripts/validate_retained_workflows.sh`

The current validator should no longer allow missing JEPA dataloader source.

Remove:

```text
ALLOW_MISSING_JEPA_DATA_SOURCE
```

and any branch that treats missing loader source as acceptable.

Add strict source checks.

## 8.1 Loader file existence and compilation

Add:

```bash
JEPA_DATA_FILES=(
  smac_jepa/data/__init__.py
  smac_jepa/data/dataset.py
  smac_jepa/data/markov_rollout_dataset.py
  smac_jepa/data/markov_rollout_visibility_dataset.py
)

for file in "${JEPA_DATA_FILES[@]}"; do
  [[ -s "$file" ]] || {
    echo "[FAIL] Missing JEPA loader source: $file" >&2
    exit 1
  }
done

"$PY" -m py_compile "${JEPA_DATA_FILES[@]}"
```

## 8.2 Loader import and visibility contract

Add:

```bash
"$PY" - <<'PY'
from smac_jepa.data import (
    SMACJEPADataset,
    load_manifest,
    load_manifest_all,
    load_npz_metadata,
)
from smac_jepa.data.markov_rollout_dataset import (
    MarkovRolloutSMACJEPADataset,
)
from smac_jepa.data.markov_rollout_visibility_dataset import (
    VisibilityMarkovRolloutSMACJEPADataset,
)

assert SMACJEPADataset is not None
assert MarkovRolloutSMACJEPADataset is not None
assert VisibilityMarkovRolloutSMACJEPADataset is not None
assert VisibilityMarkovRolloutSMACJEPADataset.explicit_visibility_mask_version >= 1

print("[OK] JEPA dataloader imports and visibility contract")
PY
```

## 8.3 Dataset and memory tests

Add:

```bash
"$PY" -m pytest -q tests/test_dataset_windows.py
"$PY" tests/test_exp33_memory_contract.py
```

If `test_exp33_memory_contract.py` is a pytest test rather than a direct script,
use:

```bash
"$PY" -m pytest -q tests/test_exp33_memory_contract.py
```

Inspect the file first and choose the correct invocation.

## 8.4 Exp-40 and Exp-45 imports

Ensure the validator runs:

```bash
"$PY" -m smac_jepa.train_jepa_exp40_dreamer --help >/dev/null
"$PY" -m smac_jepa.train_jepa_exp45_pow2_direct --help >/dev/null
```

## 8.5 Shell syntax checks

Ensure it checks:

```bash
bash -n scripts/run_exp40_dreamer_event_balanced.sh
bash -n scripts/run_exp45_pow2_direct_train.sh
bash -n scripts/eval_exp45_pow2_all.sh
bash -n scripts/eval_exp45_pow2_ordinary.sh
bash -n scripts/eval_exp45_pow2_hidden.sh
bash -n scripts/smoke_exp45_pow2_direct.sh
bash -n scripts/static_audit_exp45_pow2.sh
```

After changing to `smac-dreamer`, ensure it checks:

```bash
bash -n scripts/run_exp45_full_train_eval_resilient.sh
bash -n scripts/run_tactical_v1_2_2m.sh
bash -n scripts/run_actor_critic_h15_800k.sh
bash -n scripts/run_option_critic_v9_anchor_safe_800k.sh
bash -n scripts/run_forecast_then_actor_critic_h15_then_option_critic_v9.sh
```

## 8.6 R2-Dreamer import checks

Ensure it runs:

```bash
"$PY" scripts/train_r2dreamer_smaclite_multimap.py --help >/dev/null
"$PY" scripts/evaluate_multimap.py --help >/dev/null
"$PY" scripts/preflight_jepa_training.py --help >/dev/null
```

## 8.7 Tactical and Option-Critic tests

Ensure it runs:

```bash
"$PY" -m pytest -q \
  tests/test_tactical_policy_v1_2.py \
  tests/test_option_critic_v9_core.py \
  tests/test_option_critic_v9_migration.py \
  tests/test_option_critic_v9_auxiliary.py
```

Also preserve and run the hierarchy tests used by the V9 implementation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/external/r2dreamer:$PWD/tests${PYTHONPATH:+:$PYTHONPATH}" \
"$PY" -m pytest -q \
  tests/test_hierarchical_options.py \
  tests/test_option_critic_math.py \
  tests/test_hierarchical_auxiliary.py \
  tests/test_hierarchy_migration.py
```

## 8.8 Checkpoint-free static audits

Ensure these continue to work without bundled historical checkpoints:

```bash
SKIP_CHECKPOINT_AUDIT=1 \
  bash scripts/static_audit_tactical_v1_2.sh

SKIP_CHECKPOINT_AUDIT=1 \
  bash scripts/static_audit_actor_critic_h15_800k.sh

SKIP_CHECKPOINT_AUDIT=1 \
  bash scripts/static_audit_option_critic_v9_anchor_safe.sh
```

## 8.9 Stale-reference check

Add a repository scan that fails if active launchers still reference:

```text
CURRENT_*.txt
BUNDLE_ZIP
combined-upload
historical run directories
the old default Exp-40 checkpoint path
```

Suggested check:

```bash
if rg -n \
  'CURRENT_[A-Z0-9_]+\.txt|BUNDLE_ZIP|combined-upload|rnn_seqmem_exp40_event_balanced_5ep_20260709_083104' \
  "$ROOT/smac-jepa-wm/scripts" \
  "$ROOT/smac-dreamer/scripts"
then
  echo "[FAIL] stale runtime pointer, bundle dependency, or historical path remains" >&2
  exit 1
fi
```

---

# 9. Verify no backup snapshots remain

Run:

```bash
find . -type f \( \
  -name '*.before_*' -o \
  -name '*.pre_*' -o \
  -name '*.bak*' -o \
  -name '*.orig' -o \
  -name '*.rej' -o \
  -name '.DS_Store' \
\) -print
```

The output should be empty.

If a file appears:

1. confirm the canonical unsuffixed version exists;
2. search for references;
3. delete the backup when unreferenced.

---

# 10. Verify no generated artifacts are tracked

Run:

```bash
git ls-files | grep -E \
  '(^|/)(logs?|runs?|wandb|checkpoints?|replay|results|eval_outputs|analysis_outputs|sanity_outputs)(/|$)|\.(pt|pth|ckpt|npy|npz|log|zip|tar|tgz|pyc)$'
```

This command should return no production runtime artifact.

If it returns an intentionally committed tiny test fixture:

1. confirm a retained test directly references it;
2. document it in `CLEANUP_REPORT.md`;
3. do not retain any real training checkpoint, dataset, log, replay buffer, or
   archived bundle.

---

# 11. Verify configs were not deleted

All files under:

```text
smac-dreamer/configs/
```

must remain.

Run:

```bash
find smac-dreamer/configs -type f | sort > /tmp/jepa_dreamer_configs_after.txt
wc -l /tmp/jepa_dreamer_configs_after.txt
```

Compare against the pre-cleanup repository or Git history where available.

Do not delete configs solely because:

- their names contain an older experiment number;
- their names include `tmp`;
- their names include `backup`;
- they represent an ablation;
- they are not used by the four current launchers.

The cleanup rule explicitly preserves all configs.

---

# 12. Compile all retained source

Run:

```bash
python -m compileall -q \
  smac-jepa-wm/smac_jepa \
  smac-jepa-wm/scripts \
  smac-jepa-wm/tools \
  smac-dreamer/src \
  smac-dreamer/scripts \
  smac-dreamer/external/r2dreamer
```

Any compile failure must be fixed before committing.

---

# 13. Run the retained workflow validator

Run:

```bash
PY=python bash scripts/validate_retained_workflows.sh
```

Do not bypass a loader failure.

Do not restore `ALLOW_MISSING_JEPA_DATA_SOURCE`.

---

# 14. Run project tests

Run:

```bash
python -m pytest -q smac-jepa-wm/tests
python -m pytest -q smac-dreamer/tests
```

If a broad test suite cannot run because of an unavailable optional environment
dependency such as SMACLite, CUDA, or StarCraft assets:

1. run all source-only and protected-workflow tests;
2. record the exact failing test and missing dependency;
3. do not mark that test as passed;
4. do not delete the test merely to obtain a green suite.

---

# 15. Final status checks

Run:

```bash
set -euo pipefail

git ls-files smac-jepa-wm/smac_jepa/data

if rg -n \
  'ALLOW_MISSING_JEPA_DATA_SOURCE|loader source package.*absent|loader source is absent' \
  .
then
  echo "ERROR: stale missing-loader logic or documentation remains" >&2
  exit 1
fi

if find . -type f \( \
  -name '*.before_*' -o \
  -name '*.pre_*' -o \
  -name '*.bak*' -o \
  -name '.DS_Store' \
\) -print | grep -q .
then
  echo "ERROR: source backups or metadata remain" >&2
  exit 1
fi

if rg -n \
  'CURRENT_[A-Z0-9_]+\.txt|BUNDLE_ZIP|combined-upload|rnn_seqmem_exp40_event_balanced_5ep_20260709_083104' \
  smac-jepa-wm/scripts \
  smac-dreamer/scripts
then
  echo "ERROR: stale launcher dependency remains" >&2
  exit 1
fi

if git ls-files | grep -E \
  '(^|/)(logs?|runs?|wandb|checkpoints?|replay|results)(/|$)|\.(pt|pth|ckpt|npy|npz|log|zip|tar|tgz|pyc)$'
then
  echo "ERROR: generated artifact remains tracked" >&2
  exit 1
fi

echo "[OK] final repository hygiene checks passed"
```

---

# 16. Update `RETAINED_WORKFLOWS.md`

Ensure the documentation includes current commands for all four workflows.

## 16.1 Exp-40 JEPA

```bash
MANIFEST=/absolute/path/to/manifest.json \
OUT_DIR=/absolute/path/to/new_exp40_run \
WANDB=0 \
bash smac-jepa-wm/scripts/run_exp40_dreamer_event_balanced.sh
```

## 16.2 Exp-45 forecast

```bash
EXP40_CHECKPOINT=/absolute/path/to/exp40/checkpoint.pt \
MANIFEST=/absolute/path/to/manifest.json \
PIPE_DIR=/absolute/path/to/new_exp45_pipeline \
WANDB=0 \
bash smac-dreamer/scripts/run_exp45_full_train_eval_resilient.sh
```

## 16.3 Current actor-critic

```bash
SOURCE_CHECKPOINT=/absolute/path/to/tactical_v1_2_checkpoint.pt \
JEPA_CHECKPOINT=/absolute/path/to/exp40/checkpoint.pt \
RUN_DIR=/absolute/path/to/new_actor_critic_run \
FINAL_STEP=800000 \
bash smac-dreamer/scripts/run_actor_critic_h15_800k.sh
```

## 16.4 Current Option-Critic V9

```bash
SOURCE_CHECKPOINT=/absolute/path/to/tactical_v1_2_checkpoint.pt \
JEPA_CHECKPOINT=/absolute/path/to/exp40/checkpoint.pt \
RUN_DIR=/absolute/path/to/new_option_critic_v9_run \
FINAL_STEP=800000 \
bash smac-dreamer/scripts/run_option_critic_v9_anchor_safe_800k.sh
```

State explicitly:

- runtime `.npz` datasets are not bundled;
- production checkpoints are not bundled;
- source checkpoints must be supplied externally;
- the JEPA dataloader Python source is tracked under
  `smac-jepa-wm/smac_jepa/data/`.

---

# 17. Commit requirements

Before committing:

```bash
git status --short
git diff --check
git diff --stat
```

Review the staged diff and ensure:

- no config was deleted;
- no protected workflow file was deleted;
- only intended backup, metadata, stale utility, documentation, validator, and
  ignore-rule changes are present;
- no runtime artifact was added;
- all four JEPA dataloader files remain tracked.

Suggested commit:

```bash
git add -A
git commit -m "Finish repository cleanup and validate JEPA dataloaders"
```

Do not push until the retained workflow validator and available tests have run.

---

# 18. Definition of done

The task is complete only when all of the following are true:

- [ ] `smac-jepa-wm/smac_jepa/data/` contains the four restored Python source
      files.
- [ ] All four dataloader source files are tracked.
- [ ] Runtime `smac-jepa-wm/data/` remains ignored.
- [ ] `.DS_Store` is removed and ignored.
- [ ] All `*.before_*`, `*.pre_*`, and `*.bak*` source snapshots are removed.
- [ ] One-off debugging utilities are deleted unless a live dependency is
      documented.
- [ ] Old JEPA evaluators are deleted unless a live dependency is documented.
- [ ] `CLEANUP_REPORT.md` no longer says the dataloader package is missing.
- [ ] `ALLOW_MISSING_JEPA_DATA_SOURCE` is removed.
- [ ] `scripts/validate_retained_workflows.sh` compiles and imports the restored
      loaders.
- [ ] Dataset-window and visibility-memory contract tests run.
- [ ] Exp-40 trainer import and launcher syntax pass.
- [ ] Exp-45 trainer, forecast tests, and launcher syntax pass.
- [ ] Generic R2-Dreamer trainer and evaluator imports pass.
- [ ] Tactical-v1.2 source and tests remain.
- [ ] Current actor-critic H=15 / 800k launcher remains.
- [ ] Current Option-Critic V9 anchor-safe 8-slot H=15 / 800k launcher, source,
      audits, and tests remain.
- [ ] The canonical forecast → actor-critic → Option-Critic pipeline remains.
- [ ] No active launcher depends on `CURRENT_*.txt`, a bundle ZIP, or an old
      historical workspace path.
- [ ] No logs, runs, checkpoints, replay buffers, runtime datasets, NumPy
      outputs, or archives are tracked.
- [ ] Every file under `smac-dreamer/configs/` remains.
- [ ] `RETAINED_WORKFLOWS.md` documents all four workflows.
- [ ] `CLEANUP_REPORT.md` truthfully records validation results and unresolved
      environment dependencies.

Do not declare completion merely because the repository is smaller. Completion
means the remaining repository is clean **and** the four retained training
workflows are structurally executable.
