# Repository Cleanup Report

## Result

The extracted repository was reduced to canonical JEPA and Dreamer source,
configuration, tests, documentation, and protected launchers. Historical
datasets, logs, checkpoints, replay state, evaluation output, bundles, backup
trees, pointers, caches, and obsolete experiment launchers were removed.

The directory is not an independent Git worktree. It is untracked inside the
home-directory Git repository at `/Users/kialok`, so no cleanup branch was
created.

## Deleted Paths

The exact deletion manifest is recorded in `deleted_paths.txt`. It contains:

- all root bundle, installer, backup, `preserve_before_*`, archived pipeline,
  overnight log, duplicate `smac_jepa`, and `CURRENT_*.txt` paths specified by
  `CODEX_JEPA_DREAMER_REPO_CLEANUP.md`;
- generated JEPA/Dreamer logs, outputs, local datasets, checkpoints, replay,
  caches, archives, reports, and fix directories;
- unreferenced old JEPA trainers and launchers listed as cleanup candidates;
- superseded Option-Critic V2-V6, tactical mixture/hardening, unified-priority,
  and older forecast pipeline launchers and audits;
- obsolete top-level experiment/evaluation wrappers and source backup suffixes.

No `.pt`, `.pth`, `.ckpt`, `.npy`, `.npz`, `.log`, `.zip`, `.tar`, `.tar.gz`,
`.tgz`, or `.pyc` artifact is intentionally retained.

## Moved and Reconciled

- Moved the Exp-40 rollout gallery evaluator to
  `smac-jepa-wm/tools/eval_exp40_rollout_gallery.py`.
- Moved its launcher to
  `smac-jepa-wm/scripts/run_exp40_rollout_gallery.sh`.
- Reconciled nine R2-2100 configuration/support files from local assets into
  `smac-dreamer/configs/maps/r2_2100/`: README, manifests, family catalogs,
  curriculum metadata, generator, and validator.
- Excluded generated checksums and the historical validation report.
- Preserved all 4,505 original Dreamer config files; the final count is 4,514
  after the nine additions.

## Modified Workflows

- Exp-40 now requires explicit `MANIFEST` and `OUT_DIR`, resolves its own root,
  accepts `PYTHON`/`PY`, prints resolved paths, and retains checkpoint
  compatibility validation.
- Exp-45 training/evaluation requires external manifest/checkpoint paths and
  uses canonical source. Runtime ZIP installation and historical defaults were
  removed.
- Tactical-v1.2, actor-critic H=15/800k, and Option-Critic V9 require explicit
  source checkpoint, JEPA checkpoint, and fresh output directory.
- Source run metadata and exact SHA-256 are optional. A supplied hash is
  enforced; otherwise checkpoint architecture/state is validated semantically.
- Actor-critic and Option-Critic comparison launchers still enforce exactly
  800,000 new environment steps.
- The canonical combined pipeline passes all external inputs explicitly and
  runs forecast, ordinary actor-critic, and Option-Critic V9 sequentially.
- Checkpoint-free audit modes retain syntax, source, config, and test checks.
- `scripts/validate_retained_workflows.sh` provides the canonical repository
  validator.

## Validation

Passed:

- shell syntax for every retained shell script;
- Python compilation for retained JEPA, Dreamer, tool, and launcher trees;
- protected structural validator;
- stale active-launcher search for historical paths, bundle installers, and
  root runtime pointers;
- generated artifact and runtime-directory scans;
- configuration preservation comparison: 4,505 original, zero removed, nine
  canonical metadata additions.

Environment-bound:

- The extracted copy does not contain the Python loader source package
  `smac-jepa-wm/smac_jepa/data/`. Exp-40/45 import and dataset tests remain
  blocked until that source is supplied.
- No usable Python training environment was present. An isolated `uv` attempt
  created an empty environment but did not resolve `torch`, `omegaconf`, or
  `pytest`; protected runtime unit tests therefore were not executed.
- No long training job was launched.

Run the full checks after restoring the loader source and dependencies:

```bash
PY=/path/to/python bash scripts/validate_retained_workflows.sh
pytest -q smac-jepa-wm/tests
pytest -q smac-dreamer/tests
```
