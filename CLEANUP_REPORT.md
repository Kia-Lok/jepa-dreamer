# Repository Cleanup Report

## Result

The extracted repository was reduced to canonical JEPA and Dreamer source,
configuration, tests, documentation, and protected launchers. Historical
datasets, logs, checkpoints, replay state, evaluation output, bundles, backup
trees, pointers, caches, and obsolete experiment launchers were removed.

The cleaned directory is now an independent Git repository with `main` tracking
`origin/main`.

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
- the remaining seven tracked R2-Dreamer/JEPA `*.pre_*` and `*.before_*`
  snapshots, all `.DS_Store` files, and six unreferenced one-off debugging
  utilities.

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

## Restored JEPA dataset-loader source

The Python source package `smac-jepa-wm/smac_jepa/data/` has been restored from
the previous canonical JEPA repository. It contains:

- `__init__.py`
- `dataset.py`
- `markov_rollout_dataset.py`
- `markov_rollout_visibility_dataset.py`

All four files are tracked source. Runtime `.npz` datasets remain excluded and
must be supplied externally.

The retained-workflow validator now compiles these loaders, imports the base and
visibility-aware datasets, runs the dataset-window and anchored-memory contract
tests, checks the explicit visibility-mask contract, and validates Exp-40 and
Exp-45 trainer imports.

## Retained older-named evaluators

The following evaluators remain because they are live workflow dependencies:

- `eval_jepa_exp31_exp33.py` is imported by the anchored evaluator.
- `eval_jepa_exp31_exp33_anchored.py` is used by the Exp-40 rollout gallery and
  Exp-45 ordinary/hidden evaluation.
- `eval_jepa_hidden_belief_exp31_exp33.py` is the Exp-45 hidden evaluator.
- `eval_rnn_seqmem_dreamer_probe.py` is the active ordinary forecast fallback
  used by the direct and anchored evaluation tools.

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

- `PY=/private/tmp/jepa-dreamer-cleanup-venv/bin/python bash
  scripts/validate_retained_workflows.sh`, including loader compilation/imports,
  dataset-window tests (`3 passed`), the direct anchored-memory contract,
  Exp-40/Exp-45 imports, Dreamer CLI imports, shell syntax, and all three
  checkpoint-free static audits;
- Exp-45 forecast tests (`9 passed`);
- protected Tactical-v1.2 and Option-Critic V9 tests (`34 passed`);
- the V9-applicable hierarchy set (`53 passed, 12 deselected`) and the
  additional V9 static-audit set (`27 passed`);
- `python -m pytest -q smac-jepa-wm/tests` (`24 passed`);
- full Python compilation for retained JEPA, Dreamer, tool, and launcher trees;
- stale active-launcher search for historical paths, bundle installers, and
  root runtime pointers;
- generated artifact and runtime-directory scans;
- configuration preservation comparison: 4,505 original, zero removed, nine
  canonical metadata additions.

The broad Dreamer suite was also run outside the filesystem sandbox so PyTorch
shared-memory tests could execute. Its result was `296 passed, 1 skipped, 32
failed`. The skipped source-parity test requires the removed historical module
`smac_jepa.train_markov_rollout_rnn_visibility_seqmem_experiments`.

The 32 unresolved broad-suite failures are retained and documented rather than
hidden or deleted:

- 12 pre-V9 hierarchy assertions conflict with the current anchor-safe V9
  contract; these are the same tests excluded by the canonical V9 static audit;
- 10 tests in `test_tactical_policy.py` and
  `test_tactical_policy_hardened.py` assert pre-v1.2 initialization, metadata,
  and gating behavior, while the protected v1.2 suite passes;
- eight `test_jepa_checkpoint.py` cases monkeypatch the older two-class JEPA
  import contract, while the current checkpoint loader requires the anchored
  memory class as its third import;
- two older JEPA Dreamer update fixtures do not initialize the current adaptive
  priority state.

The shared test setup now includes `external/r2dreamer` and preloads the real
pure-PyTorch action-mask module, eliminating two collection-order errors in the
broad suite.

No long training job is part of cleanup validation.
