# Multimap Training — Verification Steps

Commands to run in the `smac-r2` conda env (Anaconda Prompt) to verify the multimap
implementation end-to-end. What was already verified on the dev machine (no torch/smaclite)
is noted; the rest must run where the full stack is installed.

## Already verified (dev machine, torch-free)
- `python -m pytest tests/` → **37 passed, 20 skipped** (skips need smaclite).
  - New: `tests/test_map_discovery.py` (split / train-max padding / safety-net) — 8 pass.
  - New: `tests/test_reward_registry.py` (all 3 rewards, dense_v3 telescoping, hashing) — 11 pass.
- All project env modules import torch-free; `MapSampler.from_entries` works.
- `dense_v3` term math: no-op telescopes to 0, HP-drop +, ally-death −, terminal win/loss, positioning off by default.
- §0 ParallelEnv routing spike: **PASS** (maps change on auto-reset, workers de-correlated, fixed shapes, no eval→buffer leak, `log_*` forwarding works).
- No-regression refactor: `build_phase4_manifest.py` now imports `validate_map`/`sha256_file` from the shared `map_discovery` module.

## Run in the conda env

### 1. Full test suite (env/padding tests now execute)
```bat
conda activate smac-r2
cd /d "c:\Users\User\OneDrive - National University of Singapore\Documents\smac-dreamer"
python -m pytest tests\ -v
```
Expect all previously-skipped env/padding tests to pass (simulator present).

### 2. No-regression: Phase-4 manifest tooling still works after the scan refactor
```bat
python Archive\build_phase4_manifest.py --map_dir configs\maps\500map_v1 ^
  --output_manifest results\_regression_phase4_manifest.yaml ^
  --output_report results\_regression_phase4_report.json ^
  --seed 42 --train_ratio 0.80 --validation_ratio 0.10 --test_ratio 0.10
```
Compare the produced manifest's split counts / padding against the committed
`configs\maps\phase4_manifest.yaml` (entry set + padding dims should match).

### 3. Provide the easy-map folder
Put ~50 ally-advantaged map JSONs under `configs\maps\easy50\` (or change `maps_folder`
in `configs\multimap.yaml`). Each must be a valid SMAClite map JSON (same schema as
`configs\maps\500map_v1\*.json`).

### 4. Discovery smoke (scan + split + padding + safety-net)
```bat
python -c "import sys; [sys.path.insert(0,p) for p in (r'src',r'external\smaclite')]; from smacdreamer.envs.map_discovery import discover, SplitSpec; tr,te,pad=discover('configs/maps/easy50', SplitSpec('ratio',0.8,0)); print('train',len(tr),'test',len(te),'pad',pad)"
```
Confirms: train/test disjoint, padding from TRAIN-max, all maps fit (else fail-fast names the offender).

### 5. Short multimap training run
```bat
python scripts\train_r2dreamer_smaclite_multimap.py --config configs\multimap.yaml --steps 500
```
Confirm:
- discovery prints the split + resolved padding;
- WM losses (`train/loss/*`) appear;
- `log_*` keys flow (invalid-action + per-term `log_reward_term_*`);
- periodic held-out eval logs `episode/eval_battle_won` + `episode/eval_reward_original`;
- `run_config.json` (or W&B config) records resolved reward name+params+padding;
- run name carries the reward hash; `latest.pt` written.
- **Buffer-leak guard:** `replay_buffer.count()` does not change across an eval (already
  guaranteed by code; the §0 spike checks the code path).

With W&B:
```bat
python scripts\train_r2dreamer_smaclite_multimap.py --config configs\multimap.yaml --steps 500 ^
  --logdir logs\r2dreamer\multimap_easy50
```
(set `wandb.project` in the YAML to enable W&B.)

### 6. Held-out evaluation
```bat
python scripts\evaluate_multimap.py --config configs\multimap.yaml ^
  --checkpoint logs\r2dreamer\multimap\latest.pt --episodes-per-map 16
```
Confirm:
- evaluates **only** held-out test maps (asserts no train map);
- per-map win rate + Wilson CI + original return;
- **headline = across-map win rate + CI** (each map one sample);
- JSON report under `results\multimap_eval_<folder>.json`.

## Notes / known limitations
- Eval action selection is deterministic (actor mode); the env's action sanitiser is the
  final safety net. Policy-side eval masking is a separate later stage.
- Generalisation is scoped to "unseen maps no larger than the largest TRAIN map" (padding
  from train-max). A held-out map exceeding train-max fails fast — exclude it or raise an
  explicit `padding` cap in `configs\multimap.yaml`.
- `dense_v3` invariance is approximate (DreamerV3 symlog/twohot + return-norm); shaping
  `gamma` must equal the agent discount. Eval reports ORIGINAL return to detect distortion.
- `dense_v3` `w_hp`/`w_ally` defaults (0.1) are placeholders — run a few episodes, read the
  per-episode `log_reward_term_*_ep_sum` vs the terminal magnitude, and tune so shaping is
  ~10–20% of |episode reward| before committing final magnitudes.
- Windows: keep `env_num: 1`. Linux/Kubeflow: raise `env_num`.
