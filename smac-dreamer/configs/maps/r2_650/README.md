# R2-Dreamer × SMAClite 650-map benchmark

This directory contains **650 deterministic custom SMAClite map JSON files** generated for combat-rich R2-Dreamer training and blind evaluation.

## Split

- `configs/train`: 400 maps from 50 seen composition families (8 variants each)
- `configs/validation`: 50 maps from the same 50 families (1 unseen variant each)
- `configs/blind_iid`: 100 maps from the same 50 families (2 unseen variants each)
- `configs/blind_compositional`: 100 maps from 20 composition families absent from train/validation (5 variants each)

The files contain only fields accepted by SMAClite's `MapInfo`; research metadata is stored separately in `manifest.jsonl` and `manifest.csv`.

## Design choices

1. **Combat-rich distances:** train contains exactly 80 immediate, 140 near, 120 medium, and 60 far variants.
2. **Winnability proxy:** every family gives allies a static combat-value advantage between 1.04× and 1.85×. This is a generation filter, not a substitute for empirical simulation.
3. **Shield correctness:** each faction is internally either entirely shielded or entirely unshielded. The faction flags exactly match the bundled unit definitions.
4. **Medivac safety:** a medivac is only used when the opposing composition contains a unit capable of targeting air.
5. **Global type vocabulary:** every map uses the same nine-entry `unit_type_ids` mapping to prevent map-local one-hot meanings.
6. **Spawn checks:** the generator emulates SMAClite's square group placement, rejects overlaps, checks map bounds, and checks ground-unit spawn cells against the selected terrain preset.
7. **Reproducibility:** fixed seed `26052026`; rerun `generate_r2_smaclite_650.py` to recreate the dataset.

## Files

- `generate_r2_smaclite_650.py`: self-contained deterministic generator and static validator
- `validate_in_smaclite.py`: dynamic environment smoke test plus scripted-policy evaluation
- `manifest.jsonl` / `manifest.csv`: per-map family, composition, engagement, formation, terrain and difficulty metadata
- `split_manifest.json`: exact files in each split
- `family_catalog.json`: composition families and split status
- `validation_report.json`: static validation results and aggregate distributions
- `checksums.sha256`: content checksums

## Static validation result

- Files: 650
- Errors: 0
- Unique content hashes: 650
- Seed: 26052026

## Required dynamic validation

Static checks cannot prove game-theoretic winnability. After copying this folder into the repository, run:

```bash
PYTHONPATH=src:external/r2dreamer:external/smaclite \
python configs/maps/r2_650/validate_in_smaclite.py \
  --root configs/maps/r2_650 \
  --episodes 5 \
  --max-steps 200
```

Use the resulting `dynamic_validation.csv` to remove or rebalance maps with high timeout rate or zero scripted-policy wins before the final expensive R2-Dreamer run. A scripted baseline is deliberately only a filter; final winnability should also be checked with a trained reference policy.

## Loader recommendation

Do not randomly split one map folder at runtime. Point the trainer and evaluator at the explicit split directories or consume `split_manifest.json`. Select checkpoints using **blind/validation win rate and original SMAClite return**, never shaped return.
