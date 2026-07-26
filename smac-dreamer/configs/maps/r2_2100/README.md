# R2-Dreamer × SMAClite General Policy 2100-map dataset

This package contains **2100 deterministic SMAClite maps** designed as a broader successor to the 1300-map pool while remaining shape-compatible with the existing R2-650/R2-1300 model.

## Split

- `configs/train`: 1,200 maps from 100 seen composition families, 12 variants per family
- `configs/validation`: 200 maps from the same 100 families, 2 unseen variants per family
- `configs/blind_iid`: 300 maps from the same 100 families, 3 unseen variants per family
- `configs/blind_compositional`: 400 maps from 40 composition families absent from train/validation, 10 variants per family

The 100 seen families retain all **50 original R2-1300 families** and add **50 new families**. The compositional blind split retains all **20 original held-out families** and adds **20 new held-out families**.

## Main improvements

1. **Broader compositions:** 100 train families instead of 50, including balanced and slightly ally-disadvantaged matchups.
2. **Balanced training terrain:** exactly 400 SIMPLE, 400 NARROW and 400 OCTAGON training maps.
3. **Choke traversal:** medium NARROW maps and some evaluation maps start teams on opposite sides of the central gate.
4. **More layout orientations:** horizontal, vertical and diagonal deployments, alongside four formation modes.
5. **Transfer compatibility:** maximum allies=9, enemies=10, actions=16 and the same nine-entry global unit vocabulary.
6. **Curriculum metadata:** every map has a stable family ID, family origin and a static seed difficulty score. Replace the seed score with empirical win/timeout/EHP statistics during training.
7. **Strict split discipline:** validation is for checkpoint selection; blind splits are for post-training evaluation only.

## Static validation

- Total maps: 2100
- Validation errors: 0
- Unique semantic configurations: 2100
- Seed: 18062026

Static checks cover schema, counts, unit vocabulary, shield consistency, medivac targetability, placement emulation, terrain walkability, overlap, engagement bucket and semantic duplicates.

## Required dynamic validation

Static validation cannot prove practical winnability or navigation quality. Run at least one scripted episode over all maps before an expensive training run:

```bash
PYTHONPATH=src:external/r2dreamer:external/smaclite \
python configs/maps/r2_smaclite_general_2100_configs/validate_in_smaclite.py \
  --root configs/maps/r2_smaclite_general_2100_configs \
  --episodes 1 --max-steps 200
```

Then rerun suspicious or hard configurations with 3–5 episodes.

## Suggested continuation setup

- Resume the existing checkpoint with the same observation/action dimensions.
- Use shuffled round-robin for the first 2–3 full passes.
- Then use a mixture such as 75% family-balanced uniform + 25% empirical hard-map sampling.
- Keep blind splits untouched until final evaluation.
- Select checkpoints using macro validation win rate, with original return as tie-breaker.
