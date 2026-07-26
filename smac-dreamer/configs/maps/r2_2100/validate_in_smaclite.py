#!/usr/bin/env python3
"""Dynamic SMAClite validation for r2_smaclite_general_2100_configs.

Run from the smac-dreamer repository root after copying the dataset to configs/maps/:

PYTHONPATH=src:external/r2dreamer:external/smaclite \
python configs/maps/r2_smaclite_general_2100_configs/validate_in_smaclite.py \
  --root configs/maps/r2_smaclite_general_2100_configs --episodes 3 --max-steps 200
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from smaclite.env.util.direction import Direction


def choose_actions(env):
    avail = env.get_avail_actions()
    actions = []
    for i in range(env.n_agents):
        valid = np.flatnonzero(avail[i])
        if i not in env.agents:
            actions.append(0); continue
        unit = env.agents[i]
        target_pool = env.enemies if unit.combat_type.name == "DAMAGE" else env.agents
        attack_choices = [a for a in valid if a >= 6]
        if attack_choices:
            scored = []
            for a in attack_choices:
                tid = a - 6
                if tid not in target_pool: continue
                target = target_pool[tid]
                if unit.combat_type.name == "HEALING":
                    score = (target.hp + target.shield) / max(target.max_hp + target.max_shield, 1e-8)
                else:
                    score = target.hp + target.shield
                scored.append((score, a))
            if scored:
                actions.append(min(scored)[1]); continue
        if unit.combat_type.name == "HEALING":
            actions.append(1 if 1 in valid else int(valid[0])); continue
        if env.enemies:
            enemy = min(env.enemies.values(), key=lambda e: float(np.linalg.norm(e.pos - unit.pos)))
            prefs = []
            for a in (2, 3, 4, 5):
                if a not in valid: continue
                dest = unit.pos + Direction(a - 2).dx_dy * 2
                prefs.append((float(np.linalg.norm(dest - enemy.pos)), a))
            if prefs:
                actions.append(min(prefs)[1]); continue
        actions.append(1 if 1 in valid else int(valid[0]))
    return [int(a) for a in actions]


def run_one(path, seed, max_steps):
    from smaclite.env.smaclite import SMACliteEnv
    env = SMACliteEnv(map_file=str(path), seed=seed)
    obs, info = env.reset(seed=seed)
    assert np.isfinite(np.asarray(obs)).all(), path
    assert np.isfinite(env.get_state()).all(), path
    total = 0.0; won = False; done = False; truncated = False
    for t in range(max_steps):
        obs, reward, done, truncated, info = env.step(choose_actions(env))
        assert np.isfinite(np.asarray(obs)).all(), path
        assert np.isfinite(float(reward)), path
        total += float(reward)
        if done or truncated:
            won = bool(info.get("battle_won", False)); break
    env.close()
    return won, not (done or truncated), total, t + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--split", choices=("all", "train", "validation", "blind_iid", "blind_compositional"), default="all")
    args = p.parse_args()
    pattern = "*/*.json" if args.split == "all" else f"{args.split}/*.json"
    maps = sorted((args.root / "configs").glob(pattern))
    rows = []
    for idx, path in enumerate(maps, 1):
        wins = timeouts = 0; returns = []; lengths = []
        for ep in range(args.episodes):
            win, timeout, ret, length = run_one(path, 1000 + ep, args.max_steps)
            wins += int(win); timeouts += int(timeout); returns.append(ret); lengths.append(length)
        row = {"path": str(path), "win_rate": wins / args.episodes, "timeout_rate": timeouts / args.episodes,
               "mean_return": sum(returns) / len(returns), "mean_length": sum(lengths) / len(lengths)}
        rows.append(row)
        print(f"[{idx:04d}/{len(maps)}] {path.name}: win={row['win_rate']:.2f} timeout={row['timeout_rate']:.2f}")
    out = args.root / f"dynamic_validation_{args.split}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
