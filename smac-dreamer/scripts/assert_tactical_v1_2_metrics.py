#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path

REQUIRED = [
    "train/tactic/mutual_information_normalized",
    "train/tactic/effect_js",
    "train/tactic/residual_to_base_ratio",
    "train/tactic/base_kl_mean",
    "train/tactic/base_kl_max",
    "train/tactic/base_kl_loss",
    "train/tactic/action_flip_rate",
    "train/tactic/mi_shortfall",
    "train/imag_post_mask_invalid_sample_rate",
]

def main():
    p=argparse.ArgumentParser(); p.add_argument("run",type=Path); a=p.parse_args()
    path=a.run/"metrics.jsonl"
    rows=[]
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: pass
    latest={}
    for row in rows:
        latest.update({k:v for k,v in row.items() if isinstance(v,(int,float))})
    missing=[k for k in REQUIRED if k not in latest]
    if missing: raise SystemExit(f"[FAIL] missing metrics: {missing}")
    bad=[k for k in REQUIRED if not math.isfinite(float(latest[k]))]
    if bad: raise SystemExit(f"[FAIL] non-finite metrics: {bad}")
    if float(latest["train/imag_post_mask_invalid_sample_rate"])>1e-6:
        raise SystemExit("[FAIL] post-mask invalid action rate is nonzero")
    if float(latest["train/tactic/base_kl_mean"])>0.10:
        raise SystemExit("[FAIL] mean tactical KL to base exceeds 0.10")
    if float(latest["train/tactic/residual_to_base_ratio"])>0.50:
        raise SystemExit("[FAIL] residual/base ratio exceeds 0.50")
    print("[OK] Tactical Mixture v1.2 metrics are finite and within safety bounds")

if __name__=="__main__": main()
