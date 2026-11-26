#!/usr/bin/env python3
"""
Build class-balanced packet FL client splits from labeled SITL packet CSVs.

Rebalances each client's packet rows so attack labels are closer to
``--target-attack-ratio`` (default 0.25), improving classification F1
without changing the feature schema.

Usage:
    python3 balance_packet_clients.py \\
        --input data/sitl_dataset/fl/packets \\
        --output data/sitl_dataset/fl/packets_balanced
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def balance_file(path: str, out_path: str, target_ratio: float, seed: int) -> dict:
    df = pd.read_csv(path)
    if "Label" not in df.columns:
        raise ValueError(f"{path} missing Label column")

    normal = df[df["Label"] == 0]
    attack = df[df["Label"] == 1]
    n_attack = len(attack)
    n_normal = len(normal)
    if n_attack == 0:
        df.to_csv(out_path, index=False)
        return {"file": os.path.basename(path), "rows": len(df), "attack": 0, "balanced": False}

    rng = np.random.default_rng(seed)
    # target: attack / (attack + normal_sampled) = target_ratio
    desired_attack = n_attack
    desired_normal = int(round(desired_attack * (1.0 - target_ratio) / target_ratio))
    desired_normal = max(desired_normal, n_attack)  # at least 1:1 if very few attacks
    desired_normal = min(desired_normal, n_normal)

    if desired_normal < n_normal:
        normal = normal.sample(n=desired_normal, random_state=seed)
    out = pd.concat([normal, attack], ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    return {
        "file": os.path.basename(path),
        "rows": len(out),
        "attack": int(out["Label"].sum()),
        "attack_ratio": float(out["Label"].mean()),
        "balanced": True,
    }


def main():
    ap = argparse.ArgumentParser(description="Balance packet FL client CSVs")
    ap.add_argument("--input", required=True, help="input packets/ directory")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--target-attack-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input, "packet_client_*.csv")))
    if not files:
        raise SystemExit(f"No packet_client_*.csv in {args.input}")

    stats = []
    for i, path in enumerate(files):
        m = re.search(r"packet_client_(\d+)", os.path.basename(path))
        cid = int(m.group(1)) if m else i
        out = os.path.join(args.output, os.path.basename(path))
        stats.append(balance_file(path, out, args.target_attack_ratio, args.seed + cid))

    print(f"Balanced {len(stats)} clients -> {args.output}")
    for s in stats:
        print(f"  {s['file']}: {s['rows']} rows, attack={s.get('attack', 0)}, "
              f"ratio={s.get('attack_ratio', 0):.3f}")


if __name__ == "__main__":
    main()
