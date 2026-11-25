#!/usr/bin/env python3
"""Rebuild combined Layer~3 rows from full-stack FedAvg/full cells.

Keeps the benign baseline from the existing combined JSON; for every shared
learning attack (Byzantine, gradient scaling, backdoor, label flip, model
replacement) copies checkpoint + Layer~3 metrics from the matching full-stack
row so both tables report identical AUROC on the same 3-client checkpoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ACFL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACFL_ROOT))

from run_drone_combined_layer3_benchmark import (  # noqa: E402
    DEFENDERS,
    LEARNING_ATTACKS,
    LAYER3_ROOT,
    _summarize,
    _write_outputs,
)

STACK_ROOT = ACFL_ROOT / "evaluation_results" / "full_stack"
COMBINED_JSON = (
    ACFL_ROOT / "evaluation_results" / "combined_layer3" / "combined_layer3_benchmark.json"
)

STACK_TO_DEFENDER = {"fedavg": "fedavg", "full": "acfl_full"}
SHARED_ATTACKS = {
    "gradient_scaling",
    "byzantine",
    "backdoor",
    "label_flip",
    "model_replacement",
}


def _attack_meta(name: str) -> dict:
    for a in LEARNING_ATTACKS:
        if a["name"] == name:
            return a
    raise KeyError(name)


def _defender_meta(name: str) -> dict:
    for d in DEFENDERS:
        if d["name"] == name:
            return d
    raise KeyError(name)


def main() -> None:
    fs_rows = json.loads((STACK_ROOT / "full_stack_benchmark.json").read_text())
    existing = json.loads(COMBINED_JSON.read_text()) if COMBINED_JSON.is_file() else []

    benign = [r for r in existing if r.get("learning_attack") == "benign"]
    fs_index = {
        (r["attack"], r["stack"], int(r["seed"])): r
        for r in fs_rows
        if r["stack"] in STACK_TO_DEFENDER and r["attack"] in SHARED_ATTACKS
    }

    new_rows = list(benign)
    missing = []
    for attack_name in sorted(SHARED_ATTACKS):
        atk = _attack_meta(attack_name)
        for stack, defender_name in STACK_TO_DEFENDER.items():
            defender = _defender_meta(defender_name)
            for seed in range(42, 52):
                src = fs_index.get((attack_name, stack, seed))
                if not src or src.get("layer3_auroc") is None:
                    missing.append((attack_name, stack, seed))
                    continue
                # Prefer full-stack's own layer3 report path; fall back to combined naming.
                report = src.get("layer3_report") or str(
                    LAYER3_ROOT / f"{attack_name}_{defender_name}_seed{seed}.json"
                )
                new_rows.append(
                    {
                        "seed": seed,
                        "learning_attack": attack_name,
                        "learning_attack_type": atk["attack_type"],
                        "defender": defender_name,
                        "defender_family": defender.get("defender_family"),
                        "scenario": src.get("scenario"),
                        "mean_final_f1": src.get("layer2_mean_f1"),
                        "trust_auc": src.get("layer1_trust_auc"),
                        "checkpoint": src.get("checkpoint"),
                        "layer3_report": report,
                        "layer3_auroc": src.get("layer3_auroc"),
                        "layer3_f1": src.get("layer3_f1"),
                        "layer3_precision": src.get("layer3_precision"),
                        "layer3_recall": src.get("layer3_recall"),
                        "layer3_waypoint_auroc": src.get("layer3_waypoint_auroc"),
                        "layer3_gps_auroc": src.get("layer3_gps_auroc"),
                        "layer3_threshold": None,
                        "layer3_waypoint_f1": None,
                        "layer3_gps_f1": None,
                        "probe_file": None,
                    }
                )

    if missing:
        print(f"WARNING: {len(missing)} full-stack cells missing Layer 3; skipping:")
        for item in missing[:20]:
            print(" ", item)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    summary = _summarize(new_rows)
    _write_outputs(new_rows, summary)

    # Quick identity check on shared means
    fs_summary = json.loads((STACK_ROOT / "full_stack_benchmark_summary.json").read_text())
    print("\nShared-cell AUROC identity check (combined vs full-stack):")
    for attack_name in sorted(SHARED_ATTACKS):
        for stack, defender_name in STACK_TO_DEFENDER.items():
            c = summary.get(f"{attack_name}/{defender_name}", {})
            f = fs_summary.get(f"{attack_name}/{stack}", {})
            cm = c.get("mean_layer3_auroc")
            fm = f.get("mean_layer3_auroc")
            match = (
                cm is not None
                and fm is not None
                and abs(cm - fm) < 1e-12
            )
            status = "OK" if match else "MISMATCH"
            print(
                f"  [{status}] {attack_name}/{defender_name}: "
                f"combined={cm} fullstack={fm}"
            )
    print(f"\nWrote combined rows: {len(new_rows)} "
          f"(benign={len(benign)}, shared={len(new_rows) - len(benign)})")


if __name__ == "__main__":
    main()
