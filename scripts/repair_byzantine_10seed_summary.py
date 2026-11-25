#!/usr/bin/env python3
"""Patch journal benchmark rows/summary after FL re-eval for seeds 45–46."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_comprehensive_validation import OUT_ROOT, _fl_eval_summary

TUNING = ROOT / "evaluation_results" / "acfl_tuning"
REPAIR = [
    ("byzantine", "fedavg", 45, "journal_byzantine_seed45_fedavg"),
    ("byzantine", "acfl_full", 45, "journal_byzantine_seed45_acfl_full"),
    ("byzantine", "acfl_full", 46, "journal_byzantine_seed46_acfl_full"),
]


def _summarize_byzantine(rows: list) -> dict:
    summary = {}
    for name in ("fedavg", "acfl_full"):
        subset = [r for r in rows if r.get("attack") == "byzantine" and r["defender"] == name]
        f1s = [r["mean_final_f1"] for r in subset if r.get("mean_final_f1") is not None]
        if f1s:
            summary[f"byzantine/{name}"] = {
                "mean_f1": statistics.mean(f1s),
                "std_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                "f1_values": f1s,
                "n_seeds": len(f1s),
            }
    fed = sorted(
        [r for r in rows if r.get("attack") == "byzantine" and r["defender"] == "fedavg"],
        key=lambda r: r["seed"],
    )
    acfl = sorted(
        [r for r in rows if r.get("attack") == "byzantine" and r["defender"] == "acfl_full"],
        key=lambda r: r["seed"],
    )
    fed_f1 = [r["mean_final_f1"] for r in fed if r.get("mean_final_f1") is not None]
    acfl_f1 = [r["mean_final_f1"] for r in acfl if r.get("mean_final_f1") is not None]
    n = min(len(fed_f1), len(acfl_f1))
    if n:
        gaps = [acfl_f1[i] - fed_f1[i] for i in range(n)]
        summary["byzantine/acfl_full_minus_fedavg"] = {
            "mean_gap": statistics.mean(gaps),
            "std_gap": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
            "gaps": gaps,
        }
    return summary


def main() -> int:
    rows_path = TUNING / "drone_journal_benchmark.json"
    summary_path = TUNING / "drone_journal_benchmark_summary.json"
    rows = json.loads(rows_path.read_text())

    for attack, defender, seed, scenario in REPAIR:
        fl_path = Path(OUT_ROOT) / scenario / "fl_eval" / "evaluation_report.json"
        if not fl_path.is_file():
            print(f"MISSING {fl_path}")
            continue
        fl = json.loads(fl_path.read_text())
        fl_sum = _fl_eval_summary(fl)
        f1 = fl_sum.get("mean_final_f1")
        print(f"{scenario}: mean_final_f1={f1}")

        metrics_path = Path(OUT_ROOT) / scenario / "metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            metrics["fl_eval"] = fl
            metrics["fl_eval_rc"] = 0
            metrics_path.write_text(json.dumps(metrics, indent=2))

        for row in rows:
            if (
                row.get("attack") == attack
                and row.get("defender") == defender
                and row.get("seed") == seed
            ):
                row["mean_final_f1"] = f1
                row["mean_final_accuracy"] = fl_sum.get("mean_final_accuracy")
                row["num_checkpoints"] = fl_sum.get("num_checkpoints")

    rows_path.write_text(json.dumps(rows, indent=2, default=str))
    full_summary = json.loads(summary_path.read_text())
    full_summary.update(_summarize_byzantine(rows))
    summary_path.write_text(json.dumps(full_summary, indent=2))

    print("\nUpdated Byzantine 10-seed summary:")
    for key in ("byzantine/fedavg", "byzantine/acfl_full", "byzantine/acfl_full_minus_fedavg"):
        print(f"  {key}: {json.dumps(full_summary[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
