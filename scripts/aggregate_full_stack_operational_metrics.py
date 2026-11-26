#!/usr/bin/env python3
"""Aggregate Layer~3 operational metrics across FedAvg, classical baselines, and ACFL.

Reads per-seed ``evaluate_layer3_sitl.py`` reports (per-checkpoint F1-tuned
threshold) from:

  * ``evaluation_results/full_stack/layer3/<defender>_<attack>_seed*.json``
    for poisoned FL checkpoints (FedAvg / ACFL-full / coordinate_median /
    trimmed_mean / krum);
  * ``evaluation_results/combined_layer3/layer3/benign_{fedavg,acfl_full}_seed*.json``
    for the matched 12-round benign control (classical baselines have no
    matched benign runs; those cells are omitted).

Writes ``evaluation_results/full_stack/full_stack_operational_metrics.json``.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
FS_LAYER3 = ROOT / "evaluation_results" / "full_stack" / "layer3"
CB_LAYER3 = ROOT / "evaluation_results" / "combined_layer3" / "layer3"
OUT_JSON = ROOT / "evaluation_results" / "full_stack" / "full_stack_operational_metrics.json"
LATENCY_JSON = ROOT / "evaluation_results" / "full_stack" / "full_stack_latency.json"

# (stack_key used in report filenames, benign combined-layer3 name or None, label)
STACKS = [
    ("fedavg", "fedavg", "FedAvg"),
    ("coordinate_median", None, "Coord.\\ median"),
    ("trimmed_mean", None, "Trimmed mean"),
    ("krum", None, "Krum"),
    ("full", "acfl_full", "ACFL (full)"),
]
ATTACKS = [
    ("benign", "Benign (no FL poison)"),
    ("byzantine", "Byzantine"),
    ("gradient_scaling", "Gradient scaling"),
    ("backdoor", "Backdoor"),
    ("label_flip", "Label flip"),
    ("model_replacement", "Model replacement"),
]
SEEDS = list(range(42, 52))

METRIC_KEYS = (
    "mean_delay_seconds",
    "mean_detection_rate",
    "aggregate_false_alarm_rate",
    "benign_missions_mean_false_alarms_per_hour",
    "threshold",
    "aggregate_auroc",
)


def _report_path(
    stack_fs: str,
    stack_cb: Optional[str],
    attack: str,
    seed: int,
) -> Optional[Path]:
    if attack == "benign":
        if stack_cb is None:
            return None
        return CB_LAYER3 / f"benign_{stack_cb}_seed{seed}.json"
    return FS_LAYER3 / f"{stack_fs}_{attack}_seed{seed}.json"


def _extract(report: Dict[str, Any]) -> Dict[str, Optional[float]]:
    ops = report.get("operational_metrics") or {}
    ttd = ops.get("time_to_detect") or {}
    fa = ops.get("false_alarms") or {}
    return {
        "mean_delay_seconds": ttd.get("mean_delay_seconds"),
        "mean_detection_rate": ttd.get("mean_detection_rate"),
        "aggregate_false_alarm_rate": fa.get("aggregate_false_alarm_rate"),
        "benign_missions_mean_false_alarms_per_hour": fa.get(
            "benign_missions_mean_false_alarms_per_hour"
        ),
        "threshold": report.get("threshold"),
        "aggregate_auroc": report.get("aggregate_auroc"),
    }


def _mean_std(vals: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    clean = [float(v) for v in vals if v is not None]
    if not clean:
        return None, None, 0
    mean = statistics.mean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean, std, len(clean)


def aggregate() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    missing: List[str] = []

    for attack, attack_label in ATTACKS:
        for stack_fs, stack_cb, stack_label in STACKS:
            seed_rows: List[Dict[str, Any]] = []
            buckets: Dict[str, List[Optional[float]]] = {k: [] for k in METRIC_KEYS}
            skipped_benign = attack == "benign" and stack_cb is None
            for seed in SEEDS:
                path = _report_path(stack_fs, stack_cb, attack, seed)
                if path is None:
                    continue
                if not path.is_file():
                    missing.append(str(path))
                    continue
                report = json.loads(path.read_text())
                metrics = _extract(report)
                if (
                    metrics["mean_detection_rate"] is None
                    and metrics["aggregate_false_alarm_rate"] is None
                ):
                    missing.append(str(path) + " (no operational_metrics)")
                    continue
                seed_rows.append({"seed": seed, "report": str(path), **metrics})
                for key in METRIC_KEYS:
                    buckets[key].append(metrics.get(key))
                rows.append(
                    {
                        "attack": attack,
                        "stack": stack_fs,
                        "seed": seed,
                        "report": str(path),
                        **metrics,
                    }
                )

            block: Dict[str, Any] = {
                "attack": attack,
                "attack_label": attack_label,
                "stack": stack_fs,
                "stack_label": stack_label,
                "n_seeds": len(seed_rows),
                "available": not skipped_benign,
                "seeds": seed_rows,
            }
            for key in METRIC_KEYS:
                mean, std, n = _mean_std(buckets[key])
                block[f"mean_{key}"] = mean
                block[f"std_{key}"] = std
                block[f"n_{key}"] = n
            summary[f"{attack}/{stack_fs}"] = block

    shared_latency = json.loads(LATENCY_JSON.read_text()) if LATENCY_JSON.is_file() else {}
    return {
        "threshold_policy": "per-checkpoint F1-tuned (evaluate_layer3_sitl default)",
        "stacks": [s[0] for s in STACKS],
        "stack_labels": {s[0]: s[2] for s in STACKS},
        "attacks": [a for a, _ in ATTACKS],
        "seeds": SEEDS,
        "rows": rows,
        "summary": summary,
        "shared_inference_latency": shared_latency,
        "missing_reports": missing,
        "provenance": {
            "fedavg_full_poisoned": (
                "evaluation_results/full_stack/layer3 "
                "(same checkpoints as tab:full-stack)"
            ),
            "classical_poisoned": (
                "journal_* for Byz./grad./backdoor; threat_* for label flip / "
                "model replacement (3-client, seeds 42--51)"
            ),
            "benign": (
                "evaluation_results/combined_layer3/layer3 "
                "(matched 12-round drone control; FedAvg and ACFL-full only)"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_JSON)
    args = parser.parse_args()
    payload = aggregate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")
    if payload["missing_reports"]:
        print(f"WARNING: {len(payload['missing_reports'])} missing/incomplete reports")
        for path in payload["missing_reports"][:30]:
            print(f"  - {path}")
    for attack, _ in ATTACKS:
        for stack_fs, _, stack_label in STACKS:
            block = payload["summary"][f"{attack}/{stack_fs}"]
            if not block.get("available", True):
                print(f"{attack:18} {stack_label:14} --- (no matched benign)")
                continue
            ttd = block.get("mean_mean_delay_seconds")
            det = block.get("mean_mean_detection_rate")
            fahr = block.get("mean_benign_missions_mean_false_alarms_per_hour")
            if ttd is None or det is None or fahr is None:
                print(f"{attack:18} {stack_label:14} incomplete (n={block['n_seeds']})")
            else:
                print(
                    f"{attack:18} {stack_label:14} "
                    f"TTD={ttd:.2f}s  det={det:.3f}  FA/hr={fahr:.1f}"
                )


if __name__ == "__main__":
    main()
