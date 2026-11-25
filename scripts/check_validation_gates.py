#!/usr/bin/env python3
"""Regression gates for scripts/run_project_validation.sh.

Fails with exit code 1 if smoke benchmark summaries fall below minimum
thresholds. Only checks artifacts that exist (skipped if not run yet).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _check(name: str, value, minimum: float, failures: List[str]) -> None:
    if value is None:
        failures.append(f"{name}: missing metric")
        return
    if float(value) < minimum:
        failures.append(f"{name}: {float(value):.4f} < {minimum:.4f}")


def _summary_get(summary: dict, key: str, field: str):
    entry = summary.get(key) or {}
    return entry.get(field)


def main() -> int:
    failures: List[str] = []
    warnings: List[str] = []

    # --- Full stack (Byzantine, full config) ---
    fs_path = ROOT / "evaluation_results/full_stack/full_stack_benchmark_summary.json"
    fs = _load_json(fs_path)
    if fs:
        _check(
            "full_stack/byzantine/full layer3_auroc",
            _summary_get(fs, "byzantine/full", "mean_layer3_auroc"),
            0.85,
            failures,
        )
        _check(
            "full_stack/byzantine/full layer2_f1",
            _summary_get(fs, "byzantine/full", "mean_layer2_f1"),
            0.65,
            failures,
        )
        _check(
            "full_stack/byzantine/full layer1_trust_auc",
            _summary_get(fs, "byzantine/full", "mean_layer1_trust_auc"),
            0.95,
            failures,
        )
        fed_l3 = _summary_get(fs, "byzantine/fedavg", "mean_layer3_auroc")
        full_l3 = _summary_get(fs, "byzantine/full", "mean_layer3_auroc")
        if fed_l3 is not None and full_l3 is not None and full_l3 <= fed_l3:
            failures.append(
                f"full_stack/byzantine: full layer3_auroc ({full_l3:.4f}) "
                f"should exceed fedavg ({fed_l3:.4f})"
            )
    else:
        warnings.append(f"skip full_stack gates (no {fs_path})")

    # --- Combined Layer 3 (Byzantine, ACFL full) ---
    cl3_path = ROOT / "evaluation_results/combined_layer3/combined_layer3_benchmark_summary.json"
    cl3 = _load_json(cl3_path)
    if cl3:
        _check(
            "combined_layer3/byzantine/acfl_full layer3_auroc",
            _summary_get(cl3, "byzantine/acfl_full", "mean_layer3_auroc"),
            0.85,
            failures,
        )
    else:
        warnings.append(f"skip combined_layer3 gates (no {cl3_path})")

    # --- Standalone Layer 3 eval (optional) ---
    l3_path = ROOT / "evaluation_results/comprehensive_validation/layer3_eval.json"
    l3 = _load_json(l3_path)
    if l3:
        _check("layer3_eval aggregate_auroc", l3.get("aggregate_auroc"), 0.90, failures)
    else:
        warnings.append(f"skip layer3_eval gate (no {l3_path})")

    for w in warnings:
        print(f"WARN: {w}")
    for f in failures:
        print(f"FAIL: {f}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} regression gate(s) failed.", file=sys.stderr)
        return 1

    print("All applicable regression gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
