#!/usr/bin/env python3
"""Diagnose why ACFL trails Krum on 5-client model-replacement runs.

The key question is whether Layer 1 identifies the malicious client but the
aggregation path still assigns enough normalized weight for a 10x
anti-direction update to hurt the global model.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "evaluation_results" / "comprehensive_validation"
DEFAULT_OUT = ROOT / "evaluation_results" / "acfl_tuning"


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    return statistics.mean(nums) if nums else None


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _scenario_dirs(root: Path, pattern: str) -> List[Path]:
    return sorted(p for p in root.glob(pattern) if p.is_dir())


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _client_diag(report: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    agg = report.get("aggregation_diagnostics", {}).get("clients", {})
    diag = agg.get(client_id, {})
    return {
        "aggregation_weight_normalized": diag.get("aggregation_weight_normalized"),
        "aggregation_weight_raw": diag.get("aggregation_weight_raw"),
        "layer1_trust_for_aggregation": diag.get("layer1_trust"),
        "robustness_score": diag.get("robustness_score"),
        "mission_context_score": diag.get("mission_context_score"),
        "combined_trust": diag.get("combined_trust"),
        "powered_trust": diag.get("powered_trust"),
        "min_weight_clipped": diag.get("min_weight_clipped"),
    }


def diagnose_scenario(
    scenario_dir: Path,
    *,
    malicious_client: str,
) -> Dict[str, Any]:
    reports_path = scenario_dir / "acfl_round_reports.json"
    metrics_path = scenario_dir / "metrics.json"
    if not reports_path.is_file():
        raise FileNotFoundError(f"missing {reports_path}")

    reports = _load_json(reports_path)
    metrics = _load_json(metrics_path) if metrics_path.is_file() else {}
    rounds = []
    for report in reports:
        clients = list(report.get("trust_scores", {}).keys())
        raw = report.get("aggregation_guard", {}).get("layer1_raw_trust")
        if raw is None:
            raw = report.get("layer1_raw_trust", {})
        diag = _client_diag(report, malicious_client)
        row = {
            "scenario": scenario_dir.name,
            "round": report.get("round"),
            "num_participants": len(clients),
            "malicious_present": malicious_client in clients,
            "layer1_raw": raw.get(malicious_client),
            **diag,
            "aggregation_skipped": report.get("aggregation_guard", {}).get("skipped", False),
        }
        rounds.append(row)

    malicious_rounds = [r for r in rounds if r["malicious_present"]]
    two_client_malicious = [
        r for r in malicious_rounds if int(r.get("num_participants") or 0) == 2
    ]
    weights = [r.get("aggregation_weight_normalized") for r in malicious_rounds]
    raw_scores = [r.get("layer1_raw") for r in malicious_rounds]
    agg_l1 = [r.get("layer1_trust_for_aggregation") for r in malicious_rounds]

    fl_summary = metrics.get("fl_eval", {}).get("summary", {})
    final_f1s = [
        entry.get("final_f1_score")
        for entry in fl_summary.values()
        if entry.get("final_f1_score") is not None
    ]

    return {
        "scenario": scenario_dir.name,
        "final_mean_f1": _mean(final_f1s),
        "trust_auc": metrics.get("acfl_eval", {}).get("trust_score_auc"),
        "rounds_total": len(rounds),
        "rounds_with_malicious_client": len(malicious_rounds),
        "two_client_malicious_rounds": len(two_client_malicious),
        "malicious_mean_layer1_raw": _mean(raw_scores),
        "malicious_mean_layer1_for_aggregation": _mean(agg_l1),
        "malicious_mean_normalized_weight": _mean(weights),
        "malicious_max_normalized_weight": (
            max(v for v in weights if v is not None)
            if any(v is not None for v in weights)
            else None
        ),
        "malicious_min_normalized_weight": (
            min(v for v in weights if v is not None)
            if any(v is not None for v in weights)
            else None
        ),
        "two_client_malicious_mean_weight": _mean(
            r.get("aggregation_weight_normalized") for r in two_client_malicious
        ),
        "rounds": rounds,
    }


def _flatten_rounds(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        rows.extend(summary.get("rounds", []))
    return rows


def write_outputs(summaries: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "model_replacement_acfl_diagnostics.json"
    rounds_path = out_dir / "model_replacement_acfl_round_diagnostics.csv"

    slim = [{k: v for k, v in s.items() if k != "rounds"} for s in summaries]
    summary_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")

    rows = _flatten_rounds(summaries)
    if rows:
        fieldnames = list(rows[0].keys())
        with rounds_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {summary_path}")
    if rows:
        print(f"Wrote {rounds_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--pattern",
        default="5client_model_replacement_seed*_acfl*",
        help="Scenario directory glob under --root",
    )
    parser.add_argument("--malicious-client", default="client_2")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    scenarios = _scenario_dirs(args.root, args.pattern)
    if not scenarios:
        raise SystemExit(f"No scenarios matched {args.root / args.pattern}")

    summaries = [
        diagnose_scenario(path, malicious_client=args.malicious_client)
        for path in scenarios
    ]
    write_outputs(summaries, args.out_dir)

    print("\nModel-replacement ACFL diagnostics:")
    for summary in summaries:
        print(
            f"  {summary['scenario']}: "
            f"F1={_fmt(summary.get('final_mean_f1'))} "
            f"trust_auc={summary.get('trust_auc')} "
            f"mal_weight_mean={_fmt(summary.get('malicious_mean_normalized_weight'))} "
            f"mal_l1_raw_mean={_fmt(summary.get('malicious_mean_layer1_raw'))} "
            f"mal_l1_agg_mean={_fmt(summary.get('malicious_mean_layer1_for_aggregation'))} "
            f"two_client_mal_rounds={summary['two_client_malicious_rounds']}"
        )


if __name__ == "__main__":
    main()
