#!/usr/bin/env python3
"""Run Layer 3 eval and export operational metrics (TTD, false alarms/hour, latency)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]

def _repo_roots() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    if (root / "fl-time-series").is_dir():
        return root, root
    nested = root / "drift-fairroad"
    if nested.is_dir():
        return root, nested
    return root, root

ACFL_ROOT, DRIFT_ROOT = _repo_roots()
ROOT = ACFL_ROOT
DEFAULT_DATA = DRIFT_ROOT / "data/sitl_dataset/fl/packets"
DEFAULT_OUT = ROOT / "evaluation_results/comprehensive_validation/layer3_operational_metrics.json"
DEFAULT_TEX = ROOT / "docs/manuscript/layer3_operational_stats.tex"
DEFAULT_TABLES_TEX = ROOT / "docs/manuscript/tables.tex"
DEFAULT_LAYER3_EVAL = ROOT / "scripts" / "evaluate_layer3_sitl.py"


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _extract_operational_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    ops = report.get("operational_metrics", {})
    ttd = ops.get("time_to_detect", {})
    fa = ops.get("false_alarms", {})
    latency = report.get("inference_latency", {})
    return {
        "checkpoint": report.get("checkpoint"),
        "threshold": report.get("threshold"),
        "aggregate_auroc": report.get("aggregate_auroc"),
        "aggregate_f1": report.get("aggregate_f1"),
        "time_to_detect": ttd,
        "false_alarms": fa,
        "inference_latency": latency,
        "by_network_attack": report.get("operational_metrics_by_network_attack", {}),
        "per_client": [
            {
                "file": entry.get("file"),
                "network_attack": entry.get("network_attack"),
                "attack_windows": entry.get("attack_windows"),
                "operational_metrics": entry.get("operational_metrics"),
                "inference_latency": entry.get("inference_latency"),
            }
            for entry in report.get("per_client", [])
        ],
    }


def render_latex(summary: Dict[str, Any]) -> str:
    ttd = summary.get("time_to_detect", {})
    fa = summary.get("false_alarms", {})
    latency = summary.get("inference_latency", {})
    lines = [
        (
            "Mean time-to-detect on attacked held-out missions is "
            f"{_fmt(ttd.get('mean_delay_seconds'))}\\,s "
            f"({_fmt(ttd.get('mean_delay_windows'), 1)} windows; "
            f"detection rate {_fmt(ttd.get('mean_detection_rate'))}). "
            f"Aggregate false-alarm rate on benign windows is "
            f"{_fmt(fa.get('aggregate_false_alarm_rate'), 3)} "
            f"({fa.get('total_false_positives', 0)} FP / "
            f"{fa.get('total_benign_windows', 0)} benign windows). "
        ),
    ]
    benign_fa = fa.get("benign_missions_mean_false_alarms_per_hour")
    if benign_fa is not None:
        lines[-1] += (
            f"Benign missions (no network attack) average "
            f"{_fmt(benign_fa, 1)} false alarms/hour."
        )
    else:
        lines[-1] += "Benign-mission false alarms/hour: N/A."
    if latency:
        lines.append(
            "Per-window inference latency (CPU, first agent): "
            f"{_fmt(latency.get('mean_ms_per_window'), 2)}\\,ms mean, "
            f"{_fmt(latency.get('p95_ms_per_window'), 2)}\\,ms p95."
        )
    return "\n".join(lines).rstrip() + "\n"


def render_latex_table(summary: Dict[str, Any]) -> str:
    ttd = summary.get("time_to_detect", {})
    fa = summary.get("false_alarms", {})
    latency = summary.get("inference_latency", {})
    by_attack = summary.get("by_network_attack", {})
    rows = [
        (
            f"Mean time-to-detect & {_fmt(ttd.get('mean_delay_seconds'))}\\,s "
            f"({_fmt(ttd.get('mean_delay_windows'), 1)} windows) & "
            f"detection rate {_fmt(ttd.get('mean_detection_rate'))} \\\\"
        ),
        (
            f"Aggregate false-alarm rate & {_fmt(fa.get('aggregate_false_alarm_rate'), 3)} & "
            f"{fa.get('total_false_positives', 0)} FP / "
            f"{fa.get('total_benign_windows', 0)} benign windows \\\\"
        ),
        (
            f"Benign missions false alarms/hour & "
            f"{_fmt(fa.get('benign_missions_mean_false_alarms_per_hour'), 1)} & "
            f"median {_fmt(fa.get('benign_missions_median_false_alarms_per_hour'), 1)} \\\\"
        ),
    ]
    if latency:
        rows.append(
            f"Inference latency (CPU) & "
            f"{_fmt(latency.get('mean_ms_per_window'), 2)}\\,ms mean & "
            f"p95 {_fmt(latency.get('p95_ms_per_window'), 2)}\\,ms \\\\"
        )
    for attack, block in sorted(by_attack.items()):
        attack_ttd = block.get("time_to_detect", {})
        attack_fa = block.get("false_alarms", {})
        rows.append(
            f"{attack} missions & "
            f"TTD {_fmt(attack_ttd.get('mean_delay_seconds'))}\\,s & "
            f"FA rate {_fmt(attack_fa.get('aggregate_false_alarm_rate'), 3)}"
        )
    if not rows:
        return ""
    body = []
    for row in rows:
        body.append(f"{row} \\\\")
    return "\n".join(body)


def render_latex_table_block(summary: Dict[str, Any]) -> str:
    rows = render_latex_table(summary)
    return (
        "% BEGIN layer3-operational-table\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Layer~3 operational metrics on held-out SITL missions "
        "(time-to-detect, false alarms, inference latency). Regenerate with "
        "\\texttt{scripts/compute\\_layer3\\_operational\\_metrics.py}.}\n"
        "\\label{tab:layer3-operational}\n"
        "\\small\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Metric & Value & Notes \\\\\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
        "% END layer3-operational-table"
    )


def patch_tables_tex(tables_path: Path, table_block: str) -> None:
    """Replace the generated layer3-operational table block in tables.tex."""
    text = tables_path.read_text(encoding="utf-8")
    begin = "% BEGIN layer3-operational-table"
    end = "% END layer3-operational-table"
    if begin not in text or end not in text:
        raise ValueError(f"Missing layer3-operational table markers in {tables_path}")
    prefix, rest = text.split(begin, 1)
    _, suffix = rest.split(end, 1)
    patched = f"{prefix}{table_block}{suffix}"
    tables_path.write_text(patched, encoding="utf-8")


def run_layer3_eval(
    *,
    checkpoint: str,
    data_dir: Path,
    output_json: Path,
    benchmark_latency: bool,
    max_false_alarms_per_hour: Optional[float],
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(DEFAULT_LAYER3_EVAL),
        "--data-dir",
        str(data_dir),
        "--checkpoint",
        checkpoint,
        "--output",
        str(output_json),
    ]
    if benchmark_latency:
        cmd.append("--benchmark-latency")
    if max_false_alarms_per_hour is not None:
        cmd.extend(
            ["--max-false-alarms-per-hour", str(max_false_alarms_per_hour)]
        )
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    with output_json.open() as handle:
        report = json.load(handle)
    report["checkpoint"] = checkpoint
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Trained global model checkpoint (.pt) for Layer 3 eval",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--layer3-report",
        type=Path,
        default=None,
        help="Reuse an existing evaluate_layer3_sitl JSON instead of re-running",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tex-out", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--tables-tex", type=Path, default=DEFAULT_TABLES_TEX)
    parser.add_argument(
        "--max-false-alarms-per-hour",
        type=float,
        default=None,
        help="Optional mission-continuity limit",
    )
    parser.add_argument(
        "--benchmark-latency",
        action="store_true",
        help="Measure per-window inference latency during eval",
    )
    args = parser.parse_args()

    layer3_json = args.output.with_name("layer3_eval_with_ops.json")
    if args.layer3_report and args.layer3_report.is_file():
        with args.layer3_report.open() as handle:
            report = json.load(handle)
        report.setdefault("checkpoint", args.checkpoint)
    else:
        report = run_layer3_eval(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            output_json=layer3_json,
            benchmark_latency=args.benchmark_latency,
            max_false_alarms_per_hour=args.max_false_alarms_per_hour,
        )

    summary = _extract_operational_summary(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(summary, handle, indent=2)

    tex = render_latex(summary)
    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    args.tex_out.write_text(tex, encoding="utf-8")
    patch_tables_tex(args.tables_tex, render_latex_table_block(summary))

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.tex_out}")
    print(f"Patched {args.tables_tex}")


if __name__ == "__main__":
    main()
