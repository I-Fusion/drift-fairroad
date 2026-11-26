#!/usr/bin/env python3
"""Stratify existing Layer 3 reports by network attack and learning-plane poison.

Reads the per-seed reports already produced for the full-stack and combined
benchmarks. Does not re-score checkpoints. Writes:

  evaluation_results/full_stack/layer3_stratified_summary.json
  docs/manuscript/layer3_stratified_table.tex

The rank table reports waypoint and GNSS AUROC and F1. The operational table
reports time-to-detect on those same attacks and false alarms per hour on
benign (no-network-attack) missions. CPU runtime is the shared TinyPacketCNN
forward pass in ``full_stack_latency.json``; it does not vary by checkpoint.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
FS_LAYER3 = ROOT / "evaluation_results" / "full_stack" / "layer3"
CB_LAYER3 = ROOT / "evaluation_results" / "combined_layer3" / "layer3"
OUT_JSON = ROOT / "evaluation_results" / "full_stack" / "layer3_stratified_summary.json"
OUT_TEX = ROOT / "docs" / "manuscript" / "layer3_stratified_table.tex"
LATENCY_JSON = ROOT / "evaluation_results" / "full_stack" / "full_stack_latency.json"

# (filename stem, benign combined stem or None, latex label, include in table)
STACKS = [
    ("fedavg", "fedavg", "FedAvg", True),
    ("l1_only", None, "Layer~1 only", True),
    ("l2_only", None, "Layer~2 only", True),
    ("coordinate_median", None, "Coord.\\ median", True),
    ("trimmed_mean", None, "Trimmed mean", True),
    ("krum", None, "Krum", True),
    ("full", "acfl_full", "ACFL (full)", True),
]
ATTACKS = [
    ("benign", "Benign"),
    ("byzantine", "Byzantine"),
    ("gradient_scaling", "Grad.\\ scaling"),
    ("backdoor", "Backdoor"),
    ("label_flip", "Label flip"),
    ("model_replacement", "Model repl."),
]
NETWORKS = ("waypoint", "gps")
SEEDS = list(range(42, 52))
RANK_KEYS = ("auroc", "f1")
OPS_KEYS = ("mean_delay_seconds", "mean_detection_rate")


def _report_path(stack: str, benign_stem: Optional[str], attack: str, seed: int) -> Optional[Path]:
    if attack == "benign":
        if benign_stem is None:
            return None
        return CB_LAYER3 / f"benign_{benign_stem}_seed{seed}.json"
    return FS_LAYER3 / f"{stack}_{attack}_seed{seed}.json"


def _mean_std(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None, 0
    mean = statistics.mean(clean)
    std = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean, std, len(clean)


def _cell(report: Dict[str, Any], network: str) -> Dict[str, Any]:
    rank = (report.get("by_network_attack") or {}).get(network) or {}
    ops_map = report.get("operational_metrics_by_network_attack") or {}
    recorded = network in ops_map
    ops = ops_map.get(network) or {}
    ttd = ops.get("time_to_detect") or {}
    return {
        "auroc": rank.get("auroc"),
        "f1": rank.get("f1"),
        "mean_delay_seconds": ttd.get("mean_delay_seconds") if recorded else None,
        "mean_detection_rate": ttd.get("mean_detection_rate") if recorded else None,
        "ops_recorded": recorded,
    }


def _pooled(report: Dict[str, Any]) -> Dict[str, Any]:
    ops = report.get("operational_metrics")
    recorded = isinstance(ops, dict) and bool(ops)
    fa = (ops or {}).get("false_alarms") or {}
    return {
        "benign_missions_mean_false_alarms_per_hour": (
            fa.get("benign_missions_mean_false_alarms_per_hour") if recorded else None
        ),
        "ops_recorded": recorded,
    }


def aggregate() -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    missing: List[str] = []
    for attack, _label in ATTACKS:
        for stack, benign_stem, stack_label, in_table in STACKS:
            block: Dict[str, Any] = {
                "attack": attack,
                "stack": stack,
                "stack_label": stack_label,
                "in_table": in_table,
                "networks": {},
                "pooled": {},
            }
            n_reports = 0
            buckets = {
                net: {k: [] for k in RANK_KEYS + OPS_KEYS}
                for net in NETWORKS
            }
            recorded = {net: 0 for net in NETWORKS}
            fa_values: List[Optional[float]] = []
            fa_recorded = 0
            for seed in SEEDS:
                path = _report_path(stack, benign_stem, attack, seed)
                if path is None:
                    continue
                if not path.is_file():
                    missing.append(str(path))
                    continue
                report = json.loads(path.read_text())
                n_reports += 1
                pooled = _pooled(report)
                if pooled["ops_recorded"]:
                    fa_recorded += 1
                    fa_values.append(pooled["benign_missions_mean_false_alarms_per_hour"])
                for net in NETWORKS:
                    metrics = _cell(report, net)
                    if metrics["ops_recorded"]:
                        recorded[net] += 1
                    for key in RANK_KEYS + OPS_KEYS:
                        buckets[net][key].append(metrics[key])
            block["n_reports"] = n_reports
            for net in NETWORKS:
                net_block: Dict[str, Any] = {"n_ops_recorded": recorded[net]}
                for key, values in buckets[net].items():
                    mean, std, n = _mean_std(values)
                    net_block[f"mean_{key}"] = mean
                    net_block[f"std_{key}"] = std
                    net_block[f"n_{key}"] = n
                block["networks"][net] = net_block
            fa_mean, fa_std, fa_n = _mean_std(fa_values)
            block["pooled"] = {
                "mean_benign_missions_mean_false_alarms_per_hour": fa_mean,
                "std_benign_missions_mean_false_alarms_per_hour": fa_std,
                "n_benign_missions_mean_false_alarms_per_hour": fa_n,
                "n_ops_recorded": fa_recorded,
            }
            summary[f"{attack}/{stack}"] = block
    latency = json.loads(LATENCY_JSON.read_text()) if LATENCY_JSON.is_file() else {}
    return {
        "seeds": SEEDS,
        "missing": missing,
        "latency": latency,
        "summary": summary,
    }


def _fmt(mean: Optional[float], std: Optional[float], digits: int = 3) -> str:
    if mean is None:
        return "---"
    if std is None or std == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _fmt_complete(block: Dict[str, Any], key: str, digits: int = 3) -> str:
    """Show a mean only when every seed recorded a finite value."""
    if block.get(f"n_{key}") != len(SEEDS):
        return "---"
    return _fmt(block.get(f"mean_{key}"), block.get(f"std_{key}"), digits)


def _fmt_delay(block: Dict[str, Any]) -> str:
    """Time-to-detect. A null delay means no onset was caught, not a missing log."""
    if block.get("n_ops_recorded") != len(SEEDS):
        return "---"
    n = block.get("n_mean_delay_seconds") or 0
    if n == 0:
        return "---"
    text = _fmt(block.get("mean_mean_delay_seconds"), block.get("std_mean_delay_seconds"), 2)
    if n < len(SEEDS):
        text += f"\\,{{\\scriptsize({n})}}"
    return text


def _best_keys(
    summary: Dict[str, Any],
    attack: str,
    network: Optional[str],
    key: str,
    *,
    higher: bool,
) -> set:
    """Best stack on a metric. ``key`` is the unsuffixed name stored as mean_/n_."""
    best = None
    winners = set()
    for stack, _benign, _label, in_table in STACKS:
        if not in_table:
            continue
        block = summary.get(f"{attack}/{stack}")
        if not block or block["n_reports"] == 0:
            continue
        src = block["networks"][network] if network else block["pooled"]
        if src.get(f"n_{key}") != len(SEEDS):
            continue
        value = src.get(f"mean_{key}")
        if value is None:
            continue
        if best is None or (value > best + 1e-12 if higher else value < best - 1e-12):
            best = value
            winners = {stack}
        elif abs(value - best) <= 1e-12:
            winners.add(stack)
    return winners


def _nrows(summary: Dict[str, Any], attack: str) -> int:
    n = 0
    for stack, _b, _l, in_table in STACKS:
        if not in_table:
            continue
        block = summary.get(f"{attack}/{stack}")
        if block and block["n_reports"]:
            n += 1
    return max(n, 1)


def _latency_sentence(latency: Dict[str, Any]) -> str:
    mean = (latency.get("mean_ms_per_window") or {}).get("mean")
    p95 = (latency.get("p95_ms_per_window") or {}).get("mean")
    if mean is None or p95 is None:
        return "CPU runtime was not recorded."
    repeats = latency.get("repeats", 10)
    windows = latency.get("windows_per_repeat")
    window_bit = f", {windows} windows each" if windows else ""
    return (
        "CPU runtime is the shared TinyPacketCNN forward pass "
        f"({repeats} sequential repeats{window_bit}): "
        f"mean {mean:.2f}\\,ms per window, p95 {p95:.2f}\\,ms. "
        "It does not depend on the aggregation checkpoint."
    )


def _attack_rows(summary: Dict[str, Any]):
    first = True
    for attack, attack_label in ATTACKS:
        rows = []
        for stack, _benign, stack_label, in_table in STACKS:
            if not in_table:
                continue
            block = summary.get(f"{attack}/{stack}")
            if not block or block["n_reports"] == 0:
                continue
            rows.append((stack, stack_label, block))
        if not rows:
            continue
        yield first, attack, attack_label, rows
        first = False


def render_rank(summary: Dict[str, Any]) -> List[str]:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Layer~3 rank metrics on held-out SITL missions, split by operational "
        "network attack (waypoint injection vs.\\ GNSS spoofing) and by the learning-plane "
        "poison that produced the checkpoint. Means $\\pm$ std over seeds 42--51. "
        "AUROC and F1 use $P(\\text{attack})$ at the per-checkpoint F1-tuned threshold. "
        "Bold marks the highest mean within an attack.}",
        "\\label{tab:layer3-stratified}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Learning-plane poison & Checkpoint & Waypoint AUROC & GNSS AUROC "
        "& Waypoint F1 & GNSS F1 \\\\",
        "\\midrule",
    ]
    for first, attack, attack_label, rows in _attack_rows(summary):
        if not first:
            lines.append("\\midrule")
        for i, (stack, stack_label, block) in enumerate(rows):
            cells = []
            for net, field in (
                ("waypoint", "mean_auroc"),
                ("gps", "mean_auroc"),
                ("waypoint", "mean_f1"),
                ("gps", "mean_f1"),
            ):
                src = block["networks"][net]
                metric = "auroc" if field.endswith("auroc") else "f1"
                text = _fmt(src.get(field), src.get(field.replace("mean_", "std_")))
                if stack in _best_keys(summary, attack, net, metric, higher=True):
                    text = f"\\textbf{{{text}}}"
                cells.append(text)
            attack_cell = (
                f"\\multirow{{{_nrows(summary, attack)}}}{{*}}{{{attack_label}}}"
                if i == 0 else ""
            )
            lines.append(
                f"{attack_cell} & {stack_label} & " + " & ".join(cells) + " \\\\"
            )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table*}",
        "",
    ])
    return lines


def render_ops(summary: Dict[str, Any], latency: Dict[str, Any]) -> List[str]:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Layer~3 operational metrics for the same checkpoints. "
        "Time-to-detect is the mean delay from labeled onset on that network attack; "
        "a parenthetical count is the number of seeds that caught at least one onset, "
        "and the cell is blank unless all ten reports stored the operational block. "
        "FA/hr is false alarms per operating hour on benign (no-network-attack) missions, "
        "blank unless all ten seeds recorded it. Bold marks the lowest mean FA/hr "
        "within an attack. "
        + _latency_sentence(latency) + "}",
        "\\label{tab:layer3-stratified-ops}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Learning-plane poison & Checkpoint & Waypoint TTD (s) & GNSS TTD (s) & FA/hr \\\\",
        "\\midrule",
    ]
    for first, attack, attack_label, rows in _attack_rows(summary):
        if not first:
            lines.append("\\midrule")
        for i, (stack, stack_label, block) in enumerate(rows):
            wpt = _fmt_delay(block["networks"]["waypoint"])
            gps = _fmt_delay(block["networks"]["gps"])
            fa = _fmt_complete(
                block["pooled"],
                "benign_missions_mean_false_alarms_per_hour",
                digits=1,
            )
            if stack in _best_keys(
                summary, attack, None,
                "benign_missions_mean_false_alarms_per_hour",
                higher=False,
            ):
                fa = f"\\textbf{{{fa}}}"
            attack_cell = (
                f"\\multirow{{{_nrows(summary, attack)}}}{{*}}{{{attack_label}}}"
                if i == 0 else ""
            )
            lines.append(f"{attack_cell} & {stack_label} & {wpt} & {gps} & {fa} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table*}",
        "",
    ])
    return lines


def render_tex(payload: Dict[str, Any]) -> str:
    lines = render_rank(payload["summary"]) + render_ops(payload["summary"], payload.get("latency") or {})
    return "\n".join(lines)


def main() -> None:
    payload = aggregate()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    OUT_TEX.write_text(render_tex(payload))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_TEX}")
    if payload["missing"]:
        print(f"Missing reports: {len(payload['missing'])}")
    lat = payload.get("latency") or {}
    mean = (lat.get("mean_ms_per_window") or {}).get("mean")
    p95 = (lat.get("p95_ms_per_window") or {}).get("mean")
    print(f"runtime mean_ms={mean} p95_ms={p95}")
    for attack, _label in ATTACKS:
        for stack, _b, _label2, in_table in STACKS:
            if not in_table:
                continue
            block = payload["summary"].get(f"{attack}/{stack}")
            if not block or not block["n_reports"]:
                continue
            wpt = block["networks"]["waypoint"]
            gps = block["networks"]["gps"]
            fa = block["pooled"]["mean_benign_missions_mean_false_alarms_per_hour"]
            print(
                f"{attack:20} {stack:18} "
                f"F1 {wpt['mean_f1']} / {gps['mean_f1']}  "
                f"TTD {wpt['mean_mean_delay_seconds']} / {gps['mean_mean_delay_seconds']} "
                f"(n {wpt['n_mean_delay_seconds']}/{gps['n_mean_delay_seconds']})  "
                f"FA/hr {fa}"
            )


if __name__ == "__main__":
    main()
