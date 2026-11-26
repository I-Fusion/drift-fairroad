#!/usr/bin/env python3
"""Paired significance tests on per-seed F1 gaps versus FedAvg from benchmark summaries."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "evaluation_results" / "acfl_tuning"
DEFAULT_OUT = TUNING / "significance_tests.json"
DEFAULT_TEX = ROOT / "docs" / "manuscript" / "significance_stats.tex"

BENCHMARK_SOURCES: Dict[str, Path] = {
    "journal_10seed": TUNING / "drone_journal_benchmark_summary.json",
    "threat_gaps_10seed": TUNING / "drone_threat_gaps_summary.json",
    "ablation_10seed": TUNING / "drone_ablation_seeded_summary.json",
    "client5_byzantine_grad": TUNING / "drone_5client_byzantine_grad_benchmark_summary.json",
    "client5_backdoor": TUNING / "drone_5client_backdoor_benchmark_summary.json",
    "client5_label_flip": TUNING / "drone_5client_label_flip_benchmark_summary.json",
    "client5_model_replacement": TUNING / "drone_5client_model_replacement_benchmark_summary.json",
}

# Keys to include per source (None = all *_minus_fedavg entries).
FOCUS_KEYS: Dict[str, Optional[Tuple[str, ...]]] = {
    "journal_10seed": (
        "byzantine/acfl_full_minus_fedavg",
        "gradient_scaling/acfl_full_minus_fedavg",
        "backdoor/acfl_full_minus_fedavg",
    ),
    "threat_gaps_10seed": (
        "label_flip/acfl_minus_fedavg",
        "model_replacement/acfl_minus_fedavg",
    ),
    "ablation_10seed": (
        "byzantine/acfl_full_minus_fedavg",
        "gradient_scaling/acfl_full_minus_fedavg",
    ),
    "client5_byzantine_grad": (
        "byzantine/acfl_full_minus_fedavg",
        "gradient_scaling/acfl_full_minus_fedavg",
    ),
    "client5_backdoor": ("backdoor/acfl_full_minus_fedavg",),
    "client5_label_flip": ("label_flip/acfl_full_minus_fedavg",),
    "client5_model_replacement": ("model_replacement/acfl_full_minus_fedavg",),
}


def _wilcoxon_two_sided(gaps: List[float]) -> Optional[float]:
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except ImportError:
        return None
    if len(gaps) < 2:
        return None
    if all(abs(g) < 1e-15 for g in gaps):
        return 1.0
    try:
        result = wilcoxon(gaps, alternative="two-sided", zero_method="wilcox")
        return float(result.pvalue)
    except ValueError:
        return None


def _sign_flip_pvalue(gaps: List[float], *, n_perm: int = 50_000, seed: int = 42) -> float:
    if len(gaps) < 2:
        return 1.0
    observed = statistics.mean(gaps)
    if all(abs(g) < 1e-15 for g in gaps):
        return 1.0
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_perm):
        flipped = [g * (1 if rng.random() < 0.5 else -1) for g in gaps]
        if abs(statistics.mean(flipped)) >= abs(observed) - 1e-15:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def _one_sided_improve_p(gaps: List[float], *, n_perm: int = 50_000, seed: int = 42) -> float:
    """H0: mean gap <= 0; H1: mean gap > 0."""
    if len(gaps) < 2:
        return 1.0
    observed = statistics.mean(gaps)
    if observed <= 0:
        return 1.0
    rng = random.Random(seed + 1)
    extreme = 0
    for _ in range(n_perm):
        flipped = [g * (1 if rng.random() < 0.5 else -1) for g in gaps]
        if statistics.mean(flipped) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def _bootstrap_ci(
    gaps: List[float],
    *,
    n_boot: int = 20_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    if len(gaps) < 2:
        m = gaps[0] if gaps else 0.0
        return m, m
    rng = random.Random(seed + 2)
    n = len(gaps)
    means = []
    for _ in range(n_boot):
        sample = [gaps[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return lo, hi


def analyze_gaps(gaps: List[float]) -> Dict[str, Any]:
    gaps = [float(g) for g in gaps]
    n = len(gaps)
    mean_gap = statistics.mean(gaps) if gaps else 0.0
    ci_lo, ci_hi = _bootstrap_ci(gaps)
    wilcoxon_p = _wilcoxon_two_sided(gaps)
    method = "wilcoxon" if wilcoxon_p is not None else "sign_flip"
    p_two_sided = wilcoxon_p if wilcoxon_p is not None else _sign_flip_pvalue(gaps)
    p_improve = _one_sided_improve_p(gaps)
    return {
        "n_seeds": n,
        "gaps": gaps,
        "mean_gap": mean_gap,
        "std_gap": statistics.pstdev(gaps) if n > 1 else 0.0,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "p_two_sided": p_two_sided,
        "p_improve": p_improve,
        "significant_0.05": p_two_sided < 0.05,
        "significant_improve_0.05": p_improve < 0.05 and mean_gap > 0,
        "test_method": method,
    }


def _collect_gap_entries(summary: Dict[str, Any], keys: Optional[Iterable[str]]) -> Dict[str, Any]:
    if keys is not None:
        return {k: summary[k] for k in keys if k in summary and "gaps" in summary[k]}
    return {k: v for k, v in summary.items() if k.endswith("_minus_fedavg") and "gaps" in v}


def run_all() -> Dict[str, Any]:
    report: Dict[str, Any] = {"sources": {}, "meta": {"alpha": 0.05, "ci": "bootstrap_percentile"}}
    for source_name, path in BENCHMARK_SOURCES.items():
        if not path.is_file():
            report["sources"][source_name] = {"error": f"missing {path}"}
            continue
        summary = json.loads(path.read_text())
        entries = _collect_gap_entries(summary, FOCUS_KEYS.get(source_name))
        source_report: Dict[str, Any] = {}
        for key, block in sorted(entries.items()):
            source_report[key] = analyze_gaps(block["gaps"])
        report["sources"][source_name] = source_report
    return report


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "$<0.001$"
    return f"${p:.3f}$"


def _fmt_gap(x: float) -> str:
    return f"{x:+.3f}"


def render_tex(report: Dict[str, Any]) -> str:
    """Build manuscript table from pre-defined rows."""
    rows = [
        # (latex benchmark label, attack, source_key, json_key)
        ("3-client main", "Byzantine", "journal_10seed", "byzantine/acfl_full_minus_fedavg"),
        ("3-client main", "Grad.\\ scaling", "journal_10seed", "gradient_scaling/acfl_full_minus_fedavg"),
        ("3-client main", "Backdoor", "journal_10seed", "backdoor/acfl_full_minus_fedavg"),
        ("Threat gaps", "Label flip", "threat_gaps_10seed", "label_flip/acfl_minus_fedavg"),
        ("Threat gaps", "Model repl.", "threat_gaps_10seed", "model_replacement/acfl_minus_fedavg"),
        ("5-client", "Byzantine", "client5_byzantine_grad", "byzantine/acfl_full_minus_fedavg"),
        ("5-client", "Grad.\\ scaling", "client5_byzantine_grad", "gradient_scaling/acfl_full_minus_fedavg"),
        ("5-client", "Backdoor", "client5_backdoor", "backdoor/acfl_full_minus_fedavg"),
        ("5-client", "Label flip", "client5_label_flip", "label_flip/acfl_full_minus_fedavg"),
        ("5-client", "Model repl.", "client5_model_replacement", "model_replacement/acfl_full_minus_fedavg"),
    ]

    lines = [
        "% Auto-generated by scripts/compute_significance_tests.py",
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Paired significance tests: ACFL full vs.\\ FedAvg per-seed F1 gap "
        "(Wilcoxon signed-rank when \\texttt{scipy} is available, else sign-flip permutation; "
        "95\\% bootstrap CI on mean gap).}",
        "\\label{tab:significance}",
        "\\small",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Benchmark & Attack & $n$ & Mean gap & 95\\% CI & $p$ (2-sided) & Sig.\\\\",
        "\\midrule",
    ]
    for bench, attack, source_key, json_key in rows:
        block = report["sources"].get(source_key, {}).get(json_key)
        if not block:
            continue
        sig = "\\checkmark" if block["significant_0.05"] else "---"
        ci = f"[{_fmt_gap(block['ci95_low'])}, {_fmt_gap(block['ci95_high'])}]"
        lines.append(
            f"{bench} & {attack} & {block['n_seeds']} "
            f"& {_fmt_gap(block['mean_gap'])} & {ci} "
            f"& {_fmt_p(block['p_two_sided'])} & {sig} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--no-tex", action="store_true")
    args = parser.parse_args()

    report = run_all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")

    if not args.no_tex:
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        args.tex.write_text(render_tex(report))
        print(f"Wrote {args.tex}")

    print("\nACFL full vs FedAvg (key rows):")
    for source_name, entries in report["sources"].items():
        if isinstance(entries, dict) and "error" in entries:
            continue
        for key, stats in entries.items():
            if "acfl" not in key:
                continue
            print(
                f"  [{source_name}] {key}: "
                f"gap={stats['mean_gap']:+.3f}, "
                f"CI=[{stats['ci95_low']:+.3f},{stats['ci95_high']:+.3f}], "
                f"p={stats['p_two_sided']:.4f}, "
                f"sig={stats['significant_0.05']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
