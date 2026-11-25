#!/usr/bin/env python3
"""Export journal benchmark results to Markdown and LaTeX tables."""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


TUNING_ROOT = Path("evaluation_results/acfl_tuning")
DEFAULT_SUMMARY = TUNING_ROOT / "drone_journal_benchmark_summary.json"
DEFAULT_ABLATION_SUMMARY = TUNING_ROOT / "drone_ablation_seeded_summary.json"
DEFAULT_CSV = TUNING_ROOT / "drone_journal_benchmark.csv"
OUT_DIR = TUNING_ROOT / "paper_tables"

ATTACK_ORDER = ["byzantine", "gradient_scaling", "backdoor"]
ABLATION_ATTACK_ORDER = [
    "byzantine",
    "gradient_scaling",
    "backdoor",
    "label_flip",
    "model_replacement",
]
ABLATION_ATTACK_LABELS = {
    "byzantine": "Byzantine",
    "gradient_scaling": "Grad. scaling",
    "backdoor": "Backdoor",
    "label_flip": "Label flip",
    "model_replacement": "Model repl.",
}
ABLATION_ATTACK_LABELS_TEX = {
    "byzantine": "Byz.",
    "gradient_scaling": "Grad.",
    "backdoor": "Backdoor",
    "label_flip": "Label flip",
    "model_replacement": "Model repl.",
}
ABLATION_DEFENDER_ORDER = [
    "fedavg",
    "acfl_no_layer1",
    "acfl_no_layer2",
    "acfl_full",
]
ABLATION_DEFENDER_LABELS = {
    "fedavg": "FedAvg",
    "acfl_no_layer1": "ACFL w/o L1",
    "acfl_no_layer2": "ACFL w/o L2",
    "acfl_full": "ACFL (full)",
}
ABLATION_DEFENDER_LABELS_TEX = {
    "fedavg": "FedAvg",
    "acfl_no_layer1": "ACFL w/o L1",
    "acfl_no_layer2": "ACFL w/o L2",
    "acfl_full": "ACFL (full)",
}
ATTACK_LABELS = {
    "byzantine": "Byzantine",
    "gradient_scaling": "Gradient scaling",
    "backdoor": "Backdoor",
}
DEFENDER_ORDER = [
    "fedavg",
    "coordinate_median",
    "trimmed_mean",
    "krum",
    "acfl_full",
]
DEFENDER_LABELS = {
    "fedavg": "FedAvg",
    "coordinate_median": "Coord.\\ median",
    "trimmed_mean": "Trimmed mean",
    "krum": "Krum",
    "acfl_full": "ACFL (full)",
}
DEFENDER_LABELS_MD = {
    "fedavg": "FedAvg",
    "coordinate_median": "Coord. median",
    "trimmed_mean": "Trimmed mean",
    "krum": "Krum",
    "acfl_full": "ACFL (full)",
}


def _fmt_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "---"
    if std is None or std == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def _fmt_mean_std_md(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "—"
    if std is None or std == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def _gap(summary: Dict, attack: str, defender: str) -> Optional[float]:
    key = f"{attack}/{defender}_minus_fedavg"
    stats = summary.get(key)
    if not stats:
        return None
    return stats.get("mean_gap")


def _load_summary(path: Path) -> Dict:
    return json.loads(path.read_text())


def _f1_cell(summary: Dict, attack: str, defender: str, latex: bool) -> str:
    stats = summary.get(f"{attack}/{defender}")
    if not stats:
        return "---" if latex else "—"
    fmt = _fmt_mean_std if latex else _fmt_mean_std_md
    return fmt(stats.get("mean_f1"), stats.get("std_f1"))


def _trigger_cell(summary: Dict, defender: str, latex: bool) -> str:
    stats = summary.get(f"backdoor/{defender}")
    if not stats:
        return "---" if latex else "—"
    prob = stats.get("mean_backdoor_attack_prob")
    if prob is None:
        return "---" if latex else "—"
    return f"{prob:.3f}"


def build_f1_table(summary: Dict, latex: bool) -> str:
    lines: List[str] = []
    if latex:
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Mean test F1 ($\\pm$ std) on SITL drone packet FL "
                     "(3 clients, 12 rounds, 3 seeds).}")
        lines.append("\\label{tab:journal-f1}")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")
        lines.append("Defender & Byzantine & Grad.\\ scaling & Backdoor \\\\")
        lines.append("\\midrule")
        for defender in DEFENDER_ORDER:
            label = DEFENDER_LABELS[defender]
            cells = [_f1_cell(summary, attack, defender, latex=True) for attack in ATTACK_ORDER]
            gap_cells = []
            if defender != "fedavg":
                gaps = [_gap(summary, a, defender) for a in ATTACK_ORDER]
                # gaps not in main table columns - add footnote in md only
            lines.append(f"{label} & {' & '.join(cells)} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
    else:
        lines.append("## Table I — Mean test F1 (± std)")
        lines.append("")
        lines.append("| Defender | Byzantine | Gradient scaling | Backdoor |")
        lines.append("|----------|-----------|------------------|----------|")
        for defender in DEFENDER_ORDER:
            label = DEFENDER_LABELS_MD[defender]
            cells = [_f1_cell(summary, attack, defender, latex=False) for attack in ATTACK_ORDER]
            lines.append(f"| {label} | {' | '.join(cells)} |")
        lines.append("")
        lines.append("**Δ vs FedAvg (mean F1 gap)**")
        lines.append("")
        lines.append("| Defender | Byzantine | Gradient scaling | Backdoor |")
        lines.append("|----------|-----------|------------------|----------|")
        for defender in DEFENDER_ORDER:
            if defender == "fedavg":
                continue
            label = DEFENDER_LABELS_MD[defender]
            gaps = []
            for attack in ATTACK_ORDER:
                g = _gap(summary, attack, defender)
                gaps.append(f"{g:+.3f}" if g is not None else "—")
            lines.append(f"| {label} | {' | '.join(gaps)} |")
    return "\n".join(lines)


def build_backdoor_trigger_table(summary: Dict, latex: bool) -> str:
    lines: List[str] = []
    if latex:
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Backdoor trigger mean $P(\\text{attack})$ on counterfactual probes "
                     "(lower is better).}")
        lines.append("\\label{tab:journal-backdoor-trigger}")
        lines.append("\\begin{tabular}{lc}")
        lines.append("\\toprule")
        lines.append("Defender & Mean $P(\\text{attack})$ \\\\")
        lines.append("\\midrule")
        for defender in DEFENDER_ORDER:
            label = DEFENDER_LABELS[defender]
            cell = _trigger_cell(summary, defender, latex=True)
            lines.append(f"{label} & {cell} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
    else:
        lines.append("## Table II — Backdoor trigger P(attack)")
        lines.append("")
        lines.append("Mean predicted attack probability on stamped trigger probes (lower = less backdoor success).")
        lines.append("")
        lines.append("| Defender | Mean P(attack) | Mean F1 (backdoor run) |")
        lines.append("|----------|----------------|-------------------------|")
        for defender in DEFENDER_ORDER:
            label = DEFENDER_LABELS_MD[defender]
            prob = _trigger_cell(summary, defender, latex=False)
            f1 = _f1_cell(summary, "backdoor", defender, latex=False)
            lines.append(f"| {label} | {prob} | {f1} |")
    return "\n".join(lines)


def build_ablation_table(ablation_summary: Dict, latex: bool) -> str:
    lines: List[str] = []
    if latex:
        ncols = "l" + ("c" * len(ABLATION_ATTACK_ORDER))
        header = " & ".join(
            ABLATION_ATTACK_LABELS_TEX[a] for a in ABLATION_ATTACK_ORDER
        )
        lines.append("\\begin{table*}[t]")
        lines.append("\\centering")
        lines.append(
            "\\caption{Layer ablation: mean test F1 ($\\pm$ std) over 10 seeds "
            "(3 clients).}"
        )
        lines.append("\\label{tab:journal-ablation}")
        lines.append("\\small")
        lines.append(f"\\begin{{tabular}}{{{ncols}}}")
        lines.append("\\toprule")
        lines.append(f"Configuration & {header} \\\\")
        lines.append("\\midrule")
        for defender in ABLATION_DEFENDER_ORDER:
            label = ABLATION_DEFENDER_LABELS_TEX[defender]
            cells = []
            for attack in ABLATION_ATTACK_ORDER:
                stats = ablation_summary.get(f"{attack}/{defender}")
                if not stats:
                    cells.append("---")
                else:
                    cells.append(
                        _fmt_mean_std(stats.get("mean_f1"), stats.get("std_f1"))
                    )
            lines.append(f"{label} & {' & '.join(cells)} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")
    else:
        header = " | ".join(
            ABLATION_ATTACK_LABELS[a] for a in ABLATION_ATTACK_ORDER
        )
        sep = "|".join(["---"] * (1 + len(ABLATION_ATTACK_ORDER)))
        lines.append("## Table III — Layer ablation (mean F1 ± std, 10 seeds)")
        lines.append("")
        lines.append(f"| Configuration | {header} |")
        lines.append(f"|{sep}|")
        for defender in ABLATION_DEFENDER_ORDER:
            label = ABLATION_DEFENDER_LABELS[defender]
            cells = []
            for attack in ABLATION_ATTACK_ORDER:
                stats = ablation_summary.get(f"{attack}/{defender}")
                if not stats:
                    cells.append("—")
                else:
                    cells.append(
                        _fmt_mean_std_md(stats.get("mean_f1"), stats.get("std_f1"))
                    )
            lines.append(f"| {label} | {' | '.join(cells)} |")
    return "\n".join(lines)


def build_threat_gaps_table(threat_summary: Dict, latex: bool) -> str:
    lines: List[str] = []
    attacks = ["label_flip", "model_replacement"]
    labels = {"label_flip": "Label flip", "model_replacement": "Model replacement"}
    defenders = [
        ("fedavg", "FedAvg"),
        ("coordinate_median", "Coord.\\ median" if latex else "Coord. median"),
        ("trimmed_mean", "Trimmed mean"),
        ("krum", "Krum"),
        ("acfl_full", "ACFL (full)"),
    ]
    present = [
        d for d, _ in defenders
        if all(f"{a}/{d}" in threat_summary for a in attacks)
    ]
    if not present:
        present = [d for d in ("fedavg", "acfl_full") if f"label_flip/{d}" in threat_summary]
    if latex:
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(
            "\\caption{Threat-model gap attacks (3 clients, 10 seeds): "
            "mean test F1 ($\\pm$ std).}"
        )
        lines.append("\\label{tab:threat-gaps}")
        lines.append("\\begin{tabular}{lcc}")
        lines.append("\\toprule")
        lines.append("Defender & Label flip & Model replacement \\\\")
        lines.append("\\midrule")
        for defender, dlabel in defenders:
            if defender not in present:
                continue
            cells = []
            for attack in attacks:
                stats = threat_summary.get(f"{attack}/{defender}")
                if not stats:
                    cells.append("---")
                else:
                    cells.append(
                        _fmt_mean_std(stats.get("mean_f1"), stats.get("std_f1"))
                    )
            lines.append(f"{dlabel} & {' & '.join(cells)} \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
    else:
        lines.append("## Table IV — Threat-model gap attacks (3-client, 10 seeds)")
        lines.append("")
        lines.append("| Defender | Label flip | Model replacement |")
        lines.append("|----------|------------|-------------------|")
        for defender, dlabel in defenders:
            if defender not in present:
                continue
            cells = []
            for attack in attacks:
                stats = threat_summary.get(f"{attack}/{defender}")
                if not stats:
                    cells.append("—")
                else:
                    cells.append(
                        _fmt_mean_std_md(stats.get("mean_f1"), stats.get("std_f1"))
                    )
            lines.append(f"| {dlabel} | {' | '.join(cells)} |")
        lines.append("")
        for attack in attacks:
            gap = threat_summary.get(f"{attack}/acfl_minus_fedavg")
            if gap:
                lines.append(
                    f"- **{labels[attack]}:** ACFL vs FedAvg gap "
                    f"{gap['mean_gap']:+.3f}"
                )
            krum_gap = threat_summary.get(f"{attack}/krum_minus_fedavg")
            if krum_gap:
                lines.append(
                    f"- **{labels[attack]}:** Krum vs FedAvg gap "
                    f"{krum_gap['mean_gap']:+.3f}"
                )
    return "\n".join(lines)


def build_results_narrative(summary: Dict) -> str:
    lines = [
        "## Results narrative (draft)",
        "",
    ]
    byz_acfl = summary.get("byzantine/acfl_full", {})
    byz_fed = summary.get("byzantine/fedavg", {})
    grad_acfl = summary.get("gradient_scaling/acfl_full", {})
    grad_fed = summary.get("gradient_scaling/fedavg", {})
    bd_acfl = summary.get("backdoor/acfl_full", {})
    bd_fed = summary.get("backdoor/fedavg", {})
    bd_krum = summary.get("backdoor/krum", {})

    if byz_acfl and byz_fed:
        lines.append(
            f"- **Byzantine:** Only ACFL preserves mission F1 "
            f"({byz_acfl.get('mean_f1', 0):.3f} vs FedAvg {byz_fed.get('mean_f1', 0):.3f}); "
            f"classical robust aggregators collapse to F1 $\\approx$ 0."
        )
    if grad_acfl and grad_fed:
        gap = summary.get("gradient_scaling/acfl_full_minus_fedavg", {}).get("mean_gap")
        gap_s = f"{gap:+.3f}" if gap is not None else "n/a"
        lines.append(
            f"- **Gradient scaling:** ACFL improves mean F1 by {gap_s} over FedAvg "
            f"({grad_acfl.get('mean_f1', 0):.3f} vs {grad_fed.get('mean_f1', 0):.3f}); "
            f"Krum and coordinate methods underperform FedAvg."
        )
    if bd_acfl and bd_fed:
        f1_gap = summary.get("backdoor/acfl_full_minus_fedavg", {}).get("mean_gap")
        f1_gap_s = f"{f1_gap:+.3f}" if f1_gap is not None else "n/a"
        p_acfl = bd_acfl.get("mean_backdoor_attack_prob")
        p_fed = bd_fed.get("mean_backdoor_attack_prob")
        trigger_note = ""
        if p_acfl is not None and p_fed is not None:
            trigger_note = (
                f" Trigger $P(\\text{{attack}})$: ACFL {p_acfl:.3f} vs FedAvg {p_fed:.3f}."
            )
        lines.append(
            f"- **Backdoor:** Clean-task F1 is comparable to FedAvg/Krum "
            f"(ACFL {bd_acfl.get('mean_f1', 0):.3f}, FedAvg {bd_fed.get('mean_f1', 0):.3f}, "
            f"Krum {bd_krum.get('mean_f1', 0):.3f}; $\\Delta$ vs FedAvg {f1_gap_s})."
            f"{trigger_note}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export journal benchmark tables")
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to drone_journal_benchmark_summary.json",
    )
    parser.add_argument(
        "--ablation-summary",
        type=Path,
        default=DEFAULT_ABLATION_SUMMARY,
        help="Path to drone_ablation_seeded_summary.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for paper tables",
    )
    args = parser.parse_args()

    summary = _load_summary(args.summary)
    ablation_summary = {}
    if args.ablation_summary.is_file():
        ablation_summary = _load_summary(args.ablation_summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    md_parts = [
        "# ACFL journal benchmark tables",
        "",
        "_Generated from `drone_journal_benchmark_summary.json`._",
        "",
        build_f1_table(summary, latex=False),
        "",
        build_backdoor_trigger_table(summary, latex=False),
        "",
    ]
    if ablation_summary:
        md_parts.extend([
            build_ablation_table(ablation_summary, latex=False),
            "",
        ])
    threat_path = TUNING_ROOT / "drone_threat_gaps_summary.json"
    if threat_path.is_file():
        md_parts.extend([
            build_threat_gaps_table(_load_summary(threat_path), latex=False),
            "",
        ])
    md_parts.append(build_results_narrative(summary))
    md_path = args.out_dir / "journal_tables.md"
    md_path.write_text("\n".join(md_parts) + "\n")

    tex_parts = [
        "% ACFL journal benchmark tables",
        "% Requires booktabs: \\usepackage{booktabs}",
        "",
        build_f1_table(summary, latex=True),
        "",
        build_backdoor_trigger_table(summary, latex=True),
    ]
    if ablation_summary:
        tex_parts.extend(["", build_ablation_table(ablation_summary, latex=True)])

    threat_summary = {}
    threat_path = TUNING_ROOT / "drone_threat_gaps_summary.json"
    if threat_path.is_file():
        threat_summary = _load_summary(threat_path)
        tex_parts.extend(["", build_threat_gaps_table(threat_summary, latex=True)])
    tex_path = args.out_dir / "journal_tables.tex"
    tex_path.write_text("\n".join(tex_parts) + "\n")

    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
