#!/usr/bin/env python3
"""Regenerate benchmark tables and narrative numbers for the ACFL manuscript."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "evaluation_results" / "acfl_tuning"
MANUSCRIPT = ROOT / "docs" / "manuscript"

SOURCES = {
    "journal": TUNING / "drone_journal_benchmark_summary.json",
    "ablation": TUNING / "drone_ablation_seeded_summary.json",
    "threat": TUNING / "drone_threat_gaps_summary.json",
    "probe_cap": TUNING / "drone_backdoor_probe_cap_summary.json",
    "probe_cap_rows": TUNING / "drone_backdoor_probe_cap.json",
    "scrub_3c": TUNING / "drone_backdoor_scrub_ablation_summary.json",
    "scrub_5c": TUNING / "drone_backdoor_scrub_ablation_5c_summary.json",
    "c5_byz_grad": TUNING / "drone_5client_byzantine_grad_benchmark_summary.json",
    "c5_backdoor": TUNING / "drone_5client_backdoor_benchmark_summary.json",
    "c5_label_flip": TUNING / "drone_5client_label_flip_benchmark_summary.json",
    "c5_model_replacement": TUNING / "drone_5client_model_replacement_benchmark_summary.json",
    "combined_layer3": ROOT / "evaluation_results/combined_layer3/combined_layer3_benchmark_summary.json",
    "full_stack": ROOT / "evaluation_results/full_stack/full_stack_benchmark_summary.json",
    "full_stack_ops": ROOT / "evaluation_results/full_stack/full_stack_operational_metrics.json",
    "aggregation_overhead": ROOT / "evaluation_results/full_stack/aggregation_overhead.json",
    "probe_config": TUNING / "drone_probe_config_ablation_summary.json",
    "attack_intensity": TUNING / "drone_attack_intensity_sweep_summary.json",
    "l1_component": TUNING / "drone_layer1_component_ablation_summary.json",
    "clean_l1l2": TUNING / "drone_clean_l1l2_ablation_summary.json",
    "layer3_ops": ROOT / "evaluation_results/comprehensive_validation/layer3_operational_metrics.json",
}

DEFENDERS = [
    "fedavg",
    "coordinate_median",
    "trimmed_mean",
    "krum",
    "acfl_full",
]
DEFENDER_TEX = {
    "fedavg": "FedAvg",
    "coordinate_median": "Coord.\\ median",
    "trimmed_mean": "Trimmed mean",
    "krum": "Krum",
    "acfl_full": "ACFL (full)",
}
ATTACKS_MAIN = ["byzantine", "gradient_scaling", "backdoor"]
ATTACKS_ABLATION = [
    "byzantine",
    "gradient_scaling",
    "backdoor",
    "label_flip",
    "model_replacement",
]
ABLATION_ATTACK_TEX = {
    "byzantine": "Byz.",
    "gradient_scaling": "Grad.",
    "backdoor": "Backdoor",
    "label_flip": "Label flip",
    "model_replacement": "Model repl.",
}
ABLATION_DEFENDERS = ["fedavg", "acfl_no_layer1", "acfl_no_layer2", "acfl_full"]
ABLATION_TEX = {
    "fedavg": "FedAvg",
    "acfl_no_layer1": "ACFL w/o L1",
    "acfl_no_layer2": "ACFL w/o L2",
    "acfl_full": "ACFL (full)",
}
L1_COMPONENT_VARIANTS = [
    ("l1_full", "All three"),
    ("l1_no_subspace", "w/o subspace"),
    ("l1_no_temporal", "w/o temporal"),
    ("l1_no_cf", "w/o CF"),
]
CLEAN_L1L2_VARIANTS = [
    ("acfl_l1_only_clean", "L1-only clean"),
    ("acfl_l2_only_clean", "L2-only clean"),
    ("acfl_l1l2_clean", "L1+L2 clean"),
    ("acfl_l1l2_ungated", "L1+L2 ungated"),
    ("acfl_full", "ACFL (full)"),
]
STACK_KEYS = [
    ("fedavg", "FedAvg (no L1/L2)"),
    ("l1_only", "Layer~1 only"),
    ("l2_only", "Layer~2 only"),
    ("full", "L1+L2+L3 (full)"),
]
COMBINED_ATTACKS = [
    ("benign", "Benign (no FL poison)"),
    ("gradient_scaling", "Gradient scaling"),
    ("byzantine", "Byzantine"),
    ("backdoor", "Backdoor"),
    ("label_flip", "Label flip"),
    ("model_replacement", "Model repl."),
]


def _load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _ms(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "---"
    if std is None or std == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def _ms_digits(
    mean: Optional[float],
    std: Optional[float],
    *,
    digits: int = 2,
) -> str:
    if mean is None:
        return "---"
    if std is None or abs(std) < 10 ** (-(digits + 1)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _gap(summary: Dict[str, Any], key: str) -> Optional[float]:
    block = summary.get(key)
    if not block:
        return None
    return block.get("mean_gap")


def _cell(summary: Dict[str, Any], attack: str, defender: str) -> str:
    stats = summary.get(f"{attack}/{defender}")
    if not stats:
        return "---"
    return _ms(stats.get("mean_f1"), stats.get("std_f1"))


def _trigger(summary: Dict[str, Any], defender: str) -> str:
    stats = summary.get(f"backdoor/{defender}")
    if not stats:
        return "---"
    prob = stats.get("mean_backdoor_attack_prob")
    return f"{prob:.3f}" if prob is not None else "---"


def _fmt_gap(g: Optional[float]) -> str:
    if g is None:
        return "---"
    return f"${g:+.3f}$"


def _pct_reduction(base: float, val: float) -> float:
    if base <= 0:
        return 0.0
    return (base - val) / base * 100.0


def build_tables_data(data: Dict[str, Any]) -> str:
    journal = data["journal"]
    ablation = data["ablation"]
    threat = data["threat"]
    scrub3 = data["scrub_3c"]
    scrub5 = data["scrub_5c"]
    c5bg = data["c5_byz_grad"]
    c5bd = data["c5_backdoor"]
    c5lf = data["c5_label_flip"]
    c5mr = data["c5_model_replacement"]
    comb = data["combined_layer3"]
    fs = data["full_stack"]
    fs_ops_payload = data.get("full_stack_ops") or {}
    fs_ops = fs_ops_payload.get("summary") or {}
    fs_latency = fs_ops_payload.get("shared_inference_latency") or {}
    agg_overhead = (data.get("aggregation_overhead") or {}).get("results") or {}
    probe_config = data.get("probe_config") or {}
    attack_intensity = data.get("attack_intensity") or {}
    l1_component = data.get("l1_component") or {}
    clean_l1l2 = data.get("clean_l1l2") or {}
    probe_rows = data.get("probe_cap_rows") or []
    layer3_ops = data.get("layer3_ops") or {}

    lines: List[str] = [
        "% Auto-generated by scripts/sync_manuscript_numbers.py — do not edit by hand.",
        "",
    ]

    # Main F1
    lines.extend([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Mean test F1 ($\\pm$ std) on the SITL drone packet FL benchmark "
        "(3 clients, 12 rounds, 10 seeds).}",
        "\\label{tab:main-f1}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Defender & Byzantine & Grad.\\ scaling & Backdoor \\\\",
        "\\midrule",
    ])
    for defender in DEFENDERS:
        cells = [_cell(journal, a, defender) for a in ATTACKS_MAIN]
        if defender == "krum":
            cells[2] = f"\\textbf{{{cells[2]}}}"
        if defender == "acfl_full":
            cells[0] = f"\\textbf{{{cells[0]}}}"
            cells[1] = f"\\textbf{{{cells[1]}}}"
        lines.append(f"{DEFENDER_TEX[defender]} & {' & '.join(cells)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])

    # Main gap
    lines.extend([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Mean test F1 gap versus FedAvg (per-seed paired difference; 10 seeds).}",
        "\\label{tab:main-gap}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Defender & Byz.\\ & Grad.\\ & Backdoor \\\\",
        "\\midrule",
    ])
    for defender in DEFENDERS[1:]:
        gaps = [_gap(journal, f"{a}/{defender}_minus_fedavg") for a in ATTACKS_MAIN]
        cells = [_fmt_gap(g) for g in gaps]
        label = DEFENDER_TEX[defender]
        if defender == "acfl_full":
            cells[0] = f"\\textbf{{{cells[0]}}}"
            cells[1] = f"\\textbf{{{cells[1]}}}"
        lines.append(f"{label} & {' & '.join(cells)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    # Backdoor trigger
    lines.extend([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Backdoor: clean-task F1 and trigger $P(\\text{attack})$ "
        "(lower is better; 10 seeds).}",
        "\\label{tab:backdoor-trigger}",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Defender & $P(\\text{attack})$ & Mean F1 \\\\",
        "\\midrule",
    ])
    for defender in DEFENDERS:
        trig = _trigger(journal, defender)
        f1 = _cell(journal, "backdoor", defender)
        label = DEFENDER_TEX[defender]
        if defender == "acfl_full":
            trig = f"\\textbf{{{trig}}}"
        lines.append(f"{label} & {trig} & {f1} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    # Ablation (F1 across full threat matrix)
    best_f1: Dict[str, float] = {}
    for attack in ATTACKS_ABLATION:
        vals = []
        for defender in ABLATION_DEFENDERS:
            block = ablation.get(f"{attack}/{defender}", {})
            if block.get("mean_f1") is not None:
                vals.append(block["mean_f1"])
        if vals:
            best_f1[attack] = max(vals)
    ncols = "l" + ("c" * len(ATTACKS_ABLATION))
    header = " & ".join(ABLATION_ATTACK_TEX[a] for a in ATTACKS_ABLATION)
    lines.extend([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Layer ablation: mean test F1 ($\\pm$ std) over 10 seeds (3 clients).}",
        "\\label{tab:ablation}",
        "\\small",
        f"\\begin{{tabular}}{{{ncols}}}",
        "\\toprule",
        f"Configuration & {header} \\\\",
        "\\midrule",
    ])
    for defender in ABLATION_DEFENDERS:
        cells = []
        for attack in ATTACKS_ABLATION:
            block = ablation.get(f"{attack}/{defender}", {})
            cell = _ms(block.get("mean_f1"), block.get("std_f1"))
            mean = block.get("mean_f1")
            if (
                mean is not None
                and attack in best_f1
                and abs(mean - best_f1[attack]) < 1e-9
            ):
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{ABLATION_TEX[defender]} & {' & '.join(cells)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])

    # Ablation trust AUC companion
    lines.extend([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Layer ablation: malicious-client trust AUC over 10 seeds (FedAvg has no trust signal).}",
        "\\label{tab:ablation-trust}",
        "\\small",
        f"\\begin{{tabular}}{{{ncols}}}",
        "\\toprule",
        f"Configuration & {header} \\\\",
        "\\midrule",
    ])
    for defender in ABLATION_DEFENDERS:
        cells = []
        for attack in ATTACKS_ABLATION:
            block = ablation.get(f"{attack}/{defender}", {})
            if defender == "fedavg" or block.get("mean_trust_auc") is None:
                cells.append("---")
            else:
                cells.append(f"{block['mean_trust_auc']:.3f}")
        lines.append(f"{ABLATION_TEX[defender]} & {' & '.join(cells)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])

    # Threat gaps (full defender set when baselines present)
    threat_defenders = [
        ("fedavg", "FedAvg"),
        ("coordinate_median", "Coord.\\ median"),
        ("trimmed_mean", "Trimmed mean"),
        ("krum", "Krum"),
        ("acfl_full", "ACFL (full)"),
    ]
    has_baselines = all(
        f"label_flip/{d}" in threat and f"model_replacement/{d}" in threat
        for d, _ in threat_defenders
    )
    lines.extend([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Threat-model gap attacks (3 clients, 10 seeds): mean test F1 ($\\pm$ std).}",
        "\\label{tab:threat-gaps}",
        "\\small",
        "\\begin{tabular}{lccr}" if has_baselines else "\\begin{tabular}{lcc}",
        "\\toprule",
        (
            "Defender & Label flip & Model repl.\\ & $\\Delta$ LF \\\\"
            if has_baselines
            else "Defender & Label flip & Model repl.\\ \\\\"
        ),
        "\\midrule",
    ])
    if has_baselines:
        for dkey, dlabel in threat_defenders:
            lf = threat[f"label_flip/{dkey}"]
            mr = threat[f"model_replacement/{dkey}"]
            lf_s = _ms(lf["mean_f1"], lf["std_f1"])
            mr_s = _ms(mr["mean_f1"], mr["std_f1"])
            if dkey == "fedavg":
                gap_s = "---"
            else:
                gkey = (
                    "label_flip/acfl_minus_fedavg"
                    if dkey == "acfl_full"
                    else f"label_flip/{dkey}_minus_fedavg"
                )
                gap_s = _fmt_gap(_gap(threat, gkey))
            if dkey == "acfl_full":
                lf_s = f"\\textbf{{{lf_s}}}"
                mr_s = f"\\textbf{{{mr_s}}}"
                gap_s = f"\\textbf{{{gap_s}}}"
            lines.append(f"{dlabel} & {lf_s} & {mr_s} & {gap_s} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    else:
        lf_f = threat["label_flip/fedavg"]
        lf_a = threat["label_flip/acfl_full"]
        mr_f = threat["model_replacement/fedavg"]
        mr_a = threat["model_replacement/acfl_full"]
        lines.extend([
            f"FedAvg & {_ms(lf_f['mean_f1'], lf_f['std_f1'])} & {_ms(mr_f['mean_f1'], mr_f['std_f1'])} \\\\",
            f"ACFL (full) & \\textbf{{{_ms(lf_a['mean_f1'], lf_a['std_f1'])}}} & "
            f"\\textbf{{{_ms(mr_a['mean_f1'], mr_a['std_f1'])}}} \\\\",
            "\\midrule",
            f"$\\Delta$ vs FedAvg & {_fmt_gap(_gap(threat, 'label_flip/acfl_minus_fedavg'))} & "
            f"{_fmt_gap(_gap(threat, 'model_replacement/acfl_minus_fedavg'))} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ])

    # Probe cap ablation
    by_seed: Dict[int, Dict[str, Any]] = defaultdict(dict)
    for row in probe_rows:
        by_seed[int(row["seed"])][row["defender"]] = row
    lines.extend([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Backdoor gating ablation: legacy min-blend vs.\\ probe-aware soft cap (3 seeds).}",
        "\\label{tab:probe-cap-ablation}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Seed & Legacy F1 & Probe F1 & Legacy $P(\\text{atk})$ & Probe $P(\\text{atk})$ \\\\",
        "\\midrule",
    ])
    for seed in sorted(by_seed):
        leg = by_seed[seed].get("acfl_legacy_blend")
        cap = by_seed[seed].get("acfl_probe_cap")
        if not leg or not cap:
            continue
        lines.append(
            f"{seed} & {leg['mean_final_f1']:.3f} & {cap['mean_final_f1']:.3f} & "
            f"{leg['backdoor_mean_attack_prob']:.3f} & {cap['backdoor_mean_attack_prob']:.3f} \\\\"
        )
    pc = data["probe_cap"]
    leg = pc["acfl_legacy_blend"]
    cap = pc["acfl_probe_cap"]
    lines.extend([
        "\\midrule",
        f"Mean & {leg['mean_f1']:.3f} & {cap['mean_f1']:.3f} & "
        f"{leg['mean_backdoor_attack_prob']:.3f} & {cap['mean_backdoor_attack_prob']:.3f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])

    # Scrub ablation
    def _scrub_block(scale_label: str, summary: Dict[str, Any]) -> List[str]:
        vo = summary["acfl_veto_only"]
        sc = summary["acfl_veto_scrub"]
        g = summary["scrub_minus_veto_only"]
        return [
            f"\\multirow{{3}}{{*}}{{{scale_label}}}",
            f"  & Veto only & {_ms(vo['mean_f1'], vo['std_f1'])} & "
            f"{_ms(vo['mean_backdoor_attack_prob'], vo['std_backdoor_attack_prob'])} \\\\",
            f"  & Veto + scrub & {_ms(sc['mean_f1'], sc['std_f1'])} & "
            f"\\textbf{{{_ms(sc['mean_backdoor_attack_prob'], sc['std_backdoor_attack_prob'])}}} \\\\",
            f"  & Scrub $-$ veto & {_fmt_gap(g['f1_gap'])} & {_fmt_gap(g['trigger_gap'])} \\\\",
        ]

    lines.extend([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Backdoor scrub ablation (10 seeds): shortlist-veto only vs.\\ veto + "
        "post-aggregation trigger scrub at $n{=}3$ (journal) and $n{=}5$ "
        "(20\\% malicious, min.\\ quorum 4).}",
        "\\label{tab:scrub-ablation}",
        "\\small",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Scale & Variant & Mean F1 & $P(\\text{attack})$ \\\\",
        "\\midrule",
    ])
    lines.extend(_scrub_block("$n{=}3$", scrub3))
    lines.append("\\midrule")
    lines.extend(_scrub_block("$n{=}5$", scrub5))
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    # 5-client byz/grad
    lines.extend([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Five-client benchmark (20\\% malicious, 10 seeds): Byzantine and "
        "gradient-scaling mean test F1 ($\\pm$ std).}",
        "\\label{tab:5client-byz-grad}",
        "\\small",
        "\\begin{tabular}{lccr}",
        "\\toprule",
        "Defender & Byzantine & Grad.\\ scaling & $\\Delta$ F1 (Byz.) \\\\",
        "\\midrule",
    ])
    for defender in DEFENDERS:
        byz = _cell(c5bg, "byzantine", defender)
        grad = _cell(c5bg, "gradient_scaling", defender)
        gap = _fmt_gap(_gap(c5bg, f"byzantine/{defender}_minus_fedavg"))
        if defender == "fedavg":
            gap = "---"
        label = DEFENDER_TEX[defender]
        if defender == "acfl_full":
            byz = f"\\textbf{{{byz}}}"
            grad = f"\\textbf{{{grad}}}"
            gap = f"\\textbf{{{gap}}}"
        lines.append(f"{label} & {byz} & {grad} & {gap} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    # 5-client backdoor
    lines.extend([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Five-client backdoor benchmark (5 clients, min.\\ 4/round, 20\\% malicious, "
        "10 seeds): clean F1 and trigger $P(\\text{attack})$.}",
        "\\label{tab:5client-backdoor}",
        "\\small",
        "\\begin{tabular}{lccr}",
        "\\toprule",
        "Defender & Mean F1 & $P(\\text{attack})$ & $\\Delta$ F1 \\\\",
        "\\midrule",
    ])
    for defender in DEFENDERS:
        f1 = _cell(c5bd, "backdoor", defender)
        trig = _trigger(c5bd, defender)
        gap = _fmt_gap(_gap(c5bd, f"backdoor/{defender}_minus_fedavg"))
        if defender == "fedavg":
            gap = "---"
        label = DEFENDER_TEX[defender]
        if defender == "acfl_full":
            f1 = f"\\textbf{{{f1}}}"
            trig = f"\\textbf{{{trig}}}"
            gap = f"\\textbf{{{gap}}}"
        lines.append(f"{label} & {f1} & {trig} & {gap} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    # 5-client label flip
    lines.extend([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Five-client label-flip benchmark (50\\% local flip, 20\\% malicious, 10 seeds): "
        "mean test F1 ($\\pm$ std).}",
        "\\label{tab:5client-label-flip}",
        "\\small",
        "\\begin{tabular}{lcr}",
        "\\toprule",
        "Defender & Mean F1 & $\\Delta$ vs FedAvg \\\\",
        "\\midrule",
    ])
    for defender in DEFENDERS:
        f1 = _cell(c5lf, "label_flip", defender)
        gap = _fmt_gap(_gap(c5lf, f"label_flip/{defender}_minus_fedavg"))
        if defender == "fedavg":
            gap = "---"
        label = DEFENDER_TEX[defender]
        if defender == "acfl_full":
            f1 = f"\\textbf{{{f1}}}"
            gap = f"\\textbf{{{gap}}}"
        lines.append(f"{label} & {f1} & {gap} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    # 5-client model replacement
    lines.extend([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Five-client model-replacement benchmark ($\\alpha{=}10$ anti-direction, "
        "20\\% malicious, minimum quorum four for ACFL/Krum, 10 seeds): mean test F1 ($\\pm$ std).}",
        "\\label{tab:5client-model-replacement}",
        "\\small",
        "\\begin{tabular}{lcr}",
        "\\toprule",
        "Defender & Mean F1 & $\\Delta$ vs FedAvg \\\\",
        "\\midrule",
    ])
    for defender in DEFENDERS:
        f1 = _cell(c5mr, "model_replacement", defender)
        gap = _fmt_gap(_gap(c5mr, f"model_replacement/{defender}_minus_fedavg"))
        if defender == "fedavg":
            gap = "---"
        label = DEFENDER_TEX[defender]
        if defender == "acfl_full":
            f1 = f"\\textbf{{{f1}}}"
            gap = f"\\textbf{{{gap}}}"
        lines.append(f"{label} & {f1} & {gap} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\[0.5em]",
        "{\\footnotesize ACFL (full): Layer~1 probe trust in aggregation, attack-scoped hard norm "
        "rejection (withheld if too few clients remain), trust AUC "
        f"{c5mr['model_replacement/acfl_full']['mean_trust_auc']:.3f}. "
        "FedAvg/median/trimmed rows from minimum-quorum-three extension; ACFL and Krum "
        "regenerated at quorum four.}",
        "\\end{table}",
        "",
    ])

    # Layer3 standalone + operational (from compute_layer3_operational_metrics.py)
    if layer3_ops:
        ttd = layer3_ops["time_to_detect"]
        fa = layer3_ops["false_alarms"]
        lat = layer3_ops["inference_latency"]
        by_attack = layer3_ops.get("by_network_attack", {})
        lines.extend([
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Layer~3 spoofing detection on held-out SITL packet missions "
            "(ACFL checkpoint, probe-calibrated $P(\\text{attack})$).}",
            "\\label{tab:layer3}",
            "\\small",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Metric & Value & Notes \\\\",
            "\\midrule",
            f"Aggregate AUROC & {layer3_ops['aggregate_auroc']:.3f} & 10 agents, held-out missions \\\\",
            f"Aggregate F1 & {layer3_ops['aggregate_f1']:.3f} & threshold tuned on eval windows \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "% BEGIN layer3-operational-table",
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Layer~3 operational metrics on held-out SITL missions "
            "(time-to-detect, false alarms, inference latency).}",
            "\\label{tab:layer3-operational}",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{@{}p{0.36\\columnwidth}p{0.30\\columnwidth}p{0.28\\columnwidth}@{}}",
            "\\toprule",
            "Metric & Value & Notes \\\\",
            "\\midrule",
            f"Mean time-to-detect & {ttd['mean_delay_seconds']:.2f}\\,s "
            f"({ttd['mean_delay_windows']:.1f} win.) & det.\\ rate {ttd['mean_detection_rate']:.2f} \\\\",
            f"False-alarm rate & {fa['aggregate_false_alarm_rate']:.3f} & "
            f"{fa['total_false_positives']} FP / {fa['total_benign_windows']} benign \\\\",
            f"False alarms/hour & {fa['benign_missions_mean_false_alarms_per_hour']:.1f} & "
            f"median {fa['benign_missions_median_false_alarms_per_hour']:.1f} \\\\",
            f"Inference latency & {lat['mean_ms_per_window']:.2f}\\,ms & "
            f"p95 {lat['p95_ms_per_window']:.2f}\\,ms \\\\",
        ])
        for atk in ("gps", "none", "waypoint"):
            block = by_attack.get(atk, {})
            t = block.get("time_to_detect", {})
            f = block.get("false_alarms", {})
            ttd_s = (
                f"{t['mean_delay_seconds']:.2f}\\,s"
                if t.get("mean_delay_seconds") is not None
                else "N/A"
            )
            lines.append(
                f"{atk} missions & TTD {ttd_s} & "
                f"FA {f.get('aggregate_false_alarm_rate', 0):.3f} \\\\"
            )
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "% END layer3-operational-table",
            "",
        ])

    # Combined layer3
    lines.extend([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Combined network+learning Layer~3 validation: poisoned FL checkpoint then "
        "Layer~3 on held-out waypoint/GNSS missions (10 seeds, FedAvg vs ACFL).}",
        "\\label{tab:combined-layer3}",
        "\\small",
        "\\begin{tabular}{lccr}",
        "\\toprule",
        "Learning attack & FedAvg AUROC & ACFL AUROC & $\\Delta$ \\\\",
        "\\midrule",
    ])
    for key, label in COMBINED_ATTACKS:
        fed = comb[f"{key}/fedavg"]
        acfl = comb[f"{key}/acfl_full"]
        delta = _gap(comb, f"{key}/acfl_minus_fedavg_layer3_auroc")
        fed_s = f"{fed['mean_layer3_auroc']:.3f}"
        if fed.get("std_layer3_auroc"):
            fed_s = _ms(fed["mean_layer3_auroc"], fed["std_layer3_auroc"])
        acfl_s = _ms(acfl["mean_layer3_auroc"], acfl["std_layer3_auroc"])
        d_s = _fmt_gap(delta)
        if key in ("byzantine", "model_replacement") and delta and delta > 0.1:
            acfl_s = f"\\textbf{{{acfl_s}}}"
            d_s = f"\\textbf{{{d_s}}}"
        lines.append(f"{label} & {fed_s} & {acfl_s} & {d_s} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\[0.5em]",
        "{\\footnotesize Shared FedAvg/ACFL-full \emph{attack} cells use the same 3-client "
        "checkpoints as Table~\\ref{tab:full-stack} (ablation for Byz./grad./backdoor; "
        "threat-gap for label flip and model replacement), so Layer~3 AUROC means match "
        "bit-for-bit. The benign row is a matched 12-round "
        "\\texttt{config\\_classification\\_drone} control (seeds 42--51), not the legacy "
        "5-round \\texttt{classification\\_benign} checkpoint. FedAvg gradient-scaling "
        "AUROC variance includes two seeds whose poisoned checkpoints emit non-finite "
        "detector scores (sanitized to chance-level AUROC~$0.5$).}",
        "\\end{table*}",
        "",
    ])

    # Full stack
    lines.extend([
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Full-stack validation: Layer~1 trust AUC, Layer~2 clean F1, and Layer~3 "
        "spoofing AUROC on the \\emph{same} checkpoint per configuration (10 seeds, $\\pm$ std "
        "over seeds). FedAvg has no Layer~1 trust signal.}",
        "\\label{tab:full-stack}",
        "\\small",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Attack & Stack & L1 trust AUC & L2 F1 & L3 AUROC \\\\",
        "\\midrule",
    ])
    fs_attacks = [
        ("byzantine", "Byzantine", [k for k, _ in STACK_KEYS]),
        ("gradient_scaling", "Grad.\\ scaling", [k for k, _ in STACK_KEYS]),
        ("backdoor", "Backdoor", [k for k, _ in STACK_KEYS]),
        ("label_flip", "Label flip", [k for k, _ in STACK_KEYS]),
        ("model_replacement", "Model repl.", [k for k, _ in STACK_KEYS]),
    ]
    stack_label_map = dict(STACK_KEYS)
    for attack, label, stack_keys in fs_attacks:
        n_stacks = len(stack_keys)
        # Best (max) per column across stacks, for bolding.
        best_l2 = max(
            (fs[f"{attack}/{k}"].get("mean_layer2_f1") or float("-inf"))
            for k in stack_keys
        )
        best_l3 = max(
            (fs[f"{attack}/{k}"].get("mean_layer3_auroc") or float("-inf"))
            for k in stack_keys
        )
        for i, stack_key in enumerate(stack_keys):
            block = fs[f"{attack}/{stack_key}"]
            stack_label = stack_label_map.get(stack_key, stack_key)
            l1 = "---"
            if block.get("mean_layer1_trust_auc") is not None:
                l1 = _ms(block["mean_layer1_trust_auc"], block.get("std_layer1_trust_auc"))
            l2 = _ms(block["mean_layer2_f1"], block["std_layer2_f1"])
            l3 = _ms(block["mean_layer3_auroc"], block["std_layer3_auroc"])
            atk_cell = f"\\multirow{{{n_stacks}}}{{*}}{{{label}}}" if i == 0 else ""
            if attack in ("byzantine", "gradient_scaling") and stack_key == "full":
                l1 = f"\\textbf{{{l1}}}"
            # Bold the best L2 F1 and L3 AUROC cell within each attack block.
            if (
                block.get("mean_layer2_f1") is not None
                and abs(block["mean_layer2_f1"] - best_l2) < 1e-9
            ):
                l2 = f"\\textbf{{{l2}}}"
            if (
                block.get("mean_layer3_auroc") is not None
                and abs(block["mean_layer3_auroc"] - best_l3) < 1e-9
            ):
                l3 = f"\\textbf{{{l3}}}"
            lines.append(f"{atk_cell} & {stack_label} & {l1} & {l2} & {l3} \\\\")
        if attack != "model_replacement":
            lines.append("\\midrule")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\\\[0.5em]",
        "{\\footnotesize Bold marks the best mean Layer~2 F1 and Layer~3 AUROC within "
        "each attack block. All rows use 3-client checkpoints (ablation for "
        "Byz./grad./backdoor; threat-gap for label flip and model replacement). "
        "FedAvg/full Layer~3 AUROC means match Table~\\ref{tab:combined-layer3} "
        "bit-for-bit on shared attacks.}",
        "\\end{table*}",
        "",
    ])

    # Full-stack operational metrics (FedAvg + classical + ACFL; per-checkpoint tuned)
    if fs_ops:
        ops_attacks = [
            ("benign", "Benign"),
            ("byzantine", "Byzantine"),
            ("gradient_scaling", "Grad.\\ scaling"),
            ("backdoor", "Backdoor"),
            ("label_flip", "Label flip"),
            ("model_replacement", "Model repl."),
        ]
        ops_stacks = [
            ("fedavg", "FedAvg"),
            ("coordinate_median", "Coord.\\ median"),
            ("trimmed_mean", "Trimmed mean"),
            ("krum", "Krum"),
            ("full", "ACFL (full)"),
        ]
        lines.extend([
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Full-stack Layer~3 operational metrics on held-out SITL missions "
            "(FedAvg, classical robust aggregators, and ACFL-full; 10 seeds; "
            "per-checkpoint F1-tuned threshold). FedAvg/ACFL-full poisoned rows reuse "
            "Table~\\ref{tab:full-stack} checkpoints; classical rows use the matching "
            "journal/threat 3-client runs; benign is the matched 12-round drone control "
            "(FedAvg and ACFL-full only).}",
            "\\label{tab:full-stack-ops}",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\begin{tabular}{llcccc}",
            "\\toprule",
            "Attack & Defender & TTD (s) & Det.\\ rate & FA/hr & "
            "Latency mean/p95 (ms) \\\\",
            "\\midrule",
        ])
        for attack, label in ops_attacks:
            # Available stacks for this attack (skip classical on benign).
            available = []
            for stack_key, stack_label in ops_stacks:
                block = fs_ops.get(f"{attack}/{stack_key}", {})
                if not block.get("available", True):
                    continue
                if block.get("mean_mean_detection_rate") is None and block.get(
                    "mean_benign_missions_mean_false_alarms_per_hour"
                ) is None:
                    continue
                available.append((stack_key, stack_label, block))
            if not available:
                continue
            # Best (lowest) FA/hr within the attack block.
            fa_vals = [
                b.get("mean_benign_missions_mean_false_alarms_per_hour")
                for _, _, b in available
                if b.get("mean_benign_missions_mean_false_alarms_per_hour") is not None
            ]
            best_fa = min(fa_vals) if fa_vals else None
            n = len(available)
            for i, (stack_key, stack_label, block) in enumerate(available):
                ttd = _ms_digits(
                    block.get("mean_mean_delay_seconds"),
                    block.get("std_mean_delay_seconds"),
                    digits=2,
                )
                det = _ms_digits(
                    block.get("mean_mean_detection_rate"),
                    block.get("std_mean_detection_rate"),
                    digits=3,
                )
                fahr = _ms_digits(
                    block.get("mean_benign_missions_mean_false_alarms_per_hour"),
                    block.get("std_benign_missions_mean_false_alarms_per_hour"),
                    digits=1,
                )
                latency_mean = (fs_latency.get("mean_ms_per_window") or {}).get("mean")
                latency_p95 = (fs_latency.get("p95_ms_per_window") or {}).get("mean")
                latency = (
                    f"{latency_mean:.2f}/{latency_p95:.2f}"
                    if latency_mean is not None and latency_p95 is not None
                    else "---"
                )
                fa = block.get("mean_benign_missions_mean_false_alarms_per_hour")
                if (
                    best_fa is not None
                    and fa is not None
                    and abs(fa - best_fa) < 1e-9
                ):
                    fahr = f"\\textbf{{{fahr}}}"
                atk_cell = f"\\multirow{{{n}}}{{*}}{{{label}}}" if i == 0 else ""
                lines.append(
                    f"{atk_cell} & {stack_label} & {ttd} & {det} & {fahr} & "
                    f"{latency} \\\\"
                )
            if attack != "model_replacement":
                lines.append("\\midrule")
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\[0.5em]",
            "{\\footnotesize TTD $=$ mean time-to-detect from labeled attack onset; "
            "det.\\ rate $=$ fraction of attack onsets detected; FA/hr $=$ false alarms "
            "per operating hour on benign (no-network-attack) missions; latency $=$ "
            "shared mean/p95 CPU inference time per packet window, measured sequentially "
            "over 10 repeats on the common TinyPacketCNN + spoofing-detector path. FA/hr "
            "is a duration-normalized \\emph{rate} (false positives divided by each mission's "
            "true span, then scaled to one hour), not an hour-long simulation: eval "
            "missions run $\\approx$2--5\\,min, so the extrapolation factor is "
            "$\\approx$12--28$\\times$. Bold marks the lowest mean FA/hr within each "
            "attack. Collapsed checkpoints (e.g.\\ FedAvg under model replacement) "
            "saturate detections by flagging nearly every benign window (det.\\ rate "
            "$\\approx 1$ with FA/hr in the thousands). Classical aggregators have no "
            "matched benign control under the 12-round drone protocol.}",
            "\\end{table*}",
            "",
        ])

    # Aggregation overhead (server-side per-round training cost)
    if agg_overhead:
        # Prefer the drone production probe path when available.
        acfl_key = (
            "acfl_full_drone"
            if "acfl_full_drone" in agg_overhead
            else "acfl_full"
        )
        overhead_rows = [
            ("fedavg", "FedAvg"),
            ("coordinate_median", "Coord.\\ median"),
            ("trimmed_mean", "Trimmed mean"),
            ("krum", "Krum"),
            (acfl_key, "ACFL (L1+L2)"),
        ]
        fedavg_mean = (agg_overhead.get("fedavg") or {}).get("mean_ms")
        lines.extend([
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Server-side per-round aggregation overhead (CPU, single thread; "
            "3 clients, drone TinyPacketCNN; mean $\\pm$ std over 50 rounds). Layers~1--2 "
            "run once per FL round at training time; this is distinct from the "
            "inference-time Layer~3 latency in Table~\\ref{tab:full-stack-ops}.}",
            "\\label{tab:aggregation-overhead}",
            "\\small",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Aggregator & Per-round (ms) & $\\times$ FedAvg \\\\",
            "\\midrule",
        ])
        for key, label in overhead_rows:
            block = agg_overhead.get(key) or {}
            mean = block.get("mean_ms")
            std = block.get("std_ms")
            cell = _ms_digits(mean, std, digits=2)
            ratio = (
                f"{mean / fedavg_mean:.1f}$\\times$"
                if mean is not None and fedavg_mean
                else "---"
            )
            lines.append(f"{label} & {cell} & {ratio} \\\\")
        l1_only = (agg_overhead.get("acfl_l1_only") or {}).get("mean_ms")
        l2_only = (agg_overhead.get("acfl_l2_only") or {}).get("mean_ms")
        full_legacy = (agg_overhead.get("acfl_full") or {}).get("mean_ms")
        full_drone = (agg_overhead.get("acfl_full_drone") or {}).get("mean_ms")
        decomp_parts = []
        if l1_only is not None and l2_only is not None:
            decomp_parts.append(
                f"With full probes, Layer~1 is {l1_only:.1f}\\,ms and Layer~2 is "
                f"{l2_only:.1f}\\,ms"
            )
        if full_legacy is not None and full_drone is not None:
            decomp_parts.append(
                f"batched probe evaluation cuts the prior $\\approx$58\\,ms stack to "
                f"{full_legacy:.1f}\\,ms; the drone path (16 windows/scenario + "
                f"attack-scoped subsets) is {full_drone:.1f}\\,ms"
            )
        decomp = (". " + "; ".join(decomp_parts) + ".") if decomp_parts else ""
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\[0.5em]",
            "{\\footnotesize Overhead is a once-per-round \\emph{training} cost, not a "
            "per-window inference cost; over a 12-round run the extra wall-clock is "
            f"sub-second in absolute terms{decomp} Client updates are weight-scaled "
            "perturbations of a real drone checkpoint so probe forward passes are "
            "realistic. Production ACFL uses the drone probe path "
            "(Table~\\ref{tab:probe-config-ablation}).}",
            "\\end{table}",
            "",
        ])

    # Probe-config quality ablation (full probes vs drone 16+scoped)
    if probe_config:
        probe_attacks = [
            ("byzantine", "Byzantine"),
            ("backdoor", "Backdoor"),
            ("label_flip", "Label flip"),
        ]
        lines.extend([
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Layer~1 probe-config ablation on ACFL-full (3 clients, 12 rounds, "
            "10 seeds): full probe set (all scenarios $\\times$ 48 windows) vs.\\ the "
            "drone production path (16 windows/scenario + attack-scoped subsets). "
            "$\\Delta$ is drone$-$full (paired).}",
            "\\label{tab:probe-config-ablation}",
            "\\small",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Attack & Full F1 & Drone F1 & $\\Delta$F1 & $\\Delta$ trust AUC \\\\",
            "\\midrule",
        ])
        for attack, label in probe_attacks:
            full = probe_config.get(f"{attack}/full_probes") or {}
            drone = probe_config.get(f"{attack}/drone_opt") or {}
            gap = probe_config.get(f"{attack}/drone_minus_full") or {}
            full_f1 = _ms(full.get("mean_f1"), full.get("std_f1"))
            drone_f1 = _ms(drone.get("mean_f1"), drone.get("std_f1"))
            df1 = gap.get("mean_f1_gap")
            dauc = gap.get("mean_trust_auc_gap")
            df1_s = f"${df1:+.3f}$" if df1 is not None else "---"
            dauc_s = f"${dauc:+.3f}$" if dauc is not None else "---"
            lines.append(
                f"{label} & {full_f1} & {drone_f1} & {df1_s} & {dauc_s} \\\\"
            )
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\[0.5em]",
            "{\\footnotesize Byzantine and label-flip F1 are essentially unchanged under "
            "the faster path. Backdoor shows a modest clean-F1 cost "
            "($\\approx$0.024) while trust AUC is slightly higher; we retain the drone "
            "path as the production default and note the backdoor tradeoff.}",
            "\\end{table}",
            "",
        ])

    # Attack-intensity sensitivity sweep
    if attack_intensity:
        triples: List[Tuple[str, str, float]] = []
        seen = set()
        for key in attack_intensity:
            if "/acfl_minus_fedavg" in key:
                continue
            try:
                left, defender = key.rsplit("/", 1)
                attack, knobval = left.split("/", 1)
                knob, val_s = knobval.split("=", 1)
                intensity = float(val_s)
            except ValueError:
                continue
            if defender not in DEFENDERS:
                continue
            trip = (attack, knob, intensity)
            if trip not in seen:
                seen.add(trip)
                triples.append(trip)
        triples.sort(key=lambda t: (t[0], t[1], t[2]))
        attack_tex = {
            "gradient_scaling": "Grad.\\ scaling",
            "byzantine": "Byzantine",
            "backdoor": "Backdoor",
            "label_flip": "Label flip",
            "model_replacement": "Model repl.",
        }
        knob_tex = {
            "scale": "scale",
            "noise_std": "$\\sigma$",
            "poison_fraction": "poison",
            "flip_fraction": "flip",
            "alpha": "$\\alpha$",
        }
        n_seeds = None
        for block in attack_intensity.values():
            if isinstance(block, dict) and block.get("n_seeds"):
                n_seeds = block["n_seeds"]
                break
        n_s = str(n_seeds) if n_seeds else "5"
        lines.extend([
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Attack-intensity sensitivity: mean test F1 ($\\pm$ std; "
            f"3 clients, 12 rounds, {n_s} seeds). One knob varies per attack; "
            "$^*$ marks the manuscript primary setting. "
            "$\\Delta$ is ACFL$-$FedAvg (paired).}",
            "\\label{tab:attack-intensity}",
            "\\small",
            "\\setlength{\\tabcolsep}{3.5pt}",
            "\\begin{tabular}{llcccccc}",
            "\\toprule",
            "Attack & Intensity & FedAvg & Coord.\\ med. & Trimmed & Krum & "
            "ACFL & $\\Delta$ \\\\",
            "\\midrule",
        ])
        last_attack = None
        for attack, knob, intensity in triples:
            blocks = {
                d: (attack_intensity.get(f"{attack}/{knob}={intensity:g}/{d}") or {})
                for d in DEFENDERS
            }
            gap = attack_intensity.get(
                f"{attack}/{knob}={intensity:g}/acfl_minus_fedavg"
            ) or {}
            primary = any(
                bool(blocks[d].get("primary_setting")) for d in DEFENDERS
            )
            star = "$^*$" if primary else ""
            attack_label = (
                attack_tex.get(attack, attack) if attack != last_attack else ""
            )
            last_attack = attack
            intensity_label = f"{knob_tex.get(knob, knob)}={intensity:g}{star}"
            cells = [
                _ms(blocks[d].get("mean_f1"), blocks[d].get("std_f1"))
                for d in DEFENDERS
            ]
            # Bold best mean F1 in the row (ACFL preferred on ties).
            means = [
                blocks[d].get("mean_f1") for d in DEFENDERS
            ]
            finite = [m for m in means if m is not None]
            if finite:
                best = max(finite)
                for i, m in enumerate(means):
                    if m is not None and abs(m - best) < 1e-9:
                        cells[i] = f"\\textbf{{{cells[i]}}}"
            dg = gap.get("mean_gap")
            dg_s = f"${dg:+.3f}$" if dg is not None else "---"
            lines.append(
                f"{attack_label} & {intensity_label} & {' & '.join(cells)} & "
                f"{dg_s} \\\\"
            )
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\[0.5em]",
            "{\\footnotesize Primary settings match Section~\\ref{sec:exp-attacks}. "
            "Coordinate median and trimmed mean often sacrifice clean utility; "
            "Krum is seed-volatile on gradient scaling and weak under label flip. "
            "ACFL remains the most stable defender as intensity increases. "
            "Backdoor trigger probabilities are in Table~\\ref{tab:attack-intensity-backdoor}.}",
            "\\end{table*}",
            "",
        ])

        # Backdoor trigger-P companion table
        bd_levels = sorted({
            intensity for attack, knob, intensity in triples
            if attack == "backdoor"
        })
        if bd_levels:
            lines.extend([
                "\\begin{table}[t]",
                "\\centering",
                "\\caption{Backdoor intensity sweep: mean trigger "
                f"$P(\\text{{attack}})$ ({n_s} seeds; lower is better)."
                "}",
                "\\label{tab:attack-intensity-backdoor}",
                "\\small",
                "\\begin{tabular}{lcccc}",
                "\\toprule",
                "Poison frac. & FedAvg & Coord.\\ med. & Krum & ACFL \\\\",
                "\\midrule",
            ])
            for intensity in bd_levels:
                primary = bool(
                    (attack_intensity.get(
                        f"backdoor/poison_fraction={intensity:g}/acfl_full"
                    ) or {}).get("primary_setting")
                )
                star = "$^*$" if primary else ""
                cells = []
                for d in ("fedavg", "coordinate_median", "krum", "acfl_full"):
                    block = attack_intensity.get(
                        f"backdoor/poison_fraction={intensity:g}/{d}"
                    ) or {}
                    p = block.get("mean_backdoor_attack_prob")
                    cells.append(f"{p:.3f}" if p is not None else "---")
                # Bold lowest trigger P
                vals = []
                for d in ("fedavg", "coordinate_median", "krum", "acfl_full"):
                    block = attack_intensity.get(
                        f"backdoor/poison_fraction={intensity:g}/{d}"
                    ) or {}
                    vals.append(block.get("mean_backdoor_attack_prob"))
                finite_p = [v for v in vals if v is not None]
                if finite_p:
                    best_p = min(finite_p)
                    for i, v in enumerate(vals):
                        if v is not None and abs(v - best_p) < 1e-9:
                            cells[i] = f"\\textbf{{{cells[i]}}}"
                lines.append(
                    f"{intensity:g}{star} & {' & '.join(cells)} \\\\"
                )
            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\\\[0.5em]",
                "{\\footnotesize ACFL keeps trigger activation near 0.02 across "
                "poison fractions while matching clean-task F1 "
                "(Table~\\ref{tab:attack-intensity}).}",
                "\\end{table}",
                "",
            ])

        # Layer~1-only vs full ACFL across the same intensity grid
        l1_triples = [
            (attack, knob, intensity)
            for attack, knob, intensity in triples
            if (attack_intensity.get(f"{attack}/{knob}={intensity:g}/acfl_l1_only") or {}).get(
                "mean_f1"
            )
            is not None
            and (attack_intensity.get(f"{attack}/{knob}={intensity:g}/acfl_full") or {}).get(
                "mean_f1"
            )
            is not None
        ]
        if l1_triples:
            lines.extend([
                "\\begin{table*}[t]",
                "\\centering",
                "\\caption{Layer~1 isolation across the attack-intensity sweep "
                f"({n_s} seeds): ACFL with Layer~2 disabled (\\texttt{{acfl\\_l1\\_only}}) "
                "versus the full stack. $\\Delta$F1 is L1-only$-$full (matched "
                "intensity/seed means). $^*$ marks the primary setting.}",
                "\\label{tab:attack-intensity-l1}",
                "\\small",
                "\\begin{tabular}{llcccc}",
                "\\toprule",
                "Attack & Intensity & L1-only F1 & Full F1 & $\\Delta$F1 & "
                "L1 / full trust AUC \\\\",
                "\\midrule",
            ])
            last_attack = None
            for attack, knob, intensity in l1_triples:
                l1 = attack_intensity.get(
                    f"{attack}/{knob}={intensity:g}/acfl_l1_only"
                ) or {}
                full = attack_intensity.get(
                    f"{attack}/{knob}={intensity:g}/acfl_full"
                ) or {}
                primary = bool(
                    l1.get("primary_setting") or full.get("primary_setting")
                )
                star = "$^*$" if primary else ""
                attack_label = (
                    attack_tex.get(attack, attack) if attack != last_attack else ""
                )
                last_attack = attack
                intensity_label = f"{knob_tex.get(knob, knob)}={intensity:g}{star}"
                l1_f1 = _ms(l1.get("mean_f1"), l1.get("std_f1"))
                full_f1 = _ms(full.get("mean_f1"), full.get("std_f1"))
                ml, mf = l1.get("mean_f1"), full.get("mean_f1")
                dg = (ml - mf) if ml is not None and mf is not None else None
                dg_s = f"${dg:+.3f}$" if dg is not None else "---"
                la = l1.get("mean_trust_auc")
                fa = full.get("mean_trust_auc")
                auc_s = (
                    f"{la:.3f} / {fa:.3f}"
                    if la is not None and fa is not None
                    else "---"
                )
                lines.append(
                    f"{attack_label} & {intensity_label} & {l1_f1} & {full_f1} & "
                    f"{dg_s} & {auc_s} \\\\"
                )
            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\\\[0.5em]",
                "{\\footnotesize On gradient scaling, Byzantine, backdoor, and label "
                "flip, L1-only tracks the full stack within $\\approx$0.06 F1 across "
                "intensities. Model replacement is the clear exception: disabling "
                "Layer~2 costs $\\approx$0.14--0.19 F1 at $\\alpha{\\ge}10$, confirming "
                "that hard-rejection/aggregation (Layer~2) is load-bearing for that "
                "threat while Layer~1 still attributes the adversary (trust AUC "
                "$\\approx 1$).}",
                "\\end{table*}",
                "",
            ])

        # Layer~2-only vs full ACFL across the same intensity grid
        l2_triples = [
            (attack, knob, intensity)
            for attack, knob, intensity in triples
            if (attack_intensity.get(f"{attack}/{knob}={intensity:g}/acfl_l2_only") or {}).get(
                "mean_f1"
            )
            is not None
            and (attack_intensity.get(f"{attack}/{knob}={intensity:g}/acfl_full") or {}).get(
                "mean_f1"
            )
            is not None
        ]
        if l2_triples:
            lines.extend([
                "\\begin{table*}[t]",
                "\\centering",
                "\\caption{Layer~2 isolation across the attack-intensity sweep "
                f"({n_s} seeds): ACFL with Layer~1 disabled (\\texttt{{acfl\\_l2\\_only}}) "
                "versus the full stack. $\\Delta$F1 is L2-only$-$full (matched "
                "intensity/seed means). $^*$ marks the primary setting.}}",
                "\\label{tab:attack-intensity-l2}",
                "\\small",
                "\\begin{tabular}{llcccc}",
                "\\toprule",
                "Attack & Intensity & L2-only F1 & Full F1 & $\\Delta$F1 & "
                "L2 / full trust AUC \\\\",
                "\\midrule",
            ])
            last_attack = None
            for attack, knob, intensity in l2_triples:
                l2 = attack_intensity.get(
                    f"{attack}/{knob}={intensity:g}/acfl_l2_only"
                ) or {}
                full = attack_intensity.get(
                    f"{attack}/{knob}={intensity:g}/acfl_full"
                ) or {}
                primary = bool(
                    l2.get("primary_setting") or full.get("primary_setting")
                )
                star = "$^*$" if primary else ""
                attack_label = (
                    attack_tex.get(attack, attack) if attack != last_attack else ""
                )
                last_attack = attack
                intensity_label = f"{knob_tex.get(knob, knob)}={intensity:g}{star}"
                l2_f1 = _ms(l2.get("mean_f1"), l2.get("std_f1"))
                full_f1 = _ms(full.get("mean_f1"), full.get("std_f1"))
                ml, mf = l2.get("mean_f1"), full.get("mean_f1")
                dg = (ml - mf) if ml is not None and mf is not None else None
                dg_s = f"${dg:+.3f}$" if dg is not None else "---"
                la = l2.get("mean_trust_auc")
                fa = full.get("mean_trust_auc")
                auc_s = (
                    f"{la:.3f} / {fa:.3f}"
                    if la is not None and fa is not None
                    else "---"
                )
                lines.append(
                    f"{attack_label} & {intensity_label} & {l2_f1} & {full_f1} & "
                    f"{dg_s} & {auc_s} \\\\"
                )
            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\\\[0.5em]",
                "{\\footnotesize Trust AUC remains at chance ($0.500$) without Layer~1 "
                "at every intensity. Clean F1 tracks the full stack on gradient "
                "scaling/Byzantine/backdoor near the primary settings, but label flip "
                "collapses as the flip rate rises ($\\Delta$F1 $-0.062$ / $-0.323$ at "
                "$0.5$ / $0.75$). Model replacement benefits from Layer~2 aggregation "
                "relative to L1-only, yet still lacks attribution.}",
                "\\end{table*}",
                "",
            ])

    # Layer~1 component leave-one-out
    if l1_component:
        n_s = None
        for _vk, _ in L1_COMPONENT_VARIANTS:
            block = l1_component.get(f"byzantine/{_vk}") or {}
            if block.get("n_seeds"):
                n_s = int(block["n_seeds"])
                break
        n_s = n_s or 10
        best_f1_l1: Dict[str, float] = {}
        for attack in ATTACKS_ABLATION:
            vals = []
            for vk, _ in L1_COMPONENT_VARIANTS:
                block = l1_component.get(f"{attack}/{vk}") or {}
                if block.get("mean_f1") is not None:
                    vals.append(block["mean_f1"])
            if vals:
                best_f1_l1[attack] = max(vals)
        ncols = "l" + ("c" * len(ATTACKS_ABLATION))
        header = " & ".join(ABLATION_ATTACK_TEX[a] for a in ATTACKS_ABLATION)
        lines.extend([
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Layer~1 component leave-one-out (default weights "
            "subspace~$0.50$ / temporal~$0.20$ / CF~$0.30$; remaining weights "
            f"renormalize; ${n_s}$ seeds). Mean test F1 ($\\pm$ std).}}",
            "\\label{tab:l1-component}",
            "\\small",
            f"\\begin{{tabular}}{{{ncols}}}",
            "\\toprule",
            f"L1 mixture & {header} \\\\",
            "\\midrule",
        ])
        for vk, label in L1_COMPONENT_VARIANTS:
            cells = []
            for attack in ATTACKS_ABLATION:
                block = l1_component.get(f"{attack}/{vk}") or {}
                cell = _ms(block.get("mean_f1"), block.get("std_f1"))
                mean = block.get("mean_f1")
                if (
                    mean is not None
                    and attack in best_f1_l1
                    and abs(mean - best_f1_l1[attack]) < 1e-9
                ):
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            lines.append(f"{label} & {' & '.join(cells)} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])

        lines.extend([
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Layer~1 component leave-one-out: malicious-client trust AUC "
            f"({n_s} seeds; same cells as Table~\\ref{{tab:l1-component}}).}}",
            "\\label{tab:l1-component-trust}",
            "\\small",
            f"\\begin{{tabular}}{{{ncols}}}",
            "\\toprule",
            f"L1 mixture & {header} \\\\",
            "\\midrule",
        ])
        for vk, label in L1_COMPONENT_VARIANTS:
            cells = []
            for attack in ATTACKS_ABLATION:
                block = l1_component.get(f"{attack}/{vk}") or {}
                auc = block.get("mean_trust_auc")
                cells.append(f"{auc:.3f}" if auc is not None else "---")
            lines.append(f"{label} & {' & '.join(cells)} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])

    # L1+L2 gating / extras ablation
    if clean_l1l2:
        def _full_block(attack: str) -> Dict[str, Any]:
            if attack in ("label_flip", "model_replacement"):
                return threat.get(f"{attack}/acfl_full") or {}
            return journal.get(f"{attack}/acfl_full") or {}

        present = [
            (vk, label)
            for vk, label in CLEAN_L1L2_VARIANTS
            if vk == "acfl_full"
            or any(
                (clean_l1l2.get(f"{a}/{vk}") or {}).get("mean_f1") is not None
                for a in ATTACKS_ABLATION
            )
        ]
        n_s = None
        for vk, _ in present:
            if vk == "acfl_full":
                continue
            block = clean_l1l2.get(f"byzantine/{vk}") or {}
            if block.get("n_seeds"):
                n_s = int(block["n_seeds"])
                break
        n_s = n_s or 10
        ncols = "l" + ("c" * len(ATTACKS_ABLATION))
        header = " & ".join(ABLATION_ATTACK_TEX[a] for a in ATTACKS_ABLATION)
        lines.extend([
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{L1/L2 and gating/extras ablation "
            f"({n_s} seeds). Clean arms disable attack-scoped extras "
            "(shortlist-veto/scrub/hard-reject); ungated keeps extras with "
            "Layer~1/2 gating off. Production ACFL (full) is shown for reference. "
            "Mean test F1 ($\\pm$ std).}}",
            "\\label{tab:l1l2-gating-ablation}",
            "\\small",
            f"\\begin{{tabular}}{{{ncols}}}",
            "\\toprule",
            f"Configuration & {header} \\\\",
            "\\midrule",
        ])
        for vk, label in present:
            cells = []
            for attack in ATTACKS_ABLATION:
                if vk == "acfl_full":
                    block = _full_block(attack)
                else:
                    block = clean_l1l2.get(f"{attack}/{vk}") or {}
                cells.append(_ms(block.get("mean_f1"), block.get("std_f1")))
            lines.append(f"{label} & {' & '.join(cells)} \\\\")
        # Backdoor trigger companion row when available
        trig_cells = []
        have_trig = False
        for vk, _ in present:
            if vk == "acfl_full":
                block = journal.get("backdoor/acfl_full") or {}
                t = block.get("mean_backdoor_attack_prob")
            else:
                block = clean_l1l2.get(f"backdoor/{vk}") or {}
                t = block.get("mean_backdoor_attack_prob")
            if t is not None:
                have_trig = True
                trig_cells.append(f"{t:.3f}")
            else:
                trig_cells.append("---")
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        if have_trig:
            lines.extend([
                "\\\\[0.5em]",
                "{\\footnotesize Backdoor trigger $P(\\text{attack})$: "
                + ", ".join(
                    f"{lab} {val}"
                    for (vk, lab), val in zip(present, trig_cells)
                    if val != "---"
                )
                + ".}",
            ])
        lines.extend(["\\end{table*}", ""])

    # Byzantine compact
    lines.extend([
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Byzantine attack compact summary (3-client FL, 10 seeds; same cells as "
        "Table~\\ref{tab:main-f1}).}",
        "\\label{tab:byzantine-10seed}",
        "\\small",
        "\\begin{tabular}{lcr}",
        "\\toprule",
        "Defender & Mean F1 & $\\Delta$ vs FedAvg \\\\",
        "\\midrule",
        f"FedAvg & {_cell(journal, 'byzantine', 'fedavg')} & --- \\\\",
        f"ACFL (full) & \\textbf{{{_cell(journal, 'byzantine', 'acfl_full')}}} & "
        f"\\textbf{{{_fmt_gap(_gap(journal, 'byzantine/acfl_full_minus_fedavg'))}}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def build_narrative(data: Dict[str, Any]) -> Dict[str, Any]:
    journal = data["journal"]
    threat = data["threat"]
    c5bd = data["c5_backdoor"]
    c5lf = data["c5_label_flip"]
    c5mr = data["c5_model_replacement"]
    c5bg = data["c5_byz_grad"]
    scrub3 = data["scrub_3c"]
    scrub5 = data["scrub_5c"]
    comb = data["combined_layer3"]
    fs = data["full_stack"]

    bd_acfl = journal["backdoor/acfl_full"]["mean_backdoor_attack_prob"]
    bd_fed = journal["backdoor/fedavg"]["mean_backdoor_attack_prob"]
    bd_krum = journal["backdoor/krum"]["mean_backdoor_attack_prob"]
    bd5_acfl = c5bd["backdoor/acfl_full"]["mean_backdoor_attack_prob"]
    bd5_krum = c5bd["backdoor/krum"]["mean_backdoor_attack_prob"]

    return {
        "byzantine_acfl_f1": journal["byzantine/acfl_full"]["mean_f1"],
        "byzantine_fedavg_f1": journal["byzantine/fedavg"]["mean_f1"],
        "byzantine_gap": _gap(journal, "byzantine/acfl_full_minus_fedavg"),
        "grad_acfl_f1": journal["gradient_scaling/acfl_full"]["mean_f1"],
        "grad_fedavg_f1": journal["gradient_scaling/fedavg"]["mean_f1"],
        "grad_gap": _gap(journal, "gradient_scaling/acfl_full_minus_fedavg"),
        "backdoor_acfl_f1": journal["backdoor/acfl_full"]["mean_f1"],
        "backdoor_fedavg_f1": journal["backdoor/fedavg"]["mean_f1"],
        "backdoor_krum_f1": journal["backdoor/krum"]["mean_f1"],
        "backdoor_f1_gap": _gap(journal, "backdoor/acfl_full_minus_fedavg"),
        "backdoor_trig_acfl": bd_acfl,
        "backdoor_trig_fedavg": bd_fed,
        "backdoor_trig_krum": bd_krum,
        "backdoor_trig_reduction_fedavg_pct": round(_pct_reduction(bd_fed, bd_acfl)),
        "backdoor_trig_reduction_krum_pct": round(_pct_reduction(bd_krum, bd_acfl)),
        "label_flip_acfl": threat["label_flip/acfl_full"]["mean_f1"],
        "label_flip_fedavg": threat["label_flip/fedavg"]["mean_f1"],
        "label_flip_gap": _gap(threat, "label_flip/acfl_minus_fedavg"),
        "mr_acfl": threat["model_replacement/acfl_full"]["mean_f1"],
        "mr_fedavg": threat["model_replacement/fedavg"]["mean_f1"],
        "mr_gap": _gap(threat, "model_replacement/acfl_minus_fedavg"),
        "c5_byz_acfl": c5bg["byzantine/acfl_full"]["mean_f1"],
        "c5_grad_acfl": c5bg["gradient_scaling/acfl_full"]["mean_f1"],
        "c5_byz_gap": _gap(c5bg, "byzantine/acfl_full_minus_fedavg"),
        "c5_grad_gap": _gap(c5bg, "gradient_scaling/acfl_full_minus_fedavg"),
        "c5_backdoor_acfl_trig": bd5_acfl,
        "c5_backdoor_krum_trig": bd5_krum,
        "c5_backdoor_acfl_f1": c5bd["backdoor/acfl_full"]["mean_f1"],
        "c5_backdoor_krum_f1": c5bd["backdoor/krum"]["mean_f1"],
        "c5_lf_acfl": c5lf["label_flip/acfl_full"]["mean_f1"],
        "c5_lf_krum": c5lf["label_flip/krum"]["mean_f1"],
        "c5_lf_fedavg": c5lf["label_flip/fedavg"]["mean_f1"],
        "c5_lf_gap": _gap(c5lf, "label_flip/acfl_full_minus_fedavg"),
        "c5_mr_acfl": c5mr["model_replacement/acfl_full"]["mean_f1"],
        "c5_mr_krum": c5mr["model_replacement/krum"]["mean_f1"],
        "c5_mr_gap": _gap(c5mr, "model_replacement/acfl_full_minus_fedavg"),
        "scrub3_veto_trig": scrub3["acfl_veto_only"]["mean_backdoor_attack_prob"],
        "scrub3_scrub_trig": scrub3["acfl_veto_scrub"]["mean_backdoor_attack_prob"],
        "scrub3_trig_gap": scrub3["scrub_minus_veto_only"]["trigger_gap"],
        "scrub5_veto_trig": scrub5["acfl_veto_only"]["mean_backdoor_attack_prob"],
        "scrub5_scrub_trig": scrub5["acfl_veto_scrub"]["mean_backdoor_attack_prob"],
        "scrub5_trig_gap": scrub5["scrub_minus_veto_only"]["trigger_gap"],
        "combined_byz_delta": _gap(comb, "byzantine/acfl_minus_fedavg_layer3_auroc"),
        "combined_grad_delta": _gap(comb, "gradient_scaling/acfl_minus_fedavg_layer3_auroc"),
        "full_byz_fedavg_l3": fs["byzantine/fedavg"]["mean_layer3_auroc"],
        "full_byz_full_l3": fs["byzantine/full"]["mean_layer3_auroc"],
        "full_byz_l3_gap": _gap(fs, "byzantine/full_minus_fedavg_layer3_auroc"),
        "full_lf_l2_gap": _gap(fs, "label_flip/full_minus_fedavg_layer2_f1"),
        "full_lf_l3_gap": _gap(fs, "label_flip/full_minus_fedavg_layer3_auroc"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync manuscript tables from benchmark JSON")
    parser.add_argument(
        "--tables-out",
        type=Path,
        default=MANUSCRIPT / "tables_data.tex",
    )
    parser.add_argument(
        "--narrative-out",
        type=Path,
        default=TUNING / "manuscript_narrative.json",
    )
    args = parser.parse_args()

    data: Dict[str, Any] = {}
    missing = []
    for name, path in SOURCES.items():
        if path.is_file():
            data[name] = _load(path)
        else:
            missing.append(str(path))
    if missing:
        print("Warning: missing sources:")
        for m in missing:
            print(f"  - {m}")

    tables_tex = build_tables_data(data)
    args.tables_out.write_text(tables_tex + "\n")
    print(f"Wrote {args.tables_out}")

    narrative = build_narrative(data)
    args.narrative_out.write_text(json.dumps(narrative, indent=2) + "\n")
    print(f"Wrote {args.narrative_out}")


if __name__ == "__main__":
    main()
