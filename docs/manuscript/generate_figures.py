#!/usr/bin/env python3
"""Generate publication figures for the ACFL journal manuscript."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "evaluation_results/acfl_tuning/drone_journal_benchmark_summary.json"
OUT_DIR = Path(__file__).resolve().parent / "figures"

ATTACKS = ["byzantine", "gradient_scaling", "backdoor"]
ATTACK_LABELS = ["Byzantine", "Grad.\ scaling", "Backdoor"]
DEFENDERS = ["fedavg", "coordinate_median", "trimmed_mean", "krum", "acfl_full"]
DEFENDER_LABELS = ["FedAvg", "Median", "Trimmed", "Krum", "ACFL"]

COLORS = {
    "fedavg": "#6c757d",
    "coordinate_median": "#adb5bd",
    "trimmed_mean": "#ced4da",
    "krum": "#495057",
    "acfl_full": "#0d6efd",
}


def _load_summary() -> dict:
    if not SUMMARY.exists():
        raise FileNotFoundError(f"Missing benchmark summary: {SUMMARY}")
    return json.loads(SUMMARY.read_text())


def _f1_matrix(summary: dict) -> np.ndarray:
    rows = []
    for defender in DEFENDERS:
        row = []
        for attack in ATTACKS:
            key = f"{attack}/{defender}"
            row.append(summary[key]["mean_f1"])
        rows.append(row)
    return np.array(rows)


def plot_main_f1(summary: dict) -> Path:
    data = _f1_matrix(summary)
    x = np.arange(len(ATTACKS))
    width = 0.15

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (defender, label) in enumerate(zip(DEFENDERS, DEFENDER_LABELS)):
        offset = (i - 2) * width
        bars = ax.bar(
            x + offset,
            data[i],
            width,
            label=label,
            color=COLORS[defender],
            edgecolor="white",
            linewidth=0.5,
        )
        if defender == "acfl_full":
            for bar in bars:
                bar.set_linewidth(1.2)
                bar.set_edgecolor("#084298")

    ax.set_ylabel("Mean test F1")
    ax.set_xticks(x)
    ax.set_xticklabels(ATTACK_LABELS)
    ax.set_ylim(0, 0.85)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.set_title("FL robustness on SITL drone packet classification (3 clients, 10 seeds)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig_main_f1.pdf"
    png = OUT_DIR / "fig_main_f1.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf


def plot_backdoor_dual(summary: dict) -> Path:
    f1_key = "backdoor"
    defenders = DEFENDERS
    f1s = [summary[f"{f1_key}/{d}"]["mean_f1"] for d in defenders]
    f1_stds = [summary[f"{f1_key}/{d}"]["std_f1"] for d in defenders]
    triggers = [summary[f"{f1_key}/{d}"]["mean_backdoor_attack_prob"] for d in defenders]

    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(defenders))
    bars = ax1.bar(
        x - 0.2,
        f1s,
        0.4,
        yerr=f1_stds,
        capsize=3,
        label="Clean-task F1",
        color="#6c757d",
        edgecolor="white",
    )
    bars[-1].set_color(COLORS["acfl_full"])
    bars[-1].set_edgecolor("#084298")
    ax1.set_ylabel("Mean test F1")
    ax1.set_ylim(0, 0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(DEFENDER_LABELS)

    ax2 = ax1.twinx()
    ax2.plot(
        x + 0.2,
        triggers,
        "o-",
        color="#dc3545",
        linewidth=2,
        markersize=7,
        label=r"Trigger $P(\mathrm{attack})$",
    )
    ax2.set_ylabel(r"Trigger $P(\mathrm{attack})$ (lower is better)")
    ax2.set_ylim(0, 0.45)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=2, frameon=False)
    ax1.set_title("Backdoor: clean utility vs.\ trigger success")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "fig_backdoor_dual.pdf"
    png = OUT_DIR / "fig_backdoor_dual.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf


def main():
    summary = _load_summary()
    f1_path = plot_main_f1(summary)
    bd_path = plot_backdoor_dual(summary)
    print(f"Wrote {f1_path}")
    print(f"Wrote {bd_path}")


if __name__ == "__main__":
    main()
