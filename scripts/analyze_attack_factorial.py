#!/usr/bin/env python3
"""Two-way ANOVA with interaction for the attack factorial (intensity × n).

Reads evaluation_results/acfl_tuning/drone_attack_factorial.json and writes:
  - drone_attack_factorial_anova.json
  - drone_attack_factorial_anova.md (human-readable)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "evaluation_results" / "acfl_tuning"
ROWS = TUNING / "drone_attack_factorial.json"
OUT_JSON = TUNING / "drone_attack_factorial_anova.json"
OUT_MD = TUNING / "drone_attack_factorial_anova.md"


def _anova_table(df: pd.DataFrame, formula: str) -> Dict[str, Any]:
    model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    out = {}
    for term, row in table.iterrows():
        out[str(term)] = {
            "sum_sq": float(row["sum_sq"]) if pd.notna(row["sum_sq"]) else None,
            "df": float(row["df"]) if pd.notna(row["df"]) else None,
            "F": float(row["F"]) if pd.notna(row.get("F")) else None,
            "PR(>F)": float(row["PR(>F)"]) if pd.notna(row.get("PR(>F)")) else None,
        }
    # Cell means for interaction plot-style reporting
    means = (
        df.groupby(["intensity_code", "num_clients"], observed=True)["mean_final_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    cell_means = [
        {
            "intensity_code": r.intensity_code,
            "num_clients": int(r.num_clients),
            "mean_f1": float(r["mean"]),
            "std_f1": float(r["std"]) if pd.notna(r["std"]) else 0.0,
            "n": int(r["count"]),
        }
        for r in means.itertuples(index=False)
    ]
    return {"anova": out, "cell_means": cell_means, "n_rows": int(len(df))}


def main() -> None:
    if not ROWS.exists():
        raise SystemExit(f"Missing {ROWS}; run benchmarks/run_drone_attack_factorial.py first")
    rows = json.loads(ROWS.read_text())
    df = pd.DataFrame(rows)
    df = df[df["mean_final_f1"].notna()].copy()
    df["num_clients"] = df["num_clients"].astype(int)
    df["intensity_code"] = df["intensity_code"].astype(str)

    results: Dict[str, Any] = {
        "design": {
            "type": "2x2 full factorial (pyDOE2.fullfact)",
            "factor_A": "attack intensity (L/H extremes)",
            "factor_B": "num_clients (3 vs 5; malicious fraction 33% vs 20%)",
            "response": "mean_final_f1",
            "formula": "mean_final_f1 ~ C(intensity_code) * C(num_clients)",
        },
        "by_attack_defender": {},
    }

    md_lines = [
        "# Attack factorial ANOVA (intensity × n_clients)",
        "",
        "Design: `pyDOE2.fullfact([2,2])` per attack.",
        "Factors: intensity (L/H) × num_clients (3/5).",
        "Model: `mean_final_f1 ~ C(intensity_code) * C(num_clients)` (Type-II ANOVA).",
        "",
    ]

    for attack in sorted(df["attack"].unique()):
        for defender in sorted(df["defender"].unique()):
            sub = df[(df["attack"] == attack) & (df["defender"] == defender)]
            # Need all 4 cells represented
            cells = sub.groupby(["intensity_code", "num_clients"]).size()
            if len(cells) < 4 or len(sub) < 8:
                continue
            key = f"{attack}/{defender}"
            try:
                block = _anova_table(
                    sub,
                    "mean_final_f1 ~ C(intensity_code) * C(num_clients)",
                )
            except Exception as exc:  # noqa: BLE001
                block = {"error": str(exc)}
            results["by_attack_defender"][key] = block

            md_lines.append(f"## {attack} / {defender}")
            md_lines.append("")
            if "error" in block:
                md_lines.append(f"Error: {block['error']}")
                md_lines.append("")
                continue
            anova = block["anova"]
            for term in (
                "C(intensity_code)",
                "C(num_clients)",
                "C(intensity_code):C(num_clients)",
                "Residual",
            ):
                t = anova.get(term) or {}
                if not t:
                    continue
                p = t.get("PR(>F)")
                p_s = f"{p:.4g}" if p is not None else "---"
                f_s = f"{t.get('F'):.3f}" if t.get("F") is not None else "---"
                md_lines.append(
                    f"- **{term}**: F={f_s}, p={p_s}, "
                    f"df={t.get('df')}"
                )
            md_lines.append("")
            md_lines.append("Cell means (F1):")
            for cm in block["cell_means"]:
                md_lines.append(
                    f"- intensity={cm['intensity_code']}, n={cm['num_clients']}: "
                    f"{cm['mean_f1']:.3f} ± {cm['std_f1']:.3f} (n={cm['n']})"
                )
            md_lines.append("")

            # Flag significant interaction
            inter = (anova.get("C(intensity_code):C(num_clients)") or {}).get("PR(>F)")
            if inter is not None and inter < 0.05:
                md_lines.append(
                    f"**Significant intensity×n interaction (p={inter:.4g}).**"
                )
                md_lines.append("")

    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
    OUT_MD.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    n_blocks = len(results["by_attack_defender"])
    n_sig = sum(
        1
        for b in results["by_attack_defender"].values()
        if (b.get("anova") or {})
        .get("C(intensity_code):C(num_clients)", {})
        .get("PR(>F)", 1.0)
        is not None
        and (b.get("anova") or {})
        .get("C(intensity_code):C(num_clients)", {})
        .get("PR(>F)", 1.0)
        < 0.05
    )
    print(f"ANOVA blocks: {n_blocks}; significant interactions (p<0.05): {n_sig}")


if __name__ == "__main__":
    main()
