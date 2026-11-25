#!/usr/bin/env python3
"""Backfill fl_eval / F1 for completed probe-config ablation scenarios.

Uses quick FL eval (final round only) so the job finishes under low RAM and
survives when launched via setsid/nohup outside the Cursor terminal.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from run_comprehensive_validation import OUT_ROOT, SITL_PKT, evaluate_scenario
from run_drone_probe_config_ablation import (
    ATTACKS,
    PROBE_CONFIGS,
    _row,
    _scenario,
    _write_outputs,
)


def main() -> None:
    rows_path = Path("evaluation_results/acfl_tuning/drone_probe_config_ablation.json")
    rows = json.loads(rows_path.read_text())
    print("data_dir ok", os.path.isdir(SITL_PKT), SITL_PKT, flush=True)

    updated = []
    for i, r in enumerate(rows, 1):
        sc_name = r["scenario"]
        sc_out = os.path.join(OUT_ROOT, sc_name)
        ckpt = os.path.join(sc_out, "fl_checkpoints")
        probe_cfg = next(c for c in PROBE_CONFIGS if c["name"] == r["probe_config"])
        attack = next(a for a in ATTACKS if a["name"] == r["attack"])
        sc = _scenario(probe_cfg, attack, int(r["seed"]))
        sc.quick_fl_eval = True
        mpath = Path(sc_out) / "metrics.json"

        if mpath.exists():
            old = json.loads(mpath.read_text())
            summary = _row(probe_cfg, attack, old, int(r["seed"]))
            if summary.get("mean_final_f1") is not None:
                print(
                    f"[{i}/{len(rows)}] skip {sc_name} "
                    f"F1={summary['mean_final_f1']:.3f}",
                    flush=True,
                )
                for k in ("aggregation_skipped_rounds", "pairwise_penalty_rounds"):
                    if summary.get(k) is None:
                        summary[k] = r.get(k)
                updated.append(summary)
                continue

        print(f"[{i}/{len(rows)}] re-eval {sc_name} ...", flush=True)
        metrics = evaluate_scenario(
            sc,
            sc_out,
            fl_checkpoint_dir=ckpt if os.path.isdir(ckpt) else None,
            final_round=r.get("final_round"),
        )
        if "acfl_eval" not in metrics and (
            Path(sc_out) / "acfl_evaluation.json"
        ).exists():
            metrics["acfl_eval"] = json.loads(
                (Path(sc_out) / "acfl_evaluation.json").read_text()
            )
        if mpath.exists():
            old = json.loads(mpath.read_text())
            old.update({k: v for k, v in metrics.items() if v is not None})
            if "fl_eval" in metrics:
                old["fl_eval"] = metrics["fl_eval"]
            mpath.write_text(json.dumps(old, indent=2, default=str))
            metrics = old
        new_row = _row(probe_cfg, attack, metrics, int(r["seed"]))
        for k in ("aggregation_skipped_rounds", "pairwise_penalty_rounds"):
            if new_row.get(k) is None:
                new_row[k] = r.get(k)
        print(
            f"  F1={new_row.get('mean_final_f1')} trust={new_row.get('trust_auc')}",
            flush=True,
        )
        updated.append(new_row)
        _write_outputs(updated)

    summary = _write_outputs(updated)
    print("\n=== SUMMARY ===", flush=True)
    for k, v in sorted(summary.items()):
        if "minus" in k:
            print(
                f"{k}: ΔF1={v.get('mean_f1_gap')} ΔAUC={v.get('mean_trust_auc_gap')}",
                flush=True,
            )
        else:
            print(
                f"{k}: F1={v.get('mean_f1')} AUC={v.get('mean_trust_auc')} "
                f"n={v.get('n_seeds')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
