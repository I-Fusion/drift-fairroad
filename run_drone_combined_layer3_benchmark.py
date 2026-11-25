#!/usr/bin/env python3
"""Combined network+learning attack validation for Layer 3.

Threat model:
  Phase A — Learning plane: train a global model under FL poisoning (malicious
            client submits Byzantine/backdoor/etc. updates).
  Phase B — Network plane: evaluate Layer 3 spoofing detection on held-out
            SITL missions that contain real waypoint/GNSS injection traffic.

Outputs live under ``evaluation_results/combined_layer3/`` so existing
``comprehensive_validation`` and ``acfl_tuning`` artifacts stay untouched.

Usage:
    # Full matrix (slow — runs FL + Layer 3 per cell):
    ./acfl_env/bin/python run_drone_combined_layer3_benchmark.py

    # Reuse journal FL checkpoints, run Layer 3 only (fast):
    ./acfl_env/bin/python run_drone_combined_layer3_benchmark.py \\
        --layer3-only \\
        --checkpoint-root evaluation_results/comprehensive_validation
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ACFL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ACFL_ROOT))

from acfl.experiment_seeds import JOURNAL_SEEDS, seeds_csv  # noqa: E402
from run_comprehensive_validation import (  # noqa: E402
    CHECKPOINT_DIR,
    OUT_ROOT,
    PYTHON,
    SITL_PKT,
    Scenario,
    _fl_eval_summary,
    generate_probes,
    run_scenario,
    sync_agent_missions,
)

COMBINED_ROOT = ACFL_ROOT / "evaluation_results" / "combined_layer3"
SCENARIO_ROOT = COMBINED_ROOT / "scenarios"
LAYER3_ROOT = COMBINED_ROOT / "layer3"
PROBE_FILE = str(Path(CHECKPOINT_DIR) / "acfl_probes_sitl.pt")
SEEDS = JOURNAL_SEEDS

LEARNING_ATTACKS = [
    {"name": "benign", "attack_type": "none", "attack_scale": 1.0},
    {
        "name": "gradient_scaling",
        "attack_type": "gradient_scaling",
        "attack_scale": 10.0,
    },
    {"name": "byzantine", "attack_type": "byzantine", "attack_scale": 1.0},
    {
        "name": "backdoor",
        "attack_type": "backdoor",
        "attack_scale": 1.0,
        "backdoor_poison_fraction": 0.35,
    },
    {
        "name": "label_flip",
        "attack_type": "label_flip",
        "attack_scale": 1.0,
        "label_flip_fraction": 0.5,
    },
    {
        "name": "model_replacement",
        "attack_type": "model_replacement",
        "attack_scale": 10.0,
    },
]


DEFENDERS = [
    {
        "name": "fedavg",
        "baseline_aggregation": "fedavg",
        "defender_family": "baseline",
    },
    {
        "name": "acfl_full",
        "baseline_aggregation": None,
        "defender_family": "acfl",
        "trust_power": 1.25,
        "min_trust": 0.03,
        "coord_thr": 2.5,
        "norm_thr": 2.0,
        "norm2_thr": 1.2,
    },
]


def _scenario(defender: Dict, attack: Dict, seed: int) -> Scenario:
    return Scenario(
        name=f"combined_{attack['name']}_seed{seed}_{defender['name']}",
        data_dir=SITL_PKT,
        config="config_classification_drone",
        learning_mode="classification",
        malicious_clients="" if attack["attack_type"] == "none" else "client_2",
        attack_type=attack["attack_type"],
        attack_scale=attack.get("attack_scale", 1.0),
        baseline_aggregation=defender.get("baseline_aggregation"),
        num_clients=3,
        min_clients=2,
        target_rounds=12,
        max_fl_rounds=12,
        probe_file=(
            PROBE_FILE
            if defender["name"] == "acfl_full" or attack["attack_type"] == "backdoor"
            else None
        ),
        acfl_trust_weight_power=defender.get("trust_power"),
        acfl_min_trust_weight=defender.get("min_trust"),
        acfl_coordinate_deviation_threshold=defender.get("coord_thr"),
        acfl_norm_ratio_threshold=defender.get("norm_thr"),
        acfl_norm_ratio_threshold_two_clients=defender.get("norm2_thr"),
        backdoor_poison_fraction=attack.get("backdoor_poison_fraction"),
        label_flip_fraction=attack.get("label_flip_fraction"),
        fl_seed=seed,
    )


def _journal_scenario_name(attack: Dict, seed: int, defender: Dict) -> str:
    if attack["attack_type"] == "none":
        return f"classification_benign"
    return f"journal_{attack['name']}_seed{seed}_{defender['name']}"


def _resolve_checkpoint(
    attack: Dict,
    seed: int,
    defender: Dict,
    checkpoint_roots: List[Path],
    scenario_root: Path,
) -> Optional[str]:
    """Find a server checkpoint aligned with full-stack provenance.

    Priority for poisoned attacks (matches ``run_drone_full_stack_benchmark``):
      ablation → threat → journal → combined → 5-client (last resort).

    Benign prefers matched 12-round ``combined_benign_seed*`` checkpoints
    (same drone config / rounds as attack rows); falls back to the legacy
    short ``classification_benign`` model only if those are missing.
    """
    combined_name = _scenario(defender, attack, seed).name
    journal_name = _journal_scenario_name(attack, seed, defender)
    five_client_name = f"5client_{attack['name']}_seed{seed}_{defender['name']}"
    threat_name = f"threat_{attack['name']}_seed{seed}_{defender['name']}"
    ablation_name = f"ablation_{defender['name']}_{attack['name']}_seed{seed}"

    if attack["attack_type"] == "none":
        rel_paths = [
            Path(combined_name) / "server_checkpoint.pt",
            Path(combined_name) / "fl_checkpoints" / "server_round_12.pt",
            Path("classification_benign") / "server_checkpoint.pt",
            Path("classification_benign") / "fl_checkpoints" / "server_round_12.pt",
            Path("classification_fedavg_benign") / "server_checkpoint.pt",
            Path("classification_fedavg_benign") / "fl_checkpoints" / "server_round_12.pt",
        ]
    else:
        candidates = [
            ablation_name,
            threat_name,
            journal_name,
            combined_name,
            five_client_name,
        ]
        rel_paths = []
        for name in candidates:
            rel_paths.append(Path(name) / "server_checkpoint.pt")
            rel_paths.append(Path(name) / "fl_checkpoints" / "server_round_12.pt")

    for root in checkpoint_roots:
        for rel in rel_paths:
            path = root / rel
            if path.is_file():
                return str(path)
    return None


def _run_layer3_eval(
    checkpoint: str,
    output_json: Path,
    probe_blend_weight: float = 0.35,
) -> Dict:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        str(ACFL_ROOT / "evaluate_layer3_sitl.py"),
        "--data-dir",
        SITL_PKT,
        "--config",
        "config_classification_drone",
        "--checkpoint",
        checkpoint,
        "--probe-file",
        PROBE_FILE,
        "--probe-blend-weight",
        str(probe_blend_weight),
        "--output",
        str(output_json),
    ]
    print(f"  Layer 3 eval -> {output_json.name}")
    rc = subprocess.run(cmd, cwd=str(ACFL_ROOT), check=False).returncode
    if rc != 0 or not output_json.is_file():
        return {"ok": False, "rc": rc}
    report = json.loads(output_json.read_text())
    report["ok"] = True
    return report


def _layer3_summary(report: Dict) -> Dict:
    if not report.get("ok"):
        return {}
    by_net = report.get("by_network_attack") or {}
    waypoint = by_net.get("waypoint", {})
    gps = by_net.get("gps", {})
    return {
        "layer3_auroc": report.get("aggregate_auroc"),
        "layer3_f1": report.get("aggregate_f1"),
        "layer3_precision": report.get("aggregate_precision"),
        "layer3_recall": report.get("aggregate_recall"),
        "layer3_threshold": report.get("threshold"),
        "layer3_waypoint_auroc": waypoint.get("auroc"),
        "layer3_gps_auroc": gps.get("auroc"),
        "layer3_waypoint_f1": waypoint.get("f1"),
        "layer3_gps_f1": gps.get("f1"),
        "probe_file": report.get("probe_file"),
    }


def _row(
    defender: Dict,
    attack: Dict,
    seed: int,
    metrics: Dict,
    layer3_report: Dict,
    checkpoint: Optional[str],
) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {})) if metrics else {}
    l3 = _layer3_summary(layer3_report)
    return {
        "seed": seed,
        "learning_attack": attack["name"],
        "learning_attack_type": attack["attack_type"],
        "defender": defender["name"],
        "defender_family": defender.get("defender_family"),
        "scenario": metrics.get("scenario") if metrics else _scenario(defender, attack, seed).name,
        "mean_final_f1": fl.get("mean_final_f1"),
        "trust_auc": (metrics.get("acfl_eval") or {}).get("trust_score_auc") if metrics else None,
        "checkpoint": checkpoint,
        "layer3_report": str(
            LAYER3_ROOT
            / f"{attack['name']}_{defender['name']}_seed{seed}.json"
        ),
        **l3,
    }


def _summarize(rows: List[Dict]) -> Dict:
    summary: Dict = {}
    for attack in LEARNING_ATTACKS:
        attack_name = attack["name"]
        for defender in DEFENDERS:
            defender_name = defender["name"]
            subset = [
                r for r in rows
                if r["learning_attack"] == attack_name and r["defender"] == defender_name
            ]
            aurocs = [
                r["layer3_auroc"] for r in subset if r.get("layer3_auroc") is not None
            ]
            f1s = [r["layer3_f1"] for r in subset if r.get("layer3_f1") is not None]
            fl_f1s = [
                r["mean_final_f1"] for r in subset if r.get("mean_final_f1") is not None
            ]
            key = f"{attack_name}/{defender_name}"
            if aurocs:
                summary[key] = {
                    "mean_layer3_auroc": statistics.mean(aurocs),
                    "std_layer3_auroc": statistics.pstdev(aurocs) if len(aurocs) > 1 else 0.0,
                    "mean_layer3_f1": statistics.mean(f1s) if f1s else None,
                    "mean_fl_f1": statistics.mean(fl_f1s) if fl_f1s else None,
                    "n_seeds": len(aurocs),
                    "layer3_auroc_values": aurocs,
                }

        fedavg_rows = sorted(
            [
                r for r in rows
                if r["learning_attack"] == attack_name and r["defender"] == "fedavg"
            ],
            key=lambda r: r["seed"],
        )
        acfl_rows = sorted(
            [
                r for r in rows
                if r["learning_attack"] == attack_name and r["defender"] == "acfl_full"
            ],
            key=lambda r: r["seed"],
        )
        n = min(len(fedavg_rows), len(acfl_rows))
        if n:
            auroc_gaps = []
            f1_gaps = []
            for i in range(n):
                fa = fedavg_rows[i].get("layer3_auroc")
                aa = acfl_rows[i].get("layer3_auroc")
                if fa is not None and aa is not None:
                    auroc_gaps.append(aa - fa)
                ff = fedavg_rows[i].get("layer3_f1")
                af = acfl_rows[i].get("layer3_f1")
                if ff is not None and af is not None:
                    f1_gaps.append(af - ff)
            if auroc_gaps:
                summary[f"{attack_name}/acfl_minus_fedavg_layer3_auroc"] = {
                    "mean_gap": statistics.mean(auroc_gaps),
                    "std_gap": statistics.pstdev(auroc_gaps) if len(auroc_gaps) > 1 else 0.0,
                    "gaps": auroc_gaps,
                }
            if f1_gaps:
                summary[f"{attack_name}/acfl_minus_fedavg_layer3_f1"] = {
                    "mean_gap": statistics.mean(f1_gaps),
                    "std_gap": statistics.pstdev(f1_gaps) if len(f1_gaps) > 1 else 0.0,
                    "gaps": f1_gaps,
                }

    return summary


def _write_outputs(rows: List[Dict], summary: Dict):
    COMBINED_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed", "learning_attack", "learning_attack_type", "defender",
        "defender_family", "scenario", "mean_final_f1", "trust_auc", "checkpoint",
        "layer3_auroc", "layer3_f1", "layer3_precision", "layer3_recall",
        "layer3_threshold", "layer3_waypoint_auroc", "layer3_gps_auroc",
        "layer3_waypoint_f1", "layer3_gps_f1", "layer3_report", "probe_file",
    ]
    csv_path = COMBINED_ROOT / "combined_layer3_benchmark.csv"
    json_path = COMBINED_ROOT / "combined_layer3_benchmark.json"
    summary_path = COMBINED_ROOT / "combined_layer3_benchmark_summary.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    json_path.write_text(json.dumps(rows, indent=2, default=str))
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")


def _load_existing_rows() -> List[Dict]:
    json_path = COMBINED_ROOT / "combined_layer3_benchmark.json"
    if not json_path.exists():
        return []
    return json.loads(json_path.read_text())


def _merge_rows(existing: List[Dict], new_rows: List[Dict]) -> List[Dict]:
    replaced = {
        (r["seed"], r["learning_attack"], r["defender"]) for r in new_rows
    }
    kept = [
        r for r in existing
        if (r.get("seed"), r.get("learning_attack"), r.get("defender")) not in replaced
    ]
    return kept + new_rows


def main():
    parser = argparse.ArgumentParser(
        description="Combined network+learning Layer 3 validation benchmark"
    )
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--layer3-only", action="store_true",
                        help="Skip FL training; reuse checkpoints from --checkpoint-root")
    parser.add_argument(
        "--checkpoint-root",
        action="append",
        default=None,
        help="Scenario checkpoint directory (repeatable; default: comprehensive_validation + combined/scenarios)",
    )
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--seeds", default=seeds_csv())
    parser.add_argument(
        "--learning-attacks",
        default=",".join(a["name"] for a in LEARNING_ATTACKS),
    )
    parser.add_argument(
        "--defenders",
        default=",".join(d["name"] for d in DEFENDERS),
    )
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument("--scenario-timeout", type=int, default=2400)
    parser.add_argument("--probe-blend-weight", type=float, default=0.35)
    args = parser.parse_args()

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    seeds = [int(s) for s in args.seeds.split(",")]
    attack_pool = [
        a for a in LEARNING_ATTACKS if a["name"] in args.learning_attacks.split(",")
    ]
    defender_pool = [d for d in DEFENDERS if d["name"] in args.defenders.split(",")]
    if args.checkpoint_root:
        checkpoint_roots = [Path(p) for p in args.checkpoint_root]
    else:
        checkpoint_roots = [
            ACFL_ROOT / "evaluation_results" / "comprehensive_validation",
            SCENARIO_ROOT,
        ]
    scenario_root = SCENARIO_ROOT

    existing = _load_existing_rows() if args.merge_existing else []
    rows: List[Dict] = []
    port = args.port

    for attack in attack_pool:
        for seed in seeds:
            for defender in defender_pool:
                sc = _scenario(defender, attack, seed)
                metrics: Dict = {}
                checkpoint: Optional[str] = None

                if args.layer3_only:
                    checkpoint = _resolve_checkpoint(
                        attack, seed, defender, checkpoint_roots, scenario_root
                    )
                    if not checkpoint:
                        roots_str = ", ".join(str(r) for r in checkpoint_roots)
                        print(f"SKIP {sc.name}: no checkpoint under [{roots_str}]")
                        continue
                    print(f"\n{'='*60}\nLAYER3-ONLY: {sc.name}\n  ckpt={checkpoint}\n{'='*60}")
                else:
                    metrics = run_scenario(
                        sc,
                        port=port,
                        scenario_timeout_s=args.scenario_timeout,
                        out_root=str(scenario_root),
                    )
                    metrics["scenario"] = sc.name
                    checkpoint = metrics.get("checkpoint")
                    port += 1

                layer3_out = LAYER3_ROOT / f"{attack['name']}_{defender['name']}_seed{seed}.json"
                if not checkpoint:
                    print(f"SKIP Layer 3 for {sc.name}: missing checkpoint")
                    continue

                layer3_report = _run_layer3_eval(
                    checkpoint,
                    layer3_out,
                    probe_blend_weight=args.probe_blend_weight,
                )
                rows.append(_row(defender, attack, seed, metrics, layer3_report, checkpoint))
                merged = _merge_rows(existing, rows)
                _write_outputs(merged, _summarize(merged))

    merged = _merge_rows(existing, rows)
    summary = _summarize(merged)
    _write_outputs(merged, summary)

    print("\nCombined Layer 3 summary (mean AUROC):")
    for attack in attack_pool:
        print(f"\n  [learning={attack['name']}]")
        for defender in defender_pool:
            key = f"{attack['name']}/{defender['name']}"
            stats = summary.get(key)
            if not stats:
                continue
            gap = summary.get(f"{attack['name']}/acfl_minus_fedavg_layer3_auroc", {})
            gap_str = ""
            if defender["name"] == "acfl_full" and gap:
                gap_str = f"  vs_fedavg={gap['mean_gap']:+.4f}"
            print(
                f"    {defender['name']}: "
                f"layer3_auroc={stats['mean_layer3_auroc']:.4f} "
                f"std={stats['std_layer3_auroc']:.4f} "
                f"n={stats['n_seeds']}"
                f"{gap_str}"
            )


if __name__ == "__main__":
    main()
