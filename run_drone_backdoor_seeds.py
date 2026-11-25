#!/usr/bin/env python3
"""Reproducible seeded evaluation: FedAvg vs ACFL full across attacks and FL seeds."""
import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List

from acfl.experiment_seeds import JOURNAL_SEEDS, seeds_csv
from run_comprehensive_validation import (
    CHECKPOINT_DIR,
    SITL_PKT,
    Scenario,
    _fl_eval_summary,
    generate_probes,
    run_scenario,
    sync_agent_missions,
)


TUNING_ROOT = Path("evaluation_results/acfl_tuning")
PROBE_FILE = str(Path(CHECKPOINT_DIR) / "acfl_probes_sitl.pt")
SEEDS = JOURNAL_SEEDS

ATTACKS = [
    {"name": "byzantine", "attack_type": "byzantine", "attack_scale": 1.0},
    {
        "name": "backdoor",
        "attack_type": "backdoor",
        "attack_scale": 1.0,
        "backdoor_poison_fraction": 0.35,
    },
]

DEFENDERS = [
    {"name": "fedavg", "fedavg": True},
    {
        "name": "acfl_full",
        "fedavg": False,
        "trust_power": 1.25,
        "min_trust": 0.03,
        "coord_thr": 2.5,
        "norm_thr": 2.0,
        "norm2_thr": 1.2,
    },
]


def _scenario(defender: Dict, attack: Dict, seed: int) -> Scenario:
    return Scenario(
        name=f"{attack['name']}_seed{seed}_{defender['name']}",
        data_dir=SITL_PKT,
        config="config_classification_drone",
        learning_mode="classification",
        malicious_clients="client_2",
        attack_type=attack["attack_type"],
        attack_scale=attack.get("attack_scale", 1.0),
        fedavg=defender["fedavg"],
        num_clients=3,
        min_clients=2,
        target_rounds=12,
        max_fl_rounds=12,
        probe_file=PROBE_FILE,
        acfl_trust_weight_power=defender.get("trust_power"),
        acfl_min_trust_weight=defender.get("min_trust"),
        acfl_coordinate_deviation_threshold=defender.get("coord_thr"),
        acfl_norm_ratio_threshold=defender.get("norm_thr"),
        acfl_norm_ratio_threshold_two_clients=defender.get("norm2_thr"),
        backdoor_poison_fraction=attack.get("backdoor_poison_fraction"),
        fl_seed=seed,
    )


def _row(defender: Dict, attack: Dict, seed: int, metrics: Dict) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {}))
    bd = metrics.get("backdoor_eval") or {}
    ckpt_round = bd.get("checkpoint_round")
    return {
        "seed": seed,
        "attack": attack["name"],
        "attack_type": attack["attack_type"],
        "defender": defender["name"],
        "fedavg": defender["fedavg"],
        "scenario": metrics.get("scenario"),
        "mean_final_f1": fl.get("mean_final_f1"),
        "mean_final_accuracy": fl.get("mean_final_accuracy"),
        "trust_auc": metrics.get("acfl_eval", {}).get("trust_score_auc"),
        "backdoor_success_rate": bd.get("backdoor_success_rate"),
        "backdoor_trigger_accuracy": bd.get("backdoor_trigger_accuracy"),
        "backdoor_mean_attack_prob": bd.get("backdoor_mean_attack_prob"),
        "checkpoint_round": ckpt_round,
        "final_round": metrics.get("final_round"),
        "num_checkpoints": fl.get("num_checkpoints"),
    }


def _write_outputs(rows: List[Dict]):
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed", "attack", "attack_type", "defender", "fedavg", "scenario", "mean_final_f1",
        "mean_final_accuracy", "trust_auc", "backdoor_success_rate",
        "backdoor_trigger_accuracy", "backdoor_mean_attack_prob",
        "checkpoint_round", "final_round", "num_checkpoints",
    ]
    csv_path = TUNING_ROOT / "drone_seeded_validation.csv"
    json_path = TUNING_ROOT / "drone_seeded_validation.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    json_path.write_text(json.dumps(rows, indent=2, default=str))

    summary = {}
    for attack in ATTACKS:
        attack_name = attack["name"]
        for defender in DEFENDERS:
            name = defender["name"]
            subset = [
                r for r in rows
                if r["defender"] == name and r.get("attack") == attack_name
            ]
            f1s = [r["mean_final_f1"] for r in subset if r.get("mean_final_f1") is not None]
            probs = [
                r["backdoor_mean_attack_prob"] for r in subset
                if r.get("backdoor_mean_attack_prob") is not None
            ]
            key = f"{attack_name}/{name}"
            if f1s:
                summary[key] = {
                    "mean_f1": statistics.mean(f1s),
                    "std_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                    "f1_values": f1s,
                    "mean_backdoor_attack_prob": (
                        statistics.mean(probs) if probs else None
                    ),
                }
        fed = [r for r in rows if r.get("attack") == attack_name and r["defender"] == "fedavg"]
        acfl = [r for r in rows if r.get("attack") == attack_name and r["defender"] == "acfl_full"]
        if fed and acfl and len(fed) == len(acfl):
            gaps = [
                a["mean_final_f1"] - f["mean_final_f1"]
                for a, f in zip(sorted(acfl, key=lambda r: r["seed"]),
                                sorted(fed, key=lambda r: r["seed"]))
            ]
            summary[f"{attack_name}/acfl_minus_fedavg"] = {
                "mean_gap": statistics.mean(gaps),
                "std_gap": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
                "gaps": gaps,
            }

    summary_path = TUNING_ROOT / "drone_seeded_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run seeded attack validation suite")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=5980)
    parser.add_argument("--seeds", default=seeds_csv())
    parser.add_argument(
        "--attacks",
        default=",".join(a["name"] for a in ATTACKS),
        help="Comma-separated attacks (byzantine, backdoor)",
    )
    parser.add_argument("--scenario-timeout", type=int, default=1200)
    args = parser.parse_args()

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    seeds = [int(s) for s in args.seeds.split(",")]
    attack_pool = [a for a in ATTACKS if a["name"] in args.attacks.split(",")]
    rows: List[Dict] = []
    port = args.port
    for attack in attack_pool:
        for seed in seeds:
            for defender in DEFENDERS:
                sc = _scenario(defender, attack, seed)
                metrics = run_scenario(
                    sc, port=port, scenario_timeout_s=args.scenario_timeout
                )
                metrics["scenario"] = sc.name
                rows.append(_row(defender, attack, seed, metrics))
                port += 1
                _write_outputs(rows)

    summary = _write_outputs(rows)
    print("\nSeeded validation summary:")
    for name, stats in summary.items():
        if name.endswith("/acfl_minus_fedavg"):
            g = stats
            print(
                f"  {name}: mean_gap={g['mean_gap']:+.4f} "
                f"std={g['std_gap']:.4f} per-seed={g['gaps']}"
            )
            continue
        print(
            f"  {name}: mean_f1={stats['mean_f1']:.4f} "
            f"std={stats['std_f1']:.4f} values={stats['f1_values']}"
        )


if __name__ == "__main__":
    main()
