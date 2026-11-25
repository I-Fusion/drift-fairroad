#!/usr/bin/env python3
"""Priority 3: drone-profile attack suite — FedAvg vs ACFL default across attacks."""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from run_comprehensive_validation import (
    CHECKPOINT_DIR,
    OUT_ROOT,
    SITL_PKT,
    Scenario,
    _fl_eval_summary,
    generate_probes,
    run_scenario,
    sync_agent_missions,
)


TUNING_ROOT = Path("evaluation_results/acfl_tuning")
FEDAVG_F1 = 0.674058437408308

ATTACKS = [
    {"name": "gradient_scaling", "attack_type": "gradient_scaling", "attack_scale": 10.0},
    {"name": "byzantine", "attack_type": "byzantine", "attack_scale": 1.0},
    {"name": "backdoor", "attack_type": "backdoor", "attack_scale": 1.0,
     "backdoor_poison_fraction": 0.35},
]

DEFENDERS = [
    {
        "name": "fedavg",
        "fedavg": True,
        "trust_power": None,
        "min_trust": None,
        "coord_thr": None,
        "norm_thr": None,
        "norm2_thr": None,
    },
    {
        "name": "acfl_default_p125_m003",
        "fedavg": False,
        "trust_power": 1.25,
        "min_trust": 0.03,
        "coord_thr": 2.5,
        "norm_thr": 2.0,
        "norm2_thr": 1.2,
    },
    {
        "name": "acfl_conservative_p075_m003",
        "fedavg": False,
        "trust_power": 0.75,
        "min_trust": 0.03,
        "coord_thr": 2.5,
        "norm_thr": 2.0,
        "norm2_thr": 1.2,
    },
]


def _guard_stats(scenario_name: str) -> Dict:
    path = Path(OUT_ROOT) / scenario_name / "acfl_round_reports.json"
    if not path.exists():
        return {}
    reports = json.loads(path.read_text())
    skipped = sum(
        1 for r in reports
        if (r.get("aggregation_guard") or {}).get("skipped")
    )
    pairwise = sum(
        1 for r in reports
        for c in (r.get("aggregation_diagnostics") or {}).get("clients", {}).values()
        if c.get("pairwise_inversion_penalty")
    )
    return {"aggregation_skipped_rounds": skipped, "pairwise_penalty_rounds": pairwise}


def _scenario(defender: Dict, attack: Dict) -> Scenario:
    tag = f"attack_suite_{defender['name']}_{attack['name']}"
    return Scenario(
        name=tag,
        data_dir=SITL_PKT,
        config="config_classification_drone",
        learning_mode="classification",
        malicious_clients="client_2",
        attack_type=attack["attack_type"],
        attack_scale=attack["attack_scale"],
        fedavg=defender["fedavg"],
        num_clients=3,
        min_clients=2,
        target_rounds=12,
        max_fl_rounds=12,
        probe_file=str(Path(CHECKPOINT_DIR) / "acfl_probes_sitl.pt"),
        acfl_trust_weight_power=defender.get("trust_power"),
        acfl_min_trust_weight=defender.get("min_trust"),
        acfl_coordinate_deviation_threshold=defender.get("coord_thr"),
        acfl_norm_ratio_threshold=defender.get("norm_thr"),
        acfl_norm_ratio_threshold_two_clients=defender.get("norm2_thr"),
        backdoor_poison_fraction=attack.get("backdoor_poison_fraction"),
    )


def _row(defender: Dict, attack: Dict, metrics: Dict) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {}))
    mean_f1 = fl.get("mean_final_f1")
    row = {
        "defender": defender["name"],
        "attack": attack["name"],
        "attack_type": attack["attack_type"],
        "attack_scale": attack["attack_scale"],
        "fedavg": defender["fedavg"],
        "trust_power": defender.get("trust_power"),
        "min_trust": defender.get("min_trust"),
        "scenario": metrics.get("scenario"),
        "mean_final_f1": mean_f1,
        "mean_final_accuracy": fl.get("mean_final_accuracy"),
        "f1_vs_fedavg_baseline": (
            mean_f1 - FEDAVG_F1 if mean_f1 is not None else None
        ),
        "trust_auc": metrics.get("acfl_eval", {}).get("trust_score_auc"),
        "final_round": metrics.get("final_round"),
        "num_checkpoints": fl.get("num_checkpoints"),
    }
    row.update(_guard_stats(metrics.get("scenario", "")))
    backdoor = metrics.get("backdoor_eval") or {}
    row["backdoor_success_rate"] = backdoor.get("backdoor_success_rate")
    row["backdoor_trigger_accuracy"] = backdoor.get("backdoor_trigger_accuracy")
    row["backdoor_mean_attack_prob"] = backdoor.get("backdoor_mean_attack_prob")
    return row


def _load_existing_rows() -> List[Dict]:
    json_path = TUNING_ROOT / "drone_attack_suite.json"
    if not json_path.exists():
        return []
    return json.loads(json_path.read_text())


def _merge_rows(existing: List[Dict], new_rows: List[Dict]) -> List[Dict]:
    replaced = {(r["defender"], r["attack"]) for r in new_rows}
    kept = [
        r for r in existing
        if (r.get("defender"), r.get("attack")) not in replaced
    ]
    return kept + new_rows


def _write_outputs(rows: List[Dict]):
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TUNING_ROOT / "drone_attack_suite.csv"
    json_path = TUNING_ROOT / "drone_attack_suite.json"
    fieldnames = [
        "defender", "attack", "attack_type", "attack_scale", "fedavg",
        "trust_power", "min_trust", "scenario", "mean_final_f1",
        "mean_final_accuracy", "f1_vs_fedavg_baseline", "trust_auc",
        "final_round", "num_checkpoints",
        "aggregation_skipped_rounds", "pairwise_penalty_rounds",
        "backdoor_success_rate", "backdoor_trigger_accuracy",
        "backdoor_mean_attack_prob",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    json_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Run drone attack suite")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=5780)
    parser.add_argument(
        "--defenders",
        default=",".join(d["name"] for d in DEFENDERS),
        help="Comma-separated defender names",
    )
    parser.add_argument(
        "--attacks",
        default=",".join(a["name"] for a in ATTACKS),
        help="Comma-separated attack names",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Keep attack-suite rows not being re-run",
    )
    parser.add_argument(
        "--scenario-timeout",
        type=int,
        default=1200,
        help="Wall-clock seconds per scenario (default 1200)",
    )
    args = parser.parse_args()

    if not args.skip_setup:
        print("Preparing agent missions and probe set...")
        sync_agent_missions()
        generate_probes()

    defender_pool = [d for d in DEFENDERS if d["name"] in args.defenders.split(",")]
    attack_pool = [a for a in ATTACKS if a["name"] in args.attacks.split(",")]

    new_rows: List[Dict] = []
    port = args.port
    for attack in attack_pool:
        for defender in defender_pool:
            sc = _scenario(defender, attack)
            metrics = run_scenario(
                sc, port=port, scenario_timeout_s=args.scenario_timeout
            )
            metrics["scenario"] = sc.name
            new_rows.append(_row(defender, attack, metrics))
            port += 1
            rows = _merge_rows(_load_existing_rows(), new_rows) if args.merge_existing else new_rows
            _write_outputs(rows)

    rows = _merge_rows(_load_existing_rows(), new_rows) if args.merge_existing else new_rows
    _write_outputs(rows)
    print("\nAttack suite summary (mean final F1):")
    for attack in ATTACKS:
        name = attack["name"]
        subset = [r for r in rows if r.get("attack") == name]
        if not subset:
            continue
        print(f"\n  [{name}]")
        subset.sort(key=lambda r: r.get("mean_final_f1") or -1, reverse=True)
        for row in subset:
            print(
                f"    {row['defender']}: f1={row.get('mean_final_f1')} "
                f"rounds={row.get('final_round')} "
                f"delta={row.get('f1_vs_fedavg_baseline')} "
                f"trust_auc={row.get('trust_auc')}"
            )


if __name__ == "__main__":
    main()
