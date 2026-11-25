#!/usr/bin/env python3
"""Journal benchmark: seeded FedAvg + robust baselines vs ACFL across FL attacks.

Runs a reproducible matrix over:
  - Defenders: fedavg, coordinate_median, trimmed_mean, krum, acfl_full
  - Attacks: gradient_scaling, byzantine, backdoor
  - Seeds: 42, 43, 44 (configurable)

Outputs:
  - evaluation_results/acfl_tuning/drone_journal_benchmark.csv
  - evaluation_results/acfl_tuning/drone_journal_benchmark.json
  - evaluation_results/acfl_tuning/drone_journal_benchmark_summary.json
"""
import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from acfl.experiment_seeds import JOURNAL_SEEDS, seeds_csv
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
PROBE_FILE = str(Path(CHECKPOINT_DIR) / "acfl_probes_sitl.pt")
SEEDS = JOURNAL_SEEDS

ATTACKS = [
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
]

DEFENDERS = [
    {
        "name": "fedavg",
        "baseline_aggregation": "fedavg",
        "defender_family": "baseline",
    },
    {
        "name": "coordinate_median",
        "baseline_aggregation": "coordinate_median",
        "defender_family": "baseline",
    },
    {
        "name": "trimmed_mean",
        "baseline_aggregation": "trimmed_mean",
        "defender_family": "baseline",
    },
    {
        "name": "krum",
        "baseline_aggregation": "krum",
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
        name=f"journal_{attack['name']}_seed{seed}_{defender['name']}",
        data_dir=SITL_PKT,
        config="config_classification_drone",
        learning_mode="classification",
        malicious_clients="client_2",
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
        fl_seed=seed,
    )


def _scenario_has_checkpoint(scenario_name: str) -> bool:
    sc_out = Path(OUT_ROOT) / scenario_name
    return (
        (sc_out / "server_checkpoint.pt").is_file()
        or (sc_out / "fl_checkpoints" / "server_round_12.pt").is_file()
    )


def _load_existing_metrics(scenario_name: str) -> Dict:
    metrics_path = Path(OUT_ROOT) / scenario_name / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
        metrics.setdefault("scenario", scenario_name)
        return metrics
    return {"scenario": scenario_name}


def _row(defender: Dict, attack: Dict, seed: int, metrics: Dict) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {}))
    bd = metrics.get("backdoor_eval") or {}
    ckpt_round = bd.get("checkpoint_round")
    return {
        "seed": seed,
        "attack": attack["name"],
        "attack_type": attack["attack_type"],
        "defender": defender["name"],
        "defender_family": defender.get("defender_family"),
        "baseline_aggregation": defender.get("baseline_aggregation"),
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


def _summarize(rows: List[Dict]) -> Dict:
    summary: Dict = {}
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
                    "n_seeds": len(f1s),
                    "mean_backdoor_attack_prob": (
                        statistics.mean(probs) if probs else None
                    ),
                }

        fedavg_rows = sorted(
            [r for r in rows if r.get("attack") == attack_name and r["defender"] == "fedavg"],
            key=lambda r: r["seed"],
        )
        fedavg_f1s = [r["mean_final_f1"] for r in fedavg_rows if r.get("mean_final_f1") is not None]
        for defender in DEFENDERS:
            name = defender["name"]
            if name == "fedavg":
                continue
            key = f"{attack_name}/{name}"
            if key not in summary or not fedavg_f1s:
                continue
            def_rows = sorted(
                [r for r in rows if r.get("attack") == attack_name and r["defender"] == name],
                key=lambda r: r["seed"],
            )
            def_f1s = [r["mean_final_f1"] for r in def_rows if r.get("mean_final_f1") is not None]
            n = min(len(def_f1s), len(fedavg_f1s))
            if n == 0:
                continue
            gaps = [def_f1s[i] - fedavg_f1s[i] for i in range(n)]
            if gaps:
                summary[f"{attack_name}/{name}_minus_fedavg"] = {
                    "mean_gap": statistics.mean(gaps),
                    "std_gap": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
                    "gaps": gaps,
                }

    return summary


def _write_outputs(rows: List[Dict], summary: Dict):
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed", "attack", "attack_type", "defender", "defender_family",
        "baseline_aggregation", "scenario", "mean_final_f1",
        "mean_final_accuracy", "trust_auc", "backdoor_success_rate",
        "backdoor_trigger_accuracy", "backdoor_mean_attack_prob",
        "checkpoint_round", "final_round", "num_checkpoints",
    ]
    csv_path = TUNING_ROOT / "drone_journal_benchmark.csv"
    json_path = TUNING_ROOT / "drone_journal_benchmark.json"
    summary_path = TUNING_ROOT / "drone_journal_benchmark_summary.json"

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
    return csv_path, json_path, summary_path


def _load_existing_rows() -> List[Dict]:
    json_path = TUNING_ROOT / "drone_journal_benchmark.json"
    if not json_path.exists():
        return []
    return json.loads(json_path.read_text())


def _merge_rows(existing: List[Dict], new_rows: List[Dict]) -> List[Dict]:
    replaced = {(r["seed"], r["attack"], r["defender"]) for r in new_rows}
    kept = [
        r for r in existing
        if (r.get("seed"), r.get("attack"), r.get("defender")) not in replaced
    ]
    return kept + new_rows


def main():
    parser = argparse.ArgumentParser(
        description="Journal benchmark: baselines + ACFL across seeded attacks"
    )
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=7100)
    parser.add_argument("--seeds", default=seeds_csv())
    parser.add_argument(
        "--attacks",
        default=",".join(a["name"] for a in ATTACKS),
        help="Comma-separated attacks",
    )
    parser.add_argument(
        "--defenders",
        default=",".join(d["name"] for d in DEFENDERS),
        help="Comma-separated defenders",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Keep benchmark rows not being re-run",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse metrics.json when scenario checkpoint already exists",
    )
    parser.add_argument("--scenario-timeout", type=int, default=1200)
    args = parser.parse_args()

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    seeds = [int(s) for s in args.seeds.split(",")]
    attack_pool = [a for a in ATTACKS if a["name"] in args.attacks.split(",")]
    defender_pool = [d for d in DEFENDERS if d["name"] in args.defenders.split(",")]

    existing = _load_existing_rows() if args.merge_existing else []
    rows: List[Dict] = []
    port = args.port

    for attack in attack_pool:
        for seed in seeds:
            for defender in defender_pool:
                sc = _scenario(defender, attack, seed)
                if args.skip_existing and _scenario_has_checkpoint(sc.name):
                    print(f"SKIP (existing): {sc.name}")
                    metrics = _load_existing_metrics(sc.name)
                else:
                    metrics = run_scenario(
                        sc, port=port, scenario_timeout_s=args.scenario_timeout
                    )
                    metrics["scenario"] = sc.name
                    port += 1
                rows.append(_row(defender, attack, seed, metrics))
                merged = _merge_rows(existing, rows)
                _write_outputs(merged, _summarize(merged))

    merged = _merge_rows(existing, rows)
    summary = _summarize(merged)
    _write_outputs(merged, summary)

    print("\nJournal benchmark summary (mean F1):")
    for attack in attack_pool:
        print(f"\n  [{attack['name']}]")
        for defender in defender_pool:
            key = f"{attack['name']}/{defender['name']}"
            stats = summary.get(key)
            if not stats:
                continue
            gap_key = f"{attack['name']}/{defender['name']}_minus_fedavg"
            gap = summary.get(gap_key, {})
            gap_str = ""
            if gap:
                gap_str = f"  gap_vs_fedavg={gap['mean_gap']:+.4f}"
            print(
                f"    {defender['name']}: "
                f"mean_f1={stats['mean_f1']:.4f} "
                f"std={stats['std_f1']:.4f} "
                f"n={stats['n_seeds']}"
                f"{gap_str}"
            )


if __name__ == "__main__":
    main()
