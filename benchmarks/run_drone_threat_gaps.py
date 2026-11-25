#!/usr/bin/env python3
"""Seeded evaluation for remaining threat-model attacks: label_flip, model_replacement.

Defenders (journal 3-client protocol): fedavg, coordinate_median, trimmed_mean,
krum, acfl_full — 10 seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

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


def _scenario(defender: Dict, attack: Dict, seed: int, quick_eval: bool) -> Scenario:
    is_acfl = defender.get("defender_family") == "acfl"
    return Scenario(
        name=f"threat_{attack['name']}_seed{seed}_{defender['name']}",
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
        probe_file=PROBE_FILE if is_acfl else None,
        acfl_trust_weight_power=defender.get("trust_power"),
        acfl_min_trust_weight=defender.get("min_trust"),
        acfl_coordinate_deviation_threshold=defender.get("coord_thr"),
        acfl_norm_ratio_threshold=defender.get("norm_thr"),
        acfl_norm_ratio_threshold_two_clients=defender.get("norm2_thr"),
        label_flip_fraction=attack.get("label_flip_fraction"),
        fl_seed=seed,
        quick_fl_eval=quick_eval,
    )


def _row(defender: Dict, attack: Dict, seed: int, metrics: Dict) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {}))
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
        "final_round": metrics.get("final_round"),
        "num_checkpoints": fl.get("num_checkpoints"),
    }


def _f1s_for(rows: List[Dict], attack_name: str, defender: str) -> List[float]:
    subset = sorted(
        [
            r
            for r in rows
            if r.get("attack") == attack_name and r.get("defender") == defender
        ],
        key=lambda r: int(r["seed"]),
    )
    return [r["mean_final_f1"] for r in subset if r.get("mean_final_f1") is not None]


def _summarize(rows: List[Dict]) -> Dict:
    summary: Dict = {}
    for attack in ATTACKS:
        attack_name = attack["name"]
        for defender in DEFENDERS:
            name = defender["name"]
            f1s = _f1s_for(rows, attack_name, name)
            key = f"{attack_name}/{name}"
            if f1s:
                summary[key] = {
                    "mean_f1": statistics.mean(f1s),
                    "std_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                    "f1_values": f1s,
                    "n_seeds": len(f1s),
                }
        fed_f1s = _f1s_for(rows, attack_name, "fedavg")
        for defender in DEFENDERS:
            name = defender["name"]
            if name == "fedavg":
                continue
            def_f1s = _f1s_for(rows, attack_name, name)
            n = min(len(fed_f1s), len(def_f1s))
            if not n:
                continue
            gaps = [def_f1s[i] - fed_f1s[i] for i in range(n)]
            gap_key = (
                f"{attack_name}/acfl_minus_fedavg"
                if name == "acfl_full"
                else f"{attack_name}/{name}_minus_fedavg"
            )
            summary[gap_key] = {
                "mean_gap": statistics.mean(gaps),
                "std_gap": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
                "gaps": gaps,
            }
    return summary


def _load_existing_rows() -> List[Dict]:
    json_path = TUNING_ROOT / "drone_threat_gaps.json"
    if not json_path.exists():
        return []
    return json.loads(json_path.read_text())


def _merge_rows(existing: List[Dict], new_rows: List[Dict]) -> List[Dict]:
    replaced = {(r.get("seed"), r.get("attack"), r.get("defender")) for r in new_rows}
    kept = [
        r
        for r in existing
        if (r.get("seed"), r.get("attack"), r.get("defender")) not in replaced
    ]
    return kept + new_rows


def _remove_cells(
    existing: List[Dict],
    attack_pool: List[Dict],
    defender_pool: List[Dict],
    seeds: List[int],
) -> List[Dict]:
    targets = {
        (seed, a["name"], d["name"])
        for seed in seeds
        for a in attack_pool
        for d in defender_pool
    }
    return [
        r
        for r in existing
        if (r.get("seed"), r.get("attack"), r.get("defender")) not in targets
    ]


def _write_outputs(rows: List[Dict], summary: Dict) -> None:
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "attack",
        "attack_type",
        "defender",
        "defender_family",
        "baseline_aggregation",
        "scenario",
        "mean_final_f1",
        "mean_final_accuracy",
        "trust_auc",
        "final_round",
        "num_checkpoints",
    ]
    csv_path = TUNING_ROOT / "drone_threat_gaps.csv"
    json_path = TUNING_ROOT / "drone_threat_gaps.json"
    summary_path = TUNING_ROOT / "drone_threat_gaps_summary.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    json_path.write_text(json.dumps(rows, indent=2, default=str))
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")


def _pending(
    attack_pool: List[Dict],
    defender_pool: List[Dict],
    seeds: List[int],
    existing: List[Dict],
):
    done = {(r.get("seed"), r.get("attack"), r.get("defender")) for r in existing}
    for attack in attack_pool:
        for seed in seeds:
            for defender in defender_pool:
                key = (seed, attack["name"], defender["name"])
                if key not in done:
                    yield attack, seed, defender


def main() -> None:
    parser = argparse.ArgumentParser(description="Threat-model gap evaluation")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=7500)
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
        help="Keep threat-gap rows not being re-run; fill missing cells",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Clear requested (seed, attack, defender) cells before running",
    )
    parser.add_argument(
        "--quick-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip FL plots and per-client eval (default: on)",
    )
    parser.add_argument("--scenario-timeout", type=int, default=1800)
    args = parser.parse_args()

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    seeds = [int(s) for s in args.seeds.split(",")]
    attack_pool = [a for a in ATTACKS if a["name"] in args.attacks.split(",")]
    defender_pool = [d for d in DEFENDERS if d["name"] in args.defenders.split(",")]
    existing = (
        _load_existing_rows() if (args.merge_existing or args.force_rerun) else []
    )
    if args.force_rerun:
        existing = _remove_cells(existing, attack_pool, defender_pool, seeds)
        print(
            f"Force rerun: cleared {len(defender_pool) * len(attack_pool) * len(seeds)} "
            f"cell(s)."
        )

    rows: List[Dict] = []
    port = args.port
    pending = list(_pending(attack_pool, defender_pool, seeds, existing))
    total_cells = len(seeds) * len(attack_pool) * len(defender_pool)
    summary: Dict = {}

    if args.merge_existing and not pending:
        print("All requested cells already complete.")
        summary = _summarize(existing)
        _write_outputs(existing, summary)
    else:
        if args.merge_existing:
            print(f"Resuming {len(pending)} missing cell(s) of {total_cells}...")
        for attack, seed, defender in pending:
            sc = _scenario(defender, attack, seed, quick_eval=args.quick_eval)
            print(
                f"\n>>> Running {sc.name} "
                f"(timeout={args.scenario_timeout}s)"
            )
            metrics = run_scenario(
                sc, port=port, scenario_timeout_s=args.scenario_timeout
            )
            metrics["scenario"] = sc.name
            rows.append(_row(defender, attack, seed, metrics))
            port += 1
            merged = _merge_rows(existing, rows)
            _write_outputs(merged, _summarize(merged))

        merged = _merge_rows(existing, rows)
        summary = _summarize(merged)
        _write_outputs(merged, summary)

    print("\nThreat-gap summary:")
    for attack in attack_pool:
        name = attack["name"]
        print(f"\n  [{name}]")
        for defender in DEFENDERS:
            key = f"{name}/{defender['name']}"
            stats = summary.get(key)
            if stats:
                print(
                    f"    {defender['name']}: mean_f1={stats['mean_f1']:.4f} "
                    f"std={stats['std_f1']:.4f} n={stats['n_seeds']}"
                )
        for defender in DEFENDERS:
            dname = defender["name"]
            if dname == "fedavg":
                continue
            gap_key = (
                f"{name}/acfl_minus_fedavg"
                if dname == "acfl_full"
                else f"{name}/{dname}_minus_fedavg"
            )
            gap = summary.get(gap_key)
            if gap:
                print(f"    {dname} vs fedavg: {gap['mean_gap']:+.4f}")


if __name__ == "__main__":
    main()
