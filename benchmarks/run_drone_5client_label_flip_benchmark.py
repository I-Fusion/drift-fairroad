#!/usr/bin/env python3
"""5-client label-flip benchmark: FedAvg + robust baselines vs ACFL.

Compares defenders when 5 FL clients participate (1 malicious, 20%):
  - Defenders: fedavg, coordinate_median, trimmed_mean, krum, acfl_full
  - Attack: label_flip (50% local label corruption)
  - Seeds: 42, 43, 44 (configurable)

Outputs:
  - evaluation_results/acfl_tuning/drone_5client_label_flip_benchmark.csv
  - evaluation_results/acfl_tuning/drone_5client_label_flip_benchmark.json
  - evaluation_results/acfl_tuning/drone_5client_label_flip_benchmark_summary.json
"""
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
NUM_CLIENTS = 5
MIN_CLIENTS = 3
BASENAME = "drone_5client_label_flip_benchmark"

ATTACK = {
    "name": "label_flip",
    "attack_type": "label_flip",
    "attack_scale": 1.0,
    "label_flip_fraction": 0.5,
}

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


def _scenario(defender: Dict, seed: int, quick_eval: bool) -> Scenario:
    needs_probes = defender["name"] == "acfl_full"
    return Scenario(
        name=f"5client_label_flip_seed{seed}_{defender['name']}",
        data_dir=SITL_PKT,
        config="config_classification_drone",
        learning_mode="classification",
        malicious_clients="client_2",
        attack_type=ATTACK["attack_type"],
        attack_scale=ATTACK.get("attack_scale", 1.0),
        baseline_aggregation=defender.get("baseline_aggregation"),
        num_clients=NUM_CLIENTS,
        min_clients=MIN_CLIENTS,
        target_rounds=12,
        max_fl_rounds=12,
        probe_file=PROBE_FILE if needs_probes else None,
        acfl_trust_weight_power=defender.get("trust_power"),
        acfl_min_trust_weight=defender.get("min_trust"),
        acfl_coordinate_deviation_threshold=defender.get("coord_thr"),
        acfl_norm_ratio_threshold=defender.get("norm_thr"),
        acfl_norm_ratio_threshold_two_clients=defender.get("norm2_thr"),
        label_flip_fraction=ATTACK.get("label_flip_fraction"),
        fl_seed=seed,
        quick_fl_eval=quick_eval,
    )


def _row(defender: Dict, seed: int, metrics: Dict) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {}))
    return {
        "seed": seed,
        "attack": ATTACK["name"],
        "attack_type": ATTACK["attack_type"],
        "label_flip_fraction": ATTACK.get("label_flip_fraction"),
        "num_clients": NUM_CLIENTS,
        "min_clients": MIN_CLIENTS,
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


def _summarize(rows: List[Dict]) -> Dict:
    summary: Dict = {}
    for defender in DEFENDERS:
        name = defender["name"]
        subset = [r for r in rows if r["defender"] == name]
        f1s = [r["mean_final_f1"] for r in subset if r.get("mean_final_f1") is not None]
        trust_aucs = [r["trust_auc"] for r in subset if r.get("trust_auc") is not None]
        key = f"label_flip/{name}"
        if f1s:
            entry = {
                "mean_f1": statistics.mean(f1s),
                "std_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                "f1_values": f1s,
                "n_seeds": len(f1s),
            }
            if trust_aucs:
                entry["mean_trust_auc"] = statistics.mean(trust_aucs)
            summary[key] = entry

    fedavg_rows = sorted(
        [r for r in rows if r["defender"] == "fedavg"],
        key=lambda r: r["seed"],
    )
    fedavg_f1s = [
        r["mean_final_f1"] for r in fedavg_rows
        if r.get("mean_final_f1") is not None
    ]
    for defender in DEFENDERS:
        name = defender["name"]
        if name == "fedavg":
            continue
        key = f"label_flip/{name}"
        if key not in summary or not fedavg_f1s:
            continue
        def_rows = sorted(
            [r for r in rows if r["defender"] == name],
            key=lambda r: r["seed"],
        )
        def_f1s = [
            r["mean_final_f1"] for r in def_rows
            if r.get("mean_final_f1") is not None
        ]
        n = min(len(def_f1s), len(fedavg_f1s))
        if n == 0:
            continue
        gaps = [def_f1s[i] - fedavg_f1s[i] for i in range(n)]
        if gaps:
            summary[f"label_flip/{name}_minus_fedavg"] = {
                "mean_gap": statistics.mean(gaps),
                "std_gap": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
                "gaps": gaps,
            }

    summary["meta"] = {
        "num_clients": NUM_CLIENTS,
        "min_clients": MIN_CLIENTS,
        "malicious_clients": "client_2",
        "malicious_fraction": 1.0 / NUM_CLIENTS,
        "label_flip_fraction": ATTACK["label_flip_fraction"],
    }
    return summary


def _write_outputs(rows: List[Dict], summary: Dict):
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed", "attack", "attack_type", "label_flip_fraction",
        "num_clients", "min_clients", "defender", "defender_family",
        "baseline_aggregation", "scenario", "mean_final_f1",
        "mean_final_accuracy", "trust_auc", "final_round", "num_checkpoints",
    ]
    csv_path = TUNING_ROOT / f"{BASENAME}.csv"
    json_path = TUNING_ROOT / f"{BASENAME}.json"
    summary_path = TUNING_ROOT / f"{BASENAME}_summary.json"

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
    json_path = TUNING_ROOT / f"{BASENAME}.json"
    if not json_path.exists():
        return []
    return json.loads(json_path.read_text())


def _merge_rows(existing: List[Dict], new_rows: List[Dict]) -> List[Dict]:
    replaced = {(r["seed"], r["defender"]) for r in new_rows}
    kept = [
        r for r in existing
        if (r.get("seed"), r.get("defender")) not in replaced
    ]
    return kept + new_rows


def _pending(defender_pool: List[Dict], seeds: List[int], existing: List[Dict]):
    done = {(r.get("seed"), r.get("defender")) for r in existing}
    for seed in seeds:
        for defender in defender_pool:
            if (seed, defender["name"]) not in done:
                yield seed, defender


def _remove_cells(
    existing: List[Dict],
    defender_pool: List[Dict],
    seeds: List[int],
) -> List[Dict]:
    targets = {(seed, d["name"]) for seed in seeds for d in defender_pool}
    return [
        r
        for r in existing
        if (r.get("seed"), r.get("defender")) not in targets
    ]


def main():
    parser = argparse.ArgumentParser(
        description="5-client label-flip benchmark: baselines + ACFL"
    )
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument("--seeds", default=seeds_csv())
    parser.add_argument(
        "--defenders",
        default=",".join(d["name"] for d in DEFENDERS),
        help="Comma-separated defenders",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Keep completed rows and only fill missing (seed, defender) cells",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Recompute requested (seed, defender) cells even if present",
    )
    parser.add_argument(
        "--quick-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip FL plots and per-client eval (default: on)",
    )
    parser.add_argument(
        "--scenario-timeout",
        type=int,
        default=2400,
        help="Per-scenario wall-clock limit in seconds",
    )
    args = parser.parse_args()

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    seeds = [int(s) for s in args.seeds.split(",")]
    defender_pool = [d for d in DEFENDERS if d["name"] in args.defenders.split(",")]
    total_cells = len(seeds) * len(defender_pool)

    existing = _load_existing_rows() if (args.merge_existing or args.force_rerun) else []
    if args.force_rerun:
        existing = _remove_cells(existing, defender_pool, seeds)
        print(f"Force rerun: cleared {total_cells} requested cell(s).")

    rows: List[Dict] = []
    port = args.port
    pending = list(_pending(defender_pool, seeds, existing))
    summary = _summarize(existing)

    if args.merge_existing and not pending:
        print("All requested cells already complete.")
        _write_outputs(existing, summary)
    else:
        if args.merge_existing:
            print(f"Resuming {len(pending)} missing cell(s) of {total_cells}...")

        for seed, defender in pending:
            sc = _scenario(defender, seed, quick_eval=args.quick_eval)
            print(
                f"\n>>> Running {sc.name} "
                f"(quick_eval={args.quick_eval}, timeout={args.scenario_timeout}s)"
            )
            metrics = run_scenario(
                sc, port=port, scenario_timeout_s=args.scenario_timeout
            )
            metrics["scenario"] = sc.name
            rows.append(_row(defender, seed, metrics))
            port += 1
            merged = _merge_rows(existing, rows)
            summary = _summarize(merged)
            _write_outputs(merged, summary)

        merged = _merge_rows(existing, rows)
        summary = _summarize(merged)
        _write_outputs(merged, summary)

    print("\n5-client label-flip benchmark summary:")
    print(
        f"  clients={NUM_CLIENTS} min_clients={MIN_CLIENTS} "
        f"malicious=client_2 (20%) flip_fraction=50%"
    )
    for defender in defender_pool:
        key = f"label_flip/{defender['name']}"
        stats = summary.get(key)
        if not stats:
            continue
        gap_key = f"label_flip/{defender['name']}_minus_fedavg"
        gap = summary.get(gap_key, {})
        gap_str = ""
        if gap:
            gap_str = f"  gap_vs_fedavg={gap['mean_gap']:+.4f}"
        auc = stats.get("mean_trust_auc")
        auc_str = f"  trust_auc={auc:.3f}" if auc is not None else ""
        print(
            f"  {defender['name']}: "
            f"mean_f1={stats['mean_f1']:.4f} "
            f"std={stats['std_f1']:.4f} "
            f"n={stats['n_seeds']}"
            f"{auc_str}"
            f"{gap_str}"
        )


if __name__ == "__main__":
    main()
