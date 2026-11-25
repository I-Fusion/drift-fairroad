#!/usr/bin/env python3
"""Drone-profile layer ablation: FedAvg vs full ACFL vs L1-only vs L2-only."""
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
from run_drone_attack_suite import ATTACKS as BASE_ATTACKS, _guard_stats


TUNING_ROOT = Path("evaluation_results/acfl_tuning")
PROBE_FILE = str(Path(CHECKPOINT_DIR) / "acfl_probes_sitl.pt")
FEDAVG_F1 = 0.674058437408308

# Layer ablation covers the full threat matrix (suite + threat-gap attacks).
ATTACKS = list(BASE_ATTACKS) + [
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
        "fedavg": True,
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        "name": "acfl_full",
        "fedavg": False,
        "trust_power": 1.25,
        "min_trust": 0.03,
        "coord_thr": 2.5,
        "norm_thr": 2.0,
        "norm2_thr": 1.2,
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        "name": "acfl_no_layer1",
        "fedavg": False,
        "trust_power": 1.25,
        "min_trust": 0.03,
        "coord_thr": 2.5,
        "norm_thr": 2.0,
        "norm2_thr": 1.2,
        "no_layer1": True,
        "no_layer2": False,
    },
    {
        "name": "acfl_no_layer2",
        "fedavg": False,
        "trust_power": 1.25,
        "min_trust": 0.03,
        "coord_thr": 2.5,
        "norm_thr": 2.0,
        "norm2_thr": 1.2,
        "no_layer1": False,
        "no_layer2": True,
    },
]


def _scenario(defender: Dict, attack: Dict, seed: Optional[int] = None) -> Scenario:
    tag = f"ablation_{defender['name']}_{attack['name']}"
    if seed is not None:
        tag = f"{tag}_seed{seed}"
    return Scenario(
        name=tag,
        data_dir=SITL_PKT,
        config="config_classification_drone",
        learning_mode="classification",
        malicious_clients="client_2",
        attack_type=attack["attack_type"],
        attack_scale=attack["attack_scale"],
        fedavg=defender["fedavg"],
        no_layer1=defender.get("no_layer1", False),
        no_layer2=defender.get("no_layer2", False),
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
        label_flip_fraction=attack.get("label_flip_fraction"),
        fl_seed=seed,
    )


def _row(
    defender: Dict,
    attack: Dict,
    metrics: Dict,
    seed: Optional[int] = None,
) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {}))
    mean_f1 = fl.get("mean_final_f1")
    backdoor = metrics.get("backdoor_eval") or {}
    return {
        "seed": seed,
        "defender": defender["name"],
        "attack": attack["name"],
        "attack_type": attack["attack_type"],
        "fedavg": defender["fedavg"],
        "no_layer1": defender.get("no_layer1", False),
        "no_layer2": defender.get("no_layer2", False),
        "scenario": metrics.get("scenario"),
        "mean_final_f1": mean_f1,
        "mean_final_accuracy": fl.get("mean_final_accuracy"),
        "f1_vs_fedavg_baseline": (
            mean_f1 - FEDAVG_F1 if mean_f1 is not None else None
        ),
        "trust_auc": metrics.get("acfl_eval", {}).get("trust_score_auc"),
        "backdoor_success_rate": backdoor.get("backdoor_success_rate"),
        "backdoor_trigger_accuracy": backdoor.get("backdoor_trigger_accuracy"),
        "backdoor_mean_attack_prob": backdoor.get("backdoor_mean_attack_prob"),
        "final_round": metrics.get("final_round"),
        "num_checkpoints": fl.get("num_checkpoints"),
        **{k: v for k, v in _guard_stats(metrics.get("scenario", "")).items()},
    }


def _write_summary(rows: List[Dict], summary_path: Path) -> Dict:
    summary = {}
    attacks = sorted({r["attack"] for r in rows if r.get("attack")})
    defenders = sorted({r["defender"] for r in rows if r.get("defender")})
    for attack in attacks:
        for defender in defenders:
            subset = [
                r for r in rows
                if r.get("attack") == attack and r.get("defender") == defender
            ]
            f1s = [r["mean_final_f1"] for r in subset if r.get("mean_final_f1") is not None]
            trust_aucs = [
                r["trust_auc"] for r in subset
                if r.get("trust_auc") is not None
            ]
            key = f"{attack}/{defender}"
            if f1s:
                summary[key] = {
                    "mean_f1": statistics.mean(f1s),
                    "std_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                    "f1_values": f1s,
                    "n_seeds": len(f1s),
                    "mean_trust_auc": (
                        statistics.mean(trust_aucs) if trust_aucs else None
                    ),
                }
        fed = [r for r in rows if r.get("attack") == attack and r["defender"] == "fedavg"]
        acfl = [r for r in rows if r.get("attack") == attack and r["defender"] == "acfl_full"]
        if fed and acfl and len(fed) == len(acfl):
            gaps = [
                a["mean_final_f1"] - f["mean_final_f1"]
                for a, f in zip(
                    sorted(acfl, key=lambda r: (r.get("seed"), r.get("scenario"))),
                    sorted(fed, key=lambda r: (r.get("seed"), r.get("scenario"))),
                )
            ]
            summary[f"{attack}/acfl_full_minus_fedavg"] = {
                "mean_gap": statistics.mean(gaps),
                "std_gap": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
                "gaps": gaps,
            }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _load_existing_rows(seeded: bool) -> List[Dict]:
    stem = "drone_ablation_seeded" if seeded else "drone_ablation"
    json_path = TUNING_ROOT / f"{stem}.json"
    if not json_path.exists():
        return []
    return json.loads(json_path.read_text())


def _merge_rows(existing: List[Dict], new_rows: List[Dict]) -> List[Dict]:
    replaced = {
        (r.get("seed"), r.get("attack"), r.get("defender")) for r in new_rows
    }
    kept = [
        r for r in existing
        if (r.get("seed"), r.get("attack"), r.get("defender")) not in replaced
    ]
    return kept + new_rows


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


def _write_outputs(rows: List[Dict], seeded: bool = False):
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    stem = "drone_ablation_seeded" if seeded else "drone_ablation"
    csv_path = TUNING_ROOT / f"{stem}.csv"
    json_path = TUNING_ROOT / f"{stem}.json"
    fieldnames = [
        "seed", "defender", "attack", "attack_type", "fedavg", "no_layer1", "no_layer2",
        "scenario", "mean_final_f1", "mean_final_accuracy", "f1_vs_fedavg_baseline",
        "trust_auc", "backdoor_success_rate", "backdoor_trigger_accuracy",
        "backdoor_mean_attack_prob", "final_round", "num_checkpoints",
        "aggregation_skipped_rounds", "pairwise_penalty_rounds",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    json_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    if seeded:
        summary = _write_summary(rows, TUNING_ROOT / f"{stem}_summary.json")
        print(f"Wrote {TUNING_ROOT / f'{stem}_summary.json'}")
        return summary
    return None


def main():
    parser = argparse.ArgumentParser(description="Run drone layer ablation suite")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=5880)
    parser.add_argument(
        "--defenders",
        default=",".join(d["name"] for d in DEFENDERS),
    )
    parser.add_argument(
        "--attacks",
        default=",".join(a["name"] for a in ATTACKS),
    )
    parser.add_argument("--scenario-timeout", type=int, default=1200)
    parser.add_argument(
        "--seeds",
        default=seeds_csv(),
        help="Comma-separated FL seeds (default: 42--51). Empty = unseeded ablation.",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Keep ablation rows not being re-run",
    )
    args = parser.parse_args()

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    defender_pool = [d for d in DEFENDERS if d["name"] in args.defenders.split(",")]
    attack_pool = [a for a in ATTACKS if a["name"] in args.attacks.split(",")]
    seeds: List[Optional[int]]
    if args.seeds.strip():
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [None]

    rows: List[Dict] = []
    port = args.port
    seeded = seeds != [None]
    existing = _load_existing_rows(seeded) if args.merge_existing else []
    int_seeds = [s for s in seeds if s is not None]
    pending = (
        list(_pending(attack_pool, defender_pool, int_seeds, existing))
        if seeded
        else [(a, None, d) for a in attack_pool for d in defender_pool]
    )
    total_cells = len(pending) if not seeded else len(int_seeds) * len(attack_pool) * len(defender_pool)

    if args.merge_existing and seeded and not pending:
        print("All requested cells already complete.")
        merged = existing
        summary = _write_outputs(merged, seeded=seeded)
    else:
        if args.merge_existing and seeded:
            print(f"Resuming {len(pending)} missing cell(s) of {total_cells}...")
        for attack, seed, defender in pending:
            sc = _scenario(defender, attack, seed=seed)
            metrics = run_scenario(
                sc, port=port, scenario_timeout_s=args.scenario_timeout
            )
            metrics["scenario"] = sc.name
            rows.append(_row(defender, attack, metrics, seed=seed))
            port += 1
            merged = _merge_rows(existing, rows) if args.merge_existing else rows
            _write_outputs(merged, seeded=seeded)

        merged = _merge_rows(existing, rows) if args.merge_existing else rows
        summary = _write_outputs(merged, seeded=seeded)

    print("\nAblation summary (mean final F1):")
    for attack in attack_pool:
        name = attack["name"]
        if seeded:
            print(f"\n  [{name}] aggregated over seeds:")
            if summary:
                for defender in defender_pool:
                    key = f"{name}/{defender['name']}"
                    stats = summary.get(key)
                    if stats:
                        print(
                            f"    {defender['name']}: "
                            f"mean={stats['mean_f1']:.4f} "
                            f"std={stats['std_f1']:.4f} "
                            f"values={stats['f1_values']}"
                        )
                gap_key = f"{name}/acfl_full_minus_fedavg"
                if gap_key in summary:
                    g = summary[gap_key]
                    print(
                        f"    acfl_full vs fedavg: "
                        f"mean_gap={g['mean_gap']:+.4f} "
                        f"std={g['std_gap']:.4f}"
                    )
        subset = [r for r in merged if r.get("attack") == name]
        if not subset:
            continue
        print(f"\n  [{name}] per run:")
        subset.sort(
            key=lambda r: (r.get("seed") is None, r.get("seed"), r.get("mean_final_f1") or -1),
            reverse=True,
        )
        for row in subset:
            extra = ""
            if name == "backdoor" and row.get("backdoor_success_rate") is not None:
                extra = (
                    f" bd_succ={row['backdoor_success_rate']:.3f}"
                    f" trig_acc={row.get('backdoor_trigger_accuracy')}"
                )
            seed_tag = f" seed={row['seed']}" if row.get("seed") is not None else ""
            print(
                f"    {row['defender']}{seed_tag}: f1={row.get('mean_final_f1')} "
                f"trust_auc={row.get('trust_auc')}{extra}"
            )


if __name__ == "__main__":
    main()
