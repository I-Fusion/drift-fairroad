#!/usr/bin/env python3
"""Compare ACFL backdoor: legacy min-blend vs probe-aware soft cap (3 seeds)."""
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

DEFENDERS = [
    {
        "name": "acfl_legacy_blend",
        "acfl_backdoor_probe_soft_cap": False,
    },
    {
        "name": "acfl_probe_cap",
        "acfl_backdoor_probe_soft_cap": True,
    },
]

ACFL_KW = dict(
    trust_power=1.25,
    min_trust=0.03,
    coord_thr=2.5,
    norm_thr=2.0,
    norm2_thr=1.2,
)


def _scenario(defender: Dict, seed: int, quick_eval: bool) -> Scenario:
    return Scenario(
        name=f"backdoor_probe_cap_seed{seed}_{defender['name']}",
        data_dir=SITL_PKT,
        config="config_classification_drone",
        learning_mode="classification",
        malicious_clients="client_2",
        attack_type="backdoor",
        attack_scale=1.0,
        num_clients=3,
        min_clients=2,
        target_rounds=12,
        max_fl_rounds=12,
        probe_file=PROBE_FILE,
        acfl_trust_weight_power=ACFL_KW["trust_power"],
        acfl_min_trust_weight=ACFL_KW["min_trust"],
        acfl_coordinate_deviation_threshold=ACFL_KW["coord_thr"],
        acfl_norm_ratio_threshold=ACFL_KW["norm_thr"],
        acfl_norm_ratio_threshold_two_clients=ACFL_KW["norm2_thr"],
        acfl_backdoor_probe_soft_cap=defender["acfl_backdoor_probe_soft_cap"],
        backdoor_poison_fraction=0.35,
        fl_seed=seed,
        quick_fl_eval=quick_eval,
    )


def _row(defender: Dict, seed: int, metrics: Dict) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {}))
    bd = metrics.get("backdoor_eval") or {}
    return {
        "seed": seed,
        "defender": defender["name"],
        "probe_soft_cap": defender["acfl_backdoor_probe_soft_cap"],
        "scenario": metrics.get("scenario"),
        "mean_final_f1": fl.get("mean_final_f1"),
        "backdoor_mean_attack_prob": bd.get("backdoor_mean_attack_prob"),
        "trust_auc": metrics.get("acfl_eval", {}).get("trust_score_auc"),
        "final_round": metrics.get("final_round"),
    }


def _summarize(rows: List[Dict]) -> Dict:
    summary = {}
    for defender in DEFENDERS:
        name = defender["name"]
        subset = [r for r in rows if r["defender"] == name]
        f1s = [r["mean_final_f1"] for r in subset if r.get("mean_final_f1") is not None]
        probs = [
            r["backdoor_mean_attack_prob"] for r in subset
            if r.get("backdoor_mean_attack_prob") is not None
        ]
        if f1s:
            summary[name] = {
                "mean_f1": statistics.mean(f1s),
                "std_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                "f1_values": f1s,
                "mean_backdoor_attack_prob": (
                    statistics.mean(probs) if probs else None
                ),
                "trigger_values": probs,
            }
    legacy = summary.get("acfl_legacy_blend")
    probe = summary.get("acfl_probe_cap")
    if legacy and probe and legacy.get("mean_f1") is not None and probe.get("mean_f1") is not None:
        summary["probe_cap_minus_legacy"] = {
            "f1_gap": probe["mean_f1"] - legacy["mean_f1"],
            "trigger_gap": (
                (probe["mean_backdoor_attack_prob"] - legacy["mean_backdoor_attack_prob"])
                if probe.get("mean_backdoor_attack_prob") is not None
                and legacy.get("mean_backdoor_attack_prob") is not None
                else None
            ),
        }
    return summary


def _write(rows: List[Dict], summary: Dict):
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TUNING_ROOT / "drone_backdoor_probe_cap.csv"
    json_path = TUNING_ROOT / "drone_backdoor_probe_cap.json"
    summary_path = TUNING_ROOT / "drone_backdoor_probe_cap_summary.json"
    fields = [
        "seed", "defender", "probe_soft_cap", "scenario",
        "mean_final_f1", "backdoor_mean_attack_prob", "trust_auc", "final_round",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})
    json_path.write_text(json.dumps(rows, indent=2, default=str))
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


def _load_existing_rows() -> List[Dict]:
    json_path = TUNING_ROOT / "drone_backdoor_probe_cap.json"
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
    """Drop selected cells so they can be recomputed with --merge-existing."""
    targets = {(seed, d["name"]) for seed in seeds for d in defender_pool}
    return [
        r
        for r in existing
        if (r.get("seed"), r.get("defender")) not in targets
    ]


def main():
    parser = argparse.ArgumentParser(description="Backdoor probe-cap A/B (3 seeds)")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--seeds", default=seeds_csv())
    parser.add_argument(
        "--defenders",
        default=",".join(d["name"] for d in DEFENDERS),
        help="Comma-separated defenders to run",
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
    parser.add_argument("--scenario-timeout", type=int, default=1800)
    args = parser.parse_args()

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    seeds = [int(s) for s in args.seeds.split(",")]
    defender_pool = [d for d in DEFENDERS if d["name"] in args.defenders.split(",")]
    existing = _load_existing_rows() if (args.merge_existing or args.force_rerun) else []
    if args.force_rerun:
        existing = _remove_cells(existing, defender_pool, seeds)
        print(
            f"Force rerun: cleared {len(defender_pool) * len(seeds)} "
            f"cell(s) from existing results."
        )
    rows: List[Dict] = []
    port = args.port

    pending = list(_pending(defender_pool, seeds, existing))
    if args.merge_existing and not pending:
        print("All requested cells already complete.")
        merged = existing
        summary = _summarize(merged)
        _write(merged, summary)
        return

    if args.merge_existing:
        print(f"Resuming {len(pending)} missing cell(s)...")

    for seed, defender in pending:
        sc = _scenario(defender, seed, quick_eval=args.quick_eval)
        print(
            f"\n>>> Running {sc.name} "
            f"(quick_eval={args.quick_eval}, timeout={args.scenario_timeout}s)"
        )
        metrics = run_scenario(sc, port=port, scenario_timeout_s=args.scenario_timeout)
        metrics["scenario"] = sc.name
        rows.append(_row(defender, seed, metrics))
        port += 1
        merged = _merge_rows(existing, rows)
        _write(merged, _summarize(merged))

    merged = _merge_rows(existing, rows)
    summary = _summarize(merged)
    _write(merged, summary)
    print("\nBackdoor probe-cap comparison:")
    for name, stats in summary.items():
        if name == "probe_cap_minus_legacy":
            g = stats
            print(
                f"  probe_cap vs legacy: f1_gap={g['f1_gap']:+.4f} "
                f"trigger_gap={g['trigger_gap']:+.4f}"
            )
            continue
        if not isinstance(stats, dict) or "mean_f1" not in stats:
            continue
        prob = stats.get("mean_backdoor_attack_prob")
        prob_s = f" trigger_P={prob:.3f}" if prob is not None else ""
        print(
            f"  {name}: f1={stats['mean_f1']:.4f} "
            f"std={stats['std_f1']:.4f}{prob_s}"
        )


if __name__ == "__main__":
    main()
