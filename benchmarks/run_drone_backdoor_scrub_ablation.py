#!/usr/bin/env python3
"""Backdoor ablation: shortlist-veto only vs veto + post-agg trigger scrub.

Compares two ACFL variants on the journal (3-client) backdoor protocol:
  - acfl_veto_only: geometry shortlist-veto, scrub OFF
  - acfl_veto_scrub: geometry shortlist-veto, scrub ON (production)

Outputs:
  - evaluation_results/acfl_tuning/drone_backdoor_scrub_ablation.csv
  - evaluation_results/acfl_tuning/drone_backdoor_scrub_ablation.json
  - evaluation_results/acfl_tuning/drone_backdoor_scrub_ablation_summary.json
"""
from __future__ import annotations

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
BASENAME_ROOT = "drone_backdoor_scrub_ablation"


def _basename(num_clients: int) -> str:
    """Journal (3c) keeps the original basename; scaling runs get an _Nc suffix."""
    if int(num_clients) == 3:
        return BASENAME_ROOT
    return f"{BASENAME_ROOT}_{int(num_clients)}c"

DEFENDERS = [
    {
        "name": "acfl_veto_only",
        "post_agg_trigger_scrub": False,
    },
    {
        "name": "acfl_veto_scrub",
        "post_agg_trigger_scrub": True,
    },
]

ACFL_KW = dict(
    trust_power=1.25,
    min_trust=0.03,
    coord_thr=2.5,
    norm_thr=2.0,
    norm2_thr=1.2,
)


def _scenario(
    defender: Dict,
    seed: int,
    quick_eval: bool,
    num_clients: int,
    min_clients: int,
) -> Scenario:
    return Scenario(
        name=f"scrub_ablation_{num_clients}c_seed{seed}_{defender['name']}",
        data_dir=SITL_PKT,
        config="config_classification_drone",
        learning_mode="classification",
        malicious_clients="client_2",
        attack_type="backdoor",
        attack_scale=1.0,
        num_clients=num_clients,
        min_clients=min_clients,
        target_rounds=12,
        max_fl_rounds=12,
        probe_file=PROBE_FILE,
        acfl_trust_weight_power=ACFL_KW["trust_power"],
        acfl_min_trust_weight=ACFL_KW["min_trust"],
        acfl_coordinate_deviation_threshold=ACFL_KW["coord_thr"],
        acfl_norm_ratio_threshold=ACFL_KW["norm_thr"],
        acfl_norm_ratio_threshold_two_clients=ACFL_KW["norm2_thr"],
        acfl_post_agg_trigger_scrub=defender["post_agg_trigger_scrub"],
        backdoor_poison_fraction=0.35,
        fl_seed=seed,
        quick_fl_eval=quick_eval,
    )


def _row(defender: Dict, seed: int, metrics: Dict, num_clients: int) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {}))
    bd = metrics.get("backdoor_eval") or {}
    return {
        "seed": seed,
        "num_clients": num_clients,
        "defender": defender["name"],
        "post_agg_trigger_scrub": defender["post_agg_trigger_scrub"],
        "scenario": metrics.get("scenario"),
        "mean_final_f1": fl.get("mean_final_f1"),
        "backdoor_mean_attack_prob": bd.get("backdoor_mean_attack_prob"),
        "trust_auc": metrics.get("acfl_eval", {}).get("trust_score_auc"),
        "final_round": metrics.get("final_round"),
    }


def _summarize(rows: List[Dict]) -> Dict:
    summary: Dict = {}
    for defender in DEFENDERS:
        name = defender["name"]
        subset = [r for r in rows if r["defender"] == name]
        f1s = [r["mean_final_f1"] for r in subset if r.get("mean_final_f1") is not None]
        probs = [
            r["backdoor_mean_attack_prob"]
            for r in subset
            if r.get("backdoor_mean_attack_prob") is not None
        ]
        if f1s:
            summary[name] = {
                "mean_f1": statistics.mean(f1s),
                "std_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                "n_seeds": len(f1s),
                "f1_values": f1s,
                "mean_backdoor_attack_prob": (
                    statistics.mean(probs) if probs else None
                ),
                "std_backdoor_attack_prob": (
                    statistics.pstdev(probs) if len(probs) > 1 else 0.0
                ),
                "trigger_values": probs,
            }
    only = summary.get("acfl_veto_only")
    scrub = summary.get("acfl_veto_scrub")
    if (
        only
        and scrub
        and only.get("mean_f1") is not None
        and scrub.get("mean_f1") is not None
    ):
        summary["scrub_minus_veto_only"] = {
            "f1_gap": scrub["mean_f1"] - only["mean_f1"],
            "trigger_gap": (
                (
                    scrub["mean_backdoor_attack_prob"]
                    - only["mean_backdoor_attack_prob"]
                )
                if scrub.get("mean_backdoor_attack_prob") is not None
                and only.get("mean_backdoor_attack_prob") is not None
                else None
            ),
        }
    return summary


def _write(rows: List[Dict], summary: Dict, basename: str) -> None:
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TUNING_ROOT / f"{basename}.csv"
    json_path = TUNING_ROOT / f"{basename}.json"
    summary_path = TUNING_ROOT / f"{basename}_summary.json"
    fields = [
        "seed",
        "num_clients",
        "defender",
        "post_agg_trigger_scrub",
        "scenario",
        "mean_final_f1",
        "backdoor_mean_attack_prob",
        "trust_auc",
        "final_round",
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


def _load_existing_rows(basename: str) -> List[Dict]:
    json_path = TUNING_ROOT / f"{basename}.json"
    if not json_path.exists():
        return []
    return json.loads(json_path.read_text())


def _merge_rows(existing: List[Dict], new_rows: List[Dict]) -> List[Dict]:
    replaced = {(r["seed"], r["defender"], r.get("num_clients")) for r in new_rows}
    kept = [
        r
        for r in existing
        if (r.get("seed"), r.get("defender"), r.get("num_clients")) not in replaced
    ]
    return kept + new_rows


def _pending(
    defender_pool: List[Dict],
    seeds: List[int],
    existing: List[Dict],
    num_clients: int,
):
    done = {
        (r.get("seed"), r.get("defender"), r.get("num_clients")) for r in existing
    }
    for seed in seeds:
        for defender in defender_pool:
            if (seed, defender["name"], num_clients) not in done:
                yield seed, defender


def _remove_cells(
    existing: List[Dict],
    defender_pool: List[Dict],
    seeds: List[int],
    num_clients: int,
) -> List[Dict]:
    targets = {
        (seed, d["name"], num_clients) for seed in seeds for d in defender_pool
    }
    return [
        r
        for r in existing
        if (r.get("seed"), r.get("defender"), r.get("num_clients")) not in targets
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backdoor scrub ablation: veto-only vs veto+scrub"
    )
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--seeds", default=seeds_csv())
    parser.add_argument(
        "--defenders",
        default=",".join(d["name"] for d in DEFENDERS),
        help="Comma-separated defenders",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=3,
        help="FL clients (3=journal, 5=scaling)",
    )
    parser.add_argument("--min-clients", type=int, default=None)
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--quick-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--scenario-timeout", type=int, default=1800)
    args = parser.parse_args()

    num_clients = int(args.num_clients)
    min_clients = (
        int(args.min_clients)
        if args.min_clients is not None
        else (4 if num_clients >= 5 else 2)
    )
    basename = _basename(num_clients)

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    seeds = [int(s) for s in args.seeds.split(",")]
    defender_pool = [d for d in DEFENDERS if d["name"] in args.defenders.split(",")]
    existing = (
        _load_existing_rows(basename)
        if (args.merge_existing or args.force_rerun)
        else []
    )
    if args.force_rerun:
        existing = _remove_cells(existing, defender_pool, seeds, num_clients)
        print(
            f"Force rerun: cleared {len(defender_pool) * len(seeds)} "
            f"cell(s) for num_clients={num_clients}."
        )
    rows: List[Dict] = []
    port = args.port

    pending = list(_pending(defender_pool, seeds, existing, num_clients))
    if args.merge_existing and not pending:
        print("All requested cells already complete.")
        merged = existing
        summary = _summarize(merged)
        _write(merged, summary, basename)
        return

    if args.merge_existing:
        print(f"Resuming {len(pending)} missing cell(s)...")

    for seed, defender in pending:
        sc = _scenario(
            defender,
            seed,
            quick_eval=args.quick_eval,
            num_clients=num_clients,
            min_clients=min_clients,
        )
        print(
            f"\n>>> Running {sc.name} "
            f"(scrub={defender['post_agg_trigger_scrub']}, "
            f"timeout={args.scenario_timeout}s)"
        )
        metrics = run_scenario(sc, port=port, scenario_timeout_s=args.scenario_timeout)
        metrics["scenario"] = sc.name
        rows.append(_row(defender, seed, metrics, num_clients))
        port += 1
        merged = _merge_rows(existing, rows)
        _write(merged, _summarize(merged), basename)

    merged = _merge_rows(existing, rows)
    summary = _summarize(merged)
    _write(merged, summary, basename)

    print(f"\nBackdoor scrub ablation (n={num_clients}):")
    for name, stats in summary.items():
        if name == "scrub_minus_veto_only":
            g = stats
            tg = g.get("trigger_gap")
            tg_s = f"{tg:+.4f}" if tg is not None else "n/a"
            print(
                f"  scrub vs veto_only: f1_gap={g['f1_gap']:+.4f} "
                f"trigger_gap={tg_s} (lower trigger is better)"
            )
            continue
        if not isinstance(stats, dict) or "mean_f1" not in stats:
            continue
        prob = stats.get("mean_backdoor_attack_prob")
        prob_s = f" trigger_P={prob:.3f}" if prob is not None else ""
        print(
            f"  {name}: f1={stats['mean_f1']:.4f}±{stats['std_f1']:.4f}"
            f"{prob_s} n={stats.get('n_seeds')}"
        )


if __name__ == "__main__":
    main()
