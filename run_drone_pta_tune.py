#!/usr/bin/env python3
"""Conservative PTA selector grid (skip penalty, tau, λ_S, λ_R).

Locked 10-seed PTA conservative used tau=0.05, λ_skip=0, λ_S=0.1, λ_R=1.0.
This grid keeps extras on and only changes the selector.

Usage:
  ./acfl_env/bin/python run_drone_pta_tune.py --skip-setup --port 9700
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

from acfl.experiment_seeds import LEGACY_THREE_SEEDS, seeds_csv
from run_comprehensive_validation import (
    CHECKPOINT_DIR,
    SITL_PKT,
    Scenario,
    _fl_eval_summary,
    generate_probes,
    run_scenario,
    sync_agent_missions,
)
from run_drone_attack_suite import _guard_stats


TUNING_ROOT = Path("evaluation_results/acfl_tuning")
PROBE_FILE = str(Path(CHECKPOINT_DIR) / "acfl_probes_sitl.pt")
STEM = "drone_pta_tune"

ATTACKS = [
    {"name": "gradient_scaling", "attack_type": "gradient_scaling", "attack_scale": 10.0},
    {
        "name": "byzantine",
        "attack_type": "byzantine",
        "attack_scale": 1.0,
        "attack_noise": 1.0,
    },
    {
        "name": "model_replacement",
        "attack_type": "model_replacement",
        "attack_scale": 10.0,
    },
]

CONFIGS = [
    {
        "name": "pta_con_base",
        "label": "conservative locked (tau=0.05 skip=0 S=0.1 R=1)",
        "tau": 0.05,
        "lambda_skip": 0.0,
        "lambda_s": 0.1,
        "lambda_r": 1.0,
        "default_branch": "ungated",
    },
    {
        "name": "pta_con_skip",
        "label": "skip penalty 0.25",
        "tau": 0.05,
        "lambda_skip": 0.25,
        "lambda_s": 0.1,
        "lambda_r": 1.0,
        "default_branch": "ungated",
    },
    {
        "name": "pta_con_tau",
        "label": "tau=0.20 + skip 0.25",
        "tau": 0.20,
        "lambda_skip": 0.25,
        "lambda_s": 0.1,
        "lambda_r": 1.0,
        "default_branch": "ungated",
    },
    {
        "name": "pta_con_nos",
        "label": "tau=0.20 skip=0.25 λ_S=0",
        "tau": 0.20,
        "lambda_skip": 0.25,
        "lambda_s": 0.0,
        "lambda_r": 1.0,
        "default_branch": "ungated",
    },
    {
        "name": "pta_con_nosr",
        "label": "tau=0.20 skip=0.25 λ_S=0 λ_R=0.3",
        "tau": 0.20,
        "lambda_skip": 0.25,
        "lambda_s": 0.0,
        "lambda_r": 0.3,
        "default_branch": "ungated",
    },
    {
        "name": "pta_con_stick",
        "label": "tau=0.20 skip=0.50 λ_S=0 λ_R=0.3",
        "tau": 0.20,
        "lambda_skip": 0.50,
        "lambda_s": 0.0,
        "lambda_r": 0.3,
        "default_branch": "ungated",
    },
    {
        "name": "pta_con_gated",
        "label": "default gated, tau=0.20 skip=0.25 λ_S=0",
        "tau": 0.20,
        "lambda_skip": 0.25,
        "lambda_s": 0.0,
        "lambda_r": 1.0,
        "default_branch": "gated",
    },
]


def _scenario(cfg: Dict, attack: Dict, seed: int) -> Scenario:
    tag = f"ptatune_{cfg['name']}_{attack['name']}_seed{seed}"
    return Scenario(
        name=tag,
        data_dir=SITL_PKT,
        config="config_classification_drone",
        learning_mode="classification",
        malicious_clients="client_2",
        attack_type=attack["attack_type"],
        attack_scale=float(attack.get("attack_scale", 1.0)),
        attack_noise=attack.get("attack_noise"),
        num_clients=3,
        min_clients=2,
        target_rounds=12,
        max_fl_rounds=12,
        probe_file=PROBE_FILE,
        acfl_trust_weight_power=1.25,
        acfl_min_trust_weight=0.03,
        acfl_coordinate_deviation_threshold=2.5,
        acfl_norm_ratio_threshold=2.0,
        acfl_norm_ratio_threshold_two_clients=1.2,
        acfl_layer1_layer2_gating=True,
        acfl_adaptive_l1l2_gating=False,
        acfl_parallel_trust_arbitration=True,
        acfl_pta_method="conservative",
        acfl_pta_default_branch=str(cfg["default_branch"]),
        acfl_pta_tau=float(cfg["tau"]),
        acfl_pta_lambda_r=float(cfg["lambda_r"]),
        acfl_pta_lambda_s=float(cfg["lambda_s"]),
        acfl_pta_lambda_skip=float(cfg["lambda_skip"]),
        acfl_backdoor_geometry_hard_reject=True,
        acfl_label_flip_geometry_hard_reject=True,
        acfl_backdoor_probe_hard_reject=True,
        acfl_post_agg_trigger_scrub=True,
        acfl_update_norm_hard_rejection_for_model_replacement=True,
        acfl_backdoor_probe_soft_cap=False,
        acfl_backdoor_trigger_hard_gate=False,
        backdoor_poison_fraction=attack.get("backdoor_poison_fraction"),
        label_flip_fraction=attack.get("label_flip_fraction"),
        fl_seed=seed,
        quick_fl_eval=True,
    )


def _row(cfg: Dict, attack: Dict, metrics: Dict, seed: int) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {}))
    return {
        "seed": seed,
        "variant": cfg["name"],
        "variant_label": cfg["label"],
        "attack": attack["name"],
        "attack_type": attack["attack_type"],
        "tau": cfg["tau"],
        "lambda_skip": cfg["lambda_skip"],
        "lambda_s": cfg["lambda_s"],
        "lambda_r": cfg["lambda_r"],
        "default_branch": cfg["default_branch"],
        "scenario": metrics.get("scenario"),
        "mean_final_f1": fl.get("mean_final_f1"),
        "mean_final_accuracy": fl.get("mean_final_accuracy"),
        "trust_auc": metrics.get("acfl_eval", {}).get("trust_score_auc"),
        "final_round": metrics.get("final_round"),
        **{k: v for k, v in _guard_stats(metrics.get("scenario", "")).items()},
    }


def _cell_key(row: Dict) -> Tuple:
    return (row.get("seed"), row.get("attack"), row.get("variant"))


def _load_existing() -> List[Dict]:
    path = TUNING_ROOT / f"{STEM}.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def _merge(existing: List[Dict], new_rows: List[Dict]) -> List[Dict]:
    by_key = {_cell_key(r): r for r in existing}
    for row in new_rows:
        by_key[_cell_key(row)] = row
    return [by_key[k] for k in sorted(by_key)]


def _pending(attack_pool, config_pool, seeds, existing):
    done = {_cell_key(r) for r in existing}
    for attack in attack_pool:
        for seed in seeds:
            for cfg in config_pool:
                key = (seed, attack["name"], cfg["name"])
                if key not in done:
                    yield attack, seed, cfg


def _write_summary(rows: List[Dict], summary_path: Path) -> Dict:
    summary: Dict = {}
    for attack in sorted({r["attack"] for r in rows}):
        for variant in sorted({r["variant"] for r in rows}):
            subset = [
                r for r in rows
                if r["attack"] == attack and r["variant"] == variant
            ]
            f1s = [r["mean_final_f1"] for r in subset if r.get("mean_final_f1") is not None]
            collapse = sum(1 for f in f1s if f < 0.25)
            key = f"{attack}/{variant}"
            summary[key] = {
                "mean_f1": statistics.mean(f1s) if f1s else None,
                "std_f1": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
                "n_collapse": collapse,
                "n_seeds": len(subset),
                "f1_values": f1s,
            }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _write_outputs(rows: List[Dict]) -> Dict:
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TUNING_ROOT / f"{STEM}.csv"
    json_path = TUNING_ROOT / f"{STEM}.json"
    summary_path = TUNING_ROOT / f"{STEM}_summary.json"
    fieldnames = [
        "seed", "variant", "variant_label", "attack", "attack_type",
        "tau", "lambda_skip", "lambda_s", "lambda_r", "default_branch",
        "scenario", "mean_final_f1", "mean_final_accuracy", "trust_auc",
        "final_round", "aggregation_skipped_rounds", "pairwise_penalty_rounds",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    json_path.write_text(json.dumps(rows, indent=2, default=str) + "\n")
    summary = _write_summary(rows, summary_path)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")
    return summary


def _print_summary(summary: Dict, attacks: List[Dict], configs: List[Dict]) -> None:
    print("\n=== Conservative PTA selector grid ===")
    for attack in attacks:
        print(f"\n{attack['name']}:")
        for cfg in configs:
            block = summary.get(f"{attack['name']}/{cfg['name']}") or {}
            if block.get("mean_f1") is None:
                continue
            print(
                f"  {cfg['name']:16} F1={block['mean_f1']:.3f}±{block['std_f1']:.3f}  "
                f"collapse={block['n_collapse']}/{block['n_seeds']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=9700)
    parser.add_argument(
        "--configs",
        default=",".join(c["name"] for c in CONFIGS),
    )
    parser.add_argument(
        "--attacks",
        default=",".join(a["name"] for a in ATTACKS),
    )
    parser.add_argument("--seeds", default=seeds_csv(LEGACY_THREE_SEEDS))
    parser.add_argument("--scenario-timeout", type=int, default=1800)
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    config_pool = [c for c in CONFIGS if c["name"] in args.configs.split(",")]
    attack_pool = [a for a in ATTACKS if a["name"] in args.attacks.split(",")]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    existing = _load_existing() if args.merge_existing else []
    pending = list(_pending(attack_pool, config_pool, seeds, existing))
    print(
        f"PTA tune: {len(pending)} pending "
        f"({len(config_pool)} configs × {len(attack_pool)} attacks × "
        f"up to {len(seeds)} seeds; {len(existing)} existing)"
    )

    new_rows: List[Dict] = []
    failures: List[str] = []
    port = args.port
    for i, (attack, seed, cfg) in enumerate(pending, 1):
        tag = f"{cfg['name']}/{attack['name']}/seed{seed}"
        sc = _scenario(cfg, attack, seed)
        print(
            f"\n[{i}/{len(pending)}] {cfg['name']} / {attack['name']} / "
            f"seed{seed}  port={port}"
        )
        try:
            metrics = run_scenario(
                sc,
                host="127.0.0.1",
                port=port,
                scenario_timeout_s=args.scenario_timeout,
            )
            row = _row(cfg, attack, metrics, seed)
            new_rows.append(row)
            print(
                f"  F1={row.get('mean_final_f1')}  trust_auc={row.get('trust_auc')}"
            )
            rows_so_far = _merge(existing, new_rows) if args.merge_existing else new_rows
            _write_outputs(rows_so_far)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{tag}: {type(exc).__name__}: {exc}")
            print(f"  FAILED {tag}: {exc}")
        port += 1

    rows = _merge(existing, new_rows) if args.merge_existing else new_rows
    summary = _write_outputs(rows)
    _print_summary(summary, attack_pool, config_pool)
    if failures:
        print(f"\n=== {len(failures)} failed cell(s) (re-run with --merge-existing) ===")
        for msg in failures:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()
