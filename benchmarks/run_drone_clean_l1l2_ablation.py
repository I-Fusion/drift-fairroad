#!/usr/bin/env python3
"""Layer / gating / extras ablation harness.

Variants:
  * acfl_l1l2_clean     — L1+L2, gating off, attack-scoped extras off
  * acfl_l1l2_ungated   — L1+L2, gating off, production extras on
  * acfl_l1_only_clean  — L1 only, attack-scoped extras off
  * acfl_l2_only_clean  — L2 only, attack-scoped extras off
  * acfl_pta_hard / soft / conservative — dual-path arbitration (extras on)

Usage:
  ./acfl_env/bin/python benchmarks/run_drone_clean_l1l2_ablation.py \\
    --skip-setup --seeds 42,43,44 --port 9300

  ./acfl_env/bin/python benchmarks/run_drone_clean_l1l2_ablation.py \\
    --skip-setup --merge-existing --variants acfl_l1_only_clean \\
    --seeds 42,43,44,45,46,47,48,49,50,51 --port 9500
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

from acfl.experiment_seeds import seeds_csv
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
STEM = "drone_clean_l1l2_ablation"

ATTACKS = [
    {"name": "gradient_scaling", "attack_type": "gradient_scaling", "attack_scale": 10.0},
    {"name": "byzantine", "attack_type": "byzantine", "attack_scale": 1.0, "attack_noise": 1.0},
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

VARIANTS = [
    {
        "name": "acfl_l1l2_clean",
        "label": "L1+L2 clean (no gating/extras)",
        "gating": False,
        "extras": False,
        "adaptive": False,
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        "name": "acfl_l1l2_ungated",
        "label": "L1+L2 ungated (extras on)",
        "gating": False,
        "extras": True,
        "adaptive": False,
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        "name": "acfl_l1l2_adaptive",
        "label": "L1+L2 adaptive g_t (extras on)",
        "gating": True,
        "extras": True,
        "adaptive": True,
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        "name": "acfl_l1l2_adaptive_hinge",
        "label": "L1+L2 per-client hinge gate (extras on)",
        "gating": True,
        "extras": True,
        "adaptive": True,
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        "name": "acfl_l1l2_adaptive_twosided",
        "label": "L1+L2 two-sided hinge gate (extras on)",
        "gating": True,
        "extras": True,
        "adaptive": True,
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        "name": "acfl_l1l2_adaptive_cohort",
        "label": "L1+L2 cohort suppression gate (extras on)",
        "gating": True,
        "extras": True,
        "adaptive": True,
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        "name": "acfl_l1l2_adaptive_selective",
        "label": "L1+L2 selective closed gate (extras on)",
        "gating": True,
        "extras": True,
        "adaptive": True,
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        "name": "acfl_l1l2_adaptive_hold",
        "label": "L1+L2 cohort gate plus withheld-outlier hold (extras on)",
        "gating": True,
        "extras": True,
        "adaptive": True,
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        "name": "acfl_l1l2_adaptive_misshold",
        "label": "L1+L2 cohort gate plus unique missed-norm hold (extras on)",
        "gating": True,
        "extras": True,
        "adaptive": True,
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        "name": "acfl_l1_only_clean",
        "label": "L1-only clean (no extras)",
        "gating": False,
        "extras": False,
        "adaptive": False,
        "no_layer1": False,
        "no_layer2": True,
    },
    {
        "name": "acfl_l2_only_clean",
        "label": "L2-only clean (no extras)",
        "gating": False,
        "extras": False,
        "adaptive": False,
        "no_layer1": True,
        "no_layer2": False,
    },
    {
        "name": "acfl_pta_hard",
        "label": "PTA hard winner (extras on)",
        "gating": True,
        "extras": True,
        "adaptive": False,
        "pta": True,
        "pta_method": "hard",
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        "name": "acfl_pta_soft",
        "label": "PTA soft mixture (extras on)",
        "gating": True,
        "extras": True,
        "adaptive": False,
        "pta": True,
        "pta_method": "soft",
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        "name": "acfl_pta_conservative",
        "label": "PTA conservative fallback (extras on)",
        "gating": True,
        "extras": True,
        "adaptive": False,
        "pta": True,
        "pta_method": "conservative",
        "no_layer1": False,
        "no_layer2": False,
    },
]


def _scenario(variant: Dict, attack: Dict, seed: int) -> Scenario:
    tag = f"cleanl1l2_{variant['name']}_{attack['name']}_seed{seed}"
    extras = bool(variant.get("extras", False))
    adaptive = bool(variant.get("adaptive", False))
    gating = bool(variant.get("gating", False)) or adaptive
    no_layer1 = bool(variant.get("no_layer1", False))
    no_layer2 = bool(variant.get("no_layer2", False))
    # Probes needed for L1 scoring and/or backdoor trigger evaluation.
    use_probes = (not no_layer1) or attack["attack_type"] == "backdoor"
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
        probe_file=PROBE_FILE if use_probes else None,
        no_layer1=no_layer1,
        no_layer2=no_layer2,
        acfl_trust_weight_power=1.25,
        acfl_min_trust_weight=0.03,
        acfl_coordinate_deviation_threshold=2.5,
        acfl_norm_ratio_threshold=2.0,
        acfl_norm_ratio_threshold_two_clients=1.2,
        # Gating builds the neutralized endpoint; adaptive blends it with raw L1.
        # PTA keeps extras identical and arbitrates gated vs ungated after agg.
        acfl_layer1_layer2_gating=gating,
        acfl_adaptive_l1l2_gating=adaptive,
        acfl_parallel_trust_arbitration=bool(variant.get("pta", False)),
        acfl_pta_method=variant.get("pta_method"),
        # Attack-scoped extras: off for clean arms, on for ungated.
        acfl_backdoor_geometry_hard_reject=extras,
        acfl_label_flip_geometry_hard_reject=extras,
        acfl_backdoor_probe_hard_reject=extras,
        acfl_post_agg_trigger_scrub=extras,
        acfl_update_norm_hard_rejection_for_model_replacement=extras,
        acfl_backdoor_probe_soft_cap=False,
        acfl_backdoor_trigger_hard_gate=False,
        backdoor_poison_fraction=attack.get("backdoor_poison_fraction"),
        label_flip_fraction=attack.get("label_flip_fraction"),
        fl_seed=seed,
        quick_fl_eval=True,
    )


def _row(variant: Dict, attack: Dict, metrics: Dict, seed: int) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {}))
    backdoor = metrics.get("backdoor_eval") or {}
    return {
        "seed": seed,
        "variant": variant["name"],
        "variant_label": variant["label"],
        "attack": attack["name"],
        "attack_type": attack["attack_type"],
        "scenario": metrics.get("scenario"),
        "mean_final_f1": fl.get("mean_final_f1"),
        "mean_final_accuracy": fl.get("mean_final_accuracy"),
        "trust_auc": metrics.get("acfl_eval", {}).get("trust_score_auc"),
        "backdoor_mean_attack_prob": backdoor.get("backdoor_mean_attack_prob"),
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


def _pending(attack_pool, variant_pool, seeds, existing):
    done = {_cell_key(r) for r in existing}
    for attack in attack_pool:
        for seed in seeds:
            for variant in variant_pool:
                key = (seed, attack["name"], variant["name"])
                if key not in done:
                    yield attack, seed, variant


def _write_summary(rows: List[Dict], summary_path: Path) -> Dict:
    summary: Dict = {}
    for attack in sorted({r["attack"] for r in rows}):
        for variant in sorted({r["variant"] for r in rows}):
            subset = [
                r for r in rows
                if r["attack"] == attack and r["variant"] == variant
            ]
            f1s = [r["mean_final_f1"] for r in subset if r.get("mean_final_f1") is not None]
            aucs = [r["trust_auc"] for r in subset if r.get("trust_auc") is not None]
            triggers = [
                r["backdoor_mean_attack_prob"]
                for r in subset
                if r.get("backdoor_mean_attack_prob") is not None
            ]
            key = f"{attack}/{variant}"
            summary[key] = {
                "mean_f1": statistics.mean(f1s) if f1s else None,
                "std_f1": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
                "f1_values": f1s,
                "mean_trust_auc": statistics.mean(aucs) if aucs else None,
                "std_trust_auc": statistics.stdev(aucs) if len(aucs) > 1 else 0.0,
                "mean_backdoor_attack_prob": (
                    statistics.mean(triggers) if triggers else None
                ),
                "std_backdoor_attack_prob": (
                    statistics.stdev(triggers) if len(triggers) > 1 else 0.0
                ),
                "n_seeds": len(subset),
            }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _write_outputs(rows: List[Dict]) -> Dict:
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TUNING_ROOT / f"{STEM}.csv"
    json_path = TUNING_ROOT / f"{STEM}.json"
    summary_path = TUNING_ROOT / f"{STEM}_summary.json"
    fieldnames = [
        "seed", "variant", "variant_label", "attack", "attack_type", "scenario",
        "mean_final_f1", "mean_final_accuracy", "trust_auc",
        "backdoor_mean_attack_prob", "final_round",
        "aggregation_skipped_rounds", "pairwise_penalty_rounds",
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


def _print_summary(summary: Dict, attacks: List[Dict], variants: List[Dict]) -> None:
    print("\n=== Clean L1+L2 ablation (no gating / no attack-scoped extras) ===")
    for attack in attacks:
        print(f"\n{attack['name']}:")
        for variant in variants:
            block = summary.get(f"{attack['name']}/{variant['name']}") or {}
            if block.get("mean_f1") is None:
                continue
            auc = block.get("mean_trust_auc")
            auc_s = f"{auc:.3f}" if auc is not None else "---"
            trig = block.get("mean_backdoor_attack_prob")
            trig_s = f"  triggerP={trig:.3f}" if trig is not None else ""
            print(
                f"  {variant['name']:18} F1={block['mean_f1']:.3f}±{block['std_f1']:.3f}  "
                f"trustAUC={auc_s}  n={block['n_seeds']}{trig_s}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument(
        "--variants",
        default=",".join(v["name"] for v in VARIANTS),
    )
    parser.add_argument(
        "--attacks",
        default=",".join(a["name"] for a in ATTACKS),
    )
    parser.add_argument("--seeds", default=seeds_csv())
    parser.add_argument("--scenario-timeout", type=int, default=1200)
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    variant_pool = [v for v in VARIANTS if v["name"] in args.variants.split(",")]
    attack_pool = [a for a in ATTACKS if a["name"] in args.attacks.split(",")]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    existing = _load_existing() if args.merge_existing else []
    pending = list(_pending(attack_pool, variant_pool, seeds, existing))
    print(
        f"Clean L1+L2 ablation: {len(pending)} pending "
        f"({len(variant_pool)} variants × {len(attack_pool)} attacks × "
        f"up to {len(seeds)} seeds; {len(existing)} existing)"
    )

    new_rows: List[Dict] = []
    failures: List[str] = []
    port = args.port
    for i, (attack, seed, variant) in enumerate(pending, 1):
        tag = f"{variant['name']}/{attack['name']}/seed{seed}"
        sc = _scenario(variant, attack, seed)
        print(
            f"\n[{i}/{len(pending)}] {variant['name']} / {attack['name']} / "
            f"seed{seed}  port={port}"
        )
        try:
            metrics = run_scenario(
                sc,
                host="127.0.0.1",
                port=port,
                scenario_timeout_s=args.scenario_timeout,
            )
            row = _row(variant, attack, metrics, seed)
            new_rows.append(row)
            print(
                f"  F1={row.get('mean_final_f1')}  trust_auc={row.get('trust_auc')}  "
                f"triggerP={row.get('backdoor_mean_attack_prob')}"
            )
            rows_so_far = _merge(existing, new_rows) if args.merge_existing else new_rows
            _write_outputs(rows_so_far)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{tag}: {type(exc).__name__}: {exc}")
            print(f"  FAILED {tag}: {exc}")
            print("  continuing with next cell (merge-existing will retry later)")
        port += 1

    rows = _merge(existing, new_rows) if args.merge_existing else new_rows
    summary = _write_outputs(rows)
    _print_summary(summary, attack_pool, variant_pool)
    if failures:
        print(f"\n=== {len(failures)} failed cell(s) (re-run with --merge-existing) ===")
        for msg in failures:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()
