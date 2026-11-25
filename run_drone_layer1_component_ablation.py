#!/usr/bin/env python3
"""Layer~1 component ablation (leave-one-out).

Layer~1 trust is a renormalized weighted sum of:
  subspace (default 0.50), temporal (0.20), counterfactual probes (0.30).

Variants zero one weight at a time (remaining weights renormalize). Setting
counterfactual weight to 0 also skips probe forwards.

Usage:
  # Smoke
  ./acfl_env/bin/python run_drone_layer1_component_ablation.py \\
    --skip-setup --seeds 42,43,44 --port 9100

  # Resume / expand
  ./acfl_env/bin/python run_drone_layer1_component_ablation.py \\
    --skip-setup --merge-existing --seeds 42,43,44,45,46
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
STEM = "drone_layer1_component_ablation"

# Primary manuscript attack settings.
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

# Leave-one-out (+ full). Weights renormalize in Layer1Validator.
VARIANTS = [
    {
        "name": "l1_full",
        "label": "All three",
        "subspace": 0.50,
        "temporal": 0.20,
        "counterfactual": 0.30,
    },
    {
        "name": "l1_no_subspace",
        "label": "w/o subspace",
        "subspace": 0.0,
        "temporal": 0.20,
        "counterfactual": 0.30,
    },
    {
        "name": "l1_no_temporal",
        "label": "w/o temporal",
        "subspace": 0.50,
        "temporal": 0.0,
        "counterfactual": 0.30,
    },
    {
        "name": "l1_no_cf",
        "label": "w/o counterfactual",
        "subspace": 0.50,
        "temporal": 0.20,
        "counterfactual": 0.0,
    },
]


def _scenario(variant: Dict, attack: Dict, seed: int) -> Scenario:
    tag = f"l1comp_{variant['name']}_{attack['name']}_seed{seed}"
    use_probes = float(variant["counterfactual"]) > 0.0 or attack["attack_type"] == "backdoor"
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
        acfl_trust_weight_power=1.25,
        acfl_min_trust_weight=0.03,
        acfl_coordinate_deviation_threshold=2.5,
        acfl_norm_ratio_threshold=2.0,
        acfl_norm_ratio_threshold_two_clients=1.2,
        acfl_subspace_weight=float(variant["subspace"]),
        acfl_temporal_weight=float(variant["temporal"]),
        acfl_counterfactual_weight=float(variant["counterfactual"]),
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
        "subspace_weight": variant["subspace"],
        "temporal_weight": variant["temporal"],
        "counterfactual_weight": variant["counterfactual"],
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


def _write_summary(rows: List[Dict], summary_path: Path) -> Dict:
    summary: Dict = {}
    attacks = sorted({r["attack"] for r in rows})
    variants = sorted({r["variant"] for r in rows})
    for attack in attacks:
        for variant in variants:
            subset = [
                r for r in rows
                if r["attack"] == attack and r["variant"] == variant
            ]
            f1s = [r["mean_final_f1"] for r in subset if r.get("mean_final_f1") is not None]
            aucs = [r["trust_auc"] for r in subset if r.get("trust_auc") is not None]
            triggers = [
                r["backdoor_mean_attack_prob"] for r in subset
                if r.get("backdoor_mean_attack_prob") is not None
            ]
            key = f"{attack}/{variant}"
            if not subset:
                continue
            summary[key] = {
                "mean_f1": statistics.mean(f1s) if f1s else None,
                "std_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                "f1_values": f1s,
                "mean_trust_auc": statistics.mean(aucs) if aucs else None,
                "std_trust_auc": statistics.pstdev(aucs) if len(aucs) > 1 else 0.0,
                "mean_backdoor_attack_prob": (
                    statistics.mean(triggers) if triggers else None
                ),
                "n_seeds": len(subset),
            }
        full = {
            r["seed"]: r for r in rows
            if r["attack"] == attack and r["variant"] == "l1_full"
            and r.get("mean_final_f1") is not None
        }
        for variant in variants:
            if variant == "l1_full":
                continue
            ablated = {
                r["seed"]: r for r in rows
                if r["attack"] == attack and r["variant"] == variant
                and r.get("mean_final_f1") is not None
            }
            shared = sorted(set(full) & set(ablated))
            if not shared:
                continue
            gaps = [
                ablated[s]["mean_final_f1"] - full[s]["mean_final_f1"]
                for s in shared
            ]
            auc_gaps = []
            for s in shared:
                a = ablated[s].get("trust_auc")
                f = full[s].get("trust_auc")
                if a is not None and f is not None:
                    auc_gaps.append(a - f)
            summary[f"{attack}/{variant}_minus_full"] = {
                "mean_f1_gap": statistics.mean(gaps),
                "std_f1_gap": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
                "f1_gaps": gaps,
                "mean_trust_auc_gap": (
                    statistics.mean(auc_gaps) if auc_gaps else None
                ),
                "n_paired": len(shared),
            }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _load_existing() -> List[Dict]:
    path = TUNING_ROOT / f"{STEM}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _merge(existing: List[Dict], new_rows: List[Dict]) -> List[Dict]:
    replaced = {_cell_key(r) for r in new_rows}
    kept = [r for r in existing if _cell_key(r) not in replaced]
    return kept + new_rows


def _pending(attack_pool, variant_pool, seeds, existing):
    done = {_cell_key(r) for r in existing}
    for attack in attack_pool:
        for seed in seeds:
            for variant in variant_pool:
                key = (seed, attack["name"], variant["name"])
                if key not in done:
                    yield attack, seed, variant


def _write_outputs(rows: List[Dict]) -> Dict:
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TUNING_ROOT / f"{STEM}.csv"
    json_path = TUNING_ROOT / f"{STEM}.json"
    summary_path = TUNING_ROOT / f"{STEM}_summary.json"
    fieldnames = [
        "seed", "variant", "variant_label", "subspace_weight", "temporal_weight",
        "counterfactual_weight", "attack", "attack_type", "scenario",
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
    print("\n=== Layer~1 component ablation ===")
    for attack in attacks:
        print(f"\n{attack['name']}:")
        for variant in variants:
            block = summary.get(f"{attack['name']}/{variant['name']}") or {}
            if block.get("mean_f1") is None:
                continue
            auc = block.get("mean_trust_auc")
            auc_s = f"{auc:.3f}" if auc is not None else "---"
            print(
                f"  {variant['name']:16} F1={block['mean_f1']:.3f}±{block['std_f1']:.3f}  "
                f"trustAUC={auc_s}  n={block['n_seeds']}"
            )
            gap = summary.get(f"{attack['name']}/{variant['name']}_minus_full") or {}
            if gap.get("mean_f1_gap") is not None:
                print(f"  {'':16} ΔF1 vs full={gap['mean_f1_gap']:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=9100)
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
        f"L1 component ablation: {len(pending)} pending "
        f"({len(variant_pool)} variants × {len(attack_pool)} attacks × "
        f"up to {len(seeds)} seeds; {len(existing)} existing)"
    )

    new_rows: List[Dict] = []
    failures: List[str] = []
    port = args.port
    for i, (attack, seed, variant) in enumerate(pending, 1):
        sc = _scenario(variant, attack, seed)
        tag = f"{variant['name']}/{attack['name']}/seed{seed}"
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
        except Exception as exc:  # noqa: BLE001 — keep sweep alive across cell failures
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
