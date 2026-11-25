#!/usr/bin/env python3
"""Probe-config ablation: full probes vs drone-optimized (16 + attack-scoped).

Compares Layer~1 counterfactual probe settings on ACFL-full only (3 clients,
12 rounds). Metrics: mean final F1 and trust AUC.

Arms:
  * full_probes  — all scenarios, full 48-window probe set
  * drone_opt    — 16 windows/scenario + attack-scoped subsets (production)

Usage:
  # Smoke (3 seeds × 3 attacks × 2 arms)
  ./acfl_env/bin/python benchmarks/run_drone_probe_config_ablation.py \\
    --skip-setup --attacks byzantine,backdoor,label_flip --seeds 42,43,44

  # Full 10-seed resume
  ./acfl_env/bin/python benchmarks/run_drone_probe_config_ablation.py \\
    --skip-setup --merge-existing --seeds 42,43,44,45,46,47,48,49,50,51
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

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
from run_drone_ablation import ATTACKS as ABLATION_ATTACKS


TUNING_ROOT = Path("evaluation_results/acfl_tuning")
PROBE_FILE = str(Path(CHECKPOINT_DIR) / "acfl_probes_sitl.pt")

# Prefer suite + label_flip from ablation list.
ATTACKS = [
    a for a in ABLATION_ATTACKS
    if a["name"] in ("byzantine", "gradient_scaling", "backdoor", "label_flip")
]

PROBE_CONFIGS = [
    {
        "name": "full_probes",
        "label": "Full probes (48×6)",
        "max_windows": 0,  # 0 => unlimited in server CLI
        "attack_scoped": False,
    },
    {
        "name": "drone_opt",
        "label": "Drone (16 + scoped)",
        "max_windows": 16,
        "attack_scoped": True,
    },
]


def _scenario(probe_cfg: Dict, attack: Dict, seed: int) -> Scenario:
    tag = f"probeablation_{probe_cfg['name']}_{attack['name']}_seed{seed}"
    return Scenario(
        name=tag,
        data_dir=SITL_PKT,
        config="config_classification_drone",
        learning_mode="classification",
        malicious_clients="client_2",
        attack_type=attack["attack_type"],
        attack_scale=attack["attack_scale"],
        fedavg=False,
        no_layer1=False,
        no_layer2=False,
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
        acfl_probe_max_windows_per_scenario=probe_cfg["max_windows"],
        acfl_probe_attack_scoped=probe_cfg["attack_scoped"],
        backdoor_poison_fraction=attack.get("backdoor_poison_fraction"),
        label_flip_fraction=attack.get("label_flip_fraction"),
        fl_seed=seed,
    )


def _row(probe_cfg: Dict, attack: Dict, metrics: Dict, seed: int) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {}))
    backdoor = metrics.get("backdoor_eval") or {}
    return {
        "seed": seed,
        "probe_config": probe_cfg["name"],
        "probe_label": probe_cfg["label"],
        "max_windows": probe_cfg["max_windows"],
        "attack_scoped": probe_cfg["attack_scoped"],
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


def _write_summary(rows: List[Dict], summary_path: Path) -> Dict:
    summary: Dict = {}
    attacks = sorted({r["attack"] for r in rows})
    configs = sorted({r["probe_config"] for r in rows})
    for attack in attacks:
        for cfg in configs:
            subset = [
                r for r in rows
                if r["attack"] == attack and r["probe_config"] == cfg
            ]
            f1s = [r["mean_final_f1"] for r in subset if r.get("mean_final_f1") is not None]
            aucs = [r["trust_auc"] for r in subset if r.get("trust_auc") is not None]
            key = f"{attack}/{cfg}"
            summary[key] = {
                "mean_f1": statistics.mean(f1s) if f1s else None,
                "std_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                "f1_values": f1s,
                "mean_trust_auc": statistics.mean(aucs) if aucs else None,
                "std_trust_auc": statistics.pstdev(aucs) if len(aucs) > 1 else 0.0,
                "trust_auc_values": aucs,
                "n_seeds": len(subset),
            }
        full = [r for r in rows if r["attack"] == attack and r["probe_config"] == "full_probes"]
        drone = [r for r in rows if r["attack"] == attack and r["probe_config"] == "drone_opt"]
        if full and drone:
            full_by_seed = {r["seed"]: r for r in full}
            drone_by_seed = {r["seed"]: r for r in drone}
            shared = sorted(set(full_by_seed) & set(drone_by_seed))
            f1_gaps = []
            auc_gaps = []
            for seed in shared:
                f = full_by_seed[seed]
                d = drone_by_seed[seed]
                if f.get("mean_final_f1") is not None and d.get("mean_final_f1") is not None:
                    f1_gaps.append(d["mean_final_f1"] - f["mean_final_f1"])
                if f.get("trust_auc") is not None and d.get("trust_auc") is not None:
                    auc_gaps.append(d["trust_auc"] - f["trust_auc"])
            summary[f"{attack}/drone_minus_full"] = {
                "mean_f1_gap": statistics.mean(f1_gaps) if f1_gaps else None,
                "std_f1_gap": statistics.pstdev(f1_gaps) if len(f1_gaps) > 1 else 0.0,
                "f1_gaps": f1_gaps,
                "mean_trust_auc_gap": statistics.mean(auc_gaps) if auc_gaps else None,
                "std_trust_auc_gap": (
                    statistics.pstdev(auc_gaps) if len(auc_gaps) > 1 else 0.0
                ),
                "trust_auc_gaps": auc_gaps,
                "n_paired": len(shared),
            }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _load_existing() -> List[Dict]:
    path = TUNING_ROOT / "drone_probe_config_ablation.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _merge(existing: List[Dict], new_rows: List[Dict]) -> List[Dict]:
    replaced = {
        (r.get("seed"), r.get("attack"), r.get("probe_config")) for r in new_rows
    }
    kept = [
        r for r in existing
        if (r.get("seed"), r.get("attack"), r.get("probe_config")) not in replaced
    ]
    return kept + new_rows


def _pending(attack_pool, config_pool, seeds, existing):
    done = {
        (r.get("seed"), r.get("attack"), r.get("probe_config")) for r in existing
    }
    for attack in attack_pool:
        for seed in seeds:
            for cfg in config_pool:
                key = (seed, attack["name"], cfg["name"])
                if key not in done:
                    yield attack, seed, cfg


def _write_outputs(rows: List[Dict]) -> Dict:
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TUNING_ROOT / "drone_probe_config_ablation.csv"
    json_path = TUNING_ROOT / "drone_probe_config_ablation.json"
    summary_path = TUNING_ROOT / "drone_probe_config_ablation_summary.json"
    fieldnames = [
        "seed", "probe_config", "probe_label", "max_windows", "attack_scoped",
        "attack", "attack_type", "scenario", "mean_final_f1", "mean_final_accuracy",
        "trust_auc", "backdoor_mean_attack_prob", "final_round",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=7900)
    parser.add_argument(
        "--probe-configs",
        default=",".join(c["name"] for c in PROBE_CONFIGS),
    )
    parser.add_argument(
        "--attacks",
        default="byzantine,backdoor,label_flip",
    )
    parser.add_argument("--seeds", default=seeds_csv())
    parser.add_argument("--scenario-timeout", type=int, default=1200)
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    config_pool = [c for c in PROBE_CONFIGS if c["name"] in args.probe_configs.split(",")]
    attack_pool = [a for a in ATTACKS if a["name"] in args.attacks.split(",")]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    existing = _load_existing() if args.merge_existing else []
    pending = list(_pending(attack_pool, config_pool, seeds, existing))
    print(
        f"Probe-config ablation: {len(pending)} pending cells "
        f"({len(config_pool)} configs × {len(attack_pool)} attacks × "
        f"up to {len(seeds)} seeds; {len(existing)} existing)"
    )

    new_rows: List[Dict] = []
    port = args.port
    for i, (attack, seed, probe_cfg) in enumerate(pending, 1):
        sc = _scenario(probe_cfg, attack, seed)
        print(
            f"\n[{i}/{len(pending)}] {probe_cfg['name']} / {attack['name']} / "
            f"seed{seed}  port={port}"
        )
        metrics = run_scenario(
            sc,
            host="127.0.0.1",
            port=port,
            scenario_timeout_s=args.scenario_timeout,
        )
        row = _row(probe_cfg, attack, metrics, seed)
        new_rows.append(row)
        print(
            f"  F1={row.get('mean_final_f1')}  trust_auc={row.get('trust_auc')}"
        )
        port += 1

    rows = _merge(existing, new_rows) if args.merge_existing else new_rows
    # Keep only requested attacks/configs/seeds in the written artifact when
    # not merging a broader set — but merge-existing keeps all history.
    summary = _write_outputs(rows)

    print("\n=== Probe-config ablation summary ===")
    for attack in sorted({a["name"] for a in attack_pool}):
        print(f"\n{attack}:")
        for cfg in ("full_probes", "drone_opt"):
            block = summary.get(f"{attack}/{cfg}") or {}
            if block.get("mean_f1") is None:
                continue
            auc = block.get("mean_trust_auc")
            auc_s = f"{auc:.3f}" if auc is not None else "---"
            print(
                f"  {cfg:12} F1={block['mean_f1']:.3f}±{block['std_f1']:.3f}  "
                f"trustAUC={auc_s}  n={block['n_seeds']}"
            )
        gap = summary.get(f"{attack}/drone_minus_full") or {}
        if gap.get("mean_f1_gap") is not None:
            print(
                f"  drone−full   ΔF1={gap['mean_f1_gap']:+.3f}  "
                f"ΔAUC={gap.get('mean_trust_auc_gap')}"
            )


if __name__ == "__main__":
    main()
