#!/usr/bin/env python3
"""Full-stack validation: Layer 1 + Layer 2 + Layer 3 together.

For each FL training configuration (FedAvg, L1-only, L2-only, L1+L2 full),
collects metrics from all three ACFL layers on the **same** scenario checkpoint:

  Layer 1 — trust_score_auc (malicious-client detection during FL)
  Layer 2 — mean_final_f1 + aggregation guard stats (robust aggregation utility)
  Layer 3 — aggregate_auroc / F1 on held-out network-attack SITL missions

Outputs live under ``evaluation_results/full_stack/`` (isolated from other studies).

Usage:
    # Fast: reuse ablation/journal checkpoints, eval Layer 1/2/3 metrics only
    ./acfl_env/bin/python benchmarks/run_drone_full_stack_benchmark.py --eval-only --merge-existing

    # Full: train + evaluate per cell (hours)
    ./acfl_env/bin/python benchmarks/run_drone_full_stack_benchmark.py --merge-existing
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ACFL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ACFL_ROOT))

from acfl.experiment_seeds import JOURNAL_SEEDS, seeds_csv  # noqa: E402
from run_comprehensive_validation import (  # noqa: E402
    CHECKPOINT_DIR,
    OUT_ROOT,
    PYTHON,
    SITL_PKT,
    Scenario,
    _fl_eval_summary,
    generate_probes,
    run_scenario,
    sync_agent_missions,
)

STACK_ROOT = ACFL_ROOT / "evaluation_results" / "full_stack"
SCENARIO_ROOT = STACK_ROOT / "scenarios"
LAYER3_ROOT = STACK_ROOT / "layer3"
PROBE_FILE = str(Path(CHECKPOINT_DIR) / "acfl_probes_sitl.pt")
SEEDS = JOURNAL_SEEDS

ATTACKS = [
    {"name": "gradient_scaling", "attack_type": "gradient_scaling", "attack_scale": 10.0},
    {"name": "byzantine", "attack_type": "byzantine", "attack_scale": 1.0},
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

# Stack configurations: which training-time layers are active.
STACK_CONFIGS = [
    {
        "name": "fedavg",
        "label": "FedAvg (no L1/L2)",
        "fedavg": True,
        "no_layer1": False,
        "no_layer2": False,
        "layer1": False,
        "layer2": False,
        "layer3": True,
    },
    {
        "name": "l1_only",
        "label": "L1 only",
        "fedavg": False,
        "no_layer1": False,
        "no_layer2": True,
        "layer1": True,
        "layer2": False,
        "layer3": True,
    },
    {
        "name": "l2_only",
        "label": "L2 only",
        "fedavg": False,
        "no_layer1": True,
        "no_layer2": False,
        "layer1": False,
        "layer2": True,
        "layer3": True,
    },
    {
        "name": "full",
        "label": "L1+L2+L3 (ACFL full)",
        "fedavg": False,
        "no_layer1": False,
        "no_layer2": False,
        "layer1": True,
        "layer2": True,
        "layer3": True,
        "trust_power": 1.25,
        "min_trust": 0.03,
        "coord_thr": 2.5,
        "norm_thr": 2.0,
        "norm2_thr": 1.2,
    },
]


def _scenario(stack: Dict, attack: Dict, seed: int) -> Scenario:
    needs_probes = (
        not stack.get("fedavg")
        or attack.get("attack_type") == "backdoor"
    )
    return Scenario(
        name=f"fullstack_{stack['name']}_{attack['name']}_seed{seed}",
        data_dir=SITL_PKT,
        config="config_classification_drone",
        learning_mode="classification",
        malicious_clients="client_2",
        attack_type=attack["attack_type"],
        attack_scale=attack.get("attack_scale", 1.0),
        fedavg=stack.get("fedavg", False),
        no_layer1=stack.get("no_layer1", False),
        no_layer2=stack.get("no_layer2", False),
        num_clients=3,
        min_clients=2,
        target_rounds=12,
        max_fl_rounds=12,
        probe_file=PROBE_FILE if needs_probes else None,
        acfl_trust_weight_power=stack.get("trust_power"),
        acfl_min_trust_weight=stack.get("min_trust"),
        acfl_coordinate_deviation_threshold=stack.get("coord_thr"),
        acfl_norm_ratio_threshold=stack.get("norm_thr"),
        acfl_norm_ratio_threshold_two_clients=stack.get("norm2_thr"),
        backdoor_poison_fraction=attack.get("backdoor_poison_fraction"),
        label_flip_fraction=attack.get("label_flip_fraction"),
        fl_seed=seed,
    )


def _legacy_scenario_names(stack: Dict, attack: Dict, seed: int) -> List[str]:
    """Names used by ablation / journal / threat-gap runs for checkpoint reuse."""
    defender_map = {
        "fedavg": "fedavg",
        "l1_only": "acfl_no_layer2",
        "l2_only": "acfl_no_layer1",
        "full": "acfl_full",
    }
    defender = defender_map.get(stack["name"], stack["name"])
    attack_name = attack["name"]
    names = [
        f"fullstack_{stack['name']}_{attack_name}_seed{seed}",
        f"ablation_{defender}_{attack_name}_seed{seed}",
        f"journal_{attack_name}_seed{seed}_{defender}",
    ]

    if attack_name == "label_flip":
        if stack["name"] in ("fedavg", "full"):
            threat_def = "fedavg" if stack["name"] == "fedavg" else "acfl_full"
            names.append(f"threat_label_flip_seed{seed}_{threat_def}")
        return names

    if attack_name == "model_replacement":
        # Prefer ablation dirs (all four stacks); fall back to threat-gap for
        # fedavg/full (those ablation cells were bootstrapped from threat JSON).
        if stack["name"] in ("fedavg", "full"):
            threat_def = "fedavg" if stack["name"] == "fedavg" else "acfl_full"
            names.append(f"threat_model_replacement_seed{seed}_{threat_def}")
        return names

    if attack_name == "backdoor":
        if stack["name"] in ("fedavg", "full"):
            journal_def = "fedavg" if stack["name"] == "fedavg" else "acfl_full"
            names.append(f"journal_backdoor_seed{seed}_{journal_def}")
        names.append(f"ablation_{defender}_backdoor")
        names.append(f"ablation_{defender}_backdoor_seed{seed}")
        return names

    return names


def _guard_stats_from_reports(reports_path: Path) -> Dict:
    if not reports_path.is_file():
        return {}
    reports = json.loads(reports_path.read_text())
    skipped = sum(
        1 for r in reports
        if (r.get("aggregation_guard") or {}).get("skipped")
    )
    pairwise = sum(
        1 for r in reports
        for c in (r.get("aggregation_diagnostics") or {}).get("clients", {}).values()
        if c.get("pairwise_inversion_penalty")
    )
    malicious_trusts = []
    for r in reports:
        if r.get("round", 0) <= 2:
            continue
        for cid, score in (r.get("trust_scores") or {}).items():
            if cid == "client_2":
                malicious_trusts.append(float(score))
    out = {
        "aggregation_skipped_rounds": skipped,
        "pairwise_penalty_rounds": pairwise,
    }
    if malicious_trusts:
        out["mean_malicious_trust"] = statistics.mean(malicious_trusts)
        out["final_malicious_trust"] = malicious_trusts[-1]
    return out


def _resolve_scenario_dir(
    stack: Dict,
    attack: Dict,
    seed: int,
    roots: List[Path],
) -> Optional[Path]:
    for root in roots:
        for name in _legacy_scenario_names(stack, attack, seed):
            path = root / name
            if (path / "server_checkpoint.pt").is_file():
                return path
            if (path / "fl_checkpoints" / "server_round_12.pt").is_file():
                return path
    return None


def _resolve_checkpoint(scenario_dir: Path) -> Optional[str]:
    for rel in ("server_checkpoint.pt", "fl_checkpoints/server_round_12.pt"):
        path = scenario_dir / rel
        if path.is_file():
            return str(path)
    return None


def _load_training_metrics(scenario_dir: Path) -> Dict:
    metrics_path = scenario_dir / "metrics.json"
    metrics: Dict = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
    if not metrics.get("fl_eval"):
        report_path = scenario_dir / "fl_eval" / "evaluation_report.json"
        if report_path.is_file():
            metrics["fl_eval"] = json.loads(report_path.read_text())
    return metrics


def _layer3_fields_from_report(layer3_report: Dict) -> Dict:
    """Extract Layer 3 summary fields from an evaluate_layer3_sitl report."""
    if not layer3_report.get("ok") and layer3_report.get("aggregate_auroc") is None:
        return {}
    by_net = layer3_report.get("by_network_attack") or {}
    return {
        "layer3_auroc": layer3_report.get("aggregate_auroc"),
        "layer3_f1": layer3_report.get("aggregate_f1"),
        "layer3_precision": layer3_report.get("aggregate_precision"),
        "layer3_recall": layer3_report.get("aggregate_recall"),
        "layer3_waypoint_auroc": (by_net.get("waypoint") or {}).get("auroc"),
        "layer3_gps_auroc": (by_net.get("gps") or {}).get("auroc"),
    }


def _load_layer3_report(path: Path) -> Dict:
    if not path.is_file():
        return {"ok": False}
    try:
        report = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"ok": False}
    if report.get("aggregate_auroc") is not None:
        report["ok"] = True
    return report


def _run_layer3_eval(checkpoint: str, output_json: Path, force: bool = False) -> Dict:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if not force:
        existing = _load_layer3_report(output_json)
        if existing.get("ok"):
            return existing

    cmd = [
        str(PYTHON),
        str(ACFL_ROOT / "evaluate_layer3_sitl.py"),
        "--data-dir", SITL_PKT,
        "--config", "config_classification_drone",
        "--checkpoint", checkpoint,
        "--probe-file", PROBE_FILE,
        "--probe-blend-weight", "0.35",
        "--output", str(output_json),
    ]
    rc = subprocess.run(
        cmd,
        cwd=str(ACFL_ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    report = _load_layer3_report(output_json)
    if report.get("ok"):
        return report
    return {"ok": False, "rc": rc}


def _compose_row(
    stack: Dict,
    attack: Dict,
    seed: int,
    metrics: Dict,
    layer3_report: Dict,
    checkpoint: Optional[str],
    scenario_dir: Optional[Path],
) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {})) if metrics else {}
    guards = _guard_stats_from_reports(
        (scenario_dir / "acfl_round_reports.json") if scenario_dir else Path(".")
    )
    l3_ok = layer3_report.get("ok", False)
    row = {
        "seed": seed,
        "attack": attack["name"],
        "stack": stack["name"],
        "stack_label": stack["label"],
        "layer1_enabled": stack["layer1"],
        "layer2_enabled": stack["layer2"],
        "layer3_enabled": stack["layer3"],
        "scenario": metrics.get("scenario") or _scenario(stack, attack, seed).name,
        "checkpoint": checkpoint,
        # Layer 1
        "layer1_trust_auc": (metrics.get("acfl_eval") or {}).get("trust_score_auc"),
        "mean_malicious_trust": guards.get("mean_malicious_trust"),
        "final_malicious_trust": guards.get("final_malicious_trust"),
        # Layer 2
        "layer2_mean_f1": fl.get("mean_final_f1"),
        "layer2_mean_accuracy": fl.get("mean_final_accuracy"),
        "aggregation_skipped_rounds": guards.get("aggregation_skipped_rounds"),
        "pairwise_penalty_rounds": guards.get("pairwise_penalty_rounds"),
        "layer3_report": str(
            LAYER3_ROOT / f"{stack['name']}_{attack['name']}_seed{seed}.json"
        ),
    }
    if l3_ok:
        row.update(_layer3_fields_from_report(layer3_report))
    else:
        row.update({
            "layer3_auroc": None,
            "layer3_f1": None,
            "layer3_precision": None,
            "layer3_recall": None,
            "layer3_waypoint_auroc": None,
            "layer3_gps_auroc": None,
        })
    return row


def _summarize(rows: List[Dict]) -> Dict:
    summary: Dict = {}
    for attack in ATTACKS:
        for stack in STACK_CONFIGS:
            subset = [
                r for r in rows
                if r["attack"] == attack["name"] and r["stack"] == stack["name"]
            ]
            if not subset:
                continue
            key = f"{attack['name']}/{stack['name']}"

            def _mean_std(field: str):
                vals = [r[field] for r in subset if r.get(field) is not None]
                if not vals:
                    return None, None, vals
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                return statistics.mean(vals), std, vals

            l1_m, l1_s, l1_v = _mean_std("layer1_trust_auc")
            l2_m, l2_s, l2_v = _mean_std("layer2_mean_f1")
            l3_m, l3_s, l3_v = _mean_std("layer3_auroc")

            summary[key] = {
                "stack_label": stack["label"],
                "layer1_enabled": stack["layer1"],
                "layer2_enabled": stack["layer2"],
                "layer3_enabled": stack["layer3"],
                "mean_layer1_trust_auc": l1_m,
                "std_layer1_trust_auc": l1_s,
                "mean_layer2_f1": l2_m,
                "std_layer2_f1": l2_s,
                "mean_layer3_auroc": l3_m,
                "std_layer3_auroc": l3_s,
                "n_seeds": len(subset),
                "layer1_trust_auc_values": l1_v,
                "layer2_f1_values": l2_v,
                "layer3_auroc_values": l3_v,
            }

        # Full stack vs FedAvg gaps
        fed = [r for r in rows if r["attack"] == attack["name"] and r["stack"] == "fedavg"]
        full = [r for r in rows if r["attack"] == attack["name"] and r["stack"] == "full"]
        fed = sorted(fed, key=lambda r: r["seed"])
        full = sorted(full, key=lambda r: r["seed"])
        n = min(len(fed), len(full))
        if n:
            for field, out_key in (
                ("layer1_trust_auc", "layer1_trust_auc"),
                ("layer2_mean_f1", "layer2_f1"),
                ("layer3_auroc", "layer3_auroc"),
            ):
                gaps = []
                for i in range(n):
                    a = full[i].get(field)
                    f = fed[i].get(field)
                    if a is not None and f is not None:
                        gaps.append(a - f)
                if gaps:
                    summary[f"{attack['name']}/full_minus_fedavg_{out_key}"] = {
                        "mean_gap": statistics.mean(gaps),
                        "std_gap": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
                        "gaps": gaps,
                    }
    return summary


def _write_outputs(rows: List[Dict], summary: Dict):
    STACK_ROOT.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed", "attack", "stack", "stack_label",
        "layer1_enabled", "layer2_enabled", "layer3_enabled",
        "scenario", "checkpoint",
        "layer1_trust_auc", "mean_malicious_trust", "final_malicious_trust",
        "layer2_mean_f1", "layer2_mean_accuracy",
        "aggregation_skipped_rounds", "pairwise_penalty_rounds",
        "layer3_auroc", "layer3_f1", "layer3_precision", "layer3_recall",
        "layer3_waypoint_auroc", "layer3_gps_auroc", "layer3_report",
    ]
    csv_path = STACK_ROOT / "full_stack_benchmark.csv"
    json_path = STACK_ROOT / "full_stack_benchmark.json"
    summary_path = STACK_ROOT / "full_stack_benchmark_summary.json"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    json_path.write_text(json.dumps(rows, indent=2, default=str))
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")


def _load_existing_rows() -> List[Dict]:
    path = STACK_ROOT / "full_stack_benchmark.json"
    return json.loads(path.read_text()) if path.is_file() else []


def _merge_rows(existing: List[Dict], new_rows: List[Dict]) -> List[Dict]:
    replaced = {(r["seed"], r["attack"], r["stack"]) for r in new_rows}
    kept = [
        r for r in existing
        if (r.get("seed"), r.get("attack"), r.get("stack")) not in replaced
    ]
    return kept + new_rows


def main():
    parser = argparse.ArgumentParser(description="Full L1+L2+L3 stack validation")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Reuse checkpoints; load L1/L2 from metrics.json, run Layer 3 eval",
    )
    parser.add_argument(
        "--checkpoint-root",
        action="append",
        default=None,
        help="Scenario dirs to search (repeatable)",
    )
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--seeds", default=seeds_csv())
    parser.add_argument("--attacks", default=",".join(a["name"] for a in ATTACKS))
    parser.add_argument(
        "--stacks",
        default=",".join(s["name"] for s in STACK_CONFIGS),
    )
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument(
        "--backfill-layer3",
        action="store_true",
        help="Reload Layer 3 metrics from existing layer3/*.json into benchmark rows",
    )
    parser.add_argument(
        "--force-layer3",
        action="store_true",
        help="Re-run Layer 3 even when layer3/*.json already exists",
    )
    parser.add_argument("--scenario-timeout", type=int, default=2400)
    args = parser.parse_args()

    if args.backfill_layer3:
        rows = _load_existing_rows()
        updated = 0
        for row in rows:
            report_path = Path(row.get("layer3_report") or "")
            report = _load_layer3_report(report_path)
            if not report.get("ok"):
                continue
            row.update(_layer3_fields_from_report(report))
            updated += 1
        summary = _summarize(rows)
        _write_outputs(rows, summary)
        print(f"Backfilled Layer 3 metrics for {updated}/{len(rows)} rows")
        return

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    seeds = [int(s) for s in args.seeds.split(",")]
    attack_pool = [a for a in ATTACKS if a["name"] in args.attacks.split(",")]
    stack_pool = [s for s in STACK_CONFIGS if s["name"] in args.stacks.split(",")]
    roots = [Path(p) for p in args.checkpoint_root] if args.checkpoint_root else [
        ACFL_ROOT / "evaluation_results" / "comprehensive_validation",
        SCENARIO_ROOT,
    ]

    existing = _load_existing_rows() if args.merge_existing else []
    rows: List[Dict] = []
    port = args.port

    for attack in attack_pool:
        for seed in seeds:
            for stack in stack_pool:
                sc = _scenario(stack, attack, seed)
                metrics: Dict = {}
                scenario_dir: Optional[Path] = None
                checkpoint: Optional[str] = None

                if args.eval_only:
                    scenario_dir = _resolve_scenario_dir(stack, attack, seed, roots)
                    if not scenario_dir:
                        print(f"SKIP {sc.name}: no scenario dir in {roots}")
                        continue
                    checkpoint = _resolve_checkpoint(scenario_dir)
                    metrics = _load_training_metrics(scenario_dir)
                    metrics.setdefault("scenario", scenario_dir.name)
                    print(
                        f"\n{'='*60}\nEVAL-ONLY: {stack['label']} / {attack['name']} "
                        f"seed{seed}\n  dir={scenario_dir}\n{'='*60}"
                    )
                else:
                    metrics = run_scenario(
                        sc,
                        port=port,
                        scenario_timeout_s=args.scenario_timeout,
                        out_root=str(SCENARIO_ROOT),
                    )
                    metrics["scenario"] = sc.name
                    scenario_dir = SCENARIO_ROOT / sc.name
                    checkpoint = metrics.get("checkpoint")
                    port += 1

                if not checkpoint:
                    print(f"SKIP Layer 3 for {sc.name}: missing checkpoint")
                    continue

                l3_out = LAYER3_ROOT / f"{stack['name']}_{attack['name']}_seed{seed}.json"
                layer3_report = _run_layer3_eval(
                    checkpoint, l3_out, force=args.force_layer3
                )
                rows.append(_compose_row(
                    stack, attack, seed, metrics, layer3_report, checkpoint, scenario_dir
                ))
                merged = _merge_rows(existing, rows)
                _write_outputs(merged, _summarize(merged))

    merged = _merge_rows(existing, rows)
    summary = _summarize(merged)
    _write_outputs(merged, summary)

    print("\nFull-stack summary (L1 trust AUC / L2 F1 / L3 AUROC):")
    for attack in attack_pool:
        print(f"\n  [{attack['name']}]")
        for stack in stack_pool:
            key = f"{attack['name']}/{stack['name']}"
            stats = summary.get(key)
            if not stats:
                continue
            l1 = stats.get("mean_layer1_trust_auc")
            l2 = stats.get("mean_layer2_f1")
            l3 = stats.get("mean_layer3_auroc")
            l1s = f"{l1:.3f}" if l1 is not None else "n/a"
            l2s = f"{l2:.3f}" if l2 is not None else "n/a"
            l3s = f"{l3:.3f}" if l3 is not None else "n/a"
            print(
                f"    {stack['name']:8s}  L1={l1s}  L2={l2s}  L3={l3s}  "
                f"({stats['stack_label']})"
            )


if __name__ == "__main__":
    main()
