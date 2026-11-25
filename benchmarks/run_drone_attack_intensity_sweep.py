#!/usr/bin/env python3
"""Attack-intensity sensitivity sweep (reviewer response).

Varies one intensity knob per attack while holding the rest of the protocol
fixed (3 clients, 12 rounds, drone probes, seeds). Primary manuscript settings
are included so curves are self-contained.

Default defenders: FedAvg, coordinate median, trimmed mean, Krum,
ACFL-full, and optional Layer~1-only / Layer~2-only ACFL arms.

Usage:
  # Smoke (3 seeds)
  ./acfl_env/bin/python benchmarks/run_drone_attack_intensity_sweep.py \\
    --skip-setup --seeds 42,43,44 --port 8000

  # Full 5-seed resume
  ./acfl_env/bin/python benchmarks/run_drone_attack_intensity_sweep.py \\
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
STEM = "drone_attack_intensity_sweep"

# One intensity axis per attack. `primary=True` marks the manuscript default.
INTENSITY_GRID: List[Dict] = [
    # Gradient scaling: scale_factor via attack_scale
    *[{"attack": "gradient_scaling", "attack_type": "gradient_scaling",
       "knob": "scale", "value": v, "attack_scale": v, "primary": v == 10.0}
      for v in (2.0, 5.0, 10.0, 20.0)],
    # Byzantine: noise_std via attack_noise
    *[{"attack": "byzantine", "attack_type": "byzantine",
       "knob": "noise_std", "value": v, "attack_scale": 1.0,
       "attack_noise": v, "primary": v == 1.0}
      for v in (0.25, 0.5, 1.0, 2.0)],
    # Backdoor: poison fraction (weight amp unused under data-poison-only)
    *[{"attack": "backdoor", "attack_type": "backdoor",
       "knob": "poison_fraction", "value": v, "attack_scale": 1.0,
       "backdoor_poison_fraction": v, "primary": v == 0.35}
      for v in (0.15, 0.35, 0.55)],
    # Label flip
    *[{"attack": "label_flip", "attack_type": "label_flip",
       "knob": "flip_fraction", "value": v, "attack_scale": 1.0,
       "label_flip_fraction": v, "primary": v == 0.5}
      for v in (0.25, 0.5, 0.75)],
    # Model replacement: alpha via attack_scale
    *[{"attack": "model_replacement", "attack_type": "model_replacement",
       "knob": "alpha", "value": v, "attack_scale": v, "primary": v == 10.0}
      for v in (5.0, 10.0, 20.0)],
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
        "no_layer1": False,
        "no_layer2": False,
    },
    {
        # Layer~1 only: counterfactual/trust validation without Layer~2 aggregation.
        "name": "acfl_l1_only",
        "baseline_aggregation": None,
        "defender_family": "acfl",
        "trust_power": 1.25,
        "min_trust": 0.03,
        "coord_thr": 2.5,
        "norm_thr": 2.0,
        "norm2_thr": 1.2,
        "no_layer1": False,
        "no_layer2": True,
    },
    {
        # Layer~2 only: mission-context aggregation without Layer~1 trust.
        "name": "acfl_l2_only",
        "baseline_aggregation": None,
        "defender_family": "acfl",
        "trust_power": 1.25,
        "min_trust": 0.03,
        "coord_thr": 2.5,
        "norm_thr": 2.0,
        "norm2_thr": 1.2,
        "no_layer1": True,
        "no_layer2": False,
    },
]


def _value_tag(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def _defenders_for(attack_name: str) -> List[Dict]:
    return list(DEFENDERS)


def _scenario(defender: Dict, cell: Dict, seed: int) -> Scenario:
    tag = (
        f"intensity_{defender['name']}_{cell['attack']}_"
        f"{cell['knob']}{_value_tag(cell['value'])}_seed{seed}"
    )
    return Scenario(
        name=tag,
        data_dir=SITL_PKT,
        config="config_classification_drone",
        learning_mode="classification",
        malicious_clients="client_2",
        attack_type=cell["attack_type"],
        attack_scale=float(cell.get("attack_scale", 1.0)),
        attack_noise=cell.get("attack_noise"),
        baseline_aggregation=defender.get("baseline_aggregation"),
        num_clients=3,
        min_clients=2,
        target_rounds=12,
        max_fl_rounds=12,
        probe_file=(
            PROBE_FILE
            if defender.get("defender_family") == "acfl"
            or cell["attack_type"] == "backdoor"
            else None
        ),
        no_layer1=bool(defender.get("no_layer1", False)),
        no_layer2=bool(defender.get("no_layer2", False)),
        acfl_trust_weight_power=defender.get("trust_power"),
        acfl_min_trust_weight=defender.get("min_trust"),
        acfl_coordinate_deviation_threshold=defender.get("coord_thr"),
        acfl_norm_ratio_threshold=defender.get("norm_thr"),
        acfl_norm_ratio_threshold_two_clients=defender.get("norm2_thr"),
        backdoor_poison_fraction=cell.get("backdoor_poison_fraction"),
        label_flip_fraction=cell.get("label_flip_fraction"),
        fl_seed=seed,
        quick_fl_eval=True,
    )


def _row(defender: Dict, cell: Dict, metrics: Dict, seed: int) -> Dict:
    fl = _fl_eval_summary(metrics.get("fl_eval", {}))
    backdoor = metrics.get("backdoor_eval") or {}
    return {
        "seed": seed,
        "defender": defender["name"],
        "attack": cell["attack"],
        "attack_type": cell["attack_type"],
        "knob": cell["knob"],
        "intensity": cell["value"],
        "primary_setting": bool(cell.get("primary")),
        "attack_scale": cell.get("attack_scale"),
        "attack_noise": cell.get("attack_noise"),
        "backdoor_poison_fraction": cell.get("backdoor_poison_fraction"),
        "label_flip_fraction": cell.get("label_flip_fraction"),
        "scenario": metrics.get("scenario"),
        "mean_final_f1": fl.get("mean_final_f1"),
        "mean_final_accuracy": fl.get("mean_final_accuracy"),
        "trust_auc": metrics.get("acfl_eval", {}).get("trust_score_auc"),
        "backdoor_mean_attack_prob": backdoor.get("backdoor_mean_attack_prob"),
        "final_round": metrics.get("final_round"),
        **{k: v for k, v in _guard_stats(metrics.get("scenario", "")).items()},
    }


def _cell_key(row: Dict) -> Tuple:
    return (
        row.get("seed"),
        row.get("attack"),
        row.get("knob"),
        float(row.get("intensity")) if row.get("intensity") is not None else None,
        row.get("defender"),
    )


def _write_summary(rows: List[Dict], summary_path: Path) -> Dict:
    summary: Dict = {}
    attacks = sorted({r["attack"] for r in rows})
    for attack in attacks:
        attack_rows = [r for r in rows if r["attack"] == attack]
        knobs = sorted({r["knob"] for r in attack_rows})
        for knob in knobs:
            intensities = sorted({
                float(r["intensity"]) for r in attack_rows if r["knob"] == knob
            })
            for intensity in intensities:
                for defender in sorted({r["defender"] for r in attack_rows}):
                    subset = [
                        r for r in attack_rows
                        if r["knob"] == knob
                        and float(r["intensity"]) == intensity
                        and r["defender"] == defender
                    ]
                    f1s = [
                        r["mean_final_f1"] for r in subset
                        if r.get("mean_final_f1") is not None
                    ]
                    aucs = [
                        r["trust_auc"] for r in subset
                        if r.get("trust_auc") is not None
                    ]
                    triggers = [
                        r["backdoor_mean_attack_prob"] for r in subset
                        if r.get("backdoor_mean_attack_prob") is not None
                    ]
                    key = f"{attack}/{knob}={intensity:g}/{defender}"
                    if not subset:
                        continue
                    summary[key] = {
                        "mean_f1": statistics.mean(f1s) if f1s else None,
                        "std_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                        "f1_values": f1s,
                        "mean_trust_auc": (
                            statistics.mean(aucs) if aucs else None
                        ),
                        "mean_backdoor_attack_prob": (
                            statistics.mean(triggers) if triggers else None
                        ),
                        "n_seeds": len(subset),
                        "primary_setting": any(r.get("primary_setting") for r in subset),
                    }
                # Paired ACFL - FedAvg gap at this intensity
                fed = [
                    r for r in attack_rows
                    if r["knob"] == knob
                    and float(r["intensity"]) == intensity
                    and r["defender"] == "fedavg"
                ]
                acfl = [
                    r for r in attack_rows
                    if r["knob"] == knob
                    and float(r["intensity"]) == intensity
                    and r["defender"] == "acfl_full"
                ]
                fed_by = {r["seed"]: r for r in fed if r.get("mean_final_f1") is not None}
                acfl_by = {r["seed"]: r for r in acfl if r.get("mean_final_f1") is not None}
                shared = sorted(set(fed_by) & set(acfl_by))
                if shared:
                    gaps = [
                        acfl_by[s]["mean_final_f1"] - fed_by[s]["mean_final_f1"]
                        for s in shared
                    ]
                    summary[f"{attack}/{knob}={intensity:g}/acfl_minus_fedavg"] = {
                        "mean_gap": statistics.mean(gaps),
                        "std_gap": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
                        "gaps": gaps,
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


def _pending(cells, defender_filter, seeds, existing):
    done = {_cell_key(r) for r in existing}
    for cell in cells:
        for seed in seeds:
            for defender in _defenders_for(cell["attack"]):
                if defender_filter and defender["name"] not in defender_filter:
                    continue
                key = (
                    seed,
                    cell["attack"],
                    cell["knob"],
                    float(cell["value"]),
                    defender["name"],
                )
                if key not in done:
                    yield cell, seed, defender


def _write_outputs(rows: List[Dict]) -> Dict:
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TUNING_ROOT / f"{STEM}.csv"
    json_path = TUNING_ROOT / f"{STEM}.json"
    summary_path = TUNING_ROOT / f"{STEM}_summary.json"
    fieldnames = [
        "seed", "defender", "attack", "attack_type", "knob", "intensity",
        "primary_setting", "attack_scale", "attack_noise",
        "backdoor_poison_fraction", "label_flip_fraction", "scenario",
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


def _print_summary(summary: Dict, cells: List[Dict]) -> None:
    print("\n=== Attack-intensity sweep summary ===")
    attacks = sorted({c["attack"] for c in cells})
    for attack in attacks:
        print(f"\n{attack}:")
        attack_cells = [c for c in cells if c["attack"] == attack]
        knob = attack_cells[0]["knob"]
        for cell in sorted(attack_cells, key=lambda c: c["value"]):
            intensity = cell["value"]
            parts = []
            for defender in _defenders_for(attack):
                block = summary.get(
                    f"{attack}/{knob}={intensity:g}/{defender['name']}"
                ) or {}
                f1 = block.get("mean_f1")
                if f1 is None:
                    continue
                mark = "*" if cell.get("primary") else " "
                parts.append(f"{defender['name']}={f1:.3f}")
            gap = summary.get(f"{attack}/{knob}={intensity:g}/acfl_minus_fedavg") or {}
            gap_s = (
                f"  Δ={gap['mean_gap']:+.3f}"
                if gap.get("mean_gap") is not None
                else ""
            )
            if parts:
                print(
                    f"  {mark}{knob}={intensity:g}: " + ", ".join(parts) + gap_s
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--attacks",
        default=",".join(sorted({c["attack"] for c in INTENSITY_GRID})),
        help="Comma-separated attack names",
    )
    parser.add_argument(
        "--defenders",
        default=(
            "fedavg,coordinate_median,trimmed_mean,krum,"
            "acfl_full,acfl_l1_only,acfl_l2_only"
        ),
        help="Comma-separated defenders",
    )
    parser.add_argument("--seeds", default=seeds_csv())
    parser.add_argument("--scenario-timeout", type=int, default=1200)
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Only run manuscript-primary intensity levels",
    )
    parser.add_argument(
        "--exclude-primary",
        action="store_true",
        help="Skip manuscript-primary levels (reuse journal narrative)",
    )
    args = parser.parse_args()

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    attack_filter = set(a.strip() for a in args.attacks.split(",") if a.strip())
    defender_filter = set(d.strip() for d in args.defenders.split(",") if d.strip())
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cells = [
        c for c in INTENSITY_GRID
        if c["attack"] in attack_filter
        and (not args.primary_only or c.get("primary"))
        and (not args.exclude_primary or not c.get("primary"))
    ]

    existing = _load_existing() if args.merge_existing else []
    pending = list(_pending(cells, defender_filter, seeds, existing))
    print(
        f"Intensity sweep: {len(pending)} pending cells "
        f"({len(cells)} intensities × up to {len(seeds)} seeds; "
        f"{len(existing)} existing)"
    )

    new_rows: List[Dict] = []
    failures: List[str] = []
    port = args.port
    for i, (cell, seed, defender) in enumerate(pending, 1):
        tag = (
            f"{defender['name']}/{cell['attack']}/"
            f"{cell['knob']}={cell['value']:g}/seed{seed}"
        )
        sc = _scenario(defender, cell, seed)
        print(
            f"\n[{i}/{len(pending)}] {defender['name']} / {cell['attack']} / "
            f"{cell['knob']}={cell['value']:g} / seed{seed}  port={port}"
        )
        try:
            metrics = run_scenario(
                sc,
                host="127.0.0.1",
                port=port,
                scenario_timeout_s=args.scenario_timeout,
            )
            row = _row(defender, cell, metrics, seed)
            new_rows.append(row)
            print(
                f"  F1={row.get('mean_final_f1')}  trust_auc={row.get('trust_auc')}  "
                f"triggerP={row.get('backdoor_mean_attack_prob')}"
            )
            # Checkpoint after each cell so kills don't lose progress
            rows_so_far = _merge(existing, new_rows) if args.merge_existing else new_rows
            _write_outputs(rows_so_far)
        except Exception as exc:  # noqa: BLE001 — keep sweep alive across cell failures
            failures.append(f"{tag}: {type(exc).__name__}: {exc}")
            print(f"  FAILED {tag}: {exc}")
            print("  continuing with next cell (merge-existing will retry later)")
        port += 1

    rows = _merge(existing, new_rows) if args.merge_existing else new_rows
    summary = _write_outputs(rows)
    _print_summary(summary, cells)
    if failures:
        print(f"\n=== {len(failures)} failed cell(s) (re-run with --merge-existing) ===")
        for msg in failures:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()
