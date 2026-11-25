#!/usr/bin/env python3
"""2×2 factorial: attack intensity × swarm size (n∈{3,5}).

Design (pyDOE2 full factorial per attack):
  Factor A — intensity: Low vs High (extremes of the OFAT sweep)
  Factor B — num_clients: 3 (33% malicious) vs 5 (20% malicious)

Reuses completed n=3 cells from drone_attack_intensity_sweep.json when present.
Runs only missing cells (typically the n=5 arms).

Usage:
  ./acfl_env/bin/python benchmarks/run_drone_attack_factorial.py --skip-setup --merge-existing \\
    --seeds 42,43,44,45,46 --port 8600
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pyDOE2 import fullfact

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
STEM = "drone_attack_factorial"
INTENSITY_JSON = TUNING_ROOT / "drone_attack_intensity_sweep.json"

# Low / High extremes from the OFAT intensity sweep.
FACTOR_A: Dict[str, Dict] = {
    "gradient_scaling": {
        "knob": "scale",
        "levels": {"L": 2.0, "H": 20.0},
        "attack_type": "gradient_scaling",
        "base": {"attack_scale": None},  # filled from level
    },
    "byzantine": {
        "knob": "noise_std",
        "levels": {"L": 0.25, "H": 2.0},
        "attack_type": "byzantine",
        "base": {"attack_scale": 1.0},
    },
    "backdoor": {
        "knob": "poison_fraction",
        "levels": {"L": 0.15, "H": 0.55},
        "attack_type": "backdoor",
        "base": {"attack_scale": 1.0},
    },
    "label_flip": {
        "knob": "flip_fraction",
        "levels": {"L": 0.25, "H": 0.75},
        "attack_type": "label_flip",
        "base": {"attack_scale": 1.0},
    },
    "model_replacement": {
        "knob": "alpha",
        "levels": {"L": 5.0, "H": 20.0},
        "attack_type": "model_replacement",
        "base": {},
    },
}

# Factor B: swarm size → (num_clients, min_clients)
FACTOR_B = {
    3: {"num_clients": 3, "min_clients": 2},
    5: {"num_clients": 5, "min_clients": 3},  # overridden per attack below
}
MIN_CLIENTS_5 = {
    "gradient_scaling": 3,
    "byzantine": 3,
    "label_flip": 3,
    "backdoor": 4,
    "model_replacement": 4,
}

DEFENDERS = [
    {"name": "fedavg", "baseline_aggregation": "fedavg", "defender_family": "baseline"},
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
    {"name": "krum", "baseline_aggregation": "krum", "defender_family": "baseline"},
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


def _value_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _design_points() -> List[Dict]:
    """Return coded full-factorial points via pyDOE2 (2×2)."""
    # fullfact([2,2]) → rows with levels in {0,1}
    coded = fullfact([2, 2])
    level_names = ["L", "H"]
    points = []
    for row in coded:
        a_idx = int(row[0])
        b_idx = int(row[1])
        points.append({
            "intensity_code": level_names[a_idx],
            "n_code": [3, 5][b_idx],
            "a_coded": float(a_idx),  # 0=L, 1=H
            "b_coded": float(b_idx),  # 0=n3, 1=n5
        })
    return points


def _cell_from_design(attack: str, point: Dict) -> Dict:
    meta = FACTOR_A[attack]
    intensity = meta["levels"][point["intensity_code"]]
    n = point["n_code"]
    num_clients = n
    if n == 3:
        min_clients = 2
    else:
        min_clients = MIN_CLIENTS_5[attack]
    cell = {
        "attack": attack,
        "attack_type": meta["attack_type"],
        "knob": meta["knob"],
        "intensity": intensity,
        "intensity_code": point["intensity_code"],
        "num_clients": num_clients,
        "min_clients": min_clients,
        "a_coded": point["a_coded"],
        "b_coded": point["b_coded"],
        "attack_scale": float(meta["base"].get("attack_scale") or intensity),
    }
    if meta["knob"] == "scale" or meta["knob"] == "alpha":
        cell["attack_scale"] = intensity
    if meta["knob"] == "noise_std":
        cell["attack_noise"] = intensity
        cell["attack_scale"] = 1.0
    if meta["knob"] == "poison_fraction":
        cell["backdoor_poison_fraction"] = intensity
        cell["attack_scale"] = 1.0
    if meta["knob"] == "flip_fraction":
        cell["label_flip_fraction"] = intensity
        cell["attack_scale"] = 1.0
    return cell


def _all_cells(attack_filter: Optional[set] = None) -> List[Dict]:
    attacks = [
        a for a in FACTOR_A
        if attack_filter is None or a in attack_filter
    ]
    out = []
    for attack in attacks:
        for point in _design_points():
            out.append(_cell_from_design(attack, point))
    return out


def _scenario(defender: Dict, cell: Dict, seed: int) -> Scenario:
    tag = (
        f"factorial_{defender['name']}_{cell['attack']}_"
        f"{cell['knob']}{_value_tag(cell['intensity'])}_"
        f"n{cell['num_clients']}_seed{seed}"
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
        num_clients=int(cell["num_clients"]),
        min_clients=int(cell["min_clients"]),
        target_rounds=12,
        max_fl_rounds=12,
        probe_file=(
            PROBE_FILE
            if defender["name"] == "acfl_full" or cell["attack_type"] == "backdoor"
            else None
        ),
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
        "intensity": cell["intensity"],
        "intensity_code": cell["intensity_code"],
        "num_clients": cell["num_clients"],
        "min_clients": cell["min_clients"],
        "malicious_fraction": 1.0 / float(cell["num_clients"]),
        "a_coded": cell["a_coded"],
        "b_coded": cell["b_coded"],
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
        "source": metrics.get("source", "run"),
        **{k: v for k, v in _guard_stats(metrics.get("scenario", "")).items()},
    }


def _cell_key(row: Dict) -> Tuple:
    return (
        row.get("seed"),
        row.get("attack"),
        row.get("knob"),
        float(row["intensity"]) if row.get("intensity") is not None else None,
        int(row["num_clients"]) if row.get("num_clients") is not None else None,
        row.get("defender"),
    )


def _load_existing() -> List[Dict]:
    path = TUNING_ROOT / f"{STEM}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _intensity_reuse_index() -> Dict[Tuple, Dict]:
    """Map (seed, attack, knob, intensity, defender) → intensity-sweep row (n=3)."""
    if not INTENSITY_JSON.exists():
        return {}
    rows = json.loads(INTENSITY_JSON.read_text())
    index = {}
    for r in rows:
        key = (
            r.get("seed"),
            r.get("attack"),
            r.get("knob"),
            float(r["intensity"]),
            r.get("defender"),
        )
        index[key] = r
    return index


def _merge(existing: List[Dict], new_rows: List[Dict]) -> List[Dict]:
    replaced = {_cell_key(r) for r in new_rows}
    kept = [r for r in existing if _cell_key(r) not in replaced]
    return kept + new_rows


def _pending(cells, defender_filter, seeds, existing):
    done = {_cell_key(r) for r in existing}
    for cell in cells:
        for seed in seeds:
            for defender in DEFENDERS:
                if defender_filter and defender["name"] not in defender_filter:
                    continue
                key = (
                    seed,
                    cell["attack"],
                    cell["knob"],
                    float(cell["intensity"]),
                    int(cell["num_clients"]),
                    defender["name"],
                )
                if key not in done:
                    yield cell, seed, defender


def _write_outputs(rows: List[Dict]) -> None:
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = TUNING_ROOT / f"{STEM}.csv"
    json_path = TUNING_ROOT / f"{STEM}.json"
    fieldnames = [
        "seed", "defender", "attack", "attack_type", "knob", "intensity",
        "intensity_code", "num_clients", "min_clients", "malicious_fraction",
        "a_coded", "b_coded", "attack_scale", "attack_noise",
        "backdoor_poison_fraction", "label_flip_fraction", "scenario",
        "mean_final_f1", "mean_final_accuracy", "trust_auc",
        "backdoor_mean_attack_prob", "final_round", "source",
        "aggregation_skipped_rounds", "pairwise_penalty_rounds",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    json_path.write_text(json.dumps(rows, indent=2, default=str) + "\n")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


def _import_n3_from_intensity(
    cells: List[Dict],
    seeds: List[int],
    defender_filter: set,
    existing: List[Dict],
) -> List[Dict]:
    """Pull matching n=3 OFAT rows into the factorial artifact."""
    index = _intensity_reuse_index()
    done = {_cell_key(r) for r in existing}
    imported = []
    for cell in cells:
        if int(cell["num_clients"]) != 3:
            continue
        for seed in seeds:
            for defender in DEFENDERS:
                if defender_filter and defender["name"] not in defender_filter:
                    continue
                key = _cell_key({
                    "seed": seed,
                    "attack": cell["attack"],
                    "knob": cell["knob"],
                    "intensity": cell["intensity"],
                    "num_clients": 3,
                    "defender": defender["name"],
                })
                if key in done:
                    continue
                src = index.get((
                    seed, cell["attack"], cell["knob"],
                    float(cell["intensity"]), defender["name"],
                ))
                if not src or src.get("mean_final_f1") is None:
                    continue
                metrics = {
                    "scenario": src.get("scenario"),
                    "final_round": src.get("final_round"),
                    "fl_eval": {
                        "summary": {
                            # reconstruct minimal summary for _row path — bypass via direct row
                        }
                    },
                    "acfl_eval": {"trust_score_auc": src.get("trust_auc")},
                    "backdoor_eval": {
                        "backdoor_mean_attack_prob": src.get("backdoor_mean_attack_prob"),
                    },
                    "source": "reuse_intensity_n3",
                }
                # Build row directly from intensity source
                row = {
                    "seed": seed,
                    "defender": defender["name"],
                    "attack": cell["attack"],
                    "attack_type": cell["attack_type"],
                    "knob": cell["knob"],
                    "intensity": cell["intensity"],
                    "intensity_code": cell["intensity_code"],
                    "num_clients": 3,
                    "min_clients": 2,
                    "malicious_fraction": 1.0 / 3.0,
                    "a_coded": cell["a_coded"],
                    "b_coded": cell["b_coded"],
                    "attack_scale": cell.get("attack_scale"),
                    "attack_noise": cell.get("attack_noise"),
                    "backdoor_poison_fraction": cell.get("backdoor_poison_fraction"),
                    "label_flip_fraction": cell.get("label_flip_fraction"),
                    "scenario": src.get("scenario"),
                    "mean_final_f1": src.get("mean_final_f1"),
                    "mean_final_accuracy": src.get("mean_final_accuracy"),
                    "trust_auc": src.get("trust_auc"),
                    "backdoor_mean_attack_prob": src.get("backdoor_mean_attack_prob"),
                    "final_round": src.get("final_round"),
                    "source": "reuse_intensity_n3",
                    "aggregation_skipped_rounds": src.get("aggregation_skipped_rounds"),
                    "pairwise_penalty_rounds": src.get("pairwise_penalty_rounds"),
                }
                imported.append(row)
                done.add(key)
    return imported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument(
        "--attacks",
        default=",".join(FACTOR_A.keys()),
    )
    parser.add_argument(
        "--defenders",
        default=",".join(d["name"] for d in DEFENDERS),
    )
    parser.add_argument("--seeds", default=seeds_csv())
    parser.add_argument("--scenario-timeout", type=int, default=1200)
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument(
        "--n-clients",
        default="3,5",
        help="Comma-separated swarm sizes to include (default 3,5)",
    )
    args = parser.parse_args()

    if not args.skip_setup:
        sync_agent_missions()
        generate_probes()

    attack_filter = {a.strip() for a in args.attacks.split(",") if a.strip()}
    defender_filter = {d.strip() for d in args.defenders.split(",") if d.strip()}
    n_filter = {int(x) for x in args.n_clients.split(",") if x.strip()}
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    cells = [
        c for c in _all_cells(attack_filter)
        if int(c["num_clients"]) in n_filter
    ]
    print("pyDOE2 2×2 design points:", _design_points())
    print(f"Factorial cells (attack×design): {len(cells)}")

    existing = _load_existing() if args.merge_existing else []
    imported = _import_n3_from_intensity(cells, seeds, defender_filter, existing)
    if imported:
        print(f"Reused {len(imported)} n=3 intensity-sweep rows")
        existing = _merge(existing, imported)
        _write_outputs(existing)

    pending = list(_pending(cells, defender_filter, seeds, existing))
    print(
        f"Factorial: {len(pending)} pending "
        f"({len(existing)} existing after reuse)"
    )

    new_rows: List[Dict] = []
    port = args.port
    for i, (cell, seed, defender) in enumerate(pending, 1):
        sc = _scenario(defender, cell, seed)
        print(
            f"\n[{i}/{len(pending)}] {defender['name']} / {cell['attack']} / "
            f"{cell['intensity_code']}={cell['intensity']:g} / "
            f"n={cell['num_clients']} / seed{seed}  port={port}"
        )
        metrics = run_scenario(
            sc,
            host="127.0.0.1",
            port=port,
            scenario_timeout_s=args.scenario_timeout,
        )
        metrics["source"] = "run"
        row = _row(defender, cell, metrics, seed)
        new_rows.append(row)
        print(
            f"  F1={row.get('mean_final_f1')}  trust_auc={row.get('trust_auc')}"
        )
        rows_so_far = _merge(existing, new_rows)
        _write_outputs(rows_so_far)
        port += 1

    rows = _merge(existing, new_rows)
    _write_outputs(rows)
    print(
        f"\nDone. Rows={len(rows)}. Next: "
        "python scripts/analyze_attack_factorial.py"
    )


if __name__ == "__main__":
    main()
