#!/usr/bin/env python3
"""Microbenchmark server-side per-round aggregation overhead.

Layers 1 and 2 of ACFL run once per FL round on the server (not at inference),
so the "three-layer cost" is a training-time aggregation overhead rather than a
per-window inference latency. This script times the actual aggregation call for
each strategy on the same set of client updates:

  * FedAvg              -- plain average (drift-fairroad FederatedAggregator)
  * Coordinate median   -- acfl.robust_baselines.coordinate_median_aggregate
  * Trimmed mean        -- acfl.robust_baselines.trimmed_mean_aggregate
  * Krum                -- acfl.robust_baselines.krum_aggregate
  * ACFL (L1+L2)        -- ACFLDefense.defend_and_aggregate (counterfactual
                           probes + trust-weighted robust aggregation)

Client updates are synthesized from a real drone TinyPacketCNN checkpoint by
adding weight-scaled Gaussian noise, so vector shapes / probe forward passes are
realistic. Latency here is defender-specific (unlike Layer 3 inference), which is
the whole point of the measurement.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
FL_TS_DIR = ROOT / "fl-time-series"
if not FL_TS_DIR.is_dir():
    FL_TS_DIR = ROOT / "drift-fairroad" / "fl-time-series"
for path in (str(ROOT), str(FL_TS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

DEFAULT_CHECKPOINT = (
    ROOT
    / "evaluation_results"
    / "comprehensive_validation"
    / "journal_backdoor_seed42_krum"
    / "server_checkpoint.pt"
)
DEFAULT_PROBE = FL_TS_DIR / "checkpoints" / "acfl_probes_sitl.pt"
DEFAULT_OUTPUT = ROOT / "evaluation_results" / "full_stack" / "aggregation_overhead.json"
CONFIG_MODULE = "config_classification_drone"


def _clone(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in state.items()}


def _make_client_updates(
    global_state: Dict[str, torch.Tensor],
    num_clients: int,
    noise_scale: float,
    generator: torch.Generator,
) -> List[Dict[str, torch.Tensor]]:
    """Weight-scaled Gaussian perturbations of the global model (benign-like)."""
    clients = []
    for _ in range(num_clients):
        update = {}
        for key, tensor in global_state.items():
            t = tensor.float()
            if t.numel() > 1:
                std = t.std().item() or 1.0
                noise = torch.randn(t.shape, generator=generator) * (noise_scale * std)
                update[key] = t + noise
            else:
                update[key] = t.clone()
        clients.append(update)
    return clients


def _mean_std(values: List[float]) -> Dict[str, float]:
    return {
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _time_strategy(
    label: str,
    call: Callable[[], object],
    repeats: int,
    warmup: int,
) -> Dict[str, float]:
    for _ in range(warmup):
        call()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        timings.append((time.perf_counter() - start) * 1000.0)
    stats = _mean_std(timings)
    print(
        f"{label:22} mean={stats['mean_ms']:.3f} ms "
        f"± {stats['std_ms']:.3f} (median {stats['median_ms']:.3f})",
        flush=True,
    )
    return stats


def _build_acfl_defense(
    model,
    probe_data,
    config_mod,
    enable_layer1,
    enable_layer2,
    *,
    evaluation_attack_type=None,
    probe_max_windows_per_scenario=None,
    probe_attack_scoped=False,
):
    from acfl.config import ACFLConfig
    from acfl.defense import ACFLDefense

    def cfg(name, default):
        return getattr(config_mod, name, default)

    max_windows = probe_max_windows_per_scenario
    if max_windows is None:
        max_windows = cfg("ACFL_PROBE_MAX_WINDOWS_PER_SCENARIO", None)
    attack_scoped = probe_attack_scoped or bool(
        cfg("ACFL_PROBE_ATTACK_SCOPED", False)
    )

    acfl_config = ACFLConfig(
        enable_layer1=enable_layer1,
        enable_layer2=enable_layer2,
        enable_layer3=False,
        min_trust_weight=float(cfg("ACFL_MIN_TRUST_WEIGHT", 0.03)),
        trust_weight_power=float(cfg("ACFL_TRUST_WEIGHT_POWER", 1.25)),
        coordinate_deviation_threshold=float(
            cfg("ACFL_COORDINATE_DEVIATION_THRESHOLD", 2.5)
        ),
        norm_ratio_threshold=float(cfg("ACFL_NORM_RATIO_THRESHOLD", 2.0)),
        norm_ratio_threshold_two_clients=float(
            cfg("ACFL_NORM_RATIO_THRESHOLD_TWO_CLIENTS", 1.2)
        ),
        min_aggregation_participants=int(cfg("ACFL_MIN_AGGREGATION_PARTICIPANTS", 2)),
        pairwise_inversion_guard_enabled=bool(
            cfg("ACFL_PAIRWISE_INVERSION_GUARD_ENABLED", True)
        ),
        robust_norm_soft_cap_enabled=bool(
            cfg("ACFL_ROBUST_NORM_SOFT_CAP_ENABLED", True)
        ),
        layer1_layer2_gating_enabled=bool(
            cfg("ACFL_LAYER1_LAYER2_GATING_ENABLED", True)
        ),
        layer1_layer2_gating_robustness_floor=float(
            cfg("ACFL_LAYER1_LAYER2_GATING_ROBUSTNESS_FLOOR", 0.5)
        ),
        evaluation_attack_type=evaluation_attack_type,
        # Isolate aggregation/probe cost from backdoor post-agg scrub SGD.
        post_agg_trigger_scrub_enabled=False,
        probe_max_windows_per_scenario=(
            int(max_windows) if max_windows is not None else None
        ),
        probe_attack_scoped=bool(attack_scoped),
    )
    return ACFLDefense(
        config=acfl_config,
        model=model,
        probe_data=probe_data,
        device=torch.device("cpu"),
        learning_mode="classification",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--probe-file", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-clients", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    from evaluate_layer3_sitl import load_model, load_probe_set
    from aggregation import FederatedAggregator
    from acfl.robust_baselines import (
        coordinate_median_aggregate,
        krum_aggregate,
        trimmed_mean_aggregate,
    )

    model, config_mod = load_model(str(args.checkpoint), CONFIG_MODULE)
    global_state = {k: v.detach().float() for k, v in model.state_dict().items()}
    probe_data = load_probe_set(str(args.probe_file))
    print(
        f"Loaded model ({sum(v.numel() for v in global_state.values())} params) "
        f"and {len(probe_data) if probe_data else 0} probe scenarios"
    )

    generator = torch.Generator().manual_seed(args.seed)
    client_ids = [f"client_{i}" for i in range(args.num_clients)]
    client_samples = [1000] * args.num_clients

    def fresh_clients():
        return _make_client_updates(
            global_state, args.num_clients, args.noise_scale, generator
        )

    results: Dict[str, Dict[str, float]] = {}

    fedavg = FederatedAggregator(strategy="fedavg")
    results["fedavg"] = _time_strategy(
        "FedAvg",
        lambda: fedavg.aggregate(fresh_clients(), client_samples),
        args.repeats,
        args.warmup,
    )
    results["coordinate_median"] = _time_strategy(
        "Coordinate median",
        lambda: coordinate_median_aggregate(fresh_clients()),
        args.repeats,
        args.warmup,
    )
    results["trimmed_mean"] = _time_strategy(
        "Trimmed mean",
        lambda: trimmed_mean_aggregate(fresh_clients()),
        args.repeats,
        args.warmup,
    )
    results["krum"] = _time_strategy(
        "Krum",
        lambda: krum_aggregate(fresh_clients(), _clone(global_state), 1),
        args.repeats,
        args.warmup,
    )

    def run_acfl(
        enable_layer1,
        enable_layer2,
        *,
        evaluation_attack_type=None,
        probe_max_windows_per_scenario=None,
        probe_attack_scoped=False,
    ):
        defense = _build_acfl_defense(
            model,
            probe_data,
            config_mod,
            enable_layer1,
            enable_layer2,
            evaluation_attack_type=evaluation_attack_type,
            probe_max_windows_per_scenario=probe_max_windows_per_scenario,
            probe_attack_scoped=probe_attack_scoped,
        )

        def call():
            defense.round_number += 1
            defense.defend_and_aggregate(
                client_ids,
                fresh_clients(),
                _clone(global_state),
                client_losses={cid: 0.5 for cid in client_ids},
                mission_context=None,
                client_samples=client_samples,
            )

        return call

    # Legacy full-probe path (engineering wins still apply: batched/pinned).
    results["acfl_l1_only"] = _time_strategy(
        "ACFL L1 only (full)",
        run_acfl(True, False, probe_max_windows_per_scenario=None, probe_attack_scoped=False),
        args.repeats,
        args.warmup,
    )
    results["acfl_l2_only"] = _time_strategy(
        "ACFL L2 only",
        run_acfl(False, True, probe_max_windows_per_scenario=None, probe_attack_scoped=False),
        args.repeats,
        args.warmup,
    )
    results["acfl_full"] = _time_strategy(
        "ACFL L1+L2 (full)",
        run_acfl(True, True, probe_max_windows_per_scenario=None, probe_attack_scoped=False),
        args.repeats,
        args.warmup,
    )
    # Drone production path: 16 windows/scenario + attack-scoped subsets.
    results["acfl_full_drone"] = _time_strategy(
        "ACFL L1+L2 (drone)",
        run_acfl(
            True,
            True,
            evaluation_attack_type="byzantine",
            probe_max_windows_per_scenario=16,
            probe_attack_scoped=True,
        ),
        args.repeats,
        args.warmup,
    )
    results["acfl_full_drone_backdoor"] = _time_strategy(
        "ACFL L1+L2 (bd)",
        run_acfl(
            True,
            True,
            evaluation_attack_type="backdoor",
            probe_max_windows_per_scenario=16,
            probe_attack_scoped=True,
        ),
        args.repeats,
        args.warmup,
    )

    payload = {
        "scope": (
            "Server-side per-round aggregation overhead (Layers 1-2 run once per "
            "FL round during training; Layer 3 is separate inference-time cost)"
        ),
        "device": "CPU (1 thread)",
        "checkpoint": str(args.checkpoint),
        "probe_file": str(args.probe_file),
        "num_clients": args.num_clients,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "noise_scale": args.noise_scale,
        "model_params": sum(v.numel() for v in global_state.values()),
        "probe_scenarios": len(probe_data) if probe_data else 0,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
