#!/usr/bin/env python3
"""
Generate the ACFL counterfactual probe set.

Layer 1 of ACFL validates client updates by measuring their *counterfactual*
effect on compact probe sets representing benign flight and a range of attack
scenarios (GNSS spoofing, ADS-B injection, MAVLink waypoint injection, jamming,
and backdoor triggers). This script builds those probe sets directly from the
drift-fairroad data, using the same windowing/normalization as training so the
probes are in-distribution for the global model.

Output: a torch ``.pt`` file mapping ``scenario_name -> (inputs, labels)`` that
``acfl.layer1_validation.ProbeEvaluator`` consumes via ``--probe-file``.

Examples
--------
Classification (packet streams):

    python3 generate_probe_set.py --mode classification \\
        --packet-file data/network_packets/mission_2_wp_23_attack_add_wp_5_alt_0005_labeled.csv \\
        --output fl-time-series/checkpoints/acfl_probes.pt

Regression (GPS/IMU sensor streams):

    python3 generate_probe_set.py --mode regression \\
        --benign-gps  data/train/gps-imu/gps_client_000.csv \\
        --benign-imu  data/train/gps-imu/imu_client_000.csv \\
        --spoof-gps   data/gps_spoofing/mission_2_wp_23_gps_input_increment_wind_10_gps.csv \\
        --spoof-imu   data/gps_spoofing/mission_2_wp_23_gps_input_increment_wind_10_imu.csv \\
        --waypoint-gps data/waypoint_injection/mission_2_wp_23_attack_add_wp_5_alt_0005_gps.csv \\
        --waypoint-imu data/waypoint_injection/mission_2_wp_23_attack_add_wp_5_alt_0005_imu.csv \\
        --output fl-time-series/checkpoints/acfl_probes_reg.pt
"""
import argparse
import glob
import logging
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# Repo root is drift-fairroad after the ACFL deliverable merge (also works when
# nested under an ACFL tree that still has a drift-fairroad/ subdirectory).
ACFL_ROOT = os.path.dirname(os.path.abspath(__file__))
_nested = os.path.join(ACFL_ROOT, "drift-fairroad")
DRIFT_ROOT = _nested if os.path.isdir(os.path.join(_nested, "fl-time-series")) else ACFL_ROOT
FL_TS_DIR = os.path.join(DRIFT_ROOT, "fl-time-series")

for path in [ACFL_ROOT, DRIFT_ROOT, FL_TS_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("acfl.probes")

# Attack scenarios the ProbeEvaluator scores (must match ACFLConfig.probe_scenarios).
ATTACK_SCENARIOS = [
    "gnss_spoofing",
    "adsb_injection",
    "mavlink_waypoint",
    "jamming",
    "backdoor_trigger",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sample_rows(arr: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Sample up to ``n`` rows (with replacement only if fewer are available)."""
    if len(arr) == 0:
        return arr
    rng = np.random.default_rng(seed)
    replace = len(arr) < n
    idx = rng.choice(len(arr), size=n, replace=replace)
    return arr[idx]


def _stamp_backdoor_trigger(
    benign_windows: np.ndarray,
    trigger_value: float = 3.0,
    trigger_feature_idxs: Optional[Tuple[int, ...]] = None,
    trigger_timesteps: int = 5,
) -> np.ndarray:
    """Stamp a fixed, rare trigger pattern into benign windows."""
    from acfl.backdoor_data import stamp_backdoor_trigger

    return stamp_backdoor_trigger(
        benign_windows,
        trigger_value=trigger_value,
        trigger_feature_idxs=trigger_feature_idxs,
        trigger_timesteps=trigger_timesteps,
    )


def _save_probe_set(
    probe_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    output_path: str,
    meta: dict,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {"probe_data": probe_data, "meta": meta}
    torch.save(payload, output_path)
    logger.info("\nProbe set saved to %s", output_path)
    logger.info("Scenarios:")
    for name, (x, y) in probe_data.items():
        pos = int(y.sum().item()) if y.numel() else 0
        logger.info(
            "  %-16s windows=%-4d shape=%s positives=%d",
            name, x.shape[0], tuple(x.shape), pos,
        )


# ---------------------------------------------------------------------------
# Classification probes (packet streams)
# ---------------------------------------------------------------------------
def build_classification_probes(
    packet_file: str,
    config,
    windows_per_scenario: int,
    seed: int,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    from data_preprocessing_time_series import DataPreprocessorTimeSeries, split_temporal_windows

    pre = DataPreprocessorTimeSeries(
        packet_file=packet_file,
        packet_features=getattr(config, "PACKET_FEATURES", None),
        packet_timestamp_col=getattr(config, "PACKET_TIMESTAMP_COL", "Timestamp"),
        label_col=getattr(config, "LABEL_COL", "Label"),
        use_labels=True,
    )
    test_split = float(getattr(config, "TEST_SPLIT", 0.2))
    use_temporal = bool(getattr(config, "TEMPORAL_TRAIN_SPLIT", False))
    norm_frac = (1.0 - test_split) if use_temporal else None
    X, y, n_features = pre.preprocess(
        window_size=config.WINDOW_SIZE,
        overlap=config.OVERLAP,
        norm_train_fraction=norm_frac,
    )
    if use_temporal:
        X, y, _, _, _ = split_temporal_windows(X, y, test_split=test_split)
    y = np.asarray(y).astype(np.int64)
    benign = X[y == 0]
    attack = X[y == 1]
    logger.info(
        "Windowed %d benign / %d attack windows (%d features)",
        len(benign), len(attack), n_features,
    )
    if len(benign) == 0 or len(attack) == 0:
        raise ValueError(
            "Need both benign and attack windows; got "
            f"{len(benign)} benign / {len(attack)} attack. "
            "Use a labeled packet file containing attack traffic."
        )

    probe_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    # Benign probe (label 0).
    benign_probe = _sample_rows(benign, windows_per_scenario, seed)
    probe_data["benign"] = (
        torch.tensor(benign_probe, dtype=torch.float32),
        torch.zeros(len(benign_probe), dtype=torch.float32),
    )

    # Attack-positive probes (label 1). The packet testbed exposes a single
    # attack class (waypoint-injection-derived traffic), so mavlink_waypoint is
    # populated from real attack windows and the remaining named buckets reuse
    # the attack-positive pool with independent sampling. This keeps the harm
    # metric meaningful (degradation in attack detection) while documenting the
    # testbed limitation.
    for i, scenario in enumerate(["gnss_spoofing", "adsb_injection", "mavlink_waypoint", "jamming"]):
        windows = _sample_rows(attack, windows_per_scenario, seed + i + 1)
        probe_data[scenario] = (
            torch.tensor(windows, dtype=torch.float32),
            torch.ones(len(windows), dtype=torch.float32),
        )

    # Backdoor-trigger probe: trigger-stamped windows that are truly attacks
    # (label 1). A clean model is largely unaffected by the rare trigger; a
    # backdoored model accepts them as benign, dropping accuracy -> harm.
    trigger_base = _sample_rows(benign, windows_per_scenario, seed + 99)
    triggered = _stamp_backdoor_trigger(trigger_base)
    probe_data["backdoor_trigger"] = (
        torch.tensor(triggered, dtype=torch.float32),
        torch.ones(len(triggered), dtype=torch.float32),
    )

    return probe_data


# ---------------------------------------------------------------------------
# Regression probes (GPS/IMU streams)
# ---------------------------------------------------------------------------
def _regression_windows(
    gps_file: str,
    imu_file: str,
    config,
) -> Tuple[np.ndarray, np.ndarray]:
    from data_preprocessing_time_series import DataPreprocessorTimeSeries

    pre = DataPreprocessorTimeSeries(
        gps_file=gps_file,
        imu_file=imu_file,
        gps_features=getattr(config, "GPS_FEATURES", None),
        imu_features=getattr(config, "IMU_FEATURES", None),
        timestamp_col=getattr(config, "TIMESTAMP_COL", "TimeUS"),
        sampling_strategy=getattr(config, "SAMPLING_STRATEGY", "downsample"),
        use_labels=False,
    )
    X, y, _ = pre.preprocess(window_size=config.WINDOW_SIZE, overlap=config.OVERLAP)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build_regression_probes(
    args,
    config,
    windows_per_scenario: int,
    seed: int,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    probe_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    sources = {
        "benign": (args.benign_gps, args.benign_imu),
        "gnss_spoofing": (args.spoof_gps, args.spoof_imu),
        "mavlink_waypoint": (args.waypoint_gps, args.waypoint_imu),
    }

    for i, (scenario, (gps, imu)) in enumerate(sources.items()):
        if not gps or not imu:
            logger.info("Skipping %s (no data provided)", scenario)
            continue
        X, ytgt = _regression_windows(gps, imu, config)
        rng = np.random.default_rng(seed + i)
        idx = rng.choice(len(X), size=min(windows_per_scenario, len(X)), replace=False)
        probe_data[scenario] = (
            torch.tensor(X[idx], dtype=torch.float32),
            torch.tensor(ytgt[idx], dtype=torch.float32),
        )

    if "benign" not in probe_data:
        raise ValueError("Regression probes require --benign-gps and --benign-imu")

    return probe_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate ACFL counterfactual probe set")
    parser.add_argument(
        "--mode", choices=["classification", "regression"], default="classification"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config module (defaults: config_classification / config_regression)",
    )
    parser.add_argument(
        "--windows-per-scenario", type=int, default=48,
        help="Number of probe windows per scenario",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Output .pt path")

    # Classification source
    parser.add_argument(
        "--packet-file", default=None,
        help="Labeled packet CSV (auto-discovered under data/network_packets if omitted)",
    )

    # Regression sources
    parser.add_argument("--benign-gps", default=None)
    parser.add_argument("--benign-imu", default=None)
    parser.add_argument("--spoof-gps", default=None)
    parser.add_argument("--spoof-imu", default=None)
    parser.add_argument("--waypoint-gps", default=None)
    parser.add_argument("--waypoint-imu", default=None)

    args = parser.parse_args()

    config_module = args.config or (
        "config_classification" if args.mode == "classification" else "config_regression"
    )
    os.environ["FL_CONFIG_MODULE"] = config_module
    import importlib
    config = importlib.import_module(config_module)

    output = args.output or os.path.join(
        FL_TS_DIR, "checkpoints",
        "acfl_probes.pt" if args.mode == "classification" else "acfl_probes_reg.pt",
    )

    if args.mode == "classification":
        packet_file = args.packet_file
        if not packet_file:
            candidates = sorted(
                glob.glob(os.path.join(DRIFT_ROOT, "data", "network_packets", "*labeled*.csv"))
            )
            if not candidates:
                logger.error(
                    "No labeled packet file found. Provide --packet-file pointing "
                    "to a CSV with a 'Label' column containing attack traffic."
                )
                sys.exit(1)
            packet_file = candidates[0]
            logger.info("Auto-selected packet file: %s", packet_file)

        probe_data = build_classification_probes(
            packet_file, config, args.windows_per_scenario, args.seed
        )
        meta = {
            "mode": "classification",
            "config": config_module,
            "window_size": config.WINDOW_SIZE,
            "features": getattr(config, "PACKET_FEATURES", None),
            "source": packet_file,
            "scenarios": list(probe_data.keys()),
            "note": "Single-attack-class testbed: named attack buckets sampled "
                    "from available attack-positive packet windows.",
        }
    else:
        probe_data = build_regression_probes(
            args, config, args.windows_per_scenario, args.seed
        )
        meta = {
            "mode": "regression",
            "config": config_module,
            "window_size": config.WINDOW_SIZE,
            "features": (getattr(config, "GPS_FEATURES", []) or [])
                        + (getattr(config, "IMU_FEATURES", []) or []),
            "scenarios": list(probe_data.keys()),
        }

    _save_probe_set(probe_data, output, meta)


if __name__ == "__main__":
    main()
