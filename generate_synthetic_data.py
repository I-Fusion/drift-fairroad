#!/usr/bin/env python3
"""
Generate additional drift-fairroad-format data.

The drift-fairroad dataset is small (a couple of mission logs). This script
synthesizes additional, statistically-faithful data using the *same method* the
drift-fairroad pipeline relies on:

  * packets are labeled per contiguous attack *time period* (as in
    raw_data_processing/label_csv_with_periods.py);
  * the column schema, feature value ranges, and timing characteristics are
    learned from a real template file and preserved exactly.

Rather than inventing values, it uses **block bootstrap** (sampling contiguous
segments of real rows per class) plus light Gaussian jitter on the continuous
features, so local sequence structure (bursts, correlations) is retained while
producing a larger, more varied dataset. Output files are drop-in inputs for
``raw_data_processing/prepare_fl_data_for_run_fl.py``.

Examples
--------
Packets (classification), 5x more rows, keep the template's attack ratio:

    python3 generate_synthetic_data.py --mode packets \\
        --template data/network_packets/mission_2_wp_23_attack_add_wp_5_alt_0005_labeled.csv \\
                   data/network_packets/mission_2_wp_23_attack_add_wp_7_alt_0001_labeled.csv \\
        --num-rows 120000 \\
        --output data/network_packets/synthetic_packets_labeled.csv \\
        --prepare-output data/train/packets --n-clients 3

GPS + IMU (regression):

    python3 generate_synthetic_data.py --mode gps-imu \\
        --gps-template data/train/gps-imu/gps_client_000.csv \\
        --imu-template data/train/gps-imu/imu_client_000.csv \\
        --num-rows 8000 \\
        --output-gps data/gps_spoofing/synthetic_gps.csv \\
        --output-imu data/gps_spoofing/synthetic_imu.csv
"""
import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ACFL_ROOT = os.path.dirname(os.path.abspath(__file__))
_nested = os.path.join(ACFL_ROOT, "drift-fairroad")
DRIFT_ROOT = _nested if os.path.isdir(os.path.join(_nested, "raw_data_processing")) else ACFL_ROOT
RAW_PROC_DIR = os.path.join(DRIFT_ROOT, "raw_data_processing")

# Continuous feature columns that are jittered; categorical-like columns
# (ports, message/protocol ids) are kept verbatim from the bootstrapped rows.
PACKET_CONTINUOUS = ["Length"]
GPS_CONTINUOUS = ["Lat", "Lng", "Alt", "Spd", "GCrs", "VZ"]
IMU_CONTINUOUS = ["GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _block_bootstrap(source: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``length`` rows by sampling a contiguous block of ``source``.

    If the source is shorter than ``length`` it is tiled (wrapped) so local
    ordering is preserved as much as possible.
    """
    n = len(source)
    if n == 0:
        raise ValueError("Empty source pool for bootstrap")
    if n >= length:
        start = int(rng.integers(0, n - length + 1))
        return source[start : start + length].copy()
    reps = int(np.ceil(length / n))
    tiled = np.tile(source, (reps, 1))
    start = int(rng.integers(0, len(tiled) - length + 1))
    return tiled[start : start + length].copy()


def _jitter(values: np.ndarray, std: float, rng: np.random.Generator,
            non_negative: bool = False) -> np.ndarray:
    """Add Gaussian noise scaled to each column's std."""
    values = values.astype(np.float64)
    col_std = np.nanstd(values, axis=0, keepdims=True)
    noise = rng.standard_normal(values.shape) * (col_std * std)
    out = values + noise
    if non_negative:
        out = np.clip(out, 0.0, None)
    return out


def _build_attack_period_mask(
    n: int,
    attack_ratio: float,
    num_periods: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Boolean mask (length n) with ``num_periods`` contiguous attack windows.

    Mirrors the drift-fairroad labeling method where attacks occur during
    distinct contiguous time periods rather than scattered rows.
    """
    mask = np.zeros(n, dtype=bool)
    target_attack = int(round(attack_ratio * n))
    if target_attack <= 0 or num_periods <= 0:
        return mask

    num_periods = max(1, min(num_periods, target_attack))
    # Split the attack budget into per-period lengths (roughly equal + noise).
    base = target_attack // num_periods
    lengths = [base] * num_periods
    for i in range(target_attack - base * num_periods):
        lengths[i % num_periods] += 1

    # Place non-overlapping periods in order across the timeline.
    gaps = n - target_attack
    if gaps < 0:
        mask[:] = True
        return mask
    # Distribute gaps before/between/after periods.
    cut_points = sorted(rng.integers(0, gaps + 1, size=num_periods))
    cursor = 0
    prev_cut = 0
    for length, cut in zip(lengths, cut_points):
        cursor += (cut - prev_cut)  # gap before this period
        start = cursor
        end = min(start + length, n)
        mask[start:end] = True
        cursor = end
        prev_cut = cut
    return mask


def _regen_timestamps_datetime(
    template_ts: pd.Series, n: int, rng: np.random.Generator
) -> List[str]:
    """Regenerate n monotonic datetime-string timestamps from template gaps."""
    parsed = pd.to_datetime(template_ts, errors="coerce").dropna()
    diffs = parsed.diff().dropna().dt.total_seconds().to_numpy()
    diffs = diffs[(diffs > 0) & (diffs < np.percentile(diffs, 99.5))]
    if len(diffs) == 0:
        diffs = np.array([0.005])
    sampled = rng.choice(diffs, size=n - 1, replace=True)
    offsets = np.concatenate([[0.0], np.cumsum(sampled)])
    start = parsed.iloc[0]
    times = start + pd.to_timedelta(offsets, unit="s")
    return [t.strftime("%Y-%m-%d %H:%M:%S.%f") for t in times]


def _regen_timestamps_int(
    template_ts: pd.Series, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Regenerate n monotonic integer (microsecond) timestamps from gaps."""
    ts = pd.to_numeric(template_ts, errors="coerce").dropna().to_numpy()
    diffs = np.diff(ts)
    diffs = diffs[(diffs > 0) & (diffs < np.percentile(diffs, 99.5))]
    if len(diffs) == 0:
        diffs = np.array([200000])  # 0.2s fallback
    sampled = rng.choice(diffs, size=n - 1, replace=True)
    start = int(ts[0]) if len(ts) else 0
    return (start + np.concatenate([[0], np.cumsum(sampled)])).astype(np.int64)


# ---------------------------------------------------------------------------
# Packets (classification)
# ---------------------------------------------------------------------------
def generate_packets(args, rng: np.random.Generator) -> pd.DataFrame:
    frames = [pd.read_csv(t) for t in args.template]
    df = pd.concat(frames, ignore_index=True)

    label_col = args.label_col
    ts_col = args.timestamp_col
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in template")

    feature_cols = [c for c in df.columns if c not in (label_col, ts_col)]
    normal = df[df[label_col] == 0]
    attack = df[df[label_col] == 1]
    print(f"Template: {len(df)} rows ({len(normal)} normal / {len(attack)} attack)")
    if len(normal) == 0 or len(attack) == 0:
        raise ValueError("Template must contain both normal and attack rows")

    n = args.num_rows
    attack_ratio = args.attack_ratio if args.attack_ratio is not None else len(attack) / len(df)
    num_periods = args.num_attack_periods
    print(f"Generating {n} rows, attack_ratio={attack_ratio:.4f}, "
          f"attack_periods={num_periods}")

    mask = _build_attack_period_mask(n, attack_ratio, num_periods, rng)

    normal_arr = normal[feature_cols].to_numpy()
    attack_arr = attack[feature_cols].to_numpy()

    # Fill the timeline period-by-period using contiguous blocks per class.
    out_rows = np.empty((n, len(feature_cols)), dtype=object)
    i = 0
    while i < n:
        j = i
        while j < n and mask[j] == mask[i]:
            j += 1
        seg_len = j - i
        pool = attack_arr if mask[i] else normal_arr
        out_rows[i:j] = _block_bootstrap(pool, seg_len, rng)
        i = j

    out = pd.DataFrame(out_rows, columns=feature_cols)

    # Jitter continuous features; coerce numeric columns back to numbers.
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    cont = [c for c in PACKET_CONTINUOUS if c in feature_cols]
    if cont:
        out[cont] = _jitter(out[cont].to_numpy(), args.jitter, rng, non_negative=True)
        out[cont] = out[cont].round(1)

    # Keep categorical-like columns (ports, MsgID, Protocol) as integers so the
    # output matches the template formatting exactly.
    for col in feature_cols:
        if col not in cont:
            out[col] = out[col].round().astype("int64")

    out[ts_col] = _regen_timestamps_datetime(df[ts_col], n, rng)
    out[label_col] = mask.astype(int)

    out = out[[ts_col] + feature_cols + [label_col]]
    pos = int(out[label_col].sum())
    print(f"Generated packets: {len(out)} rows, {pos} attack ({pos/len(out)*100:.2f}%)")
    return out


# ---------------------------------------------------------------------------
# GPS / IMU (regression)
# ---------------------------------------------------------------------------
def generate_sensor(
    template_path: str,
    continuous_cols: List[str],
    ts_col: str,
    n: int,
    jitter: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = pd.read_csv(template_path)
    cols = list(df.columns)
    present_cont = [c for c in continuous_cols if c in cols]
    if not present_cont:
        raise ValueError(
            f"None of the continuous columns {continuous_cols} found in "
            f"{template_path} (columns: {cols})"
        )

    source = df.to_numpy()
    block = _block_bootstrap(source, n, rng)
    out = pd.DataFrame(block, columns=cols)

    # Numeric coercion + jitter on sensor features only.
    for c in present_cont:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[present_cont] = _jitter(out[present_cont].to_numpy(), jitter, rng)

    # Regenerate monotonic TimeUS so windowing/sorting stays valid.
    if ts_col in cols:
        out[ts_col] = _regen_timestamps_int(df[ts_col], n, rng)

    print(f"Generated {os.path.basename(template_path)} -> {len(out)} rows "
          f"({len(present_cont)} jittered features)")
    return out


# ---------------------------------------------------------------------------
# Optional inline client splitting (reuses drift-fairroad's own method)
# ---------------------------------------------------------------------------
def prepare_clients_packets(packet_csv: str, output_dir: str, n_clients: int):
    if RAW_PROC_DIR not in sys.path:
        sys.path.insert(0, RAW_PROC_DIR)
    from prepare_fl_data_for_run_fl import (
        load_packet_data, split_among_clients, save_client_files,
        create_config_template,
    )
    packet_df = load_packet_data(packet_csv, "Timestamp")
    client_data = split_among_clients(
        packet_df=packet_df, n_clients=n_clients, packet_timestamp_col="Timestamp"
    )
    mapping = save_client_files(client_data, output_dir)
    create_config_template(mapping, output_dir)
    print(f"\nClient splits written to {output_dir}")


def prepare_clients_gps_imu(gps_csv, imu_csv, output_dir, n_clients):
    if RAW_PROC_DIR not in sys.path:
        sys.path.insert(0, RAW_PROC_DIR)
    from prepare_fl_data_for_run_fl import (
        load_gps_data, load_imu_data, merge_gps_imu, split_among_clients,
        save_client_files, create_config_template,
    )
    gps_df = load_gps_data(gps_csv, "TimeUS")
    imu_df = load_imu_data(imu_csv, "TimeUS")
    merged = merge_gps_imu(gps_df, imu_df, timestamp_col="TimeUS")
    client_data = split_among_clients(
        gps_df=gps_df, imu_df=imu_df, merged_df=merged,
        n_clients=n_clients, timestamp_col="TimeUS",
    )
    mapping = save_client_files(client_data, output_dir)
    create_config_template(mapping, output_dir)
    print(f"\nClient splits written to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate drift-fairroad-format data")
    parser.add_argument("--mode", choices=["packets", "gps-imu"], default="packets")
    parser.add_argument("--num-rows", type=int, default=120000,
                        help="Number of rows to generate")
    parser.add_argument("--jitter", type=float, default=0.05,
                        help="Gaussian jitter std as a fraction of each feature's std")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-clients", type=int, default=3)

    # Packet mode
    parser.add_argument("--template", nargs="+", default=None,
                        help="One or more labeled packet CSV templates")
    parser.add_argument("--timestamp-col", default="Timestamp")
    parser.add_argument("--label-col", default="Label")
    parser.add_argument("--attack-ratio", type=float, default=None,
                        help="Target attack fraction (default: template's ratio)")
    parser.add_argument("--num-attack-periods", type=int, default=8)
    parser.add_argument("--output", default=None, help="Output labeled packet CSV")
    parser.add_argument("--prepare-output", default=None,
                        help="If set, also write client splits to this directory")

    # GPS/IMU mode
    parser.add_argument("--gps-template", default=None)
    parser.add_argument("--imu-template", default=None)
    parser.add_argument("--output-gps", default=None)
    parser.add_argument("--output-imu", default=None)

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.mode == "packets":
        if not args.template:
            parser.error("--template is required for --mode packets")
        out = generate_packets(args, rng)
        output = args.output or os.path.join(
            DRIFT_ROOT, "data", "network_packets", "synthetic_packets_labeled.csv"
        )
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        out.to_csv(output, index=False)
        print(f"Saved synthetic packets to {output}")

        if args.prepare_output:
            prepare_clients_packets(output, args.prepare_output, args.n_clients)
        else:
            print("\nTo split into FL clients, run:")
            print(f"  python3 raw_data_processing/"
                  f"prepare_fl_data_for_run_fl.py --packet-file {output} "
                  f"--output data/train/packets")

    else:  # gps-imu
        if not args.gps_template or not args.imu_template:
            parser.error("--gps-template and --imu-template are required for --mode gps-imu")
        gps = generate_sensor(args.gps_template, GPS_CONTINUOUS, "TimeUS",
                              args.num_rows, args.jitter, rng)
        # IMU usually sampled faster than GPS; keep proportional length.
        imu_rows = args.num_rows * 5
        imu = generate_sensor(args.imu_template, IMU_CONTINUOUS, "TimeUS",
                             imu_rows, args.jitter, rng)

        out_gps = args.output_gps or os.path.join(
            DRIFT_ROOT, "data", "gps_spoofing", "synthetic_gps.csv")
        out_imu = args.output_imu or os.path.join(
            DRIFT_ROOT, "data", "gps_spoofing", "synthetic_imu.csv")
        os.makedirs(os.path.dirname(out_gps) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(out_imu) or ".", exist_ok=True)
        gps.to_csv(out_gps, index=False)
        imu.to_csv(out_imu, index=False)
        print(f"Saved synthetic GPS to {out_gps}")
        print(f"Saved synthetic IMU to {out_imu}")

        if args.prepare_output:
            prepare_clients_gps_imu(out_gps, out_imu, args.prepare_output, args.n_clients)
        else:
            print("\nTo split into FL clients, run:")
            print(f"  python3 raw_data_processing/"
                  f"prepare_fl_data_for_run_fl.py --gps-file {out_gps} "
                  f"--imu-file {out_imu} --output data/train/gps-imu")


if __name__ == "__main__":
    main()
