# -*- coding: utf-8 -*-
"""
Generate the attack-onset-file (JSON) required by evaluate_fl_system.py from the
output of find_time_periods.py.

Uses a single pre-split packet CSV so timestamps match time_periods from the same
source. Finds global window indices where each attack period starts, then maps
them to per-client onset indices using the same splitting logic as
prepare_fl_data_for_run_fl.py (consecutive row split + equalize to min_packets).

Output format: {"client_1": [0, 42, ...], "client_2": [10, ...], "client_3": [...]}

Usage (from project root):

  python raw_data_processing/generate_attack_onset_file.py ^
    --packet-file path/to/mission_xxx_labeled.csv ^
    --time-periods path/to/mission_xxx_time_periods.csv ^
    --output data/validate/attack_onset.json
"""

import os
import sys
import json
import argparse
import pandas as pd

# Add project root and fl-classification for config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FL_CLASS_DIR = os.path.join(PROJECT_ROOT, "fl-classification")
if FL_CLASS_DIR not in sys.path:
    sys.path.insert(0, FL_CLASS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    """
    Parse timestamps: numeric microseconds since epoch (packet CSV from same
    source as find_time_periods) or datetime strings. Returns timezone-naive.
    """
    sample = series.dropna().iloc[0] if len(series.dropna()) else None
    if sample is None:
        return pd.to_datetime(series, errors="coerce")
    if isinstance(sample, (int, float)) or (isinstance(sample, str) and sample.replace(".", "").replace("-", "").isdigit()):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.dropna().max() > 1e18:
            out = pd.to_datetime(numeric, unit="ns", errors="coerce")
        else:
            out = pd.to_datetime(numeric, unit="us", errors="coerce")
    else:
        out = pd.to_datetime(series, errors="coerce")
    if out.dt.tz is not None:
        out = out.dt.tz_localize(None)
    return out


def load_packet_df(packet_file: str, timestamp_col: str = "Timestamp"):
    """Load packet CSV, sort by timestamp, dropna."""
    df = pd.read_csv(packet_file)
    if timestamp_col not in df.columns:
        for c in ["Timestamp", "timestamp", "TimeUS", "time"]:
            if c in df.columns:
                timestamp_col = c
                break
        else:
            timestamp_col = df.columns[0]
    df[timestamp_col] = _parse_timestamp_series(df[timestamp_col])
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)
    return df, timestamp_col


def window_bounds(num_samples: int, window_size: int, overlap: int):
    """Yield (w, start_row, end_row) for each window."""
    stride = window_size - overlap
    num_windows = (num_samples - window_size) // stride
    for w in range(num_windows):
        start_row = w * stride
        end_row = start_row + window_size
        yield w, start_row, end_row


def get_attack_start_times(time_periods_path: str, attack_periods: list = None):
    """Load attack period start times from CSV or JSON. Returns list of naive Timestamps."""
    start_times = []
    path_lower = time_periods_path.lower()

    if path_lower.endswith(".csv"):
        df = pd.read_csv(time_periods_path)
        if "start_time" not in df.columns:
            raise ValueError(f"CSV must have a 'start_time' column. Found: {list(df.columns)}")
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["start_time"])
        if attack_periods is not None:
            if "period" in df.columns:
                df = df[df["period"].astype(int).isin(attack_periods)]
            else:
                df = df.iloc[[i - 1 for i in attack_periods if 1 <= i <= len(df)]]
        start_times = df["start_time"].tolist()
    elif path_lower.endswith(".json"):
        with open(time_periods_path) as f:
            data = json.load(f)
        if "attack_onset_timestamps" in data:
            start_times = [
                (lambda x: x.tz_convert("UTC").tz_localize(None) if x.tz else x)(pd.Timestamp(t))
                for t in data["attack_onset_timestamps"]
            ]
        elif "periods" in data:
            for p in data["periods"]:
                if attack_periods is not None and "period" in p and p["period"] not in attack_periods:
                    continue
                t = p.get("start_time") or p.get("start")
                if t is not None:
                    x = pd.Timestamp(t)
                    start_times.append(x.tz_convert("UTC").tz_localize(None) if x.tz else x)
        else:
            raise ValueError("JSON must contain 'attack_onset_timestamps' or 'periods' with start_time/start")
    else:
        raise ValueError("time_periods_path must be a .csv or .json file")
    return start_times


def compute_global_onset_indices(
    df: pd.DataFrame,
    ts_col: str,
    time_periods_path: str,
    window_size: int,
    overlap: int,
    attack_periods: list = None,
) -> list:
    """Return sorted list of global window indices (0-based) that contain an attack start."""
    num_samples = len(df)
    attack_starts = get_attack_start_times(time_periods_path, attack_periods)
    if not attack_starts:
        return []

    stride = window_size - overlap
    num_windows = (num_samples - window_size) // stride
    if num_windows <= 0:
        return []

    ts_series = df[ts_col]
    onset_indices = []
    for t_attack in attack_starts:
        for w, start_row, end_row in window_bounds(num_samples, window_size, overlap):
            if end_row > len(ts_series):
                break
            t_start = ts_series.iloc[start_row]
            t_end = ts_series.iloc[end_row - 1]
            if t_start <= t_attack <= t_end:
                onset_indices.append(w)
                break
    return sorted(set(onset_indices))


def client_row_ranges(n_samples: int, n_clients: int):
    """
    Replicate prepare_fl_data_for_run_fl split_among_clients row assignment.
    Returns list of (start_row, end_row) per client (0-based, exclusive end).
    After equalization each client gets min_packets rows: [start, start+min_packets).
    """
    samples_per_client = n_samples // n_clients
    remainder = n_samples % n_clients
    ranges = []
    for client_id in range(n_clients):
        start_idx = client_id * samples_per_client + min(client_id, remainder)
        if client_id < remainder:
            end_idx = start_idx + samples_per_client + 1
        else:
            end_idx = start_idx + samples_per_client
        ranges.append((start_idx, end_idx))
    # Equalize: each client keeps only first min_packets of their segment (same as prepare_fl_data)
    lengths = [end - start for start, end in ranges]
    min_packets = min(lengths)
    return [(start, start + min_packets) for start, _ in ranges]


def global_window_range_for_client(r0: int, r1: int, window_size: int, stride: int):
    """
    Client has rows [r0, r1). Return (w_min, w_max) global window indices that belong to this client.
    A window w belongs to client if its start row w*stride is in [r0, r1) and window fits: w*stride+window_size <= r1.
    """
    # First w with w*stride >= r0
    w_min = (r0 + stride - 1) // stride if r0 > 0 else 0
    # Last w with w*stride + window_size <= r1
    w_max = (r1 - window_size) // stride
    if w_max < w_min:
        return (w_min, w_min - 1)  # no full window inside
    return (w_min, w_max)


def map_global_onsets_to_per_client(
    global_onsets: list,
    n_samples: int,
    n_clients: int,
    window_size: int,
    overlap: int,
) -> dict:
    """
    Map global window onset indices to per-client local indices using the same
    row split and equalization as prepare_fl_data_for_run_fl.
    Returns {"client_1": [...], "client_2": [...], ...}.
    """
    stride = window_size - overlap
    client_ranges = client_row_ranges(n_samples, n_clients)
    out = {}
    for c in range(n_clients):
        r0, r1 = client_ranges[c]
        w_min, w_max = global_window_range_for_client(r0, r1, window_size, stride)
        local_onsets = [g - w_min for g in global_onsets if w_min <= g <= w_max]
        out[f"client_{c + 1}"] = sorted(set(local_onsets))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate attack_onset JSON from find_time_periods output using a single pre-split packet file, then map to per-client indices."
    )
    parser.add_argument(
        "--packet-file", "-p",
        type=str,
        required=True,
        help="Path to a single packet CSV (pre-split; same source as time_periods for consistent timestamps).",
    )
    parser.add_argument(
        "--time-periods", "-t",
        type=str,
        required=True,
        help="CSV from find_time_periods (start_time, end_time) or JSON with periods / attack_onset_timestamps.",
    )
    parser.add_argument(
        "--output", "-o",
        default="attack_onset.json",
        help="Output JSON path (default: attack_onset.json).",
    )
    parser.add_argument(
        "--n-clients", "-n",
        type=int,
        default=3,
        help="Number of clients for split (must match prepare_fl_data, default: 3).",
    )
    parser.add_argument(
        "--config",
        default="config_classification",
        help="Config module for WINDOW_SIZE, OVERLAP (default: config_classification).",
    )
    parser.add_argument(
        "--attack-periods",
        type=str,
        default=None,
        help="Comma-separated period numbers to use as attacks (e.g. 2,4). Default: all periods.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override config WINDOW_SIZE.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Override config OVERLAP.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print global onsets and per-client mapping.",
    )
    args = parser.parse_args()

    # Config
    try:
        config = __import__(args.config)
    except ImportError:
        window_size = args.window_size or 50
        overlap = args.overlap or 25
        timestamp_col = "Timestamp"
    else:
        window_size = args.window_size or getattr(config, "WINDOW_SIZE", 50)
        overlap = args.overlap or getattr(config, "OVERLAP", 25)
        timestamp_col = getattr(config, "PACKET_TIMESTAMP_COL", "Timestamp")

    attack_periods = None
    if args.attack_periods:
        attack_periods = [int(x.strip()) for x in args.attack_periods.split(",")]

    # Load single packet file
    df, ts_col = load_packet_df(args.packet_file, timestamp_col)
    n_samples = len(df)

    # Global onset indices (all windows; no test split)
    global_onsets = compute_global_onset_indices(
        df=df,
        ts_col=ts_col,
        time_periods_path=args.time_periods,
        window_size=window_size,
        overlap=overlap,
        attack_periods=attack_periods,
    )

    if args.verbose:
        print(f"  Global onset indices: {global_onsets}")

    # Map to per-client using prepare_fl_data split logic
    out = map_global_onsets_to_per_client(
        global_onsets=global_onsets,
        n_samples=n_samples,
        n_clients=args.n_clients,
        window_size=window_size,
        overlap=overlap,
    )

    for k, v in out.items():
        print(f"  {k}: {len(v)} onset(s)")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote to {args.output}")


if __name__ == "__main__":
    main()
