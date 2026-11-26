#!/usr/bin/env python3
"""
packet_pipeline.py - turn a captured pcap + dataflash CSVs into labeled data.

Reproduces drift-fairroad's network-data pipeline, parametrized by the attacker
port so it works for any SITL instance:

  1. Parse the pcap into MAVLink packet rows
     [Timestamp, SrcPort, DstPort, Length, MsgID, Protocol]  (like
     raw_data_processing/packets_analysis_pyshark_tcp_5762.py).
  2. Derive attack time-periods from the *attacker-port* subset using the
     existing find_time_periods.find_time_periods.
  3. Label the packet rows (excluding attacker-port packets) by those periods
     -> packets_labeled.csv  (Timestamp,SrcPort,DstPort,Length,MsgID,Protocol,Label).
  4. Label GPS/IMU CSVs (from convert_logs.py) by the same periods via their
     wall-clock Timestamp -> *_labeled.csv.

Packet parsing needs system tshark (installed by setup_sitl.sh); pyshark wraps it.
"""
import argparse
import os
import struct
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

# Reuse the generic period detector from the raw_data_processing package.
RAW_PROC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "raw_data_processing",
)
if RAW_PROC_DIR not in sys.path:
    sys.path.insert(0, RAW_PROC_DIR)
from find_time_periods import find_time_periods  # noqa: E402


# ---------------------------------------------------------------------------
# MAVLink 2 framing (same layout as the reference pyshark parser)
# ---------------------------------------------------------------------------
class _Mavlink2:
    def __init__(self, pkt_buf: bytes):
        msg_len = pkt_buf[1]
        unpacked = struct.unpack(f"BBBBBBB3s{msg_len}s2s", pkt_buf[: 12 + msg_len])
        self.magic, self.len = unpacked[0], unpacked[1]
        self.seq, self.sysid, self.compid = unpacked[4], unpacked[5], unpacked[6]
        msgid = bytearray(unpacked[7])
        msgid.reverse()
        self.msgid = int(msgid.hex(), 16)


def _parse_msgid_from_tcp(raw_hex: str) -> Optional[int]:
    try:
        raw = bytes.fromhex(raw_hex.replace(":", ""))
    except ValueError:
        return None
    start = raw.find(0xFD)
    if start == -1:
        return None
    try:
        return _Mavlink2(raw[start:]).msgid
    except Exception:
        return None


def parse_pcap_to_npy(pcap_path: str, npy_path: str) -> str:
    """Parse a pcap into a 6-column packet array and save as .npy.

    Columns: Timestamp, SrcPort, DstPort, Length, MsgID, Protocol (1=TCP, 0=UDP).
    """
    import pyshark  # imported lazily so the rest of the module loads without tshark

    rows = []
    cap = pyshark.FileCapture(pcap_path, keep_packets=False)
    try:
        for pkt in cap:
            try:
                ts = pkt.sniff_time
                if hasattr(pkt, "tcp"):
                    src = int(getattr(pkt.tcp, "srcport", 0))
                    dst = int(getattr(pkt.tcp, "dstport", 0))
                    length = int(float(getattr(pkt.tcp, "len", 0) or 0))
                    msgid = None
                    if hasattr(pkt.tcp, "payload"):
                        msgid = _parse_msgid_from_tcp(str(pkt.tcp.payload))
                    rows.append([ts, src, dst, length, msgid if msgid is not None else -1, 1])
                elif hasattr(pkt, "udp"):
                    src = int(getattr(pkt.udp, "srcport", 0))
                    dst = int(getattr(pkt.udp, "dstport", 0))
                    length = int(float(getattr(pkt.udp, "length", 0) or 0))
                    rows.append([ts, src, dst, length, -1, 0])
            except (AttributeError, ValueError):
                continue
    finally:
        cap.close()

    arr = np.array(rows, dtype=object)
    np.save(npy_path, arr)
    print(f"[packets] parsed {len(arr)} packets -> {npy_path}")
    return npy_path


# ---------------------------------------------------------------------------
# Attack periods + labeling
# ---------------------------------------------------------------------------
def attack_periods(npy_path: str, attack_port: int, gap_threshold_s: float = 1.0):
    """Detect attack time-periods from the attacker-port packet subset."""
    arr = np.load(npy_path, allow_pickle=True)
    if len(arr) == 0:
        return []
    df = pd.DataFrame(arr, columns=["Timestamp", "SrcPort", "DstPort", "Length", "MsgID", "Protocol"])
    df["SrcPort"] = pd.to_numeric(df["SrcPort"], errors="coerce")
    df["DstPort"] = pd.to_numeric(df["DstPort"], errors="coerce")
    sub = df[(df["SrcPort"] == attack_port) | (df["DstPort"] == attack_port)]
    if len(sub) == 0:
        print(f"[periods] no packets on attacker port {attack_port}; no attack periods")
        return []
    tmp = npy_path.replace(".npy", "_attackport.npy")
    np.save(tmp, sub.to_numpy())
    periods, _ = find_time_periods(tmp, gap_threshold_seconds=gap_threshold_s)
    print(f"[periods] {len(periods)} attack period(s) on port {attack_port}")
    return periods


def _in_periods(ts, periods) -> int:
    for p in periods:
        if p["start_time"] <= ts <= p["end_time"]:
            return 1
    return 0


def label_packets(npy_path: str, periods, attack_port: int, csv_out: str) -> str:
    """Label packets by attack periods, dropping attacker-port packets."""
    arr = np.load(npy_path, allow_pickle=True)
    df = pd.DataFrame(arr, columns=["Timestamp", "SrcPort", "DstPort", "Length", "MsgID", "Protocol"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    for c in ["SrcPort", "DstPort", "Length", "MsgID", "Protocol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop attacker-port rows (the labeling key, not a model feature).
    df = df[(df["SrcPort"] != attack_port) & (df["DstPort"] != attack_port)].reset_index(drop=True)
    df["Label"] = [_in_periods(ts, periods) for ts in df["Timestamp"]]
    df.to_csv(csv_out, index=False)
    pos = int(df["Label"].sum())
    print(f"[label] packets: {len(df)} rows, {pos} attack ({pos/max(1,len(df))*100:.2f}%) -> {csv_out}")
    return csv_out


def label_sensor_csv(csv_in: str, periods, csv_out: str,
                     timestamp_col: str = "Timestamp") -> str:
    """Add a Label column to a GPS/IMU CSV by matching its wall-clock Timestamp."""
    df = pd.read_csv(csv_in)
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    df["Label"] = [0 if pd.isna(t) else _in_periods(t, periods) for t in ts]
    df.to_csv(csv_out, index=False)
    pos = int(df["Label"].sum())
    print(f"[label] {os.path.basename(csv_in)}: {len(df)} rows, {pos} attack -> {csv_out}")
    return csv_out


def label_sensor_csv_by_timeus(csv_in: str, timeus_periods, csv_out: str,
                               timeus_col: str = "TimeUS") -> str:
    """Add a Label column to a GPS/IMU CSV by matching the boot-time TimeUS.

    ``timeus_periods`` is a list of (start_us, end_us) tuples. This is the
    speedup/timezone-robust labeling path: TimeUS is the dataflash's own clock.
    """
    df = pd.read_csv(csv_in)
    tus = pd.to_numeric(df[timeus_col], errors="coerce")

    def _lab(v):
        if pd.isna(v):
            return 0
        for s, e in timeus_periods:
            if s <= v <= e:
                return 1
        return 0

    df["Label"] = [_lab(v) for v in tus]
    df.to_csv(csv_out, index=False)
    pos = int(df["Label"].sum())
    print(f"[label] {os.path.basename(csv_in)}: {len(df)} rows, {pos} attack -> {csv_out}")
    return csv_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="pcap + sensor CSV labeling pipeline")
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--attack-port", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gps-csv", default=None, help="GPS CSV from convert_logs.py")
    ap.add_argument("--imu-csv", default=None, help="IMU CSV from convert_logs.py")
    ap.add_argument("--gap-threshold", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    npy = os.path.join(args.out_dir, "packets.npy")
    parse_pcap_to_npy(args.pcap, npy)
    periods = attack_periods(npy, args.attack_port, args.gap_threshold)

    label_packets(npy, periods, args.attack_port,
                  os.path.join(args.out_dir, "packets_labeled.csv"))
    if args.gps_csv:
        label_sensor_csv(args.gps_csv, periods,
                         os.path.join(args.out_dir, "gps_labeled.csv"))
    if args.imu_csv:
        label_sensor_csv(args.imu_csv, periods,
                         os.path.join(args.out_dir, "imu_labeled.csv"))


if __name__ == "__main__":
    main()
