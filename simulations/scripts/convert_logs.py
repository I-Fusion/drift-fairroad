#!/usr/bin/env python3
"""
convert_logs.py - convert an ArduPilot dataflash (.BIN) log to GPS and IMU CSVs.

Produces CSVs whose columns match the drift-fairroad GPS/IMU schema (the same
header as the existing FL client files), so the output is a drop-in input for
``raw_data_processing/prepare_fl_data_for_run_fl.py`` and for the GPS/IMU
labeling step.

GPS header: Unnamed: 0,Timestamp,Unnamed: 2,TimeUS,GPS_ID,Status,GMS,GWk,
            Nstats,HDop,Lat,Lng,Alt,Spd,GCrs,VZ,Yaw,U
IMU header: Unnamed: 0,Timestamp,Unnamed: 2,TimeUS,I,GyrX,GyrY,GyrZ,AccX,
            AccY,AccZ,EG,EA,T,GH,AH,GHz,AHz

The ``Timestamp`` column is a real wall-clock datetime derived from the log's
GPS time (week/ms). For IMU rows (which have no GPS time) the wall clock is
interpolated from the TimeUS<->GPS-time relationship. This makes GPS/IMU rows
align with the pcap wall-clock timestamps so attack-period labeling works.

Usage:
    python3 convert_logs.py --bin run0/sitl/logs/00000001.BIN \\
        --out-gps gps.csv --out-imu imu.csv
"""
import argparse
import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pymavlink import mavutil

# GPS epoch and current leap-second offset (UTC = GPS - 18s).
GPS_EPOCH = dt.datetime(1980, 1, 6, 0, 0, 0)
LEAP_SECONDS = 18

GPS_HEADER = ["Unnamed: 0", "Timestamp", "Unnamed: 2", "TimeUS", "GPS_ID",
              "Status", "GMS", "GWk", "Nstats", "HDop", "Lat", "Lng", "Alt",
              "Spd", "GCrs", "VZ", "Yaw", "U"]
IMU_HEADER = ["Unnamed: 0", "Timestamp", "Unnamed: 2", "TimeUS", "I", "GyrX",
              "GyrY", "GyrZ", "AccX", "AccY", "AccZ", "EG", "EA", "T", "GH",
              "AH", "GHz", "AHz"]

# dataflash field -> output GPS column
GPS_FIELDS = {
    "I": "GPS_ID", "Status": "Status", "GMS": "GMS", "GWk": "GWk",
    "NSats": "Nstats", "HDop": "HDop", "Lat": "Lat", "Lng": "Lng", "Alt": "Alt",
    "Spd": "Spd", "GCrs": "GCrs", "VZ": "VZ", "Yaw": "Yaw", "U": "U",
}
IMU_FIELDS = {
    "I": "I", "GyrX": "GyrX", "GyrY": "GyrY", "GyrZ": "GyrZ",
    "AccX": "AccX", "AccY": "AccY", "AccZ": "AccZ", "EG": "EG", "EA": "EA",
    "T": "T", "GH": "GH", "AH": "AH", "GHz": "GHz", "AHz": "AHz",
}


def _gps_to_utc(gwk: float, gms: float) -> dt.datetime:
    return GPS_EPOCH + dt.timedelta(weeks=int(gwk),
                                    milliseconds=float(gms)) - dt.timedelta(seconds=LEAP_SECONDS)


def read_messages(bin_path: str, types: List[str]) -> Dict[str, List[dict]]:
    """Read selected message types from a dataflash log into dict rows."""
    mlog = mavutil.mavlink_connection(bin_path, dialect="ardupilotmega")
    out = {t: [] for t in types}
    while True:
        m = mlog.recv_match(type=types)
        if m is None:
            break
        out[m.get_type()].append(m.to_dict())
    return out


def _build_time_map(gps_rows: List[dict]):
    """Linear fit mapping TimeUS (sim us) -> UTC epoch seconds, from GPS time."""
    tus, secs = [], []
    for r in gps_rows:
        if "GWk" in r and "GMS" in r and r.get("GWk", 0):
            utc = _gps_to_utc(r["GWk"], r["GMS"])
            tus.append(float(r["TimeUS"]))
            secs.append(utc.timestamp())
    if len(tus) >= 2:
        a, b = np.polyfit(tus, secs, 1)
        return lambda t: a * float(t) + b
    if len(tus) == 1:
        offset = secs[0] - tus[0] / 1e6
        return lambda t: float(t) / 1e6 + offset
    # Fallback: treat TimeUS as seconds-since-epoch-less; use wall clock now.
    base = dt.datetime.utcnow().timestamp()
    t0 = tus[0] if tus else 0.0
    return lambda t: base + (float(t) - t0) / 1e6


def _fmt_ts(epoch_s: float) -> str:
    return dt.datetime.utcfromtimestamp(epoch_s).strftime("%Y-%m-%d %H:%M:%S.%f")


def convert(
    bin_path: str,
    out_gps: Optional[str] = None,
    out_imu: Optional[str] = None,
    instance: int = 0,
) -> Dict[str, str]:
    msgs = read_messages(bin_path, ["GPS", "IMU"])
    gps_rows = msgs.get("GPS", [])
    imu_rows = msgs.get("IMU", [])
    time_map = _build_time_map(gps_rows)

    written = {}

    if out_gps is not None:
        rows = []
        for i, r in enumerate(gps_rows):
            if instance is not None and int(r.get("I", 0)) != instance:
                continue
            ts = _fmt_ts(time_map(r["TimeUS"]))
            row = {"Unnamed: 0": i, "Timestamp": ts, "Unnamed: 2": "GPS",
                   "TimeUS": int(r["TimeUS"])}
            for fld, col in GPS_FIELDS.items():
                row[col] = r.get(fld, "")
            rows.append(row)
        df = pd.DataFrame(rows, columns=GPS_HEADER)
        df.to_csv(out_gps, index=False)
        written["gps"] = out_gps
        print(f"[convert] GPS rows (I={instance}): {len(df)} -> {out_gps}")

    if out_imu is not None:
        rows = []
        for i, r in enumerate(imu_rows):
            if instance is not None and int(r.get("I", 0)) != instance:
                continue
            ts = _fmt_ts(time_map(r["TimeUS"]))
            row = {"Unnamed: 0": i, "Timestamp": ts, "Unnamed: 2": "IMU",
                   "TimeUS": int(r["TimeUS"])}
            for fld, col in IMU_FIELDS.items():
                row[col] = r.get(fld, "")
            rows.append(row)
        df = pd.DataFrame(rows, columns=IMU_HEADER)
        df.to_csv(out_imu, index=False)
        written["imu"] = out_imu
        print(f"[convert] IMU rows (I={instance}): {len(df)} -> {out_imu}")

    return written


def main():
    ap = argparse.ArgumentParser(description="Convert dataflash .BIN to GPS/IMU CSV")
    ap.add_argument("--bin", required=True, help="path to .BIN dataflash log")
    ap.add_argument("--out-gps", default=None)
    ap.add_argument("--out-imu", default=None)
    ap.add_argument("--instance", type=int, default=0,
                    help="sensor instance to export (0=primary); -1 for all")
    args = ap.parse_args()
    inst = None if args.instance < 0 else args.instance
    if not args.out_gps and not args.out_imu:
        ap.error("provide at least one of --out-gps / --out-imu")
    convert(args.bin, args.out_gps, args.out_imu, inst)


if __name__ == "__main__":
    main()
