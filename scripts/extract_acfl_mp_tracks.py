#!/usr/bin/env python3
"""Extract leaflet tracks.json from ACFL SITL BINs + tlogs (no ACM_TRUST).

body  ← DataFlash SIM (truth)
pos   ← DataFlash POS (EKF)
gps   ← DataFlash GPS
onset ← run_meta attack start (fallback: ATTACK:* STATUSTEXT in tlog)

Usage:
  ./acfl_env/bin/python scripts/extract_acfl_mp_tracks.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pymavlink import DFReader, mavutil


def _resolve_roots() -> tuple[Path, Path]:
    """Repo root + simulations/fl-time-series root (flat PR or nested ACFL)."""
    root = Path(__file__).resolve().parents[1]
    if (root / "fl-time-series").is_dir():
        return root, root
    nested = root / "drift-fairroad"
    if nested.is_dir():
        return root, nested
    return root, root

ACFL_ROOT, DRIFT_ROOT = _resolve_roots()
PACK = ACFL_ROOT / "results" / "demo_mission_planner"
RUNS = PACK / "runs"
SCENES = (
    "scene_c_clean",
    "scene_a_gps_spoof",
    "scene_b_waypoint_collude",
    "scene_d_fleet_collapse",
    "scene_e_reference_uav",
)
COLORS = {"uav_0": "#d62728", "uav_1": "#1f77b4", "uav_2": "#2ca02c"}
DT_S = 0.2


def _decimate(samples: List[dict], dt: float) -> List[dict]:
    if not samples:
        return []
    out = [samples[0]]
    last = samples[0]["t"]
    for s in samples[1:]:
        if s["t"] - last >= dt:
            out.append(s)
            last = s["t"]
    if out[-1] is not samples[-1]:
        out.append(samples[-1])
    return out


def _text(msg) -> str:
    t = msg.text
    if isinstance(t, bytes):
        t = t.decode("utf-8", "ignore")
    return str(t).rstrip("\x00")


def extract_bin(path: Path) -> Dict[str, Any]:
    log = DFReader.DFReader_binary(str(path))
    body: List[dict] = []
    pos: List[dict] = []
    gps: List[dict] = []
    t0: Optional[float] = None

    while True:
        msg = log.recv_msg()
        if msg is None:
            break
        ts = float(getattr(msg, "_timestamp", 0.0) or 0.0)
        if t0 is None and ts > 0:
            t0 = ts
        if t0 is None:
            continue
        t = ts - t0
        typ = msg.get_type()
        if typ == "SIM":
            # SIM has Lat/Lng/Alt (deg / m)
            body.append(
                {
                    "t": round(t, 3),
                    "lat": float(msg.Lat),
                    "lng": float(msg.Lng),
                    "alt": float(msg.Alt),
                }
            )
        elif typ == "POS":
            pos.append(
                {
                    "t": round(t, 3),
                    "lat": float(msg.Lat),
                    "lng": float(msg.Lng),
                    "alt": float(msg.Alt),
                }
            )
        elif typ == "GPS" and int(getattr(msg, "Status", 0) or 0) >= 2:
            gps.append(
                {
                    "t": round(t, 3),
                    "lat": float(msg.Lat),
                    "lng": float(msg.Lng),
                    "alt": float(msg.Alt),
                }
            )

    body_d = _decimate(body, DT_S)
    pos_d = _decimate(pos, DT_S)
    gps_d = _decimate(gps, DT_S)
    duration = 0.0
    for series in (body_d, pos_d, gps_d):
        if series:
            duration = max(duration, float(series[-1]["t"]))
    return {
        "source": path.name,
        "dt_s": DT_S,
        "duration_s": round(duration, 2),
        "n_body": len(body_d),
        "n_gps": len(gps_d),
        "n_pos": len(pos_d),
        "body": body_d,
        "gps": gps_d,
        "pos": pos_d,
    }


def onset_from_tlog(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    m = mavutil.mavlink_connection(str(path))
    t0: Optional[float] = None
    while True:
        msg = m.recv_msg()
        if msg is None:
            break
        ts = float(getattr(msg, "_timestamp", 0.0) or 0.0)
        if t0 is None and ts > 0:
            t0 = ts
        if msg.get_type() != "STATUSTEXT" or t0 is None:
            continue
        text = _text(msg)
        if text.startswith("ATTACK:"):
            return round(ts - t0, 2)
    return None


CONTROLLERS = ("fedavg", "acfl")


def build_scene(scene_id: str, controller: str = "acfl") -> dict:
    run = RUNS / scene_id / controller
    if not (run / "run_meta.json").is_file():
        run = RUNS / scene_id  # legacy
    meta = json.loads((run / "run_meta.json").read_text())
    vehicles = []
    onset = None
    duration = 0.0
    for v in meta.get("vehicles") or []:
        uid = v["uav_id"]
        bin_rel = v.get("bin")
        tlog_rel = v.get("tlog")
        if not bin_rel:
            continue
        track = extract_bin(run / bin_rel)
        if tlog_rel:
            o = onset_from_tlog(run / tlog_rel)
            if o is not None:
                onset = o if onset is None else min(onset, o)
        duration = max(duration, float(track["duration_s"]))
        role = "attacked" if v.get("attacked") else "benign"
        if v.get("recovered"):
            role += "+recovered"
        vehicles.append(
            {
                "uav_id": uid,
                "role": role,
                "bin": bin_rel,
                "track": track,
            }
        )
    for ev in meta.get("events") or []:
        if str(ev.get("action", "")).startswith("start_"):
            onset = float(ev["t_s"])
            break
    return {
        "scene_id": scene_id,
        "controller": controller,
        "title": meta.get("title"),
        "duration_s": round(duration, 2),
        "onset_s": onset,
        "note": "Extracted from ACFL SITL DataFlash (SIM=body, POS=pos).",
        "sync": "per-uav t0 = first DataFlash timestamp",
        "colors": COLORS,
        "vehicles": vehicles,
        "end_reason": meta.get("end_reason"),
        "recovered": meta.get("recovered"),
        "break_events": meta.get("break_events"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="all", choices=["all", *SCENES])
    ap.add_argument("--controller", default="both", choices=["both", *CONTROLLERS])
    args = ap.parse_args()
    scenes = SCENES if args.scene == "all" else (args.scene,)
    controllers = CONTROLLERS if args.controller == "both" else (args.controller,)
    for sid in scenes:
        for ctrl in controllers:
            run = RUNS / sid / ctrl
            if not (run / "run_meta.json").is_file():
                print(f"skip {sid}/{ctrl}: no run")
                continue
            out_dir = PACK / "leaflet" / sid / ctrl
            out_dir.mkdir(parents=True, exist_ok=True)
            data = build_scene(sid, ctrl)
            payload = json.dumps(data) + "\n"
            (out_dir / "tracks.json").write_text(payload)
            # Convenience copy: ACFL tracks also live at leaflet/<scene>/tracks.json
            # for single-pane film / legacy paths.
            if ctrl == "acfl":
                legacy = PACK / "leaflet" / sid
                legacy.mkdir(parents=True, exist_ok=True)
                (legacy / "tracks.json").write_text(payload)
            print(
                f"{sid}/{ctrl}: duration={data['duration_s']}s onset={data['onset_s']} "
                f"vehicles={len(data['vehicles'])} end={data['end_reason']}"
            )


if __name__ == "__main__":
    main()
