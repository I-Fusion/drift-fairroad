#!/usr/bin/env python3
"""Merge per-UAV ACFL demo tlogs into dual-track Mission Planner files.

For each vehicle index i (sysid = i+1):
  * EKF track  → keep / rewrite messages as sysid 1/2/3
  * SIM truth  → convert SIM_STATE → GLOBAL_POSITION_INT on sysid 11/12/13
                 (Mission Planner maps GPI, not SIM_STATE)

Also injects:
  * NAMED_VALUE_FLOAT metrics (~1 Hz): EKF_SIM_M, MISS_CNT, HOLD_CNT, DET_CNT,
    FAVG_SCR, ACFL_SCR — visible in MP Status / Quick / HUD User Items / Tuning
  * MISSION_COUNT + MISSION_ITEM_INT for the demo waypoint file (and copies
    ``mission.waypoints`` beside the tlog for Flight Plan → Load WP)

Layout:
  results/demo_mission_planner/runs/<scene>/<controller>/sitl/uav_*/telemetry.tlog
  → mission_planner/<scene>/<controller>/multivehicle_sync.tlog
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pymavlink import mavutil


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
RUNS = ACFL_ROOT / "results" / "demo_mission_planner" / "runs"
OUT = ACFL_ROOT / "results" / "demo_mission_planner" / "mission_planner"
MISSION = DRIFT_ROOT / "simulations" / "missions" / "mission_2_wp_23.waypoints"
CONTROLLERS = ("fedavg", "acfl")
SCENES = (
    "scene_c_clean",
    "scene_a_gps_spoof",
    "scene_b_waypoint_collude",
    "scene_d_fleet_collapse",
    "scene_e_reference_uav",
)

Row = Tuple[float, bytes, int]  # ts, buf, sysid
METRIC_DT_S = 1.0


def _repack(msg, sysid: int) -> Optional[bytes]:
    try:
        mav = mavutil.mavlink.MAVLink(None)
        mav.srcSystem = int(sysid)
        mav.srcComponent = int(msg.get_srcComponent() or 1)
        return bytes(msg.pack(mav))
    except Exception:
        return None


def _pack_sysid(mav, msg, sysid: int, compid: int = 1) -> bytes:
    old_s, old_c = mav.srcSystem, mav.srcComponent
    mav.srcSystem = int(sysid)
    mav.srcComponent = int(compid)
    try:
        return bytes(msg.pack(mav))
    finally:
        mav.srcSystem, mav.srcComponent = old_s, old_c


def _sim_to_gpi(msg, sysid: int) -> Optional[bytes]:
    """SIM_STATE → GLOBAL_POSITION_INT so MP draws the truth icon."""
    try:
        mav = mavutil.mavlink.MAVLink(None)
        mav.srcSystem = int(sysid)
        mav.srcComponent = 1
        lat_i = int(getattr(msg, "lat_int", 0) or 0)
        lon_i = int(getattr(msg, "lon_int", 0) or 0)
        if lat_i == 0 and lon_i == 0:
            lat_f = float(msg.lat)
            lon_f = float(msg.lon)
            if abs(lat_f) <= 90 and abs(lon_f) <= 180:
                lat_i = int(lat_f * 1e7)
                lon_i = int(lon_f * 1e7)
            else:
                lat_i = int(lat_f)
                lon_i = int(lon_f)
        alt_mm = int(float(msg.alt) * 1000.0)
        vn = float(getattr(msg, "vn", 0.0) or 0.0)
        ve = float(getattr(msg, "ve", 0.0) or 0.0)
        vd = float(getattr(msg, "vd", 0.0) or 0.0)
        gpi = mav.global_position_int_encode(
            0,
            lat_i,
            lon_i,
            alt_mm,
            alt_mm,
            int(round(vn * 100.0)),
            int(round(ve * 100.0)),
            int(round(vd * 100.0)),
            65535,
        )
        return bytes(gpi.pack(mav))
    except Exception:
        return None


def _named_float(sysid: int, name: str, value: float, boot_ms: int = 0) -> bytes:
    mav = mavutil.mavlink.MAVLink(None)
    # MAVLink name field is 10 chars
    nm = (name or "")[:10]
    msg = mav.named_value_float_encode(int(boot_ms) & 0xFFFFFFFF, nm.encode("ascii"), float(value))
    return _pack_sysid(mav, msg, sysid)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Horizontal separation (m). Fast equirectangular near CMAC is fine."""
    east = (lon2 - lon1) * 111320.0 * math.cos(math.radians(0.5 * (lat1 + lat2)))
    north = (lat2 - lat1) * 110540.0
    return float(math.hypot(east, north))


def _status_text(msg) -> str:
    t = msg.text
    if isinstance(t, bytes):
        t = t.decode("utf-8", "ignore")
    return str(t).rstrip("\x00")


def _load_waypoints(path: Path) -> List[dict]:
    items = []
    lines = [ln.rstrip("\n") for ln in path.read_text().splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("QGC WPL"):
        raise ValueError(f"{path} is not a QGC WPL mission file")
    for ln in lines[1:]:
        p = ln.split("\t") if "\t" in ln else ln.split()
        if len(p) < 12:
            continue
        items.append(
            {
                "seq": int(p[0]),
                "current": int(p[1]),
                "frame": int(p[2]),
                "command": int(p[3]),
                "p1": float(p[4]),
                "p2": float(p[5]),
                "p3": float(p[6]),
                "p4": float(p[7]),
                "x": float(p[8]),
                "y": float(p[9]),
                "z": float(p[10]),
                "autocontinue": int(float(p[11])),
            }
        )
    return items


def _mission_rows(items: List[dict], sysid: int, t0: float = -0.05) -> List[Row]:
    """Emit MISSION_COUNT + MISSION_ITEM_INT so tlogs carry the route."""
    mav = mavutil.mavlink.MAVLink(None)
    rows: List[Row] = []
    # Dialect here is MAVLink1 ardupilotmega (no mission_type arg).
    count = mav.mission_count_encode(sysid, 1, len(items))
    rows.append((t0, _pack_sysid(mav, count, sysid), sysid))
    for it in items:
        msg = mav.mission_item_int_encode(
            sysid,
            1,
            int(it["seq"]),
            int(it["frame"]),
            int(it["command"]),
            int(it["current"]),
            int(it["autocontinue"]),
            float(it["p1"]),
            float(it["p2"]),
            float(it["p3"]),
            float(it["p4"]),
            int(round(it["x"] * 1e7)),
            int(round(it["y"] * 1e7)),
            float(it["z"]),
        )
        rows.append((t0 + 0.001 * (it["seq"] + 1), _pack_sysid(mav, msg, sysid), sysid))
    return rows


def _load_dual(path: Path, ekf_sysid: int) -> Tuple[List[Row], dict]:
    """Load tlog: EKF @ ekf_sysid, SIM GPI @ ekf_sysid+10, plus metric NVFs."""
    sim_sysid = ekf_sysid + 10
    conn = mavutil.mavlink_connection(str(path))
    rows: List[Row] = []
    flight_t0: Optional[float] = None
    last_ekf: Optional[Tuple[float, float]] = None
    last_sim: Optional[Tuple[float, float]] = None
    last_metric_ts = -1e9
    miss_cnt = 0
    hold_cnt = 0
    det_cnt = 0
    last_miss_inc = -1e9
    last_hold_inc = -1e9
    favg_scr = 0.0
    acfl_scr = 0.0
    max_sep = 0.0
    attack_on = False
    metrics_summary = {
        "max_ekf_sim_m": 0.0,
        "miss_cnt": 0,
        "hold_cnt": 0,
        "det_cnt": 0,
    }

    re_favg = re.compile(r"FEDAVG:(\S+)\s+([0-9.]+)")
    re_acfl = re.compile(r"ACFL:(\S+)\s+([0-9.]+)")

    while True:
        msg = conn.recv_msg()
        if msg is None:
            break
        if msg.get_type() == "BAD_DATA":
            continue
        ts = getattr(msg, "_timestamp", None)
        if ts is None:
            continue
        typ = msg.get_type()
        if (
            flight_t0 is None
            and typ == "GLOBAL_POSITION_INT"
            and abs(int(msg.lat)) > 1000
            and int(getattr(msg, "relative_alt", 0) or 0) > 5000
        ):
            flight_t0 = float(ts)
        if flight_t0 is None:
            continue
        if float(ts) < flight_t0:
            continue
        rel_t = float(ts) - flight_t0
        boot_ms = int(rel_t * 1000.0) & 0xFFFFFFFF

        if typ == "SIM_STATE":
            gpi = _sim_to_gpi(msg, sim_sysid)
            if gpi is not None:
                rows.append((float(ts), gpi, sim_sysid))
                lat_i = int(getattr(msg, "lat_int", 0) or 0)
                lon_i = int(getattr(msg, "lon_int", 0) or 0)
                if lat_i == 0 and lon_i == 0:
                    lat_f = float(msg.lat)
                    lon_f = float(msg.lon)
                    if abs(lat_f) <= 90 and abs(lon_f) <= 180:
                        lat_i = int(lat_f * 1e7)
                        lon_i = int(lon_f * 1e7)
                    else:
                        lat_i, lon_i = int(lat_f), int(lon_f)
                last_sim = (lat_i / 1e7, lon_i / 1e7)
            raw_sim = _repack(msg, sim_sysid)
            if raw_sim is not None:
                rows.append((float(ts), raw_sim, sim_sysid))
        else:
            raw = _repack(msg, ekf_sysid)
            if raw is None:
                continue
            rows.append((float(ts), raw, ekf_sysid))
            if typ == "HEARTBEAT":
                hb_sim = _repack(msg, sim_sysid)
                if hb_sim is not None:
                    rows.append((float(ts), hb_sim, sim_sysid))
            elif typ == "GLOBAL_POSITION_INT" and abs(int(msg.lat)) > 1000:
                last_ekf = (msg.lat / 1e7, msg.lon / 1e7)
            elif typ == "STATUSTEXT":
                text = _status_text(msg)
                if text.startswith("ATTACK:"):
                    attack_on = True
                if "RECOVER" in text or text.startswith("RESULT:"):
                    attack_on = False
                    det_cnt += 1
                # Debounce: STATUSTEXT is emitted every detect loop (~many Hz).
                # Count at most once per second → ≈ seconds under miss / hold.
                if "ACTIVE:" in text and "miss" in text.lower():
                    if float(ts) - last_miss_inc >= 1.0:
                        miss_cnt += 1
                        last_miss_inc = float(ts)
                if "ACTIVE:" in text and "HOLD" in text:
                    if float(ts) - last_hold_inc >= 1.0:
                        hold_cnt += 1
                        last_hold_inc = float(ts)
                m = re_favg.search(text)
                if m:
                    favg_scr = float(m.group(2))
                m = re_acfl.search(text)
                if m:
                    acfl_scr = float(m.group(2))

        # ~1 Hz metrics on EKF sysid (HUD / Status / Quick / Tuning)
        if float(ts) - last_metric_ts >= METRIC_DT_S and last_ekf and last_sim:
            sep = _haversine_m(last_ekf[0], last_ekf[1], last_sim[0], last_sim[1])
            max_sep = max(max_sep, sep)
            rows.append((float(ts) + 1e-4, _named_float(ekf_sysid, "EKF_SIM_M", sep, boot_ms), ekf_sysid))
            rows.append((float(ts) + 2e-4, _named_float(ekf_sysid, "MISS_CNT", float(miss_cnt), boot_ms), ekf_sysid))
            rows.append((float(ts) + 3e-4, _named_float(ekf_sysid, "HOLD_CNT", float(hold_cnt), boot_ms), ekf_sysid))
            rows.append((float(ts) + 4e-4, _named_float(ekf_sysid, "DET_CNT", float(det_cnt), boot_ms), ekf_sysid))
            rows.append((float(ts) + 5e-4, _named_float(ekf_sysid, "FAVG_SCR", favg_scr, boot_ms), ekf_sysid))
            rows.append((float(ts) + 6e-4, _named_float(ekf_sysid, "ACFL_SCR", acfl_scr, boot_ms), ekf_sysid))
            rows.append(
                (
                    float(ts) + 7e-4,
                    _named_float(ekf_sysid, "ATK_ON", 1.0 if attack_on else 0.0, boot_ms),
                    ekf_sysid,
                )
            )
            last_metric_ts = float(ts)

    metrics_summary.update(
        {
            "max_ekf_sim_m": round(max_sep, 2),
            "miss_cnt": miss_cnt,
            "hold_cnt": hold_cnt,
            "det_cnt": det_cnt,
        }
    )
    return rows, metrics_summary


def merge_run(scene_id: str, controller: str, mission_path: Path = MISSION) -> Optional[Path]:
    run = RUNS / scene_id / controller
    meta_path = run / "run_meta.json"
    if not meta_path.is_file():
        run = RUNS / scene_id
        meta_path = run / "run_meta.json"
        if not meta_path.is_file() or controller != "acfl":
            print(f"skip {scene_id}/{controller}: no run_meta.json")
            return None
    meta = json.loads(meta_path.read_text())
    paths = []
    for idx, v in enumerate(meta.get("vehicles") or []):
        rel = v.get("tlog")
        if not rel:
            continue
        p = run / rel
        if p.is_file():
            paths.append((p, int(v.get("sysid") or (idx + 1))))
    if len(paths) < 2:
        print(f"{scene_id}/{controller}: need ≥2 tlogs")
        return None

    print(f"merge {scene_id}/{controller}: {[(p.parent.name, sid) for p, sid in paths]}")
    per = []
    metrics_by_uav = {}
    for p, sid in paths:
        rows, msum = _load_dual(p, ekf_sysid=sid)
        if not rows:
            print(f"  {p.parent.name}: empty")
            continue
        print(
            f"  {p.parent.name}→EKF {sid}/SIM {sid+10}: {len(rows)} msgs "
            f"span={rows[-1][0]-rows[0][0]:.1f}s  max_sep={msum['max_ekf_sim_m']}m "
            f"miss={msum['miss_cnt']} hold={msum['hold_cnt']} det={msum['det_cnt']}"
        )
        per.append(rows)
        metrics_by_uav[f"uav_{sid-1}"] = msum
    if len(per) < 2:
        raise SystemExit(f"{scene_id}/{controller}: <2 non-empty tlogs")

    aligned = []
    for rows in per:
        t_start = rows[0][0]
        aligned.append([(ts - t_start, buf, sid) for ts, buf, sid in rows])
    t1 = max(r[-1][0] for r in aligned)
    t_min = min(r[-1][0] for r in aligned)
    print(f"  relative spans max={t1:.1f}s min={t_min:.1f}s (keeping full)")

    merged: List[Tuple[float, bytes]] = []
    # Waypoints first (negative / early time) on each EKF sysid
    if mission_path.is_file():
        items = _load_waypoints(mission_path)
        for sid in sorted({r[2] for rows in aligned for r in rows if 1 <= r[2] <= 3}):
            for ts, buf, s in _mission_rows(items, sid, t0=-0.05):
                merged.append((ts, buf))
        print(f"  mission: {len(items)} WPs injected on EKF sysids")
    else:
        print(f"  WARNING: mission file missing: {mission_path}")

    for rows in aligned:
        for ts, buf, _sid in rows:
            if ts >= 0.0:
                merged.append((ts, buf))
    merged.sort(key=lambda r: r[0])

    out_dir = OUT / scene_id / controller
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multivehicle_sync.tlog"
    base_usec = int(1_700_000_000_000_000)
    with out_path.open("wb") as f:
        for ts, buf in merged:
            # allow small negative pre-roll for mission dump
            usec = base_usec + int(ts * 1e6)
            if usec < 0:
                usec = 0
            f.write(struct.pack(">Q", usec))
            f.write(buf)

    if mission_path.is_file():
        shutil.copy2(mission_path, out_dir / "mission.waypoints")

    # Verify
    m = mavutil.mavlink_connection(str(out_path))
    sys_counts: Counter = Counter()
    gpi: Counter = Counter()
    nvf: Counter = Counter()
    mission_n = 0
    nvf_names: Counter = Counter()
    while True:
        msg = m.recv_msg()
        if msg is None:
            break
        sid = int(msg.get_srcSystem())
        sys_counts[sid] += 1
        typ = msg.get_type()
        if typ == "GLOBAL_POSITION_INT":
            gpi[sid] += 1
        elif typ == "NAMED_VALUE_FLOAT":
            nvf[sid] += 1
            name = msg.name
            if isinstance(name, bytes):
                name = name.decode("ascii", "ignore")
            nvf_names[str(name).rstrip("\x00")] += 1
        elif typ in ("MISSION_ITEM_INT", "MISSION_ITEM", "MISSION_COUNT"):
            mission_n += 1
    hb: Counter = Counter()
    m2 = mavutil.mavlink_connection(str(out_path))
    while True:
        msg = m2.recv_msg()
        if msg is None:
            break
        if msg.get_type() == "HEARTBEAT":
            hb[int(msg.get_srcSystem())] += 1
    print(f"  verify sysids={dict(sys_counts)}")
    print(f"  GPI EKF/SIM={dict(gpi)}")
    print(f"  HEARTBEAT={dict(hb)}")
    print(f"  NAMED_VALUE_FLOAT={dict(nvf)} names={dict(nvf_names)}")
    print(f"  mission_msgs={mission_n}")
    if any(hb.get(s, 0) == 0 for s in (11, 12, 13)):
        print("  WARNING: SIM sysids missing HEARTBEAT (MP will hide them)")

    side = {
        "scene_id": scene_id,
        "controller": controller,
        "merged_tlog": str(out_path.relative_to(OUT.parents[0])),
        "n_messages": len(merged),
        "overlap_s": round(t1, 3),
        "run_meta": str(meta_path),
        "end_reason": meta.get("end_reason"),
        "recovered": meta.get("recovered"),
        "break_events": meta.get("break_events"),
        "metrics_by_uav": metrics_by_uav,
        "named_value_float": {
            "EKF_SIM_M": "Horizontal EKF vs SIM separation (meters)",
            "MISS_CNT": "≈ seconds of ACTIVE miss (debounced 1 Hz)",
            "HOLD_CNT": "≈ seconds of ACTIVE HOLD / detect (debounced 1 Hz)",
            "DET_CNT": "Cumulative RESULT recovered / RECOVER events",
            "FAVG_SCR": "Latest FedAvg anomaly score from STATUSTEXT",
            "ACFL_SCR": "Latest ACFL anomaly score from STATUSTEXT",
            "ATK_ON": "1 while attack active, else 0",
        },
        "waypoints": {
            "file": "mission.waypoints" if mission_path.is_file() else None,
            "injected_mission_msgs": mission_n,
            "mp_hint": (
                "Flight Plan → Load WP → mission.waypoints (keeps route on map "
                "during tlog play). Also MISSION_ITEM_INT is in the tlog."
            ),
        },
        "map_note": (
            "Dual icons: sysid 1/2/3 = EKF GLOBAL_POSITION_INT; "
            "sysid 11/12/13 = SIM_STATE projected as GLOBAL_POSITION_INT. "
            "Metrics: NAMED_VALUE_FLOAT on EKF sysids (Status / Quick / HUD User Items)."
        ),
        "sysid_map": {
            "uav_0_EKF": 1,
            "uav_1_EKF": 2,
            "uav_2_EKF": 3,
            "uav_0_SIM": 11,
            "uav_1_SIM": 12,
            "uav_2_SIM": 13,
        },
        "gpi_by_sysid": {str(k): v for k, v in sorted(gpi.items())},
    }
    (out_dir / "meta.json").write_text(json.dumps(side, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        f"""# Mission Planner — `{scene_id}` / **{controller}**

Controller: **{controller}** (only this detector may recover).

## Map sysids

| Sysid | Track |
|------:|-------|
| 1/2/3 | EKF `GLOBAL_POSITION_INT` (uav_0/1/2) |
| 11/12/13 | SIM truth as `GLOBAL_POSITION_INT` (uav_0/1/2) |

## Live metrics (during tlog play)

`NAMED_VALUE_FLOAT` on each EKF vehicle (~1 Hz):

| Name | Meaning |
|------|---------|
| `EKF_SIM_M` | EKF vs SIM horizontal error (m) |
| `MISS_CNT` | Cumulative miss events |
| `HOLD_CNT` | Cumulative detect/hold events |
| `DET_CNT` | Cumulative recoveries |
| `FAVG_SCR` / `ACFL_SCR` | Latest anomaly scores |
| `ATK_ON` | Attack active (0/1) |

In Mission Planner: **Status** tab, **Quick** (double-click to add), HUD
**right-click → User Items**, or **Tuning** graph under the map.

Per-UAV summary from this merge: `{json.dumps(metrics_by_uav)}`

## Waypoints on the map

1. **Flight Plan → Load WP** → open `mission.waypoints` in this folder  
   (Mission Planner does not always draw tlog `MISSION_ITEM`s automatically.)
2. Switch back to Flight Data and play `multivehicle_sync.tlog`.
3. The route stays overlaid while icons move.

`MISSION_COUNT` / `MISSION_ITEM_INT` are also embedded at the start of the tlog
({mission_n} mission messages).

## Play checklist

1. Load `mission.waypoints` in Flight Plan.
2. Flight Data → Ctrl+F → Telemetry Logs → open `multivehicle_sync.tlog`.
3. Enable vehicles **1–3** and **11–13**.
4. Add Quick/HUD items: `EKF_SIM_M`, `MISS_CNT`, `HOLD_CNT`.
5. Play. Under spoof, EKF slides off SIM and `EKF_SIM_M` climbs if the controller misses.

- Messages: {len(merged)}
- Span: {t1:.1f}s
- End: `{meta.get('end_reason')}`
- Recovered: {meta.get('recovered')}
- Breaks: {meta.get('break_events')}
"""
    )
    print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", action="append")
    ap.add_argument(
        "--controller",
        action="append",
        choices=list(CONTROLLERS),
        help="Repeatable; default both",
    )
    ap.add_argument("--mission", type=Path, default=MISSION)
    args = ap.parse_args()
    scenes = args.scene or list(SCENES)
    controllers = args.controller or list(CONTROLLERS)
    for sid in scenes:
        for ctrl in controllers:
            merge_run(sid, ctrl, mission_path=args.mission)


if __name__ == "__main__":
    main()
