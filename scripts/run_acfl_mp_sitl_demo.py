#!/usr/bin/env python3
"""Fly ACFL-owned 3-UAV Mission Planner demos with separate detector logs.

For each scene, flies **twice** (or once with --controller):

  * ``fedavg`` — only FedAvg may recover; misses leave the attack running so
    EKF (GPI) diverges from SIM truth until break / max_wall
  * ``acfl``   — only ACFL may recover; stops spoof / restores mission

Each run writes ``results/demo_mission_planner/runs/<scene>/<controller>/``.
Merge projects SIM_STATE → GPI on sysid 11/12/13 so Mission Planner shows
truth vs EKF side-by-side.

Usage (drift-fairroad / ACFL repo root):
  python3 scripts/run_acfl_mp_sitl_demo.py --scene all
  python3 scripts/run_acfl_mp_sitl_demo.py --scene scene_a_gps_spoof --controller fedavg
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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
SIM_SCRIPTS = DRIFT_ROOT / "simulations" / "scripts"
SIM_CONFIGS = DRIFT_ROOT / "simulations" / "configs"
MISSION = DRIFT_ROOT / "simulations" / "missions" / "mission_2_wp_23.waypoints"
OUT_ROOT = ACFL_ROOT / "results" / "demo_mission_planner" / "runs"
FL_TS = DRIFT_ROOT / "fl-time-series"

sys.path.insert(0, str(ACFL_ROOT))
sys.path.insert(0, str(FL_TS))
sys.path.insert(0, str(SIM_SCRIPTS))

import attacks_sitl  # noqa: E402
import run_simulation as sitl  # noqa: E402
from acfl.layer3_detection import SpoofingDetector  # noqa: E402
from evaluate_layer3_sitl import DEFAULT_PROBE_FILE, load_model, load_probe_set  # noqa: E402

DEFAULT_HOME = (-35.3632621, 149.1652374, 584.09, 353.0)
HOME_EAST_M = [0.0, 40.0, 0.0]
HOME_NORTH_M = [0.0, 0.0, 40.0]
INSTANCE_BASE = 70
SYSIDS = [1, 2, 3]


def _first_existing(*cands: Path) -> Path:
    for c in cands:
        if c.is_file():
            return c
    return cands[0]


FEDAVG_CKPT = _first_existing(
    ACFL_ROOT
    / "evaluation_results/comprehensive_validation/journal_byzantine_seed42_fedavg/server_checkpoint.pt",
    FL_TS / "checkpoints/acfl_examples/byzantine_seed42_fedavg.pt",
)
ACFL_CKPT = _first_existing(
    ACFL_ROOT
    / "evaluation_results/comprehensive_validation/journal_byzantine_seed42_acfl_full/server_checkpoint.pt",
    FL_TS / "checkpoints/acfl_examples/byzantine_seed42_acfl_full.pt",
)

# Wave-based scenes. Each is flown under controller=fedavg and controller=acfl.
SCENES = {
    "scene_c_clean": {
        "title": "Clean three-UAV mission (no attack)",
        "waves": [],
        "min_duration_s": 60.0,
        "max_wall_s": 120.0,
        "post_recover_tail_s": 0.0,
    },
    "scene_a_gps_spoof": {
        "title": "Multi-wave GPS spoof — controller recover vs miss→break",
        "waves": [
            {
                "t_s": 15.0,
                "uavs": [0],
                "attack": "gps",
                "probe": "gnss_spoofing",
                "soak_s": 8.0,
                "label": "wave1_gps_uav0",
            },
            {
                "t_s": 50.0,
                "uavs": [1],
                "attack": "gps",
                "probe": "gnss_spoofing",
                "soak_s": 8.0,
                "label": "wave2_gps_uav1",
            },
            {
                "t_s": 90.0,
                "uavs": [0, 2],
                "attack": "gps",
                "probe": "gnss_spoofing",
                "soak_s": 8.0,
                "label": "wave3_gps_uav0_uav2",
            },
        ],
        "gps_duration_s": 300.0,
        # deg offset scale → ~100 m+ EKF slide vs SIM when GPS2 is forced
        "gps_drift_per_step": 0.0012,
        "min_duration_s": 140.0,
        "max_wall_s": 240.0,
        "fedavg_max_wall_s": 300.0,  # give miss path time to diverge / break
        "post_recover_tail_s": 25.0,
    },
    "scene_b_waypoint_collude": {
        "title": "Multi-wave waypoint + GPS — controller restore vs miss→break",
        "waves": [
            {
                "t_s": 15.0,
                "uavs": [0, 1],
                "attack": "waypoint",
                "probe": "mavlink_waypoint",
                "soak_s": 8.0,
                "label": "wave1_wp_uav0_uav1",
            },
            {
                "t_s": 55.0,
                "uavs": [0],
                "attack": "waypoint",
                "probe": "mavlink_waypoint",
                "soak_s": 8.0,
                "label": "wave2_wp_uav0",
            },
            {
                "t_s": 95.0,
                "uavs": [1],
                "attack": "gps",
                "probe": "gnss_spoofing",
                "soak_s": 8.0,
                "label": "wave3_gps_uav1",
            },
        ],
        "gps_duration_s": 300.0,
        "gps_drift_per_step": 0.0012,
        "min_duration_s": 150.0,
        "max_wall_s": 260.0,
        "fedavg_max_wall_s": 320.0,
        "post_recover_tail_s": 25.0,
    },
    # Hero cut: entire fleet spoofed at once — FedAvg collapse vs ACFL fleet recover.
    "scene_d_fleet_collapse": {
        "title": "Fleet-wide GPS spoof — collapse vs ACFL recover-all",
        "waves": [
            {
                "t_s": 20.0,
                "uavs": [0, 1, 2],
                "attack": "gps",
                "probe": "gnss_spoofing",
                "soak_s": 8.0,
                "label": "wave1_gps_all",
            },
        ],
        "gps_duration_s": 360.0,
        "gps_drift_per_step": 0.0016,  # ~ steeper slide for a dramatic map split
        "min_duration_s": 100.0,
        "max_wall_s": 200.0,
        "fedavg_max_wall_s": 280.0,
        "post_recover_tail_s": 30.0,
    },
    # Clean UAV stays locked as a visual reference while two craft are spoofed.
    "scene_e_reference_uav": {
        "title": "2-of-3 GPS spoof — clean reference vs attacked drift",
        "waves": [
            {
                "t_s": 18.0,
                "uavs": [0, 1],
                "attack": "gps",
                "probe": "gnss_spoofing",
                "soak_s": 8.0,
                "label": "wave1_gps_uav0_uav1",
            },
            {
                "t_s": 70.0,
                "uavs": [0, 1],
                "attack": "gps",
                "probe": "gnss_spoofing",
                "soak_s": 8.0,
                "label": "wave2_gps_uav0_uav1",
            },
        ],
        "gps_duration_s": 320.0,
        "gps_drift_per_step": 0.0014,
        "min_duration_s": 130.0,
        "max_wall_s": 220.0,
        "fedavg_max_wall_s": 280.0,
        "post_recover_tail_s": 25.0,
    },
}

CONTROLLERS = ("fedavg", "acfl")


def _offset_home(origin, east_m: float, north_m: float):
    lat0, lon0, alt, yaw = origin
    dlat = north_m / 110540.0
    dlon = east_m / (111320.0 * max(1e-6, abs(np.cos(np.radians(lat0)))))
    return (lat0 + dlat, lon0 + dlon, alt, yaw)


def _arducopter() -> Path:
    root = Path(os.environ.get("ARDUPILOT_DIR", Path.home() / "ardupilot"))
    binary = root / "build" / "sitl" / "bin" / "arducopter"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    return binary


def _set_param(conn, name: str, value: float) -> None:
    conn.mav.param_set_send(
        conn.target_system,
        conn.target_component,
        name.encode() if isinstance(name, str) else name,
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )


def _enable_dataflash(conn) -> None:
    _set_param(conn, "LOG_BACKEND_TYPE", 1.0)
    _set_param(conn, "LOG_DISARMED", 1.0)
    _set_param(conn, "LOG_FILE_DSRMROT", 1.0)


def _pack_statustext(sysid: int, text: str, severity: int = 6) -> bytes:
    """Pack a STATUSTEXT with the vehicle sysid (for injection into .tlog)."""
    mav = mavutil.mavlink.MAVLink(None)
    mav.srcSystem = int(sysid)
    mav.srcComponent = 1
    raw = (text or "")[:50].encode("ascii", errors="ignore")
    msg = mav.statustext_encode(int(severity), raw)
    return bytes(msg.pack(mav))


def _flush_annotations(tlog_path: Path, annotations: List[Tuple[float, bytes]]) -> None:
    """Append buffered STATUSTEXT after setup_logfile's wb handle is closed.

    Appending while the live logfile FD is open races and gets overwritten.
    """
    import struct

    if tlog_path is None or not annotations:
        return
    with tlog_path.open("ab") as f:
        for ts, buf in annotations:
            f.write(struct.pack(">Q", int(ts * 1e6)))
            f.write(buf)


def _status_text(
    conn,
    text: str,
    severity: int = 6,
    *,
    tlog_path: Path = None,
    sysid: int = 1,
    annot_buf: Optional[List[Tuple[float, bytes]]] = None,
) -> None:
    # Buffer for post-close injection (setup_logfile uses wb; concurrent ab races).
    buf = _pack_statustext(sysid, text, severity)
    ts = time.time()
    if annot_buf is not None:
        annot_buf.append((ts, buf))
        return
    # Live path: prefer the open logfile handle so writes share one FD.
    import struct

    lf = getattr(conn, "logfile", None) if conn is not None else None
    if lf is not None:
        try:
            lf.write(struct.pack(">Q", int(ts * 1e6)))
            lf.write(buf)
            lf.flush()
            return
        except Exception:
            pass
    if tlog_path is None:
        return
    with tlog_path.open("ab") as f:
        f.write(struct.pack(">Q", int(ts * 1e6)))
        f.write(buf)


def _build_detectors() -> Tuple[SpoofingDetector, SpoofingDetector, dict]:
    probe_data = load_probe_set(str(DEFAULT_PROBE_FILE)) or {}
    fed_model, _ = load_model(str(FEDAVG_CKPT))
    acfl_model, _ = load_model(str(ACFL_CKPT))
    fed = SpoofingDetector(
        fed_model, threshold=0.5, learning_mode="classification", probe_data=probe_data
    )
    acfl = SpoofingDetector(
        acfl_model, threshold=0.5, learning_mode="classification", probe_data=probe_data
    )
    return fed, acfl, probe_data


def _probe_window(probe_data: dict, scenario: str, rng: np.random.Generator) -> torch.Tensor:
    key = scenario if scenario in probe_data else "benign"
    if key not in probe_data and "benign" in probe_data:
        key = "benign"
    x, _y = probe_data[key]
    idx = int(rng.integers(0, x.shape[0]))
    return x[idx : idx + 1].float()


def recover_vehicle(
    *,
    attack: str,
    control_conn,
    attack_port: int,
    mission_items: List[dict],
    gps_worker: Optional[attacks_sitl.GpsSpoofWorker],
    detector_name: str,
    events: List[dict],
    t0: float,
    tlog_path: Optional[Path] = None,
    sysid: int = 1,
    annot_buf: Optional[List[Tuple[float, bytes]]] = None,
) -> None:
    wall = time.time() - t0
    if attack == "gps" and gps_worker is not None:
        gps_worker.stop()
        # Restore primary GNSS (GPS1) as sole / preferred source.
        _set_param(control_conn, "GPS_PRIMARY", 0.0)
        _set_param(control_conn, "GPS_AUTO_SWITCH", 1.0)  # UseBest
        _status_text(
            control_conn,
            f"{detector_name}:RECOVER gps_stop",
            4,
            tlog_path=tlog_path,
            sysid=sysid,
            annot_buf=annot_buf,
        )
        events.append(
            {
                "t_s": round(wall, 2),
                "detector": detector_name,
                "action": "stop_gps_spoof",
                "attack_port": attack_port,
            }
        )
        print(f"[recover] {detector_name} stopped GPS spoof port={attack_port}")
    elif attack == "waypoint":
        ok = attacks_sitl.restore_mission_items("127.0.0.1", attack_port, mission_items)
        sitl.set_mode(control_conn, "AUTO")
        _status_text(
            control_conn,
            f"{detector_name}:RECOVER mission_ok",
            4,
            tlog_path=tlog_path,
            sysid=sysid,
            annot_buf=annot_buf,
        )
        events.append(
            {
                "t_s": round(wall, 2),
                "detector": detector_name,
                "action": "restore_mission",
                "ok": ok,
                "attack_port": attack_port,
            }
        )
        print(f"[recover] {detector_name} restored mission port={attack_port} ok={ok}")


def _param_with_sysid(base: Path, sysid: int, cwd: Path) -> Path:
    """Copy defaults and force MAV_SYSID (defaults file overrides --sysid)."""
    text = base.read_text()
    lines = []
    found = False
    for line in text.splitlines():
        if line.strip().startswith("MAV_SYSID"):
            lines.append(f"MAV_SYSID        {int(sysid)}")
            found = True
        else:
            lines.append(line)
    if not found:
        lines.append(f"MAV_SYSID        {int(sysid)}")
    out = cwd / f"defaults_sysid_{sysid}.param"
    out.write_text("\n".join(lines) + "\n")
    return out


def _start_vehicle(
    binary: Path,
    instance: int,
    sysid: int,
    home,
    cwd: Path,
    speedup: float,
    attack: str,
) -> subprocess.Popen:
    cwd.mkdir(parents=True, exist_ok=True)
    control_port = 5760 + 10 * instance
    attack_port = 5762 + 10 * instance
    base = SIM_CONFIGS / (
        "config_gps_injection.param" if attack == "gps" else "config_waypoint_inject.param"
    )
    param = _param_with_sysid(base, sysid, cwd)
    home_str = f"{home[0]},{home[1]},{home[2]},{home[3]}"
    cmd = [
        str(binary),
        "-S",
        "--model",
        "+",
        "-I",
        str(instance),
        "--sysid",
        str(sysid),
        "--serial0",
        f"tcp:{control_port}",
        "--serial1",
        f"tcp:{attack_port}",
        "--speedup",
        str(speedup),
        "--home",
        home_str,
        "--defaults",
        str(param),
        "-w",
    ]
    log = open(cwd / "sitl.log", "w")
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    proc._sitl_log = log  # type: ignore[attr-defined]
    return proc


def _attacked_uavs(spec: dict) -> List[int]:
    out = set()
    for w in spec.get("waves") or []:
        out.update(w.get("uavs") or [])
    return sorted(out)


def _check_break(conn, proc: subprocess.Popen, armed_once: bool) -> Optional[str]:
    """Return a break reason string if the craft/sim has failed."""
    if proc.poll() is not None:
        return "sitl_exit"
    # Non-blocking drain with health checks
    alt_m = None
    gps_fix = None
    armed = None
    ekf_bad = False
    for _ in range(40):
        msg = conn.recv_match(blocking=False)
        if msg is None:
            break
        t = msg.get_type()
        if t == "HEARTBEAT":
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        elif t == "GLOBAL_POSITION_INT":
            alt_m = msg.relative_alt / 1000.0
        elif t == "GPS_RAW_INT":
            gps_fix = int(msg.fix_type)
        elif t == "EKF_STATUS_REPORT":
            # flags bitfield; low healthy flags indicate trouble
            flags = int(getattr(msg, "flags", 0) or 0)
            if flags and flags < 0x1F:
                ekf_bad = True
        elif t == "STATUSTEXT":
            text = (msg.text or "").lower()
            if any(k in text for k in ("ekf", "crash", "failsafe", "gps glitch", "bad gps")):
                if "prearm" not in text:
                    return f"statustext:{msg.text[:40]}"
    if armed_once and armed is False:
        return "unexpected_disarm"
    if alt_m is not None and alt_m < 1.5 and armed_once:
        return f"alt_collapse:{alt_m:.1f}m"
    if gps_fix is not None and gps_fix < 2 and armed_once:
        return f"gps_lost:fix={gps_fix}"
    if ekf_bad:
        return "ekf_unhealthy"
    return None


def run_scene(scene_id: str, controller: str = "acfl", speedup: float = 5.0) -> dict:
    if controller not in CONTROLLERS:
        raise ValueError(f"controller must be one of {CONTROLLERS}")
    spec = SCENES[scene_id]
    out_dir = OUT_ROOT / scene_id / controller
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    binary = _arducopter()
    mission_items = sitl.load_waypoints(str(MISSION))
    fed_det, acfl_det, probe_data = _build_detectors()
    ctrl_det = fed_det if controller == "fedavg" else acfl_det
    ctrl_label = "FEDAVG" if controller == "fedavg" else "ACFL"
    rng = np.random.default_rng(42)

    procs: List[subprocess.Popen] = []
    conns = []
    tlog_paths: List[Path] = []
    annot_bufs: List[List[Tuple[float, bytes]]] = [[], [], []]
    gps_workers: Dict[int, attacks_sitl.GpsSpoofWorker] = {}
    recovered_pairs = set()
    uav_recovered = set()
    events: List[dict] = []
    break_events: List[dict] = []
    armed_once = [False, False, False]
    end_reason = "max_wall"

    try:
        attacked = set(_attacked_uavs(spec))
        uses_gps = any(w.get("attack") == "gps" for w in (spec.get("waves") or []))
        for i in range(3):
            inst = INSTANCE_BASE + i
            home = _offset_home(DEFAULT_HOME, HOME_EAST_M[i], HOME_NORTH_M[i])
            cwd = out_dir / "sitl" / f"uav_{i}"
            # GPS-capable defaults whenever any wave uses GPS (supports mixed scenes).
            param_attack = "gps" if (uses_gps or i in attacked) else "waypoint"
            procs.append(
                _start_vehicle(binary, inst, SYSIDS[i], home, cwd, speedup, param_attack)
            )
            time.sleep(0.4)
        time.sleep(3.0)

        for i in range(3):
            inst = INSTANCE_BASE + i
            port = 5760 + 10 * inst
            conn = mavutil.mavlink_connection(f"tcp:127.0.0.1:{port}")
            if conn.wait_heartbeat(timeout=90) is None:
                raise TimeoutError(f"no heartbeat on {port}")
            tlog = out_dir / "sitl" / f"uav_{i}" / "telemetry.tlog"
            conn.setup_logfile(str(tlog))
            conns.append(conn)
            tlog_paths.append(tlog)
            _set_param(conn, "ARMING_CHECK", 0.0)
            _set_param(conn, "MAV_SYSID", float(SYSIDS[i]))
            _enable_dataflash(conn)
            # Request SIM_STATE (~5 Hz) so MP / tracks can show truth vs EKF.
            try:
                conn.mav.command_long_send(
                    conn.target_system,
                    conn.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    mavutil.mavlink.MAVLINK_MSG_ID_SIM_STATE,
                    200000,  # usec
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            except Exception:
                pass
            hb = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
            if hb is not None:
                print(f"[sysid] uav_{i} heartbeat src={hb.get_srcSystem()} want={SYSIDS[i]}")
            # Primary GPS for reliable arming; GPS-spoof craft still gets secondary input on attacker port.
            _set_param(conn, "GPS_AUTO_SWITCH", 1.0)
            conn.mav.request_data_stream_send(
                conn.target_system,
                conn.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,
                1,
            )
            for msgid, us in ((33, 100000), (24, 200000), (27, 40000), (30, 100000), (193, 200000)):
                conn.mav.command_long_send(
                    conn.target_system,
                    conn.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    msgid,
                    us,
                    0,
                    0,
                    0,
                    0,
                    0,
                )

        time.sleep(2.0)
        for i, conn in enumerate(conns):
            sitl.wait_armable(conn, timeout=90)
            sitl.upload_mission(conn, mission_items)
            sitl.set_mode(conn, "GUIDED")
            if not sitl.arm(conn, timeout=20, force=True):
                sitl.arm(conn, timeout=20, force=True)
            armed_once[i] = True
            sitl.takeoff(conn, 10.0, timeout=max(40.0, 40.0 / speedup))
            sitl.set_mode(conn, "AUTO")
            _status_text(
                conn,
                f"{ctrl_label} demo {scene_id} uav_{i}"[:50],
                tlog_path=tlog_paths[i],
                sysid=SYSIDS[i],
                annot_buf=annot_bufs[i],
            )

        def emit(i: int, text: str, severity: int = 6) -> None:
            _status_text(
                conns[i],
                text,
                severity,
                tlog_path=tlog_paths[i],
                sysid=SYSIDS[i],
                annot_buf=annot_bufs[i],
            )

        def start_attack(i: int, attack: str, wave_label: str, now: float, soak_s: float) -> None:
            aport = 5762 + 10 * (INSTANCE_BASE + i)
            attack_active[i] = True
            attack_kind[i] = attack
            attack_started_at[i] = now
            attack_wave[i] = wave_label
            attack_soak[i] = soak_s
            uav_recovered.discard(i)
            if attack == "gps":
                # Force EKF onto MAVLink GPS2 (spoofed) so map EKF≠SIM is obvious.
                _set_param(conns[i], "GPS_PRIMARY", 1.0)  # SecondGPS
                _set_param(conns[i], "GPS_AUTO_SWITCH", 0.0)  # Use primary
                prev = gps_workers.get(i)
                if prev is not None:
                    prev.stop()
                worker = attacks_sitl.GpsSpoofWorker(
                    "127.0.0.1",
                    aport,
                    duration_s=float(spec.get("gps_duration_s", 300.0)),
                    drift_per_step=float(spec.get("gps_drift_per_step", 0.0012)),
                    rate_hz=10.0,
                )
                worker.start()
                gps_workers[i] = worker
                emit(i, f"ATTACK:gps {wave_label}"[:50], 4)
            else:
                attacks_sitl.inject_waypoint(
                    "127.0.0.1", aport, count=1, seed=100 + i + int(now)
                )
                emit(i, f"ATTACK:wp {wave_label}"[:50], 4)
            events.append(
                {
                    "t_s": round(now, 2),
                    "uav": i,
                    "action": f"start_{attack}",
                    "wave": wave_label,
                }
            )
            print(f"[attack] {wave_label} uav_{i} {attack} at t={now:.1f}s")

        t0 = time.time()
        attack_active = {i: False for i in range(3)}
        attack_kind: Dict[int, str] = {}
        attack_started_at: Dict[int, float] = {}
        attack_wave: Dict[int, str] = {}
        attack_probe: Dict[int, str] = {}
        attack_soak: Dict[int, float] = {}
        waves = list(spec.get("waves") or [])
        waves_fired: set = set()
        min_dur = float(spec["min_duration_s"])
        if controller == "fedavg" and "fedavg_max_wall_s" in spec:
            max_wall = float(spec["fedavg_max_wall_s"])
        else:
            max_wall = float(spec["max_wall_s"])
        tail_s = float(spec.get("post_recover_tail_s", 20.0))
        last_recover_at: Optional[float] = None
        scene_done = False

        for i in range(3):
            emit(i, f"CTRL:{ctrl_label} recovers on hold"[:50], 5)
            emit(i, "MAP:EKF=sys1-3 SIM=sys11-13", 5)

        while not scene_done:
            now = time.time() - t0
            if now >= max_wall:
                end_reason = "max_wall"
                break

            for wi, wave in enumerate(waves):
                if wi in waves_fired:
                    continue
                if now < float(wave["t_s"]):
                    continue
                waves_fired.add(wi)
                label = str(wave.get("label") or f"wave{wi}")
                for i in wave.get("uavs") or []:
                    ii = int(i)
                    attack_probe[ii] = str(wave.get("probe") or "benign")
                    start_attack(
                        ii,
                        str(wave["attack"]),
                        label,
                        now,
                        float(wave.get("soak_s", 8.0)),
                    )

            if not waves and now >= min_dur:
                end_reason = "clean_complete"
                break

            for i, conn in enumerate(conns):
                br = _check_break(conn, procs[i], armed_once[i])
                if br:
                    break_events.append({"t_s": round(now, 2), "uav": i, "break": br})
                    emit(i, f"BREAK:{br}"[:50], 2)
                    emit(i, f"MISSCONSEQ:{ctrl_label} failed"[:50], 2)
                    print(f"[break] uav_{i} {br} at t={now:.1f}s ({controller})")
                    end_reason = f"break_uav_{i}:{br}"
                    scene_done = True
                    break

                scenario = (
                    attack_probe.get(i, "benign") if attack_active.get(i) else "benign"
                )
                window = _probe_window(probe_data, scenario, rng)
                # Score both for Messages contrast; only controller may recover.
                fed_s = fed_det.detect(window)
                acfl_s = acfl_det.detect(window)
                ctrl_s = ctrl_det.detect(window)
                fed_action = str(fed_s.get("recommended_action", "continue"))
                acfl_action = str(acfl_s.get("recommended_action", "continue"))
                ctrl_action = str(ctrl_s.get("recommended_action", "continue"))
                fed_score = float(fed_s.get("anomaly_score", 0))
                acfl_score = float(acfl_s.get("anomaly_score", 0))
                emit(
                    i,
                    f"FEDAVG:{fed_action[:4]} {fed_score:.2f}",
                    5 if attack_active.get(i) else 6,
                )
                emit(
                    i,
                    f"ACFL:{acfl_action[:4]} {acfl_score:.2f}",
                    4 if acfl_action == "safe_hold" else 6,
                )
                if attack_active.get(i):
                    if ctrl_action != "safe_hold":
                        emit(i, f"ACTIVE:{ctrl_label} miss attack on"[:50], 3)
                    else:
                        emit(i, f"ACTIVE:{ctrl_label} HOLD"[:50], 4)

                if not attack_active.get(i):
                    continue
                soak = float(attack_soak.get(i, 8.0))
                if (now - attack_started_at.get(i, now)) < soak:
                    continue
                if ctrl_action != "safe_hold":
                    continue
                recover_vehicle(
                    attack=attack_kind.get(i, "gps"),
                    control_conn=conn,
                    attack_port=5762 + 10 * (INSTANCE_BASE + i),
                    mission_items=mission_items,
                    gps_worker=gps_workers.get(i),
                    detector_name=ctrl_label,
                    events=events,
                    t0=t0,
                    tlog_path=tlog_paths[i],
                    sysid=SYSIDS[i],
                    annot_buf=annot_bufs[i],
                )
                emit(i, f"RESULT:{ctrl_label} recovered"[:50], 4)
                attack_active[i] = False
                uav_recovered.add(i)
                recovered_pairs.add((i, ctrl_label))
                last_recover_at = now

            if scene_done:
                break

            # ACFL path: healthy tail after waves cleared.
            # FedAvg miss path: do NOT end on waves_complete while attacks still on —
            # keep flying until break / max_wall so EKF↔SIM divergence is recorded.
            if waves and len(waves_fired) >= len(waves) and not any(attack_active.values()):
                end_reason = "waves_complete"
                if last_recover_at is not None and (now - last_recover_at) >= tail_s:
                    break
                if last_recover_at is None and now >= min_dur:
                    break

            time.sleep(1.0)

        for conn in conns:
            try:
                sitl.set_mode(conn, "RTL")
            except Exception:
                pass
        time.sleep(max(2.0, 6.0 / speedup))
    finally:
        for w in gps_workers.values():
            w.stop()
        for conn in conns:
            # mavtcp.close() does NOT close setup_logfile's wb handle; close it
            # explicitly so a later flush cannot overwrite appended STATUSTEXT.
            try:
                lf = getattr(conn, "logfile", None)
                if lf is not None:
                    lf.flush()
                    lf.close()
                    conn.logfile = None
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        # Inject buffered STATUSTEXT after logfile handles are closed.
        for i, path in enumerate(tlog_paths):
            n = len(annot_bufs[i])
            _flush_annotations(path, annot_bufs[i])
            print(f"[annot] flushed {n} STATUSTEXT → {path}")
        for proc in procs:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
            log = getattr(proc, "_sitl_log", None)
            if log:
                log.close()

    attacked_ids = _attacked_uavs(spec)
    vehicles = []
    for i in range(3):
        cwd = out_dir / "sitl" / f"uav_{i}"
        bins = sorted((cwd / "logs").glob("*.BIN")) if (cwd / "logs").is_dir() else []
        tlog = cwd / "telemetry.tlog"
        vehicles.append(
            {
                "uav_id": f"uav_{i}",
                "sysid": SYSIDS[i],
                "tlog": str(tlog.relative_to(out_dir)) if tlog.is_file() else None,
                "bin": str(bins[0].relative_to(out_dir)) if bins else None,
                "attacked": i in attacked_ids,
                "recovered": i in uav_recovered,
            }
        )

    meta = {
        "scene_id": scene_id,
        "controller": controller,
        "title": spec["title"],
        "waves": spec.get("waves") or [],
        "attack_uavs": attacked_ids,
        "end_reason": end_reason,
        "speedup": speedup,
        "fedavg_checkpoint": str(FEDAVG_CKPT),
        "acfl_checkpoint": str(ACFL_CKPT),
        "policy": {
            "controller": controller,
            "on_detect": f"{ctrl_label} safe_hold → recover.",
            "on_miss": "Keep attack active until break (or max_wall) so EKF diverges from SIM.",
            "map": "Merge emits EKF GPI on sysid 1/2/3 and SIM→GPI on 11/12/13.",
        },
        "events": events,
        "break_events": break_events,
        "recovered": [{"uav": a, "detector": b} for a, b in sorted(recovered_pairs)],
        "vehicles": vehicles,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        json.dumps(
            {
                "scene": scene_id,
                "controller": controller,
                "end_reason": end_reason,
                "recovered": meta["recovered"],
                "breaks": break_events,
            },
            indent=2,
        )
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="all", choices=["all", *SCENES.keys()])
    ap.add_argument(
        "--controller",
        default="both",
        choices=["both", *CONTROLLERS],
        help="Which detector controls recovery (default: fly both separately)",
    )
    ap.add_argument("--speedup", type=float, default=5.0)
    args = ap.parse_args()
    order = [
        "scene_c_clean",
        "scene_a_gps_spoof",
        "scene_b_waypoint_collude",
        "scene_d_fleet_collapse",
        "scene_e_reference_uav",
    ]
    scenes = order if args.scene == "all" else [args.scene]
    controllers = list(CONTROLLERS) if args.controller == "both" else [args.controller]
    summary: dict = {}
    for sid in scenes:
        summary[sid] = {}
        for ctrl in controllers:
            print(f"\n=== flying {sid} / {ctrl} ===")
            summary[sid][ctrl] = run_scene(sid, controller=ctrl, speedup=args.speedup)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
