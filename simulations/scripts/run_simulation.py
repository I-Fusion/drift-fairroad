#!/usr/bin/env python3
"""
run_simulation.py - run one ArduCopter SITL flight and collect raw data.

Reproduces the drift-fairroad single-run procedure (AESS_Challenge_Simulation):
  start SITL -> load mission -> arm -> guided -> takeoff 10 -> auto ->
  (inject attack) -> RTL, while capturing the attacker-port traffic and the
  automatically-saved dataflash (.BIN) log.

Instance ``I`` port convention (verified against ArduPilot SITL):
    control / serial0 : 5760 + 10*I
    attacker / serial1: 5762 + 10*I   <- traffic here marks attack periods

Outputs (under ``--run-dir``):
    <run-dir>/sitl/logs/*.BIN   dataflash log (GPS/IMU/EKF)
    <run-dir>/capture.pcapng    captured attacker-port packets
    <run-dir>/run_meta.json     run metadata (ports, attack windows, paths)

Standalone usage:
    python3 run_simulation.py --instance 0 \\
        --mission ../missions/mission_2_wp_23.waypoints \\
        --attack waypoint --run-dir /tmp/run0
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from typing import Dict, List, Optional

from pymavlink import mavutil

import attacks_sitl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(SCRIPT_DIR)              # drift-fairroad/simulations
CONFIG_DIR = os.path.join(SIM_DIR, "configs")

DEFAULT_HOME = (-35.3632621, 149.1652374, 584.09, 353.0)

CONFIG_FOR_ATTACK = {
    "none": "config_waypoint_inject.param",
    "waypoint": "config_waypoint_inject.param",
    "gps": "config_gps_injection.param",
}


def _ardupilot_dir() -> str:
    d = os.environ.get("ARDUPILOT_DIR", os.path.expanduser("~/ardupilot"))
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"ArduPilot not found at {d}. Set ARDUPILOT_DIR or run setup_sitl.sh."
        )
    return d


def _arducopter_bin() -> str:
    b = os.path.join(_ardupilot_dir(), "build", "sitl", "bin", "arducopter")
    if not os.path.isfile(b):
        raise FileNotFoundError(
            f"arducopter binary not found at {b}. Build it with setup_sitl.sh."
        )
    return b


# ---------------------------------------------------------------------------
# Mission file parsing + upload
# ---------------------------------------------------------------------------
def load_waypoints(path: str):
    """Parse a QGC WPL 110 file into a list of mission-item dicts."""
    items = []
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if not lines or not lines[0].startswith("QGC WPL"):
        raise ValueError(f"{path} is not a QGC WPL mission file")
    for ln in lines[1:]:
        p = ln.split("\t")
        if len(p) < 12:
            p = ln.split()
        items.append({
            "seq": int(p[0]), "current": int(p[1]), "frame": int(p[2]),
            "command": int(p[3]),
            "p1": float(p[4]), "p2": float(p[5]), "p3": float(p[6]), "p4": float(p[7]),
            "x": float(p[8]), "y": float(p[9]), "z": float(p[10]),
            "autocontinue": int(float(p[11])),
        })
    return items


def upload_mission(conn, items, timeout: float = 30.0) -> bool:
    """Upload mission items using the standard MISSION protocol."""
    conn.mav.mission_clear_all_send(conn.target_system, conn.target_component)
    conn.mav.mission_count_send(conn.target_system, conn.target_component, len(items))
    sent = 0
    deadline = time.time() + timeout
    while sent < len(items) and time.time() < deadline:
        req = conn.recv_match(type=["MISSION_REQUEST", "MISSION_REQUEST_INT"],
                              blocking=True, timeout=timeout)
        if req is None:
            break
        it = items[req.seq]
        conn.mav.mission_item_int_send(
            conn.target_system, conn.target_component,
            it["seq"], it["frame"], it["command"], it["current"], it["autocontinue"],
            it["p1"], it["p2"], it["p3"], it["p4"],
            int(it["x"] * 1e7), int(it["y"] * 1e7), it["z"],
        )
        sent = req.seq + 1
    ack = conn.recv_match(type="MISSION_ACK", blocking=True, timeout=timeout)
    ok = ack is not None and getattr(ack, "type", 1) == 0
    print(f"[mission] uploaded {sent}/{len(items)} items, ack_ok={ok}")
    return ok


# ---------------------------------------------------------------------------
# Flight control helpers
# ---------------------------------------------------------------------------
def set_mode(conn, mode: str, timeout: float = 15.0) -> bool:
    mapping = conn.mode_mapping() or {}
    if mode not in mapping:
        print(f"[mode] unknown mode {mode}; available: {list(mapping)[:8]}...")
        return False
    conn.mav.set_mode_send(
        conn.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mapping[mode],
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        hb = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=timeout)
        if hb and hb.custom_mode == mapping[mode]:
            print(f"[mode] now {mode}")
            return True
    print(f"[mode] failed to confirm {mode}")
    return False


def wait_armable(conn, timeout: float = 120.0):
    """Wait until the EKF/GPS report a usable position (pre-arm readiness)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = conn.recv_match(type=["GPS_RAW_INT", "EKF_STATUS_REPORT"],
                              blocking=True, timeout=5)
        if msg is None:
            continue
        if msg.get_type() == "GPS_RAW_INT" and msg.fix_type >= 3:
            print("[arm] GPS 3D fix acquired")
            return True
    print("[arm] timeout waiting for GPS fix; proceeding anyway")
    return False


def arm(conn, timeout: float = 90.0, force: bool = False) -> bool:
    deadline = time.time() + timeout
    last = 0.0
    while time.time() < deadline:
        now = time.time()
        if now - last >= 2.0:
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                1, 21196 if force else 0, 0, 0, 0, 0, 0,
            )
            last = now
        msg = conn.recv_match(type=["HEARTBEAT", "COMMAND_ACK", "STATUSTEXT"],
                              blocking=True, timeout=2)
        if msg is None:
            continue
        t = msg.get_type()
        if t == "HEARTBEAT" and (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print("[arm] armed")
            return True
        if t == "COMMAND_ACK" and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            if msg.result != 0:
                print(f"[arm] arm rejected (result={msg.result})")
        if t == "STATUSTEXT" and "PreArm" in (msg.text or ""):
            print(f"[arm] {msg.text}")
    print("[arm] failed to arm")
    return False


def takeoff(conn, alt: float, timeout: float = 60.0) -> bool:
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = conn.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
        if msg and msg.relative_alt / 1000.0 >= alt * 0.95:
            print(f"[takeoff] reached {msg.relative_alt/1000.0:.1f} m")
            return True
    print("[takeoff] timeout/incomplete")
    return False


def get_vehicle_boot_us(conn, timeout: float = 5.0):
    """Return the vehicle's current boot time in microseconds (SYSTEM_TIME).

    The dataflash GPS/IMU ``TimeUS`` column is microseconds-since-boot, and
    SYSTEM_TIME.time_boot_ms is the same boot clock, so an attack window
    recorded in this unit labels the sensor CSVs directly -- with no timezone
    or SIM_SPEEDUP ambiguity (unlike absolute UTC/wall-clock matching).
    """
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        2, 100000, 0, 0, 0, 0, 0,  # msgid 2 = SYSTEM_TIME, 0.1s
    )
    # Drain any buffered SYSTEM_TIME and keep the freshest one (telemetry may
    # have queued up while a blocking attack was running on another socket).
    latest = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = conn.recv_match(type="SYSTEM_TIME", blocking=True, timeout=1.0)
        if msg is None:
            break
        if msg.time_boot_ms:
            latest = msg.time_boot_ms * 1000
        nxt = conn.recv_match(type="SYSTEM_TIME", blocking=False)
        if nxt is None:
            break
        if nxt.time_boot_ms:
            latest = nxt.time_boot_ms * 1000
    return latest


def wait_mission_complete(conn, last_seq: int, timeout: float = 600.0) -> bool:
    """Block until the final waypoint is reached (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = conn.recv_match(type=["MISSION_CURRENT", "MISSION_ITEM_REACHED"],
                              blocking=True, timeout=10)
        if msg is None:
            continue
        seq = msg.seq
        if seq >= last_seq:
            print(f"[mission] reached final waypoint (seq {seq})")
            return True
    print("[mission] timeout waiting for completion")
    return False


# ---------------------------------------------------------------------------
# Packet capture
# ---------------------------------------------------------------------------
def start_capture(control_port: int, attack_port: int, pcap_path: str, iface: str = "lo"):
    """Start a tshark capture of both control and attacker ports.

    The control port carries normal GCS telemetry (-> normal-labeled packets);
    the attacker port carries the injection traffic (-> marks attack periods).
    Returns the Popen handle (or None if tshark is unavailable).
    """
    tshark = shutil.which("tshark")
    if tshark is None:
        print("[capture] tshark not found; skipping packet capture")
        return None
    bpf = f"tcp port {control_port} or tcp port {attack_port}"
    cmd = [tshark, "-i", iface, "-f", bpf, "-w", pcap_path, "-q"]
    print(f"[capture] {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


# ---------------------------------------------------------------------------
# Main single-run
# ---------------------------------------------------------------------------
def run(
    instance: int,
    mission: str,
    run_dir: str,
    attack: str = "waypoint",
    wind: float = 0.0,
    speedup: float = 10.0,
    takeoff_alt: float = 10.0,
    attack_delay_s: float = 8.0,
    attack_count: int = 1,
    gps_attack_duration_s: float = 20.0,
    iface: str = "lo",
    home=DEFAULT_HOME,
    mission_timeout_s: float = 900.0,
    seed: Optional[int] = None,
) -> Dict:
    os.makedirs(run_dir, exist_ok=True)
    sitl_cwd = os.path.join(run_dir, "sitl")
    os.makedirs(sitl_cwd, exist_ok=True)
    pcap_path = os.path.join(run_dir, "capture.pcapng")

    control_port = 5760 + 10 * instance
    attack_port = 5762 + 10 * instance

    param_file = os.path.join(CONFIG_DIR, CONFIG_FOR_ATTACK.get(attack, "config_waypoint_inject.param"))
    home_str = f"{home[0]},{home[1]},{home[2]},{home[3]}"

    # Explicitly map serial0 (control) and serial1 (attacker) to deterministic
    # TCP server ports. SITL only opens serial1 after a GCS connects on serial0.
    cmd = [
        _arducopter_bin(), "-S", "--model", "+",
        "-I", str(instance),
        "--serial0", f"tcp:{control_port}",
        "--serial1", f"tcp:{attack_port}",
        "--speedup", str(speedup),
        "--home", home_str,
        "--defaults", param_file,
        "-w",
    ]
    print(f"[sitl] instance {instance} ports control={control_port} attack={attack_port}")
    print(f"[sitl] {' '.join(cmd)} (cwd={sitl_cwd})")
    sitl_log = open(os.path.join(run_dir, "sitl.log"), "w")
    sitl_proc = subprocess.Popen(cmd, cwd=sitl_cwd, stdout=sitl_log, stderr=subprocess.STDOUT)

    capture_proc = None
    attack_windows: List[Dict] = []
    result = {
        "instance": instance, "mission": mission, "attack": attack, "wind": wind,
        "run_dir": run_dir, "control_port": control_port, "attack_port": attack_port,
        "pcap": pcap_path, "bin": None, "attack_windows": attack_windows,
        "success": False,
    }

    try:
        # Connect for control.
        conn = mavutil.mavlink_connection(f"tcp:127.0.0.1:{control_port}")
        if conn.wait_heartbeat(timeout=60) is None:
            raise TimeoutError("No heartbeat from SITL control port")
        print("[sitl] heartbeat received")

        # Apply wind (other attack params come from --defaults).
        conn.mav.param_set_send(
            conn.target_system, conn.target_component,
            b"SIM_WIND_SPD", float(wind), mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        # Relax arming checks so the automated flight arms reliably in SITL.
        conn.mav.param_set_send(
            conn.target_system, conn.target_component,
            b"ARMING_CHECK", 0.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32,
        )
        # GPS blending (GPS_AUTO_SWITCH=3 in the param files) is unhealthy with a
        # single GPS; only the GPS-spoofing scenario provides a 2nd GPS, so use
        # the primary GPS directly for the other scenarios to arm cleanly/fast.
        if attack != "gps":
            conn.mav.param_set_send(
                conn.target_system, conn.target_component,
                b"GPS_AUTO_SWITCH", 1.0, mavutil.mavlink.MAV_PARAM_TYPE_INT32,
            )
        time.sleep(1.0)

        # Act like a GCS: request steady telemetry so the control port carries
        # a realistic normal packet stream (the source of normal-labeled rows).
        conn.mav.request_data_stream_send(
            conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1,
        )

        # Start packet capture before any attack traffic.
        capture_proc = start_capture(control_port, attack_port, pcap_path, iface)
        time.sleep(1.0)

        items = load_waypoints(mission)
        last_seq = max(it["seq"] for it in items)
        upload_mission(conn, items)

        wait_armable(conn, timeout=120)
        set_mode(conn, "GUIDED")
        if not arm(conn, timeout=30):
            print("[arm] retrying with force-arm")
            arm(conn, timeout=20, force=True)
        takeoff(conn, takeoff_alt)
        set_mode(conn, "AUTO")

        # Run the attack inline so we can record the exact window in both
        # wall-clock time (for pcap/packet labeling) and vehicle UTC (for
        # GPS/IMU labeling, which is robust to SIM_SPEEDUP).
        if attack != "none":
            time.sleep(attack_delay_s)
            wall0, boot0 = time.time(), get_vehicle_boot_us(conn)
            if attack == "waypoint":
                attacks_sitl.inject_waypoint(
                    "127.0.0.1", attack_port, count=attack_count, seed=seed)
            elif attack == "gps":
                attacks_sitl.inject_gps_spoof(
                    "127.0.0.1", attack_port, duration_s=gps_attack_duration_s)
            wall1, boot1 = time.time(), get_vehicle_boot_us(conn)
            attack_windows.append({
                "type": attack, "start": wall0, "end": wall1,
                "timeus_start": boot0, "timeus_end": boot1,
            })

        wait_mission_complete(conn, last_seq, timeout=mission_timeout_s)

        set_mode(conn, "RTL")
        # Give RTL/landing a little wall-clock time to log final samples.
        time.sleep(max(2.0, 10.0 / max(1.0, speedup)))
        result["success"] = True
    finally:
        # Stop capture (flush pcap) and SITL.
        if capture_proc is not None:
            capture_proc.send_signal(signal.SIGINT)
            try:
                capture_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                capture_proc.kill()
        sitl_proc.send_signal(signal.SIGINT)
        try:
            sitl_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sitl_proc.kill()
        sitl_log.close()

        # Locate the dataflash log.
        logs_dir = os.path.join(sitl_cwd, "logs")
        if os.path.isdir(logs_dir):
            bins = sorted(
                (os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".BIN")),
                key=os.path.getmtime,
            )
            if bins:
                result["bin"] = bins[-1]

    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[run] done. bin={result['bin']} pcap={pcap_path} success={result['success']}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Run one SITL flight + data capture")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--mission", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--attack", choices=["none", "waypoint", "gps"], default="waypoint")
    ap.add_argument("--wind", type=float, default=0.0, help="SIM_WIND_SPD m/s (0/6/10)")
    ap.add_argument("--speedup", type=float, default=10.0)
    ap.add_argument("--takeoff-alt", type=float, default=10.0)
    ap.add_argument("--attack-delay", type=float, default=8.0)
    ap.add_argument("--attack-count", type=int, default=1)
    ap.add_argument("--gps-duration", type=float, default=20.0)
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--mission-timeout", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    run(
        instance=args.instance, mission=args.mission, run_dir=args.run_dir,
        attack=args.attack, wind=args.wind, speedup=args.speedup,
        takeoff_alt=args.takeoff_alt, attack_delay_s=args.attack_delay,
        attack_count=args.attack_count, gps_attack_duration_s=args.gps_duration,
        iface=args.iface, mission_timeout_s=args.mission_timeout, seed=args.seed,
    )


if __name__ == "__main__":
    main()
