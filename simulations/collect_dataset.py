#!/usr/bin/env python3
"""
collect_dataset.py - orchestrate multi-drone SITL data collection.

Each "drone" is an independent ArduCopter SITL instance flying one mission with
one scenario (attack type + wind), producing its own pcap + dataflash log, which
are post-processed into labeled packet/GPS/IMU CSVs. Drones run in batches of
``--concurrency`` (resource-bound on small machines) but an arbitrary
``--num-drones`` is supported, so the dataset is no longer limited to 3 agents.

Each drone's labeled output becomes one federated client, yielding an N-client
FL dataset directly (drone i -> client i).

Pipeline per drone:
    run_simulation.py  -> capture.pcapng + sitl/logs/*.BIN
    convert_logs.py    -> gps.csv, imu.csv
    packet_pipeline.py -> packets_labeled.csv, gps_labeled.csv, imu_labeled.csv

Example:
    python3 collect_dataset.py --num-drones 6 --concurrency 2 \\
        --attacks waypoint gps none --winds 0 6 10 \\
        --output runs/dataset1 --fl-output runs/dataset1/fl --speedup 15
"""
import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(SIM_DIR, "scripts")
MISSIONS_DIR = os.path.join(SIM_DIR, "missions")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import generate_missions  # noqa: E402
import convert_logs  # noqa: E402
import packet_pipeline  # noqa: E402

PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Scenario planning
# ---------------------------------------------------------------------------
def build_scenarios(
    num_drones: int,
    attacks: List[str],
    winds: List[float],
    missions: List[str],
    start_instance: int = 0,
    fl_index_start: int = 0,
) -> List[Dict]:
    scenarios = []
    for i in range(num_drones):
        inst = start_instance + i
        scenarios.append({
            "instance": inst,
            "fl_index": fl_index_start + i,
            "drone": f"drone_{inst:03d}",
            "mission": missions[i % len(missions)],
            "attack": attacks[i % len(attacks)],
            "wind": float(winds[i % len(winds)]),
        })
    return scenarios


def ensure_missions(args) -> List[str]:
    if args.missions_dir and os.path.isdir(args.missions_dir):
        found = sorted(
            os.path.join(args.missions_dir, f)
            for f in os.listdir(args.missions_dir) if f.endswith(".waypoints")
        )
        if found:
            print(f"[missions] using {len(found)} existing missions from {args.missions_dir}")
            return found
    # Generate a pool of missions (at least num_drones).
    out = os.path.join(args.output, "missions")
    os.makedirs(out, exist_ok=True)
    rng_seed = args.seed
    import random
    rng = random.Random(rng_seed)
    paths = []
    n = max(args.num_drones, args.mission_pool)
    for idx in range(n):
        pattern = rng.choice(generate_missions.PATTERNS)
        n_wp = rng.randint(8, 24)
        lines = generate_missions.build_mission(
            n_wp, pattern, args.area_radius, 20.0, 60.0, rng)
        p = os.path.join(out, f"mission_gen_{idx:03d}_{pattern}_wp_{n_wp}.waypoints")
        with open(p, "w") as f:
            f.write("\n".join(lines) + "\n")
        paths.append(p)
    print(f"[missions] generated {len(paths)} missions in {out}")
    return paths


# ---------------------------------------------------------------------------
# Run a single drone via subprocess (true parallelism + port isolation)
# ---------------------------------------------------------------------------
def launch_drone(sc: Dict, args) -> subprocess.Popen:
    run_dir = os.path.join(args.output, "runs", sc["drone"])
    os.makedirs(run_dir, exist_ok=True)
    sc["run_dir"] = run_dir
    cmd = [
        PYTHON, os.path.join(SCRIPTS_DIR, "run_simulation.py"),
        "--instance", str(sc["instance"]),
        "--mission", sc["mission"],
        "--run-dir", run_dir,
        "--attack", sc["attack"],
        "--wind", str(sc["wind"]),
        "--speedup", str(args.speedup),
        "--attack-count", str(args.attack_count),
        "--gps-duration", str(args.gps_duration),
        "--mission-timeout", str(args.mission_timeout),
        "--iface", args.iface,
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed + sc["instance"])]
    if getattr(args, "attack_delay", None) is not None:
        cmd += ["--attack-delay", str(args.attack_delay)]
    log = open(os.path.join(run_dir, "drone.log"), "w")
    print(f"[launch] {sc['drone']} instance={sc['instance']} attack={sc['attack']} wind={sc['wind']}")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def run_batches(scenarios: List[Dict], args):
    pending = list(scenarios)
    while pending:
        batch = pending[: args.concurrency]
        pending = pending[args.concurrency:]
        procs = [(sc, launch_drone(sc, args)) for sc in batch]
        for sc, p in procs:
            rc = p.wait()
            sc["sim_returncode"] = rc
            print(f"[batch] {sc['drone']} finished rc={rc}")


# ---------------------------------------------------------------------------
# Post-processing (no SITL needed)
# ---------------------------------------------------------------------------
def _timeus_periods_from_meta(meta: Dict) -> List:
    """Build (start_us, end_us) TimeUS periods from the attack windows.

    TimeUS is the dataflash boot clock, so these match the GPS/IMU CSVs exactly
    regardless of SIM_SPEEDUP or timezone.
    """
    periods = []
    for w in meta.get("attack_windows", []):
        s, e = w.get("timeus_start"), w.get("timeus_end")
        if s and e:
            periods.append((float(s), float(e)))
    return periods


def postprocess(sc: Dict, args) -> Dict:
    run_dir = sc["run_dir"]
    meta_path = os.path.join(run_dir, "run_meta.json")
    if not os.path.isfile(meta_path):
        print(f"[post] {sc['drone']}: no run_meta.json; skipping")
        sc["postprocessed"] = False
        return sc
    with open(meta_path) as f:
        meta = json.load(f)
    sc.update({"bin": meta.get("bin"), "pcap": meta.get("pcap"),
               "attack_port": meta.get("attack_port")})

    out = os.path.join(run_dir, "processed")
    os.makedirs(out, exist_ok=True)
    gps_csv = imu_csv = None

    # 1. dataflash -> GPS/IMU CSV
    if meta.get("bin") and os.path.isfile(meta["bin"]):
        gps_csv = os.path.join(out, "gps.csv")
        imu_csv = os.path.join(out, "imu.csv")
        try:
            convert_logs.convert(meta["bin"], gps_csv, imu_csv, instance=0)
        except Exception as e:
            print(f"[post] {sc['drone']}: log conversion failed: {e}")
            gps_csv = imu_csv = None

    # 2. pcap -> periods -> labeled packets + labeled GPS/IMU
    periods = []
    if meta.get("pcap") and os.path.isfile(meta["pcap"]) and meta.get("attack_port"):
        try:
            npy = os.path.join(out, "packets.npy")
            packet_pipeline.parse_pcap_to_npy(meta["pcap"], npy)
            periods = packet_pipeline.attack_periods(npy, meta["attack_port"], args.gap_threshold)
            packet_pipeline.label_packets(
                npy, periods, meta["attack_port"],
                os.path.join(out, "packets_labeled.csv"))
            sc["packets_labeled"] = os.path.join(out, "packets_labeled.csv")
        except Exception as e:
            print(f"[post] {sc['drone']}: packet pipeline failed: {e}")

    # Sensor data is labeled by the attack window in TimeUS (the dataflash's own
    # boot clock) -- robust to SIM_SPEEDUP and timezone, unlike wall-clock/UTC.
    sensor_periods = _timeus_periods_from_meta(meta)
    if gps_csv and os.path.isfile(gps_csv):
        sc["gps_labeled"] = packet_pipeline.label_sensor_csv_by_timeus(
            gps_csv, sensor_periods, os.path.join(out, "gps_labeled.csv"))
    if imu_csv and os.path.isfile(imu_csv):
        sc["imu_labeled"] = packet_pipeline.label_sensor_csv_by_timeus(
            imu_csv, sensor_periods, os.path.join(out, "imu_labeled.csv"))

    sc["postprocessed"] = True
    return sc


# ---------------------------------------------------------------------------
# Assemble per-drone outputs into an FL client dataset
# ---------------------------------------------------------------------------
def assemble_fl(scenarios: List[Dict], fl_output: str):
    import shutil
    pkt_dir = os.path.join(fl_output, "packets")
    gi_dir = os.path.join(fl_output, "gps-imu")
    os.makedirs(pkt_dir, exist_ok=True)
    os.makedirs(gi_dir, exist_ok=True)
    n_pkt = n_gi = 0
    for sc in scenarios:
        cid = sc.get("fl_index", sc["instance"])
        if sc.get("packets_labeled") and os.path.isfile(sc["packets_labeled"]):
            shutil.copy(sc["packets_labeled"],
                        os.path.join(pkt_dir, f"packet_client_{cid:03d}.csv"))
            n_pkt += 1
        if sc.get("gps_labeled") and os.path.isfile(sc["gps_labeled"]):
            shutil.copy(sc["gps_labeled"], os.path.join(gi_dir, f"gps_client_{cid:03d}.csv"))
        if sc.get("imu_labeled") and os.path.isfile(sc["imu_labeled"]):
            shutil.copy(sc["imu_labeled"], os.path.join(gi_dir, f"imu_client_{cid:03d}.csv"))
            n_gi += 1
    print(f"[fl] assembled {n_pkt} packet clients and {n_gi} gps/imu clients in {fl_output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Multi-drone SITL data collection")
    ap.add_argument("--num-drones", type=int, default=6)
    ap.add_argument("--start-instance", type=int, default=0,
                    help="first SITL instance id (drone_{start:03d}); use 10+ for held-out validate")
    ap.add_argument("--fl-index-start", type=int, default=None,
                    help="FL client file index offset (default: same as --start-instance; "
                         "set 0 when writing a separate validate fl/ tree)")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="max simultaneous SITL instances (resource-bound)")
    ap.add_argument("--attacks", nargs="+", default=["waypoint", "gps", "none"],
                    choices=["waypoint", "gps", "none"])
    ap.add_argument("--winds", nargs="+", type=float, default=[0.0, 6.0, 10.0])
    ap.add_argument("--missions-dir", default=None,
                    help="use existing .waypoints here instead of generating")
    ap.add_argument("--mission-pool", type=int, default=10,
                    help="how many missions to generate if none provided")
    ap.add_argument("--area-radius", type=float, default=300.0)
    ap.add_argument("--speedup", type=float, default=10.0)
    ap.add_argument("--attack-count", type=int, default=1)
    ap.add_argument("--attack-delay", type=float, default=None,
                    help="seconds after AUTO before attack (default: run_simulation default)")
    ap.add_argument("--gps-duration", type=float, default=20.0)
    ap.add_argument("--mission-timeout", type=float, default=900.0)
    ap.add_argument("--gap-threshold", type=float, default=1.0)
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", "-o", required=True, help="dataset output root")
    ap.add_argument("--fl-output", default=None, help="assemble FL client dataset here")
    ap.add_argument("--manifest-name", default="manifest.json",
                    help="manifest filename under --output (e.g. validate_manifest.json)")
    ap.add_argument("--no-sim", action="store_true",
                    help="skip flights; only (re)post-process existing runs")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.fl_index_start is None:
        args.fl_index_start = args.start_instance
    missions = ensure_missions(args)
    scenarios = build_scenarios(
        args.num_drones, args.attacks, args.winds, missions,
        start_instance=args.start_instance,
        fl_index_start=args.fl_index_start,
    )

    if not args.no_sim:
        run_batches(scenarios, args)
    else:
        for sc in scenarios:
            sc["run_dir"] = os.path.join(args.output, "runs", sc["drone"])

    for sc in scenarios:
        postprocess(sc, args)

    if args.fl_output:
        assemble_fl(scenarios, args.fl_output)

    manifest = {
        "args": vars(args),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "drones": scenarios,
    }
    manifest_path = os.path.join(args.output, args.manifest_name)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n[done] {len(scenarios)} drones. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
