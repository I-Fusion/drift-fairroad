#!/usr/bin/env python3
"""
Build agent-centric FL packet layout from SITL run outputs.

Each **agent** (drone identity) owns one directory with **many mission** CSVs.
Validate-collected missions are assigned to agents by index (drone_010 -> agent_0
as a second mission), not as separate eval-only agents.

Usage:
    python3 build_agent_mission_dataset.py \\
        --train-runs data/sitl_dataset/runs \\
        --extra-runs data/sitl_validate/runs \\
        --output data/sitl_dataset/fl/packets
"""
import argparse
import glob
import json
import os
import re
import shutil


def _drone_index(path: str) -> int:
    m = re.search(r"drone_(\d+)", path)
    if not m:
        raise ValueError(f"cannot parse drone id from {path}")
    return int(m.group(1))


def _agent_for_drone(drone_idx: int, num_agents: int) -> int:
    return drone_idx % num_agents


def ingest_runs(
    runs_root: str,
    output_packets: str,
    num_agents: int,
    manifest_out: list,
) -> int:
    n = 0
    for run_dir in sorted(glob.glob(os.path.join(runs_root, "drone_*"))):
        labeled = os.path.join(run_dir, "processed", "packets_labeled.csv")
        if not os.path.isfile(labeled):
            continue
        drone_idx = _drone_index(run_dir)
        agent_idx = _agent_for_drone(drone_idx, num_agents)
        agent_dir = os.path.join(output_packets, f"agent_{agent_idx:03d}")
        os.makedirs(agent_dir, exist_ok=True)
        dst = os.path.join(agent_dir, f"mission_drone_{drone_idx:03d}.csv")
        shutil.copy2(labeled, dst)
        manifest_out.append({
            "drone": os.path.basename(run_dir),
            "agent": f"agent_{agent_idx:03d}",
            "mission_file": dst,
            "source": labeled,
        })
        n += 1
    return n


def write_legacy_aliases(output_packets: str, num_agents: int) -> None:
    """First mission per agent -> packet_client_XXX.csv for backward compatibility."""
    for agent_idx in range(num_agents):
        agent_dir = os.path.join(output_packets, f"agent_{agent_idx:03d}")
        missions = sorted(glob.glob(os.path.join(agent_dir, "mission_*.csv")))
        if not missions:
            continue
        shutil.copy2(
            missions[0],
            os.path.join(output_packets, f"packet_client_{agent_idx:03d}.csv"),
        )


def main():
    ap = argparse.ArgumentParser(description="Build agent/mission FL packet layout")
    ap.add_argument("--train-runs", required=True, help="primary SITL runs/ directory")
    ap.add_argument("--extra-runs", action="append", default=[],
                    help="additional run roots (e.g. sitl_validate/runs)")
    ap.add_argument("--output", required=True, help="fl/packets output directory")
    ap.add_argument("--num-agents", type=int, default=10)
    ap.add_argument("--clean", action="store_true", help="remove existing agent_* dirs first")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.clean:
        for d in glob.glob(os.path.join(args.output, "agent_*")):
            shutil.rmtree(d, ignore_errors=True)

    manifest = {"agents": args.num_agents, "missions": []}
    total = ingest_runs(args.train_runs, args.output, args.num_agents, manifest["missions"])
    for extra in args.extra_runs:
        if os.path.isdir(extra):
            total += ingest_runs(extra, args.output, args.num_agents, manifest["missions"])

    write_legacy_aliases(args.output, args.num_agents)

    manifest_path = os.path.join(os.path.dirname(args.output), "agent_mission_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Ingested {total} missions into {args.num_agents} agents -> {args.output}")
    print(f"Manifest: {manifest_path}")
    for agent_idx in range(args.num_agents):
        missions = sorted(glob.glob(os.path.join(args.output, f"agent_{agent_idx:03d}", "mission_*.csv")))
        print(f"  agent_{agent_idx:03d}: {len(missions)} mission(s)")


if __name__ == "__main__":
    main()
