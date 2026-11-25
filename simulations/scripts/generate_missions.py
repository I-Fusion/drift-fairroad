#!/usr/bin/env python3
"""
generate_missions.py - generate varied ArduPilot mission plans.

Produces QGC WPL 110 ``.waypoints`` files (the same format as
``missions/mission_2_wp_23.waypoints``) so the harness is not limited to a
single fixed mission. Missions are generated around the SITL home location
(default: the CMAC field used by ArduPilot, matching ``SIM_OPOS_LAT/LNG`` in
the provided param files) using several spatial patterns.

QGC WPL 110 line format (tab separated):
    seq  current  frame  command  p1  p2  p3  p4  x(lat)  y(lon)  z(alt)  autocontinue

Frame 0  = MAV_FRAME_GLOBAL (home, absolute altitude)
Frame 3  = MAV_FRAME_GLOBAL_RELATIVE_ALT (waypoints, relative altitude)
Command 16 = MAV_CMD_NAV_WAYPOINT

Example:
    python3 generate_missions.py --count 10 --min-wp 8 --max-wp 24 \\
        --output ../missions/generated
"""
import argparse
import math
import os
import random
from typing import List, Tuple

# ArduPilot default SITL home (CMAC). Matches SIM_OPOS_LAT / SIM_OPOS_LNG.
HOME_LAT = -35.3632621
HOME_LON = 149.1652374
HOME_ALT = 584.09  # absolute AMSL altitude of home

NAV_WAYPOINT = 16
FRAME_GLOBAL = 0
FRAME_REL_ALT = 3

PATTERNS = ["polygon", "grid", "star", "circle", "random"]


def _meters_to_deg(dlat_m: float, dlon_m: float, lat_deg: float) -> Tuple[float, float]:
    """Convert north/east meter offsets to lat/lon degree offsets."""
    dlat = dlat_m / 111320.0
    dlon = dlon_m / (111320.0 * math.cos(math.radians(lat_deg)))
    return dlat, dlon


def _polygon(n: int, radius_m: float, rng: random.Random) -> List[Tuple[float, float]]:
    """Convex-ish polygon: evenly spaced angles with jittered radius."""
    pts = []
    start = rng.uniform(0, 2 * math.pi)
    for i in range(n):
        ang = start + 2 * math.pi * i / n
        r = radius_m * rng.uniform(0.5, 1.0)
        pts.append((r * math.cos(ang), r * math.sin(ang)))
    return pts


def _circle(n: int, radius_m: float, rng: random.Random) -> List[Tuple[float, float]]:
    start = rng.uniform(0, 2 * math.pi)
    return [
        (radius_m * math.cos(start + 2 * math.pi * i / n),
         radius_m * math.sin(start + 2 * math.pi * i / n))
        for i in range(n)
    ]


def _star(n: int, radius_m: float, rng: random.Random) -> List[Tuple[float, float]]:
    start = rng.uniform(0, 2 * math.pi)
    pts = []
    for i in range(n):
        ang = start + 2 * math.pi * i / n
        r = radius_m if i % 2 == 0 else radius_m * 0.45
        pts.append((r * math.cos(ang), r * math.sin(ang)))
    return pts


def _grid(n: int, radius_m: float, rng: random.Random) -> List[Tuple[float, float]]:
    """Lawnmower / boustrophedon grid covering a square area."""
    cols = max(2, int(round(math.sqrt(n))))
    rows = max(2, int(math.ceil(n / cols)))
    step = (2 * radius_m) / max(1, cols - 1)
    pts = []
    for r in range(rows):
        col_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in col_range:
            east = -radius_m + c * step
            north = -radius_m + r * ((2 * radius_m) / max(1, rows - 1))
            pts.append((north, east))
            if len(pts) >= n:
                return pts
    return pts


def _random(n: int, radius_m: float, rng: random.Random) -> List[Tuple[float, float]]:
    return [
        (rng.uniform(-radius_m, radius_m), rng.uniform(-radius_m, radius_m))
        for _ in range(n)
    ]


_GENERATORS = {
    "polygon": _polygon,
    "circle": _circle,
    "star": _star,
    "grid": _grid,
    "random": _random,
}


def build_mission(
    n_wp: int,
    pattern: str,
    radius_m: float,
    min_alt: float,
    max_alt: float,
    rng: random.Random,
    home_lat: float = HOME_LAT,
    home_lon: float = HOME_LON,
    home_alt: float = HOME_ALT,
) -> List[str]:
    """Return QGC WPL 110 lines for one mission (including the home item)."""
    offsets = _GENERATORS[pattern](n_wp, radius_m, rng)

    lines = ["QGC WPL 110"]
    # seq 0: home (absolute frame, autocontinue=1)
    lines.append(
        f"0\t1\t{FRAME_GLOBAL}\t{NAV_WAYPOINT}\t0\t0\t0\t0\t"
        f"{home_lat:.7f}\t{home_lon:.7f}\t{home_alt:.6f}\t1"
    )

    for i, (north_m, east_m) in enumerate(offsets, start=1):
        dlat, dlon = _meters_to_deg(north_m, east_m, home_lat)
        lat = home_lat + dlat
        lon = home_lon + dlon
        alt = round(rng.uniform(min_alt, max_alt), 6)
        lines.append(
            f"{i}\t0\t{FRAME_REL_ALT}\t{NAV_WAYPOINT}\t"
            f"0.00000000\t0.00000000\t0.00000000\t0.00000000\t"
            f"{lat:.8f}\t{lon:.8f}\t{alt:.6f}\t1"
        )
    return lines


def main():
    ap = argparse.ArgumentParser(description="Generate ArduPilot mission plans")
    ap.add_argument("--count", type=int, default=10, help="number of missions to generate")
    ap.add_argument("--min-wp", type=int, default=8, help="min waypoints per mission")
    ap.add_argument("--max-wp", type=int, default=24, help="max waypoints per mission")
    ap.add_argument("--radius", type=float, default=300.0,
                    help="mission area radius in meters around home")
    ap.add_argument("--min-alt", type=float, default=20.0, help="min relative altitude (m)")
    ap.add_argument("--max-alt", type=float, default=60.0, help="max relative altitude (m)")
    ap.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS,
                    help="spatial patterns to draw from")
    ap.add_argument("--home-lat", type=float, default=HOME_LAT)
    ap.add_argument("--home-lon", type=float, default=HOME_LON)
    ap.add_argument("--home-alt", type=float, default=HOME_ALT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", "-o", required=True, help="output directory")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output, exist_ok=True)

    generated = []
    for idx in range(args.count):
        pattern = rng.choice(args.patterns)
        n_wp = rng.randint(args.min_wp, args.max_wp)
        lines = build_mission(
            n_wp, pattern, args.radius, args.min_alt, args.max_alt, rng,
            args.home_lat, args.home_lon, args.home_alt,
        )
        fname = f"mission_gen_{idx:03d}_{pattern}_wp_{n_wp}.waypoints"
        path = os.path.join(args.output, fname)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        generated.append(path)
        print(f"  [{idx:03d}] {pattern:8s} {n_wp:2d} wp -> {path}")

    print(f"\nGenerated {len(generated)} missions in {args.output}")
    return generated


if __name__ == "__main__":
    main()
