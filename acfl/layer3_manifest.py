"""Map SITL mission files to network-plane attack types (waypoint/gps/none)."""
import json
import os
import re
from typing import Dict, Optional


def _load_manifest_drones(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        payload = json.load(f)
    out: Dict[str, str] = {}
    for entry in payload.get("drones", []):
        drone = entry.get("drone")
        attack = entry.get("attack")
        if drone and attack:
            out[str(drone)] = str(attack)
    return out


def load_network_attack_types(acfl_root: Optional[str] = None) -> Dict[str, str]:
    """
    Return mission basename -> network attack type.

    Keys look like ``mission_drone_010.csv``; values are ``waypoint``, ``gps``,
    or ``none``.
    """
    root = acfl_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    drift = os.path.join(root, "drift-fairroad", "data")
    drone_attack: Dict[str, str] = {}
    for rel in (
        os.path.join("sitl_dataset", "manifest.json"),
        os.path.join("sitl_validate", "validate_manifest.json"),
        os.path.join("sitl_drone_extra", "drone_extra_manifest.json"),
    ):
        drone_attack.update(_load_manifest_drones(os.path.join(drift, rel)))

    mission_attack: Dict[str, str] = {}
    manifest_path = os.path.join(
        drift, "sitl_dataset", "fl", "agent_mission_manifest.json"
    )
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            missions = json.load(f).get("missions", [])
        for entry in missions:
            drone = entry.get("drone")
            mission_file = entry.get("mission_file", "")
            if not drone or not mission_file:
                continue
            attack = drone_attack.get(drone, "unknown")
            mission_attack[os.path.basename(mission_file)] = attack
    return mission_attack


def network_attack_from_mission_path(
    mission_path: str,
    attack_types: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve network attack type from a mission CSV path or basename."""
    basename = os.path.basename(mission_path)
    if attack_types is None:
        attack_types = load_network_attack_types()
    if basename in attack_types:
        return attack_types[basename]
    match = re.search(r"mission_drone_(\d+)\.csv", basename)
    if match:
        drone = f"drone_{match.group(1)}"
        for key, attack in attack_types.items():
            if key.endswith(f"{match.group(1)}.csv"):
                return attack
    return "unknown"
