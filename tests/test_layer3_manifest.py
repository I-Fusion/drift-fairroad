"""Tests for Layer 3 network attack manifest lookup."""
import os
import sys

ACFL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ACFL_ROOT not in sys.path:
    sys.path.insert(0, ACFL_ROOT)

from acfl.layer3_manifest import (
    load_network_attack_types,
    network_attack_from_mission_path,
)


class TestLayer3Manifest:
    def test_loads_eval_mission_attack_types(self):
        types = load_network_attack_types(ACFL_ROOT)
        assert types.get("mission_drone_010.csv") == "waypoint"
        assert types.get("mission_drone_011.csv") == "gps"
        assert types.get("mission_drone_016.csv") == "none"

    def test_resolves_from_full_path(self):
        types = load_network_attack_types(ACFL_ROOT)
        path = "drift-fairroad/data/sitl_dataset/fl/packets/agent_001/mission_drone_011.csv"
        assert network_attack_from_mission_path(path, types) == "gps"
