"""Unit tests for Layer 3 operational metrics."""
import os
import sys

import pytest

ACFL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ACFL_ROOT not in sys.path:
    sys.path.insert(0, ACFL_ROOT)

from acfl.operational_metrics import (
    aggregate_operational_metrics,
    attack_onset_indices,
    compute_agent_operational_metrics,
    compute_false_alarm_metrics,
    compute_time_to_detect,
    mission_duration_seconds,
    seconds_per_window_step,
    window_stride_packets,
)

MISSION_010 = os.path.join(
    ACFL_ROOT,
    "drift-fairroad/data/sitl_dataset/fl/packets/agent_000/mission_drone_010.csv",
)
MISSION_016 = os.path.join(
    ACFL_ROOT,
    "drift-fairroad/data/sitl_dataset/fl/packets/agent_006/mission_drone_016.csv",
)


def test_attack_onset_indices_single_segment():
    labels = [0, 0, 1, 1, 0, 1]
    assert attack_onset_indices(labels) == [2, 5]


def test_compute_time_to_detect_with_seconds():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.4, 0.9, 0.8]
    result = compute_time_to_detect(
        labels,
        scores,
        threshold=0.5,
        seconds_per_window_step=2.0,
    )
    assert result["attack_onsets"] == 1
    assert result["detected_onsets"] == 1
    assert result["delays_windows"] == [1]
    assert result["mean_delay_seconds"] == 2.0


def test_compute_time_to_detect_missed_onset():
    labels = [0, 1, 1]
    scores = [0.1, 0.2, 0.3]
    result = compute_time_to_detect(labels, scores, threshold=0.5)
    assert result["attack_onsets"] == 1
    assert result["detected_onsets"] == 0
    assert result["delays_windows"] == []


def test_compute_false_alarm_metrics():
    labels = [0, 0, 0, 1]
    scores = [0.9, 0.1, 0.8, 0.9]
    result = compute_false_alarm_metrics(
        labels,
        scores,
        threshold=0.5,
        mission_duration_seconds=3600.0,
        max_false_alarms_per_hour=5.0,
    )
    assert result["false_positives"] == 2
    assert result["false_alarm_rate"] == 2 / 3
    assert result["false_alarms_per_hour"] == 2.0
    assert result["within_operational_limit"] is True


def test_window_stride_packets():
    assert window_stride_packets(50, 25) == 25


@pytest.mark.skipif(not os.path.isfile(MISSION_010), reason="SITL mission CSV missing")
def test_mission_duration_positive():
    duration = mission_duration_seconds(MISSION_010)
    assert duration is not None
    assert duration > 0


@pytest.mark.skipif(not os.path.isfile(MISSION_010), reason="SITL mission CSV missing")
def test_seconds_per_window_step_positive():
    step = seconds_per_window_step(MISSION_010, window_size=50, overlap=25)
    assert step is not None
    assert step > 0


@pytest.mark.skipif(
    not os.path.isfile(MISSION_010) or not os.path.isfile(MISSION_016),
    reason="SITL mission CSV missing",
)
def test_compute_agent_operational_metrics_on_real_mission():
    labels = [0] * 10 + [1] * 5
    scores = [0.1] * 12 + [0.9] * 3
    stats = [{"mission_file": MISSION_010}]
    result = compute_agent_operational_metrics(
        labels,
        scores,
        threshold=0.5,
        split_stats=stats,
        window_size=50,
        overlap=25,
    )
    ttd = result["time_to_detect"]
    assert ttd["attack_onsets"] == 1
    assert ttd["detected_onsets"] == 1
    assert ttd["mean_delay_seconds"] is not None
    assert result["false_alarms"]["mission_duration_seconds"] is not None


def test_aggregate_operational_metrics():
    per_agent = [
        {
            "attack_windows": 5,
            "network_attack": "waypoint",
            "operational_metrics": {
                "time_to_detect": {
                    "mean_delay_seconds": 4.0,
                    "mean_delay_windows": 2.0,
                    "detection_rate": 1.0,
                },
                "false_alarms": {
                    "false_positives": 1,
                    "benign_windows": 10,
                    "false_alarm_rate": 0.1,
                    "false_alarms_per_hour": 3.0,
                },
            },
        },
        {
            "attack_windows": 0,
            "network_attack": "none",
            "operational_metrics": {
                "time_to_detect": {
                    "mean_delay_seconds": None,
                    "mean_delay_windows": None,
                    "detection_rate": None,
                },
                "false_alarms": {
                    "false_positives": 0,
                    "benign_windows": 20,
                    "false_alarm_rate": 0.0,
                    "false_alarms_per_hour": 0.0,
                },
            },
        },
    ]
    agg = aggregate_operational_metrics(per_agent)
    assert agg["agents_with_attacks"] == 1
    assert agg["agents_benign"] == 1
    assert agg["time_to_detect"]["mean_delay_seconds"] == 4.0
    assert agg["false_alarms"]["total_false_positives"] == 1
    assert agg["false_alarms"]["benign_missions_mean_false_alarms_per_hour"] == 0.0
