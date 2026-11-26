"""Layer 3 operational metrics: time-to-detect, false alarms, inference latency."""
from __future__ import annotations

import csv
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)


def _parse_timestamp(value: str) -> Optional[datetime]:
    text = str(value).strip()
    if not text:
        return None
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def window_stride_packets(window_size: int, overlap: int) -> int:
    """Packet advance between consecutive sliding windows."""
    return max(1, int(window_size) - int(overlap))


def labels_to_binary(labels: Sequence[Union[int, float]]) -> np.ndarray:
    return (np.asarray(labels).reshape(-1) >= 0.5).astype(int)


def scores_to_predictions(scores: Sequence[float], threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)


def attack_onset_indices(labels: Sequence[Union[int, float]]) -> List[int]:
    """Window indices where labeled attack segments begin (0 -> 1 transitions)."""
    y = labels_to_binary(labels)
    if len(y) == 0:
        return []
    onsets = [0] if y[0] == 1 else []
    for idx in range(1, len(y)):
        if y[idx - 1] == 0 and y[idx] == 1:
            onsets.append(idx)
    return onsets


def mission_duration_seconds(
    mission_file: str,
    *,
    timestamp_col: str = "Timestamp",
) -> Optional[float]:
    """Mission span from first to last packet timestamp (seconds)."""
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    try:
        with open(mission_file, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or timestamp_col not in reader.fieldnames:
                return None
            for row in reader:
                ts = _parse_timestamp(row.get(timestamp_col, ""))
                if ts is None:
                    continue
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
    except OSError:
        return None
    if first_ts is None or last_ts is None or last_ts <= first_ts:
        return None
    return float((last_ts - first_ts).total_seconds())


def mission_packet_count(
    mission_file: str,
) -> Optional[int]:
    try:
        with open(mission_file, newline="") as handle:
            # Subtract header row.
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return None


def seconds_per_window_step(
    mission_file: str,
    *,
    window_size: int,
    overlap: int,
    timestamp_col: str = "Timestamp",
) -> Optional[float]:
    """
    Approximate wall-clock seconds between consecutive window starts.

    Uses mission duration and uniform packet spacing: duration * stride / packets.
    """
    duration = mission_duration_seconds(mission_file, timestamp_col=timestamp_col)
    packets = mission_packet_count(mission_file)
    if duration is None or not packets or packets <= 1:
        return None
    stride = window_stride_packets(window_size, overlap)
    return float(duration * stride / (packets - 1))


def compute_confusion(
    labels: Sequence[Union[int, float]],
    predictions: Sequence[int],
) -> Dict[str, int]:
    y = labels_to_binary(labels)
    preds = np.asarray(predictions, dtype=int).reshape(-1)
    if len(y) != len(preds):
        raise ValueError("labels and predictions must have the same length")
    tp = int(np.sum((y == 1) & (preds == 1)))
    tn = int(np.sum((y == 0) & (preds == 0)))
    fp = int(np.sum((y == 0) & (preds == 1)))
    fn = int(np.sum((y == 1) & (preds == 0)))
    return {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "total_windows": len(y),
    }


def compute_time_to_detect(
    labels: Sequence[Union[int, float]],
    scores: Sequence[float],
    threshold: float,
    *,
    window_stride_packets: int = 1,
    seconds_per_window_step: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Delay from each attack onset until the first positive prediction at/after onset.
    """
    preds = scores_to_predictions(scores, threshold)
    onsets = attack_onset_indices(labels)
    if not onsets:
        return {
            "attack_onsets": 0,
            "detected_onsets": 0,
            "detection_rate": None,
            "delays_windows": [],
            "mean_delay_windows": None,
            "median_delay_windows": None,
            "min_delay_windows": None,
            "max_delay_windows": None,
            "mean_delay_seconds": None,
            "median_delay_seconds": None,
        }

    delays_windows: List[int] = []
    for onset in onsets:
        for idx in range(onset, len(preds)):
            if preds[idx] == 1:
                delays_windows.append(int(idx - onset))
                break

    detected = len(delays_windows)
    result: Dict[str, Any] = {
        "attack_onsets": len(onsets),
        "detected_onsets": detected,
        "detection_rate": float(detected / len(onsets)) if onsets else None,
        "delays_windows": delays_windows,
        "mean_delay_windows": (
            float(np.mean(delays_windows)) if delays_windows else None
        ),
        "median_delay_windows": (
            float(np.median(delays_windows)) if delays_windows else None
        ),
        "min_delay_windows": int(np.min(delays_windows)) if delays_windows else None,
        "max_delay_windows": int(np.max(delays_windows)) if delays_windows else None,
        "window_stride_packets": int(window_stride_packets),
        "seconds_per_window_step": seconds_per_window_step,
    }
    if seconds_per_window_step is not None and delays_windows:
        delays_sec = [d * seconds_per_window_step for d in delays_windows]
        result["delays_seconds"] = delays_sec
        result["mean_delay_seconds"] = float(np.mean(delays_sec))
        result["median_delay_seconds"] = float(np.median(delays_sec))
        result["min_delay_seconds"] = float(np.min(delays_sec))
        result["max_delay_seconds"] = float(np.max(delays_sec))
    return result


def compute_false_alarm_metrics(
    labels: Sequence[Union[int, float]],
    scores: Sequence[float],
    threshold: float,
    *,
    mission_duration_seconds: Optional[float] = None,
    max_false_alarms_per_hour: Optional[float] = None,
) -> Dict[str, Any]:
    """False positives on benign (label=0) windows and optional per-hour rate."""
    preds = scores_to_predictions(scores, threshold)
    confusion = compute_confusion(labels, preds)
    fp = confusion["false_positives"]
    tn = confusion["true_negatives"]
    benign_windows = fp + tn
    false_alarm_rate = float(fp / benign_windows) if benign_windows > 0 else None

    out: Dict[str, Any] = {
        **confusion,
        "benign_windows": benign_windows,
        "false_alarm_rate": false_alarm_rate,
        "mission_duration_seconds": mission_duration_seconds,
    }
    if mission_duration_seconds and mission_duration_seconds > 0:
        hours = mission_duration_seconds / 3600.0
        out["mission_duration_hours"] = hours
        out["false_alarms_per_hour"] = float(fp / hours)
        if max_false_alarms_per_hour is not None:
            out["max_false_alarms_per_hour_limit"] = float(max_false_alarms_per_hour)
            out["within_operational_limit"] = (
                out["false_alarms_per_hour"] <= max_false_alarms_per_hour
            )
    return out


def _eval_mission_files(split_stats: Optional[Sequence[dict]]) -> List[str]:
    if not split_stats:
        return []
    files = []
    for entry in split_stats:
        mission_file = entry.get("mission_file")
        if mission_file:
            files.append(str(mission_file))
    return files


def compute_agent_operational_metrics(
    labels: Sequence[Union[int, float]],
    scores: Sequence[float],
    threshold: float,
    *,
    split_stats: Optional[Sequence[dict]] = None,
    window_size: int = 50,
    overlap: int = 25,
    timestamp_col: str = "Timestamp",
    max_false_alarms_per_hour: Optional[float] = None,
) -> Dict[str, Any]:
    """Per-agent Layer 3 operational metrics from labeled eval windows."""
    mission_files = _eval_mission_files(split_stats)
    stride = window_stride_packets(window_size, overlap)

    duration_sec: Optional[float] = None
    sec_per_step: Optional[float] = None
    if mission_files:
        durations = [
            mission_duration_seconds(path, timestamp_col=timestamp_col)
            for path in mission_files
        ]
        durations = [d for d in durations if d is not None and d > 0]
        if durations:
            duration_sec = float(sum(durations))
        step_estimates = [
            seconds_per_window_step(
                path,
                window_size=window_size,
                overlap=overlap,
                timestamp_col=timestamp_col,
            )
            for path in mission_files
        ]
        step_estimates = [s for s in step_estimates if s is not None and s > 0]
        if step_estimates:
            sec_per_step = float(np.mean(step_estimates))

    return {
        "time_to_detect": compute_time_to_detect(
            labels,
            scores,
            threshold,
            window_stride_packets=stride,
            seconds_per_window_step=sec_per_step,
        ),
        "false_alarms": compute_false_alarm_metrics(
            labels,
            scores,
            threshold,
            mission_duration_seconds=duration_sec,
            max_false_alarms_per_hour=max_false_alarms_per_hour,
        ),
        "mission_files": mission_files,
    }


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    return float(np.mean(nums)) if nums else None


def _median_or_none(values: List[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    return float(np.median(nums)) if nums else None


def aggregate_operational_metrics(
    per_agent: Sequence[Dict[str, Any]],
    *,
    benign_network_attacks: Sequence[str] = ("none",),
) -> Dict[str, Any]:
    """Summarize operational metrics across agents."""
    attack_agents = [
        entry
        for entry in per_agent
        if entry.get("attack_windows", 0) > 0
    ]
    benign_agents = [
        entry
        for entry in per_agent
        if entry.get("network_attack", "unknown") in benign_network_attacks
        or entry.get("attack_windows", 0) == 0
    ]

    ttd_means = [
        entry.get("operational_metrics", {})
        .get("time_to_detect", {})
        .get("mean_delay_seconds")
        for entry in attack_agents
    ]
    ttd_windows = [
        entry.get("operational_metrics", {})
        .get("time_to_detect", {})
        .get("mean_delay_windows")
        for entry in attack_agents
    ]
    detection_rates = [
        entry.get("operational_metrics", {})
        .get("time_to_detect", {})
        .get("detection_rate")
        for entry in attack_agents
    ]

    benign_fa_rates = [
        entry.get("operational_metrics", {})
        .get("false_alarms", {})
        .get("false_alarm_rate")
        for entry in per_agent
        if entry.get("operational_metrics", {}).get("false_alarms", {}).get("benign_windows", 0) > 0
    ]
    benign_fa_per_hour = [
        entry.get("operational_metrics", {})
        .get("false_alarms", {})
        .get("false_alarms_per_hour")
        for entry in benign_agents
        if entry.get("operational_metrics", {}).get("false_alarms", {}).get("false_alarms_per_hour")
        is not None
    ]

    total_fp = sum(
        entry.get("operational_metrics", {})
        .get("false_alarms", {})
        .get("false_positives", 0)
        for entry in per_agent
    )
    total_benign = sum(
        entry.get("operational_metrics", {})
        .get("false_alarms", {})
        .get("benign_windows", 0)
        for entry in per_agent
    )

    return {
        "agents_total": len(per_agent),
        "agents_with_attacks": len(attack_agents),
        "agents_benign": len(benign_agents),
        "time_to_detect": {
            "mean_delay_seconds": _mean_or_none(ttd_means),
            "median_delay_seconds": _median_or_none(ttd_means),
            "mean_delay_windows": _mean_or_none(ttd_windows),
            "median_delay_windows": _median_or_none(ttd_windows),
            "mean_detection_rate": _mean_or_none(detection_rates),
        },
        "false_alarms": {
            "aggregate_false_alarm_rate": (
                float(total_fp / total_benign) if total_benign > 0 else None
            ),
            "total_false_positives": int(total_fp),
            "total_benign_windows": int(total_benign),
            "benign_missions_mean_false_alarms_per_hour": _mean_or_none(
                benign_fa_per_hour
            ),
            "benign_missions_median_false_alarms_per_hour": _median_or_none(
                benign_fa_per_hour
            ),
            "per_agent_false_alarm_rate_mean": _mean_or_none(benign_fa_rates),
        },
    }


def aggregate_operational_by_network_attack(
    per_agent: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Operational summary stratified by network attack type."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for entry in per_agent:
        attack = str(entry.get("network_attack", "unknown"))
        buckets.setdefault(attack, []).append(entry)
    return {
        attack: aggregate_operational_metrics(agents)
        for attack, agents in sorted(buckets.items())
    }
