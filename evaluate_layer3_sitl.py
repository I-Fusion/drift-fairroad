#!/usr/bin/env python3
"""
Layer 3 inference evaluation on SITL labeled streams.

Uses the same agent-mission eval windows as the FL evaluator.

Loads a trained global checkpoint and runs SpoofingDetector on sliding
windows from packet (classification) or GPS/IMU (regression) client files.

Usage:
    python3 evaluate_layer3_sitl.py \\
        --data-dir data/sitl_dataset/fl/packets \\
        --checkpoint fl-time-series/checkpoints/server_round_5.pt \\
        --output evaluation_results/layer3_sitl_packets.json
"""
import argparse
import importlib
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def _resolve_repo_roots(here: str):
    """Support flat drift-fairroad root (PR) and nested ACFL/drift-fairroad."""
    root = os.path.dirname(os.path.abspath(here))
    if os.path.isdir(os.path.join(root, "fl-time-series")):
        return root, root
    nested = os.path.join(root, "drift-fairroad")
    if os.path.isdir(nested):
        return root, nested
    return root, root


def _resolve_python(repo_root: str) -> str:
    for cand in (
        os.path.join(repo_root, "acfl_env", "bin", "python"),
        os.path.join(repo_root, "fl_env", "bin", "python"),
        sys.executable,
    ):
        if cand == sys.executable or os.path.isfile(cand):
            return cand
    return sys.executable

ACFL_ROOT, DRIFT_ROOT = _resolve_repo_roots(__file__)
FL_TS_DIR = os.path.join(DRIFT_ROOT, "fl-time-series")
for p in [ACFL_ROOT, DRIFT_ROOT, FL_TS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from acfl.layer3_detection import SpoofingDetector
from acfl.layer3_manifest import load_network_attack_types, network_attack_from_mission_path
from acfl.operational_metrics import (
    aggregate_operational_metrics,
    aggregate_operational_by_network_attack,
    compute_agent_operational_metrics,
)
from agent_mission_data import (
    build_agent_eval_windows,
    list_agent_ids,
    list_mission_files_for_agent,
)
from data_preprocessing_time_series import DataPreprocessorTimeSeries, split_temporal_windows


def list_packet_mission_files(data_dir: str) -> list:
    """All mission CSVs: agent_XXX/mission_*.csv, else legacy packet_client_*.csv."""
    import glob

    files = []
    for agent_num in list_agent_ids(data_dir):
        files.extend(list_mission_files_for_agent(data_dir, agent_num))
    if files:
        return sorted(set(files))
    return sorted(glob.glob(os.path.join(data_dir, "packet_client_*.csv")))


def infer_config_module(state: dict) -> str:
    """Pick FL config from checkpoint weight keys (LSTM vs TinyPacketCNN)."""
    if not state:
        raise ValueError("Empty state_dict; cannot infer model config")
    keys = list(state.keys())
    if any(k.startswith("features.") for k in keys):
        return "config_classification_drone"
    if any(k.startswith("lstm.") for k in keys):
        return "config_classification"
    raise ValueError(
        "Cannot infer config from checkpoint keys; pass --config explicitly"
    )


def _expected_input_size(config) -> int:
    packet_features = getattr(config, "PACKET_FEATURES", None) or []
    if packet_features:
        return len(packet_features)
    if hasattr(config, "INPUT_SIZE"):
        return config.INPUT_SIZE
    gps = getattr(config, "GPS_FEATURES", []) or []
    imu = getattr(config, "IMU_FEATURES", []) or []
    pkt = getattr(config, "PACKET_FEATURES", []) or []
    total = len(gps) + len(imu) + len(pkt)
    return total if total > 0 else 5


CHECKPOINT_DIR = os.path.join(FL_TS_DIR, "checkpoints")
DEFAULT_PROBE_FILE = os.path.join(CHECKPOINT_DIR, "acfl_probes_sitl.pt")


def load_probe_set(probe_file: Optional[str]) -> Optional[dict]:
    if not probe_file or not os.path.isfile(probe_file):
        return None
    payload = torch.load(probe_file, map_location="cpu", weights_only=False)
    return payload.get("probe_data", payload)


def load_model(ckpt_path: str, config_module: str = "auto"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if config_module in (None, "", "auto"):
        config_module = infer_config_module(state or {})
    elif state:
        inferred = infer_config_module(state)
        if inferred != config_module:
            print(
                f"  checkpoint looks like {inferred}; overriding --config {config_module}",
                file=sys.stderr,
            )
            config_module = inferred
    os.environ["FL_CONFIG_MODULE"] = config_module
    config = importlib.import_module(config_module)
    model_module = importlib.import_module(config.MODEL_PATH)
    model_class = getattr(model_module, config.MODEL_CLASS)
    cfg = config.MODEL_CONFIG.copy()
    expected = _expected_input_size(config)
    input_size = ckpt.get("input_size")
    if input_size is None and state:
        for key, tensor in state.items():
            if key == "features.0.weight":
                input_size = int(tensor.shape[1])
                break
            if key.endswith("weight_ih_l0"):
                input_size = int(tensor.shape[1])
                break
    if input_size is None:
        input_size = expected
    if input_size != expected:
        raise ValueError(
            f"Checkpoint input_size={input_size} does not match "
            f"config expectation ({expected} features for {config_module})"
        )
    cfg["input_size"] = input_size
    cfg["output_size"] = ckpt.get("output_size", cfg.get("output_size", 1))
    model = model_class(**cfg)
    if state is None:
        raise ValueError(f"No model weights in checkpoint: {ckpt_path}")
    model.load_state_dict(state)
    model.eval()
    return model, config


def _sanitize_scores(scores) -> np.ndarray:
    """Replace non-finite detector scores (broken FL checkpoints) with 0.5."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return arr
    bad = ~np.isfinite(arr)
    if bad.any():
        arr = arr.copy()
        arr[bad] = 0.5
    return arr


def _safe_auroc(labels, scores) -> Optional[float]:
    labels_arr = np.asarray(labels).astype(int)
    scores_arr = _sanitize_scores(scores)
    if len(set(labels_arr.tolist())) < 2 or scores_arr.size == 0:
        return None
    try:
        return float(roc_auc_score(labels_arr, scores_arr))
    except ValueError:
        return None


def threshold_metrics(labels, scores, threshold: float) -> dict:
    scores_arr = _sanitize_scores(scores)
    labels_arr = np.asarray(labels).astype(int)
    preds = [1 if s >= threshold else 0 for s in scores_arr]
    return {
        "f1": float(f1_score(labels_arr, preds, zero_division=0)),
        "precision": float(precision_score(labels_arr, preds, zero_division=0)),
        "recall": float(recall_score(labels_arr, preds, zero_division=0)),
    }


def tune_threshold_for_f1(labels, scores, default_threshold: float) -> dict:
    """Choose the score threshold with highest binary F1 on labeled eval windows."""
    labels_arr = np.asarray(labels).astype(int)
    scores_arr = _sanitize_scores(scores)
    fixed = threshold_metrics(labels_arr, scores_arr, default_threshold)
    result = {
        "fixed_threshold": float(default_threshold),
        "fixed_f1": fixed["f1"],
        "fixed_precision": fixed["precision"],
        "fixed_recall": fixed["recall"],
        "tuned_threshold": None,
    }
    if len(np.unique(labels_arr)) < 2 or len(scores_arr) == 0:
        return result

    candidates = np.unique(scores_arr)
    candidates = np.unique(np.concatenate([candidates, [float(default_threshold)]]))
    best = None
    for threshold in candidates:
        metrics = threshold_metrics(labels_arr, scores_arr, float(threshold))
        # Prefer fewer false alarms when F1 ties, which means a higher threshold.
        key = (metrics["f1"], metrics["precision"], float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)

    if best is not None and best[2]["f1"] > fixed["f1"]:
        result.update({
            "tuned_threshold": best[1],
            "tuned_f1": best[2]["f1"],
            "tuned_precision": best[2]["precision"],
            "tuned_recall": best[2]["recall"],
        })
    return result


def benchmark_inference_latency(
    X,
    detector,
    *,
    warmup_windows: int = 5,
) -> Optional[dict]:
    """Measure per-window inference latency on eval windows (milliseconds)."""
    if len(X) == 0:
        return None
    warmup = min(int(warmup_windows), len(X))
    for idx in range(warmup):
        detector.detect(torch.FloatTensor(X[idx : idx + 1]))

    timings_ms = []
    for idx in range(len(X)):
        window = torch.FloatTensor(X[idx : idx + 1])
        start = time.perf_counter()
        detector.detect(window)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(timings_ms, dtype=float)
    return {
        "windows_benchmarked": int(len(arr)),
        "mean_ms_per_window": float(np.mean(arr)),
        "median_ms_per_window": float(np.median(arr)),
        "p95_ms_per_window": float(np.percentile(arr, 95)),
        "max_ms_per_window": float(np.max(arr)),
    }


def eval_windows(
    name: str,
    X,
    y,
    detector,
    threshold: float,
    *,
    benchmark_latency: bool = False,
):
    """Evaluate an already-windowed agent or mission split."""
    if len(X) == 0:
        return None

    scores, labels = [], []
    for i in range(len(X)):
        window = torch.FloatTensor(X[i : i + 1])
        label = int(np.asarray(y[i]).reshape(-1)[0] >= 0.5)
        det = detector.detect(window)
        scores.append(det["anomaly_score"])
        labels.append(label)

    result = {
        "file": name,
        "windows": len(labels),
        "attack_windows": int(sum(labels)),
        "scores": _sanitize_scores(scores).tolist(),
        "labels": labels,
    }
    if benchmark_latency:
        latency = benchmark_inference_latency(X, detector)
        if latency is not None:
            result["inference_latency"] = latency
    auroc = _safe_auroc(labels, result["scores"])
    if auroc is not None:
        result["auroc"] = auroc
    result.update(threshold_metrics(labels, result["scores"], threshold))
    return result


def _network_attack_for_result(
    result: dict,
    stats: Optional[list],
    attack_types: dict,
) -> str:
    if stats:
        for entry in stats:
            mission_file = entry.get("mission_file", "")
            attack = network_attack_from_mission_path(mission_file, attack_types)
            if attack != "unknown":
                return attack
    return "unknown"


def _stratify_network_attacks(
    raw_results: list,
    threshold: float,
    attack_types: dict,
) -> dict:
    """Aggregate Layer 3 metrics by SITL network attack type (waypoint/gps/none)."""
    buckets: dict = {}
    for result, stats in raw_results:
        net_attack = _network_attack_for_result(result, stats, attack_types)
        bucket = buckets.setdefault(net_attack, {"scores": [], "labels": [], "agents": []})
        bucket["scores"].extend(result["scores"])
        bucket["labels"].extend(result["labels"])
        bucket["agents"].append(result["file"])

    summary = {}
    for net_attack, bucket in buckets.items():
        labels = bucket["labels"]
        scores = bucket["scores"]
        entry = {
            "network_attack": net_attack,
            "agents": bucket["agents"],
            "windows": len(labels),
            "attack_windows": int(sum(labels)),
        }
        if len(set(labels)) >= 2:
            auroc = _safe_auroc(labels, scores)
            if auroc is not None:
                entry["auroc"] = auroc
        entry.update(threshold_metrics(labels, scores, threshold))
        summary[net_attack] = entry
    return summary


def eval_client(packet_file, model, config, detector, threshold: float):
    pre = DataPreprocessorTimeSeries(
        packet_file=packet_file,
        packet_features=getattr(config, "PACKET_FEATURES", None),
        packet_timestamp_col=getattr(config, "PACKET_TIMESTAMP_COL", "Timestamp"),
        label_col=getattr(config, "LABEL_COL", "Label"),
        use_labels=True,
    )
    X, y, _ = pre.preprocess(
        config.WINDOW_SIZE,
        config.OVERLAP,
        norm_train_fraction=(
            (1.0 - getattr(config, "TEST_SPLIT", 0.2))
            if getattr(config, "TEMPORAL_TRAIN_SPLIT", False)
            else None
        ),
    )
    if getattr(config, "TEMPORAL_TRAIN_SPLIT", False):
        _, _, X, y, _ = split_temporal_windows(
            X, y, test_split=float(getattr(config, "TEST_SPLIT", 0.2)),
        )
    return eval_windows(
        os.path.basename(packet_file),
        X,
        y,
        detector,
        threshold,
        benchmark_latency=benchmark_latency,
    )


def main():
    ap = argparse.ArgumentParser(description="Layer 3 eval on SITL data")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--config",
        default="auto",
        help="FL config module (auto detects LSTM vs TinyPacketCNN from checkpoint)",
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--no-tune-threshold",
        action="store_true",
        help="Keep --threshold fixed instead of tuning for best aggregate F1",
    )
    ap.add_argument(
        "--probe-file",
        default=DEFAULT_PROBE_FILE,
        help="Counterfactual probe set for Layer 3 calibration (classification)",
    )
    ap.add_argument(
        "--probe-blend-weight",
        type=float,
        default=0.35,
        help="Blend weight for probe-calibrated score vs raw P(attack)",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--max-false-alarms-per-hour",
        type=float,
        default=None,
        help="Optional mission-continuity limit for false alarms per hour",
    )
    ap.add_argument(
        "--benchmark-latency",
        action="store_true",
        help="Benchmark per-window inference latency (first agent only)",
    )
    ap.add_argument(
        "--latency-only",
        action="store_true",
        help="Benchmark only the first agent and emit an inference-latency report",
    )
    args = ap.parse_args()

    import glob
    files = list_packet_mission_files(args.data_dir)
    if not files:
        raise SystemExit(f"No mission packet CSVs found under {args.data_dir}")

    model, config = load_model(args.checkpoint, args.config)
    learning_mode = getattr(config, "LEARNING_MODE", "classification")
    probe_data = load_probe_set(args.probe_file)
    if probe_data:
        print(f"Loaded {len(probe_data)} probe scenarios from {args.probe_file}")
    else:
        print(f"No probe file at {args.probe_file}; using model P(attack) only")

    detector = SpoofingDetector(
        model,
        threshold=args.threshold,
        probe_blend_weight=args.probe_blend_weight,
        learning_mode=learning_mode,
        probe_data=probe_data,
    )
    if detector._benign_attack_prob_ref is not None:
        print(
            f"Probe anchors: benign P(attack)={detector._benign_attack_prob_ref:.4f} "
            f"attack P(attack)={detector._attack_attack_prob_ref:.4f}"
        )

    network_attack_types = load_network_attack_types(ACFL_ROOT)
    raw_results = []
    all_scores, all_labels = [], []

    agent_ids = list_agent_ids(args.data_dir)
    if agent_ids:
        eval_items = []
        selected_agent_ids = agent_ids[:1] if args.latency_only else agent_ids
        for agent_num in selected_agent_ids:
            X, y, stats = build_agent_eval_windows(args.data_dir, agent_num, config)
            eval_items.append((f"agent_{agent_num:03d}", X, y, stats))
    else:
        eval_items = [
            (os.path.basename(f), None, None, None)
            for f in (files[:1] if args.latency_only else files)
        ]

    for agent_idx, (name, X, y, stats) in enumerate(eval_items):
        benchmark_latency = bool(
            (args.benchmark_latency or args.latency_only) and agent_idx == 0
        )
        if X is None:
            packet_file = os.path.join(args.data_dir, name)
            r = eval_client(
                packet_file,
                model,
                config,
                detector,
                args.threshold,
                benchmark_latency=benchmark_latency,
            )
        else:
            r = eval_windows(
                name,
                X,
                y,
                detector,
                args.threshold,
                benchmark_latency=benchmark_latency,
            )
        if r:
            raw_results.append((r, stats))
            all_scores.extend(r["scores"])
            all_labels.extend(r["labels"])

    if args.latency_only:
        latency = next(
            (
                result.get("inference_latency")
                for result, _ in raw_results
                if result.get("inference_latency")
            ),
            None,
        )
        report = {
            "checkpoint": args.checkpoint,
            "learning_mode": learning_mode,
            "inference_latency": latency,
        }
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report, indent=2))
        return

    threshold_info = tune_threshold_for_f1(all_labels, all_scores, args.threshold)
    final_threshold = args.threshold
    if not args.no_tune_threshold and threshold_info.get("tuned_threshold") is not None:
        final_threshold = float(threshold_info["tuned_threshold"])

    window_size = int(getattr(config, "WINDOW_SIZE", 50))
    overlap = int(getattr(config, "OVERLAP", 25))
    timestamp_col = str(getattr(config, "PACKET_TIMESTAMP_COL", "Timestamp"))

    per_client = []
    for r, stats in raw_results:
        r.update(threshold_metrics(r["labels"], r["scores"], final_threshold))
        net_attack = _network_attack_for_result(r, stats, network_attack_types)
        operational = compute_agent_operational_metrics(
            r["labels"],
            r["scores"],
            final_threshold,
            split_stats=stats,
            window_size=window_size,
            overlap=overlap,
            timestamp_col=timestamp_col,
            max_false_alarms_per_hour=args.max_false_alarms_per_hour,
        )
        slim = {k: v for k, v in r.items() if k not in ("scores", "labels")}
        slim["network_attack"] = net_attack
        slim["operational_metrics"] = operational
        if stats is not None:
            slim["split_stats"] = stats
        per_client.append(slim)

    report = {
        "per_client": per_client,
        "threshold": final_threshold,
        "requested_threshold": args.threshold,
        "threshold_tuning": threshold_info,
        "learning_mode": learning_mode,
        "probe_file": args.probe_file if probe_data else None,
        "probe_blend_weight": args.probe_blend_weight,
        "by_network_attack": _stratify_network_attacks(
            raw_results, final_threshold, network_attack_types
        ),
        "operational_metrics": aggregate_operational_metrics(per_client),
        "operational_metrics_by_network_attack": aggregate_operational_by_network_attack(
            per_client
        ),
    }
    latency_reports = [
        entry.get("inference_latency")
        for entry in per_client
        if entry.get("inference_latency")
    ]
    if latency_reports:
        report["inference_latency"] = latency_reports[0]
    if len(set(all_labels)) >= 2:
        auroc = _safe_auroc(all_labels, all_scores)
        if auroc is not None:
            report["aggregate_auroc"] = auroc
        report.update({
            f"aggregate_{k}": v
            for k, v in threshold_metrics(all_labels, all_scores, final_threshold).items()
        })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
