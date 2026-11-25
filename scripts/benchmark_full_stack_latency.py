#!/usr/bin/env python3
"""Benchmark shared Layer~3 per-window CPU latency without parallel contention.

All operational-table defenders use the same TinyPacketCNN architecture and
SpoofingDetector path, so latency is an implementation property rather than a
defender/attack metric. This script benchmarks one canonical checkpoint
sequentially and reports variation across repeated runs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _repo_roots() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    if (root / "fl-time-series").is_dir():
        return root, root
    nested = root / "drift-fairroad"
    if nested.is_dir():
        return root, nested
    return root, root

ACFL_ROOT, DRIFT_ROOT = _repo_roots()
ROOT = ACFL_ROOT
DEFAULT_CHECKPOINT = (
    ROOT
    / "evaluation_results"
    / "comprehensive_validation"
    / "journal_backdoor_seed42_krum"
    / "server_checkpoint.pt"
)
DEFAULT_OUTPUT = (
    ROOT / "evaluation_results" / "full_stack" / "full_stack_latency.json"
)
DATA_DIR = DRIFT_ROOT / "data" / "sitl_dataset" / "fl" / "packets"
PROBE_FILE = DRIFT_ROOT / "fl-time-series" / "checkpoints" / "acfl_probes_sitl.pt"


def _remove_per_checkpoint_latency() -> int:
    """Remove prior parallel timings, which are invalid under CPU contention."""
    report_dirs = (
        ROOT / "evaluation_results" / "full_stack" / "layer3",
        ROOT / "evaluation_results" / "combined_layer3" / "layer3",
    )
    removed = 0
    for report_dir in report_dirs:
        for path in report_dir.glob("*.json"):
            report = json.loads(path.read_text())
            if report.pop("inference_latency", None) is not None:
                path.write_text(json.dumps(report, indent=2) + "\n")
                removed += 1
    return removed


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    removed = _remove_per_checkpoint_latency()
    if removed:
        print(f"Removed {removed} contention-affected per-checkpoint timings")

    runs = []
    for repeat in range(args.repeats):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            report_path = Path(handle.name)
        try:
            cmd = [
                str(ROOT / "acfl_env" / "bin" / "python"),
                str(ROOT / "evaluate_layer3_sitl.py"),
                "--data-dir",
                str(DATA_DIR),
                "--config",
                "config_classification_drone",
                "--checkpoint",
                str(args.checkpoint),
                "--probe-file",
                str(PROBE_FILE),
                "--probe-blend-weight",
                "0.35",
                "--latency-only",
                "--output",
                str(report_path),
            ]
            subprocess.run(
                cmd,
                cwd=str(ROOT),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            latency = json.loads(report_path.read_text())["inference_latency"]
            runs.append(latency)
            print(
                f"[{repeat + 1}/{args.repeats}] "
                f"mean={latency['mean_ms_per_window']:.3f} ms "
                f"p95={latency['p95_ms_per_window']:.3f} ms",
                flush=True,
            )
        finally:
            report_path.unlink(missing_ok=True)

    payload = {
        "scope": (
            "Shared TinyPacketCNN + SpoofingDetector inference path; "
            "latency is not defender- or attack-specific"
        ),
        "device": "CPU",
        "checkpoint": str(args.checkpoint),
        "probe_file": str(PROBE_FILE),
        "probe_blend_weight": 0.35,
        "repeats": args.repeats,
        "windows_per_repeat": runs[0]["windows_benchmarked"],
        "mean_ms_per_window": _mean_std(
            [run["mean_ms_per_window"] for run in runs]
        ),
        "median_ms_per_window": _mean_std(
            [run["median_ms_per_window"] for run in runs]
        ),
        "p95_ms_per_window": _mean_std(
            [run["p95_ms_per_window"] for run in runs]
        ),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
