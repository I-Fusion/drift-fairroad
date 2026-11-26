#!/usr/bin/env python3
"""Re-score Layer 3 reports that lack operational metrics.

Seeds 42--44 of Layer~1-only and Layer~2-only on Byzantine, gradient scaling,
and backdoor were written before time-to-detect and false alarms per hour were
stored. This reruns ``evaluate_layer3_sitl.py`` with the same probe blend as
the later seeds and replaces the report only when aggregate AUROC matches.
"""
from __future__ import annotations

import json
import subprocess
import sys
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
LAYER3 = ROOT / "evaluation_results" / "full_stack" / "layer3"
CKPT_ROOT = ROOT / "evaluation_results" / "comprehensive_validation"
DATA_DIR = DRIFT_ROOT / "data" / "sitl_dataset" / "fl" / "packets"
PROBE = DRIFT_ROOT / "fl-time-series" / "checkpoints" / "acfl_probes_sitl.pt"
PYTHON = ROOT / "acfl_env" / "bin" / "python"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)
TMP = ROOT / "evaluation_results" / "full_stack" / "layer3_ops_fill"

STACK_DIR = {
    "l1_only": "ablation_acfl_no_layer2",
    "l2_only": "ablation_acfl_no_layer1",
}
ATTACKS = ("byzantine", "gradient_scaling", "backdoor")
SEEDS = (42, 43, 44)
AUROC_TOL = 1e-6


def _cells():
    for stack, prefix in STACK_DIR.items():
        for attack in ATTACKS:
            for seed in SEEDS:
                report = LAYER3 / f"{stack}_{attack}_seed{seed}.json"
                ckpt = CKPT_ROOT / f"{prefix}_{attack}_seed{seed}" / "server_checkpoint.pt"
                yield stack, attack, seed, report, ckpt


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    failures = 0
    for stack, attack, seed, report, ckpt in _cells():
        tag = f"{stack}_{attack}_seed{seed}"
        if not report.is_file() or not ckpt.is_file():
            print(f"MISSING {tag}")
            failures += 1
            continue
        old = json.loads(report.read_text())
        if old.get("operational_metrics") and old.get("operational_metrics_by_network_attack"):
            print(f"SKIP {tag}: operational metrics already present")
            continue
        side = TMP / f"{tag}.json"
        cmd = [
            str(PYTHON), "-u", str(ROOT / "evaluate_layer3_sitl.py"),
            "--data-dir", str(DATA_DIR),
            "--config", "config_classification_drone",
            "--checkpoint", str(ckpt),
            "--probe-file", str(PROBE),
            "--probe-blend-weight", "0.35",
            "--output", str(side),
        ]
        log_path = TMP / f"{tag}.log"
        print(f"RUN {tag}", flush=True)
        with log_path.open("w") as log:
            rc = subprocess.run(cmd, cwd=str(ROOT), check=False, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not side.is_file():
            print(f"FAIL {tag}: rc={rc}")
            failures += 1
            continue
        new = json.loads(side.read_text())
        old_au = old.get("aggregate_auroc")
        new_au = new.get("aggregate_auroc")
        if old_au is None or new_au is None or abs(float(old_au) - float(new_au)) > AUROC_TOL:
            print(f"FAIL {tag}: AUROC {old_au} -> {new_au}")
            failures += 1
            continue
        report.write_text(json.dumps(new, indent=2) + "\n")
        ttd = (new.get("operational_metrics") or {}).get("time_to_detect") or {}
        fa = (new.get("operational_metrics") or {}).get("false_alarms") or {}
        print(
            f"OK {tag} auroc={new_au:.6f} "
            f"ttd={ttd.get('mean_delay_seconds')} "
            f"fa/hr={fa.get('benign_missions_mean_false_alarms_per_hour')}",
            flush=True,
        )
    print(f"DONE failures={failures}", flush=True)
    return failures


if __name__ == "__main__":
    sys.exit(main())
