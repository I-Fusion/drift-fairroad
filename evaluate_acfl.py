"""
ACFL evaluation helper — runs poisoning sweeps and exports metrics.

Usage:
    python evaluate_acfl.py --checkpoint-dir fl-time-series/checkpoints
"""
import argparse
import json
import os
import sys

# Deliverable layout: this tree is intended to live at the root of
# I-Fusion/drift-fairroad (sibling to fl-time-series/ and data/).
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ACFL_ROOT = REPO_ROOT
DRIFT_ROOT = REPO_ROOT
FL_TS_DIR = os.path.join(REPO_ROOT, "fl-time-series")

for path in [REPO_ROOT, FL_TS_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)


def load_acfl_reports(checkpoint_dir: str) -> dict:
    """Load ACFL round reports and trust scores."""
    reports_path = os.path.join(checkpoint_dir, "acfl_round_reports.json")
    trust_path = os.path.join(checkpoint_dir, "acfl_trust_scores.json")

    result = {"reports": [], "trust_scores": []}
    if os.path.exists(reports_path):
        with open(reports_path) as f:
            result["reports"] = json.load(f)
    if os.path.exists(trust_path):
        with open(trust_path) as f:
            result["trust_scores"] = json.load(f)
    return result


def compute_trust_auc(
    trust_scores: list, malicious_clients: set, warmup_rounds: int = 2
) -> float:
    """Compute trust score AUC vs ground-truth malicious labels.

    Rounds with index <= warmup_rounds are excluded: during warm-up the global
    model has not converged, so a poisoned update is not yet distinguishable
    from benign ones and every client legitimately scores ~1.0.
    """
    from sklearn.metrics import roc_auc_score

    scores, labels = [], []
    for record in trust_scores:
        if record.get("round", 0) <= warmup_rounds:
            continue
        for cid, trust in record.get("trust_scores", {}).items():
            scores.append(trust)
            labels.append(1 if cid in malicious_clients else 0)

    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, [-s for s in scores]))


def compute_drift_stability(reports: list) -> dict:
    """Compute validation accuracy variance over final rounds."""
    if len(reports) < 2:
        return {"variance": None, "rounds": len(reports)}

    losses = [r.get("avg_loss", 0) for r in reports]
    tail = losses[-min(10, len(losses)):]
    import numpy as np
    return {
        "loss_variance_tail": float(np.var(tail)),
        "loss_mean_tail": float(np.mean(tail)),
        "total_rounds": len(reports),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACFL defense metrics")
    parser.add_argument(
        "--checkpoint-dir",
        default=os.path.join(FL_TS_DIR, "checkpoints"),
    )
    parser.add_argument(
        "--malicious-clients",
        default="",
        help="Comma-separated malicious client IDs",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ACFL_ROOT, "evaluation_results"),
    )
    parser.add_argument(
        "--run-drift-fairroad-eval",
        action="store_true",
        help="Also run drift-fairroad evaluate_fl_system.py",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--config", default="config_classification")
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=2,
        help="Exclude these initial (pre-convergence) rounds from trust AUC",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    malicious = set(c.strip() for c in args.malicious_clients.split(",") if c.strip())

    acfl_data = load_acfl_reports(args.checkpoint_dir)
    metrics = {
        "acfl_rounds": len(acfl_data["reports"]),
        "drift_stability": compute_drift_stability(acfl_data["reports"]),
    }

    if malicious and acfl_data["trust_scores"]:
        auc = compute_trust_auc(
            acfl_data["trust_scores"], malicious, warmup_rounds=args.warmup_rounds
        )
        metrics["trust_score_auc"] = auc
        metrics["trust_auc_warmup_excluded"] = args.warmup_rounds

    if acfl_data["reports"]:
        latest = acfl_data["reports"][-1]
        metrics["latest_trust_scores"] = latest.get("trust_scores", {})
        metrics["latest_trust_breakdown"] = latest.get("trust_breakdown", {})

    output_path = os.path.join(args.output_dir, "acfl_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"ACFL evaluation saved to {output_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.run_drift_fairroad_eval and args.data_dir:
        import subprocess
        eval_script = os.path.join(FL_TS_DIR, "evaluate_fl_system.py")
        cmd = [
            sys.executable, eval_script,
            "--data-dir", args.data_dir,
            "--task", "classification",
            "--config", args.config,
            "--checkpoint-dir", args.checkpoint_dir,
            "--output-dir", os.path.join(args.output_dir, "drift_fairroad"),
        ]
        print(f"\nRunning drift-fairroad evaluation: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=DRIFT_ROOT)


if __name__ == "__main__":
    main()
