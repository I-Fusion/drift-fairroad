"""
Comprehensive Evaluation Script for Federated Learning CAE (Convolutional Autoencoder).

Evaluation dimensions (aligned with root evaluate_fl_system.py):

1. Reconstruction quality and responsiveness
   - MSE, SSIM (overall and per-client), best/final round

2. Robustness and non-IID resilience
   - Client heterogeneity (MSE/SSIM variance across clients), consistency

3. Drift resistance and model stability
   - Drift: MSE/SSIM on fixed validation set at round 0 vs later rounds
   - Convergence stability: round at which metric stabilizes, variance in tail rounds

4. Mission continuity
   - Optional: fraction of samples below reconstruction error threshold

5. Efficiency and practicality
   - Model size (parameters, MB), checkpoint size, communication overhead per round

Outputs: evaluation_report.json, evaluation_report.txt, and plots (metrics over rounds,
client heterogeneity, drift).
"""
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_preprocessing_cae import CAEDataPreprocessor
import config


def _round_key(name: str) -> int:
    """Extract round number from server_round_N.pt."""
    base = name.replace('server_round_', '').replace('.pt', '')
    return int(base) if base.isdigit() else 0


def calculate_mse(img1, img2):
    """MSE between two images (tensors or numpy)."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    return float(np.mean((img1 - img2) ** 2))


def calculate_ssim(img1, img2):
    """SSIM between two images (C,H,W) or (H,W,C)."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    if img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    return float(ssim(
        img1, img2,
        multichannel=True,
        channel_axis=2,
        data_range=1.0,
    ))


class CAEEvaluator:
    """Comprehensive evaluator for FL CAE (single checkpoint or full checkpoint_dir)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Args:
            model_path: Single checkpoint path (optional).
            checkpoint_dir: Directory with server_round_*.pt for multi-round evaluation (optional).
        """
        self.model_path = model_path
        self.checkpoint_dir = checkpoint_dir
        self.model_module = importlib.import_module(config.MODEL_PATH)
        self.model_class = getattr(self.model_module, config.MODEL_CLASS)

        self.model = self.model_class(**config.MODEL_CONFIG)
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model.load(model_path)

        self.device = self.model.device
        self.checkpoints: Dict[int, str] = {}  # round -> path
        self.evaluation_results: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)  # round -> client_id -> metrics
        self._param_count: Optional[int] = None

        print(f"Model loaded on {self.device}")
        print(f"Parameters: {self.model.count_parameters():,}")

    def _get_param_count(self) -> int:
        if self._param_count is None:
            self._param_count = self.model.count_parameters()
        return self._param_count

    def load_checkpoints(self, rounds: Optional[List[int]] = None) -> None:
        """Load checkpoint paths from checkpoint_dir (server_round_*.pt)."""
        if not self.checkpoint_dir or not os.path.isdir(self.checkpoint_dir):
            return
        pat = re.compile(r"server_round_(\d+)\.pt")
        for f in os.listdir(self.checkpoint_dir):
            m = pat.match(f)
            if m:
                r = int(m.group(1))
                if rounds is None or r in rounds:
                    self.checkpoints[r] = os.path.join(self.checkpoint_dir, f)
        self.checkpoints = dict(sorted(self.checkpoints.items()))

    def evaluate_on_dataset(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate current model on a dataloader. Returns mse, ssim, num_images."""
        self.model.model.eval()
        total_mse = 0.0
        total_ssim = 0.0
        num_images = 0
        with torch.no_grad():
            for batch_idx, (noisy_batch, clean_batch) in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                noisy_batch = noisy_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)
                reconstructed_batch = self.model.model(noisy_batch)
                batch_size = noisy_batch.size(0)
                for i in range(batch_size):
                    total_mse += calculate_mse(reconstructed_batch[i], clean_batch[i])
                    total_ssim += calculate_ssim(reconstructed_batch[i], clean_batch[i])
                    num_images += 1
        n = max(num_images, 1)
        return {
            "mse": total_mse / n,
            "ssim": total_ssim / n,
            "num_images": num_images,
        }

    def evaluate_all_clients(
        self,
        clean_dir: str,
        noisy_dir: str,
        num_clients: int = 3,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate current model on all clients. Returns {client_1: {mse, ssim, num_images}, ...}."""
        preprocessor = CAEDataPreprocessor(
            clean_dir=clean_dir,
            noisy_dir=noisy_dir,
            img_size=config.AUTOENCODER_IMAGE_SIZE,
            num_clients=num_clients,
        )
        all_results = {}
        for cid in range(num_clients):
            dataloader, _ = preprocessor.get_dataloader(
                client_id=cid,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
            )
            res = self.evaluate_on_dataset(dataloader)
            all_results[f"client_{cid + 1}"] = res
        return all_results

    def get_fixed_dataloader(
        self,
        clean_dir: str,
        noisy_dir: str,
        num_clients: int = 3,
        client_id: int = 0,
        max_batches: Optional[int] = None,
    ) -> DataLoader:
        """Return a single fixed dataloader for drift/convergence (same data every round)."""
        preprocessor = CAEDataPreprocessor(
            clean_dir=clean_dir,
            noisy_dir=noisy_dir,
            img_size=config.AUTOENCODER_IMAGE_SIZE,
            num_clients=num_clients,
        )
        dl, _ = preprocessor.get_dataloader(
            client_id=client_id,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
        )
        return dl

    def evaluate_all_checkpoints(
        self,
        clean_dir: str,
        noisy_dir: str,
        num_clients: int = 3,
        rounds: Optional[List[int]] = None,
        evaluate_clients: bool = True,
    ) -> None:
        """
        Evaluate each checkpoint in checkpoint_dir. Fills self.evaluation_results.
        Uses a fixed dataloader (client 0) for drift; optionally per-client for heterogeneity.
        """
        if not self.checkpoints:
            print("No checkpoints to evaluate.")
            return
        fixed_dl = self.get_fixed_dataloader(clean_dir, noisy_dir, num_clients, client_id=0)
        round_list = sorted(self.checkpoints.keys())
        for round_num in round_list:
            path = self.checkpoints[round_num]
            self.model.load(path)
            # Fixed validation set (for drift/convergence)
            res_fixed = self.evaluate_on_dataset(fixed_dl)
            self.evaluation_results[round_num]["fixed_validation"] = res_fixed
            if evaluate_clients:
                client_results = self.evaluate_all_clients(clean_dir, noisy_dir, num_clients)
                for cid, data in client_results.items():
                    self.evaluation_results[round_num][cid] = data
            # print(f"  Round {round_num}: MSE={res_fixed['mse']:.6f}, SSIM={res_fixed['ssim']:.4f}")
        # Restore last checkpoint for any subsequent single-model use
        if round_list:
            self.model.load(self.checkpoints[round_list[-1]])

    def compute_summary(self) -> Dict[str, Any]:
        """Summary: best round (by MSE/SSIM), final MSE/SSIM."""
        if not self.evaluation_results:
            return {}
        rounds = sorted(self.evaluation_results.keys())
        key = "fixed_validation" if "fixed_validation" in self.evaluation_results[rounds[0]] else next(iter(self.evaluation_results[rounds[0]].keys()))
        best_mse_round = min(rounds, key=lambda r: self.evaluation_results[r][key]["mse"])
        best_ssim_round = max(rounds, key=lambda r: self.evaluation_results[r][key]["ssim"])
        last = rounds[-1]
        return {
            "best_round_mse": best_mse_round,
            "best_round_ssim": best_ssim_round,
            "best_mse": self.evaluation_results[best_mse_round][key]["mse"],
            "best_ssim": self.evaluation_results[best_ssim_round][key]["ssim"],
            "final_round": last,
            "final_mse": self.evaluation_results[last][key]["mse"],
            "final_ssim": self.evaluation_results[last][key]["ssim"],
            "n_rounds": len(rounds),
        }

    def compute_drift_metrics(self, metric: str = "mse") -> Dict[str, Any]:
        """Drift: metric on fixed set at round 0 vs each round. For MSE, positive drift = got worse."""
        if not self.evaluation_results:
            return {}
        rounds = sorted(self.evaluation_results.keys())
        key = "fixed_validation" if "fixed_validation" in self.evaluation_results[rounds[0]] else next(iter(self.evaluation_results[rounds[0]].keys()))
        val_at_0 = self.evaluation_results[rounds[0]][key][metric]
        drift_per_round = {}
        for r in rounds:
            v = self.evaluation_results[r][key][metric]
            if metric == "ssim":
                drift_per_round[r] = float(val_at_0 - v)  # positive = SSIM dropped
            else:
                drift_per_round[r] = float(v - val_at_0)  # positive = MSE increased
        last = rounds[-1]
        return {
            "reference": key,
            "metric": metric,
            f"{metric}_at_round_0": float(val_at_0),
            f"{metric}_at_final_round": float(self.evaluation_results[last][key][metric]),
            "drift_at_final_round": float(drift_per_round[last]),
            "drift_per_round": {int(k): float(v) for k, v in drift_per_round.items()},
        }

    def compute_convergence_stability(
        self,
        metric: str = "mse",
        epsilon: float = 0.01,
        tail_rounds: int = 5,
    ) -> Dict[str, Any]:
        """Round at which metric stabilizes within epsilon of final; variance in last tail_rounds."""
        if not self.evaluation_results:
            return {}
        rounds = sorted(self.evaluation_results.keys())
        key = "fixed_validation" if "fixed_validation" in self.evaluation_results[rounds[0]] else next(iter(self.evaluation_results[rounds[0]].keys()))
        values = [(r, self.evaluation_results[r][key][metric]) for r in rounds]
        if len(values) < 2:
            return {}
        _, vals = zip(*values)
        final_val = vals[-1]
        stable_round = None
        for r, v in values:
            if abs(v - final_val) <= epsilon:
                stable_round = r
                break
        tail = values[-min(tail_rounds, len(values)):]
        tail_vals = [v for _, v in tail]
        return {
            "metric": metric,
            "final_value": float(final_val),
            "round_stabilized": stable_round,
            "epsilon": epsilon,
            "variance_last_n_rounds": float(np.var(tail_vals)) if len(tail_vals) > 1 else 0.0,
            "std_last_n_rounds": float(np.std(tail_vals)) if len(tail_vals) > 1 else 0.0,
        }

    def compute_efficiency_metrics(self) -> Dict[str, Any]:
        """Model size, checkpoint file sizes, communication overhead."""
        n_params = self._get_param_count()
        bytes_per_param = 4
        model_size_mb = (n_params * bytes_per_param) / (1024 * 1024)
        out = {
            "total_parameters": n_params,
            "model_size_mb": round(model_size_mb, 4),
        }
        if self.checkpoint_dir:
            paths = list(Path(self.checkpoint_dir).glob("server_round_*.pt"))
            paths = sorted(paths, key=lambda p: _round_key(p.name))
            file_sizes = []
            for p in paths:
                try:
                    file_sizes.append(p.stat().st_size)
                except OSError:
                    pass
            if file_sizes:
                out["checkpoint_file_sizes_bytes"] = file_sizes
                out["mean_checkpoint_size_mb"] = round(np.mean(file_sizes) / (1024 * 1024), 4)
                out["communication_overhead_per_round_mb"] = round(
                    (np.mean(file_sizes) * 2 * getattr(config, "NUM_CLIENTS", 3)) / (1024 * 1024), 4
                )
        return out

    def calculate_client_statistics(self) -> Dict[str, Dict[str, float]]:
        """Per-metric stats across clients (mean, std, min, max) for last round."""
        if not self.evaluation_results:
            return {}
        rounds = sorted(self.evaluation_results.keys())
        last = rounds[-1]
        data = self.evaluation_results[last]
        client_keys = [k for k in data if k.startswith("client_")]
        if not client_keys:
            return {}
        stats = {}
        for metric in ("mse", "ssim"):
            vals = [data[c][metric] for c in client_keys]
            stats[metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        return stats

    def compute_non_iid_resilience(self) -> Dict[str, Any]:
        """Non-IID resilience from client MSE/SSIM variance (1 - coefficient of variation)."""
        stats = self.calculate_client_statistics()
        if not stats:
            return {}
        out = {}
        for metric in ("mse", "ssim"):
            m, s = stats[metric]["mean"], stats[metric]["std"]
            if m <= 0:
                cv = 1.0
            else:
                cv = s / m
            # For MSE lower is better: resilience = 1 - cv (high variance = low resilience)
            # For SSIM higher is better: same 1 - cv
            out[f"{metric}_cv"] = float(cv)
            out[f"{metric}_mean"] = float(m)
            out[f"{metric}_std"] = float(s)
        out["non_iid_resilience_mse"] = float(max(0, min(1, 1 - stats["mse"]["std"] / max(stats["mse"]["mean"], 1e-8))))
        out["non_iid_resilience_ssim"] = float(max(0, min(1, 1 - stats["ssim"]["std"] / max(1 - stats["ssim"]["mean"], 1e-8))))
        return out

    def plot_metrics_over_rounds(self, output_dir: str) -> None:
        """Plot MSE and SSIM over rounds (fixed validation set)."""
        if not self.evaluation_results:
            return
        os.makedirs(output_dir, exist_ok=True)
        rounds = sorted(self.evaluation_results.keys())
        key = "fixed_validation" if "fixed_validation" in self.evaluation_results[rounds[0]] else next(iter(self.evaluation_results[rounds[0]].keys()))
        mse_vals = [self.evaluation_results[r][key]["mse"] for r in rounds]
        ssim_vals = [self.evaluation_results[r][key]["ssim"] for r in rounds]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(rounds, mse_vals, "o-", linewidth=2, markersize=4)
        ax1.set_xlabel("Round")
        ax1.set_ylabel("MSE")
        ax1.set_title("Reconstruction MSE over Rounds (fixed validation set)")
        ax1.grid(True, alpha=0.3)
        ax2.plot(rounds, ssim_vals, "o-", linewidth=2, markersize=4, color="green")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("SSIM")
        ax2.set_title("Reconstruction SSIM over Rounds (fixed validation set)")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "cae_metrics_over_rounds.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        # print(f"  Saved: {path}")

    def plot_client_heterogeneity(self, output_dir: str) -> None:
        """Plot per-client MSE/SSIM for last round."""
        if not self.evaluation_results:
            return
        rounds = sorted(self.evaluation_results.keys())
        last = rounds[-1]
        data = self.evaluation_results[last]
        client_keys = sorted([k for k in data if k.startswith("client_")])
        if not client_keys:
            return
        os.makedirs(output_dir, exist_ok=True)
        mse_vals = [data[c]["mse"] for c in client_keys]
        ssim_vals = [data[c]["ssim"] for c in client_keys]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(client_keys, mse_vals, color="steelblue", alpha=0.8)
        ax1.set_ylabel("MSE")
        ax1.set_title("MSE per Client (last round)")
        ax1.tick_params(axis="x", rotation=15)
        ax2.bar(client_keys, ssim_vals, color="seagreen", alpha=0.8)
        ax2.set_ylabel("SSIM")
        ax2.set_title("SSIM per Client (last round)")
        ax2.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        path = os.path.join(output_dir, "cae_client_heterogeneity.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        # print(f"  Saved: {path}")

    def plot_drift(self, output_dir: str, metric: str = "mse") -> None:
        """Plot drift over rounds (metric change from round 0)."""
        drift = self.compute_drift_metrics(metric=metric)
        if not drift or "drift_per_round" not in drift:
            return
        os.makedirs(output_dir, exist_ok=True)
        rounds = sorted(drift["drift_per_round"].keys())
        drifts = [drift["drift_per_round"][r] for r in rounds]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rounds, drifts, "o-", linewidth=2, markersize=4, color="coral")
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xlabel("Round")
        ax.set_ylabel(f"{metric.upper()} drift (vs round 0)")
        ax.set_title(f"CAE {metric.upper()} drift on fixed validation set (positive = worse than round 0)")
        ax.grid(True, alpha=0.3)
        path = os.path.join(output_dir, "cae_drift_over_rounds.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        # print(f"  Saved: {path}")

    def generate_report(self, output_dir: str) -> None:
        """Write evaluation_report.json and evaluation_report.txt (same structure as root)."""
        os.makedirs(output_dir, exist_ok=True)
        rounds = sorted(self.evaluation_results.keys()) if self.evaluation_results else []
        summary = self.compute_summary()
        drift_mse = self.compute_drift_metrics("mse")
        drift_ssim = self.compute_drift_metrics("ssim")
        convergence_mse = self.compute_convergence_stability("mse")
        convergence_ssim = self.compute_convergence_stability("ssim")
        efficiency = self.compute_efficiency_metrics()
        non_iid = self.compute_non_iid_resilience()
        client_stats = self.calculate_client_statistics()

        report = {
            "task": "autoencoder",
            "checkpoint_dir": self.checkpoint_dir or "",
            "model_path": self.model_path or "",
            "num_checkpoints": len(self.checkpoints),
            "rounds_evaluated": rounds,
            "summary": summary,
            "reconstruction_quality": {
                "final_mse": summary.get("final_mse"),
                "final_ssim": summary.get("final_ssim"),
                "best_round_mse": summary.get("best_round_mse"),
                "best_round_ssim": summary.get("best_round_ssim"),
            },
            "drift_resistance": {"mse": drift_mse, "ssim": drift_ssim},
            "convergence_stability": {"mse": convergence_mse, "ssim": convergence_ssim},
            "mission_continuity": {"note": "N/A for reconstruction (no false alarms)"},
            "efficiency": efficiency,
            "non_iid_resilience": non_iid,
            "client_heterogeneity": client_stats,
        }
        if self.evaluation_results:
            report["evaluation_results"] = {
                int(r): {
                    k: {x: float(v) if isinstance(v, (np.floating, np.integer)) else v for x, v in vv.items()}
                    for k, vv in self.evaluation_results[r].items()
                }
                for r in rounds
            }

        json_path = os.path.join(output_dir, "evaluation_report.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        # print(f"  Saved: {json_path}")

        txt_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(txt_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("FEDERATED LEARNING CAE EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Task: autoencoder\n")
            f.write(f"Checkpoint directory: {self.checkpoint_dir or 'N/A'}\n")
            f.write(f"Model path: {self.model_path or 'N/A'}\n")
            f.write(f"Rounds evaluated: {min(rounds) if rounds else 0} to {max(rounds) if rounds else 0}\n\n")

            f.write("=" * 70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 70 + "\n\n")
            if summary:
                f.write(f"  Best round (MSE):  {summary.get('best_round_mse')}\n")
                f.write(f"  Best MSE:          {summary.get('best_mse', 0):.6f}\n")
                f.write(f"  Best round (SSIM): {summary.get('best_round_ssim')}\n")
                f.write(f"  Best SSIM:         {summary.get('best_ssim', 0):.4f}\n")
                f.write(f"  Final round:       {summary.get('final_round')}\n")
                f.write(f"  Final MSE:         {summary.get('final_mse', 0):.6f}\n")
                f.write(f"  Final SSIM:        {summary.get('final_ssim', 0):.4f}\n\n")

            f.write("=" * 70 + "\n")
            f.write("DRIFT RESISTANCE\n")
            f.write("=" * 70 + "\n\n")
            if drift_mse:
                f.write(f"  MSE at round 0: {drift_mse.get('mse_at_round_0', 0):.6f}\n")
                f.write(f"  MSE at final:   {drift_mse.get('mse_at_final_round', 0):.6f}\n")
                f.write(f"  MSE drift (final): {drift_mse.get('drift_at_final_round', 0):.6f}\n\n")
            if drift_ssim:
                f.write(f"  SSIM at round 0: {drift_ssim.get('ssim_at_round_0', 0):.4f}\n")
                f.write(f"  SSIM at final:   {drift_ssim.get('ssim_at_final_round', 0):.4f}\n\n")

            f.write("=" * 70 + "\n")
            f.write("CONVERGENCE STABILITY\n")
            f.write("=" * 70 + "\n\n")
            if convergence_mse:
                f.write(f"  MSE round stabilized: {convergence_mse.get('round_stabilized')}\n")
                f.write(f"  MSE variance (last n rounds): {convergence_mse.get('variance_last_n_rounds', 0):.6f}\n\n")
            if convergence_ssim:
                f.write(f"  SSIM round stabilized: {convergence_ssim.get('round_stabilized')}\n\n")

            f.write("=" * 70 + "\n")
            f.write("EFFICIENCY\n")
            f.write("=" * 70 + "\n\n")
            for k, v in efficiency.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("NON-IID RESILIENCE (CLIENT HETEROGENEITY)\n")
            f.write("=" * 70 + "\n\n")
            for k, v in non_iid.items():
                f.write(f"  {k}: {v}\n")
        # print(f"  Saved: {txt_path}")

    def run_full_evaluation(
        self,
        clean_dir: str,
        noisy_dir: str,
        num_clients: int = 3,
        output_dir: str = "evaluation_results",
        rounds: Optional[List[int]] = None,
        evaluate_clients: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full evaluation pipeline: load checkpoints, evaluate, plot, report.
        If checkpoint_dir was not set, uses single model_path and only runs one-round metrics/report.
        """
        print("\n" + "=" * 70)
        print("COMPREHENSIVE CAE FL EVALUATION")
        print("=" * 70)

        if self.checkpoint_dir:
            self.load_checkpoints(rounds=rounds)
            if not self.checkpoints:
                print("No checkpoints found. Exiting.")
                return {}
            self.evaluate_all_checkpoints(
                clean_dir=clean_dir,
                noisy_dir=noisy_dir,
                num_clients=num_clients,
                rounds=rounds,
                evaluate_clients=evaluate_clients,
            )
            self.plot_metrics_over_rounds(output_dir)
            self.plot_client_heterogeneity(output_dir)
            self.plot_drift(output_dir, metric="mse")
        else:
            # Single checkpoint: one-round evaluation
            if self.model_path:
                client_results = self.evaluate_all_clients(clean_dir, noisy_dir, num_clients)
                r = 1
                self.evaluation_results[r] = {"fixed_validation": client_results.get("client_1", {})}
                for cid, data in client_results.items():
                    self.evaluation_results[r][cid] = data
                self.checkpoints[r] = self.model_path

        self.generate_report(output_dir)
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {output_dir}/")
        print("  - evaluation_report.json")
        print("  - evaluation_report.txt")
        if self.checkpoints:
            print("  - cae_metrics_over_rounds.png")
            print("  - cae_client_heterogeneity.png")
            print("  - cae_drift_over_rounds.png")
        return self.compute_summary()

    def save_reconstructions(self, dataloader: DataLoader, output_dir: str, num_samples: int = 10) -> None:
        """Save sample reconstructions (noisy, reconstructed, clean)."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.model.eval()
        saved = 0
        with torch.no_grad():
            for noisy_batch, clean_batch in dataloader:
                if saved >= num_samples:
                    break
                noisy_batch = noisy_batch.to(self.device)
                rec = self.model.model(noisy_batch)
                for i in range(noisy_batch.size(0)):
                    if saved >= num_samples:
                        break
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    for ax, img, title in [
                        (axes[0], noisy_batch[i].cpu().permute(1, 2, 0).numpy(), "Noisy"),
                        (axes[1], rec[i].cpu().permute(1, 2, 0).numpy(), "Reconstructed"),
                        (axes[2], clean_batch[i].permute(1, 2, 0).numpy(), "Clean"),
                    ]:
                        ax.imshow(img)
                        ax.set_title(title)
                        ax.axis("off")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"reconstruction_{saved + 1}.png"), dpi=100)
                    plt.close()
                    saved += 1
        # print(f"  Saved {saved} reconstructions to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive CAE FL evaluation (aligned with evaluate_fl_system dimensions)")
    parser.add_argument("--model-path", type=str, default=None, help="Single checkpoint path")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory with server_round_*.pt for multi-round evaluation")
    parser.add_argument("--clean-dir", type=str, default="../data/payload/images/clean", help="Clean images directory")
    parser.add_argument("--noisy-dir", type=str, default="../data/payload/images/noise", help="Noisy images directory")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory for reports and plots")
    parser.add_argument("--rounds", type=int, nargs="+", default=None, help="Specific rounds to evaluate (default: all)")
    parser.add_argument("--no-client-eval", action="store_true", help="Skip per-client evaluation (faster)")
    parser.add_argument("--save-reconstructions", action="store_true", help="Save sample reconstructions")
    args = parser.parse_args()

    if not args.model_path and not args.checkpoint_dir:
        parser.error("Provide either --model-path or --checkpoint-dir")

    evaluator = CAEEvaluator(model_path=args.model_path, checkpoint_dir=args.checkpoint_dir)
    evaluator.run_full_evaluation(
        clean_dir=args.clean_dir,
        noisy_dir=args.noisy_dir,
        num_clients=args.num_clients,
        output_dir=args.output_dir,
        rounds=args.rounds,
        evaluate_clients=not args.no_client_eval,
    )

    if args.save_reconstructions:
        preprocessor = CAEDataPreprocessor(
            clean_dir=args.clean_dir,
            noisy_dir=args.noisy_dir,
            img_size=config.AUTOENCODER_IMAGE_SIZE,
            num_clients=1,
        )
        dl, _ = preprocessor.get_dataloader(client_id=0, batch_size=8, shuffle=False)
        evaluator.save_reconstructions(dl, os.path.join(args.output_dir, "reconstructions"), num_samples=10)


if __name__ == "__main__":
    main()
