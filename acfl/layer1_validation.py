"""
Layer 1: Lightweight Counterfactual Update Validation and Poisoning Detection.

Components:
  1) Subspace Projection Analysis (randomized SVD)
  2) Round-Consistency Temporal Scoring
  3) Counterfactual attack-probe update validation
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from acfl.config import ACFLConfig
from acfl.weight_utils import compute_update_delta, l2_norm, norm_ratio_threshold


class UpdateValidationModule:
    """Pre-aggregation update validation for ACFL Layer 1."""

    def __init__(self, config: ACFLConfig):
        self.config = config
        self.client_update_history: Dict[str, List[np.ndarray]] = {}
        self.trust_ema: Dict[str, float] = {}
        self.round_number = 0

    def score_updates(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        client_losses: Optional[Dict[str, float]] = None,
        probe_evaluator: Optional["ProbeEvaluator"] = None,
    ) -> Dict[str, float]:
        """
        Compute per-client trust scores in [min_trust_weight, 1.0].

        Lower scores indicate higher poisoning risk.
        """
        deltas = [
            compute_update_delta(cw, global_weights) for cw in client_weights
        ]

        subspace_scores = self._subspace_projection_scores(deltas)
        temporal_scores = self._temporal_consistency_scores(client_ids, deltas)

        # Counterfactual probe scoring is only meaningful when a probe set is
        # available and its weight is positive. Without it, its weight is
        # redistributed across the other signals so an unavailable signal does
        # not impose a high trust floor on poisoned updates.
        w_sub = float(self.config.subspace_weight)
        w_tmp = float(self.config.temporal_consistency_weight)
        w_cf_cfg = float(self.config.counterfactual_weight)
        has_cf = probe_evaluator is not None and w_cf_cfg > 0.0
        cf_scores = {cid: 1.0 for cid in client_ids}
        if has_cf:
            cf_scores = probe_evaluator.counterfactual_scores(
                client_ids, client_weights, global_weights
            )

        w_cf = w_cf_cfg if has_cf else 0.0
        total_w = w_sub + w_tmp + w_cf
        if total_w <= 0:
            total_w = 1.0
        w_sub, w_tmp, w_cf = w_sub / total_w, w_tmp / total_w, w_cf / total_w

        trust = {}
        for i, cid in enumerate(client_ids):
            raw = (
                w_sub * subspace_scores[i]
                + w_tmp * temporal_scores[i]
                + w_cf * cf_scores[cid]
            )
            trust[cid] = self._soft_filter(raw)

        # EMA smoothing: one benign round should not fully rehabilitate a poisoner.
        alpha = 0.35
        for cid in client_ids:
            prev = self.trust_ema.get(cid, trust[cid])
            trust[cid] = float(alpha * trust[cid] + (1.0 - alpha) * prev)
            self.trust_ema[cid] = trust[cid]

        for cid, delta in zip(client_ids, deltas):
            self.client_update_history.setdefault(cid, []).append(delta.copy())

        self.round_number += 1
        return trust

    def _subspace_projection_scores(self, deltas: List[np.ndarray]) -> List[float]:
        """Robust subspace + norm anomaly scoring.

        Combines two complementary signals that are reliable even when only a
        few clients participate and one is an outlier:

          * Subspace deviation from the coordinate-wise *median* update (a
            robust reference that, unlike the mean, is not dragged toward an
            outlier), normalized by the median-absolute-deviation (MAD).
          * Update-norm ratio relative to the median norm, which directly
            catches gradient-scaling / boosting attacks that inflate the L2
            norm while keeping direction plausible.
        """
        if not deltas or deltas[0].size == 0:
            return [1.0] * len(deltas)

        matrix = np.stack(deltas).astype(np.float32)
        n, d = matrix.shape

        # Robust reference: coordinate-wise median update.
        median_delta = np.median(matrix, axis=0)

        # Lightweight random projection (rSVD-style) for the subspace signal.
        k = min(self.config.subspace_dim, n, d)
        if k >= 1:
            rng = np.random.default_rng(self.round_number)
            omega = rng.standard_normal((d, k)).astype(np.float32)
            proj = matrix @ omega
            proj_med = median_delta @ omega
            sub_dev = np.linalg.norm(proj - proj_med, axis=1)
        else:
            sub_dev = np.linalg.norm(matrix - median_delta, axis=1)

        # MAD-normalized deviation (robust z-score).
        mad = float(np.median(np.abs(sub_dev - np.median(sub_dev)))) + 1e-8
        robust_z = np.abs(sub_dev - np.median(sub_dev)) / (1.4826 * mad)

        # Norm-ratio signal.
        norms = np.linalg.norm(matrix, axis=1)
        median_norm = float(np.median(norms)) + 1e-8
        norm_ratio = norms / median_norm
        norm_thr = norm_ratio_threshold(
            n,
            self.config.norm_ratio_threshold,
            self.config.norm_ratio_threshold_two_clients,
        )

        scores = []
        for i in range(n):
            sub_score = self._decay(robust_z[i], self.config.subspace_anomaly_threshold)
            norm_score = self._decay(norm_ratio[i], norm_thr)
            score = min(sub_score, norm_score)
            scores.append(float(np.clip(score, self.config.min_trust_weight, 1.0)))

        return scores

    def _temporal_consistency_scores(
        self,
        client_ids: List[str],
        deltas: List[np.ndarray],
    ) -> List[float]:
        """Score based on inter-round update magnitude stability.

        Legitimate clients exhibit smooth update trajectories whose magnitude
        decays as training converges. Poisoning clients injecting boosted or
        abrupt updates show sudden magnitude spikes relative to their own
        history. We score the ratio of the current update norm to the client's
        historical median norm, so a consistently *inflated* attacker is still
        penalized (its norm stays far above the benign convergence trend) while
        benign direction changes on tiny near-converged updates are not.
        """
        scores = []
        for cid, delta in zip(client_ids, deltas):
            history = self.client_update_history.get(cid, [])
            if len(history) < 2:
                scores.append(1.0)
                continue

            recent = history[-min(5, len(history)) :]
            hist_norms = np.array([l2_norm(h) for h in recent])
            hist_median_norm = float(np.median(hist_norms)) + 1e-8
            delta_norm = l2_norm(delta)

            # Penalize abrupt growth above the client's own historical norm.
            norm_ratio = delta_norm / hist_median_norm
            score = self._decay(norm_ratio, 2.0)
            scores.append(
                float(np.clip(score, self.config.min_trust_weight, 1.0))
            )

        return scores

    @staticmethod
    def _decay(value: float, threshold: float) -> float:
        """1.0 below threshold, exponentially decaying above it."""
        if value <= threshold:
            return 1.0
        return float(np.exp(-(value - threshold)))

    def _soft_filter(self, raw_score: float) -> float:
        return float(np.clip(raw_score, self.config.min_trust_weight, 1.0))


class ProbeEvaluator:
    """
    Counterfactual attack-probe validation.

    Measures effect of a client update on compact probe sets representing
    benign flight and attack scenarios.

    Optimizations vs. the original loop:
      * Probe tensors are pinned on ``device`` once at construction.
      * All active scenarios are evaluated in a single batched forward.
      * Optional per-scenario window cap and attack-scoped scenario subsets.
      * ``set_baseline`` skips re-evaluation when weights are unchanged.
    """

    ATTACK_SCENARIOS = {
        "gnss_spoofing", "adsb_injection", "mavlink_waypoint",
        "jamming", "backdoor_trigger",
    }

    # Attack-scoped subsets used when ``probe_attack_scoped`` is enabled.
    # Keeps the scenarios that drive that attack's Layer~1 signal and drops
    # the rest (especially backdoor_trigger under non-backdoor attacks).
    ATTACK_SCOPED_SCENARIOS = {
        "backdoor": ("benign", "backdoor_trigger"),
        "label_flip": (
            "benign",
            "gnss_spoofing",
            "adsb_injection",
            "mavlink_waypoint",
            "jamming",
        ),
        "byzantine": (
            "benign",
            "gnss_spoofing",
            "mavlink_waypoint",
            "jamming",
        ),
        "gradient_scaling": (
            "benign",
            "gnss_spoofing",
            "mavlink_waypoint",
            "jamming",
        ),
        "model_replacement": (
            "benign",
            "gnss_spoofing",
            "mavlink_waypoint",
            "jamming",
        ),
    }

    @classmethod
    def scenarios_for_attack(cls, attack_type: Optional[str]) -> Optional[List[str]]:
        if not attack_type or attack_type in ("none", ""):
            return None
        scoped = cls.ATTACK_SCOPED_SCENARIOS.get(str(attack_type))
        return list(scoped) if scoped else None

    def __init__(
        self,
        model: nn.Module,
        probe_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        learning_mode: str = "classification",
        tolerance: float = 0.05,
        gain: float = 6.0,
        min_score: float = 0.05,
        backdoor_probe_tolerance: float = 0.01,
        backdoor_probe_gain: float = 10.0,
        max_windows_per_scenario: Optional[int] = None,
        active_scenarios: Optional[List[str]] = None,
    ):
        """
        Args:
            model: Global model (temporarily modified for counterfactual eval)
            probe_data: {scenario_name: (inputs, labels)}
            device: Torch device
            learning_mode: 'classification' or 'regression'
            tolerance: per-scenario confidence drop treated as benign drift
            gain: amplification applied to mean harm beyond the tolerance
            min_score: floor for the counterfactual trust score
            max_windows_per_scenario: optional cap on windows kept per scenario
            active_scenarios: optional subset of scenario names to evaluate
        """
        self.model = model
        self.device = device
        self.learning_mode = learning_mode
        self.tolerance = tolerance
        self.gain = gain
        self.min_score = min_score
        self.backdoor_probe_tolerance = backdoor_probe_tolerance
        self.backdoor_probe_gain = backdoor_probe_gain
        self.max_windows_per_scenario = max_windows_per_scenario
        self._baseline_metrics: Optional[Dict[str, float]] = None
        self._baseline_global: Optional[Dict[str, torch.Tensor]] = None

        self.probe_data = self._prepare_probe_data(probe_data, active_scenarios)
        self._rebuild_batch()

    def _prepare_probe_data(
        self,
        probe_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        active_scenarios: Optional[List[str]],
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        if active_scenarios:
            keep = [s for s in active_scenarios if s in probe_data]
            if not keep:
                keep = list(probe_data.keys())
        else:
            keep = list(probe_data.keys())

        prepared: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        max_windows = self.max_windows_per_scenario
        for scenario in keep:
            inputs, labels = probe_data[scenario]
            inputs = inputs.detach().to(self.device)
            labels = labels.detach().to(self.device)
            if max_windows is not None and int(max_windows) > 0:
                n = min(int(max_windows), int(inputs.shape[0]))
                inputs = inputs[:n]
                labels = labels[:n]
            prepared[scenario] = (inputs, labels)
        return prepared

    def _rebuild_batch(self) -> None:
        """Concatenate active scenarios into one forward batch."""
        self._scenario_order = list(self.probe_data.keys())
        if not self._scenario_order:
            self._batched_inputs = None
            self._batched_labels = None
            self._scenario_slices: Dict[str, Tuple[int, int]] = {}
            return

        pieces_x = []
        pieces_y = []
        slices: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        for scenario in self._scenario_order:
            inputs, labels = self.probe_data[scenario]
            n = int(inputs.shape[0])
            pieces_x.append(inputs)
            pieces_y.append(labels.reshape(-1))
            slices[scenario] = (cursor, cursor + n)
            cursor += n
        self._batched_inputs = torch.cat(pieces_x, dim=0)
        self._batched_labels = torch.cat(pieces_y, dim=0)
        self._scenario_slices = slices

    @staticmethod
    def _weights_equal(
        left: Dict[str, torch.Tensor],
        right: Dict[str, torch.Tensor],
    ) -> bool:
        if left.keys() != right.keys():
            return False
        for key in left:
            if left[key].shape != right[key].shape:
                return False
            if not torch.equal(left[key], right[key]):
                return False
        return True

    def set_baseline(self, global_weights: Dict[str, torch.Tensor]):
        """Record baseline detection metrics for the current global model."""
        if (
            self._baseline_global is not None
            and self._baseline_metrics is not None
            and self._weights_equal(self._baseline_global, global_weights)
        ):
            return
        self._baseline_global = {
            key: tensor.clone() for key, tensor in global_weights.items()
        }
        self.model.set_weights(self._baseline_global)
        self._baseline_metrics = self._evaluate_all_probes()

    def evaluate_candidate(
        self,
        weights: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Probe metrics for a temporary candidate model (restores weights)."""
        previous = {
            key: tensor.detach().clone()
            for key, tensor in self.model.state_dict().items()
        }
        try:
            self.model.set_weights(weights)
            metrics = self._evaluate_all_probes()
            p_attack, benign_rate = self._evaluate_backdoor_trigger_stats()
            return {
                "metrics": metrics,
                "benign_quality": float(metrics.get("benign", 0.0)),
                "p_attack_trigger": float(p_attack),
                "trigger_benign_rate": float(benign_rate),
            }
        finally:
            self.model.set_weights(previous)

    def candidate_probe_terms(
        self,
        candidate_weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Runtime probe terms for PTA: task loss, harm, trigger residual."""
        self.set_baseline(global_weights)
        baseline = self._baseline_metrics or {}
        base_pa, _base_br = self._evaluate_backdoor_trigger_stats()
        evaluated = self.evaluate_candidate(candidate_weights)
        metrics = evaluated.get("metrics") or {}
        harm = float(self._compute_harm_score(baseline, metrics))
        benign_quality = float(evaluated.get("benign_quality", 0.0))
        p_attack = float(evaluated.get("p_attack_trigger", 0.0))
        return {
            "L_task": float(max(0.0, 1.0 - benign_quality)),
            "H": harm,
            "R_trigger": float(max(0.0, base_pa - p_attack)),
            "benign_quality": benign_quality,
            "p_attack_trigger": p_attack,
            "baseline_p_attack_trigger": float(base_pa),
        }

    def counterfactual_scores(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Score each client by counterfactual effect on probe sets.

        The baseline is recomputed from the *current* global model every round
        so that, as the global model converges, benign updates are compared
        against an up-to-date reference rather than a stale one.
        """
        self.set_baseline(global_weights)

        scores = {}
        for cid, cw in zip(client_ids, client_weights):
            self.model.set_weights(cw)
            cf_metrics = self._evaluate_all_probes()
            harm = self._compute_harm_score(self._baseline_metrics, cf_metrics)
            score = np.exp(-harm)
            scores[cid] = float(np.clip(score, self.min_score, 1.0))

        self.model.set_weights(self._baseline_global)
        return scores

    def _evaluate_all_probes(self) -> Dict[str, float]:
        """Return a smooth detection-quality score per scenario in [0, 1].

        Higher means the model handles that scenario better. Using a continuous
        confidence/quality measure (rather than thresholded 0/1 accuracy) avoids
        large harm jumps when a benign update nudges a prediction across the
        decision boundary.
        """
        if self._batched_inputs is None or not self._scenario_order:
            return {}

        self.model.eval()
        metrics = {}
        with torch.no_grad():
            outputs = self.model(self._batched_inputs)
            if self.learning_mode == "classification":
                probs = torch.sigmoid(outputs).squeeze().clamp(1e-6, 1 - 1e-6)
                lbl = self._batched_labels.float().reshape(probs.shape)
                correct = torch.where(lbl > 0.5, probs, 1.0 - probs)
                for scenario in self._scenario_order:
                    start, end = self._scenario_slices[scenario]
                    metrics[scenario] = float(correct[start:end].mean().item())
            else:
                # Per-scenario MSE so regression quality stays scenario-local.
                for scenario in self._scenario_order:
                    start, end = self._scenario_slices[scenario]
                    pred = outputs[start:end]
                    target = self._batched_labels[start:end].reshape_as(pred)
                    mse = nn.functional.mse_loss(pred, target).item()
                    metrics[scenario] = float(np.exp(-mse))
        return metrics

    def _evaluate_backdoor_trigger_stats(self) -> Tuple[float, float]:
        """Return mean P(attack) and benign-misclass rate on trigger probes."""
        if "backdoor_trigger" not in self.probe_data:
            return 0.0, 0.0
        inputs, labels = self.probe_data["backdoor_trigger"]
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            if self.learning_mode == "classification":
                probs = torch.sigmoid(outputs).squeeze().clamp(1e-6, 1 - 1e-6)
                mean_attack_prob = float(probs.mean().item())
                benign_rate = float((probs < 0.5).float().mean().item())
                return mean_attack_prob, benign_rate
            mse = nn.functional.mse_loss(outputs, labels).item()
            quality = float(np.exp(-mse))
            return quality, 0.0

    def _compute_harm_score(
        self,
        baseline: Dict[str, float],
        counterfactual: Dict[str, float],
    ) -> float:
        """Tolerance-banded mean degradation in attack-detection confidence.

        For every attack scenario we measure how much the client's update drops
        detection confidence below the current global baseline. Drops within
        ``tolerance`` are ignored (benign drift); the remainder is averaged
        across scenarios (so harm does not scale with scenario count) and scaled
        by ``gain``. The backdoor-trigger scenario is weighted double.
        """
        degs = []
        weights = []
        for scenario in self.ATTACK_SCENARIOS:
            if scenario not in baseline:
                continue
            base_val = baseline[scenario]
            cf_val = counterfactual.get(scenario, base_val)
            # Both metrics are "higher = better", so a positive base - cf is a
            # degradation regardless of classification/regression mode.
            degradation = max(0.0, (base_val - cf_val) - self.tolerance)
            w = 2.0 if scenario == "backdoor_trigger" else 1.0
            degs.append(w * degradation)
            weights.append(w)

        if not weights:
            return 0.0
        mean_deg = sum(degs) / sum(weights)
        return self.gain * mean_deg

    def clean_probe_quality_details(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, float]]:
        """Per-client clean-task probe quality for label-flip selection.

        Label-flip poisons local labels without large update norms, so Krum
        geometry alone is the strongest classical signal. Clean probe quality
        (true-class confidence, heavily weighted to ``benign``, excluding the
        backdoor trigger probe) is a secondary discriminator used only when it
        clearly beats the geometric winner.
        """
        self.set_baseline(global_weights)
        baseline = self._baseline_metrics or {}
        details: Dict[str, Dict[str, float]] = {}
        for cid, cw in zip(client_ids, client_weights):
            self.model.set_weights(cw)
            cf = self._evaluate_all_probes()
            benign_q = float(cf.get("benign", 0.0))
            # Exclude trigger probe: irrelevant for label-flip discrimination.
            clean_vals = [
                float(v)
                for s, v in cf.items()
                if s != "backdoor_trigger"
            ]
            mean_q = (
                float(sum(clean_vals) / len(clean_vals)) if clean_vals else 0.0
            )
            drops = []
            for scenario, base_val in baseline.items():
                if scenario == "backdoor_trigger":
                    continue
                drops.append(
                    max(0.0, float(base_val) - float(cf.get(scenario, base_val)))
                )
            mean_drop = float(sum(drops) / len(drops)) if drops else 0.0
            benign_drop = max(
                0.0, float(baseline.get("benign", benign_q)) - benign_q
            )
            harm = float(self.gain * (0.75 * benign_drop + 0.25 * mean_drop))
            details[cid] = {
                "benign_quality": benign_q,
                "mean_quality": mean_q,
                # Selection score: benign-dominant (higher is better).
                "clean_quality": float(0.85 * benign_q + 0.15 * mean_q),
                "benign_drop": benign_drop,
                "mean_drop": mean_drop,
                "harm": harm,
            }
        self.model.set_weights(self._baseline_global)
        return details

    def backdoor_trigger_harm_details(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, float]]:
        """Per-client backdoor_trigger harm with probe statistics."""
        self.set_baseline(global_weights)
        base_pa, base_br = self._evaluate_backdoor_trigger_stats()
        details: Dict[str, Dict[str, float]] = {}
        for cid, cw in zip(client_ids, client_weights):
            self.model.set_weights(cw)
            cf_pa, cf_br = self._evaluate_backdoor_trigger_stats()
            harm = self._compute_backdoor_trigger_harm(base_pa, cf_pa, base_br, cf_br)
            details[cid] = {
                "baseline_p_attack": base_pa,
                "counterfactual_p_attack": cf_pa,
                "baseline_benign_rate": base_br,
                "counterfactual_benign_rate": cf_br,
                "harm": harm,
                "probe_cap": float(np.clip(np.exp(-harm), self.min_score, 1.0)),
            }
        self.model.set_weights(self._baseline_global)
        return details

    def backdoor_trigger_harm_scores(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Per-client harm from counterfactual effect on backdoor_trigger probes only."""
        details = self.backdoor_trigger_harm_details(
            client_ids, client_weights, global_weights
        )
        return {cid: float(info["harm"]) for cid, info in details.items()}

    def backdoor_probe_cap_scores(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Soft trust ceiling from backdoor_trigger harm: exp(-harm) in [min_score, 1]."""
        details = self.backdoor_trigger_harm_details(
            client_ids, client_weights, global_weights
        )
        return {cid: float(info["probe_cap"]) for cid, info in details.items()}

    def _compute_backdoor_trigger_harm(
        self,
        baseline_p_attack: float,
        counterfactual_p_attack: float,
        baseline_benign_rate: float,
        counterfactual_benign_rate: float,
    ) -> float:
        """Harm when an update increases backdoor trigger acceptance."""
        tol = self.backdoor_probe_tolerance
        gain = self.backdoor_probe_gain
        # Embedding: trigger windows classified more benign (lower P(attack)).
        pa_drop = max(0.0, (baseline_p_attack - counterfactual_p_attack) - tol)
        # Thresholded misclassification rate on attack-labeled triggers rises.
        br_rise = max(
            0.0, (counterfactual_benign_rate - baseline_benign_rate) - tol
        )
        return gain * (pa_drop + br_rise)

    def _compute_backdoor_trigger_harm_legacy(
        self,
        baseline: Dict[str, float],
        counterfactual: Dict[str, float],
    ) -> float:
        """Previous single-signal harm (kept for diagnostics)."""
        scenario = "backdoor_trigger"
        if scenario not in baseline:
            return 0.0
        base_val = baseline[scenario]
        cf_val = counterfactual.get(scenario, base_val)
        degradation = max(0.0, (base_val - cf_val) - self.tolerance)
        return self.gain * degradation
