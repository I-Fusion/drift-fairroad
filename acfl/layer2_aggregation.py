"""
Layer 2: Mission-Context-Conditioned Byzantine-Tolerant Trust-Weighted Aggregation.

Stage 1: Coordinate-wise robust pre-filtering (median / trimmed mean)
Stage 2: Mission-context-conditioned client trust weighting
"""
from typing import Dict, List, Optional

import numpy as np
import torch

from acfl.config import ACFLConfig
from acfl.weight_utils import (
    coordinate_wise_trimmed_mean,
    compute_client_update_norm_stats,
    norm_ratio_threshold,
    robustness_score_from_update_stats,
    weighted_average_state_dicts,
)


class TrustWeightedAggregator:
    """Byzantine-tolerant aggregation with mission-context trust weighting."""

    def __init__(self, config: ACFLConfig):
        self.config = config
        self.benign_deviation_history: List[float] = []
        self.client_participation: Dict[str, int] = {}
        self.last_aggregation_diagnostics: Dict = {}

    def aggregate(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_trust: Optional[Dict[str, float]] = None,
        mission_context: Optional[Dict[str, Dict[str, float]]] = None,
        client_samples: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform two-stage robust aggregation.

        Args:
            client_ids: Client identifiers
            client_weights: Client model state dicts
            global_weights: Current global model weights
            layer1_trust: Trust scores from Layer 1 validation
            mission_context: Per-client cyber-physical consistency features
            client_samples: Optional sample counts for sample-weighted aggregation
        """
        if not client_weights:
            raise ValueError("No client weights to aggregate")

        layer1_trust = layer1_trust or {cid: 1.0 for cid in client_ids}
        mission_context = mission_context or {}

        # Stage 1: Coordinate-wise robust pre-filtering
        filtered_weights, robustness_scores = self._robust_prefilter(
            client_ids, client_weights, global_weights
        )

        # Stage 2: Mission-context trust weighting
        final_weights = []
        client_diagnostics = {}
        for i, cid in enumerate(client_ids):
            stat_score = robustness_scores[i]
            mission_score = self._mission_context_score(cid, mission_context)
            l1_score = layer1_trust.get(cid, 1.0)

            combined = (
                self.config.statistical_consistency_weight * stat_score
                + self.config.mission_context_weight * mission_score
            ) * l1_score
            combined = max(float(combined), 0.0)
            powered = combined ** max(float(self.config.trust_weight_power), 0.0)

            if client_samples is not None:
                sample_factor = client_samples[i] / max(sum(client_samples), 1)
                powered *= (0.5 + 0.5 * sample_factor)

            effective = max(powered, self.config.min_trust_weight)
            final_weights.append(effective)
            self.client_participation[cid] = self.client_participation.get(cid, 0) + 1

            client_diagnostics[cid] = {
                "layer1_trust": float(l1_score),
                "robustness_score": float(stat_score),
                "mission_context_score": float(mission_score),
                "combined_trust": float(combined),
                "powered_trust": float(powered),
                "aggregation_weight_raw": float(effective),
                "min_weight_clipped": bool(effective == self.config.min_trust_weight),
                "num_samples": int(client_samples[i]) if client_samples is not None else None,
            }

        penalties, penalty_reasons = self._pairwise_inversion_penalties(
            client_ids, client_diagnostics
        )
        for i, cid in enumerate(client_ids):
            if penalties[i] != 1.0:
                final_weights[i] *= penalties[i]
                client_diagnostics[cid]["pairwise_inversion_penalty"] = float(penalties[i])
                client_diagnostics[cid]["pairwise_inversion_reason"] = penalty_reasons[i]

        total_weight = float(sum(final_weights))
        if total_weight > 0:
            for i, cid in enumerate(client_ids):
                client_diagnostics[cid]["aggregation_weight_normalized"] = float(
                    final_weights[i] / total_weight
                )

        self.last_aggregation_diagnostics = {
            "trust_weight_power": float(self.config.trust_weight_power),
            "min_trust_weight": float(self.config.min_trust_weight),
            "pairwise_inversion_guard_enabled": bool(
                self.config.pairwise_inversion_guard_enabled
            ),
            "clients": client_diagnostics,
        }

        return weighted_average_state_dicts(filtered_weights, final_weights)

    def aggregate_readonly(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_trust: Optional[Dict[str, float]] = None,
        mission_context: Optional[Dict[str, Dict[str, float]]] = None,
        client_samples: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate without mutating participation or deviation history."""
        history = list(self.benign_deviation_history)
        participation = dict(self.client_participation)
        diagnostics = dict(self.last_aggregation_diagnostics)
        try:
            return self.aggregate(
                client_ids,
                client_weights,
                global_weights,
                layer1_trust,
                mission_context,
                client_samples,
            )
        finally:
            self.benign_deviation_history = history
            self.client_participation = participation
            self.last_aggregation_diagnostics = diagnostics

    def record_round(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        diagnostics: Optional[Dict] = None,
    ) -> None:
        """Apply one round of Layer-2 history after a PTA commit."""
        if client_weights:
            self._robust_prefilter(client_ids, client_weights, global_weights)
        for cid in client_ids:
            self.client_participation[cid] = self.client_participation.get(cid, 0) + 1
        if diagnostics is not None:
            self.last_aggregation_diagnostics = diagnostics

    def _pairwise_inversion_penalties(
        self,
        client_ids: List[str],
        client_diagnostics: Dict[str, Dict],
    ) -> tuple:
        """Downweight a 2-client participant that mimics consensus while peer is filtered."""
        n = len(client_ids)
        penalties = [1.0] * n
        reasons = [""] * n
        if n != 2 or not self.config.pairwise_inversion_guard_enabled:
            return penalties, reasons

        margin = float(self.config.pairwise_trust_margin)
        asymmetry = float(self.config.pairwise_robustness_asymmetry)
        floor = float(self.config.pairwise_high_robustness_floor)
        multiplier = float(self.config.pairwise_suspicious_weight_multiplier)

        for suspect_idx, reference_idx in ((0, 1), (1, 0)):
            suspect_id = client_ids[suspect_idx]
            reference_id = client_ids[reference_idx]
            suspect = client_diagnostics[suspect_id]
            reference = client_diagnostics[reference_id]

            if (
                reference["layer1_trust"] > suspect["layer1_trust"] + margin
                and suspect["robustness_score"]
                > reference["robustness_score"] + asymmetry
                and suspect["robustness_score"] >= floor
            ):
                penalties[suspect_idx] = min(penalties[suspect_idx], multiplier)
                reasons[suspect_idx] = (
                    "trust_robustness_inversion:"
                    f"{suspect_id}:rob={suspect['robustness_score']:.3f},"
                    f"l1={suspect['layer1_trust']:.3f} vs "
                    f"{reference_id}:rob={reference['robustness_score']:.3f},"
                    f"l1={reference['layer1_trust']:.3f}"
                )

        return penalties, reasons

    def _robust_prefilter(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
    ) -> tuple:
        """Soft-filter updates deviating from coordinate-wise robust statistics.

        Operates in *delta space* (client - global) so that boosting /
        gradient-scaling attacks, which inflate the update relative to the
        robust consensus, are caught independently of Layer 1. Scoring is
        self-contained per round (scale-relative to the median update norm),
        avoiding the history-pollution failure mode where an attacker present
        every round inflates its own detection threshold.
        """
        n = len(client_weights)
        trimmed_weights = coordinate_wise_trimmed_mean(
            client_weights, self.config.trim_fraction
        )

        stats = compute_client_update_norm_stats(client_weights, global_weights)
        if not stats:
            return list(client_weights), [1.0] * n

        coord_thr = self.config.coordinate_deviation_threshold
        norm_thr = norm_ratio_threshold(
            n,
            self.config.norm_ratio_threshold,
            self.config.norm_ratio_threshold_two_clients,
        )
        robustness_scores = []
        filtered = []

        for i, cw in enumerate(client_weights):
            score = robustness_score_from_update_stats(
                stats[i],
                coord_thr,
                norm_thr,
                self.config.min_trust_weight,
            )
            self.benign_deviation_history.append(float(stats[i]["rel_dev"]))

            if score >= 0.999:
                filtered.append(cw)
            else:
                filtered.append(self._blend_toward_robust(cw, trimmed_weights, score))

            robustness_scores.append(score)

        return filtered, robustness_scores

    @staticmethod
    def _decay(value: float, threshold: float) -> float:
        """1.0 below threshold, exponentially decaying above it."""
        if value <= threshold:
            return 1.0
        return float(np.exp(-(value - threshold)))

    def _robustness_scores_readonly(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
    ) -> List[float]:
        """Compute robustness scores without mutating any internal state."""
        n = len(client_weights)
        stats = compute_client_update_norm_stats(client_weights, global_weights)
        if not stats:
            return [1.0] * n

        coord_thr = self.config.coordinate_deviation_threshold
        norm_thr = norm_ratio_threshold(
            n,
            self.config.norm_ratio_threshold,
            self.config.norm_ratio_threshold_two_clients,
        )
        return [
            robustness_score_from_update_stats(
                stats[i], coord_thr, norm_thr, self.config.min_trust_weight
            )
            for i in range(n)
        ]

    def _blend_toward_robust(
        self,
        weights: Dict[str, torch.Tensor],
        robust: Dict[str, torch.Tensor],
        trust: float,
    ) -> Dict[str, torch.Tensor]:
        blended = {}
        for key in weights:
            blended[key] = trust * weights[key].float() + (1 - trust) * robust[key].float()
        return blended

    def _mission_context_score(
        self,
        client_id: str,
        mission_context: Dict[str, Dict[str, float]],
    ) -> float:
        """Compute cyber-physical consistency trust score."""
        ctx = mission_context.get(client_id, {})
        if not ctx:
            participation = self.client_participation.get(client_id, 0)
            return min(1.0, 0.5 + 0.1 * participation)

        score = 0.0
        total_weight = 0.0
        for feature, weight in self.config.mission_context_features.items():
            if feature in ctx:
                score += weight * np.clip(ctx[feature], 0.0, 1.0)
                total_weight += weight

        if total_weight > 0:
            return score / total_weight
        return 1.0

    def get_trust_breakdown(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_trust: Dict[str, float],
        mission_context: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """Return detailed trust score breakdown for evaluation."""
        robustness_scores = self._robustness_scores_readonly(
            client_weights, global_weights
        )
        breakdown = {}
        for i, cid in enumerate(client_ids):
            breakdown[cid] = {
                "layer1_trust": layer1_trust.get(cid, 1.0),
                "robustness_score": robustness_scores[i],
                "mission_context_score": self._mission_context_score(
                    cid, mission_context
                ),
                "participation_count": self.client_participation.get(cid, 0),
            }
        return breakdown
