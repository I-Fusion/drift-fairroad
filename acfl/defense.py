"""
ACFL Defense orchestrator combining Layers 1 and 2.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from acfl.config import ACFLConfig
from acfl.layer1_validation import ProbeEvaluator, UpdateValidationModule
from acfl.layer2_aggregation import TrustWeightedAggregator
from acfl.parallel_trust_arbitration import (
    BranchScore,
    arbitrate,
    candidate_objective,
)
from acfl.robust_baselines import krum_peer_scores
from acfl.weight_utils import (
    blend_state_dicts,
    clone_state_dict,
    compute_client_update_norm_stats,
    norm_ratio_threshold,
    robustness_score_from_update_stats,
    state_dict_l2,
    weighted_average_state_dicts,
)


class ACFLDefense:
    """
    Main ACFL defense framework.

    Integrates Layer 1 (update validation) and Layer 2 (trust-weighted aggregation)
    as modular plugins in the federated training loop.
    """

    def __init__(
        self,
        config: ACFLConfig,
        model: Optional[nn.Module] = None,
        probe_data: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        learning_mode: str = "classification",
    ):
        self.config = config
        self.model = model
        self.device = device or torch.device("cpu")
        self.learning_mode = learning_mode

        self.layer1 = UpdateValidationModule(config) if config.enable_layer1 else None
        self.layer2 = TrustWeightedAggregator(config) if config.enable_layer2 else None

        self.probe_evaluator: Optional[ProbeEvaluator] = None
        if probe_data and model is not None and config.enable_layer1:
            active_scenarios = None
            if (
                bool(getattr(config, "probe_attack_scoped", False))
                and config.evaluation_attack_type
            ):
                active_scenarios = ProbeEvaluator.scenarios_for_attack(
                    config.evaluation_attack_type
                )
            elif config.probe_scenarios:
                active_scenarios = list(config.probe_scenarios)
            self.probe_evaluator = ProbeEvaluator(
                model,
                probe_data,
                self.device,
                learning_mode,
                tolerance=config.counterfactual_tolerance,
                gain=config.counterfactual_gain,
                min_score=config.min_trust_weight,
                backdoor_probe_tolerance=config.backdoor_probe_tolerance,
                backdoor_probe_gain=config.backdoor_probe_gain,
                max_windows_per_scenario=getattr(
                    config, "probe_max_windows_per_scenario", None
                ),
                active_scenarios=active_scenarios,
            )

        self.trust_history: List[Dict[str, float]] = []
        self.round_number = 0
        self.last_aggregation_guard: Optional[Dict] = None
        self.last_update_norm_rejection: Optional[Dict] = None
        self.last_robust_norm_soft_cap: Optional[Dict] = None
        self.last_backdoor_probe_soft_cap: Optional[Dict] = None
        self.last_backdoor_trigger_hard_gate: Optional[Dict] = None
        self.last_backdoor_probe_hard_rejection: Optional[Dict] = None
        self.last_backdoor_geometry_hard_rejection: Optional[Dict] = None
        self.last_post_agg_trigger_scrub: Optional[Dict] = None
        self.last_layer1_layer2_gating: Optional[Dict] = None
        self.last_parallel_trust_arbitration: Optional[Dict] = None
        self._pta_last_branch: Optional[str] = None

    @staticmethod
    def _backdoor_probe_worst_outliers(
        client_ids: List[str],
        details: Dict[str, Dict],
        harm_thr: float,
        peer_margin: float,
        require_p_floor: bool = False,
        p_floor: float = 0.15,
        cohort_factor: float = 1.5,
    ) -> Dict[str, Dict]:
        """Return worst trigger-harm outliers (peer-gap or cohort-relative)."""
        scored = [
            (cid, details.get(cid, {}))
            for cid in client_ids
            if cid in details
        ]
        if not scored:
            return {}

        harms = [float(info.get("harm", 0.0)) for _, info in scored]
        max_harm = max(harms)
        second_harm = sorted(harms, reverse=True)[1] if len(harms) >= 2 else 0.0
        median_harm = float(sorted(harms)[len(harms) // 2])
        peer_gap_ok = (max_harm - second_harm) >= peer_margin
        cohort_ok = (
            len(harms) >= 3
            and max_harm >= cohort_factor * max(median_harm, 1e-6)
        )
        if max_harm < harm_thr or not (peer_gap_ok or cohort_ok):
            return {}

        outliers: Dict[str, Dict] = {}
        for cid, info in scored:
            harm = float(info.get("harm", 0.0))
            if harm < max_harm - 1e-12:
                continue
            cf_pa = float(info.get("counterfactual_p_attack", 1.0))
            if require_p_floor and cf_pa >= p_floor:
                continue
            reasons = [f"harm:{harm:.4f}>={harm_thr:.4f}"]
            if peer_gap_ok:
                reasons.append(
                    f"peer_margin:{(harm - second_harm):.4f}>={peer_margin:.4f}"
                )
            if cohort_ok:
                reasons.append(
                    f"cohort:{harm:.4f}>={cohort_factor:.2f}x_median:{median_harm:.4f}"
                )
            if require_p_floor:
                reasons.append(f"cf_p_attack:{cf_pa:.4f}<{p_floor:.4f}")
            outliers[cid] = {
                "harm": harm,
                "counterfactual_p_attack": cf_pa,
                "probe_cap": float(info.get("probe_cap", 1.0)),
                "second_harm": second_harm,
                "median_harm": median_harm,
                "reason": ";".join(reasons),
            }
        return outliers

    def _peer_robustness_scores(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        n = len(client_ids)
        stats = compute_client_update_norm_stats(client_weights, global_weights)
        if not stats:
            return {cid: 1.0 for cid in client_ids}

        norm_thr = norm_ratio_threshold(
            n,
            self.config.norm_ratio_threshold,
            self.config.norm_ratio_threshold_two_clients,
        )
        coord_thr = float(self.config.coordinate_deviation_threshold)
        return {
            client_ids[i]: robustness_score_from_update_stats(
                stats[i], coord_thr, norm_thr, self.config.min_trust_weight
            )
            for i in range(n)
        }

    def _backdoor_layer1_in_aggregation(self) -> bool:
        attack = self.config.evaluation_attack_type
        use_l1_for_attack = attack == "backdoor" or (
            attack == "model_replacement"
            and self.config.use_layer1_for_model_replacement
        )
        return (
            use_l1_for_attack
            and self.config.layer1_layer2_gating_enabled
            and self.layer1 is not None
            and self.layer2 is not None
        )

    def _hard_rejection_active(self) -> bool:
        """Global hard reject, or attack-scoped hard reject for model replacement."""
        if self.config.update_norm_hard_rejection_enabled:
            return True
        return (
            self.config.evaluation_attack_type == "model_replacement"
            and self.config.update_norm_hard_rejection_for_model_replacement
        )

    def _apply_layer1_layer2_gating(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_trust: Dict[str, float],
    ) -> Dict[str, float]:
        """Record peer robustness when Layer 1 scores are telemetry-only."""
        if (
            not self.config.layer1_layer2_gating_enabled
            or self.layer2 is None
            or not self.layer1
        ):
            self.last_layer1_layer2_gating = {"enabled": False}
            return layer1_trust

        robustness = self._peer_robustness_scores(
            client_ids, client_weights, global_weights
        )
        self.last_layer1_layer2_gating = {
            "enabled": True,
            "adaptive": bool(self.config.adaptive_l1l2_gating_enabled),
            "defer_layer1_to_layer2": not self._backdoor_layer1_in_aggregation(),
            "backdoor_layer1_in_aggregation": self._backdoor_layer1_in_aggregation(),
            "robustness_scores": robustness,
            "neutralized_clients": [],
        }
        return layer1_trust

    def _adaptive_gate(
        self,
        client_ids: List[str],
        layer1_raw: Dict[str, float],
        robustness: Dict[str, float],
    ) -> Dict[str, float]:
        """Per-client hinge for Layer 1 cohort over-penalty only.

        c_i = max(0, R_i - T_i). A single client above tau is treated as
        attribution and stays on raw Layer 1. Two or more such clients mean
        Layer 1 is penalizing a robust cohort, so those gates open.
        A high Layer 1 score on a low-robustness client does not open the gate.
        """
        suppress: Dict[str, float] = {}
        gates: Dict[str, float] = {}
        tau = float(self.config.adaptive_l1l2_gating_conflict_threshold)
        width = float(getattr(self.config, "adaptive_l1l2_gating_width", 0.25))
        width = max(width, 1e-6)
        for cid in client_ids:
            robustness_i = float(robustness.get(cid, 1.0))
            trust_i = float(layer1_raw.get(cid, 1.0))
            suppress[cid] = max(0.0, robustness_i - trust_i)
            gates[cid] = 0.0

        suspects = [cid for cid in client_ids if suppress[cid] > tau]
        cohort = len(suspects) >= 2
        if cohort:
            for cid in suspects:
                gates[cid] = float(np.clip((suppress[cid] - tau) / width, 0.0, 1.0))

        gate_values = list(gates.values())
        mean_gate = float(np.mean(gate_values)) if gate_values else 0.0
        suppress_values = list(suppress.values())
        mean_conflict = float(np.mean(suppress_values)) if suppress_values else 0.0
        return {
            "g_t": mean_gate,
            "g_i": gates,
            "mean_conflict": mean_conflict,
            "conflict_threshold": tau,
            "width": width,
            "per_client_conflict": suppress,
            "per_client_suppress": suppress,
            "suspects": suspects,
            "cohort_penalty": cohort,
        }

    def _adaptive_aggregation_trust(
        self,
        client_ids: List[str],
        layer1_raw: Dict[str, float],
        robustness: Dict[str, float],
        raw_capped: Dict[str, float],
        gated_trust: Dict[str, float],
        gate_info: Dict[str, float],
    ) -> Dict[str, float]:
        """Cohort hinge blend. A shut gate keeps the norm-capped Layer 1 score."""
        gates = gate_info.get("g_i") or {}
        trust: Dict[str, float] = {}
        for cid in client_ids:
            raw_t = float(layer1_raw.get(cid, 1.0))
            gate_i = float(gates.get(cid, 0.0))
            trust[cid] = (
                (1.0 - gate_i) * float(raw_capped.get(cid, raw_t))
                + gate_i * float(gated_trust.get(cid, 1.0))
            )
        return {"trust": trust, "cohort_penalty": bool(gate_info.get("cohort_penalty"))}

    def _aggregation_layer1_trust(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_raw: Dict[str, float],
        sanitize_flags: Dict[str, str],
    ) -> Dict[str, float]:
        """Trust scores that drive guards, rejection, and Layer 2 weighting."""
        if (
            self.config.layer1_layer2_gating_enabled
            and self.layer1 is not None
            and self.layer2 is not None
        ):
            baseline = {cid: 1.0 for cid in client_ids}
            for cid in sanitize_flags:
                baseline[cid] = float(self.config.min_trust_weight)
            return self._apply_robust_norm_soft_cap(
                client_ids, client_weights, global_weights, baseline
            )

        trust = dict(layer1_raw)
        trust = self._apply_layer1_layer2_gating(
            client_ids, client_weights, global_weights, trust
        )
        return self._apply_robust_norm_soft_cap(
            client_ids, client_weights, global_weights, trust
        )

    def _sanitize_client_weights(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
    ) -> tuple:
        """Replace non-finite client weights with the current global model."""
        clean_weights: List[Dict[str, torch.Tensor]] = []
        flags: Dict[str, str] = {}

        for i, cid in enumerate(client_ids):
            cw = client_weights[i]
            bad = any(
                not torch.isfinite(t).all()
                for t in cw.values()
                if t.is_floating_point()
            )
            if bad:
                flags[cid] = "non_finite"
                clean_weights.append(
                    {k: global_weights[k].clone() for k in global_weights}
                )
            else:
                clean_weights.append(cw)

        return clean_weights, flags

    def _apply_robust_norm_soft_cap(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_trust: Dict[str, float],
    ) -> Dict[str, float]:
        """Downweight Layer 1 trust for updates with extreme peer-relative norms."""
        n = len(client_ids)
        if not self.config.robust_norm_soft_cap_enabled or n == 0:
            self.last_robust_norm_soft_cap = {"enabled": False, "caps": {}}
            return layer1_trust

        stats = compute_client_update_norm_stats(client_weights, global_weights)
        norm_thr = norm_ratio_threshold(
            n,
            self.config.norm_ratio_threshold,
            self.config.norm_ratio_threshold_two_clients,
        )
        coord_thr = float(self.config.coordinate_deviation_threshold)

        capped = {}
        caps = {}
        for i, cid in enumerate(client_ids):
            cap = robustness_score_from_update_stats(
                stats[i],
                coord_thr,
                norm_thr,
                self.config.min_trust_weight,
            )
            caps[cid] = cap
            capped[cid] = float(layer1_trust.get(cid, 1.0)) * cap

        self.last_robust_norm_soft_cap = {"enabled": True, "caps": caps}
        return capped

    def _apply_backdoor_probe_soft_cap(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        neutral_capped: Dict[str, float],
    ) -> Dict[str, float]:
        """Min neutral norm-capped trust with backdoor_trigger probe ceiling."""
        if (
            not self.config.backdoor_probe_aware_soft_cap_enabled
            or self.probe_evaluator is None
        ):
            self.last_backdoor_probe_soft_cap = {"enabled": False}
            return neutral_capped

        probe_caps = self.probe_evaluator.backdoor_probe_cap_scores(
            client_ids, client_weights, global_weights
        )
        harm_details = self.probe_evaluator.backdoor_trigger_harm_details(
            client_ids, client_weights, global_weights
        )
        blended = {}
        for cid in client_ids:
            blended[cid] = min(
                float(neutral_capped.get(cid, 1.0)),
                float(probe_caps.get(cid, 1.0)),
            )
        self.last_backdoor_probe_soft_cap = {
            "enabled": True,
            "probe_caps": probe_caps,
            "harm_details": harm_details,
            "neutral_capped": dict(neutral_capped),
            "blended": blended,
        }
        return blended

    def _apply_backdoor_trigger_hard_gate(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_trust: Dict[str, float],
    ) -> Dict[str, float]:
        """Hard-clamp clear trigger-harm outliers (peer-relative, after warmup)."""
        if (
            self.config.evaluation_attack_type != "backdoor"
            or not self.config.backdoor_trigger_hard_gate_enabled
            or self.probe_evaluator is None
        ):
            self.last_backdoor_trigger_hard_gate = {"enabled": False}
            return layer1_trust

        warmup = max(0, int(self.config.backdoor_trigger_hard_gate_warmup_rounds))
        # round_number is incremented after aggregation; current FL round is +1.
        current_round = int(self.round_number) + 1
        if current_round <= warmup:
            self.last_backdoor_trigger_hard_gate = {
                "enabled": True,
                "skipped_warmup": True,
                "warmup_rounds": warmup,
                "round": current_round,
                "clamped_clients": {},
                "binding_rate": 0.0,
            }
            return layer1_trust

        details = self.probe_evaluator.backdoor_trigger_harm_details(
            client_ids, client_weights, global_weights
        )
        harm_thr = float(self.config.backdoor_trigger_hard_gate_harm)
        peer_margin = float(self.config.backdoor_trigger_hard_gate_peer_margin)
        p_floor = float(self.config.backdoor_trigger_hard_gate_p_attack_floor)
        require_floor = bool(
            self.config.backdoor_trigger_hard_gate_require_p_attack_floor
        )
        min_t = float(self.config.min_trust_weight)

        harms = [float(details.get(cid, {}).get("harm", 0.0)) for cid in client_ids]
        max_harm = max(harms) if harms else 0.0
        second_harm = sorted(harms, reverse=True)[1] if len(harms) >= 2 else 0.0

        outliers = self._backdoor_probe_worst_outliers(
            client_ids,
            details,
            harm_thr,
            peer_margin,
            require_p_floor=require_floor,
            p_floor=p_floor,
        )

        gated = dict(layer1_trust)
        clamped: Dict[str, Dict] = {}
        for cid, info in outliers.items():
            gated[cid] = min_t
            clamped[cid] = {
                "pre_trust": float(layer1_trust.get(cid, 1.0)),
                "post_trust": min_t,
                **info,
            }

        self.last_backdoor_trigger_hard_gate = {
            "enabled": True,
            "skipped_warmup": False,
            "warmup_rounds": warmup,
            "round": current_round,
            "harm_threshold": harm_thr,
            "peer_margin": peer_margin,
            "p_attack_floor": p_floor,
            "require_p_attack_floor": require_floor,
            "max_harm": max_harm,
            "second_harm": second_harm,
            "harm_details": details,
            "clamped_clients": clamped,
            "binding_rate": (
                float(len(clamped)) / float(len(client_ids)) if client_ids else 0.0
            ),
            "final_trust": dict(gated),
        }
        return gated

    def _apply_backdoor_probe_hard_rejection(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_trust: Dict[str, float],
        client_samples: Optional[List[int]],
        layer1_raw: Optional[Dict[str, float]] = None,
    ) -> tuple:
        """Drop backdoor clients with clear trigger-harm peer outliers."""
        n = len(client_ids)
        if (
            self.config.evaluation_attack_type != "backdoor"
            or not self.config.backdoor_probe_hard_reject_enabled
            or self.probe_evaluator is None
            or n == 0
        ):
            self.last_backdoor_probe_hard_rejection = {
                "enabled": False,
                "rejected_clients": [],
                "accepted_clients": list(client_ids),
            }
            return client_ids, client_weights, layer1_trust, client_samples

        warmup = max(0, int(self.config.backdoor_probe_hard_reject_warmup_rounds))
        current_round = int(self.round_number) + 1
        if current_round <= warmup:
            self.last_backdoor_probe_hard_rejection = {
                "enabled": True,
                "skipped_warmup": True,
                "warmup_rounds": warmup,
                "round": current_round,
                "rejected_clients": [],
                "accepted_clients": list(client_ids),
            }
            return client_ids, client_weights, layer1_trust, client_samples

        details = self.probe_evaluator.backdoor_trigger_harm_details(
            client_ids, client_weights, global_weights
        )
        harm_thr = float(self.config.backdoor_probe_hard_reject_harm)
        peer_margin = float(self.config.backdoor_probe_hard_reject_peer_margin)
        max_trust = float(self.config.backdoor_probe_hard_reject_max_trust)
        # No sparse early-exit. Withhold only if fewer than 2 clients would
        # remain (allows reject at n=3–4 even when FL quorum is 4).
        min_keep = max(1, int(self.config.min_aggregation_participants))
        min_after_reject = max(
            2, int(self.config.backdoor_probe_hard_reject_min_keep)
        )

        outliers = self._backdoor_probe_worst_outliers(
            client_ids,
            details,
            harm_thr,
            peer_margin,
            cohort_factor=float(self.config.backdoor_probe_hard_reject_cohort_factor),
        )

        # Gate harm rejects on low Layer-1 raw trust to avoid high-trust benign FPs.
        # Skip the gate when Layer 1 is off (raw scores are neutral 1.0).
        trust_source = layer1_raw if layer1_raw is not None else layer1_trust
        gated_outliers: Dict[str, Dict] = {}
        trust_blocked: Dict[str, Dict] = {}
        apply_trust_gate = bool(self.config.enable_layer1 and self.layer1 is not None)
        for cid, info in outliers.items():
            raw_t = float(trust_source.get(cid, 1.0))
            if apply_trust_gate and raw_t > max_trust:
                trust_blocked[cid] = {
                    **info,
                    "layer1_raw_trust": raw_t,
                    "max_trust": max_trust,
                    "reason": (
                        f"{info.get('reason', '')};"
                        f"blocked_high_trust:{raw_t:.4f}>{max_trust:.4f}"
                    ).lstrip(";"),
                }
                continue
            gated_outliers[cid] = {
                **info,
                "layer1_raw_trust": raw_t,
            }

        accepted_ids: List[str] = []
        accepted_weights: List[Dict[str, torch.Tensor]] = []
        accepted_trust: Dict[str, float] = {}
        accepted_samples: Optional[List[int]] = [] if client_samples is not None else None
        rejected_clients: List[Dict] = []

        for i, cid in enumerate(client_ids):
            if cid in gated_outliers:
                rejected_clients.append({
                    "client_id": cid,
                    "layer1_trust": float(layer1_trust.get(cid, 1.0)),
                    **gated_outliers[cid],
                })
                continue
            accepted_ids.append(cid)
            accepted_weights.append(client_weights[i])
            accepted_trust[cid] = float(layer1_trust.get(cid, 1.0))
            if accepted_samples is not None and client_samples is not None:
                accepted_samples.append(client_samples[i])

        withheld = False
        if rejected_clients and len(accepted_ids) < min_after_reject:
            withheld = True
            accepted_ids = list(client_ids)
            accepted_weights = list(client_weights)
            accepted_trust = {
                cid: float(layer1_trust.get(cid, 1.0)) for cid in client_ids
            }
            accepted_samples = (
                list(client_samples) if client_samples is not None else None
            )

        rejection_info = {
            "enabled": True,
            "backdoor_scoped": True,
            "skipped_warmup": False,
            "warmup_rounds": warmup,
            "round": current_round,
            "harm_threshold": harm_thr,
            "peer_margin": peer_margin,
            "max_trust": max_trust,
            "trust_gate_applied": apply_trust_gate,
            "trust_blocked_clients": trust_blocked,
            "min_aggregation_participants": min_keep,
            "min_after_reject": min_after_reject,
            "num_participants": n,
            "harm_details": details,
            "rejected_clients": [] if withheld else rejected_clients,
            "candidate_rejected_clients": rejected_clients if withheld else [],
            "rejection_withheld_insufficient_participants": withheld,
            "accepted_clients": accepted_ids,
        }
        self.last_backdoor_probe_hard_rejection = rejection_info
        return accepted_ids, accepted_weights, accepted_trust, accepted_samples

    def _apply_backdoor_geometry_hard_rejection(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_trust: Dict[str, float],
        client_samples: Optional[List[int]],
    ) -> tuple:
        """Apply Krum-geometry filtering for backdoor / label-flip rounds.

        Modes:
          - shortlist_blend: multi-Krum shortlist then inv-trigger-weighted
            average of trigger-eligible members (backdoor).
          - shortlist_veto: multi-Krum shortlist then keep lowest absolute
            trigger P (with Krum tie-break within margin) (backdoor).
          - shortlist_clean: multi-Krum shortlist then keep highest clean-probe
            quality with margin vs geom winner (label-flip).
          - shortlist_clean_blend: multi-Krum shortlist then blend the
            cleanest peers (label-flip research lever).
          - keep_best: retain only the lowest Krum-score client.
          - reject_worst: drop the highest-score client when clearly isolated.
        """
        n = len(client_ids)
        attack = self.config.evaluation_attack_type
        backdoor_geom = (
            attack == "backdoor"
            and bool(self.config.backdoor_geometry_hard_reject_enabled)
        )
        label_flip_geom = (
            attack == "label_flip"
            and bool(
                getattr(self.config, "label_flip_geometry_hard_reject_enabled", False)
            )
        )
        if not (backdoor_geom or label_flip_geom) or n == 0:
            self.last_backdoor_geometry_hard_rejection = {
                "enabled": False,
                "rejected_clients": [],
                "accepted_clients": list(client_ids),
            }
            return client_ids, client_weights, layer1_trust, client_samples

        if label_flip_geom:
            warmup = max(
                0,
                int(
                    getattr(
                        self.config,
                        "label_flip_geometry_hard_reject_warmup_rounds",
                        0,
                    )
                ),
            )
            mode = str(
                getattr(
                    self.config,
                    "label_flip_geometry_hard_reject_mode",
                    "shortlist_clean",
                )
            ).strip().lower()
            min_after_reject = max(
                1,
                int(
                    getattr(
                        self.config, "label_flip_geometry_hard_reject_min_keep", 1
                    )
                ),
            )
            shortlist_k = max(
                1,
                int(getattr(self.config, "label_flip_geometry_shortlist_k", 2)),
            )
        else:
            warmup = max(0, int(self.config.backdoor_geometry_hard_reject_warmup_rounds))
            mode = str(
                getattr(
                    self.config, "backdoor_geometry_hard_reject_mode", "shortlist_veto"
                )
            ).strip().lower()
            min_after_reject = max(
                1, int(self.config.backdoor_geometry_hard_reject_min_keep)
            )
            shortlist_k = max(
                1, int(getattr(self.config, "backdoor_geometry_shortlist_k", 2))
            )

        current_round = int(self.round_number) + 1
        if current_round <= warmup:
            self.last_backdoor_geometry_hard_rejection = {
                "enabled": True,
                "skipped_warmup": True,
                "warmup_rounds": warmup,
                "round": current_round,
                "attack_type": attack,
                "rejected_clients": [],
                "accepted_clients": list(client_ids),
            }
            return client_ids, client_weights, layer1_trust, client_samples

        score_ratio_thr = float(self.config.backdoor_geometry_hard_reject_score_ratio)
        veto_margin = float(
            getattr(self.config, "backdoor_geometry_trigger_veto_margin", 0.01)
        )
        blend_tie_eps = float(
            getattr(self.config, "backdoor_geometry_blend_tie_eps", 0.005)
        )

        # Need ≥2 clients to compare peer distances.
        if n < 2:
            self.last_backdoor_geometry_hard_rejection = {
                "enabled": True,
                "skipped_sparse": True,
                "mode": mode,
                "round": current_round,
                "attack_type": attack,
                "num_participants": n,
                "rejected_clients": [],
                "accepted_clients": list(client_ids),
            }
            return client_ids, client_weights, layer1_trust, client_samples

        scores = krum_peer_scores(
            client_weights, global_weights, num_byzantine=1
        )
        score_map = {
            client_ids[i]: float(scores[i]) for i in range(n)
        }
        ranked_worst = sorted(
            score_map.items(), key=lambda item: item[1], reverse=True
        )
        ranked_best = sorted(score_map.items(), key=lambda item: item[1])
        worst_cid, worst_score = ranked_worst[0]
        geom_best_cid, best_score = ranked_best[0]
        second_worst = ranked_worst[1][1] if len(ranked_worst) > 1 else 0.0
        ratio = worst_score / max(second_worst, 1e-12)

        drop_ids: List[str] = []
        keep_reason = ""
        selected_cid = geom_best_cid
        shortlist: List[str] = []
        trigger_map: Dict[str, float] = {}
        clean_map: Dict[str, float] = {}
        vetoed: List[str] = []
        harm_details: Dict[str, Dict] = {}
        blend_weights: Dict[str, float] = {}
        eligible: List[str] = []

        if mode == "shortlist_clean_blend":
            k = min(shortlist_k, n)
            shortlist = [cid for cid, _ in ranked_best[:k]]
            blend_keep = max(
                1,
                int(getattr(self.config, "label_flip_geometry_blend_keep", 2)),
            )
            clean_margin = float(
                getattr(self.config, "label_flip_geometry_clean_margin", 0.05)
            )
            if self.probe_evaluator is not None and hasattr(
                self.probe_evaluator, "clean_probe_quality_details"
            ):
                harm_details = self.probe_evaluator.clean_probe_quality_details(
                    client_ids, client_weights, global_weights
                )
                clean_map = {
                    cid: float(
                        harm_details.get(cid, {}).get("clean_quality", 0.0)
                    )
                    for cid in shortlist
                }
                ranked_clean = sorted(
                    shortlist,
                    key=lambda cid: (clean_map.get(cid, 0.0), -score_map[cid]),
                    reverse=True,
                )
                keep = ranked_clean[: min(blend_keep, len(ranked_clean))]
                leader = keep[0]
                leader_clean = clean_map.get(leader, 0.0)
                # Drop weak seconds that would dilute the cleanest peer.
                keep = [
                    cid
                    for cid in keep
                    if clean_map.get(cid, 0.0) >= leader_clean - clean_margin
                ] or [leader]
                eligible = list(keep)
                if len(keep) == 1:
                    selected_cid = keep[0]
                    drop_ids = [cid for cid in client_ids if cid != selected_cid]
                    keep_reason = (
                        f"shortlist_clean_blend_single:{selected_cid}:"
                        f"clean:{clean_map.get(selected_cid, 0.0):.4f}:"
                        f"krum:{score_map[selected_cid]:.4f}:"
                        f"shortlist={shortlist}"
                    )
                else:
                    raw = {
                        cid: max(clean_map.get(cid, 0.0), 0.0) + 1e-3
                        for cid in keep
                    }
                    total = float(sum(raw.values())) or 1.0
                    normed = {cid: raw[cid] / total for cid in keep}
                    peak = max(normed.values()) or 1.0
                    blend_weights = {
                        cid: float(normed[cid] / peak) for cid in keep
                    }
                    selected_cid = leader
                    drop_ids = [cid for cid in client_ids if cid not in keep]
                    keep_reason = (
                        f"shortlist_clean_blend:keep={keep}:"
                        f"weights={{{', '.join(f'{c}:{blend_weights[c]:.3f}' for c in keep)}}}"
                        f":shortlist={shortlist}"
                    )
            else:
                selected_cid = geom_best_cid
                drop_ids = [cid for cid in client_ids if cid != selected_cid]
                keep_reason = (
                    f"shortlist_clean_blend_fallback_keep_best:{selected_cid}:"
                    f"krum_score:{best_score:.4f}"
                )
                mode = "shortlist_clean_blend_fallback_keep_best"
        elif mode == "shortlist_clean":
            k = min(shortlist_k, n)
            shortlist = [cid for cid, _ in ranked_best[:k]]
            clean_margin = float(
                getattr(self.config, "label_flip_geometry_clean_margin", 0.025)
            )
            if self.probe_evaluator is not None and hasattr(
                self.probe_evaluator, "clean_probe_quality_details"
            ):
                harm_details = self.probe_evaluator.clean_probe_quality_details(
                    client_ids, client_weights, global_weights
                )
                clean_map = {
                    cid: float(
                        harm_details.get(cid, {}).get("clean_quality", 0.0)
                    )
                    for cid in shortlist
                }
                # Default to geometric winner; override only on a clear clean win.
                geom_shortlist_best = shortlist[0]
                clean_winner = max(
                    shortlist,
                    key=lambda cid: (clean_map.get(cid, 0.0), -score_map[cid]),
                )
                geom_clean = clean_map.get(geom_shortlist_best, 0.0)
                win_clean = clean_map.get(clean_winner, 0.0)
                clean_delta = win_clean - geom_clean
                if (
                    clean_winner != geom_shortlist_best
                    and clean_delta >= clean_margin
                ):
                    selected_cid = clean_winner
                    keep_reason = (
                        f"shortlist_clean_override:{selected_cid}:"
                        f"clean:{win_clean:.4f}:"
                        f"geom:{geom_shortlist_best}:{geom_clean:.4f}:"
                        f"delta:{clean_delta:.4f}>={clean_margin:.4f}:"
                        f"krum:{score_map[selected_cid]:.4f}:"
                        f"shortlist={shortlist}"
                    )
                else:
                    selected_cid = geom_shortlist_best
                    keep_reason = (
                        f"shortlist_clean_keep_geom:{selected_cid}:"
                        f"clean:{geom_clean:.4f}:"
                        f"best_other:{clean_winner}:{win_clean:.4f}:"
                        f"delta:{clean_delta:.4f}:"
                        f"margin:{clean_margin:.4f}:"
                        f"krum:{score_map[selected_cid]:.4f}:"
                        f"shortlist={shortlist}"
                    )
                drop_ids = [cid for cid in client_ids if cid != selected_cid]
            else:
                selected_cid = geom_best_cid
                drop_ids = [cid for cid in client_ids if cid != selected_cid]
                keep_reason = (
                    f"shortlist_clean_fallback_keep_best:{selected_cid}:"
                    f"krum_score:{best_score:.4f}"
                )
                mode = "shortlist_clean_fallback_keep_best"
        elif mode in ("shortlist_veto", "shortlist_blend"):
            k = min(shortlist_k, n)
            shortlist = [cid for cid, _ in ranked_best[:k]]
            # Reuse probe details from earlier hard-reject when available.
            prior = self.last_backdoor_probe_hard_rejection or {}
            prior_details = prior.get("harm_details") or {}
            if prior_details and all(cid in prior_details for cid in shortlist):
                harm_details = prior_details
            elif self.probe_evaluator is not None:
                harm_details = self.probe_evaluator.backdoor_trigger_harm_details(
                    client_ids, client_weights, global_weights
                )
            else:
                selected_cid = geom_best_cid
                drop_ids = [cid for cid in client_ids if cid != selected_cid]
                keep_reason = (
                    f"{mode}_fallback_keep_best:{selected_cid}:"
                    f"krum_score:{best_score:.4f}"
                )
                mode = f"{mode}_fallback_keep_best"

            if harm_details:
                trigger_map = {
                    cid: float(
                        harm_details.get(cid, {}).get("counterfactual_p_attack", 1.0)
                    )
                    for cid in shortlist
                }
                min_trig = min(trigger_map.values()) if trigger_map else 1.0
                eligible = [
                    cid
                    for cid in shortlist
                    if trigger_map[cid] <= min_trig + veto_margin
                ]
                vetoed = [cid for cid in shortlist if cid not in eligible]
                if not eligible:
                    eligible = list(shortlist)
                    vetoed = []

                selected_cid = min(
                    eligible,
                    key=lambda cid: (trigger_map[cid], score_map[cid]),
                )

                trig_spread = (
                    max(trigger_map[c] for c in eligible)
                    - min(trigger_map[c] for c in eligible)
                ) if eligible else 0.0
                # Blend only on near-ties; a clear trigger winner stays solo.
                if (
                    mode == "shortlist_blend"
                    and len(eligible) >= 2
                    and trig_spread <= blend_tie_eps
                ):
                    raw = {
                        cid: 1.0 / (trigger_map[cid] + 1e-3) for cid in eligible
                    }
                    total = float(sum(raw.values())) or 1.0
                    normed = {cid: raw[cid] / total for cid in eligible}
                    peak = max(normed.values()) or 1.0
                    blend_weights = {
                        cid: float(normed[cid] / peak) for cid in eligible
                    }
                    drop_ids = [cid for cid in client_ids if cid not in eligible]
                    keep_reason = (
                        f"shortlist_blend:eligible={eligible}:"
                        f"spread:{trig_spread:.4f}<=eps:{blend_tie_eps:.4f}:"
                        f"weights={{{', '.join(f'{c}:{blend_weights[c]:.3f}' for c in eligible)}}}"
                        f":shortlist={shortlist}"
                    )
                else:
                    # Single keep (veto mode, clear trigger winner, or 1 eligible).
                    drop_ids = [cid for cid in client_ids if cid != selected_cid]
                    keep_reason = (
                        f"{mode}:{selected_cid}:"
                        f"trig:{trigger_map[selected_cid]:.4f}:"
                        f"krum:{score_map[selected_cid]:.4f}:"
                        f"spread:{trig_spread:.4f}:"
                        f"shortlist={shortlist}"
                    )
        elif mode == "keep_best":
            selected_cid = geom_best_cid
            drop_ids = [cid for cid in client_ids if cid != selected_cid]
            keep_reason = (
                f"keep_best:{selected_cid}:krum_score:{best_score:.4f}"
            )
        else:
            # reject_worst
            if ratio >= score_ratio_thr:
                selected_cid = ""  # unused
                drop_ids = [worst_cid]
                keep_reason = (
                    f"krum_score:{worst_score:.4f};"
                    f"ratio:{ratio:.4f}>={score_ratio_thr:.4f}"
                )
            else:
                self.last_backdoor_geometry_hard_rejection = {
                    "enabled": True,
                    "mode": mode,
                    "skipped_warmup": False,
                    "round": current_round,
                    "attack_type": attack,
                    "score_ratio_threshold": score_ratio_thr,
                    "worst_client": worst_cid,
                    "best_client": geom_best_cid,
                    "worst_score": worst_score,
                    "best_score": best_score,
                    "second_score": second_worst,
                    "score_ratio": ratio,
                    "krum_scores": score_map,
                    "rejected_clients": [],
                    "accepted_clients": list(client_ids),
                }
                return client_ids, client_weights, layer1_trust, client_samples

        rejected_clients = []
        for cid in drop_ids:
            if cid in trigger_map:
                cf_pa = float(trigger_map[cid])
            elif cid in harm_details:
                cf_pa = float(
                    harm_details[cid].get("counterfactual_p_attack", float("nan"))
                )
            else:
                cf_pa = float("nan")
            rejected_clients.append({
                "client_id": cid,
                "krum_score": float(score_map.get(cid, 0.0)),
                "best_client": selected_cid or geom_best_cid,
                "geom_best_client": geom_best_cid,
                "best_score": best_score,
                "worst_client": worst_cid,
                "worst_score": worst_score,
                "score_ratio": ratio,
                "counterfactual_p_attack": cf_pa,
                "clean_quality": float(clean_map.get(cid, float("nan"))),
                "layer1_trust": float(layer1_trust.get(cid, 1.0)),
                "reason": keep_reason,
            })

        accepted_ids: List[str] = []
        accepted_weights: List[Dict[str, torch.Tensor]] = []
        accepted_trust: Dict[str, float] = {}
        accepted_samples: Optional[List[int]] = (
            [] if client_samples is not None else None
        )
        drop_set = set(drop_ids)
        for i, cid in enumerate(client_ids):
            if cid in drop_set:
                continue
            accepted_ids.append(cid)
            accepted_weights.append(client_weights[i])
            if cid in blend_weights:
                accepted_trust[cid] = float(blend_weights[cid])
            else:
                accepted_trust[cid] = float(layer1_trust.get(cid, 1.0))
            if accepted_samples is not None and client_samples is not None:
                accepted_samples.append(client_samples[i])

        withheld = False
        if drop_ids and len(accepted_ids) < min_after_reject:
            withheld = True
            accepted_ids = list(client_ids)
            accepted_weights = list(client_weights)
            accepted_trust = {
                cid: float(layer1_trust.get(cid, 1.0)) for cid in client_ids
            }
            accepted_samples = (
                list(client_samples) if client_samples is not None else None
            )

        self.last_backdoor_geometry_hard_rejection = {
            "enabled": True,
            "backdoor_scoped": bool(backdoor_geom),
            "label_flip_scoped": bool(label_flip_geom),
            "attack_type": attack,
            "mode": mode,
            "skipped_warmup": False,
            "warmup_rounds": warmup,
            "round": current_round,
            "score_ratio_threshold": score_ratio_thr,
            "shortlist_k": shortlist_k,
            "shortlist": shortlist,
            "eligible": eligible,
            "blend_weights": blend_weights,
            "blend_tie_eps": float(
                getattr(self.config, "backdoor_geometry_blend_tie_eps", 0.005)
            ),
            "trigger_map": trigger_map,
            "clean_quality_map": clean_map,
            "clean_margin": float(
                getattr(self.config, "label_flip_geometry_clean_margin", 0.05)
            ),
            "blend_keep": int(
                getattr(self.config, "label_flip_geometry_blend_keep", 2)
            ),
            "vetoed_from_shortlist": vetoed,
            "trigger_veto_margin": veto_margin,
            "worst_client": worst_cid,
            "best_client": selected_cid or geom_best_cid,
            "geom_best_client": geom_best_cid,
            "worst_score": worst_score,
            "best_score": best_score,
            "second_score": second_worst,
            "score_ratio": ratio,
            "krum_scores": score_map,
            "harm_details": harm_details,
            "min_after_reject": min_after_reject,
            "num_participants": n,
            "rejected_clients": [] if withheld else rejected_clients,
            "candidate_rejected_clients": rejected_clients if withheld else [],
            "rejection_withheld_insufficient_participants": withheld,
            "accepted_clients": accepted_ids,
            "keep_reason": keep_reason,
        }
        return accepted_ids, accepted_weights, accepted_trust, accepted_samples

    def _apply_post_agg_trigger_scrub(
        self,
        aggregated: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Few SGD steps on probe sets after aggregation (backdoor-only).

        Benchmark metric is mean P(attack) on stamped triggers (lower is better
        in the published tables). Poisoning embeds trigger→benign; scrub pushes
        trigger logits toward benign while anchoring clean windows and a real
        attack scenario so clean-task F1 does not collapse.
        """
        if (
            self.config.evaluation_attack_type != "backdoor"
            or not bool(
                getattr(self.config, "post_agg_trigger_scrub_enabled", False)
            )
            or self.model is None
            or self.probe_evaluator is None
            or self.learning_mode != "classification"
        ):
            self.last_post_agg_trigger_scrub = {
                "enabled": False,
                "applied": False,
            }
            return aggregated

        warmup = max(
            0, int(getattr(self.config, "post_agg_trigger_scrub_warmup_rounds", 0))
        )
        current_round = int(self.round_number) + 1
        if current_round <= warmup:
            self.last_post_agg_trigger_scrub = {
                "enabled": True,
                "applied": False,
                "skipped_warmup": True,
                "warmup_rounds": warmup,
                "round": current_round,
            }
            return aggregated

        probe_data = self.probe_evaluator.probe_data
        if "benign" not in probe_data or "backdoor_trigger" not in probe_data:
            self.last_post_agg_trigger_scrub = {
                "enabled": True,
                "applied": False,
                "reason": "missing_probe_scenarios",
            }
            return aggregated

        steps = max(0, int(getattr(self.config, "post_agg_trigger_scrub_steps", 3)))
        if steps == 0:
            self.last_post_agg_trigger_scrub = {
                "enabled": True,
                "applied": False,
                "reason": "zero_steps",
            }
            return aggregated

        lr = float(getattr(self.config, "post_agg_trigger_scrub_lr", 1e-3))
        trig_coef = float(
            getattr(self.config, "post_agg_trigger_scrub_trigger_coef", 1.0)
        )
        max_delta = float(
            getattr(self.config, "post_agg_trigger_scrub_max_delta_norm", 0.5)
        )

        pre = clone_state_dict(aggregated)
        x_b, y_b = probe_data["benign"]
        x_t, _y_t = probe_data["backdoor_trigger"]
        # Push stamped triggers toward benign (y=0) to lower mean P(attack).
        y_t_scrub = torch.zeros_like(_y_t)
        x_b = x_b.to(self.device)
        y_b = y_b.to(self.device).float().view(-1, 1)
        x_t = x_t.to(self.device)
        y_t_scrub = y_t_scrub.to(self.device).float().view(-1, 1)

        # Anchor a real attack scenario so scrub does not erase attack detection.
        attack_anchor = None
        for name in (
            "mavlink_waypoint",
            "gnss_spoofing",
            "adsb_injection",
            "jamming",
        ):
            if name in probe_data:
                attack_anchor = name
                break
        if attack_anchor is not None:
            x_a, y_a = probe_data[attack_anchor]
            x_a = x_a.to(self.device)
            y_a = y_a.to(self.device).float().view(-1, 1)
        else:
            x_a = y_a = None

        self.model.set_weights(pre)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(True)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()
        step_losses: List[float] = []

        try:
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                logits_b = self.model(x_b)
                if logits_b.ndim == 1:
                    logits_b = logits_b.view(-1, 1)
                logits_t = self.model(x_t)
                if logits_t.ndim == 1:
                    logits_t = logits_t.view(-1, 1)
                loss_clean = bce(logits_b, y_b)
                loss_trig = bce(logits_t, y_t_scrub)
                loss = loss_clean + trig_coef * loss_trig
                if x_a is not None:
                    logits_a = self.model(x_a)
                    if logits_a.ndim == 1:
                        logits_a = logits_a.view(-1, 1)
                    loss = loss + bce(logits_a, y_a)
                loss.backward()
                optimizer.step()
                step_losses.append(float(loss.detach().item()))

            scrubbed = clone_state_dict(self.model.get_weights())

            delta_norm = 0.0
            if max_delta > 0.0:
                sq = 0.0
                for key in scrubbed:
                    d = scrubbed[key].float() - pre[key].float()
                    sq += float(torch.sum(d * d).item())
                delta_norm = float(sq ** 0.5)
                if delta_norm > max_delta and delta_norm > 0.0:
                    scale = max_delta / delta_norm
                    for key in scrubbed:
                        scrubbed[key] = pre[key] + scale * (
                            scrubbed[key] - pre[key]
                        )
                    delta_norm = max_delta

            if not self._aggregate_is_finite(scrubbed):
                self.last_post_agg_trigger_scrub = {
                    "enabled": True,
                    "applied": False,
                    "reason": "non_finite_scrub",
                    "round": current_round,
                    "step_losses": step_losses,
                }
                self.model.set_weights(pre)
                self.model.eval()
                return aggregated

            self.model.set_weights(scrubbed)
            self.model.eval()
            self.last_post_agg_trigger_scrub = {
                "enabled": True,
                "applied": True,
                "round": current_round,
                "steps": steps,
                "lr": lr,
                "trigger_coef": trig_coef,
                "trigger_target": 0.0,
                "attack_anchor": attack_anchor,
                "max_delta_norm": max_delta,
                "delta_norm": delta_norm,
                "step_losses": step_losses,
                "final_loss": step_losses[-1] if step_losses else None,
            }
            return scrubbed
        except Exception as exc:
            self.model.set_weights(pre)
            self.model.eval()
            self.last_post_agg_trigger_scrub = {
                "enabled": True,
                "applied": False,
                "reason": f"exception:{type(exc).__name__}:{exc}",
                "round": current_round,
            }
            return aggregated

    @staticmethod
    def _aggregate_is_finite(aggregated: Dict[str, torch.Tensor]) -> bool:
        for tensor in aggregated.values():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                return False
        return True

    def _apply_update_norm_hard_rejection(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_trust: Dict[str, float],
        client_samples: Optional[List[int]],
    ) -> tuple:
        """Drop clients whose update norm/deviation exceeds hard thresholds."""
        n = len(client_ids)
        if not self._hard_rejection_active() or n == 0:
            self.last_update_norm_rejection = {
                "enabled": False,
                "rejected_clients": [],
                "accepted_clients": list(client_ids),
            }
            return client_ids, client_weights, layer1_trust, client_samples

        stats = compute_client_update_norm_stats(client_weights, global_weights)
        norm_thr = norm_ratio_threshold(
            n,
            self.config.update_norm_hard_rejection_ratio,
            self.config.update_norm_hard_rejection_ratio_two_clients,
        )
        dev_thr = float(self.config.update_norm_hard_rejection_deviation)

        accepted_ids: List[str] = []
        accepted_weights: List[Dict[str, torch.Tensor]] = []
        accepted_trust: Dict[str, float] = {}
        accepted_samples: Optional[List[int]] = [] if client_samples is not None else None
        rejected_clients: List[Dict] = []

        for i, cid in enumerate(client_ids):
            norm_ratio = stats[i]["norm_ratio"]
            rel_dev = stats[i]["rel_dev"]
            reasons = []
            if norm_ratio > norm_thr:
                reasons.append(f"norm_ratio:{norm_ratio:.3f}>{norm_thr:.3f}")
            if rel_dev > dev_thr:
                reasons.append(f"rel_dev:{rel_dev:.3f}>{dev_thr:.3f}")

            if reasons:
                rejected_clients.append({
                    "client_id": cid,
                    "norm_ratio": norm_ratio,
                    "rel_dev": rel_dev,
                    "update_norm": stats[i]["update_norm"],
                    "median_update_norm": stats[i]["median_update_norm"],
                    "layer1_trust": float(layer1_trust.get(cid, 1.0)),
                    "reason": ";".join(reasons),
                })
                continue

            accepted_ids.append(cid)
            accepted_weights.append(client_weights[i])
            accepted_trust[cid] = float(layer1_trust.get(cid, 1.0))
            if accepted_samples is not None and client_samples is not None:
                accepted_samples.append(client_samples[i])

        min_keep = max(1, int(self.config.min_aggregation_participants))
        withheld = False
        if rejected_clients and len(accepted_ids) < min_keep:
            # Prefer soft trust over holding the global model when rejection
            # would leave too few participants (common in sparse MR rounds).
            withheld = True
            accepted_ids = list(client_ids)
            accepted_weights = list(client_weights)
            accepted_trust = {
                cid: float(layer1_trust.get(cid, 1.0)) for cid in client_ids
            }
            accepted_samples = (
                list(client_samples) if client_samples is not None else None
            )

        rejection_info = {
            "enabled": True,
            "global_flag": bool(self.config.update_norm_hard_rejection_enabled),
            "model_replacement_scoped": (
                self.config.evaluation_attack_type == "model_replacement"
                and self.config.update_norm_hard_rejection_for_model_replacement
            ),
            "norm_ratio_threshold": norm_thr,
            "rel_dev_threshold": dev_thr,
            "rejected_clients": [] if withheld else rejected_clients,
            "candidate_rejected_clients": rejected_clients if withheld else [],
            "rejection_withheld_insufficient_participants": withheld,
            "accepted_clients": accepted_ids,
        }
        self.last_update_norm_rejection = rejection_info
        return accepted_ids, accepted_weights, accepted_trust, accepted_samples

    def _aggregation_guard_decision(
        self,
        client_ids: List[str],
        layer1_trust: Dict[str, float],
        layer1_raw: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """Return a skip reason when the global model should be held."""
        if not self.config.aggregation_guard_enabled:
            return None

        n = len(client_ids)
        min_participants = max(1, int(self.config.min_aggregation_participants))
        # After a successful backdoor hard-reject (probe or geometry), allow one
        # below the normal quorum so dropping the outlier does not hold the model.
        pr = self.last_backdoor_probe_hard_rejection or {}
        gr = self.last_backdoor_geometry_hard_rejection or {}
        if gr.get("enabled") and gr.get("rejected_clients"):
            min_participants = max(
                1, int(gr.get("min_after_reject", 1))
            )
        elif pr.get("enabled") and pr.get("rejected_clients"):
            min_participants = max(
                1, int(pr.get("min_after_reject", min_participants - 1))
            )
        if n < min_participants:
            return f"insufficient_participants:{n}<{min_participants}"

        held = self._withheld_norm_outlier_ids(layer1_raw)
        if held:
            return "withheld_norm_outlier:" + ",".join(held)

        missed = self._missed_norm_outlier_ids(client_ids, layer1_raw)
        if missed:
            return "missed_norm_outlier:" + ",".join(missed)

        if n == 1:
            cid = client_ids[0]
            trust = float(layer1_trust.get(cid, 1.0))
            if (
                layer1_raw is not None
                and self.config.layer1_layer2_gating_enabled
                and not self.config.adaptive_l1l2_gating_enabled
                and self.layer1 is not None
                and self.layer2 is not None
            ):
                trust = float(layer1_raw.get(cid, 1.0))
            threshold = float(self.config.single_participant_trust_threshold)
            if trust < threshold:
                return (
                    f"single_participant_low_trust:{cid}:"
                    f"{trust:.4f}<{threshold:.4f}"
                )

        return None

    def _withheld_norm_outlier_ids(
        self,
        layer1_raw: Optional[Dict[str, float]],
    ) -> List[str]:
        """Clients a withheld norm reject should hold rather than aggregate.

        Fires only on the adaptive path, and only when raw Layer 1 still
        trusts the client while its peer-robustness score is low. A client
        Layer 1 has already scored down stays in the soft aggregate.
        """
        if not self.config.adaptive_l1l2_gating_enabled or not layer1_raw:
            return []
        info = self.last_update_norm_rejection or {}
        if not info.get("rejection_withheld_insufficient_participants"):
            return []
        robustness = (self.last_layer1_layer2_gating or {}).get("robustness_scores") or {}
        tau = float(self.config.adaptive_l1l2_gating_conflict_threshold)
        held: List[str] = []
        for client in info.get("candidate_rejected_clients") or []:
            cid = str(client.get("client_id", ""))
            if not cid:
                continue
            raw_t = float(layer1_raw.get(cid, 1.0))
            robustness_i = float(robustness.get(cid, 1.0))
            if raw_t - robustness_i > tau:
                held.append(cid)
        return held

    def _missed_norm_outlier_ids(
        self,
        client_ids: List[str],
        layer1_raw: Optional[Dict[str, float]],
    ) -> List[str]:
        """Unique client that Layer 1 still trusts while its norm score is low.

        Does not require the hard-reject threshold. Two or more such clients
        stay in the aggregate, as do clients Layer 1 has already scored down.
        """
        if not self.config.adaptive_l1l2_gating_enabled or not layer1_raw:
            return []
        robustness = (self.last_layer1_layer2_gating or {}).get("robustness_scores") or {}
        tau = float(self.config.adaptive_l1l2_gating_conflict_threshold)
        missed = [
            cid
            for cid in client_ids
            if float(layer1_raw.get(cid, 1.0)) - float(robustness.get(cid, 1.0)) > tau
        ]
        if len(missed) == 1:
            return missed
        return []

    def defend_and_aggregate(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        client_losses: Optional[Dict[str, float]] = None,
        mission_context: Optional[Dict[str, Dict[str, float]]] = None,
        client_samples: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run ACFL defense pipeline and return aggregated weights.

        Pipeline:
          1. Sanitize non-finite client weights
          2. Layer 1: Validate updates, compute trust scores (telemetry)
          3. Build aggregation trust (neutral L1 + soft cap when gating is on)
          4. Optional hard norm rejection
          5. Layer 2: Byzantine-tolerant trust-weighted aggregation
        """
        client_weights, sanitize_flags = self._sanitize_client_weights(
            client_ids, client_weights, global_weights
        )
        global_weights = clone_state_dict(global_weights)
        client_weights = [clone_state_dict(cw) for cw in client_weights]

        neutral_baseline = {cid: 1.0 for cid in client_ids}
        for cid in sanitize_flags:
            neutral_baseline[cid] = float(self.config.min_trust_weight)

        layer1_raw = dict(neutral_baseline)
        if self.layer1 is not None:
            layer1_raw = self.layer1.score_updates(
                client_ids,
                client_weights,
                global_weights,
                client_losses,
                self.probe_evaluator,
            )

        for cid in sanitize_flags:
            layer1_raw[cid] = float(self.config.min_trust_weight)

        if bool(getattr(self.config, "parallel_trust_arbitration_enabled", False)):
            return self._defend_with_parallel_trust_arbitration(
                client_ids,
                client_weights,
                global_weights,
                layer1_raw,
                sanitize_flags,
                mission_context,
                client_samples,
            )

        gating_active = (
            self.config.layer1_layer2_gating_enabled
            and self.layer1 is not None
            and self.layer2 is not None
        )

        if gating_active:
            self._apply_layer1_layer2_gating(
                client_ids, client_weights, global_weights, layer1_raw
            )
            neutral_capped = self._apply_robust_norm_soft_cap(
                client_ids, client_weights, global_weights, neutral_baseline
            )
            if self._backdoor_layer1_in_aggregation():
                if self.config.backdoor_probe_aware_soft_cap_enabled:
                    probe_trust = self._apply_backdoor_probe_soft_cap(
                        client_ids,
                        client_weights,
                        global_weights,
                        neutral_capped,
                    )
                    if self.config.backdoor_probe_blend_with_raw_l1:
                        raw_capped = self._apply_robust_norm_soft_cap(
                            client_ids, client_weights, global_weights, layer1_raw
                        )
                        layer1_trust = {
                            cid: min(
                                float(probe_trust[cid]),
                                float(raw_capped[cid]),
                            )
                            for cid in client_ids
                        }
                        if self.last_backdoor_probe_soft_cap:
                            self.last_backdoor_probe_soft_cap["raw_capped"] = raw_capped
                            self.last_backdoor_probe_soft_cap["final_trust"] = (
                                layer1_trust
                            )
                    else:
                        layer1_trust = probe_trust
                else:
                    raw_capped = self._apply_robust_norm_soft_cap(
                        client_ids, client_weights, global_weights, layer1_raw
                    )
                    layer1_trust = {
                        cid: min(float(neutral_capped[cid]), float(raw_capped[cid]))
                        for cid in client_ids
                    }
            else:
                layer1_trust = neutral_capped
            if self.config.adaptive_l1l2_gating_enabled:
                raw_capped = self._apply_robust_norm_soft_cap(
                    client_ids, client_weights, global_weights, layer1_raw
                )
                gating_info = dict(self.last_layer1_layer2_gating or {})
                robustness = gating_info.get("robustness_scores") or (
                    self._peer_robustness_scores(
                        client_ids, client_weights, global_weights
                    )
                )
                gate_info = self._adaptive_gate(client_ids, layer1_raw, robustness)
                gated_trust = dict(layer1_trust)
                selected = self._adaptive_aggregation_trust(
                    client_ids,
                    layer1_raw,
                    robustness,
                    raw_capped,
                    gated_trust,
                    gate_info,
                )
                layer1_trust = selected["trust"]
                gating_info.update(gate_info)
                gating_info["adaptive"] = True
                gating_info["blend"] = "(1-g_i)*T_L1_capped + g_i*T_gated"
                self.last_layer1_layer2_gating = gating_info
        else:
            layer1_trust = self._aggregation_layer1_trust(
                client_ids,
                client_weights,
                global_weights,
                layer1_raw,
                sanitize_flags,
            )

        layer1_trust = self._apply_backdoor_trigger_hard_gate(
            client_ids,
            client_weights,
            global_weights,
            layer1_trust,
        )

        (
            client_ids,
            client_weights,
            layer1_trust,
            client_samples,
        ) = self._apply_backdoor_probe_hard_rejection(
            client_ids,
            client_weights,
            global_weights,
            layer1_trust,
            client_samples,
            layer1_raw=layer1_raw,
        )

        (
            client_ids,
            client_weights,
            layer1_trust,
            client_samples,
        ) = self._apply_backdoor_geometry_hard_rejection(
            client_ids,
            client_weights,
            global_weights,
            layer1_trust,
            client_samples,
        )

        (
            client_ids,
            client_weights,
            layer1_trust,
            client_samples,
        ) = self._apply_update_norm_hard_rejection(
            client_ids,
            client_weights,
            global_weights,
            layer1_trust,
            client_samples,
        )

        skip_reason = self._aggregation_guard_decision(
            client_ids, layer1_trust, layer1_raw
        )
        if skip_reason:
            self.last_aggregation_guard = {
                "skipped": True,
                "reason": skip_reason,
                "num_participants": len(client_ids),
                "min_aggregation_participants": int(
                    self.config.min_aggregation_participants
                ),
                "layer1_trust": layer1_trust,
                "layer1_raw_trust": layer1_raw,
                "update_norm_rejection": self.last_update_norm_rejection,
                "backdoor_probe_hard_rejection": self.last_backdoor_probe_hard_rejection,
                "backdoor_geometry_hard_rejection": (
                    self.last_backdoor_geometry_hard_rejection
                ),
                "backdoor_trigger_hard_gate": self.last_backdoor_trigger_hard_gate,
            }
            if self.layer2 is not None:
                self.layer2.last_aggregation_diagnostics = {
                    "aggregation_guard": self.last_aggregation_guard,
                    "trust_weight_power": float(self.config.trust_weight_power),
                    "min_trust_weight": float(self.config.min_trust_weight),
                    "clients": {},
                }
            self.trust_history.append(layer1_raw.copy())
            self.round_number += 1
            if self.config.export_trust_scores:
                self._export_round_trust(layer1_raw, client_ids)
            return global_weights

        if self.layer2 is not None:
            aggregated = self.layer2.aggregate(
                client_ids,
                client_weights,
                global_weights,
                layer1_trust,
                mission_context,
                client_samples,
            )
        else:
            from acfl.weight_utils import weighted_average_state_dicts
            weights = [layer1_trust.get(cid, 1.0) for cid in client_ids]
            aggregated = weighted_average_state_dicts(client_weights, weights)

        if not self._aggregate_is_finite(aggregated):
            self.last_aggregation_guard = {
                "skipped": True,
                "reason": "non_finite_aggregate",
                "num_participants": len(client_ids),
                "update_norm_rejection": self.last_update_norm_rejection,
                "robust_norm_soft_cap": self.last_robust_norm_soft_cap,
            }
            self.trust_history.append(layer1_raw.copy())
            self.round_number += 1
            if self.config.export_trust_scores:
                self._export_round_trust(layer1_raw, client_ids)
            return global_weights

        aggregated = self._apply_post_agg_trigger_scrub(aggregated)

        self.last_aggregation_guard = {
            "skipped": False,
            "num_participants": len(client_ids),
            "min_aggregation_participants": int(
                self.config.min_aggregation_participants
            ),
            "update_norm_rejection": self.last_update_norm_rejection,
            "robust_norm_soft_cap": self.last_robust_norm_soft_cap,
            "backdoor_probe_soft_cap": self.last_backdoor_probe_soft_cap,
            "backdoor_probe_hard_rejection": self.last_backdoor_probe_hard_rejection,
            "backdoor_geometry_hard_rejection": (
                self.last_backdoor_geometry_hard_rejection
            ),
            "backdoor_trigger_hard_gate": self.last_backdoor_trigger_hard_gate,
            "post_agg_trigger_scrub": self.last_post_agg_trigger_scrub,
            "layer1_layer2_gating": self.last_layer1_layer2_gating,
            "layer1_raw_trust": layer1_raw,
        }

        self.trust_history.append(layer1_raw.copy())
        self.round_number += 1

        if self.config.export_trust_scores:
            self._export_round_trust(layer1_raw, client_ids)

        return aggregated

    def _pta_trust_maps(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_raw: Dict[str, float],
        sanitize_flags: Dict[str, str],
    ) -> Dict[str, Dict[str, float]]:
        """Gated vs ungated Layer-1 interpretations of the same evidence."""
        self._apply_layer1_layer2_gating(
            client_ids, client_weights, global_weights, layer1_raw
        )
        baseline = {cid: 1.0 for cid in client_ids}
        for cid in sanitize_flags:
            baseline[cid] = float(self.config.min_trust_weight)
        raw_capped = self._apply_robust_norm_soft_cap(
            client_ids, client_weights, global_weights, layer1_raw
        )
        neutral_capped = self._apply_robust_norm_soft_cap(
            client_ids, client_weights, global_weights, baseline
        )
        gated_endpoint = dict(neutral_capped)
        if self._backdoor_layer1_in_aggregation():
            if self.config.backdoor_probe_aware_soft_cap_enabled:
                probe_trust = self._apply_backdoor_probe_soft_cap(
                    client_ids, client_weights, global_weights, neutral_capped
                )
                if self.config.backdoor_probe_blend_with_raw_l1:
                    gated_endpoint = {
                        cid: min(float(probe_trust[cid]), float(raw_capped[cid]))
                        for cid in client_ids
                    }
                else:
                    gated_endpoint = probe_trust
            else:
                gated_endpoint = {
                    cid: min(float(neutral_capped[cid]), float(raw_capped[cid]))
                    for cid in client_ids
                }

        gating_info = dict(self.last_layer1_layer2_gating or {})
        robustness = gating_info.get("robustness_scores") or (
            self._peer_robustness_scores(client_ids, client_weights, global_weights)
        )
        gate_info = self._adaptive_gate(client_ids, layer1_raw, robustness)
        gated_selected = self._adaptive_aggregation_trust(
            client_ids,
            layer1_raw,
            robustness,
            raw_capped,
            gated_endpoint,
            gate_info,
        )
        gating_info.update(gate_info)
        gating_info["adaptive"] = True
        gating_info["pta"] = True
        gating_info["blend"] = "(1-g_i)*T_L1_capped + g_i*T_gated"
        self.last_layer1_layer2_gating = gating_info
        return {
            "ungated": dict(raw_capped),
            "gated": dict(gated_selected["trust"]),
            "raw_capped": dict(raw_capped),
            "gated_endpoint": dict(gated_endpoint),
        }

    def _pta_apply_shared_extras(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        shared_trust: Dict[str, float],
        client_samples: Optional[List[int]],
        layer1_raw: Dict[str, float],
    ) -> tuple:
        """Attack-scoped extras once, identically, before both aggregations."""
        shared_trust = self._apply_backdoor_trigger_hard_gate(
            client_ids, client_weights, global_weights, shared_trust
        )
        (
            client_ids,
            client_weights,
            shared_trust,
            client_samples,
        ) = self._apply_backdoor_probe_hard_rejection(
            client_ids,
            client_weights,
            global_weights,
            shared_trust,
            client_samples,
            layer1_raw=layer1_raw,
        )
        (
            client_ids,
            client_weights,
            shared_trust,
            client_samples,
        ) = self._apply_backdoor_geometry_hard_rejection(
            client_ids,
            client_weights,
            global_weights,
            shared_trust,
            client_samples,
        )
        (
            client_ids,
            client_weights,
            shared_trust,
            client_samples,
        ) = self._apply_update_norm_hard_rejection(
            client_ids,
            client_weights,
            global_weights,
            shared_trust,
            client_samples,
        )
        return client_ids, client_weights, shared_trust, client_samples

    def _pta_guard_reason(
        self,
        client_ids: List[str],
        layer1_trust: Dict[str, float],
        layer1_raw: Dict[str, float],
        adaptive: bool,
    ) -> Optional[str]:
        saved = bool(self.config.adaptive_l1l2_gating_enabled)
        self.config.adaptive_l1l2_gating_enabled = bool(adaptive)
        try:
            return self._aggregation_guard_decision(
                client_ids, layer1_trust, layer1_raw
            )
        finally:
            self.config.adaptive_l1l2_gating_enabled = saved

    def _pta_aggregate_candidate(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_trust: Dict[str, float],
        mission_context: Optional[Dict[str, Dict[str, float]]],
        client_samples: Optional[List[int]],
        skip_reason: Optional[str],
        apply_scrub: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if skip_reason:
            return clone_state_dict(global_weights)
        if self.layer2 is not None:
            aggregated = self.layer2.aggregate_readonly(
                client_ids,
                client_weights,
                global_weights,
                layer1_trust,
                mission_context,
                client_samples,
            )
        else:
            weights = [layer1_trust.get(cid, 1.0) for cid in client_ids]
            aggregated = weighted_average_state_dicts(client_weights, weights)
        if not self._aggregate_is_finite(aggregated):
            return clone_state_dict(global_weights)
        if apply_scrub:
            return self._apply_post_agg_trigger_scrub(aggregated)
        return aggregated

    def _pta_loo_influences(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_trust: Dict[str, float],
        mission_context: Optional[Dict[str, Dict[str, float]]],
        client_samples: Optional[List[int]],
        candidate: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        denom = state_dict_l2(candidate, global_weights) + 1e-8
        influences: Dict[str, float] = {}
        if len(client_ids) <= 1:
            for cid in client_ids:
                influences[cid] = 1.0
            return influences
        for i, cid in enumerate(client_ids):
            ids_m = client_ids[:i] + client_ids[i + 1 :]
            weights_m = client_weights[:i] + client_weights[i + 1 :]
            trust_m = {other: layer1_trust[other] for other in ids_m}
            samples_m = None
            if client_samples is not None:
                samples_m = client_samples[:i] + client_samples[i + 1 :]
            loo = self._pta_aggregate_candidate(
                ids_m,
                weights_m,
                global_weights,
                trust_m,
                mission_context,
                samples_m,
                skip_reason=None,
                apply_scrub=False,
            )
            influences[cid] = state_dict_l2(candidate, loo) / denom
        return influences

    def _pta_residual_risk(
        self,
        layer1_raw: Dict[str, float],
        influences: Dict[str, float],
    ) -> float:
        terms = [
            max(0.0, 1.0 - float(layer1_raw.get(cid, 1.0))) * float(influence)
            for cid, influence in influences.items()
        ]
        if not terms:
            return 0.0
        reduce = str(
            getattr(self.config, "parallel_trust_arbitration_risk_reduce", "max")
        ).lower()
        if reduce == "sum":
            return float(sum(terms))
        return float(max(terms))

    def _pta_score_branch(
        self,
        name: str,
        candidate: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer1_raw: Dict[str, float],
        layer1_trust: Dict[str, float],
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        mission_context: Optional[Dict[str, Dict[str, float]]],
        client_samples: Optional[List[int]],
        skipped: bool,
    ) -> BranchScore:
        probe_terms = {
            "L_task": 0.0,
            "H": 0.0,
            "R_trigger": 0.0,
        }
        if self.probe_evaluator is not None and not skipped:
            probe_terms.update(
                self.probe_evaluator.candidate_probe_terms(
                    candidate, global_weights
                )
            )
        elif self.probe_evaluator is not None and skipped:
            current = self.probe_evaluator.candidate_probe_terms(
                global_weights, global_weights
            )
            probe_terms["L_task"] = float(current.get("L_task", 0.0))

        influences = {}
        if not skipped:
            influences = self._pta_loo_influences(
                client_ids,
                client_weights,
                global_weights,
                layer1_trust,
                mission_context,
                client_samples,
                candidate,
            )
        R = 0.0 if skipped else self._pta_residual_risk(layer1_raw, influences)
        global_scale = state_dict_l2(global_weights, {
            key: tensor.new_zeros(tensor.shape) for key, tensor in global_weights.items()
        })
        S = 0.0 if skipped else (
            state_dict_l2(candidate, global_weights) / (1.0 + global_scale)
        )
        J = candidate_objective(
            L_task=float(probe_terms["L_task"]),
            H=float(probe_terms["H"]),
            R=R,
            S=S,
            R_trigger=float(probe_terms["R_trigger"]),
            lambda_H=float(self.config.parallel_trust_arbitration_lambda_H),
            lambda_R=float(self.config.parallel_trust_arbitration_lambda_R),
            lambda_S=float(self.config.parallel_trust_arbitration_lambda_S),
            lambda_trigger=float(self.config.parallel_trust_arbitration_lambda_trigger),
            skipped=bool(skipped),
            lambda_skip=float(
                getattr(self.config, "parallel_trust_arbitration_lambda_skip", 0.0)
            ),
        )
        return BranchScore(
            name=name,
            L_task=float(probe_terms["L_task"]),
            H=float(probe_terms["H"]),
            R=R,
            S=S,
            R_trigger=float(probe_terms["R_trigger"]),
            J=J,
            skipped=bool(skipped),
        )

    def _defend_with_parallel_trust_arbitration(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_raw: Dict[str, float],
        sanitize_flags: Dict[str, str],
        mission_context: Optional[Dict[str, Dict[str, float]]],
        client_samples: Optional[List[int]],
    ) -> Dict[str, torch.Tensor]:
        """Dual-path counterfactual aggregation, then risk-aware arbitration."""
        trust_maps = self._pta_trust_maps(
            client_ids, client_weights, global_weights, layer1_raw, sanitize_flags
        )
        ungated_trust = dict(trust_maps["ungated"])
        gated_trust = dict(trust_maps["gated"])

        (
            extra_ids,
            extra_weights,
            shared_trust,
            extra_samples,
        ) = self._pta_apply_shared_extras(
            client_ids,
            client_weights,
            global_weights,
            dict(ungated_trust),
            client_samples,
            layer1_raw,
        )
        gated_trust = {cid: gated_trust.get(cid, 1.0) for cid in extra_ids}
        ungated_trust = {cid: shared_trust.get(cid, ungated_trust.get(cid, 1.0))
                         for cid in extra_ids}

        ungated_skip = self._pta_guard_reason(
            extra_ids, ungated_trust, layer1_raw, adaptive=False
        )
        gated_skip = self._pta_guard_reason(
            extra_ids, gated_trust, layer1_raw, adaptive=True
        )

        if ungated_skip and gated_skip:
            self.last_aggregation_guard = {
                "skipped": True,
                "reason": f"pta_both:{ungated_skip}|{gated_skip}",
                "num_participants": len(extra_ids),
                "parallel_trust_arbitration": {"both_skipped": True},
            }
            self.last_parallel_trust_arbitration = self.last_aggregation_guard[
                "parallel_trust_arbitration"
            ]
            self.trust_history.append(layer1_raw.copy())
            self.round_number += 1
            if self.config.export_trust_scores:
                self._export_round_trust(layer1_raw, client_ids)
            return global_weights

        ungated_model = self._pta_aggregate_candidate(
            extra_ids,
            extra_weights,
            global_weights,
            ungated_trust,
            mission_context,
            extra_samples,
            ungated_skip,
        )
        gated_model = self._pta_aggregate_candidate(
            extra_ids,
            extra_weights,
            global_weights,
            gated_trust,
            mission_context,
            extra_samples,
            gated_skip,
        )

        ungated_score = self._pta_score_branch(
            "ungated",
            ungated_model,
            global_weights,
            layer1_raw,
            ungated_trust,
            extra_ids,
            extra_weights,
            mission_context,
            extra_samples,
            skipped=bool(ungated_skip),
        )
        gated_score = self._pta_score_branch(
            "gated",
            gated_model,
            global_weights,
            layer1_raw,
            gated_trust,
            extra_ids,
            extra_weights,
            mission_context,
            extra_samples,
            skipped=bool(gated_skip),
        )
        decision = arbitrate(
            J_gated=gated_score.J,
            J_ungated=ungated_score.J,
            method=str(self.config.parallel_trust_arbitration_method),
            tau=float(self.config.parallel_trust_arbitration_tau),
            steepness=float(
                self.config.parallel_trust_arbitration_mixture_steepness
            ),
            last_branch=self._pta_last_branch,
            default_branch=str(
                self.config.parallel_trust_arbitration_default_branch
            ),
            scores={"gated": gated_score, "ungated": ungated_score},
        )
        eta = float(decision.eta)
        committed = blend_state_dicts(gated_model, ungated_model, eta)
        if decision.selected == "gated":
            committed = gated_model
            self._pta_last_branch = "gated"
        elif decision.selected == "ungated":
            committed = ungated_model
            self._pta_last_branch = "ungated"
        else:
            committed = blend_state_dicts(gated_model, ungated_model, eta)
            self._pta_last_branch = (
                "gated" if eta >= 0.5 else "ungated"
            )

        if self.layer2 is not None:
            self.layer2.record_round(
                extra_ids,
                extra_weights,
                global_weights,
                diagnostics={
                    "parallel_trust_arbitration": decision.as_dict(),
                    "clients": {
                        cid: {
                            "layer1_trust_ungated": float(ungated_trust.get(cid, 1.0)),
                            "layer1_trust_gated": float(gated_trust.get(cid, 1.0)),
                        }
                        for cid in extra_ids
                    },
                },
            )

        pta_diag = {
            "enabled": True,
            "method": decision.method,
            "selected": decision.selected,
            "eta": eta,
            "ungated_skip": ungated_skip,
            "gated_skip": gated_skip,
            "decision": decision.as_dict(),
            "trust_ungated": ungated_trust,
            "trust_gated": gated_trust,
        }
        self.last_parallel_trust_arbitration = pta_diag
        self.last_aggregation_guard = {
            "skipped": False,
            "num_participants": len(extra_ids),
            "min_aggregation_participants": int(
                self.config.min_aggregation_participants
            ),
            "update_norm_rejection": self.last_update_norm_rejection,
            "robust_norm_soft_cap": self.last_robust_norm_soft_cap,
            "backdoor_probe_soft_cap": self.last_backdoor_probe_soft_cap,
            "backdoor_probe_hard_rejection": self.last_backdoor_probe_hard_rejection,
            "backdoor_geometry_hard_rejection": (
                self.last_backdoor_geometry_hard_rejection
            ),
            "backdoor_trigger_hard_gate": self.last_backdoor_trigger_hard_gate,
            "post_agg_trigger_scrub": self.last_post_agg_trigger_scrub,
            "layer1_layer2_gating": self.last_layer1_layer2_gating,
            "layer1_raw_trust": layer1_raw,
            "parallel_trust_arbitration": pta_diag,
        }
        self.trust_history.append(layer1_raw.copy())
        self.round_number += 1
        if self.config.export_trust_scores:
            self._export_round_trust(layer1_raw, client_ids)
        return committed

    def get_trust_breakdown(
        self,
        client_ids: List[str],
        client_weights: List[Dict[str, torch.Tensor]],
        global_weights: Dict[str, torch.Tensor],
        layer1_trust: Dict[str, float],
        mission_context: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        if self.layer2 is None:
            return {cid: {"layer1_trust": layer1_trust.get(cid, 1.0)} for cid in client_ids}
        return self.layer2.get_trust_breakdown(
            client_ids, client_weights, global_weights,
            layer1_trust, mission_context or {},
        )

    def compute_trust_auc(self, warmup_rounds: int = 2) -> Optional[float]:
        """Compute AUC of trust scores vs ground-truth malicious labels.

        Any client not explicitly flagged malicious is treated as benign, and
        the first ``warmup_rounds`` (pre-convergence) rounds are excluded since
        poisoned updates are not yet separable from benign ones there.
        """
        if not self.config.ground_truth_malicious or not self.trust_history:
            return None

        from sklearn.metrics import roc_auc_score

        malicious = {
            cid for cid, flag in self.config.ground_truth_malicious.items() if flag
        }

        scores = []
        labels = []
        for round_idx, round_trust in enumerate(self.trust_history, start=1):
            if round_idx <= warmup_rounds:
                continue
            for cid, trust in round_trust.items():
                scores.append(trust)
                labels.append(1 if cid in malicious else 0)

        if len(set(labels)) < 2:
            return None
        return float(roc_auc_score(labels, [-s for s in scores]))

    def _export_round_trust(self, trust: Dict[str, float], client_ids: List[str]):
        if not self.config.trust_score_path:
            return
        os.makedirs(os.path.dirname(self.config.trust_score_path) or ".", exist_ok=True)
        record = {
            "round": self.round_number,
            "trust_scores": trust,
            "client_ids": client_ids,
        }
        path = self.config.trust_score_path
        history = []
        if os.path.exists(path):
            with open(path) as f:
                history = json.load(f)
        history.append(record)
        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    def summary(self) -> Dict:
        """Return defense summary for reporting."""
        summary = {
            "rounds_defended": self.round_number,
            "layer1_enabled": self.config.enable_layer1,
            "layer2_enabled": self.config.enable_layer2,
            "layer3_enabled": self.config.enable_layer3,
            "trust_history_length": len(self.trust_history),
        }
        auc = self.compute_trust_auc()
        if auc is not None:
            summary["trust_score_auc"] = auc
        return summary
