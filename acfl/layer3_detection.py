"""
Layer 3: Poisoning-Resilient FL Detector for Control Signal Spoofing.

Uses the globally defended model to score live telemetry for network-plane
attacks.  Classification (packet) models use predicted P(attack) with optional
probe-set calibration; regression (GPS/IMU) models retain cross-modal
command-vs-state inconsistency scoring.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Attack probe scenarios used to calibrate packet classifiers at inference.
_ATTACK_PROBE_SCENARIOS = (
    "gnss_spoofing",
    "adsb_injection",
    "mavlink_waypoint",
    "jamming",
)


class SpoofingDetector:
    """
    Operational anomaly detector for adversarial control / network-plane attacks.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.5,
        cross_modal_weight: float = 0.7,
        probe_blend_weight: float = 0.35,
        learning_mode: str = "classification",
        probe_data: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.threshold = threshold
        self.cross_modal_weight = cross_modal_weight
        self.probe_blend_weight = float(np.clip(probe_blend_weight, 0.0, 1.0))
        self.learning_mode = learning_mode
        self.probe_data = probe_data or {}
        self.device = device or torch.device("cpu")
        self._benign_attack_prob_ref: Optional[float] = None
        self._attack_attack_prob_ref: Optional[float] = None
        if self.learning_mode == "classification" and self.probe_data:
            self._calibrate_probe_references()

    def _calibrate_probe_references(self) -> None:
        """Anchor live P(attack) scores to benign vs attack probe baselines."""
        benign_ref = self._mean_attack_prob_on_probe("benign")
        attack_probs = [
            self._mean_attack_prob_on_probe(name)
            for name in _ATTACK_PROBE_SCENARIOS
            if name in self.probe_data
        ]
        if benign_ref is None or not attack_probs:
            self._benign_attack_prob_ref = None
            self._attack_attack_prob_ref = None
            return
        self._benign_attack_prob_ref = benign_ref
        self._attack_attack_prob_ref = float(np.mean(attack_probs))

    def _mean_attack_prob_on_probe(self, scenario: str) -> Optional[float]:
        if scenario not in self.probe_data:
            return None
        inputs, _ = self.probe_data[scenario]
        if inputs.numel() == 0:
            return None
        self.model.eval()
        with torch.no_grad():
            batch = inputs.to(self.device)
            outputs = self.model(batch)
            probs = torch.sigmoid(outputs).reshape(-1)
            return float(probs.mean().item())

    def detect(
        self,
        sensor_window: torch.Tensor,
        command_state: Optional[torch.Tensor] = None,
        telemetry_context: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Run spoofing detection on a sensor window.

        Args:
            sensor_window: (1, seq_len, features) input window
            command_state: Optional MAVLink command features (regression path)
            telemetry_context: Optional mission headers, e.g.
                gnss_imu_consistency in [0, 1]
        """
        self.model.eval()
        sensor_window = sensor_window.to(self.device)

        with torch.no_grad():
            model_output = self.model(sensor_window)

            if self.learning_mode == "classification":
                model_score = self._classification_attack_prob(model_output)
                probe_score = self._probe_calibrated_score(model_score)
                telemetry_score = self._telemetry_inconsistency(telemetry_context)
                combined = self._combine_classification_scores(
                    model_score, probe_score, telemetry_score
                )
                cross_modal_score = telemetry_score
            else:
                model_score = self._regression_anomaly_score(
                    model_output, sensor_window
                )
                cross_modal_score = 0.0
                if command_state is not None:
                    cross_modal_score = self._cross_modal_inconsistency(
                        sensor_window, command_state.to(self.device)
                    )
                combined = (
                    (1 - self.cross_modal_weight) * model_score
                    + self.cross_modal_weight * cross_modal_score
                )

        is_adversarial = combined >= self.threshold
        return {
            "anomaly_score": float(combined),
            "model_score": float(model_score),
            "probe_score": float(probe_score) if self.learning_mode == "classification" else 0.0,
            "cross_modal_score": float(cross_modal_score),
            "is_adversarial": bool(is_adversarial),
            "recommended_action": "safe_hold" if is_adversarial else "continue",
            "benign_probe_ref": self._benign_attack_prob_ref,
            "attack_probe_ref": self._attack_attack_prob_ref,
        }

    @staticmethod
    def _classification_attack_prob(output: torch.Tensor) -> float:
        """P(attack) — higher means more likely network-plane attack traffic."""
        prob = torch.sigmoid(output).squeeze().item()
        return float(np.clip(prob, 0.0, 1.0))

    def _probe_calibrated_score(self, live_attack_prob: float) -> float:
        """Map live P(attack) onto [0, 1] using probe anchors."""
        if (
            self._benign_attack_prob_ref is None
            or self._attack_attack_prob_ref is None
        ):
            return live_attack_prob
        span = self._attack_attack_prob_ref - self._benign_attack_prob_ref
        if span <= 1e-6:
            return live_attack_prob
        calibrated = (live_attack_prob - self._benign_attack_prob_ref) / span
        return float(np.clip(calibrated, 0.0, 1.0))

    def _combine_classification_scores(
        self,
        model_score: float,
        probe_score: float,
        telemetry_score: float,
    ) -> float:
        if self._benign_attack_prob_ref is not None:
            w = self.probe_blend_weight
            base = (1.0 - w) * model_score + w * probe_score
        else:
            base = model_score
        if telemetry_score > 0.0:
            t_w = min(0.25, self.probe_blend_weight)
            return float(np.clip((1.0 - t_w) * base + t_w * telemetry_score, 0.0, 1.0))
        return float(np.clip(base, 0.0, 1.0))

    @staticmethod
    def _telemetry_inconsistency(
        telemetry_context: Optional[Dict[str, float]],
    ) -> float:
        """Low GNSS/MAVLink consistency from client headers raises anomaly score."""
        if not telemetry_context:
            return 0.0
        gnss = telemetry_context.get("gnss_imu_consistency")
        mav = telemetry_context.get("mavlink_state_consistency")
        scores = []
        if gnss is not None:
            scores.append(float(np.clip(1.0 - float(gnss), 0.0, 1.0)))
        if mav is not None:
            scores.append(float(np.clip(1.0 - float(mav), 0.0, 1.0)))
        if not scores:
            return 0.0
        return float(np.mean(scores))

    def _regression_anomaly_score(
        self,
        output: torch.Tensor,
        input_window: torch.Tensor,
    ) -> float:
        if output.shape == input_window.shape[-1:]:
            target = input_window[:, -1, : output.shape[-1]]
            error = nn.functional.mse_loss(output, target).item()
            return float(np.tanh(error))
        prob = torch.sigmoid(output).squeeze().item()
        return float(abs(prob - 0.5) * 2)

    def _cross_modal_inconsistency(
        self,
        sensor_window: torch.Tensor,
        command_state: torch.Tensor,
    ) -> float:
        last_state = sensor_window[0, -1, :]
        n_features = last_state.shape[0]

        if n_features < 6 or command_state.numel() < 2:
            return 0.0

        gps_spd = last_state[3].item()
        gps_course = last_state[4].item()
        gps_vz = last_state[5].item()

        cmd_heading = (
            command_state[0, 0].item()
            if command_state.dim() > 1
            else command_state[0].item()
        )
        cmd_speed = (
            command_state[0, 1].item()
            if command_state.shape[-1] > 1
            else 0.0
        )

        heading_diff = abs(gps_course - cmd_heading)
        heading_diff = min(heading_diff, 360 - heading_diff) / 180.0

        speed_diff = abs(gps_spd - cmd_speed) / (
            max(abs(cmd_speed), abs(gps_spd), 1.0)
        )

        inconsistency = (
            0.5 * heading_diff + 0.3 * speed_diff + 0.2 * abs(gps_vz) / 10.0
        )
        return float(np.clip(inconsistency, 0.0, 1.0))

    def batch_detect(
        self,
        windows: torch.Tensor,
        command_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, list]:
        results = {
            "anomaly_scores": [],
            "is_adversarial": [],
            "recommended_actions": [],
        }
        for i in range(windows.shape[0]):
            cmd = command_states[i : i + 1] if command_states is not None else None
            det = self.detect(windows[i : i + 1], cmd)
            results["anomaly_scores"].append(det["anomaly_score"])
            results["is_adversarial"].append(det["is_adversarial"])
            results["recommended_actions"].append(det["recommended_action"])
        return results
