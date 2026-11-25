"""Tests for Layer 3 packet-classification spoofing detection."""
import os
import sys

import pytest
import torch
import torch.nn as nn

ACFL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ACFL_ROOT not in sys.path:
    sys.path.insert(0, ACFL_ROOT)

from acfl.layer3_detection import SpoofingDetector


class _BinaryCNN(nn.Module):
    def __init__(self, attack_logit: float = 0.0):
        super().__init__()
        self.attack_logit = nn.Parameter(torch.tensor(attack_logit))

    def forward(self, x):
        # Zeros vs ones shift logits so probe benign/attack batches differ.
        input_bias = (x.mean(dim=(1, 2)) - 0.5) * 6.0
        logit = self.attack_logit + input_bias
        return logit.reshape(x.shape[0], 1)

    def get_weights(self):
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights)


def _probe_data(benign_logit: float, attack_logit: float):
    benign_x = torch.zeros(4, 10, 5)
    attack_x = torch.ones(4, 10, 5)
    return {
        "benign": (benign_x, torch.zeros(4)),
        "mavlink_waypoint": (attack_x, torch.ones(4)),
        "gnss_spoofing": (attack_x, torch.ones(4)),
    }


class TestLayer3Classification:
    def test_attack_prob_not_symmetric_around_half(self):
        """P(attack) should rank attack-like outputs above benign ones."""
        benign_model = _BinaryCNN(attack_logit=-4.0)
        attack_model = _BinaryCNN(attack_logit=4.0)

        det_benign = SpoofingDetector(benign_model, learning_mode="classification")
        det_attack = SpoofingDetector(attack_model, learning_mode="classification")

        benign_window = torch.zeros(1, 10, 5)
        attack_window = torch.ones(1, 10, 5)
        score_benign = det_benign.detect(benign_window)["model_score"]
        score_attack = det_attack.detect(attack_window)["model_score"]

        assert score_benign < 0.1
        assert score_attack > 0.9

    def test_probe_calibration_anchors_scores(self):
        model = _BinaryCNN(attack_logit=2.0)
        probes = _probe_data(benign_logit=-3.0, attack_logit=3.0)
        # Override probe responses by using separate models is hard; use fixed logits model
        # and distinct probe tensors — calibration uses model on probe batches.
        model.load_state_dict(_BinaryCNN(-3.0).state_dict())
        detector = SpoofingDetector(
            model,
            learning_mode="classification",
            probe_data=probes,
            probe_blend_weight=0.5,
        )
        assert detector._benign_attack_prob_ref is not None
        assert detector._attack_attack_prob_ref is not None
        assert detector._attack_attack_prob_ref > detector._benign_attack_prob_ref

        model.load_state_dict(_BinaryCNN(3.0).state_dict())
        live = detector.detect(torch.ones(1, 10, 5))
        assert live["anomaly_score"] > 0.5
        assert live["is_adversarial"]

    def test_telemetry_context_boosts_anomaly(self):
        model = _BinaryCNN(attack_logit=0.0)
        detector = SpoofingDetector(model, learning_mode="classification", threshold=0.5)
        window = torch.randn(1, 10, 5)
        clean = detector.detect(window)
        inconsistent = detector.detect(
            window,
            telemetry_context={"gnss_imu_consistency": 0.1, "mavlink_state_consistency": 0.1},
        )
        assert inconsistent["anomaly_score"] >= clean["anomaly_score"]
