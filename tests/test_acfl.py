"""Unit tests for ACFL defense framework."""
import sys
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import pytest

ACFL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ACFL_ROOT)

from acfl.config import ACFLConfig
from acfl.defense import ACFLDefense
from acfl.layer1_validation import UpdateValidationModule
from acfl.layer2_aggregation import TrustWeightedAggregator
from acfl.layer3_detection import SpoofingDetector
from acfl.weight_utils import (
    compute_client_update_norm_stats,
    compute_update_delta,
    coordinate_wise_median,
    flatten_state_dict,
    norm_ratio_threshold,
    weighted_average_state_dicts,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x[:, -1, :])

    def get_weights(self):
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights)


def _make_state_dict(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = TinyModel()
    return model.state_dict()


def _make_state_dicts(n: int) -> list:
    return [_make_state_dict(i) for i in range(n)]


class TestWeightUtils:
    def test_flatten_and_delta(self):
        global_w = _make_state_dict(0)
        client_w = _make_state_dict(1)
        delta = compute_update_delta(client_w, global_w)
        assert delta.ndim == 1
        assert delta.size > 0

    def test_weighted_average(self):
        weights_list = _make_state_dicts(3)
        result = weighted_average_state_dicts(weights_list, [0.5, 0.3, 0.2])
        assert set(result.keys()) == set(weights_list[0].keys())

    def test_coordinate_median(self):
        weights_list = _make_state_dicts(5)
        median = coordinate_wise_median(weights_list)
        for key in median:
            assert median[key].shape == weights_list[0][key].shape

    def test_norm_ratio_threshold_two_clients(self):
        assert norm_ratio_threshold(2) == 1.5
        assert norm_ratio_threshold(3) == 2.0

    def test_update_norm_stats_peer_comparison_for_two_clients(self):
        global_w = _make_state_dict(0)
        benign_w = _make_state_dict(1)
        extreme_w = {
            key: benign_w[key] + torch.randn_like(benign_w[key]) * 100.0
            for key in benign_w
        }
        stats = compute_client_update_norm_stats([benign_w, extreme_w], global_w)
        assert stats[1]["norm_ratio"] > 10.0
        assert stats[0]["norm_ratio"] < 1.0


class TestLayer1Validation:
    def test_subspace_scoring_benign_clients(self):
        config = ACFLConfig()
        validator = UpdateValidationModule(config)
        global_w = _make_state_dict(0)
        client_weights = [_make_state_dict(i) for i in range(1, 4)]
        client_ids = ["c1", "c2", "c3"]

        scores = validator.score_updates(client_ids, client_weights, global_w)
        assert len(scores) == 3
        for cid in client_ids:
            assert 0.05 <= scores[cid] <= 1.0

    def test_poisoned_update_gets_lower_trust(self):
        config = ACFLConfig()
        validator = UpdateValidationModule(config)
        global_w = _make_state_dict(0)

        benign = [_make_state_dict(i) for i in range(1, 3)]
        poisoned = _make_state_dict(99)
        for key in poisoned:
            poisoned[key] = poisoned[key] * 100.0

        client_ids = ["benign_1", "benign_2", "poisoned"]
        client_weights = benign + [poisoned]

        scores_r1 = validator.score_updates(client_ids, client_weights, global_w)
        scores_r2 = validator.score_updates(client_ids, client_weights, global_w)

        assert scores_r1["poisoned"] <= scores_r1["benign_1"]

    def test_two_client_gradient_scaling_detected(self):
        """10x scaling is invisible at threshold 2.0 with two clients but not 1.5."""
        config = ACFLConfig()
        validator = UpdateValidationModule(config)
        global_w = _make_state_dict(0)
        benign_w = _make_state_dict(1)

        scaled_w = {}
        for key in benign_w:
            delta = benign_w[key] - global_w[key]
            scaled_w[key] = global_w[key] + 10.0 * delta

        client_ids = ["benign", "attacker"]
        scores = validator.score_updates(
            client_ids, [benign_w, scaled_w], global_w
        )
        assert scores["attacker"] < scores["benign"]
        assert scores["attacker"] < 1.0


class TestLayer2Aggregation:
    def test_robust_aggregation(self):
        config = ACFLConfig()
        aggregator = TrustWeightedAggregator(config)
        global_w = _make_state_dict(0)
        client_weights = [_make_state_dict(i) for i in range(1, 4)]
        client_ids = ["c1", "c2", "c3"]

        result = aggregator.aggregate(
            client_ids, client_weights, global_w,
            layer1_trust={"c1": 1.0, "c2": 1.0, "c3": 0.1},
        )
        assert set(result.keys()) == set(global_w.keys())

    def test_mission_context_affects_trust(self):
        config = ACFLConfig()
        aggregator = TrustWeightedAggregator(config)
        global_w = _make_state_dict(0)
        client_weights = _make_state_dicts(2)
        client_ids = ["good", "bad"]

        mission_ctx = {
            "good": {"gnss_imu_consistency": 0.95, "mavlink_state_consistency": 0.9},
            "bad": {"gnss_imu_consistency": 0.2, "mavlink_state_consistency": 0.1},
        }

        breakdown = aggregator.get_trust_breakdown(
            client_ids, client_weights, global_w,
            {"good": 1.0, "bad": 1.0}, mission_ctx,
        )
        assert breakdown["good"]["mission_context_score"] > breakdown["bad"]["mission_context_score"]

    def test_trust_power_diagnostics_downweight_low_trust_client(self):
        config = ACFLConfig(trust_weight_power=2.0)
        aggregator = TrustWeightedAggregator(config)
        global_w = _make_state_dict(0)
        client_weights = [copy.deepcopy(global_w), copy.deepcopy(global_w)]
        client_ids = ["good", "bad"]

        aggregator.aggregate(
            client_ids,
            client_weights,
            global_w,
            layer1_trust={"good": 1.0, "bad": 0.2},
            client_samples=[10, 10],
        )

        diag = aggregator.last_aggregation_diagnostics["clients"]
        assert diag["good"]["aggregation_weight_normalized"] > diag["bad"]["aggregation_weight_normalized"]
        assert diag["bad"]["aggregation_weight_raw"] <= diag["bad"]["combined_trust"]
        assert aggregator.last_aggregation_diagnostics["trust_weight_power"] == 2.0

    def test_pairwise_inversion_guard_downweights_suspicious_client(self):
        config = ACFLConfig(
            trust_weight_power=1.25,
            min_trust_weight=0.05,
            pairwise_inversion_guard_enabled=True,
        )
        aggregator = TrustWeightedAggregator(config)
        global_w = _make_state_dict(0)
        benign_w = _make_state_dict(1)
        attacker_w = _make_state_dict(2)

        # Force diagnostics path by patching robustness after aggregate internals.
        # Simulate round-9 pattern: trusted benign soft-filtered, attacker at 1.0.
        aggregator._robust_prefilter = lambda *args, **kwargs: (
            [benign_w, attacker_w],
            [0.607, 1.0],
        )

        aggregator.aggregate(
            ["client_1", "client_2"],
            [benign_w, attacker_w],
            global_w,
            layer1_trust={"client_1": 0.985, "client_2": 0.877},
            client_samples=[10, 10],
        )

        diag = aggregator.last_aggregation_diagnostics["clients"]
        assert "pairwise_inversion_penalty" in diag["client_2"]
        assert diag["client_2"]["pairwise_inversion_penalty"] == 0.4
        assert diag["client_2"]["aggregation_weight_normalized"] < 0.45
        assert (
            diag["client_1"]["aggregation_weight_normalized"]
            > diag["client_2"]["aggregation_weight_normalized"]
        )

    def test_pairwise_inversion_guard_preserves_balanced_round(self):
        config = ACFLConfig(pairwise_inversion_guard_enabled=True)
        aggregator = TrustWeightedAggregator(config)
        global_w = _make_state_dict(0)
        client_weights = [_make_state_dict(1), _make_state_dict(2)]

        aggregator._robust_prefilter = lambda *args, **kwargs: (
            client_weights,
            [1.0, 0.607],
        )

        aggregator.aggregate(
            ["client_1", "client_2"],
            client_weights,
            global_w,
            layer1_trust={"client_1": 0.979, "client_2": 0.831},
            client_samples=[10, 10],
        )

        diag = aggregator.last_aggregation_diagnostics["clients"]
        assert "pairwise_inversion_penalty" not in diag["client_1"]
        assert "pairwise_inversion_penalty" not in diag["client_2"]


class TestLayer3Detection:
    def test_spoofing_detection(self):
        model = TinyModel()
        detector = SpoofingDetector(model, threshold=0.3)

        sensor = torch.randn(1, 10, 4)
        command = torch.tensor([[90.0, 5.0]])

        result = detector.detect(sensor, command)
        assert "anomaly_score" in result
        assert "is_adversarial" in result
        assert result["recommended_action"] in ("safe_hold", "continue")


class TestACFLDefense:
    def test_full_pipeline(self):
        config = ACFLConfig(
            ground_truth_malicious={"client_3": True},
        )
        global_w = _make_state_dict(0)
        client_weights = [_make_state_dict(i) for i in range(1, 4)]
        client_ids = ["client_1", "client_2", "client_3"]

        defense = ACFLDefense(config=config, learning_mode="classification")

        for _ in range(3):
            aggregated = defense.defend_and_aggregate(
                client_ids, client_weights, global_w,
                client_losses={"client_1": 0.1, "client_2": 0.15, "client_3": 0.8},
                mission_context={
                    "client_1": {"gnss_imu_consistency": 0.9},
                    "client_2": {"gnss_imu_consistency": 0.85},
                    "client_3": {"gnss_imu_consistency": 0.2},
                },
            )
            global_w = aggregated

        summary = defense.summary()
        assert summary["rounds_defended"] == 3
        assert summary["layer1_enabled"]

    def test_ablation_layers(self):
        config = ACFLConfig(enable_layer1=False, enable_layer2=True)
        defense = ACFLDefense(config=config)
        global_w = _make_state_dict(0)
        client_weights = _make_state_dicts(2)
        result = defense.defend_and_aggregate(
            ["c1", "c2"], client_weights, global_w
        )
        assert result is not None

    def test_aggregation_guard_skips_insufficient_participants(self):
        config = ACFLConfig(
            enable_layer1=False,
            min_aggregation_participants=2,
        )
        defense = ACFLDefense(config=config)
        global_w = _make_state_dict(0)
        client_w = _make_state_dict(1)

        result = defense.defend_and_aggregate(
            ["client_2"], [client_w], global_w
        )

        assert defense.last_aggregation_guard["skipped"] is True
        assert "insufficient_participants:1<2" in defense.last_aggregation_guard["reason"]
        for key in global_w:
            assert torch.allclose(result[key], global_w[key])

    def test_aggregation_guard_skips_below_fl_quorum(self):
        config = ACFLConfig(
            enable_layer1=False,
            min_aggregation_participants=4,
        )
        defense = ACFLDefense(config=config)
        global_w = _make_state_dict(0)
        client_weights = [_make_state_dict(i) for i in range(3)]

        result = defense.defend_and_aggregate(
            ["client_1", "client_2", "client_3"], client_weights, global_w
        )

        assert defense.last_aggregation_guard["skipped"] is True
        assert "insufficient_participants:3<4" in defense.last_aggregation_guard["reason"]
        assert defense.last_aggregation_guard["min_aggregation_participants"] == 4
        for key in global_w:
            assert torch.allclose(result[key], global_w[key])

    def test_aggregation_guard_skips_single_low_trust_participant(self):
        config = ACFLConfig(
            enable_layer1=True,
            min_aggregation_participants=1,
            single_participant_trust_threshold=0.95,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        global_w = _make_state_dict(0)
        client_w = _make_state_dict(1)
        defense.layer1.score_updates = lambda *args, **kwargs: {"client_2": 0.5}

        result = defense.defend_and_aggregate(["client_2"], [client_w], global_w)

        assert defense.last_aggregation_guard["skipped"] is True
        assert "single_participant_low_trust" in defense.last_aggregation_guard["reason"]
        for key in global_w:
            assert torch.allclose(result[key], global_w[key])

    def test_aggregation_guard_allows_multi_client_round(self):
        config = ACFLConfig(enable_layer1=False, min_aggregation_participants=2)
        defense = ACFLDefense(config=config)
        global_w = _make_state_dict(0)
        client_weights = _make_state_dicts(2)

        result = defense.defend_and_aggregate(
            ["client_1", "client_2"], client_weights, global_w
        )

        assert defense.last_aggregation_guard["skipped"] is False
        assert set(result.keys()) == set(global_w.keys())

    def test_update_norm_hard_rejection_drops_extreme_update(self):
        config = ACFLConfig(
            enable_layer1=False,
            update_norm_hard_rejection_enabled=True,
            update_norm_hard_rejection_ratio=2.0,
            update_norm_hard_rejection_ratio_two_clients=2.0,
            update_norm_hard_rejection_deviation=4.0,
            min_aggregation_participants=1,
        )
        defense = ACFLDefense(config=config)
        global_w = _make_state_dict(0)
        benign_w = _make_state_dict(1)
        extreme_w = {}
        for key in benign_w:
            extreme_w[key] = benign_w[key] + torch.randn_like(benign_w[key]) * 100.0

        result = defense.defend_and_aggregate(
            ["benign", "attacker"],
            [benign_w, extreme_w],
            global_w,
        )

        rejection = defense.last_update_norm_rejection
        assert rejection["enabled"] is True
        assert len(rejection["rejected_clients"]) == 1
        assert rejection["rejected_clients"][0]["client_id"] == "attacker"
        assert rejection["accepted_clients"] == ["benign"]
        for key in global_w:
            assert torch.allclose(result[key], benign_w[key], atol=1e-5)

    def test_update_norm_hard_rejection_withholds_when_too_few_remain(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=False,
            robust_norm_soft_cap_enabled=False,
            update_norm_hard_rejection_enabled=True,
            update_norm_hard_rejection_ratio=1.05,
            update_norm_hard_rejection_ratio_two_clients=1.05,
            min_aggregation_participants=2,
        )
        defense = ACFLDefense(config=config)
        global_w = _make_state_dict(0)
        w1 = _make_state_dict(1)
        w2 = _make_state_dict(2)

        result = defense.defend_and_aggregate(
            ["client_1", "client_2"],
            [w1, w2],
            global_w,
        )

        rejection = defense.last_update_norm_rejection
        assert rejection["rejection_withheld_insufficient_participants"] is True
        assert rejection["rejected_clients"] == []
        assert set(rejection["accepted_clients"]) == {"client_1", "client_2"}
        assert defense.last_aggregation_guard["skipped"] is False
        # Soft path keeps both updates rather than holding the global model.
        assert set(result.keys()) == set(global_w.keys())


class TestRobustAggregationHardening:
    def test_layer2_two_client_outlier_gets_low_robustness(self):
        config = ACFLConfig(
            coordinate_deviation_threshold=2.5,
            norm_ratio_threshold=2.0,
            norm_ratio_threshold_two_clients=1.2,
        )
        aggregator = TrustWeightedAggregator(config)
        global_w = _make_state_dict(0)
        benign_w = _make_state_dict(1)
        attacker_w = {}
        for key in benign_w:
            attacker_w[key] = benign_w[key] + torch.randn_like(benign_w[key]) * 50.0

        scores = aggregator._robustness_scores_readonly(
            [benign_w, attacker_w], global_w
        )
        assert scores[0] > scores[1]
        assert scores[1] < 0.5

    def test_robust_norm_soft_cap_downweights_extreme_update(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=False,
            robust_norm_soft_cap_enabled=True,
            coordinate_deviation_threshold=2.5,
            norm_ratio_threshold_two_clients=1.2,
            update_norm_hard_rejection_enabled=False,
        )
        defense = ACFLDefense(config=config)
        global_w = _make_state_dict(0)
        benign_w = _make_state_dict(1)
        attacker_w = {}
        for key in benign_w:
            attacker_w[key] = benign_w[key] + torch.randn_like(benign_w[key]) * 50.0

        defense.defend_and_aggregate(
            ["benign", "attacker"],
            [benign_w, attacker_w],
            global_w,
        )

        caps = defense.last_robust_norm_soft_cap["caps"]
        assert caps["benign"] > caps["attacker"]

    def test_sanitize_non_finite_client_weights(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=False,
            robust_norm_soft_cap_enabled=False,
            update_norm_hard_rejection_enabled=False,
            min_trust_weight=0.05,
        )
        defense = ACFLDefense(config=config)
        global_w = _make_state_dict(0)
        benign_w = _make_state_dict(1)
        bad_w = copy.deepcopy(benign_w)
        key = next(k for k, t in bad_w.items() if t.is_floating_point())
        bad_w[key] = torch.full_like(bad_w[key], float("nan"))

        defense.defend_and_aggregate(
            ["benign", "bad"],
            [benign_w, bad_w],
            global_w,
        )

        assert defense.trust_history[-1]["bad"] == pytest.approx(0.05)

    def test_layer1_layer2_gating_defers_l1_to_layer2(self):
        config = ACFLConfig(
            enable_layer1=True,
            enable_layer2=True,
            layer1_layer2_gating_enabled=True,
            robust_norm_soft_cap_enabled=True,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        defense.layer1.score_updates = lambda *args, **kwargs: {
            "benign": 0.3,
            "attacker": 0.9,
        }
        global_w = _make_state_dict(0)
        benign_w = _make_state_dict(1)
        attacker_w = _make_state_dict(2)

        defense_no_l1 = ACFLDefense(
            config=ACFLConfig(
                enable_layer1=False,
                enable_layer2=True,
                robust_norm_soft_cap_enabled=True,
                update_norm_hard_rejection_enabled=False,
                min_aggregation_participants=1,
            ),
            learning_mode="classification",
        )

        defense.defend_and_aggregate(
            ["benign", "attacker"],
            [benign_w, attacker_w],
            global_w,
        )
        defense_no_l1.defend_and_aggregate(
            ["benign", "attacker"],
            [benign_w, attacker_w],
            global_w,
        )

        gating = defense.last_layer1_layer2_gating
        assert gating["enabled"] is True
        assert gating["defer_layer1_to_layer2"] is True
        assert defense.last_robust_norm_soft_cap.get("enabled") is True
        assert defense.trust_history[-1]["benign"] == pytest.approx(0.3)
        assert defense.trust_history[-1]["attacker"] == pytest.approx(0.9)

        full_diag = defense.layer2.last_aggregation_diagnostics["clients"]
        no_l1_diag = defense_no_l1.layer2.last_aggregation_diagnostics["clients"]
        for cid in ("benign", "attacker"):
            assert full_diag[cid]["layer1_trust"] == pytest.approx(
                no_l1_diag[cid]["layer1_trust"]
            )

    def test_adaptive_gate_opens_when_l1_conflicts_with_robustness(self):
        defense = ACFLDefense(
            config=ACFLConfig(adaptive_l1l2_gating_enabled=True),
            learning_mode="classification",
        )
        agree = defense._adaptive_gate(
            ["benign", "attacker"],
            {"benign": 0.95, "attacker": 0.10},
            {"benign": 0.95, "attacker": 0.12},
        )
        single = defense._adaptive_gate(
            ["benign", "attacker"],
            {"benign": 0.20, "attacker": 0.90},
            {"benign": 0.95, "attacker": 0.90},
        )
        cohort = defense._adaptive_gate(
            ["benign_a", "benign_b", "attacker"],
            {"benign_a": 0.20, "benign_b": 0.30, "attacker": 0.90},
            {"benign_a": 0.95, "benign_b": 0.95, "attacker": 0.90},
        )
        missed = defense._adaptive_gate(
            ["benign", "attacker"],
            {"benign": 1.0, "attacker": 0.89},
            {"benign": 1.0, "attacker": 0.25},
        )
        assert agree["g_t"] == pytest.approx(0.0)
        assert single["cohort_penalty"] is False
        assert all(value == pytest.approx(0.0) for value in single["g_i"].values())
        assert cohort["cohort_penalty"] is True
        assert cohort["g_i"]["benign_a"] == pytest.approx(1.0)
        assert cohort["g_i"]["benign_b"] == pytest.approx(1.0)
        assert cohort["g_i"]["attacker"] == pytest.approx(0.0)
        assert missed["g_i"]["attacker"] == pytest.approx(0.0)

    def test_closed_gate_keeps_norm_capped_layer1(self):
        defense = ACFLDefense(
            config=ACFLConfig(adaptive_l1l2_gating_enabled=True),
            learning_mode="classification",
        )
        attributed = defense._adaptive_gate(
            ["benign", "attacker"],
            {"benign": 0.95, "attacker": 0.20},
            {"benign": 0.95, "attacker": 0.90},
        )
        attributed_trust = defense._adaptive_aggregation_trust(
            ["benign", "attacker"],
            {"benign": 0.95, "attacker": 0.20},
            {"benign": 0.95, "attacker": 0.90},
            {"benign": 0.90, "attacker": 0.18},
            {"benign": 0.95, "attacker": 0.90},
            attributed,
        )
        assert attributed_trust["trust"]["benign"] == pytest.approx(0.90)
        assert attributed_trust["trust"]["attacker"] == pytest.approx(0.18)

        replacement = defense._adaptive_gate(
            ["benign", "attacker"],
            {"benign": 1.0, "attacker": 0.89},
            {"benign": 1.0, "attacker": 0.25},
        )
        replacement_trust = defense._adaptive_aggregation_trust(
            ["benign", "attacker"],
            {"benign": 1.0, "attacker": 0.89},
            {"benign": 1.0, "attacker": 0.25},
            {"benign": 1.0, "attacker": 0.22},
            {"benign": 1.0, "attacker": 0.25},
            replacement,
        )
        assert replacement_trust["trust"]["attacker"] == pytest.approx(0.22)
        assert replacement_trust["trust"]["benign"] == pytest.approx(1.0)

        cohort = defense._adaptive_gate(
            ["benign_a", "benign_b", "attacker"],
            {"benign_a": 0.20, "benign_b": 0.30, "attacker": 0.90},
            {"benign_a": 0.95, "benign_b": 0.95, "attacker": 0.90},
        )
        cohort_trust = defense._adaptive_aggregation_trust(
            ["benign_a", "benign_b", "attacker"],
            {"benign_a": 0.20, "benign_b": 0.30, "attacker": 0.90},
            {"benign_a": 0.95, "benign_b": 0.95, "attacker": 0.90},
            {"benign_a": 0.19, "benign_b": 0.28, "attacker": 0.81},
            {"benign_a": 0.95, "benign_b": 0.95, "attacker": 0.90},
            cohort,
        )
        assert cohort_trust["trust"]["benign_a"] == pytest.approx(0.95)
        assert cohort_trust["trust"]["benign_b"] == pytest.approx(0.95)
        assert cohort_trust["trust"]["attacker"] == pytest.approx(0.81)

    def test_withheld_norm_outlier_holds_the_round(self):
        defense = ACFLDefense(
            config=ACFLConfig(adaptive_l1l2_gating_enabled=True),
            learning_mode="classification",
        )
        defense.last_layer1_layer2_gating = {
            "robustness_scores": {"benign": 1.0, "attacker": 0.25},
        }
        defense.last_update_norm_rejection = {
            "rejection_withheld_insufficient_participants": True,
            "candidate_rejected_clients": [
                {"client_id": "attacker", "norm_ratio": 3.0},
            ],
        }
        reason = defense._aggregation_guard_decision(
            ["benign", "attacker"],
            {"benign": 1.0, "attacker": 0.22},
            {"benign": 1.0, "attacker": 0.89},
        )
        assert reason == "withheld_norm_outlier:attacker"

        defense.last_layer1_layer2_gating = {
            "robustness_scores": {"benign": 0.95, "attacker": 0.20},
        }
        low_trust = defense._aggregation_guard_decision(
            ["benign", "attacker"],
            {"benign": 0.90, "attacker": 0.16},
            {"benign": 0.95, "attacker": 0.20},
        )
        assert low_trust is None

        defense.last_update_norm_rejection = {
            "rejection_withheld_insufficient_participants": False,
            "candidate_rejected_clients": [],
        }
        defense.last_layer1_layer2_gating = {
            "robustness_scores": {"benign": 1.0, "attacker": 0.499},
        }
        under_threshold = defense._aggregation_guard_decision(
            ["benign", "attacker"],
            {"benign": 1.0, "attacker": 0.43},
            {"benign": 1.0, "attacker": 0.866},
        )
        assert under_threshold == "missed_norm_outlier:attacker"

        defense.last_layer1_layer2_gating = {
            "robustness_scores": {"benign": 0.40, "attacker": 0.40},
        }
        both_low = defense._aggregation_guard_decision(
            ["benign", "attacker"],
            {"benign": 0.40, "attacker": 0.40},
            {"benign": 0.90, "attacker": 0.90},
        )
        assert both_low is None

    def test_backdoor_attack_blends_layer1_raw_conservatively(self):
        config = ACFLConfig(
            enable_layer1=True,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            layer1_layer2_gating_enabled=True,
            robust_norm_soft_cap_enabled=True,
            update_norm_hard_rejection_enabled=False,
            backdoor_probe_hard_reject_enabled=False,
            backdoor_geometry_hard_reject_enabled=False,
            min_aggregation_participants=1,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        defense.layer1.score_updates = lambda *args, **kwargs: {
            "benign": 0.95,
            "attacker": 0.25,
        }
        global_w = _make_state_dict(0)
        benign_w = _make_state_dict(1)
        attacker_w = _make_state_dict(2)

        defense.defend_and_aggregate(
            ["benign", "attacker"],
            [benign_w, attacker_w],
            global_w,
        )

        gating = defense.last_layer1_layer2_gating
        assert gating["backdoor_layer1_in_aggregation"] is True
        clients = defense.layer2.last_aggregation_diagnostics["clients"]
        assert clients["attacker"]["layer1_trust"] <= clients["benign"]["layer1_trust"]

    def test_backdoor_probe_aware_soft_cap_uses_trigger_harm(self):
        config = ACFLConfig(
            enable_layer1=True,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            layer1_layer2_gating_enabled=True,
            backdoor_probe_aware_soft_cap_enabled=True,
            backdoor_probe_blend_with_raw_l1=False,
            backdoor_trigger_hard_gate_enabled=False,
            backdoor_probe_hard_reject_enabled=False,
            backdoor_geometry_hard_reject_enabled=False,
            robust_norm_soft_cap_enabled=False,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
            min_trust_weight=0.05,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        defense.layer1.score_updates = lambda *args, **kwargs: {
            "benign": 0.05,
            "attacker": 0.05,
        }

        class _ProbeStub:
            def backdoor_probe_cap_scores(self, client_ids, client_weights, global_weights):
                return {"benign": 0.95, "attacker": 0.20}

            def backdoor_trigger_harm_details(self, client_ids, client_weights, global_weights):
                return {
                    "benign": {"harm": 0.05, "probe_cap": 0.95},
                    "attacker": {"harm": 1.60, "probe_cap": 0.20},
                }

        defense.probe_evaluator = _ProbeStub()
        global_w = _make_state_dict(0)
        benign_w = _make_state_dict(1)
        attacker_w = _make_state_dict(2)

        defense.defend_and_aggregate(
            ["benign", "attacker"],
            [benign_w, attacker_w],
            global_w,
        )

        cap = defense.last_backdoor_probe_soft_cap
        assert cap["enabled"] is True
        assert cap["probe_caps"]["attacker"] == pytest.approx(0.20)
        clients = defense.layer2.last_aggregation_diagnostics["clients"]
        assert clients["attacker"]["layer1_trust"] == pytest.approx(0.20)
        assert clients["benign"]["layer1_trust"] == pytest.approx(0.95)

    def test_backdoor_probe_blend_with_raw_l1(self):
        config = ACFLConfig(
            enable_layer1=True,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            layer1_layer2_gating_enabled=True,
            backdoor_probe_aware_soft_cap_enabled=True,
            backdoor_probe_blend_with_raw_l1=True,
            backdoor_trigger_hard_gate_enabled=False,
            backdoor_probe_hard_reject_enabled=False,
            backdoor_geometry_hard_reject_enabled=False,
            robust_norm_soft_cap_enabled=False,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
            min_trust_weight=0.03,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        defense.layer1.score_updates = lambda *args, **kwargs: {
            "benign": 0.90,
            "attacker": 0.40,
        }

        class _ProbeStub:
            def backdoor_probe_cap_scores(self, client_ids, client_weights, global_weights):
                return {"benign": 0.95, "attacker": 0.55}

            def backdoor_trigger_harm_details(self, client_ids, client_weights, global_weights):
                return {
                    "benign": {"harm": 0.05, "probe_cap": 0.95},
                    "attacker": {"harm": 0.60, "probe_cap": 0.55},
                }

        defense.probe_evaluator = _ProbeStub()
        global_w = _make_state_dict(0)
        benign_w = _make_state_dict(1)
        attacker_w = _make_state_dict(2)

        defense.defend_and_aggregate(
            ["benign", "attacker"],
            [benign_w, attacker_w],
            global_w,
        )

        clients = defense.layer2.last_aggregation_diagnostics["clients"]
        assert clients["attacker"]["layer1_trust"] <= 0.55

    def test_model_replacement_defaults_use_l1_and_hard_reject(self):
        config = ACFLConfig(evaluation_attack_type="model_replacement")
        assert config.use_layer1_for_model_replacement is True
        assert config.update_norm_hard_rejection_for_model_replacement is True
        assert config.update_norm_hard_rejection_enabled is False

    def test_model_replacement_uses_layer1_by_default(self):
        config = ACFLConfig(
            enable_layer1=True,
            enable_layer2=True,
            evaluation_attack_type="model_replacement",
            layer1_layer2_gating_enabled=True,
            use_layer1_for_model_replacement=True,
            update_norm_hard_rejection_enabled=False,
            update_norm_hard_rejection_for_model_replacement=False,
            robust_norm_soft_cap_enabled=False,
            min_aggregation_participants=1,
            min_trust_weight=0.05,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        defense.layer1.score_updates = lambda *args, **kwargs: {
            "benign": 0.95,
            "attacker": 0.20,
        }
        global_w = _make_state_dict(0)
        benign_w = _make_state_dict(1)
        attacker_w = _make_state_dict(2)

        defense.defend_and_aggregate(
            ["benign", "attacker"],
            [benign_w, attacker_w],
            global_w,
        )

        gating = defense.last_layer1_layer2_gating
        assert gating["backdoor_layer1_in_aggregation"] is True
        clients = defense.layer2.last_aggregation_diagnostics["clients"]
        assert clients["attacker"]["layer1_trust"] < clients["benign"]["layer1_trust"]

    def test_model_replacement_auto_hard_rejects_extreme_norm(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=False,
            evaluation_attack_type="model_replacement",
            update_norm_hard_rejection_enabled=False,
            update_norm_hard_rejection_for_model_replacement=True,
            update_norm_hard_rejection_ratio=2.0,
            update_norm_hard_rejection_ratio_two_clients=2.0,
            update_norm_hard_rejection_deviation=4.0,
            min_aggregation_participants=1,
        )
        defense = ACFLDefense(config=config)
        global_w = _make_state_dict(0)
        benign_w = _make_state_dict(1)
        extreme_w = {}
        for key in benign_w:
            extreme_w[key] = benign_w[key] + torch.randn_like(benign_w[key]) * 100.0

        result = defense.defend_and_aggregate(
            ["benign", "attacker"],
            [benign_w, extreme_w],
            global_w,
        )

        rejection = defense.last_update_norm_rejection
        assert rejection["enabled"] is True
        assert rejection["model_replacement_scoped"] is True
        assert rejection["global_flag"] is False
        assert len(rejection["rejected_clients"]) == 1
        assert rejection["rejected_clients"][0]["client_id"] == "attacker"
        for key in global_w:
            assert torch.allclose(result[key], benign_w[key], atol=1e-5)

    def test_backdoor_defaults_use_probe_hard_reject(self):
        config = ACFLConfig(evaluation_attack_type="backdoor")
        assert config.backdoor_probe_hard_reject_enabled is True
        assert config.backdoor_geometry_hard_reject_enabled is True
        assert config.backdoor_geometry_hard_reject_mode == "shortlist_veto"
        assert config.backdoor_geometry_shortlist_k == 2
        assert config.post_agg_trigger_scrub_enabled is True
        assert config.backdoor_trigger_hard_gate_enabled is False

    def test_label_flip_defaults_use_geometry_shortlist_clean(self):
        config = ACFLConfig(evaluation_attack_type="label_flip")
        assert config.label_flip_geometry_hard_reject_enabled is True
        assert config.label_flip_geometry_hard_reject_mode == "shortlist_clean"
        assert config.label_flip_geometry_shortlist_k == 2
        assert config.label_flip_geometry_clean_margin == 0.025

    def test_label_flip_shortlist_clean_blend_keeps_two_cleanest(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=True,
            evaluation_attack_type="label_flip",
            label_flip_geometry_hard_reject_enabled=True,
            label_flip_geometry_hard_reject_mode="shortlist_clean_blend",
            label_flip_geometry_shortlist_k=3,
            label_flip_geometry_blend_keep=2,
            label_flip_geometry_clean_margin=0.05,
            label_flip_geometry_hard_reject_warmup_rounds=0,
            label_flip_geometry_hard_reject_min_keep=1,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
            single_participant_trust_threshold=0.0,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        global_w = _make_state_dict(0)
        benign_a = _make_state_dict(1)
        benign_b = {k: v + 0.02 for k, v in benign_a.items()}
        attacker = {k: v + 0.04 for k, v in benign_a.items()}

        class _ProbeStub:
            def clean_probe_quality_details(self, client_ids, client_weights, global_weights):
                return {
                    "benign_a": {"clean_quality": 0.90, "harm": 0.01},
                    "benign_b": {"clean_quality": 0.88, "harm": 0.02},
                    "attacker": {"clean_quality": 0.40, "harm": 0.50},
                }

        defense.probe_evaluator = _ProbeStub()
        defense.defend_and_aggregate(
            ["benign_a", "benign_b", "attacker"],
            [benign_a, benign_b, attacker],
            global_w,
        )
        rejection = defense.last_backdoor_geometry_hard_rejection
        assert rejection["mode"] == "shortlist_clean_blend"
        assert set(rejection["accepted_clients"]) == {"benign_a", "benign_b"}
        assert "attacker" not in rejection["accepted_clients"]
        assert "benign_a" in (rejection.get("blend_weights") or {})
        assert "benign_b" in (rejection.get("blend_weights") or {})

    def test_label_flip_shortlist_clean_prefers_higher_clean_quality(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=True,
            evaluation_attack_type="label_flip",
            label_flip_geometry_hard_reject_enabled=True,
            label_flip_geometry_hard_reject_mode="shortlist_clean",
            label_flip_geometry_shortlist_k=2,
            label_flip_geometry_clean_margin=0.025,
            label_flip_geometry_hard_reject_warmup_rounds=0,
            label_flip_geometry_hard_reject_min_keep=1,
            backdoor_geometry_hard_reject_enabled=False,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
            single_participant_trust_threshold=0.0,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        # Geometric winner is attacker (close to benign_a); clean quality should
        # prefer benign_a over attacker within the shortlist when margin clears.
        global_w = _make_state_dict(0)
        benign_a = _make_state_dict(1)
        attacker = {k: v + 0.02 for k, v in benign_a.items()}
        benign_b = {k: v + 1.0 for k, v in benign_a.items()}

        class _ProbeStub:
            def clean_probe_quality_details(self, client_ids, client_weights, global_weights):
                return {
                    "benign_a": {"clean_quality": 0.90, "harm": 0.01},
                    "attacker": {"clean_quality": 0.40, "harm": 0.50},
                    "benign_b": {"clean_quality": 0.85, "harm": 0.05},
                }

        defense.probe_evaluator = _ProbeStub()
        defense.defend_and_aggregate(
            ["benign_a", "attacker", "benign_b"],
            [benign_a, attacker, benign_b],
            global_w,
        )
        rejection = defense.last_backdoor_geometry_hard_rejection
        assert rejection["enabled"] is True
        assert rejection["label_flip_scoped"] is True
        assert rejection["mode"] == "shortlist_clean"
        assert rejection["best_client"] == "benign_a"
        assert set(rejection["accepted_clients"]) == {"benign_a"}
        assert "attacker" not in rejection["accepted_clients"]

    def test_label_flip_shortlist_clean_keeps_geom_on_small_margin(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=True,
            evaluation_attack_type="label_flip",
            label_flip_geometry_hard_reject_enabled=True,
            label_flip_geometry_hard_reject_mode="shortlist_clean",
            label_flip_geometry_shortlist_k=2,
            label_flip_geometry_clean_margin=0.025,
            label_flip_geometry_hard_reject_warmup_rounds=0,
            label_flip_geometry_hard_reject_min_keep=1,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
            single_participant_trust_threshold=0.0,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        global_w = _make_state_dict(0)
        # Make benign_a the clear geometric winner (close cluster).
        benign_a = _make_state_dict(1)
        benign_b = {k: v + 0.01 for k, v in benign_a.items()}
        far = {k: v + 5.0 for k, v in benign_a.items()}

        class _ProbeStub:
            def clean_probe_quality_details(self, client_ids, client_weights, global_weights):
                # Tiny clean edge for benign_b — should NOT override geom winner.
                return {
                    "benign_a": {"clean_quality": 0.500, "harm": 0.10},
                    "benign_b": {"clean_quality": 0.510, "harm": 0.09},
                    "far": {"clean_quality": 0.400, "harm": 0.20},
                }

        defense.probe_evaluator = _ProbeStub()
        defense.defend_and_aggregate(
            ["benign_a", "benign_b", "far"],
            [benign_a, benign_b, far],
            global_w,
        )
        rejection = defense.last_backdoor_geometry_hard_rejection
        assert rejection["best_client"] == rejection["geom_best_client"]
        assert "keep_geom" in (rejection.get("keep_reason") or "")

    def test_label_flip_keep_best_matches_krum_winner(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=True,
            evaluation_attack_type="label_flip",
            label_flip_geometry_hard_reject_enabled=True,
            label_flip_geometry_hard_reject_mode="keep_best",
            label_flip_geometry_hard_reject_warmup_rounds=0,
            label_flip_geometry_hard_reject_min_keep=1,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
            single_participant_trust_threshold=0.0,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        global_w = _make_state_dict(0)
        benign_a = _make_state_dict(1)
        benign_b = {k: v + 0.01 for k, v in benign_a.items()}
        attacker = {k: v + 50.0 for k, v in benign_a.items()}
        defense.defend_and_aggregate(
            ["benign_a", "benign_b", "attacker"],
            [benign_a, benign_b, attacker],
            global_w,
        )
        rejection = defense.last_backdoor_geometry_hard_rejection
        assert rejection["mode"] == "keep_best"
        assert "attacker" not in rejection["accepted_clients"]
        assert set(rejection["accepted_clients"]) == {rejection["best_client"]}

    def test_backdoor_probe_hard_reject_drops_worst_outlier(self):
        config = ACFLConfig(
            enable_layer1=True,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            layer1_layer2_gating_enabled=True,
            backdoor_probe_hard_reject_enabled=True,
            backdoor_probe_hard_reject_harm=0.20,
            backdoor_probe_hard_reject_peer_margin=0.04,
            backdoor_probe_hard_reject_warmup_rounds=0,
            backdoor_probe_hard_reject_max_trust=0.90,
            backdoor_geometry_hard_reject_enabled=False,
            backdoor_trigger_hard_gate_enabled=False,
            robust_norm_soft_cap_enabled=False,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=2,
            single_participant_trust_threshold=0.5,
            min_trust_weight=0.05,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        defense.layer1.score_updates = lambda *args, **kwargs: {
            "benign_a": 0.90,
            "benign_b": 0.88,
            "attacker": 0.85,
        }

        class _ProbeStub:
            def backdoor_trigger_harm_details(self, client_ids, client_weights, global_weights):
                return {
                    "benign_a": {
                        "harm": 0.10,
                        "counterfactual_p_attack": 0.80,
                        "probe_cap": 0.90,
                    },
                    "benign_b": {
                        "harm": 0.12,
                        "counterfactual_p_attack": 0.78,
                        "probe_cap": 0.88,
                    },
                    "attacker": {
                        "harm": 0.35,
                        "counterfactual_p_attack": 0.40,
                        "probe_cap": 0.70,
                    },
                }

        defense.probe_evaluator = _ProbeStub()
        global_w = _make_state_dict(0)
        defense.defend_and_aggregate(
            ["benign_a", "benign_b", "attacker"],
            [_make_state_dict(1), _make_state_dict(2), _make_state_dict(3)],
            global_w,
        )

        rejection = defense.last_backdoor_probe_hard_rejection
        assert rejection["enabled"] is True
        assert rejection["backdoor_scoped"] is True
        assert len(rejection["rejected_clients"]) == 1
        assert rejection["rejected_clients"][0]["client_id"] == "attacker"
        assert set(rejection["accepted_clients"]) == {"benign_a", "benign_b"}
        assert defense.last_aggregation_guard["skipped"] is False

    def test_backdoor_probe_hard_reject_skips_tied_harms(self):
        config = ACFLConfig(
            enable_layer1=True,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            layer1_layer2_gating_enabled=True,
            backdoor_probe_hard_reject_enabled=True,
            backdoor_probe_hard_reject_harm=0.20,
            backdoor_probe_hard_reject_peer_margin=0.04,
            backdoor_probe_hard_reject_warmup_rounds=0,
            backdoor_geometry_hard_reject_enabled=False,
            backdoor_trigger_hard_gate_enabled=False,
            robust_norm_soft_cap_enabled=False,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        defense.layer1.score_updates = lambda *args, **kwargs: {
            "benign": 0.90,
            "attacker": 0.85,
        }

        class _ProbeStub:
            def backdoor_trigger_harm_details(self, client_ids, client_weights, global_weights):
                return {
                    "benign": {"harm": 0.30, "counterfactual_p_attack": 0.5, "probe_cap": 0.7},
                    "attacker": {"harm": 0.31, "counterfactual_p_attack": 0.5, "probe_cap": 0.7},
                }

        defense.probe_evaluator = _ProbeStub()
        global_w = _make_state_dict(0)
        defense.defend_and_aggregate(
            ["benign", "attacker"],
            [_make_state_dict(1), _make_state_dict(2)],
            global_w,
        )

        rejection = defense.last_backdoor_probe_hard_rejection
        assert rejection["enabled"] is True
        assert rejection["rejected_clients"] == []

    def test_backdoor_probe_hard_reject_withholds_two_client_rounds(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            backdoor_probe_hard_reject_enabled=True,
            backdoor_probe_hard_reject_harm=0.10,
            backdoor_probe_hard_reject_peer_margin=0.02,
            backdoor_probe_hard_reject_warmup_rounds=0,
            backdoor_probe_hard_reject_cohort_factor=1.5,
            backdoor_probe_hard_reject_min_keep=2,
            backdoor_geometry_hard_reject_enabled=False,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=2,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")

        class _ProbeStub:
            def backdoor_trigger_harm_details(self, client_ids, client_weights, global_weights):
                return {
                    "benign": {"harm": 0.10, "counterfactual_p_attack": 0.8, "probe_cap": 0.9},
                    "attacker": {"harm": 0.35, "counterfactual_p_attack": 0.4, "probe_cap": 0.7},
                }

        defense.probe_evaluator = _ProbeStub()
        global_w = _make_state_dict(0)
        defense.defend_and_aggregate(
            ["benign", "attacker"],
            [_make_state_dict(1), _make_state_dict(2)],
            global_w,
        )

        # n=2: reject would leave 1 < min_after_reject(2) → withhold soft path.
        rejection = defense.last_backdoor_probe_hard_rejection
        assert rejection.get("skipped_sparse_participation") is not True
        assert rejection["rejection_withheld_insufficient_participants"] is True
        assert rejection["rejected_clients"] == []
        assert len(rejection["candidate_rejected_clients"]) == 1
        assert defense.last_aggregation_guard["skipped"] is False

    def test_backdoor_probe_hard_reject_fires_at_three_clients(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            backdoor_probe_hard_reject_enabled=True,
            backdoor_probe_hard_reject_harm=0.20,
            backdoor_probe_hard_reject_peer_margin=0.04,
            backdoor_probe_hard_reject_warmup_rounds=0,
            backdoor_geometry_hard_reject_enabled=False,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=2,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")

        class _ProbeStub:
            def backdoor_trigger_harm_details(self, client_ids, client_weights, global_weights):
                return {
                    "benign_a": {"harm": 0.10, "counterfactual_p_attack": 0.8, "probe_cap": 0.9},
                    "benign_b": {"harm": 0.12, "counterfactual_p_attack": 0.75, "probe_cap": 0.88},
                    "attacker": {"harm": 0.35, "counterfactual_p_attack": 0.4, "probe_cap": 0.7},
                }

        defense.probe_evaluator = _ProbeStub()
        global_w = _make_state_dict(0)
        defense.defend_and_aggregate(
            ["benign_a", "benign_b", "attacker"],
            [_make_state_dict(1), _make_state_dict(2), _make_state_dict(3)],
            global_w,
        )

        rejection = defense.last_backdoor_probe_hard_rejection
        assert rejection["rejection_withheld_insufficient_participants"] is False
        assert len(rejection["rejected_clients"]) == 1
        assert rejection["rejected_clients"][0]["client_id"] == "attacker"
        assert defense.last_aggregation_guard["skipped"] is False

    def test_backdoor_probe_hard_reject_blocks_high_trust_harm_outlier(self):
        config = ACFLConfig(
            enable_layer1=True,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            layer1_layer2_gating_enabled=True,
            backdoor_probe_hard_reject_enabled=True,
            backdoor_probe_hard_reject_harm=0.20,
            backdoor_probe_hard_reject_peer_margin=0.04,
            backdoor_probe_hard_reject_warmup_rounds=0,
            backdoor_probe_hard_reject_max_trust=0.85,
            backdoor_geometry_hard_reject_enabled=False,
            backdoor_trigger_hard_gate_enabled=False,
            robust_norm_soft_cap_enabled=False,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=2,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        defense.layer1.score_updates = lambda *args, **kwargs: {
            "benign_a": 0.90,
            "benign_b": 0.88,
            "attacker": 0.95,
        }

        class _ProbeStub:
            def backdoor_trigger_harm_details(self, client_ids, client_weights, global_weights):
                return {
                    "benign_a": {"harm": 0.10, "counterfactual_p_attack": 0.8, "probe_cap": 0.9},
                    "benign_b": {"harm": 0.12, "counterfactual_p_attack": 0.75, "probe_cap": 0.88},
                    "attacker": {"harm": 0.35, "counterfactual_p_attack": 0.4, "probe_cap": 0.7},
                }

        defense.probe_evaluator = _ProbeStub()
        global_w = _make_state_dict(0)
        defense.defend_and_aggregate(
            ["benign_a", "benign_b", "attacker"],
            [_make_state_dict(1), _make_state_dict(2), _make_state_dict(3)],
            global_w,
        )

        rejection = defense.last_backdoor_probe_hard_rejection
        assert rejection["rejected_clients"] == []
        assert "attacker" in rejection["trust_blocked_clients"]

    def test_backdoor_geometry_hard_reject_drops_isolated_client(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            backdoor_probe_hard_reject_enabled=False,
            backdoor_geometry_hard_reject_enabled=True,
            backdoor_geometry_hard_reject_mode="reject_worst",
            backdoor_geometry_hard_reject_score_ratio=1.25,
            backdoor_geometry_hard_reject_warmup_rounds=0,
            backdoor_geometry_hard_reject_min_keep=2,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=2,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        global_w = _make_state_dict(0)
        benign_a = _make_state_dict(1)
        benign_b = {
            k: v + 0.01 for k, v in benign_a.items()
        }
        attacker = {
            k: v + 50.0 for k, v in benign_a.items()
        }
        defense.defend_and_aggregate(
            ["benign_a", "benign_b", "attacker"],
            [benign_a, benign_b, attacker],
            global_w,
        )

        rejection = defense.last_backdoor_geometry_hard_rejection
        assert rejection["enabled"] is True
        assert len(rejection["rejected_clients"]) == 1
        assert rejection["rejected_clients"][0]["client_id"] == "attacker"
        assert defense.last_aggregation_guard["skipped"] is False

    def test_backdoor_geometry_keep_best_selects_krum_winner(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            backdoor_probe_hard_reject_enabled=False,
            backdoor_geometry_hard_reject_enabled=True,
            backdoor_geometry_hard_reject_mode="keep_best",
            backdoor_geometry_hard_reject_warmup_rounds=0,
            backdoor_geometry_hard_reject_min_keep=1,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=2,
            single_participant_trust_threshold=0.0,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        global_w = _make_state_dict(0)
        benign_a = _make_state_dict(1)
        benign_b = {k: v + 0.01 for k, v in benign_a.items()}
        attacker = {k: v + 50.0 for k, v in benign_a.items()}
        defense.defend_and_aggregate(
            ["benign_a", "benign_b", "attacker"],
            [benign_a, benign_b, attacker],
            global_w,
        )

        rejection = defense.last_backdoor_geometry_hard_rejection
        assert rejection["mode"] == "keep_best"
        assert set(rejection["accepted_clients"]) == {rejection["best_client"]}
        assert "attacker" not in rejection["accepted_clients"]
        assert defense.last_aggregation_guard["skipped"] is False

    def test_backdoor_geometry_shortlist_veto_prefers_lower_trigger(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            backdoor_probe_hard_reject_enabled=False,
            backdoor_geometry_hard_reject_enabled=True,
            backdoor_geometry_hard_reject_mode="shortlist_veto",
            backdoor_geometry_shortlist_k=2,
            backdoor_geometry_trigger_veto_margin=0.01,
            backdoor_geometry_hard_reject_warmup_rounds=0,
            backdoor_geometry_hard_reject_min_keep=1,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
            single_participant_trust_threshold=0.0,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        # Geometric winner is attacker (close to one benign); second-best is
        # the other benign. Absolute trigger veto should pick the benign.
        global_w = _make_state_dict(0)
        benign_a = _make_state_dict(1)
        attacker = {k: v + 0.02 for k, v in benign_a.items()}
        benign_b = {k: v + 1.0 for k, v in benign_a.items()}

        class _ProbeStub:
            def backdoor_trigger_harm_details(self, client_ids, client_weights, global_weights):
                return {
                    "benign_a": {
                        "harm": 0.10,
                        "counterfactual_p_attack": 0.20,
                        "probe_cap": 0.9,
                    },
                    "attacker": {
                        "harm": 0.01,
                        "counterfactual_p_attack": 0.55,
                        "probe_cap": 0.99,
                    },
                    "benign_b": {
                        "harm": 0.12,
                        "counterfactual_p_attack": 0.22,
                        "probe_cap": 0.88,
                    },
                }

        defense.probe_evaluator = _ProbeStub()
        defense.defend_and_aggregate(
            ["benign_a", "attacker", "benign_b"],
            [benign_a, attacker, benign_b],
            global_w,
        )

        rejection = defense.last_backdoor_geometry_hard_rejection
        assert rejection["mode"] == "shortlist_veto"
        assert "attacker" in rejection["shortlist"] or rejection["geom_best_client"] == "attacker"
        assert rejection["accepted_clients"] == ["benign_a"]
        assert "attacker" in (rejection.get("vetoed_from_shortlist") or ["attacker"])
        assert defense.last_aggregation_guard["skipped"] is False

    def test_backdoor_geometry_shortlist_blend_dilutes_tied_trigger(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            backdoor_probe_hard_reject_enabled=False,
            backdoor_geometry_hard_reject_enabled=True,
            backdoor_geometry_hard_reject_mode="shortlist_blend",
            backdoor_geometry_shortlist_k=3,
            backdoor_geometry_trigger_veto_margin=0.01,
            backdoor_geometry_blend_tie_eps=0.01,
            backdoor_geometry_hard_reject_warmup_rounds=0,
            backdoor_geometry_hard_reject_min_keep=1,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
            single_participant_trust_threshold=0.0,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        global_w = _make_state_dict(0)
        benign_a = _make_state_dict(1)
        attacker = {k: v + 0.02 for k, v in benign_a.items()}
        benign_b = {k: v + 0.05 for k, v in benign_a.items()}
        outlier = {k: v + 40.0 for k, v in benign_a.items()}

        class _ProbeStub:
            def backdoor_trigger_harm_details(self, client_ids, client_weights, global_weights):
                # Attacker tied with benigns on absolute trigger → must blend.
                return {
                    "benign_a": {"harm": 0.1, "counterfactual_p_attack": 0.40, "probe_cap": 0.9},
                    "attacker": {"harm": 0.0, "counterfactual_p_attack": 0.40, "probe_cap": 1.0},
                    "benign_b": {"harm": 0.1, "counterfactual_p_attack": 0.405, "probe_cap": 0.9},
                    "outlier": {"harm": 0.5, "counterfactual_p_attack": 0.70, "probe_cap": 0.6},
                }

        defense.probe_evaluator = _ProbeStub()
        defense.defend_and_aggregate(
            ["benign_a", "attacker", "benign_b", "outlier"],
            [benign_a, attacker, benign_b, outlier],
            global_w,
        )

        rejection = defense.last_backdoor_geometry_hard_rejection
        assert rejection["mode"] == "shortlist_blend"
        assert "outlier" not in rejection["accepted_clients"]
        assert len(rejection["accepted_clients"]) >= 2
        assert "attacker" in rejection["accepted_clients"]
        assert "attacker" in rejection["blend_weights"]
        # Attacker must not be the sole update (dilution).
        assert set(rejection["accepted_clients"]) != {"attacker"}
        assert defense.last_aggregation_guard["skipped"] is False

    def test_post_agg_trigger_scrub_applies_and_moves_weights(self):
        torch.manual_seed(0)
        model = TinyModel()
        # Benign windows → y=0; trigger windows → y=1.
        benign_x = torch.randn(8, 3, 4)
        trig_x = torch.randn(8, 3, 4) + 2.0
        probe_data = {
            "benign": (benign_x, torch.zeros(8)),
            "backdoor_trigger": (trig_x, torch.ones(8)),
        }
        config = ACFLConfig(
            enable_layer1=True,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            backdoor_probe_hard_reject_enabled=False,
            backdoor_geometry_hard_reject_enabled=False,
            post_agg_trigger_scrub_enabled=True,
            post_agg_trigger_scrub_steps=5,
            post_agg_trigger_scrub_lr=0.05,
            post_agg_trigger_scrub_trigger_coef=1.0,
            post_agg_trigger_scrub_max_delta_norm=10.0,
            post_agg_trigger_scrub_warmup_rounds=0,
            update_norm_hard_rejection_enabled=False,
            robust_norm_soft_cap_enabled=False,
            min_aggregation_participants=1,
        )
        defense = ACFLDefense(
            config=config,
            model=model,
            probe_data=probe_data,
            learning_mode="classification",
        )
        # Bias the model toward predicting attack on everything (high P).
        with torch.no_grad():
            model.fc.bias.fill_(3.0)

        from acfl.weight_utils import clone_state_dict as _clone
        global_w = _clone(model.get_weights())
        client_w = _clone(global_w)
        pre_trig = float(torch.sigmoid(model(trig_x)).mean().item())

        defense.layer1.score_updates = lambda *args, **kwargs: {"c1": 1.0}
        out = defense.defend_and_aggregate(["c1"], [client_w], global_w)

        scrub = defense.last_post_agg_trigger_scrub
        assert scrub["enabled"] is True
        assert scrub["applied"] is True
        assert scrub["steps"] == 5
        assert scrub.get("trigger_target") == 0.0
        model.set_weights(out)
        post_trig = float(torch.sigmoid(model(trig_x)).mean().item())
        assert post_trig < pre_trig
        assert defense.last_aggregation_guard["post_agg_trigger_scrub"]["applied"] is True

    def test_backdoor_trigger_hard_gate_clamps_peer_outlier(self):
        config = ACFLConfig(
            enable_layer1=True,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            layer1_layer2_gating_enabled=True,
            backdoor_probe_aware_soft_cap_enabled=False,
            backdoor_probe_hard_reject_enabled=False,
            backdoor_geometry_hard_reject_enabled=False,
            backdoor_trigger_hard_gate_enabled=True,
            backdoor_trigger_hard_gate_harm=2.0,
            backdoor_trigger_hard_gate_peer_margin=2.0,
            backdoor_trigger_hard_gate_warmup_rounds=0,
            backdoor_trigger_hard_gate_require_p_attack_floor=False,
            robust_norm_soft_cap_enabled=False,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
            min_trust_weight=0.05,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        defense.layer1.score_updates = lambda *args, **kwargs: {
            "benign": 0.90,
            "attacker": 0.85,
        }

        class _ProbeStub:
            def backdoor_trigger_harm_details(self, client_ids, client_weights, global_weights):
                return {
                    "benign": {
                        "harm": 0.2,
                        "counterfactual_p_attack": 0.80,
                        "probe_cap": 0.82,
                    },
                    "attacker": {
                        "harm": 5.0,
                        "counterfactual_p_attack": 0.40,
                        "probe_cap": float(__import__("math").exp(-5.0)),
                    },
                }

        defense.probe_evaluator = _ProbeStub()
        global_w = _make_state_dict(0)
        defense.defend_and_aggregate(
            ["benign", "attacker"],
            [_make_state_dict(1), _make_state_dict(2)],
            global_w,
        )

        gate = defense.last_backdoor_trigger_hard_gate
        assert gate["enabled"] is True
        assert gate.get("skipped_warmup") is False
        assert "attacker" in gate["clamped_clients"]
        assert "benign" not in gate["clamped_clients"]
        clients = defense.layer2.last_aggregation_diagnostics["clients"]
        assert clients["attacker"]["layer1_trust"] == pytest.approx(0.05)
        assert clients["benign"]["layer1_trust"] == pytest.approx(0.90)

    def test_backdoor_trigger_hard_gate_skips_tied_harms(self):
        config = ACFLConfig(
            enable_layer1=True,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            layer1_layer2_gating_enabled=True,
            backdoor_probe_aware_soft_cap_enabled=False,
            backdoor_trigger_hard_gate_enabled=True,
            backdoor_trigger_hard_gate_harm=2.0,
            backdoor_trigger_hard_gate_peer_margin=2.0,
            backdoor_trigger_hard_gate_warmup_rounds=0,
            robust_norm_soft_cap_enabled=False,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
            min_trust_weight=0.05,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        defense.layer1.score_updates = lambda *args, **kwargs: {
            "benign": 0.90,
            "attacker": 0.85,
        }

        class _ProbeStub:
            def backdoor_trigger_harm_details(self, client_ids, client_weights, global_weights):
                return {
                    "benign": {"harm": 8.5, "counterfactual_p_attack": 0.5, "probe_cap": 0.03},
                    "attacker": {"harm": 9.0, "counterfactual_p_attack": 0.5, "probe_cap": 0.03},
                }

        defense.probe_evaluator = _ProbeStub()
        defense.defend_and_aggregate(
            ["benign", "attacker"],
            [_make_state_dict(1), _make_state_dict(2)],
            _make_state_dict(0),
        )
        gate = defense.last_backdoor_trigger_hard_gate
        assert gate["clamped_clients"] == {}

    def test_backdoor_trigger_hard_gate_warmup_skips(self):
        config = ACFLConfig(
            enable_layer1=True,
            enable_layer2=True,
            evaluation_attack_type="backdoor",
            layer1_layer2_gating_enabled=True,
            backdoor_probe_aware_soft_cap_enabled=False,
            backdoor_probe_hard_reject_enabled=False,
            backdoor_geometry_hard_reject_enabled=False,
            backdoor_trigger_hard_gate_enabled=True,
            backdoor_trigger_hard_gate_harm=2.0,
            backdoor_trigger_hard_gate_peer_margin=2.0,
            backdoor_trigger_hard_gate_warmup_rounds=3,
            robust_norm_soft_cap_enabled=False,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
            min_trust_weight=0.05,
        )
        defense = ACFLDefense(config=config, learning_mode="classification")
        defense.round_number = 0  # first aggregation -> round 1 <= warmup
        defense.layer1.score_updates = lambda *args, **kwargs: {
            "benign": 0.90,
            "attacker": 0.85,
        }

        class _ProbeStub:
            def backdoor_trigger_harm_details(self, client_ids, client_weights, global_weights):
                raise AssertionError("probe harm should not run during warmup")

        defense.probe_evaluator = _ProbeStub()
        defense.defend_and_aggregate(
            ["benign", "attacker"],
            [_make_state_dict(1), _make_state_dict(2)],
            _make_state_dict(0),
        )
        gate = defense.last_backdoor_trigger_hard_gate
        assert gate["skipped_warmup"] is True
        assert gate["clamped_clients"] == {}

    def test_hold_on_non_finite_aggregate(self):
        config = ACFLConfig(
            enable_layer1=False,
            enable_layer2=True,
            robust_norm_soft_cap_enabled=False,
            update_norm_hard_rejection_enabled=False,
            min_aggregation_participants=1,
        )
        defense = ACFLDefense(config=config)
        global_w = _make_state_dict(0)
        client_w = _make_state_dict(1)

        def _nan_aggregate(*_args, **_kwargs):
            bad = copy.deepcopy(global_w)
            key = next(k for k, t in bad.items() if t.is_floating_point())
            bad[key] = torch.full_like(bad[key], float("nan"))
            return bad

        defense.layer2.aggregate = _nan_aggregate
        result = defense.defend_and_aggregate(["attacker"], [client_w], global_w)

        assert defense.last_aggregation_guard["skipped"] is True
        assert defense.last_aggregation_guard["reason"] == "non_finite_aggregate"
        for k in global_w:
            assert torch.allclose(result[k], global_w[k])
