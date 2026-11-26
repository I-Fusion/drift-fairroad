"""Tests for Parallel Trust Arbitration (dual-path counterfactual aggregation)."""
import os
import sys

import pytest
import torch
import torch.nn as nn

ACFL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ACFL_ROOT)

from acfl.config import ACFLConfig
from acfl.defense import ACFLDefense
from acfl.parallel_trust_arbitration import (
    ORACLE_FEATURE_KEYS,
    arbitrate,
    assert_runtime_features,
    candidate_objective,
    conservative_select,
    hard_select,
    mixture_eta,
)
from acfl.weight_utils import blend_state_dicts, clone_state_dict, state_dict_l2


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


def _state(seed: int):
    torch.manual_seed(seed)
    return TinyModel().state_dict()


def _pta_config(**kwargs) -> ACFLConfig:
    defaults = dict(
        enable_layer1=True,
        enable_layer2=True,
        parallel_trust_arbitration_enabled=True,
        parallel_trust_arbitration_method="hard",
        layer1_layer2_gating_enabled=True,
        adaptive_l1l2_gating_enabled=False,
        backdoor_probe_hard_reject_enabled=False,
        backdoor_geometry_hard_reject_enabled=False,
        post_agg_trigger_scrub_enabled=False,
        update_norm_hard_rejection_enabled=False,
        aggregation_guard_enabled=False,
        min_aggregation_participants=1,
        robust_norm_soft_cap_enabled=True,
        export_trust_scores=False,
    )
    defaults.update(kwargs)
    return ACFLConfig(**defaults)


class TestArbitrationMath:
    def test_hard_picks_lower_objective(self):
        assert hard_select(0.1, 0.4) == "gated"
        assert hard_select(0.5, 0.2) == "ungated"
        assert hard_select(0.3, 0.3, default_branch="ungated") == "ungated"

    def test_soft_mixture_is_sigmoid_of_gap(self):
        assert mixture_eta(0.0, 2.0, steepness=12.0) > 0.95
        assert mixture_eta(2.0, 0.0, steepness=12.0) < 0.05
        assert 0.49 < mixture_eta(0.2, 0.2, steepness=12.0) < 0.51

    def test_conservative_hysteresis(self):
        assert conservative_select(0.10, 0.12, tau=0.05, last_branch="ungated") == "ungated"
        assert conservative_select(0.10, 0.20, tau=0.05, last_branch="ungated") == "gated"
        assert conservative_select(0.20, 0.10, tau=0.05, last_branch="gated") == "ungated"
        assert conservative_select(0.12, 0.10, tau=0.05, last_branch="gated") == "gated"

    def test_arbitrate_soft_reports_mixture(self):
        decision = arbitrate(0.2, 0.8, method="soft", steepness=12.0)
        assert decision.selected == "mixture"
        assert decision.eta > 0.9

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError):
            arbitrate(0.1, 0.2, method="oracle")

    def test_selector_rejects_oracle_features(self):
        with pytest.raises(ValueError):
            assert_runtime_features(["L_task", "H", "evaluation_attack_type"])
        assert_runtime_features(["L_task", "H", "R", "S", "R_trigger"])
        assert "ground_truth_malicious" in ORACLE_FEATURE_KEYS

    def test_objective_is_weighted_sum(self):
        assert candidate_objective(0.2, 0.1, 0.3, 0.4, 0.5, 1, 1, 1, 1) == pytest.approx(1.5)

    def test_skip_penalty_raises_hold_objective(self):
        live = candidate_objective(0.2, 0.05, 0.05, 0.5, 0.0, 1, 1, 0.1, 1)
        hold = candidate_objective(
            0.2, 0.0, 0.0, 0.0, 0.0, 1, 1, 0.1, 1, skipped=True, lambda_skip=0.0
        )
        hold_penalized = candidate_objective(
            0.2, 0.0, 0.0, 0.0, 0.0, 1, 1, 0.1, 1, skipped=True, lambda_skip=0.25
        )
        assert hold < live
        assert hold_penalized == pytest.approx(hold + 0.25)
        assert hard_select(hold, live) == "gated"
        assert hard_select(hold_penalized, live) == "ungated"


class TestBlendUtils:
    def test_blend_recovers_endpoints(self):
        gated = _state(1)
        ungated = _state(2)
        only_gated = blend_state_dicts(gated, ungated, 1.0)
        only_ungated = blend_state_dicts(gated, ungated, 0.0)
        assert state_dict_l2(only_gated, gated) == pytest.approx(0.0)
        assert state_dict_l2(only_ungated, ungated) == pytest.approx(0.0)
        mid = blend_state_dicts(gated, ungated, 0.5)
        assert state_dict_l2(mid, gated) > 0.0
        assert state_dict_l2(mid, ungated) > 0.0


class TestDefensePTA:
    def test_round_commits_one_model_and_one_participation(self):
        defense = ACFLDefense(
            config=_pta_config(),
            learning_mode="classification",
        )
        global_w = _state(0)
        clients = [_state(1), _state(2), _state(3)]
        ids = ["benign_a", "benign_b", "attacker"]
        out = defense.defend_and_aggregate(ids, clients, global_w)
        assert set(out.keys()) == set(global_w.keys())
        assert defense.round_number == 1
        diag = defense.last_parallel_trust_arbitration
        assert diag["enabled"] is True
        assert diag["method"] == "hard"
        assert diag["selected"] in ("gated", "ungated")
        assert 0.0 <= float(diag["eta"]) <= 1.0
        for cid in ids:
            assert defense.layer2.client_participation[cid] == 1

    def test_ground_truth_labels_do_not_change_decision(self):
        torch.manual_seed(0)
        ids = ["benign", "attacker"]
        global_w = _state(0)
        clients = [_state(4), _state(9)]

        clean = ACFLDefense(config=_pta_config(), learning_mode="classification")
        labeled = ACFLDefense(
            config=_pta_config(
                ground_truth_malicious={"attacker": True, "benign": False}
            ),
            learning_mode="classification",
        )
        out_clean = clean.defend_and_aggregate(
            ids,
            [clone_state_dict(clients[0]), clone_state_dict(clients[1])],
            clone_state_dict(global_w),
        )
        out_labeled = labeled.defend_and_aggregate(
            ids,
            [clone_state_dict(clients[0]), clone_state_dict(clients[1])],
            clone_state_dict(global_w),
        )
        assert clean.last_parallel_trust_arbitration["selected"] == (
            labeled.last_parallel_trust_arbitration["selected"]
        )
        assert clean.last_parallel_trust_arbitration["eta"] == pytest.approx(
            labeled.last_parallel_trust_arbitration["eta"]
        )
        assert state_dict_l2(out_clean, out_labeled) == pytest.approx(0.0)

    def test_soft_method_can_blend(self):
        defense = ACFLDefense(
            config=_pta_config(parallel_trust_arbitration_method="soft"),
            learning_mode="classification",
        )
        out = defense.defend_and_aggregate(
            ["c0", "c1", "c2"],
            [_state(1), _state(2), _state(3)],
            _state(0),
        )
        diag = defense.last_parallel_trust_arbitration
        assert diag["method"] == "soft"
        assert diag["selected"] == "mixture"
        assert set(out.keys())

    def test_conservative_starts_on_default_ungated(self):
        defense = ACFLDefense(
            config=_pta_config(
                parallel_trust_arbitration_method="conservative",
                parallel_trust_arbitration_tau=10.0,
            ),
            learning_mode="classification",
        )
        defense.defend_and_aggregate(
            ["c0", "c1"],
            [_state(1), _state(2)],
            _state(0),
        )
        assert defense.last_parallel_trust_arbitration["selected"] == "ungated"

    def test_aggregate_readonly_does_not_mutate_history(self):
        defense = ACFLDefense(config=_pta_config(), learning_mode="classification")
        ids = ["c0", "c1"]
        clients = [_state(1), _state(2)]
        global_w = _state(0)
        before_hist = list(defense.layer2.benign_deviation_history)
        before_part = dict(defense.layer2.client_participation)
        defense.layer2.aggregate_readonly(
            ids, clients, global_w, {"c0": 1.0, "c1": 0.2}
        )
        assert defense.layer2.benign_deviation_history == before_hist
        assert defense.layer2.client_participation == before_part

    def test_probe_terms_restore_model_weights(self):
        model = TinyModel()
        with torch.no_grad():
            model.fc.bias.fill_(0.0)
        probes = {
            "benign": (torch.randn(6, 3, 4), torch.zeros(6)),
            "gnss_spoofing": (torch.randn(6, 3, 4), torch.ones(6)),
            "backdoor_trigger": (torch.randn(6, 3, 4) + 1.5, torch.ones(6)),
        }
        defense = ACFLDefense(
            config=_pta_config(),
            model=model,
            probe_data=probes,
            learning_mode="classification",
        )
        original = clone_state_dict(model.get_weights())
        candidate = _state(11)
        terms = defense.probe_evaluator.candidate_probe_terms(
            candidate, original
        )
        assert "L_task" in terms and "H" in terms and "R_trigger" in terms
        assert state_dict_l2(model.get_weights(), original) == pytest.approx(0.0)
