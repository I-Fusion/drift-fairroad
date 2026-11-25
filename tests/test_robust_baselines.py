"""Unit tests for classical robust FL aggregation baselines."""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

ACFL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ACFL_ROOT)

from acfl.robust_baselines import (
    BASELINE_STRATEGIES,
    coordinate_median_aggregate,
    krum_aggregate,
    trimmed_mean_aggregate,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x[:, -1, :])


def _state_dicts(n: int, scale: float = 1.0) -> list:
    out = []
    for i in range(n):
        torch.manual_seed(i)
        model = TinyModel()
        sd = model.state_dict()
        if scale != 1.0:
            sd = {k: v * scale for k, v in sd.items()}
        out.append(sd)
    return out


class TestRobustBaselines:
    def test_baseline_strategy_names(self):
        assert "coordinate_median" in BASELINE_STRATEGIES
        assert "krum" in BASELINE_STRATEGIES

    def test_coordinate_median_shape(self):
        weights = _state_dicts(3)
        agg = coordinate_median_aggregate(weights)
        assert set(agg.keys()) == set(weights[0].keys())

    def test_trimmed_mean_prefers_robust_center_with_outlier(self):
        benign = _state_dicts(2, scale=1.0)
        outlier = _state_dicts(1, scale=50.0)[0]
        agg = trimmed_mean_aggregate(benign + [outlier])
        ref = coordinate_median_aggregate(benign + [outlier])
        for key in agg:
            diff_median = (agg[key].float() - ref[key].float()).abs().mean()
            diff_outlier = (agg[key].float() - outlier[key].float()).abs().mean()
            assert diff_median.item() < diff_outlier.item()

    def test_krum_selects_benign_cluster(self):
        global_w = _state_dicts(1, scale=1.0)[0]
        benign_a = _state_dicts(1, scale=1.0)[0]
        torch.manual_seed(99)
        benign_b = TinyModel().state_dict()
        outlier = _state_dicts(1, scale=100.0)[0]

        picked = krum_aggregate([benign_a, benign_b, outlier], global_weights=global_w)
        dist_a = sum(
            (picked[k].float() - benign_a[k].float()).abs().sum().item()
            for k in picked
        )
        dist_out = sum(
            (picked[k].float() - outlier[k].float()).abs().sum().item()
            for k in picked
        )
        assert dist_a < dist_out

    def test_federated_aggregator_exposes_baselines(self):
        # Works when this tree is merged into drift-fairroad, or when a
        # sibling development checkout still nests drift-fairroad/.
        candidates = [
            os.path.join(ACFL_ROOT, "fl-time-series"),
            os.path.join(ACFL_ROOT, "drift-fairroad", "fl-time-series"),
        ]
        drift_root = next((p for p in candidates if os.path.isdir(p)), None)
        if drift_root is None:
            pytest.skip("fl-time-series not present (merge into drift-fairroad to enable)")
        sys.path.insert(0, drift_root)
        from aggregation import FederatedAggregator

        weights = _state_dicts(3)
        for strategy in ("coordinate_median", "trimmed_mean", "krum"):
            agg = FederatedAggregator(strategy=strategy)
            result = agg.aggregate(weights)
            assert set(result.keys()) == set(weights[0].keys())
