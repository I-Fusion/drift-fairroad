"""Tests for Layer 3 checkpoint config inference and threshold tuning."""
import os
import sys

import pytest

ACFL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ACFL_ROOT not in sys.path:
    sys.path.insert(0, ACFL_ROOT)

from evaluate_layer3_sitl import infer_config_module, tune_threshold_for_f1


def test_infer_lstm_config():
    state = {"lstm.weight_ih_l0": None, "fc.0.weight": None}
    assert infer_config_module(state) == "config_classification"


def test_infer_tiny_cnn_config():
    state = {"features.0.weight": None, "head.1.weight": None}
    assert infer_config_module(state) == "config_classification_drone"


def test_infer_empty_raises():
    with pytest.raises(ValueError, match="Empty state_dict"):
        infer_config_module({})


def test_tune_threshold_improves_fixed_f1():
    labels = [0, 0, 1, 1]
    scores = [0.10, 0.20, 0.30, 0.40]

    result = tune_threshold_for_f1(labels, scores, default_threshold=0.5)

    assert result["fixed_f1"] == 0.0
    assert result["tuned_threshold"] == 0.3
    assert result["tuned_f1"] == 1.0
