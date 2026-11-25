"""Tests for FL attack simulators."""
import os
import sys

import torch
import pytest

ACFL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ACFL_ROOT)
sys.path.insert(0, os.path.join(ACFL_ROOT, "drift-fairroad", "fl-time-series"))

from acfl.attacks import AttackType, ClientAttackSimulator
from models.tiny_packet_cnn import TinyPacketCNN


@pytest.fixture
def tiny_cnn_weights():
    model = TinyPacketCNN(input_size=5, output_size=1, seq_len=50)
    return model.state_dict()


def test_byzantine_preserves_integer_buffers(tiny_cnn_weights):
    sim = ClientAttackSimulator(
        attack_type=AttackType.BYZANTINE,
        noise_std=1.0,
        attack_seed_base=42,
    )
    attacked = sim.attack_update(tiny_cnn_weights, tiny_cnn_weights, round_number=1)

    for key, tensor in tiny_cnn_weights.items():
        if not tensor.is_floating_point():
            assert attacked[key].dtype == tensor.dtype
            assert torch.equal(attacked[key], tensor)
        else:
            assert not torch.equal(attacked[key], tensor)


def test_model_replacement_preserves_integer_buffers(tiny_cnn_weights):
    sim = ClientAttackSimulator(
        attack_type=AttackType.MODEL_REPLACEMENT,
        scale_factor=10.0,
        attack_seed_base=42,
    )
    attacked = sim.attack_update(
        tiny_cnn_weights, tiny_cnn_weights, round_number=1
    )

    for key, tensor in tiny_cnn_weights.items():
        if not tensor.is_floating_point():
            assert attacked[key].dtype == tensor.dtype
            assert torch.equal(attacked[key], tensor)
        else:
            assert attacked[key].dtype.is_floating_point


def test_model_replacement_opposes_benign_update_direction(tiny_cnn_weights):
    global_w = {k: torch.zeros_like(v) for k, v in tiny_cnn_weights.items()}
    local_w = {}
    for k, v in tiny_cnn_weights.items():
        if v.is_floating_point():
            local_w[k] = v + 1.0
        else:
            local_w[k] = v.clone()

    sim = ClientAttackSimulator(
        attack_type=AttackType.MODEL_REPLACEMENT,
        scale_factor=2.0,
    )
    attacked = sim.attack_update(local_w, global_w, round_number=0)
    float_key = next(k for k, t in tiny_cnn_weights.items() if t.is_floating_point())
    # delta = +1, replacement = global - scale*delta => negative direction
    assert attacked[float_key].mean() < 0


def test_label_flip_leaves_weights_unchanged(tiny_cnn_weights):
    sim = ClientAttackSimulator(attack_type=AttackType.LABEL_FLIP)
    attacked = sim.attack_update(tiny_cnn_weights, tiny_cnn_weights, round_number=0)
    for key in tiny_cnn_weights:
        assert torch.equal(attacked[key], tiny_cnn_weights[key])


def test_byzantine_noise_is_deterministic(tiny_cnn_weights):
    sim = ClientAttackSimulator(
        attack_type=AttackType.BYZANTINE,
        noise_std=1.0,
        attack_seed_base=44,
    )
    first = sim.attack_update(tiny_cnn_weights, tiny_cnn_weights, round_number=3)
    second = sim.attack_update(tiny_cnn_weights, tiny_cnn_weights, round_number=3)
    other_round = sim.attack_update(tiny_cnn_weights, tiny_cnn_weights, round_number=4)

    float_key = next(k for k, t in tiny_cnn_weights.items() if t.is_floating_point())
    assert torch.equal(first[float_key], second[float_key])
    assert not torch.equal(first[float_key], other_round[float_key])
