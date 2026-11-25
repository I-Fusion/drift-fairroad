"""Tests for backdoor trigger probe evaluation."""
import os
import sys

import numpy as np
import torch
import pytest

ACFL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FL_TS_DIR = os.path.join(ACFL_ROOT, "drift-fairroad", "fl-time-series")
sys.path.insert(0, ACFL_ROOT)
sys.path.insert(0, FL_TS_DIR)

from acfl.backdoor_data import stamp_backdoor_trigger
from acfl.backdoor_eval import evaluate_backdoor_trigger
from models.tiny_packet_cnn import TinyPacketCNN


@pytest.fixture
def trigger_probe_file(tmp_path):
    benign = np.zeros((16, 50, 5), dtype=np.float32)
    triggered = stamp_backdoor_trigger(benign)
    probe_data = {
        "backdoor_trigger": (
            torch.tensor(triggered, dtype=torch.float32),
            torch.ones(16, dtype=torch.float32),
        ),
    }
    path = tmp_path / "probes.pt"
    torch.save({"probe_data": probe_data}, path)
    return str(path)


@pytest.fixture
def checkpoint_paths(tmp_path):
    model = TinyPacketCNN(input_size=5, output_size=1, seq_len=50)
    ckpt = {
        "round": 3,
        "model_state_dict": model.state_dict(),
        "learning_mode": "classification",
        "input_size": 5,
        "output_size": 1,
    }
    clean_path = tmp_path / "server_round_3.pt"
    torch.save(ckpt, clean_path)

    backdoored = TinyPacketCNN(input_size=5, output_size=1, seq_len=50)
    with torch.no_grad():
        backdoored.head[-1].bias.fill_(-8.0)
    ckpt["model_state_dict"] = backdoored.state_dict()
    weak_path = tmp_path / "server_round_4.pt"
    torch.save(ckpt, weak_path)
    return str(clean_path), str(weak_path)


def test_evaluate_backdoor_trigger_reports_success_rate(
    checkpoint_paths, trigger_probe_file
):
    clean_path, weak_path = checkpoint_paths
    clean = evaluate_backdoor_trigger(
        clean_path,
        trigger_probe_file,
        config_module="config_classification_drone",
        fl_ts_dir=FL_TS_DIR,
    )
    weak = evaluate_backdoor_trigger(
        weak_path,
        trigger_probe_file,
        config_module="config_classification_drone",
        fl_ts_dir=FL_TS_DIR,
    )

    assert clean["backdoor_trigger_samples"] == 16
    assert 0.0 <= clean["backdoor_success_rate"] <= 1.0
    assert weak["backdoor_mean_attack_prob"] < clean["backdoor_mean_attack_prob"]
    assert weak["backdoor_mean_attack_prob"] < clean["backdoor_mean_attack_prob"]
