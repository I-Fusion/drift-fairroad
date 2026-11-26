"""Tests for backdoor data poisoning helpers."""
import os
import sys

import numpy as np
import pytest

ACFL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ACFL_ROOT)

from acfl.backdoor_data import poison_training_windows, stamp_backdoor_trigger


def test_stamp_backdoor_trigger_writes_msgid_protocol_channels():
    X = np.zeros((2, 10, 5), dtype=np.float32)
    out = stamp_backdoor_trigger(X, trigger_value=3.0, trigger_timesteps=5)
    assert out.shape == X.shape
    assert np.all(out[:, -5:, -2:] == 3.0)
    assert np.all(out[:, :-5, :] == 0.0)


def test_poison_training_windows_relabels_benign_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 10, 5)).astype(np.float32)
    y = np.array([0] * 16 + [1] * 4, dtype=np.float32)

    X_p, y_p, stats = poison_training_windows(
        X, y, poison_fraction=0.25, target_label=0.0, seed=1
    )

    assert stats["num_poisoned"] == 4
    assert stats["benign_windows"] == 16
    assert np.all(y_p >= 0)
    assert np.sum(y_p >= 0.5) == 4

    poisoned_mask = np.any(X_p != X, axis=(1, 2))
    assert poisoned_mask.sum() == 4
    assert np.all(y_p[poisoned_mask] == 0.0)


def test_poison_training_windows_matches_probe_trigger_pattern():
    X = np.zeros((8, 50, 5), dtype=np.float32)
    y = np.zeros(8, dtype=np.float32)
    X_p, _, stats = poison_training_windows(X, y, poison_fraction=0.5, seed=0)
    poisoned_mask = np.any(X_p != X, axis=(1, 2))
    probe = stamp_backdoor_trigger(X[poisoned_mask])
    assert poisoned_mask.sum() == stats["num_poisoned"]
    assert np.allclose(X_p[poisoned_mask, -5:, -2:], probe[:, -5:, -2:])
