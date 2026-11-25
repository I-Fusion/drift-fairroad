"""Tests for label-flipping data poisoning."""
import os
import sys

import numpy as np
import pytest

ACFL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ACFL_ROOT)

from acfl.label_flip_data import flip_training_labels


def test_flip_training_labels_changes_subset():
    y = np.array([0, 0, 0, 1, 1], dtype=np.float32)
    X = np.zeros((5, 10, 5), dtype=np.float32)
    _, y_out, stats = flip_training_labels(X, y, flip_fraction=0.4, seed=7)
    assert stats["num_flipped"] == 2
    assert not np.array_equal(y, y_out)
    flipped = y != y_out
    assert flipped.sum() == 2
    assert np.all(y_out[flipped] == 1.0 - y[flipped])


def test_flip_training_labels_is_reproducible():
    y = np.array([0, 0, 1, 1, 0, 1], dtype=np.float32)
    X = np.zeros((6, 10, 5), dtype=np.float32)
    _, y1, _ = flip_training_labels(X, y, flip_fraction=0.5, seed=3)
    _, y2, _ = flip_training_labels(X, y, flip_fraction=0.5, seed=3)
    assert np.array_equal(y1, y2)
