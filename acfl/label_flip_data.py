"""
Label-flipping data poisoning for malicious FL clients.

Flips supervised training labels on a fraction of local windows so the
client optimizes toward the wrong class without modifying outgoing weights.
"""
from typing import Dict, Tuple

import numpy as np


def flip_training_labels(
    X: np.ndarray,
    y: np.ndarray,
    flip_fraction: float = 0.5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Flip labels on a random subset of training windows."""
    X_out = np.array(X, copy=True)
    y_out = np.asarray(y).reshape(-1).astype(np.float32, copy=True)

    n = len(y_out)
    if n == 0:
        return X_out, y_out, {
            "flip_fraction_requested": flip_fraction,
            "num_flipped": 0,
            "total_windows": 0,
        }

    rng = np.random.default_rng(seed)
    n_flip = max(1, int(round(n * flip_fraction))) if flip_fraction > 0 else 0
    n_flip = min(n_flip, n)
    chosen = rng.choice(n, size=n_flip, replace=False)

    for idx in chosen:
        y_out[idx] = 1.0 - y_out[idx]

    return X_out, y_out, {
        "flip_fraction_requested": float(flip_fraction),
        "flip_fraction_actual": float(len(chosen) / n),
        "num_flipped": int(len(chosen)),
        "total_windows": int(n),
        "flipped_indices": chosen.tolist(),
    }
