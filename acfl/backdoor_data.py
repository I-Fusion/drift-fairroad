"""
Backdoor trigger stamping aligned with counterfactual ``backdoor_trigger`` probes.

The trigger writes a constant high-magnitude pattern into MsgID/Protocol channels
over the final timesteps of a window. Malicious clients poison local training
data so the global model learns trigger -> benign, while probes remain labeled
attack for counterfactual harm scoring.
"""
from typing import Dict, Optional, Tuple

import numpy as np


def stamp_backdoor_trigger(
    windows: np.ndarray,
    trigger_value: float = 3.0,
    trigger_feature_idxs: Optional[Tuple[int, ...]] = None,
    trigger_timesteps: int = 5,
) -> np.ndarray:
    """Stamp the fixed trigger pattern into window tensors."""
    triggered = np.array(windows, copy=True)
    if triggered.size == 0:
        return triggered

    n_features = triggered.shape[2]
    if trigger_feature_idxs is None:
        trigger_feature_idxs = tuple(
            i for i in (n_features - 2, n_features - 1) if i >= 0
        )
    t = min(trigger_timesteps, triggered.shape[1])
    for f in trigger_feature_idxs:
        triggered[:, -t:, f] = trigger_value
    return triggered


def poison_training_windows(
    X: np.ndarray,
    y: np.ndarray,
    poison_fraction: float = 0.25,
    target_label: float = 0.0,
    trigger_value: float = 3.0,
    trigger_feature_idxs: Optional[Tuple[int, ...]] = None,
    trigger_timesteps: int = 5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Poison benign windows with trigger pattern and relabel to the target class."""
    X_out = np.array(X, copy=True)
    y_out = np.asarray(y).reshape(-1).astype(np.float32, copy=True)

    benign_idx = np.where(y_out < 0.5)[0]
    if len(benign_idx) == 0:
        return X_out, y_out, {
            "num_poisoned": 0,
            "poison_fraction_requested": poison_fraction,
            "target_label": float(target_label),
        }

    n_poison = max(1, int(round(len(benign_idx) * poison_fraction)))
    n_poison = min(n_poison, len(benign_idx))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(benign_idx, size=n_poison, replace=False)

    X_out[chosen] = stamp_backdoor_trigger(
        X_out[chosen],
        trigger_value=trigger_value,
        trigger_feature_idxs=trigger_feature_idxs,
        trigger_timesteps=trigger_timesteps,
    )
    y_out[chosen] = target_label

    return X_out, y_out, {
        "num_poisoned": int(len(chosen)),
        "benign_windows": int(len(benign_idx)),
        "poison_fraction_requested": float(poison_fraction),
        "poison_fraction_actual": float(len(chosen) / len(benign_idx)),
        "target_label": float(target_label),
        "trigger_value": float(trigger_value),
        "trigger_timesteps": int(trigger_timesteps),
    }
