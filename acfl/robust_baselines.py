"""
Classical Byzantine-robust FL aggregation baselines for journal evaluation.

Implements coordinate-wise median, trimmed mean, and Krum-style selection
on client model state dicts (same interface as drift-fairroad FederatedAggregator).
"""
from typing import Dict, List

import numpy as np
import torch

from acfl.weight_utils import (
    clone_state_dict,
    compute_update_delta,
    coordinate_wise_median,
)


def coordinate_median_aggregate(
    client_weights: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Coordinate-wise median across client updates."""
    if not client_weights:
        raise ValueError("No client weights to aggregate")
    return coordinate_wise_median(client_weights)


def trimmed_mean_aggregate(
    client_weights: List[Dict[str, torch.Tensor]],
    trim_fraction: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Coordinate-wise trimmed mean across client updates."""
    if not client_weights:
        raise ValueError("No client weights to aggregate")
    n = len(client_weights)
    trim_count = max(0, int(n * trim_fraction))
    if n >= 3 and trim_count == 0:
        trim_count = 1
    if trim_count == 0 or n <= 2 * trim_count:
        return coordinate_wise_median(client_weights)

    keys = client_weights[0].keys()
    result = {}
    for key in keys:
        stacked = torch.stack([cw[key].float() for cw in client_weights])
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[trim_count : n - trim_count]
        result[key] = trimmed.mean(dim=0)
    return result


def krum_peer_scores(
    client_weights: List[Dict[str, torch.Tensor]],
    global_weights: Dict[str, torch.Tensor] | None = None,
    num_byzantine: int = 1,
) -> List[float]:
    """Per-client Krum scores (higher = more geometrically isolated)."""
    if not client_weights:
        return []
    if len(client_weights) == 1:
        return [0.0]

    if global_weights is not None:
        vectors = [
            compute_update_delta(cw, global_weights).astype(np.float64)
            for cw in client_weights
        ]
    else:
        vectors = []
        for cw in client_weights:
            parts = [
                cw[key].detach().cpu().float().numpy().ravel()
                for key in sorted(cw.keys())
            ]
            vectors.append(np.concatenate(parts).astype(np.float64))

    n = len(vectors)
    pairwise = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.sum((vectors[i] - vectors[j]) ** 2))
            pairwise[i, j] = dist
            pairwise[j, i] = dist

    neighbors = max(1, n - num_byzantine - 1)
    scores = []
    for i in range(n):
        dists = sorted(pairwise[i, j] for j in range(n) if j != i)
        scores.append(float(sum(dists[:neighbors])))
    return scores


def krum_aggregate(
    client_weights: List[Dict[str, torch.Tensor]],
    global_weights: Dict[str, torch.Tensor] | None = None,
    num_byzantine: int = 1,
) -> Dict[str, torch.Tensor]:
    """Krum / multi-Krum: select the client with the lowest peer-distance score.

    Uses update deltas when ``global_weights`` is provided; otherwise compares
    full weight vectors. For small swarms (n=3) we score each client by the
    sum of squared distances to its ``n - num_byzantine - 1`` nearest peers,
    falling back to all peers when that count is non-positive.
    """
    if not client_weights:
        raise ValueError("No client weights to aggregate")
    if len(client_weights) == 1:
        return clone_state_dict(client_weights[0])

    scores = krum_peer_scores(client_weights, global_weights, num_byzantine)
    best = int(np.argmin(scores))
    return clone_state_dict(client_weights[best])


BASELINE_STRATEGIES = frozenset({
    "fedavg",
    "coordinate_median",
    "trimmed_mean",
    "krum",
})
