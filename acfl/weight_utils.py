"""
Utilities for flattening and comparing federated model updates.
"""
from typing import Dict, List, Tuple

import numpy as np
import torch


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Deep-copy a state dict so probe eval cannot alias live model tensors."""
    return {key: tensor.clone() for key, tensor in state_dict.items()}


def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    """Flatten a PyTorch state dict into a 1-D float32 vector."""
    parts = []
    for key in sorted(state_dict.keys()):
        parts.append(state_dict[key].detach().cpu().float().numpy().ravel())
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def compute_update_delta(
    client_weights: Dict[str, torch.Tensor],
    global_weights: Dict[str, torch.Tensor],
) -> np.ndarray:
    """Compute flattened weight delta (client - global)."""
    delta_parts = []
    for key in sorted(client_weights.keys()):
        diff = (
            client_weights[key].float() - global_weights[key].float()
        ).detach().cpu().numpy().ravel()
        delta_parts.append(diff)
    return np.concatenate(delta_parts) if delta_parts else np.array([], dtype=np.float32)


def apply_delta_to_weights(
    global_weights: Dict[str, torch.Tensor],
    delta_vector: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """Apply a flattened delta vector back to a state dict."""
    updated = {}
    offset = 0
    for key in sorted(global_weights.keys()):
        tensor = global_weights[key]
        size = tensor.numel()
        chunk = delta_vector[offset : offset + size]
        updated[key] = torch.tensor(
            chunk.reshape(tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        offset += size
    return updated


def weighted_average_state_dicts(
    client_weights: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """Compute weighted average of client state dicts."""
    if not client_weights:
        raise ValueError("No client weights to aggregate")
    if len(client_weights) != len(weights):
        raise ValueError("Mismatch between weights and client count")

    total = sum(weights)
    if total <= 0:
        uniform = 1.0 / len(client_weights)
        weights = [uniform] * len(client_weights)
        total = 1.0

    norm_weights = [w / total for w in weights]
    avg = {k: torch.zeros_like(client_weights[0][k], dtype=torch.float32)
           for k in client_weights[0]}

    for cw, w in zip(client_weights, norm_weights):
        for key in avg:
            avg[key] += w * cw[key].float()

    return avg


def blend_state_dicts(
    gated: Dict[str, torch.Tensor],
    ungated: Dict[str, torch.Tensor],
    eta: float,
) -> Dict[str, torch.Tensor]:
    """Convex mix: eta * gated + (1 - eta) * ungated."""
    eta = float(min(1.0, max(0.0, eta)))
    if gated.keys() != ungated.keys():
        raise ValueError("Cannot blend state dicts with different keys")
    mixed = {}
    for key in gated:
        mixed[key] = (
            eta * gated[key].float() + (1.0 - eta) * ungated[key].float()
        ).to(dtype=gated[key].dtype)
    return mixed


def state_dict_l2(
    left: Dict[str, torch.Tensor],
    right: Dict[str, torch.Tensor],
) -> float:
    """Euclidean distance between two state dicts."""
    total = 0.0
    for key in left:
        if key not in right:
            continue
        diff = left[key].float() - right[key].float()
        total += float(diff.pow(2).sum().item())
    return float(total ** 0.5)


def coordinate_wise_median(
    client_weights: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Compute coordinate-wise median across client state dicts."""
    keys = client_weights[0].keys()
    result = {}
    for key in keys:
        stacked = torch.stack([cw[key].float() for cw in client_weights])
        result[key] = torch.median(stacked, dim=0).values
    return result


def coordinate_wise_trimmed_mean(
    client_weights: List[Dict[str, torch.Tensor]],
    trim_fraction: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Compute coordinate-wise trimmed mean across client state dicts."""
    keys = client_weights[0].keys()
    result = {}
    n = len(client_weights)
    trim_count = max(0, int(n * trim_fraction))

    for key in keys:
        stacked = torch.stack([cw[key].float() for cw in client_weights])
        if trim_count > 0 and n > 2 * trim_count:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_count : n - trim_count]
            result[key] = trimmed.mean(dim=0)
        else:
            result[key] = stacked.mean(dim=0)
    return result


def l2_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def norm_ratio_threshold(
    n_clients: int,
    default: float = 2.0,
    two_client: float = 1.5,
) -> float:
    """Norm-ratio decay threshold; lower when only two clients participate."""
    if n_clients <= 2:
        return two_client
    return default


def compute_client_update_norm_stats(
    client_weights: List[Dict[str, torch.Tensor]],
    global_weights: Dict[str, torch.Tensor],
) -> List[Dict[str, float]]:
    """Per-client update norm relative to the round's median update norm."""
    n = len(client_weights)
    if n == 0:
        return []

    deltas = [compute_update_delta(cw, global_weights) for cw in client_weights]
    if not deltas or deltas[0].size == 0:
        return [
            {"update_norm": 0.0, "median_update_norm": 0.0, "norm_ratio": 1.0, "rel_dev": 0.0}
            for _ in range(n)
        ]

    matrix = np.stack(deltas).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1)

    if n == 2:
        median_norm = float(np.median(norms)) + 1e-8
        min_norm = float(min(norms))
        stats = []
        for i in range(2):
            j = 1 - i
            peer_norm = float(norms[j]) + 1e-8
            if min_norm <= 1e-12:
                norm_ratio = float(norms[i]) / median_norm
                ref_norm = median_norm
            else:
                norm_ratio = float(norms[i]) / peer_norm
                ref_norm = peer_norm
            stats.append({
                "update_norm": float(norms[i]),
                "median_update_norm": ref_norm,
                "norm_ratio": norm_ratio,
                "rel_dev": float(np.linalg.norm(matrix[i] - matrix[j]) / ref_norm),
            })
        return stats

    median_delta = np.median(matrix, axis=0)
    median_norm = float(np.median(norms)) + 1e-8
    deviations = np.linalg.norm(matrix - median_delta, axis=1)

    stats = []
    for i in range(n):
        stats.append({
            "update_norm": float(norms[i]),
            "median_update_norm": median_norm,
            "norm_ratio": float(norms[i] / median_norm),
            "rel_dev": float(deviations[i] / median_norm),
        })
    return stats


def robustness_score_from_update_stats(
    stat: Dict[str, float],
    coordinate_deviation_threshold: float,
    norm_ratio_threshold_value: float,
    min_trust_weight: float = 0.05,
) -> float:
    """Soft robustness in (0, 1] from peer-aware update statistics."""
    if stat["norm_ratio"] != stat["norm_ratio"] or stat["rel_dev"] != stat["rel_dev"]:
        return float(min_trust_weight)

    def _decay(value: float, threshold: float) -> float:
        if value <= threshold:
            return 1.0
        return float(np.exp(-(value - threshold)))

    score = min(
        _decay(stat["rel_dev"], coordinate_deviation_threshold),
        _decay(stat["norm_ratio"], norm_ratio_threshold_value),
    )
    return float(max(score, min_trust_weight))
