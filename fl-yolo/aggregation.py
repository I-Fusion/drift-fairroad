"""
Aggregation methods for Federated Learning
"""
import torch
from typing import List, Dict
import copy


class FederatedAggregator:
    """
    Handles aggregation of model weights from multiple clients.
    Supports multiple aggregation strategies.
    """

    def __init__(self, strategy="fedavg"):
        """
        Args:
            strategy: Aggregation strategy ('fedavg', 'fedavgm', 'weighted')
        """
        self.strategy = strategy
        self.momentum_buffer = None
        self.beta = 0.9  # Momentum parameter for FedAvgM

    def aggregate(self, client_weights: List[Dict], client_samples: List[int] = None):
        """
        Aggregate client weights based on the selected strategy.

        Args:
            client_weights: List of state dicts from clients
            client_samples: List of number of samples each client trained on

        Returns:
            Aggregated state dict
        """
        if self.strategy == "fedavg":
            return self._fedavg(client_weights)
        elif self.strategy == "fedavgm":
            return self._fedavgm(client_weights)
        elif self.strategy == "weighted":
            if client_samples is None:
                raise ValueError("client_samples required for weighted aggregation")
            return self._weighted_avg(client_weights, client_samples)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.strategy}")

    def _fedavg(self, client_weights: List[Dict]):
        """
        Federated Averaging (FedAvg) - Simple average of all client weights.

        Args:
            client_weights: List of state dicts from clients

        Returns:
            Averaged state dict
        """
        if not client_weights:
            raise ValueError("No client weights to aggregate")

        # Initialize with zeros
        avg_weights = copy.deepcopy(client_weights[0])
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)

        # Sum all client weights
        for client_weight in client_weights:
            for key in avg_weights.keys():
                avg_weights[key] += client_weight[key].float()

        # Average
        num_clients = len(client_weights)
        for key in avg_weights.keys():
            avg_weights[key] = avg_weights[key] / num_clients

        return avg_weights

    def _fedavgm(self, client_weights: List[Dict]):
        """
        Federated Averaging with Momentum (FedAvgM).
        Applies momentum to the aggregated updates.

        Args:
            client_weights: List of state dicts from clients

        Returns:
            Averaged state dict with momentum
        """
        # First get standard FedAvg
        avg_weights = self._fedavg(client_weights)

        # Initialize momentum buffer on first call
        if self.momentum_buffer is None:
            self.momentum_buffer = copy.deepcopy(avg_weights)
            return avg_weights

        # Apply momentum
        for key in avg_weights.keys():
            self.momentum_buffer[key] = (
                self.beta * self.momentum_buffer[key] +
                (1 - self.beta) * avg_weights[key]
            )

        return copy.deepcopy(self.momentum_buffer)

    def _weighted_avg(self, client_weights: List[Dict], client_samples: List[int]):
        """
        Weighted average based on number of samples each client has.

        Args:
            client_weights: List of state dicts from clients
            client_samples: List of number of samples per client

        Returns:
            Weighted averaged state dict
        """
        if not client_weights or not client_samples:
            raise ValueError("No client weights or samples to aggregate")

        if len(client_weights) != len(client_samples):
            raise ValueError("Mismatch between number of weights and samples")

        # Calculate total samples
        total_samples = sum(client_samples)

        # Initialize with zeros
        avg_weights = copy.deepcopy(client_weights[0])
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)

        # Weighted sum
        for client_weight, num_samples in zip(client_weights, client_samples):
            weight = num_samples / total_samples
            for key in avg_weights.keys():
                avg_weights[key] += weight * client_weight[key].float()

        return avg_weights

    def reset_momentum(self):
        """Reset momentum buffer (useful when starting new training)"""
        self.momentum_buffer = None
