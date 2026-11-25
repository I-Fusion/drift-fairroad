"""
Federated learning attack models for ACFL evaluation.

Simulates poisoning, backdoor, and Byzantine attacks on client updates
for systematic evaluation against ACFL defenses.
"""
from enum import Enum
from typing import Dict, Optional

import torch


class AttackType(Enum):
    NONE = "none"
    GRADIENT_SCALING = "gradient_scaling"
    LABEL_FLIP = "label_flip"
    BACKDOOR = "backdoor"
    BYZANTINE = "byzantine"
    MODEL_REPLACEMENT = "model_replacement"


class ClientAttackSimulator:
    """Apply learning-plane attacks to client model updates."""

    def __init__(
        self,
        attack_type: AttackType = AttackType.NONE,
        scale_factor: float = 10.0,
        noise_std: float = 1.0,
        backdoor_scale: float = 5.0,
        backdoor_data_poison_only: bool = False,
        attack_seed_base: Optional[int] = None,
    ):
        self.attack_type = attack_type
        self.scale_factor = scale_factor
        self.noise_std = noise_std
        self.backdoor_scale = backdoor_scale
        self.backdoor_data_poison_only = backdoor_data_poison_only
        self.attack_seed_base = attack_seed_base

    def _attack_seed_for_round(self, round_number: int) -> int:
        base = int(self.attack_seed_base) if self.attack_seed_base is not None else 0
        return base + int(round_number)

    def attack_update(
        self,
        client_weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        round_number: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Apply attack transformation to client weights before submission."""
        if self.attack_type == AttackType.NONE:
            return client_weights

        if self.attack_type == AttackType.GRADIENT_SCALING:
            return self._scale_gradient(client_weights, global_weights)

        if self.attack_type == AttackType.BYZANTINE:
            return self._byzantine_noise(client_weights, round_number)

        if self.attack_type == AttackType.BACKDOOR:
            if self.backdoor_data_poison_only:
                return client_weights
            return self._backdoor_trigger(client_weights, global_weights)

        if self.attack_type == AttackType.LABEL_FLIP:
            return client_weights

        if self.attack_type == AttackType.MODEL_REPLACEMENT:
            return self._model_replacement(
                client_weights, global_weights, round_number
            )

        return client_weights

    @staticmethod
    def _clone_non_float(tensor: torch.Tensor) -> torch.Tensor:
        """Preserve integer buffers (e.g. BatchNorm num_batches_tracked)."""
        return tensor.clone()

    def _scale_gradient(
        self,
        client_weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        attacked = {}
        for key in client_weights:
            delta = client_weights[key] - global_weights[key]
            attacked[key] = global_weights[key] + self.scale_factor * delta
        return attacked

    def _byzantine_noise(
        self,
        client_weights: Dict[str, torch.Tensor],
        round_number: int = 0,
    ) -> Dict[str, torch.Tensor]:
        attacked = {}
        gen = torch.Generator()
        gen.manual_seed(self._attack_seed_for_round(round_number))
        for key in sorted(client_weights.keys()):
            tensor = client_weights[key]
            if not tensor.is_floating_point():
                attacked[key] = self._clone_non_float(tensor)
                continue
            noise = torch.randn(
                tensor.shape,
                generator=gen,
                dtype=tensor.dtype,
                device=tensor.device,
            ) * self.noise_std
            attacked[key] = tensor + noise
        return attacked

    def _backdoor_trigger(
        self,
        client_weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Amplify the malicious client's update (optional; data poisoning is primary).

        When ``backdoor_data_poison_only`` is set, local trigger poisoning on the
        client drives the backdoor and outgoing weights are left unchanged.
        """
        attacked = {}
        for key in client_weights:
            delta = client_weights[key] - global_weights[key]
            attacked[key] = global_weights[key] + self.backdoor_scale * delta
        return attacked

    def _model_replacement(
        self,
        client_weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        round_number: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Scaled anti-direction model replacement (distinct from gradient scaling).

        Pushes the global model opposite to the malicious client's benign update
        direction, a common FL model-replacement objective under aggregation.
        """
        attacked = {}
        for key in sorted(client_weights.keys()):
            tensor = client_weights[key]
            global_t = global_weights[key]
            if not tensor.is_floating_point():
                attacked[key] = self._clone_non_float(tensor)
                continue
            delta = tensor - global_t
            attacked[key] = global_t - self.scale_factor * delta
        return attacked
