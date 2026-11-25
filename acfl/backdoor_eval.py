"""
Evaluate global models on ``backdoor_trigger`` counterfactual probes.

Reports how often a backdoored global model misclassifies stamped trigger
windows as benign — the proposal's backdoor success metric.
"""
from __future__ import annotations

import glob
import importlib
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _resolve_checkpoint_path(checkpoint_dir: str) -> Optional[str]:
    """Pick the best server checkpoint under a scenario directory."""
    if not checkpoint_dir:
        return None
    if os.path.isfile(checkpoint_dir) and checkpoint_dir.endswith(".pt"):
        return checkpoint_dir

    final_ckpt = os.path.join(checkpoint_dir, "server_checkpoint.pt")
    if os.path.isfile(final_ckpt):
        return final_ckpt

    patterns = [
        os.path.join(checkpoint_dir, "server_round_*.pt"),
        os.path.join(checkpoint_dir, "fl_checkpoints", "server_round_*.pt"),
    ]
    candidates = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                rnd = int(os.path.basename(path).split("_")[-1].split(".")[0])
            except ValueError:
                continue
            candidates.append((rnd, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _load_probe_tensors(probe_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(probe_file, map_location="cpu", weights_only=False)
    probe_data = payload.get("probe_data", payload)
    if "backdoor_trigger" not in probe_data:
        raise KeyError(
            f"Probe file missing backdoor_trigger scenario: {probe_file}"
        )
    inputs, labels = probe_data["backdoor_trigger"]
    return inputs.float(), labels.float().reshape(-1)


def _build_model(
    checkpoint: Dict,
    config_module: str,
    fl_ts_dir: str,
) -> nn.Module:
    if fl_ts_dir not in sys.path:
        sys.path.insert(0, fl_ts_dir)

    config = importlib.import_module(config_module)
    model_module = importlib.import_module(config.MODEL_PATH)
    model_class = getattr(model_module, config.MODEL_CLASS)

    model_config = dict(config.MODEL_CONFIG)
    model_config["input_size"] = int(
        checkpoint.get("input_size", model_config.get("input_size", 5))
    )
    model_config["output_size"] = int(
        checkpoint.get("output_size", model_config.get("output_size", 1))
    )
    if "seq_len" not in model_config:
        model_config["seq_len"] = int(getattr(config, "WINDOW_SIZE", 50))

    model = model_class(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_backdoor_trigger(
    checkpoint_path: str,
    probe_file: str,
    config_module: str = "config_classification_drone",
    fl_ts_dir: Optional[str] = None,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Score a global checkpoint on stamped trigger probes.

    Returns:
        backdoor_success_rate: fraction of attack-labeled triggers called benign
        backdoor_trigger_accuracy: fraction correctly flagged as attack
        backdoor_mean_attack_prob: mean predicted P(attack) on triggers
    """
    if fl_ts_dir is None:
        fl_ts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "drift-fairroad",
            "fl-time-series",
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = _build_model(checkpoint, config_module, fl_ts_dir)
    inputs, labels = _load_probe_tensors(probe_file)

    attack_probs = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(-1)
            attack_probs.append(probs.cpu())

    attack_prob = torch.cat(attack_probs).numpy()
    predicted_attack = attack_prob >= threshold
    true_attack = labels.numpy() >= 0.5

    n = len(true_attack)
    if n == 0:
        return {
            "backdoor_success_rate": None,
            "backdoor_trigger_accuracy": None,
            "backdoor_mean_attack_prob": None,
            "backdoor_trigger_samples": 0,
        }

    misclassified_as_benign = (~predicted_attack) & true_attack
    success = misclassified_as_benign
    correct_attack = predicted_attack & true_attack

    return {
        "backdoor_success_rate": float(success.sum() / true_attack.sum()),
        "backdoor_trigger_accuracy": float(correct_attack.sum() / n),
        "backdoor_mean_attack_prob": float(attack_prob.mean()),
        "backdoor_trigger_samples": int(n),
        "checkpoint_round": int(checkpoint.get("round", -1)),
        "checkpoint_path": checkpoint_path,
    }


def evaluate_backdoor_trigger_from_dir(
    checkpoint_dir: str,
    probe_file: str,
    config_module: str = "config_classification_drone",
    fl_ts_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Resolve the latest server checkpoint in a scenario directory."""
    path = _resolve_checkpoint_path(checkpoint_dir)
    if not path:
        return {
            "backdoor_success_rate": None,
            "backdoor_trigger_accuracy": None,
            "backdoor_mean_attack_prob": None,
            "backdoor_trigger_samples": 0,
            "error": "no_server_checkpoint",
        }
    metrics = evaluate_backdoor_trigger(
        path, probe_file, config_module=config_module, fl_ts_dir=fl_ts_dir
    )
    return metrics
