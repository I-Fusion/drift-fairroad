"""
Federated-learning weight serialization tuned for bandwidth/storage limits.

Typical drone / edge constraints:
  - Narrow MAVLink or mesh radio links (prefer smaller payloads)
  - Limited flash (avoid large checkpoint histories)
  - Limited RAM (fp32 state dicts are 2x fp16)
"""
from __future__ import annotations

import gzip
import pickle
from typing import Any, Dict, Tuple

import torch


def _to_fp16_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if torch.is_floating_point(v):
            out[k] = v.detach().cpu().to(torch.float16)
        else:
            out[k] = v.detach().cpu()
    return out


def _to_fp32_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if v.dtype == torch.float16:
            out[k] = v.to(torch.float32)
        else:
            out[k] = v
    return out


def serialize_weights(
    state_dict: Dict[str, torch.Tensor],
    *,
    use_fp16: bool = False,
    compress: bool = False,
) -> bytes:
    """Pickle (optionally fp16 + gzip) a model state dict for upload/download."""
    payload: Dict[str, Any] = {"v": 1, "fp16": use_fp16, "state": state_dict}
    if use_fp16:
        payload["state"] = _to_fp16_state(state_dict)
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    if compress:
        return gzip.compress(raw, compresslevel=6)
    return raw


def deserialize_weights(data: bytes, *, compressed: bool = False) -> Dict[str, torch.Tensor]:
    """Restore a state dict from serialize_weights output."""
    raw = gzip.decompress(data) if compressed else data
    payload = pickle.loads(raw)
    if isinstance(payload, dict) and "state" in payload:
        state = payload["state"]
        if payload.get("fp16"):
            return _to_fp32_state(state)
        return state
    # Legacy: bare state dict
    if isinstance(payload, dict):
        return payload
    raise ValueError("unrecognized weight payload")


def payload_stats(data: bytes) -> Dict[str, float]:
    return {"bytes": len(data), "kb": round(len(data) / 1024.0, 2)}


def is_gzip_payload(data: bytes) -> bool:
    return len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B


def transfer_settings_from_config(config_module) -> Tuple[bool, bool]:
    fp16 = bool(getattr(config_module, "WEIGHT_TRANSFER_FP16", False))
    compress = bool(getattr(config_module, "WEIGHT_TRANSFER_GZIP", False))
    return fp16, compress


def estimate_state_dict_bytes(state_dict: Dict[str, torch.Tensor], fp16: bool = False) -> int:
    """Rough parameter payload size (weights only, no pickle overhead)."""
    total = 0
    for v in state_dict.values():
        if not torch.is_tensor(v):
            continue
        bpe = 2 if fp16 and torch.is_floating_point(v) else v.element_size()
        total += v.numel() * bpe
    return total
