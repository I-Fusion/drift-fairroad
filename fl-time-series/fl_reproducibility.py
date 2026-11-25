"""Deterministic seeding for federated learning runs."""
import os
import random
from typing import Optional

import numpy as np
import torch


def get_fl_seed(default: Optional[int] = None) -> Optional[int]:
    raw = os.environ.get("FL_SEED")
    if raw is None or raw == "":
        return default
    return int(raw)


def set_fl_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
