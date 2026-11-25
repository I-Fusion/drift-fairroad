"""Parallel Trust Arbitration (PTA) / Dual-Path Counterfactual Aggregation.

At each round both a gated and an ungated expert produce a candidate global
update from the same client payloads. A server-side score that uses only
runtime quantities (probe loss, counterfactual harm, residual influence,
update size, trigger-probe response) then selects or blends them.

The selector must not observe ground-truth attack labels.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Optional


PTA_METHODS = ("hard", "soft", "conservative")
PTA_BRANCHES = ("gated", "ungated")
ORACLE_FEATURE_KEYS = (
    "ground_truth_malicious",
    "attack_label",
    "is_malicious",
    "evaluation_attack_type",
    "malicious_clients",
)


@dataclass
class BranchScore:
    """Runtime score of one counterfactual candidate."""

    name: str
    L_task: float
    H: float
    R: float
    S: float
    R_trigger: float
    J: float
    skipped: bool = False

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ArbitrationDecision:
    """Attack-agnostic choice between gated and ungated candidates."""

    method: str
    selected: str
    eta: float
    tau: float
    default_branch: str
    scores: Dict[str, BranchScore] = field(default_factory=dict)
    last_branch: Optional[str] = None

    def as_dict(self) -> Dict:
        payload = asdict(self)
        payload["scores"] = {
            name: score.as_dict() if isinstance(score, BranchScore) else score
            for name, score in self.scores.items()
        }
        return payload


def candidate_objective(
    L_task: float,
    H: float,
    R: float,
    S: float,
    R_trigger: float = 0.0,
    lambda_H: float = 1.0,
    lambda_R: float = 1.0,
    lambda_S: float = 0.1,
    lambda_trigger: float = 1.0,
    skipped: bool = False,
    lambda_skip: float = 0.0,
) -> float:
    """J = L_task + λ_H H + λ_R R + λ_S S + λ_trig R_trig + λ_skip·1_skip."""
    skip_cost = float(lambda_skip) if skipped else 0.0
    return (
        float(L_task)
        + float(lambda_H) * float(H)
        + float(lambda_R) * float(R)
        + float(lambda_S) * float(S)
        + float(lambda_trigger) * float(R_trigger)
        + skip_cost
    )


def logistic(x: float) -> float:
    x = max(-60.0, min(60.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def mixture_eta(J_gated: float, J_ungated: float, steepness: float = 12.0) -> float:
    """Weight on the gated candidate: σ(k (J_U - J_G)).

    J_G ≪ J_U → η ≈ 1 (gated). J_U ≪ J_G → η ≈ 0 (ungated).
    """
    return logistic(float(steepness) * (float(J_ungated) - float(J_gated)))


def hard_select(
    J_gated: float,
    J_ungated: float,
    default_branch: str = "ungated",
) -> str:
    if float(J_gated) < float(J_ungated):
        return "gated"
    if float(J_ungated) < float(J_gated):
        return "ungated"
    return _valid_branch(default_branch)


def conservative_select(
    J_gated: float,
    J_ungated: float,
    tau: float,
    last_branch: Optional[str] = None,
    default_branch: str = "ungated",
) -> str:
    """Hysteresis around the production default.

    Stay on the current branch unless the other is safer by more than ``tau``.
    """
    current = _valid_branch(last_branch or default_branch)
    margin = float(tau)
    if current == "ungated":
        return "gated" if (float(J_ungated) - float(J_gated)) > margin else "ungated"
    return "ungated" if (float(J_gated) - float(J_ungated)) > margin else "gated"


def arbitrate(
    J_gated: float,
    J_ungated: float,
    method: str = "hard",
    tau: float = 0.05,
    steepness: float = 12.0,
    last_branch: Optional[str] = None,
    default_branch: str = "ungated",
    scores: Optional[Dict[str, BranchScore]] = None,
) -> ArbitrationDecision:
    method = str(method or "hard").strip().lower()
    if method not in PTA_METHODS:
        raise ValueError(f"Unknown PTA method {method!r}; expected {PTA_METHODS}")
    default_branch = _valid_branch(default_branch)

    if method == "hard":
        selected = hard_select(J_gated, J_ungated, default_branch)
        eta = 1.0 if selected == "gated" else 0.0
    elif method == "soft":
        eta = mixture_eta(J_gated, J_ungated, steepness)
        selected = "mixture"
    else:
        selected = conservative_select(
            J_gated, J_ungated, tau, last_branch, default_branch
        )
        eta = 1.0 if selected == "gated" else 0.0

    return ArbitrationDecision(
        method=method,
        selected=selected,
        eta=float(eta),
        tau=float(tau),
        default_branch=default_branch,
        scores=scores or {},
        last_branch=last_branch,
    )


def assert_runtime_features(feature_names: Iterable[str]) -> None:
    """Raise if a selector feature is an attack-label oracle."""
    illegal = [
        name for name in feature_names if str(name) in ORACLE_FEATURE_KEYS
    ]
    if illegal:
        raise ValueError(
            "PTA selector cannot use ground-truth attack labels: "
            + ", ".join(illegal)
        )


def _valid_branch(name: str) -> str:
    value = str(name or "ungated").strip().lower()
    if value not in PTA_BRANCHES:
        raise ValueError(f"Unknown PTA branch {name!r}; expected {PTA_BRANCHES}")
    return value
