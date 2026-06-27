"""Composite profiler score (Ω) and jacket derivation (F2, US2).

Implements the math from research/foundation/architecture/02-model-profiler-math-and-sio.md §2:

    C = sigma(theta_hat)                         capability (IRT)
    K = 1 - ECE                                  calibration confidence
    F = (latency/latency_max)*(cost/cost_ref)    cost+latency penalty
    Omega = 0.7*C*K - 0.3*F

and maps (Omega, theta_hat, SE, C) to a Jacket via the locked band table, with escalation
when  P_route >= alpha  OR  Omega < 0.25  OR  SE > 0.4.
"""
from __future__ import annotations

import math
from typing import Optional

from .jacket import Jacket


def capability(theta_hat: float) -> float:
    """C = sigma(theta_hat) in [0, 1]."""
    return 1.0 / (1.0 + math.exp(-theta_hat))


def calibration_k(ece: float, *, sc: Optional[float] = None, sem_entropy: Optional[float] = None) -> float:
    """Calibration confidence.

    MVP: K = 1 - ECE. If the optional self-consistency (``sc``) and semantic-entropy
    (``sem_entropy``) signals are supplied, use the weighted form
    K = 0.4*(1-ECE) + 0.35*sc + 0.25*exp(-sem_entropy).
    """
    if sc is None and sem_entropy is None:
        return 1.0 - ece
    sc = 0.0 if sc is None else sc
    sem = 0.0 if sem_entropy is None else sem_entropy
    return 0.4 * (1.0 - ece) + 0.35 * sc + 0.25 * math.exp(-sem)


def cost_penalty(latency: float, latency_max: float, cost: float, cost_ref: float) -> float:
    """F = (latency/latency_max) * (cost/cost_ref). Zero refs → 0 (no penalty signal)."""
    if latency_max <= 0 or cost_ref <= 0:
        return 0.0
    return (latency / latency_max) * (cost / cost_ref)


def omega(C: float, K: float, F: float) -> float:
    """Omega = 0.7*C*K - 0.3*F."""
    return 0.7 * C * K - 0.3 * F


# (Omega lower-bound inclusive, tier, budget, mode, compress_at) — highest band first.
_BANDS = [
    (0.55, "lean", 2000, "single-shot", 0.85),
    (0.40, "standard", 4000, "single-shot+retry", 0.75),
    (0.25, "rich", 8000, "heavy-loop", 0.60),
    (float("-inf"), "max-rich", 16000, "escalate", 0.50),
]


def _band(omega_value: float):
    # lean is strictly Omega > 0.55; the rest are inclusive lower bounds.
    if omega_value > 0.55:
        return _BANDS[0]
    for lower, tier, budget, mode, compress in _BANDS[1:]:
        if omega_value >= lower:
            return (lower, tier, budget, mode, compress)
    return _BANDS[-1]


def _role(C: float) -> str:
    if C >= 0.65:
        return "planner"
    if C < 0.40:
        return "executor"
    return "both"


def derive_jacket(
    *,
    omega: float,
    theta: float,
    se: float,
    capability: float,
    degradation_turn: Optional[int] = None,
    p_route: float = 0.0,
    alpha: float = 0.5,
    effective_context: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> Jacket:
    """Map (Omega, theta, SE, C) → a Jacket per the locked band table.

    mode is taken purely from the Omega band; ``escalate`` is a separate flag set when
    P_route >= alpha OR Omega < 0.25 OR SE > 0.4 (the max-rich band's mode is already
    "escalate", so the two agree there).
    """
    _lower, tier, budget, mode, compress = _band(omega)

    # budget may be tightened to the model's effective context window when known
    # (clamp 0.03–0.05 × ctx_eff into [300, budget]); design doc 01.
    if effective_context:
        ctx_budget = int(max(300, min(budget, 0.05 * effective_context)))
        budget = min(budget, ctx_budget) if ctx_budget >= 300 else budget

    escalate = (p_route >= alpha) or (omega < 0.25) or (se > 0.4)

    max_turns = (degradation_turn - 1) if degradation_turn else 10

    return Jacket(
        tier=tier,
        budget_tokens=budget,
        mode=mode,
        spawn_temp=1.0 - capability,
        compress_at=compress,
        max_turns=max_turns,
        role=_role(capability),
        escalate=escalate,
        system_prompt=system_prompt,
    )
