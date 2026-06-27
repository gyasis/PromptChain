"""irt — IRT 3PL math for the Model Profiler (F2).

Pure, deterministic, no I/O. Implements the 3PL probability, Fisher information,
and theta estimators (EAP / WLE / entry point) per
specs/013-model-profiler/research.md (D1) and contracts/profiler-api.md.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit


def prob_3pl(theta: float, a: float, b: float, c: float) -> float:
    """3PL success probability: c + (1-c)*sigmoid(a*(theta-b))."""
    return float(c + (1.0 - c) * expit(a * (theta - b)))


def _prob_3pl_vec(theta: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Vectorized 3PL over a theta grid (internal helper)."""
    return c + (1.0 - c) * expit(a * (theta - b))


def fisher_information(theta: float, a: float, b: float, c: float) -> float:
    """Fisher information I = a^2 (P-c)^2 (1-P) / (P (1-c)^2).

    Reduces to a^2 * P * (1-P) when c == 0. Returns 0.0 safely at degenerate
    boundaries (P at 0 or 1, or c == 1) to avoid division by zero.
    """
    P = prob_3pl(theta, a, b, c)
    denom = P * (1.0 - c) ** 2
    if denom <= 0.0 or P >= 1.0:
        return 0.0
    return float(a * a * (P - c) ** 2 * (1.0 - P) / denom)


def estimate_theta_eap(
    responses: Sequence[int],
    items: Sequence["ProbeItem"],
    *,
    grid: Optional[np.ndarray] = None,
    prior: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[float, float]:
    """Expected a posteriori theta estimate over a grid with a normal prior.

    Returns (theta_hat, posterior_sd).
    """
    if grid is None:
        grid = np.linspace(-4.0, 4.0, 81)
    grid = np.asarray(grid, dtype=float)

    mu, sigma = prior
    # normal prior weight (unnormalized is fine — posterior is renormalized below)
    w = np.exp(-0.5 * ((grid - mu) / sigma) ** 2)

    # log-likelihood over the grid for numerical stability
    log_lik = np.zeros_like(grid)
    for x, it in zip(responses, items):
        P = _prob_3pl_vec(grid, it.a, it.b, it.c)
        P = np.clip(P, 1e-12, 1.0 - 1e-12)
        log_lik += x * np.log(P) + (1 - x) * np.log(1.0 - P)

    log_lik -= log_lik.max()
    posterior = np.exp(log_lik) * w
    total = posterior.sum()
    if total <= 0.0:
        return float(mu), float(sigma)

    posterior /= total
    theta_hat = float(np.sum(grid * posterior))
    var = float(np.sum((grid - theta_hat) ** 2 * posterior))
    posterior_sd = float(np.sqrt(max(var, 1e-12)))
    return theta_hat, posterior_sd


def _fisher_se(theta_hat: float, items: Sequence["ProbeItem"]) -> float:
    """Fisher standard error 1/sqrt(sum I_i) with a guard for ~zero info."""
    total_info = sum(
        fisher_information(theta_hat, it.a, it.b, it.c) for it in items
    )
    if total_info <= 1e-12:
        return 1e6
    return float(1.0 / np.sqrt(total_info))


def estimate_theta_wle(
    responses: Sequence[int],
    items: Sequence["ProbeItem"],
) -> Tuple[float, float]:
    """Bounded maximum-likelihood theta via grid argmax over [-4, 4].

    No prior — the all-correct pattern lands at +4.0 and all-incorrect at -4.0,
    both finite. Returns (theta_hat, fisher_se).
    """
    grid = np.linspace(-4.0, 4.0, 161)
    log_lik = np.zeros_like(grid)
    for x, it in zip(responses, items):
        P = _prob_3pl_vec(grid, it.a, it.b, it.c)
        P = np.clip(P, 1e-12, 1.0 - 1e-12)
        log_lik += x * np.log(P) + (1 - x) * np.log(1.0 - P)

    theta_hat = float(grid[int(np.argmax(log_lik))])
    se = _fisher_se(theta_hat, items)
    return theta_hat, se


def estimate_theta(
    responses: Sequence[int],
    items: Sequence["ProbeItem"],
) -> Tuple[float, float]:
    """Entry point: WLE for all-correct/all-incorrect, EAP otherwise.

    Always returns (theta_hat, fisher_se) with a finite, positive SE.
    """
    resp = list(responses)
    s = sum(resp)
    n = len(resp)
    if n > 0 and (s == 0 or s == n):
        theta_hat, _ = estimate_theta_wle(resp, items)
    else:
        theta_hat, _ = estimate_theta_eap(resp, items)
    se = _fisher_se(theta_hat, items)
    return theta_hat, se


# Late import to keep the type hints readable without a hard load-order cycle.
from promptchain.profiler.item_bank import ProbeItem  # noqa: E402,F401
