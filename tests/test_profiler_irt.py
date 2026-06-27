"""RED (test-first) tests for promptchain.profiler.irt (NOT YET IMPLEMENTED).

Pins the IRT 3PL / EAP / WLE contract from
specs/013-model-profiler/contracts/profiler-api.md + research.md (D1).
These MUST fail now (irt.py is a stub) and pass once the module is written.
"""
from __future__ import annotations

import math

import pytest

from promptchain.profiler.irt import (
    prob_3pl,
    fisher_information,
    estimate_theta_eap,
    estimate_theta_wle,
    estimate_theta,
)
from promptchain.profiler.item_bank import ItemBank


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------- #
# prob_3pl
# --------------------------------------------------------------------------- #
def test_prob_3pl_at_difficulty_is_closed_form():
    # at theta == b, sigmoid(0) == 0.5, so P == c + (1-c)*0.5
    a, b, c = 1.3, 0.4, 0.2
    P = prob_3pl(b, a, b, c)
    assert P == pytest.approx(c + (1 - c) * 0.5, abs=1e-9)
    assert P == pytest.approx(0.6, abs=1e-9)  # 0.2 + 0.8*0.5


def test_prob_3pl_monotonic_increasing():
    a, b, c = 1.0, 0.0, 0.1
    thetas = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]
    vals = [prob_3pl(t, a, b, c) for t in thetas]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def test_prob_3pl_asymptotes():
    a, b, c = 1.5, 0.0, 0.25
    # theta -> -inf gives the lower asymptote c
    assert prob_3pl(-50.0, a, b, c) == pytest.approx(c, abs=1e-3)
    # theta -> +inf gives 1
    assert prob_3pl(50.0, a, b, c) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# fisher_information
# --------------------------------------------------------------------------- #
def test_fisher_information_hand_computed_value():
    # I = a^2 (P-c)^2 (1-P) / (P (1-c)^2)   at (theta,a,b,c)=(0.5,1.2,0.0,0.1)
    I = fisher_information(0.5, 1.2, 0.0, 0.1)
    assert I == pytest.approx(0.281078462541982, abs=1e-9)
    assert I >= 0.0


def test_fisher_information_reduces_to_a2_P_1minusP_when_c_zero():
    # For c == 0: I = a^2 * P * (1-P)
    a, b, c = 1.5, -0.2, 0.0
    theta = 0.3
    P = _sigmoid(a * (theta - b))
    expected = a * a * P * (1 - P)
    assert fisher_information(theta, a, b, c) == pytest.approx(expected, abs=1e-9)
    assert fisher_information(theta, a, b, c) == pytest.approx(0.4902637359640816, abs=1e-9)


# --------------------------------------------------------------------------- #
# estimate_theta_eap
# --------------------------------------------------------------------------- #
def _bank_items(n=20, seed=0):
    bank = ItemBank.synthetic(dimensions=("reasoning_depth",), n_per_dim=n, seed=seed)
    return bank.by_dimension("reasoning_depth")


def _deterministic_responses(items, true_theta):
    return [1 if prob_3pl(true_theta, it.a, it.b, it.c) >= 0.5 else 0 for it in items]


def test_estimate_theta_eap_recovers_positive_theta():
    items = _bank_items()
    true_theta = 1.0
    responses = _deterministic_responses(items, true_theta)
    theta_hat, posterior_sd = estimate_theta_eap(responses, items)
    assert theta_hat == pytest.approx(true_theta, abs=0.6)
    assert posterior_sd > 0.0


def test_estimate_theta_eap_recovers_negative_theta():
    items = _bank_items()
    true_theta = -1.0
    responses = _deterministic_responses(items, true_theta)
    theta_hat, posterior_sd = estimate_theta_eap(responses, items)
    assert theta_hat == pytest.approx(true_theta, abs=0.6)
    assert posterior_sd > 0.0


# --------------------------------------------------------------------------- #
# estimate_theta_wle  (bounded, finite at the extremes)
# --------------------------------------------------------------------------- #
def test_estimate_theta_wle_all_correct_is_large_and_finite():
    items = _bank_items(n=8)
    responses = [1] * len(items)
    theta_hat, se = estimate_theta_wle(responses, items)
    assert math.isfinite(theta_hat)
    assert theta_hat >= 2.0
    assert se > 0.0


def test_estimate_theta_wle_all_incorrect_is_small_and_finite():
    items = _bank_items(n=8)
    responses = [0] * len(items)
    theta_hat, se = estimate_theta_wle(responses, items)
    assert math.isfinite(theta_hat)
    assert theta_hat <= -2.0
    assert se > 0.0


# --------------------------------------------------------------------------- #
# estimate_theta  (entry point: EAP for mixed, WLE at extremes; Fisher SE)
# --------------------------------------------------------------------------- #
def test_estimate_theta_recovers_known_theta_for_mixed_pattern():
    items = _bank_items()
    true_theta = 1.0
    responses = _deterministic_responses(items, true_theta)
    # sanity: this is a genuinely mixed pattern (not all-correct/all-incorrect)
    assert 0 < sum(responses) < len(responses)
    theta_hat, se = estimate_theta(responses, items)
    assert theta_hat == pytest.approx(true_theta, abs=0.6)
    assert se > 0.0
    assert math.isfinite(se)


def test_estimate_theta_uses_wle_not_eap_for_all_correct():
    items = _bank_items(n=8)
    responses = [1] * len(items)
    entry_theta, entry_se = estimate_theta(responses, items)
    eap_theta, _ = estimate_theta_eap(responses, items)
    # WLE has less prior shrinkage than the N(0,1)-prior EAP on an all-correct pattern
    assert entry_theta > eap_theta
    assert math.isfinite(entry_theta)
    assert entry_se > 0.0
    assert math.isfinite(entry_se)
