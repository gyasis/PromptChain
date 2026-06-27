"""US2 tests (FR-017) for promptchain.profiler.composite (test-first, red until W8).

Pins the composite formulas + the Ω→jacket band table from design doc
research/foundation/architecture/02-model-profiler-math-and-sio.md §2.
"""
from __future__ import annotations

import math

import pytest

from promptchain.profiler.composite import (
    capability,
    calibration_k,
    cost_penalty,
    omega,
    derive_jacket,
)
from promptchain.profiler.jacket import Jacket


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------- #
# scalar formulas
# --------------------------------------------------------------------------- #
def test_capability_is_sigmoid():
    assert capability(0.0) == pytest.approx(0.5, abs=1e-9)
    assert capability(1.0) == pytest.approx(_sigmoid(1.0), abs=1e-9)


def test_calibration_k_is_one_minus_ece():
    assert calibration_k(0.2) == pytest.approx(0.8, abs=1e-9)
    assert calibration_k(0.0) == pytest.approx(1.0, abs=1e-9)


def test_cost_penalty_formula():
    # (latency/latency_max) * (cost/cost_ref)
    assert cost_penalty(100.0, 200.0, 2.0, 4.0) == pytest.approx(0.25, abs=1e-9)


def test_omega_formula():
    # Ω = 0.7·C·K − 0.3·F
    C, K, F = 0.8, 0.8, 0.25
    assert omega(C, K, F) == pytest.approx(0.7 * C * K - 0.3 * F, abs=1e-9)
    assert omega(C, K, F) == pytest.approx(0.373, abs=1e-9)


# --------------------------------------------------------------------------- #
# Ω → jacket bands
# --------------------------------------------------------------------------- #
def test_jacket_band_lean():
    j = derive_jacket(omega=0.60, theta=1.6, se=0.15, capability=0.8)
    assert j.tier == "lean"
    assert j.budget_tokens == 2000
    assert j.mode == "single-shot"
    assert j.compress_at == pytest.approx(0.85)
    assert j.spawn_temp == pytest.approx(1 - 0.8)
    assert j.escalate is False


def test_jacket_band_standard():
    j = derive_jacket(omega=0.45, theta=0.5, se=0.25, capability=0.6)
    assert j.tier == "standard"
    assert j.budget_tokens == 4000
    assert j.mode == "single-shot+retry"
    assert j.compress_at == pytest.approx(0.75)


def test_jacket_band_rich():
    j = derive_jacket(omega=0.30, theta=0.0, se=0.30, capability=0.45)
    assert j.tier == "rich"
    assert j.budget_tokens == 8000
    assert j.mode == "heavy-loop"
    assert j.compress_at == pytest.approx(0.60)


def test_jacket_band_max_rich_escalates():
    j = derive_jacket(omega=0.20, theta=-1.2, se=0.35, capability=0.3)
    assert j.tier == "max-rich"
    assert j.budget_tokens == 16000
    assert j.mode == "escalate"
    assert j.compress_at == pytest.approx(0.50)
    assert j.escalate is True  # Ω < 0.25


# --------------------------------------------------------------------------- #
# escalation triggers (P_route≥α OR Ω<0.25 OR SE>0.4) — mode stays the band mode
# --------------------------------------------------------------------------- #
def test_escalate_on_high_se_keeps_band_mode():
    j = derive_jacket(omega=0.50, theta=0.5, se=0.45, capability=0.6)
    assert j.escalate is True               # SE > 0.4
    assert j.mode == "single-shot+retry"    # band mode unchanged (Ω in standard band)


def test_escalate_on_route_probability():
    j = derive_jacket(omega=0.50, theta=0.5, se=0.2, capability=0.6, p_route=0.6, alpha=0.5)
    assert j.escalate is True               # P_route ≥ α


def test_no_escalate_when_all_clear():
    j = derive_jacket(omega=0.50, theta=0.5, se=0.2, capability=0.6, p_route=0.1, alpha=0.5)
    assert j.escalate is False


# --------------------------------------------------------------------------- #
# spawn_temp, role, max_turns
# --------------------------------------------------------------------------- #
def test_spawn_temp_is_one_minus_capability():
    j = derive_jacket(omega=0.45, theta=0.5, se=0.25, capability=0.62)
    assert j.spawn_temp == pytest.approx(1 - 0.62)


@pytest.mark.parametrize("cap,role", [(0.8, "planner"), (0.5, "both"), (0.3, "executor")])
def test_role_by_capability(cap, role):
    j = derive_jacket(omega=0.45, theta=0.5, se=0.25, capability=cap)
    assert j.role == role


def test_max_turns_is_degradation_turn_minus_one():
    j = derive_jacket(omega=0.45, theta=0.5, se=0.25, capability=0.6, degradation_turn=9)
    assert j.max_turns == 8


# --------------------------------------------------------------------------- #
# Jacket round-trip (foundational dataclass exercised here, per plan)
# --------------------------------------------------------------------------- #
def test_jacket_roundtrip():
    j = derive_jacket(omega=0.45, theta=0.5, se=0.25, capability=0.6)
    j2 = Jacket.from_dict(j.to_dict())
    assert j2 == j
