"""RED (test-first) tests for promptchain.profiler.cat (NOT YET IMPLEMENTED).

Pins the CAT item-selection / SE / stop-rule contract from
specs/013-model-profiler/contracts/profiler-api.md + research.md (D1).
MUST fail now (cat.py is a stub).
"""
from __future__ import annotations

import math

import pytest

from promptchain.profiler.cat import (
    select_next_item,
    standard_error,
    cat_should_stop,
)
from promptchain.profiler.item_bank import ProbeItem, ItemBank


def _ref_fisher(theta: float, a: float, b: float, c: float) -> float:
    """Independent reference for Fisher information (mirrors the locked formula)."""
    P = c + (1 - c) * (1.0 / (1.0 + math.exp(-a * (theta - b))))
    return a * a * (P - c) ** 2 * (1 - P) / (P * (1 - c) ** 2)


def _bank():
    # reasoning_depth has c == 0; build items with KNOWN params and distinct info at theta=0
    items = [
        ProbeItem(item_id="A", dimension="reasoning_depth", a=1.0, b=2.0, c=0.0),
        ProbeItem(item_id="B", dimension="reasoning_depth", a=2.0, b=0.0, c=0.0),
        ProbeItem(item_id="C", dimension="reasoning_depth", a=1.2, b=-2.0, c=0.0),
    ]
    return ItemBank(items=items)


# --------------------------------------------------------------------------- #
# select_next_item
# --------------------------------------------------------------------------- #
def test_select_next_item_picks_max_info_at_theta():
    bank = _bank()
    theta_hat = 0.0
    # compute the expected max-info item independently
    expected = max(bank.items, key=lambda it: _ref_fisher(theta_hat, it.a, it.b, it.c))
    chosen = select_next_item(theta_hat, bank, set())
    assert chosen.item_id == expected.item_id
    assert expected.item_id == "B"  # the a=2,b=0 item dominates at theta=0


def test_select_next_item_skips_administered():
    bank = _bank()
    theta_hat = 0.0
    administered = {"B"}
    chosen = select_next_item(theta_hat, bank, administered)
    assert chosen.item_id not in administered
    # among the remaining {A, C}, the larger-info one is expected
    remaining = [it for it in bank.items if it.item_id not in administered]
    expected = max(remaining, key=lambda it: _ref_fisher(theta_hat, it.a, it.b, it.c))
    assert chosen.item_id == expected.item_id


# --------------------------------------------------------------------------- #
# standard_error
# --------------------------------------------------------------------------- #
def test_standard_error_hand_computed_value():
    items = [
        ProbeItem(item_id="x1", dimension="instruction_following", a=1.2, b=0.0, c=0.1),
        ProbeItem(item_id="x2", dimension="reasoning_depth", a=0.8, b=0.5, c=0.0),
    ]
    se = standard_error(0.4, items)
    assert se == pytest.approx(1.4941315349736553, abs=1e-9)


def test_standard_error_decreases_as_items_added():
    base = [ProbeItem(item_id="i0", dimension="reasoning_depth", a=1.5, b=0.0, c=0.0)]
    more = base + [
        ProbeItem(item_id="i1", dimension="reasoning_depth", a=1.5, b=0.1, c=0.0),
        ProbeItem(item_id="i2", dimension="reasoning_depth", a=1.5, b=-0.1, c=0.0),
    ]
    se_base = standard_error(0.0, base)
    se_more = standard_error(0.0, more)
    assert se_more < se_base


# --------------------------------------------------------------------------- #
# cat_should_stop
# --------------------------------------------------------------------------- #
def test_cat_should_stop_at_se_threshold_after_min_items():
    assert cat_should_stop(0.3, 5) is True


def test_cat_should_not_stop_when_se_above_tau():
    assert cat_should_stop(0.31, 5) is False


def test_cat_should_not_stop_below_min_items():
    assert cat_should_stop(0.1, 4) is False


def test_cat_should_stop_at_max_items_regardless_of_se():
    assert cat_should_stop(5.0, 30) is True
    assert cat_should_stop(0.9, 30) is True
