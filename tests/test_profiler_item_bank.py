"""Foundational tests for the Model Profiler item bank (F2, W2 — test-first).

These FAIL until promptchain/profiler/item_bank.py is implemented (W3).
"""
import math

import pytest

from promptchain.profiler.item_bank import DIMENSIONS, ProbeItem, ItemBank


def test_probe_item_validation_ok():
    it = ProbeItem(item_id="i1", dimension="reasoning_depth", a=1.2, b=0.0, c=0.2)
    assert it.a == 1.2 and it.c == 0.2
    assert it.dimension in DIMENSIONS


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(item_id="x", dimension="reasoning_depth", a=0.0, b=0.0, c=0.1),   # a must be > 0
        dict(item_id="x", dimension="reasoning_depth", a=-1.0, b=0.0, c=0.1),  # a must be > 0
        dict(item_id="x", dimension="reasoning_depth", a=1.0, b=0.0, c=1.0),   # c must be < 1
        dict(item_id="x", dimension="reasoning_depth", a=1.0, b=0.0, c=-0.1),  # c must be >= 0
        dict(item_id="x", dimension="not_a_real_dimension", a=1.0, b=0.0, c=0.1),  # bad dimension
    ],
)
def test_probe_item_validation_rejects_bad(kwargs):
    with pytest.raises((ValueError, AssertionError)):
        ProbeItem(**kwargs)


def test_synthetic_bank_is_deterministic_with_known_params():
    b1 = ItemBank.synthetic(dimensions=("reasoning_depth",), n_per_dim=5, seed=7)
    b2 = ItemBank.synthetic(dimensions=("reasoning_depth",), n_per_dim=5, seed=7)
    assert b1.calibrated is True
    assert len(b1.items) == 5
    # deterministic given the seed: same (a,b,c) sequence
    for x, y in zip(b1.items, b2.items):
        assert (x.a, x.b, x.c) == (y.a, y.b, y.c)
    # known/valid params
    for it in b1.items:
        assert it.a > 0 and 0.0 <= it.c < 1.0


def test_by_dimension_filters():
    bank = ItemBank.synthetic(dimensions=("reasoning_depth", "tool_call_reliability"), n_per_dim=3, seed=1)
    rd = bank.by_dimension("reasoning_depth")
    assert len(rd) == 3
    assert all(it.dimension == "reasoning_depth" for it in rd)


def test_empty_bank_is_uncalibrated():
    bank = ItemBank(items=[])
    assert bank.calibrated is False


def test_filter_quality_drops_nonvariant_and_nondiscriminating_items():
    # 8 calibration subjects ordered weak..strong; a clean latent-ability matrix
    # (Guttman-style thresholds) so the rest-score is a valid ability proxy.
    items = [
        ProbeItem(item_id="f1", dimension="reasoning_depth", a=1.0, b=-1.0, c=0.0),
        ProbeItem(item_id="f2", dimension="reasoning_depth", a=1.0, b=0.0, c=0.0),
        ProbeItem(item_id="f3", dimension="reasoning_depth", a=1.0, b=1.0, c=0.0),
        ProbeItem(item_id="good", dimension="reasoning_depth", a=1.0, b=0.0, c=0.0),
        ProbeItem(item_id="no_var", dimension="reasoning_depth", a=1.0, b=-5.0, c=0.0),
        ProbeItem(item_id="anti", dimension="reasoning_depth", a=1.0, b=0.0, c=0.0),
    ]
    responses = {
        "f1":     [0, 0, 1, 1, 1, 1, 1, 1],  # discriminating filler
        "f2":     [0, 0, 0, 0, 1, 1, 1, 1],  # discriminating filler
        "f3":     [0, 0, 0, 0, 0, 0, 1, 1],  # discriminating filler
        "good":   [0, 0, 0, 1, 1, 1, 1, 1],  # tracks ability -> high point-biserial -> kept
        "no_var": [1, 1, 1, 1, 1, 1, 1, 1],  # zero variance -> dropped
        "anti":   [1, 1, 1, 0, 0, 0, 0, 0],  # reversed -> negative point-biserial -> dropped
    }
    kept_ids = {it.item_id for it in ItemBank.filter_quality(items, responses)}
    assert "good" in kept_ids          # discriminating -> kept
    assert {"f1", "f2", "f3"} <= kept_ids
    assert "no_var" not in kept_ids    # no variance -> dropped
    assert "anti" not in kept_ids      # negative point-biserial -> dropped


def test_filter_quality_drops_near_ceiling_items():
    # 20 subjects: an item correct for 19/20 (acc 0.95) is fine; 39/40 (>0.95) is dropped.
    items = [
        ProbeItem(item_id="disc", dimension="reasoning_depth", a=1.0, b=0.0, c=0.0),
        ProbeItem(item_id="ceiling", dimension="reasoning_depth", a=1.0, b=-4.0, c=0.0),
    ]
    n = 40
    disc = [0] * (n // 2) + [1] * (n // 2)            # discriminating
    ceiling = [0] + [1] * (n - 1)                      # acc = 39/40 = 0.975 > 0.95 -> dropped
    kept_ids = {it.item_id for it in ItemBank.filter_quality(items, {"disc": disc, "ceiling": ceiling})}
    assert "ceiling" not in kept_ids
