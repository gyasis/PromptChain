"""Tests for validity_suite (issue #38) — each assertion: a PASS case and a FAIL case, plus the runner.
These encode the exact failure modes from the 2026-07 experiment session."""
from promptchain.utils import validity_suite as vs


def test_technique_fired():
    assert vs.technique_fired(["a", "b", "c"], ["a", "x", "y"]).passed          # differs -> fired
    assert not vs.technique_fired(["a", "b"], ["a", "b"]).passed                # byte-identical -> did NOT fire


def test_no_regression():
    assert vs.no_regression([True, True, False], [True, True, True]).passed     # base-correct kept
    assert not vs.no_regression([True, True, True, True], [True, False, False, True]).passed  # broke 2 correct


def test_above_noise():
    # a real, tight lift over enough reps
    assert vs.above_noise([40, 41, 39, 40, 40], [55, 56, 54, 55, 55]).passed
    # a "delta" inside the base's own swing
    assert not vs.above_noise([40, 60, 45, 55, 50], [52, 48, 58, 42, 50]).passed
    # n=1 must fail outright (the foundational rule)
    assert not vs.above_noise([84], [79]).passed


def test_harness_faithful():
    assert vs.harness_faithful(0.69, 0.70).passed                              # reproduces known score
    assert not vs.harness_faithful(0.40, 0.70).passed                          # glm 40 vs known 70 -> rig broken


def test_no_silent_defaults():
    assert vs.no_silent_defaults(["ok", "fine"], ["ERROR", "(unparsed)"]).passed
    assert not vs.no_silent_defaults(["ok", "(unparsed)"], ["ERROR", "(unparsed)"]).passed  # silent default hit


def test_variance_bounded():
    assert vs.variance_bounded([50, 51, 49, 50], max_std=2).passed
    assert not vs.variance_bounded([40, 70, 50, 60], max_std=2).passed         # ±31.9-style instability


def test_negative_control():
    assert vs.negative_control(0.52, chance_level=0.5).passed                   # shuffled -> ~chance, no leak
    assert not vs.negative_control(0.88, chance_level=0.5).passed               # shuffled still high -> leakage


def test_monotonic():
    assert vs.monotonic([1, 2, 3], [40, 55, 62]).passed                        # knob moves the metric
    assert not vs.monotonic([0, 1, 2], [40, 40, 40]).passed                    # FLAT -> knob is a no-op


def test_suite_runner_warn_and_raise():
    s = vs.ValiditySuite("moe")
    s.add(vs.technique_fired(["a", "b"], ["a", "b"]))     # fails
    s.add(vs.harness_faithful(0.70, 0.70))               # passes
    assert not s.ok and len(s.failed) == 1
    assert "1/2 checks passed" in s.report()
    s.warn()                                             # should not raise
    try:
        s.raise_if_failed(); assert False
    except AssertionError:
        pass


def test_suite_all_pass():
    s = vs.ValiditySuite("clean")
    s.add(vs.technique_fired(["a"], ["b"]))
    s.add(vs.no_regression([True], [True]))
    assert s.ok
    s.raise_if_failed()  # no raise
