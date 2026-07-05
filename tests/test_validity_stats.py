"""Tests for validity_stats (issue #40) — known-value checks per statistical measure."""
import math
from promptchain.utils import validity_stats as vst


def test_mcnemar_significant_improvement():
    # treatment fixes 12, breaks 1 -> clearly significant improvement
    base = [False]*12 + [True]*8
    treat = [True]*12 + [False] + [True]*7
    r = vst.mcnemar(base, treat)
    assert r["n01_treatment_wins"] == 12 and r["n10_treatment_breaks"] == 1
    assert r["p"] < 0.05 and r["direction"] == "treatment better"


def test_mcnemar_no_discordant_is_null():
    r = vst.mcnemar([True, False, True], [True, False, True])
    assert r["p"] == 1.0  # identical -> indistinguishable


def test_wilson_ci_bounds_valid_at_edges():
    lo, hi = vst.wilson_ci(10, 10)      # 100% passes
    assert 0.0 <= lo <= hi <= 1.0 and hi == 1.0 and lo > 0.6   # not the impossible >1 of Wald
    lo0, hi0 = vst.wilson_ci(0, 20)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.3


def test_holm_bonferroni_controls_fwer():
    res = vst.holm_bonferroni([0.001, 0.04, 0.03, 0.9], alpha=0.05)
    # smallest p rejected; the 0.9 never; step-down thresholds a/(k-rank)
    assert res[0]["reject"] is True and res[3]["reject"] is False
    assert res[0]["threshold"] == 0.05/4


def test_min_detectable_effect_shrinks_with_n():
    small = vst.min_detectable_effect(20, 0.5)
    big = vst.min_detectable_effect(2000, 0.5)
    assert small > big > 0   # bigger eval detects smaller effects


def test_cohens_h_ceiling_effect():
    # 90->95 should be a LARGER effect size than 50->55 (ceiling)
    assert abs(vst.cohens_h(0.95, 0.90)) > abs(vst.cohens_h(0.55, 0.50))


def test_cohens_d_known():
    d = vst.cohens_d([2,2,2,2], [0,0,0,0])   # mean diff 2, ~0 within-var
    assert d > 3


def test_cohens_kappa_perfect_and_chance():
    assert vst.cohens_kappa(["a","b","a","b"], ["a","b","a","b"]) == 1.0     # perfect
    k = vst.cohens_kappa(["a","a","b","b"], ["a","b","a","b"])               # chance-ish
    assert -0.6 < k < 0.6


def test_bootstrap_ci_contains_mean():
    lo, hi = vst.bootstrap_ci([10,11,9,10,12,8,10,11], conf=0.95)
    assert lo <= 10 <= hi


def test_compare_paired_binary_end_to_end():
    base = [False]*12 + [True]*8
    treat = [True]*12 + [False] + [True]*7
    r = vst.compare_paired_binary(base, treat)
    assert r["significant"] and "STRONG" in r["verdict"]
    assert r["treatment_rate"] > r["base_rate"]
    assert 0.0 <= r["base_ci"][0] <= r["base_ci"][1] <= 1.0
