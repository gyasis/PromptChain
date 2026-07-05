"""validity_stats — the STATISTICAL INFERENCE layer for validity_suite (issue #40).

validity_suite asserts PROCEDURAL validity (did-it-fire, no-regression, harness-faithful...). This
module answers the other half: **is an observed difference statistically STRONG or WEAK?** It ships the
must-have measures so any comparison we bless is sound. Stdlib-first; scipy is an OPTIONAL lazy import
(only for the rank tests). Each function documents its FORMULA, the CONDITION under which it's correct,
and the MISTAKE it prevents.

THE ONE THAT MATTERS MOST FOR US: our experiments compare the SAME scenarios base-vs-treatment with
pass/fail outcomes = PAIRED BINARY data. The correct test is **McNemar's** (Dietterich 1998), NOT a
t-test on scores (which ignores the pairing and overstates significance). Use `compare_paired_binary`.

Refs: Dietterich (1998) approximate tests for comparing classifiers; Card et al. (2020) With Little
Power (power/Type-M); Cohen (1988) effect sizes d/h; Demsar (2006) comparisons over multiple datasets.
"""
import math
import statistics


# --------------------------------------------------------- helpers (inverse normal for power/MDE)
def _probit(p):
    """Inverse standard-normal CDF (Acklam's rational approximation). z such that Phi(z)=p."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1-p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _chi2_sf_df1(x):
    """Survival function of chi-square with df=1 (the McNemar p-value). = erfc(sqrt(x/2))."""
    return math.erfc(math.sqrt(x / 2.0)) if x > 0 else 1.0


# --------------------------------------------------------- MUST-HAVE 1: McNemar (paired binary)
def mcnemar(base_correct, treatment_correct, continuity=True):
    """McNemar's test — the CORRECT significance test for PAIRED BINARY outcomes (same items, pass/fail
    under base vs treatment). Formula (with continuity correction): chi2 = (|n01-n10|-1)^2 / (n01+n10),
    p = chi2_sf(chi2, df=1). n01 = base wrong & treatment RIGHT (treatment wins), n10 = base right &
    treatment WRONG (treatment loses). PREVENTS: using a t-test/chi-square that ignores the pairing and
    overstates significance (Dietterich 1998)."""
    n01 = sum(1 for b, t in zip(base_correct, treatment_correct) if (not b) and t)   # treatment fixes
    n10 = sum(1 for b, t in zip(base_correct, treatment_correct) if b and (not t))   # treatment breaks
    disc = n01 + n10
    if disc == 0:
        return {"stat": 0.0, "p": 1.0, "n01_treatment_wins": 0, "n10_treatment_breaks": 0,
                "note": "no discordant pairs — the two arms are indistinguishable here"}
    d = abs(n01 - n10) - (1 if continuity else 0)
    chi2 = (d * d) / disc if d > 0 else 0.0
    return {"stat": chi2, "p": _chi2_sf_df1(chi2), "n01_treatment_wins": n01, "n10_treatment_breaks": n10,
            "direction": "treatment better" if n01 > n10 else ("treatment worse" if n10 > n01 else "tie")}


# --------------------------------------------------------- MUST-HAVE 2: Wilson score CI (proportions)
def wilson_ci(k, n, conf=0.95):
    """Wilson score interval for a pass-RATE (k passes of n). The gold standard for binomial proportions.
    PREVENTS: the naive Wald interval (p +/- 1.96*SE) producing impossible bounds (<0 or >1) near 0%/100%."""
    if n == 0:
        return (0.0, 1.0)
    z = _probit(1 - (1 - conf) / 2); ph = k / n; d = 1 + z*z/n
    center = (ph + z*z/(2*n)) / d
    margin = z * math.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / d
    return (max(0.0, center - margin), min(1.0, center + margin))


# --------------------------------------------------------- MUST-HAVE 3: multiple-comparison correction
def holm_bonferroni(pvalues, alpha=0.05):
    """Holm-Bonferroni step-down correction controlling FAMILY-WISE error rate across K comparisons.
    PREVENTS: running K arms/scenarios and calling a chance winner significant (P(>=1 false pos) = 1-(1-a)^K).
    Returns list of {index, p, threshold, reject} in the ORIGINAL order."""
    idx = sorted(range(len(pvalues)), key=lambda i: pvalues[i]); k = len(pvalues)
    out = {i: None for i in range(k)}; still = True
    for rank, i in enumerate(idx):
        thr = alpha / (k - rank)
        rej = still and pvalues[i] <= thr
        if not rej:
            still = False
        out[i] = {"index": i, "p": pvalues[i], "threshold": thr, "reject": rej}
    return [out[i] for i in range(k)]


# --------------------------------------------------------- MUST-HAVE 4: power / minimum detectable effect
def min_detectable_effect(n, p_baseline, alpha=0.05, power=0.8):
    """Minimum Detectable Effect for a two-proportion comparison at sample size n. Approx:
    MDE ~= (z_alpha/2 + z_power) * sqrt(2*p*(1-p)/n). PREVENTS: claiming (or chasing) a delta your eval
    set is too small to see — underpowered evals give Type-M (magnitude) errors (Card et al. 2020)."""
    za = _probit(1 - alpha/2); zb = _probit(power); p = p_baseline
    return (za + zb) * math.sqrt(2 * p * (1 - p) / max(n, 1))


# --------------------------------------------------------- MUST-HAVE 5: effect sizes
def cohens_h(p1, p2):
    """Effect size for two PROPORTIONS: h = 2*asin(sqrt(p1)) - 2*asin(sqrt(p2)). |h|: 0.2 small, 0.5 med,
    0.8 large. PREVENTS: treating a 90->95% gain as equal to 50->55% (h accounts for the ceiling)."""
    return 2*math.asin(math.sqrt(p1)) - 2*math.asin(math.sqrt(p2))


def cohens_d(a, b):
    """Effect size for two CONTINUOUS samples: (mean_a - mean_b)/pooled_sd. |d|: 0.2 small, 0.5 med, 0.8 large."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp = math.sqrt(((na-1)*statistics.variance(a) + (nb-1)*statistics.variance(b)) / (na+nb-2)) or 1e-9
    return (statistics.mean(a) - statistics.mean(b)) / sp


# --------------------------------------------------------- NICE: bootstrap CI, kappa, rank tests
def bootstrap_ci(data, statistic=None, n_resamples=2000, conf=0.95, seed=0):
    """Assumption-free CI via percentile bootstrap. For custom metrics with no closed-form CI."""
    import random
    statistic = statistic or statistics.mean
    rng = random.Random(seed); n = len(data)
    if n == 0:
        return (0.0, 0.0)
    stats_ = sorted(statistic([data[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    lo = stats_[int((1-conf)/2 * n_resamples)]; hi = stats_[int((1+conf)/2 * n_resamples) - 1]
    return (lo, hi)


def cohens_kappa(rater1, rater2):
    """Inter-rater agreement (e.g. two judges / a judge vs gold): kappa = (po - pe)/(1 - pe), correcting
    for chance agreement. PREVENTS: trusting a judge/metric whose agreement is no better than chance."""
    n = len(rater1) or 1; cats = set(rater1) | set(rater2)
    po = sum(1 for a, b in zip(rater1, rater2) if a == b) / n
    pe = sum((sum(1 for x in rater1 if x == c)/n) * (sum(1 for x in rater2 if x == c)/n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def wilcoxon_signed_rank(a, b):
    """Paired NON-parametric test (ordinal/non-normal paired scores). Lazy scipy."""
    from scipy import stats
    r = stats.wilcoxon(a, b)
    return {"stat": float(r.statistic), "p": float(r.pvalue)}


def mann_whitney_u(a, b):
    """Unpaired NON-parametric test (independent groups, non-normal). Lazy scipy."""
    from scipy import stats
    r = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"stat": float(r.statistic), "p": float(r.pvalue)}


# --------------------------------------------------------- the convenience: paired pass/fail, done right
def compare_paired_binary(base_correct, treatment_correct, alpha=0.05):
    """THE function for our data shape: base vs treatment pass/fail on the SAME scenarios. Runs McNemar
    (correct test) + Wilson CIs on each pass-rate + Cohen's h (effect) + a STRONG/WEAK/INCONCLUSIVE verdict."""
    n = len(base_correct)
    bp = sum(bool(x) for x in base_correct); tp = sum(bool(x) for x in treatment_correct)
    mc = mcnemar(base_correct, treatment_correct)
    p1, p2 = bp/n, tp/n
    h = cohens_h(p2, p1)
    sig = mc["p"] < alpha
    if not sig:
        verdict = "INCONCLUSIVE (not significant — likely noise or underpowered)"
    elif abs(h) < 0.2:
        verdict = "WEAK (significant but trivial effect size)"
    else:
        verdict = f"STRONG ({'improvement' if p2>p1 else 'regression'})"
    return {"n": n, "base_rate": p1, "treatment_rate": p2,
            "base_ci": wilson_ci(bp, n), "treatment_ci": wilson_ci(tp, n),
            "mcnemar": mc, "cohens_h": h, "significant": sig, "verdict": verdict,
            "mde_at_this_n": min_detectable_effect(n, p1, alpha)}
