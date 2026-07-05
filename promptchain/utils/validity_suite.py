"""validity_suite — experiment-VALIDITY assertions for PromptChain (issue #38).

The "Great Expectations for experiments": Great Expectations asserts DATA quality and DeepEval/Giskard
assert MODEL quality, but NOTHING asserts that an experiment RUN was scientifically valid. Deep research
(2026-07) confirmed this gap. This module fills it — a small, stdlib-first set of assertions you run
BEFORE trusting an experiment's aggregate score, so you catch the ways a number lies:

  • a technique that never fired (treatment byte-identical to control)      -> technique_fired
  • an intervention that made a correct baseline item WORSE                 -> no_regression
  • a "delta" that's inside the base's own run-to-run variance              -> above_noise
  • a harness so broken a known-good model can't reproduce its known score  -> harness_faithful
  • a silent parse/exception fallback that quietly "passes"                 -> no_silent_defaults
  • an unstable experiment (high variance) that can't support a claim       -> variance_bounded
  • data leakage (shuffled labels still score high)                         -> negative_control
  • a "strength" knob that doesn't move the metric (no-op / broken pipe)    -> monotonic

Grounded in: Breck et al. "ML Test Score" (2017); Henderson et al. "Deep RL that Matters" (2018);
Dodge et al. "Show Your Work" (2019); Card et al. "With Little Power" (2020); Sculley et al. "Hidden
Technical Debt in ML" (2015, CACE); Ribeiro et al. "CheckList" (2020); Masood "decision-grade ablations"
(2026). scipy is an OPTIONAL lazy dependency (only for the exact Welch p-value); everything works without it.

The meta-rule: an experiment number is a HYPOTHESIS to verify, not a conclusion. Run the suite first.
"""
import statistics


class Check:
    """One assertion's outcome. `passed` False = the experiment is INVALID on this dimension."""

    def __init__(self, name, passed, message, detail=None):
        self.name = name
        self.passed = passed
        self.message = message
        self.detail = detail or {}

    def __repr__(self):
        return f"{'PASS' if self.passed else 'FAIL'} [{self.name}] {self.message}"


# ---------------------------------------------------------------- individual assertions
def technique_fired(control_outputs, treatment_outputs, max_identical=0.95):
    """No-op detection (Sculley glue-code / CACE): if the treatment's per-item outputs are (near-)
    identical to control, the intervention DIDN'T FIRE — a NON-result, not evidence it doesn't help.
    Pass = fewer than max_identical of items are byte-identical."""
    n = min(len(control_outputs), len(treatment_outputs)) or 1
    identical = sum(1 for c, t in zip(control_outputs, treatment_outputs) if c == t)
    ratio = identical / n
    return Check("technique_fired", ratio < max_identical,
                 f"{ratio:.0%} of outputs identical to control (fired if < {max_identical:.0%})",
                 {"identical_ratio": ratio})


def no_regression(control_correct, treatment_correct, max_regression=0.02):
    """No-regression (TFX ModelValidator; the 'techniques must not lower a correct base' rule): an
    intervention must not break items the baseline got right. control/treatment_correct = bools per item."""
    n = min(len(control_correct), len(treatment_correct)) or 1
    regressed = sum(1 for c, t in zip(control_correct, treatment_correct) if c and not t)
    rate = regressed / n
    return Check("no_regression", rate <= max_regression,
                 f"{rate:.0%} of baseline-correct items regressed (allowed <= {max_regression:.0%})",
                 {"regressed": regressed, "regression_rate": rate})


def above_noise(control_scores, treatment_scores, min_reps=3, p_threshold=0.05):
    """Above-noise (Henderson/Dodge/Card): NEVER conclude from one run. Needs >= min_reps scores per arm.
    Fails if the delta is within the base's own variance (or, if scipy present, if Welch p >= threshold)."""
    if len(control_scores) < min_reps or len(treatment_scores) < min_reps:
        return Check("above_noise", False,
                     f"only {len(control_scores)}/{len(treatment_scores)} reps — need >= {min_reps} (n=1 proves nothing)")
    cm, tm = statistics.mean(control_scores), statistics.mean(treatment_scores)
    csd = statistics.pstdev(control_scores); tsd = statistics.pstdev(treatment_scores)
    delta = tm - cm
    pooled = (csd + tsd) / 2 or 1e-9
    p = None
    try:
        from scipy import stats  # optional
        p = float(stats.ttest_ind(treatment_scores, control_scores, equal_var=False).pvalue)
    except Exception:
        pass
    if p is not None:
        ok = (p < p_threshold) and (delta > 0)
        msg = f"delta={delta:+.2f} (base {cm:.1f}±{csd:.1f}), Welch p={p:.3f} (sig if < {p_threshold} & delta>0)"
    else:
        ok = delta > pooled                      # stdlib fallback: delta must exceed the pooled spread
        msg = f"delta={delta:+.2f} vs pooled±{pooled:.2f} (base {cm:.1f}±{csd:.1f}) — sig if delta>spread"
    return Check("above_noise", ok, msg, {"delta": delta, "control_std": csd, "p": p})


def harness_faithful(known_score, expected_score, tol=0.05):
    """Harness-faithful (Breck Infra Test 2 / positive control): a KNOWN-GOOD model must reproduce its
    known score through your eval rig. If it doesn't, the RIG is broken — every other number is suspect."""
    dev = abs(known_score - expected_score)
    return Check("harness_faithful", dev <= tol,
                 f"known model scored {known_score:.2f}, expected {expected_score:.2f} (tol {tol}) — "
                 + ("rig OK" if dev <= tol else "RIG BROKEN, distrust all results"),
                 {"deviation": dev})


def no_silent_defaults(outputs, fallback_markers, max_ratio=0.0):
    """No-silent-defaults (Sculley): a parse/exception fallback must not quietly 'pass'. Fails if any
    output equals/contains a fallback marker beyond max_ratio (default 0 = none tolerated)."""
    def hit(o):
        return any((m == o) or (isinstance(o, str) and isinstance(m, str) and m in o) for m in fallback_markers)
    n = len(outputs) or 1
    fired = sum(1 for o in outputs if hit(o))
    ratio = fired / n
    return Check("no_silent_defaults", ratio <= max_ratio,
                 f"{ratio:.0%} of outputs hit a silent fallback/default (allowed <= {max_ratio:.0%})",
                 {"fallback_ratio": ratio})


def variance_bounded(scores, max_std, min_reps=3):
    """Variance-bounded (Card): an unstable experiment can't support a claim. Fails if std across reps
    exceeds max_std (or too few reps)."""
    if len(scores) < min_reps:
        return Check("variance_bounded", False, f"only {len(scores)} reps — need >= {min_reps}")
    sd = statistics.pstdev(scores)
    return Check("variance_bounded", sd <= max_std,
                 f"std={sd:.2f} across {len(scores)} reps (cap {max_std})", {"std": sd})


def negative_control(shuffled_score, chance_level, tol=0.1):
    """Negative control (Adebayo sanity checks): with SHUFFLED labels the score must collapse to chance.
    If it stays high, your harness is LEAKING. Pass = shuffled score is near chance."""
    dev = abs(shuffled_score - chance_level)
    return Check("negative_control", dev <= tol,
                 f"shuffled-label score {shuffled_score:.2f} vs chance {chance_level:.2f} (tol {tol}) — "
                 + ("no leak" if dev <= tol else "DATA LEAK / broken eval"),
                 {"deviation": dev})


def monotonic(knob_values, scores, direction="increasing"):
    """Directional expectation (CheckList DIR / no-op detection): turning a 'strength' knob must move the
    metric. A flat line = the knob never reached the model (no-op) or the pipe is broken. Pass = the
    scores trend in `direction` across the sorted knob values."""
    pairs = sorted(zip(knob_values, scores))
    ys = [s for _, s in pairs]
    if len(set(ys)) <= 1:
        return Check("monotonic", False, f"metric FLAT across knob {[k for k, _ in pairs]} — knob is a no-op / pipe broken")
    ups = sum(1 for a, b in zip(ys, ys[1:]) if b > a)
    downs = sum(1 for a, b in zip(ys, ys[1:]) if b < a)
    ok = (ups >= downs) if direction == "increasing" else (downs >= ups)
    return Check("monotonic", ok, f"metric trend {ys} ({'up' if ups>=downs else 'down'}-leaning), expected {direction}")


# ---------------------------------------------------------------- the runner
class ValiditySuite:
    """Collects Checks (like Great Expectations collects expectations). Run your assertions, then either
    .warn() (print failures, keep going) or .raise_if_failed() (STRICT: block a conclusion). The whole
    point: you may report 'X scored Y' only after the suite passes; otherwise report 'Y BUT check N failed'."""

    def __init__(self, name="experiment"):
        self.name = name
        self.checks = []

    def add(self, check):
        self.checks.append(check)
        return check

    @property
    def failed(self):
        return [c for c in self.checks if not c.passed]

    @property
    def ok(self):
        return not self.failed

    def report(self):
        lines = [f"[validity:{self.name}] {len(self.checks)-len(self.failed)}/{len(self.checks)} checks passed"]
        lines += ["  " + repr(c) for c in self.checks]
        return "\n".join(lines)

    def warn(self):
        if self.failed:
            import warnings
            warnings.warn(f"[validity:{self.name}] {len(self.failed)} assertion(s) FAILED — do NOT trust the "
                          f"result: {[c.name for c in self.failed]}", stacklevel=2)
        return self

    def raise_if_failed(self):
        if self.failed:
            raise AssertionError(f"[validity:{self.name}] experiment INVALID — "
                                 + "; ".join(repr(c) for c in self.failed))
        return self
