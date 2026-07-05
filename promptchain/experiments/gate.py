"""The graduation GATE — the turnstile of spec-015's technique pipeline.

A technique earns promotion from the experimental bench (EnhancedAgenticStepProcessor) to a production
`AgenticStepProcessor` param ONLY by passing this gate on a HELD-OUT scenario set. The gate is code, not a
claim: it runs the base vs. the treatment over N reps and applies `promptchain.validity` —
`technique_fired` (not a no-op), `no_regression` (never worsens a base-correct item), `above_noise` (delta
beyond run-to-run variance, N>=3), optional `harness_faithful` (a known control reproduces its known
score), and `compare_paired_binary`/`mcnemar` for the paired pass/fail delta. A `GateReport.passed` is True
only if ALL of them pass.

This is the machinery that makes "does technique X actually help?" answerable — the exact discipline whose
absence sank the original enhanced_agentic_step_processor ("10x" claimed, never measured; see spec-015).

Usage:
    rep = run_gate(base_fn, treatment_fn, held_out_scenarios, score_fn=is_correct, reps=3)
    rep.raise_if_failed()          # block graduation unless proven
    if rep.passed: graduate(...)   # attach rep.summary() to the PR (FR-7)
"""
from promptchain.validity import (
    Check, ValiditySuite,
    technique_fired, no_regression, above_noise, harness_faithful,
    compare_paired_binary,
)


def split(scenarios, frac=0.5, seed=0):
    """Deterministic tune/test split so a technique can't be gated on the data it was tuned on (FR-3).
    Returns (tune, held_out). Interleaved by a fixed stride — no RNG (keeps runs reproducible)."""
    ordered = list(scenarios)
    step = max(1, round(1 / max(1e-9, 1 - frac))) if frac < 1 else 1
    held = [s for i, s in enumerate(ordered) if (i + 1 + seed) % 2 == 0][: max(1, int(len(ordered) * (1 - frac)) or 1)]
    heldset = set(map(id, held))
    tune = [s for s in ordered if id(s) not in heldset]
    return tune, held


class GateReport:
    """The result of a graduation gate. `passed` is True only if every validity check passed."""

    def __init__(self, suite, paired, base_scores, treatment_scores):
        self.suite = suite
        self.paired = paired
        self.base_scores = base_scores
        self.treatment_scores = treatment_scores

    @property
    def passed(self):
        return self.suite.ok

    def summary(self):
        bm = sum(self.base_scores) / len(self.base_scores)
        tm = sum(self.treatment_scores) / len(self.treatment_scores)
        head = (f"GATE {'PASS' if self.passed else 'FAIL'} — base {bm:.1f} -> treatment {tm:.1f} "
                f"(verdict: {self.paired['verdict']})")
        return head + "\n" + self.suite.report()

    def raise_if_failed(self):
        """Block graduation: raises AssertionError unless every check passed."""
        self.suite.raise_if_failed()
        return self


def run_gate(base_fn, treatment_fn, scenarios, *, score_fn, reps=3,
             known_fn=None, expected_score=None, alpha=0.05, name="gate",
             max_identical=0.95, max_regression=0.02):
    """Run the graduation gate.

    base_fn / treatment_fn : (scenario) -> output. Called `reps` times per scenario (stochastic in prod).
    score_fn               : (scenario, output) -> bool  (did this output pass?).
    scenarios              : the HELD-OUT set to gate on (use `split` first).
    reps                   : >= 3 for `above_noise` (per Henderson/Dodge/Card).
    known_fn/expected_score: optional positive control for `harness_faithful`.

    Returns a GateReport. `passed` iff technique_fired AND no_regression AND above_noise AND
    (harness_faithful if a control was given) AND a significant paired improvement (McNemar).
    """
    n = len(scenarios)
    base_scores, treat_scores = [], []
    base_out0, treat_out0 = [], []                       # rep-0 outputs, for no-op detection
    per_base = [[] for _ in range(n)]                    # per-scenario correctness across reps
    per_treat = [[] for _ in range(n)]

    for r in range(reps):
        bc = tc = 0
        for i, sc in enumerate(scenarios):
            bo = base_fn(sc)
            to = treatment_fn(sc)
            bok = bool(score_fn(sc, bo))
            tok = bool(score_fn(sc, to))
            if r == 0:
                base_out0.append(bo)
                treat_out0.append(to)
            per_base[i].append(bok)
            per_treat[i].append(tok)
            bc += bok
            tc += tok
        base_scores.append(100.0 * bc / n)
        treat_scores.append(100.0 * tc / n)

    # majority-vote correctness per scenario (stable across stochastic reps)
    base_correct = [sum(x) > len(x) / 2 for x in per_base]
    treat_correct = [sum(x) > len(x) / 2 for x in per_treat]

    suite = ValiditySuite(name)
    suite.add(technique_fired(base_out0, treat_out0, max_identical))
    suite.add(no_regression(base_correct, treat_correct, max_regression))
    suite.add(above_noise(base_scores, treat_scores, min_reps=3))  # hard floor: n>=3 (spec FR-3)
    if known_fn is not None and expected_score is not None:
        known = 100.0 * sum(bool(score_fn(sc, known_fn(sc))) for sc in scenarios) / n
        suite.add(harness_faithful(known, expected_score))

    paired = compare_paired_binary(base_correct, treat_correct, alpha)
    suite.add(Check(
        "mcnemar_improvement",
        bool(paired["significant"]) and paired["treatment_rate"] > paired["base_rate"],
        f"paired pass/fail: {paired['verdict']} (McNemar p={paired['mcnemar']['p']:.3f})",
        paired,
    ))
    return GateReport(suite, paired, base_scores, treat_scores)
