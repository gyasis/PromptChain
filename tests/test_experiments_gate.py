"""Gate tests (spec-015, T002). The gate must ACCEPT a genuinely-better technique and REJECT the three
ways a technique lies: a no-op, a regression, and a fires-but-doesn't-help (in-noise) change."""
import warnings
from promptchain.experiments import run_gate, split

N = 10
SCEN = list(range(N))


def _agent(correct_by_rep, tag=""):
    """Deterministic stand-in for a stochastic step: correctness varies per rep, output is
    stable per (scenario, correctness, tag). Called N times per rep, in scenario order."""
    st = {"c": 0}

    def fn(scenario):
        rep = (st["c"] // N) % len(correct_by_rep)
        st["c"] += 1
        ok = scenario in correct_by_rep[rep]
        return f"{'PASS' if ok else 'FAIL'}:{scenario}{tag}"
    return fn


def score(scenario, output):
    return output.startswith("PASS")


def test_gate_accepts_a_genuinely_better_technique():
    # base ~20%, treatment ~90% — fixes 7 items (enough for McNemar significance), never regresses
    base = _agent([{0, 1}, {0, 1}, {0, 1, 2}])
    treat = _agent([{0, 1, 2, 3, 4, 5, 6, 7, 8}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7, 8}])
    rep = run_gate(base, treat, SCEN, score_fn=score, reps=3)
    assert rep.passed, rep.summary()
    rep.raise_if_failed()  # must not raise


def test_gate_rejects_a_no_op_negative_control():
    sets = [{0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 2, 3}]
    base = _agent(sets)
    treat = _agent(sets)                                               # identical -> did not fire
    rep = run_gate(base, treat, SCEN, score_fn=score, reps=3)
    assert not rep.passed
    assert any(c.name == "technique_fired" and not c.passed for c in rep.suite.checks)


def test_gate_rejects_a_regression():
    base = _agent([{0, 1, 2, 3, 4}] * 3)                              # 50%, {0..4} correct
    treat = _agent([{2, 3, 4, 5, 6}] * 3)                            # fixes 5,6 but BREAKS 0,1
    rep = run_gate(base, treat, SCEN, score_fn=score, reps=3)
    assert not rep.passed
    assert any(c.name == "no_regression" and not c.passed for c in rep.suite.checks)


def test_gate_rejects_fires_but_no_improvement_in_noise():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                              # zero-variance t-test edge
        base = _agent([{0, 1, 2, 3, 4}] * 3)                         # 50%
        treat = _agent([{0, 1, 2, 3, 4}] * 3, tag=":v2")            # SAME correctness, DIFFERENT output
        rep = run_gate(base, treat, SCEN, score_fn=score, reps=3)
    assert not rep.passed
    # it fired (outputs differ) and didn't regress, but the delta is noise -> above_noise / mcnemar reject
    assert any(c.name == "technique_fired" and c.passed for c in rep.suite.checks)
    assert any(c.name in ("above_noise", "mcnemar_improvement") and not c.passed for c in rep.suite.checks)


def test_split_is_deterministic_and_holds_out():
    tune, held = split(SCEN, frac=0.5, seed=0)
    assert held and tune
    assert split(SCEN, frac=0.5, seed=0)[1] == held   # reproducible


def test_gate_requires_min_reps():
    base = _agent([{0, 1}])
    treat = _agent([{0, 1, 2, 3}])
    rep = run_gate(base, treat, SCEN, score_fn=score, reps=1)         # < 3 reps
    assert not rep.passed
    assert any(c.name == "above_noise" and not c.passed for c in rep.suite.checks)
