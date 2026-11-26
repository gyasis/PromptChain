"""Offline tests for promptchain.utils.model_router (stubs only; no model, no network)."""
import ast
import asyncio
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

from promptchain import PromptChain
from promptchain.profiler.irt import prob_3pl
from promptchain.profiler.jacket import CapabilityProfile, SkillEstimate
from promptchain.profiler.store import ProfileStore
from promptchain.utils.decision_step import DecisionError
from promptchain.utils.model_router import (
    Difficulty,
    DifficultyLink,
    ModelRouter,
    RoutingOutcome,
    decision_difficulty,
    read_outcomes,
    route_model,
    write_outcomes,
)

MODELS = ["small", "mid", "big"]
ABILITIES = {"small": -1.0, "mid": 0.0, "big": 1.5}
# P(pass) = sigmoid(2 * (theta - (0 + 2 * d)))
LINK = DifficultyLink(a=2.0, alpha=0.0, beta=2.0, n=100)


def run(coro):
    return asyncio.run(coro)


def fixed(value):
    async def _fn(task):
        return Difficulty(value, source="stub")
    return _fn


class StubBackend:
    name = "stub"
    model = "stub-model"

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    async def decide(self, state, questions):
        self.calls.append((state, questions))
        return {"model": self.model, "answers": self.answers}


LEVELS = ["trivial", "easy", "medium", "hard", "expert"]


def router(**kw):
    kw.setdefault("threshold", 0.7)
    kw.setdefault("timeout_s", 1.0)
    kw.setdefault("abilities", ABILITIES)
    kw.setdefault("link", LINK)
    kw.setdefault("difficulty", fixed(0.1))
    return ModelRouter(MODELS, **kw)


# ---- decision_difficulty


def test_decision_difficulty_wire_shape_and_probability_mapping():
    stub = StubBackend({"difficulty": {
        "score": 3, "confidence": 0.8,
        "probabilities": {"easy": 0.5, "hard": 0.5},
    }})
    fn = decision_difficulty(stub, instructions="How hard?", levels=LEVELS,
                             prepare_state=lambda t: t[:5])
    d = run(fn("abcdefghij"))
    state, qs = stub.calls[0]
    assert state == "abcde"
    assert qs == {"difficulty": {"type": "score", "instructions": "How hard?", "criteria": LEVELS}}
    assert d.value == pytest.approx(((1 + 3) / 2) / 4)
    assert d.source == "decision:stub" and d.model == "stub-model"
    assert d.distribution == {"easy": 0.5, "hard": 0.5}


def test_decision_difficulty_score_only_and_clipping():
    stub = StubBackend({"difficulty": {"score": 9, "confidence": 0.9}})
    d = run(decision_difficulty(stub, instructions="x", levels=LEVELS)("t"))
    assert d.value == 1.0
    stub = StubBackend({"difficulty": {"score": 1, "confidence": 0.9}})
    assert run(decision_difficulty(stub, instructions="x", levels=LEVELS)("t")).value == 0.25


def test_decision_difficulty_rejects_bad_levels_and_missing_backend():
    with pytest.raises(DecisionError):
        decision_difficulty(StubBackend({}), instructions="x", levels=["only"])
    with pytest.raises(DecisionError):
        decision_difficulty(None, instructions="x", levels=LEVELS)


# ---- plan: choice and order


def test_cheapest_passing_model_and_suffix_order():
    plan = run(router(difficulty=fixed(0.1)).plan("t"))
    # small: sigmoid(2*(-1-0.2)) ~ 0.08; mid: sigmoid(2*(0-0.2)) ~ 0.40; big: sigmoid(2*(1.5-0.2)) ~ 0.93
    assert plan.chosen == "big" and plan.order == ["big"] and plan.degraded is None
    plan = run(router(difficulty=fixed(0.0), abilities={"small": 1.0, "mid": 2.0, "big": 3.0}).plan("t"))
    assert plan.chosen == "small" and plan.order == ["small", "mid", "big"]
    plan = run(router(difficulty=fixed(0.0), abilities={"small": -3.0, "mid": 1.0, "big": 2.0}).plan("t"))
    assert plan.chosen == "mid" and plan.order == ["mid", "big"]
    assert plan.p_success["small"] < 0.7 <= plan.p_success["mid"]
    assert plan.to_dict()["difficulty"]["value"] == 0.0
    json.dumps(plan.to_dict())


def test_nothing_passes_uses_last_with_reason():
    plan = run(router(difficulty=fixed(1.0)).plan("t"))
    assert plan.chosen == "big" and plan.degraded is None
    assert "no model reaches" in plan.reason


def test_no_theta_and_no_link_use_last_model_with_reason(caplog):
    plan = run(router(abilities={}).plan("t"))
    assert plan.chosen == "big" and "no ability" in plan.reason
    plan = run(router(link=None).plan("t"))
    assert plan.chosen == "big" and plan.order == ["big"]
    assert "no link" in plan.degraded and "no link" in plan.reason
    plan = run(router(difficulty=None).plan("t"))
    assert plan.chosen == "big" and plan.degraded and plan.difficulty is None


def test_model_without_theta_is_skipped_not_chosen():
    plan = run(router(difficulty=fixed(0.0), abilities={"mid": 2.0, "big": 3.0}).plan("t"))
    assert plan.chosen == "mid" and plan.p_success["small"] is None


# ---- never block


def test_slow_difficulty_times_out_with_one_warning(caplog):
    async def slow(task):
        await asyncio.sleep(5)

    r = router(difficulty=slow, timeout_s=0.05)
    with caplog.at_level(logging.WARNING, logger="promptchain.utils.model_router"):
        t0 = time.monotonic()
        plan = run(r.plan("t"))
        elapsed = time.monotonic() - t0
    assert elapsed < 0.05 + 0.5
    assert plan.chosen == "big" and "timed out" in plan.degraded
    warnings = [x for x in caplog.records if x.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_difficulty_errors_fall_back_without_raising(caplog):
    async def boom(task):
        raise DecisionError("backend down")

    async def wrong_type(task):
        return 0.5

    for fn in (boom, wrong_type):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="promptchain.utils.model_router"):
            plan = run(router(difficulty=fn).plan("t"))
        assert plan.chosen == "big" and "failed" in plan.degraded
        assert len([x for x in caplog.records if x.levelno == logging.WARNING]) == 1


def test_backend_decision_error_through_decision_difficulty(caplog):
    stub = StubBackend({"difficulty": {"error": "no model"}})
    fn = decision_difficulty(stub, instructions="x", levels=LEVELS)
    plan = run(router(difficulty=fn).plan("t"))
    assert plan.chosen == "big" and "DecisionError" in plan.degraded


def test_constructor_validation():
    with pytest.raises(ValueError):
        ModelRouter([], threshold=0.5, timeout_s=1)
    with pytest.raises(ValueError):
        ModelRouter(["a", "a"], threshold=0.5, timeout_s=1)
    with pytest.raises(ValueError):
        ModelRouter(["a"], threshold=0.0, timeout_s=1)
    with pytest.raises(ValueError):
        ModelRouter(["a"], threshold=0.5, timeout_s=0)
    with pytest.raises(ValueError):
        Difficulty(1.5, "x")


# ---- fit


def _synthetic(n, a, alpha, beta, seed):
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        m = rnd.choice(MODELS)
        theta = ABILITIES[m]
        d = rnd.random()
        p = prob_3pl(theta, a, alpha + beta * d, 0.0)
        out.append(RoutingOutcome(f"t{i}", "task", d, "stub", m, theta, rnd.random() < p, "explore"))
    return out


def test_fit_recovers_seeded_parameters():
    link = DifficultyLink.fit(_synthetic(5000, a=1.8, alpha=-0.3, beta=2.4, seed=7))
    assert link.n == 5000
    assert link.a == pytest.approx(1.8, abs=0.25)
    assert link.alpha == pytest.approx(-0.3, abs=0.2)
    assert link.beta == pytest.approx(2.4, abs=0.3)
    assert 0.5 < link.auroc <= 1.0
    assert link.p_success(1.5, 0.1) > link.p_success(-1.0, 0.1)


def test_fit_rejects_bad_data():
    rows = _synthetic(200, 1.8, 0.0, 2.0, seed=1)
    for r in rows:
        r.passed = True
    with pytest.raises(ValueError, match="same result"):
        DifficultyLink.fit(rows)
    with pytest.raises(ValueError, match="at least 3"):
        DifficultyLink.fit(rows[:2])
    # stronger models fail more: a <= 0
    rnd = random.Random(3)
    inverted = [RoutingOutcome(f"{i}", "t", rnd.random(), "s", m, ABILITIES[m],
                               rnd.random() < (0.9 if m == "small" else 0.1), "x")
                for i in range(600) for m in MODELS]
    with pytest.raises(ValueError, match="not positive"):
        DifficultyLink.fit(inverted)
    one_theta = [RoutingOutcome(f"{i}", "t", i / 40, "s", "mid", 0.0, i % 2 == 0, "x") for i in range(40)]
    with pytest.raises(ValueError, match="distinct"):
        DifficultyLink.fit(one_theta)


# ---- profile thetas


def _profile(model, thetas, low=False):
    skills = {d: SkillEstimate(dimension=d, theta=t, se=0.3, low_precision=low)
              for d, t in thetas.items()}
    return CapabilityProfile(model_id=model, skills=skills)


def test_theta_from_store_dimension_and_mean(tmp_path):
    path = tmp_path / "profiles.json"
    store = ProfileStore(path)
    store.upsert(_profile("small", {"reasoning_depth": -1.0, "instruction_following": -3.0}))
    store.upsert(_profile("mid", {"reasoning_depth": 2.0, "instruction_following": 0.0}))
    store.upsert(_profile("big", {"reasoning_depth": 3.0, "instruction_following": 1.0}, low=True))

    r = ModelRouter(MODELS, threshold=0.5, timeout_s=1, store_path=path, dimension="reasoning_depth")
    assert [r._ability(m, r._read_store())[0] for m in MODELS] == [-1.0, 2.0, 3.0]
    r = ModelRouter(MODELS, threshold=0.5, timeout_s=1, store_path=path)
    assert [r._ability(m, r._read_store())[0] for m in MODELS] == [-2.0, 1.0, 2.0]
    r = ModelRouter(MODELS, threshold=0.5, timeout_s=1, store_path=path, link=LINK,
                    difficulty=fixed(0.0), abilities={"small": 5.0})
    assert r._ability("small", r._read_store())[0] == 5.0  # explicit ability wins
    plan = run(r.plan("t"))
    assert plan.chosen == "small"
    plan = run(ModelRouter(["big"], threshold=0.99, timeout_s=1, store_path=path,
                           link=LINK, difficulty=fixed(0.9)).plan("t"))
    assert "low_precision" in plan.reason


def test_missing_store_gives_no_theta(tmp_path):
    r = ModelRouter(MODELS, threshold=0.5, timeout_s=1, store_path=tmp_path / "none.json",
                    link=LINK, difficulty=fixed(0.0))
    plan = run(r.plan("t"))
    assert plan.chosen == "big" and "no ability" in plan.reason


# ---- outcomes JSONL


def test_outcomes_round_trip_and_append(tmp_path):
    rows = _synthetic(5, 1.8, 0.0, 2.0, seed=2)
    path = tmp_path / "sub" / "o.jsonl"
    assert write_outcomes(rows, path) == 5
    assert write_outcomes(rows[:2], path) == 2
    got = read_outcomes(path)
    assert got[:5] == rows and len(got) == 7
    assert write_outcomes(rows[:1], path, append=False) == 1
    assert read_outcomes(path) == rows[:1]
    path.write_text('{"task_id": "x"}\n')
    with pytest.raises(ValueError, match=r"o\.jsonl:1"):
        read_outcomes(path)


# ---- route_model in a chain


def test_route_model_in_promptchain_with_on_plan():
    seen, ran = [], []

    async def runner(model, task):
        ran.append((model, task))
        return f"{model}:{task}"

    step = route_model(router(difficulty=fixed(0.0), abilities={"small": 1.0, "mid": 2.0, "big": 3.0}),
                       runner, on_plan=seen.append)
    chain = PromptChain(models=[], instructions=[step, lambda s: s.upper()])
    assert run(chain.process_prompt_async("hello")) == "SMALL:HELLO"
    assert ran == [("small", "hello")] and seen[0].chosen == "small"


def test_route_model_survives_failing_on_plan(caplog):
    def bad(plan):
        raise RuntimeError("sink down")

    async def runner(model, task):
        return model

    with caplog.at_level(logging.WARNING, logger="promptchain.utils.model_router"):
        assert run(route_model(router(), runner, on_plan=bad)("t")) == "big"
    assert any("on_plan" in r.message for r in caplog.records)


# ---- import hygiene


def test_import_does_not_pull_profiler():
    code = ("import sys, promptchain.utils.model_router as m\n"
            "bad=[k for k in sys.modules if k.startswith('promptchain.profiler')]\n"
            "print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(__file__).resolve().parents[1]), check=True).stdout
    assert out.strip() == "[]"


def test_module_level_imports_exclude_profiler_scipy_numpy():
    src = Path(sys.modules[ModelRouter.__module__].__file__).read_text()
    names = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0] + ("." + node.module.split(".")[1]
                                                            if node.module and node.module.startswith("promptchain.") else ""))
    assert "scipy" not in names and "numpy" not in names
    assert not any("profiler" in n for n in names)


# ---- audit fixes

import numpy as np  # noqa: E402

from promptchain.utils.decision_step import SystemOneBackend  # noqa: E402
from promptchain.utils import model_router as mr  # noqa: E402


class HangingBackend(SystemOneBackend):
    """A SystemOneBackend whose executor thread blocks for ``timeout`` seconds."""

    def _post(self, body):
        time.sleep(self.timeout)
        raise TimeoutError("hung")


def test_backend_timeout_longer_than_router_bound_warns_at_construction(caplog):
    fn = decision_difficulty(HangingBackend(model="tev1:4b", timeout=0.4),
                             instructions="x", levels=LEVELS)
    with caplog.at_level(logging.WARNING, logger="promptchain.utils.model_router"):
        r = ModelRouter(MODELS, threshold=0.7, timeout_s=0.05, difficulty=fn, link=LINK,
                        abilities=ABILITIES)
    assert any("backend timeout" in x.message and "executor" in x.message for x in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="promptchain.utils.model_router"):
        ModelRouter(MODELS, threshold=0.7, timeout_s=1.0, difficulty=fn, link=LINK,
                    abilities=ABILITIES)
    assert not caplog.records
    t0 = time.monotonic()
    plan = run(r.plan("t"))  # the executor thread keeps running; the router does not wait
    assert time.monotonic() - t0 < 0.5
    assert plan.chosen == "big" and "timed out" in plan.degraded


def test_executor_thread_backend_exit_waits_for_hung_thread():
    """asyncio.run joins the default executor at exit, so a hung backend delays exit."""
    fn = decision_difficulty(HangingBackend(model="tev1:4b", timeout=0.5),
                             instructions="x", levels=LEVELS)
    r = ModelRouter(MODELS, threshold=0.7, timeout_s=0.05, difficulty=fn, link=LINK,
                    abilities=ABILITIES)
    t0 = time.monotonic()
    run(r.plan("t"))
    assert time.monotonic() - t0 >= 0.45


def test_fit_rejects_separable_and_degenerate_data():
    rnd = random.Random(5)
    by_theta = [RoutingOutcome(str(i), "t", rnd.random(), "s", "m", t, t > 0, "x")
                for i in range(20) for t in (-1.0, 1.0)]
    with pytest.raises(ValueError, match="separable|converge|degenerate"):
        DifficultyLink.fit(by_theta)
    by_d = [RoutingOutcome(str(i), "t", d, "s", "m", t, d < 0.5, "x")
            for i, d in enumerate(np.linspace(0, 1, 20)) for t in (-1.0, 0.0, 1.0)]
    with pytest.raises(ValueError, match="separable|converge|degenerate|not positive"):
        DifficultyLink.fit(by_d)
    few = _synthetic(29, 1.8, 0.0, 2.0, seed=1)
    with pytest.raises(ValueError, match="at least 30"):
        DifficultyLink.fit(few)


def test_fit_rejects_non_bool_passed():
    rows = _synthetic(60, 1.8, 0.0, 2.0, seed=4)
    rows[0].passed = "false"
    with pytest.raises(ValueError, match="bool"):
        DifficultyLink.fit(rows)


def test_fit_warns_when_beta_not_positive(caplog):
    rnd = random.Random(9)
    rows = []
    for i in range(1500):
        m = rnd.choice(MODELS)
        d = rnd.random()
        rows.append(RoutingOutcome(str(i), "t", d, "s", m, ABILITIES[m],
                                   rnd.random() < prob_3pl(ABILITIES[m], 1.5, -1.0 * d, 0.0), "x"))
    with caplog.at_level(logging.WARNING, logger="promptchain.utils.model_router"):
        DifficultyLink.fit(rows)
    assert any("beta" in x.message for x in caplog.records)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1, None, "1"])
def test_timeout_s_must_be_finite_positive(bad):
    with pytest.raises(ValueError, match="timeout_s"):
        ModelRouter(["a"], threshold=0.5, timeout_s=bad)


def test_read_outcomes_validates_types(tmp_path):
    good = asdict_row()
    path = tmp_path / "o.jsonl"
    for field, value in [("passed", "false"), ("passed", 1), ("difficulty", "0.9"),
                         ("theta", None), ("difficulty", True), ("model", 3)]:
        path.write_text(json.dumps(good) + "\n" + json.dumps({**good, field: value}) + "\n")
        with pytest.raises(ValueError, match=r"o\.jsonl:2"):
            read_outcomes(path)
    path.write_text("[1, 2]\n")
    with pytest.raises(ValueError, match=r"o\.jsonl:1"):
        read_outcomes(path)


def asdict_row():
    return {"task_id": "a", "task": "t", "difficulty": 0.5, "difficulty_source": "s",
            "model": "m", "theta": 0.1, "passed": True, "source": "explore", "reason": ""}


def test_write_outcomes_rejects_nan_without_partial_write(tmp_path):
    path = tmp_path / "o.jsonl"
    ok = RoutingOutcome("a", "t", 0.5, "s", "m", 0.1, True, "x")
    bad = RoutingOutcome("b", "t", float("nan"), "s", "m", 0.1, True, "x")
    with pytest.raises(ValueError, match="finite"):
        write_outcomes([ok, bad], path)
    assert not path.exists()


def test_write_outcomes_appends_after_missing_newline(tmp_path):
    path = tmp_path / "o.jsonl"
    ok = RoutingOutcome("a", "t", 0.5, "s", "m", 0.1, True, "x")
    path.write_text(json.dumps(asdict_row()))
    write_outcomes([ok], path)
    assert len(read_outcomes(path)) == 2


def test_low_precision_note_clears_when_profile_becomes_precise(tmp_path):
    path = tmp_path / "p.json"
    store = ProfileStore(path)
    r = ModelRouter(["big"], threshold=0.99, timeout_s=1, store_path=path, link=LINK,
                    difficulty=fixed(0.9))
    store.upsert(_profile("big", {"reasoning_depth": 3.0}, low=True))
    assert "low_precision" in run(r.plan("t")).reason
    store.upsert(_profile("big", {"reasoning_depth": 3.0}, low=False))
    assert "low_precision" not in run(r.plan("t")).reason


def test_misordered_models_warn_once_and_are_not_called_most_able(caplog):
    r = ModelRouter(["big", "small"], threshold=0.99, timeout_s=1, difficulty=fixed(0.9),
                    link=LINK, abilities={"big": 3.0, "small": -1.0})
    with caplog.at_level(logging.WARNING, logger="promptchain.utils.model_router"):
        p1 = run(r.plan("t"))
        run(r.plan("t"))
    assert len([x for x in caplog.records if "non-decreasing" in x.message]) == 1
    assert "most able" not in p1.reason and "last model in the list" in p1.reason
    assert "not non-decreasing" in p1.reason


def test_missing_theta_is_named_in_reason():
    r = router(difficulty=fixed(1.0), abilities={"small": -1.0, "mid": 0.0})
    p = run(r.plan("t"))
    assert p.chosen == "big" and "no ability for ['big']" in p.reason
    assert "most able" not in p.reason
    p = run(router(abilities={"small": float("nan"), "mid": "x", "big": 2.0}).plan("t"))
    assert "no ability for ['small', 'mid']" in p.reason and "invalid ability" in p.reason


def test_link_none_skips_difficulty_call(caplog):
    calls = []

    async def counting(task):
        calls.append(task)
        return Difficulty(0.3, "s")

    async def boom(task):
        raise RuntimeError("down")

    with caplog.at_level(logging.WARNING, logger="promptchain.utils.model_router"):
        p = run(router(link=None, difficulty=counting).plan("t"))
        run(router(link=None, difficulty=boom).plan("t"))
    assert calls == [] and p.chosen == "big" and "no link" in p.degraded
    assert p.difficulty is None
    assert not [x for x in caplog.records if x.levelno >= logging.WARNING]


def test_corrupt_store_warns_once_and_store_read_once_per_plan(tmp_path, caplog, monkeypatch):
    path = tmp_path / "p.json"
    path.write_text("{not json")
    r = ModelRouter(MODELS, threshold=0.7, timeout_s=1, store_path=path, link=LINK,
                    difficulty=fixed(0.0))
    with caplog.at_level(logging.WARNING, logger="promptchain.utils.model_router"):
        run(r.plan("t"))
        run(r.plan("t"))
    assert len([x for x in caplog.records if "corrupt" in x.message]) == 1
    from promptchain.profiler import store as store_mod

    reads = []
    real = store_mod.ProfileStore.load
    monkeypatch.setattr(store_mod.ProfileStore, "load",
                        lambda self: (reads.append(1), real(self))[1])
    run(r.plan("t"))
    assert len(reads) == 1  # one read for three models


def test_decision_difficulty_skips_non_finite_probabilities(caplog):
    stub = StubBackend({"difficulty": {"score": 2, "confidence": 0.9,
                                       "probabilities": {"0": float("nan"), "4": 1.0,
                                                         "1": float("inf")}}})
    d = run(decision_difficulty(stub, instructions="x", levels=LEVELS)("t"))
    assert d.value == 1.0
    stub = StubBackend({"difficulty": {"score": 2, "confidence": 0.9,
                                       "probabilities": {"zz": 1.0}}})
    with caplog.at_level(logging.DEBUG, logger="promptchain.utils.model_router"):
        d = run(decision_difficulty(stub, instructions="x", levels=LEVELS)("t"))
    assert d.value == 0.5 and any("no probability key" in x.message for x in caplog.records)
    stub = StubBackend({"difficulty": {"score": float("nan"), "confidence": 0.9}})
    with pytest.raises(DecisionError):
        run(decision_difficulty(stub, instructions="x", levels=LEVELS)("t"))


def test_decision_difficulty_parses_real_tev1_reply_shape():
    reply = {"type": "score", "score": 2.924, "confidence": 0.269,
             "legend": {str(i): v for i, v in enumerate(LEVELS)},
             "probabilities": {"0": 0.02, "1": 0.1, "2": 0.25, "3": 0.5, "4": 0.13}}
    d = run(decision_difficulty(StubBackend({"difficulty": reply}), instructions="x",
                                levels=LEVELS)("t"))
    expected = (0 * 0.02 + 1 * 0.1 + 2 * 0.25 + 3 * 0.5 + 4 * 0.13) / 4
    assert d.value == pytest.approx(expected)


def test_cancelled_error_propagates():
    async def cancelled(task):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        run(router(difficulty=cancelled).plan("t"))


def test_probability_equal_to_threshold_is_chosen():
    flat = DifficultyLink(a=1.0, alpha=0.0, beta=0.0, n=100)  # P = sigmoid(theta)
    r = ModelRouter(["small", "big"], threshold=0.5, timeout_s=1, difficulty=fixed(0.5),
                    link=flat, abilities={"small": 0.0, "big": 1.0})
    p = run(r.plan("t"))
    assert p.p_success["small"] == 0.5 and p.chosen == "small"


def test_wrong_return_type_reports_typeerror(caplog):
    async def wrong(task):
        return 0.5

    p = run(router(difficulty=wrong).plan("t"))
    assert "difficulty call failed: TypeError" in p.degraded and "not Difficulty" in p.degraded


def test_last_resort_guard_catches_a_raising_link(caplog):
    class BrokenLink:
        def p_success(self, theta, d):
            raise RuntimeError("link exploded")

    with caplog.at_level(logging.WARNING, logger="promptchain.utils.model_router"):
        p = run(router(link=BrokenLink()).plan("t"))
    assert p.chosen == "big" and "planning failed: RuntimeError" in p.degraded
    assert any("planning failed" in x.message for x in caplog.records)
