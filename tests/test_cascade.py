"""Offline tests for the verify-then-escalate cascade (no model or network calls)."""

import asyncio
import json
from unittest.mock import patch

import pytest

from promptchain import PromptChain
from promptchain.utils.cascade import (
    CascadeExhausted,
    cascade,
    decision_check,
)


def run(coro):
    return asyncio.run(coro)


class Tier:
    def __init__(self, out, name):
        self.out, self.calls = out, []
        self.__name__ = name

    async def __call__(self, state):
        self.calls.append(state)
        return self.out


def contains_ok(step_input, output):
    return "ok" in output


def test_cheap_passes_frontier_untouched():
    cheap, big = Tier("ok cheap", "cheap"), Tier("ok big", "big")
    step = cascade([cheap, big], check=contains_ok, name="t")
    assert run(step("q")) == "ok cheap"
    assert big.calls == []
    assert step.last_result.answered_by == "cheap"
    assert not step.last_result.escalated


def test_cheap_fails_frontier_passes_with_reasons():
    cheap, big = Tier("bad", "cheap"), Tier("ok big", "big")
    step = cascade([cheap, big], check=lambda i, o: ("ok" in o, f"saw {o}"), name="t")
    assert run(step("q")) == "ok big"
    r = step.last_result
    assert r.escalated and r.tier == 1 and r.answered_by == "big"
    assert [(a.passed, a.reason) for a in r.attempts] == [(False, "saw bad"), (True, "saw ok big")]
    assert big.calls == ["q"]  # the next tier gets the original input


def test_tier_exception_escalates_and_is_recorded():
    async def boom(state):
        raise PermissionError("401 bad key")

    step = cascade([boom, Tier("ok", "big")], check=contains_ok, name="t")
    assert run(step("q")) == "ok"
    a = step.last_result.attempts[0]
    assert not a.passed and "PermissionError: 401 bad key" in a.error and a.output is None


def test_check_exception_propagates():
    def bad_check(i, o):
        raise RuntimeError("verifier down")

    big = Tier("ok", "big")
    step = cascade([Tier("x", "cheap"), big], check=bad_check, name="t")
    with pytest.raises(RuntimeError, match="verifier down"):
        run(step("q"))
    assert big.calls == []


def test_exhausted_raises_with_every_attempt_and_error():
    async def boom(state):
        raise ValueError("nope")

    step = cascade([boom, Tier("bad", "big")], check=contains_ok, name="t")
    with pytest.raises(CascadeExhausted) as ei:
        run(step("q"))
    res = ei.value.result
    assert len(res.attempts) == 2 and res.output is None and res.tier is None
    assert "ValueError: nope" in str(ei.value)


def test_exhausted_uses_on_exhausted_with_original_state():
    fallback = Tier("human", "fb")
    step = cascade([Tier("bad", "a")], check=contains_ok, name="t", on_exhausted=fallback)
    assert run(step("q")) == "human"
    assert fallback.calls == ["q"]
    assert step.last_result.output is None


def test_async_check_tuple_and_on_result():
    seen = []

    async def acheck(i, o):
        return ("ok" in o, "async verdict")

    step = cascade([Tier("ok", "a")], check=acheck, name="t", on_result=seen.append)
    run(step("q"))
    assert seen[0].attempts[0].reason == "async verdict"


def test_labels_default_and_explicit():
    chain = PromptChain(models=["openai/gpt-x"], instructions=["{input}"])
    step = cascade([chain, Tier("ok", "fn")], check=contains_ok, name="t")
    with patch.object(PromptChain, "process_prompt_async", return_value="bad"):
        run(step("q"))
    assert [a.label for a in step.last_result.attempts] == ["openai/gpt-x", "fn"]
    named = cascade([Tier("ok", "fn")], check=contains_ok, name="t", labels=["L"])
    run(named("q"))
    assert named.last_result.answered_by == "L"


def test_promptchain_tier_called_with_state():
    chain = PromptChain(models=["m"], instructions=["{input}"])
    with patch.object(PromptChain, "process_prompt_async", return_value="ok from chain") as m:
        step = cascade([chain], check=contains_ok, name="t")
        assert run(step("q")) == "ok from chain"
    m.assert_called_once_with("q")


def test_to_dict_is_json_serialisable():
    step = cascade([Tier("bad", "a"), Tier("ok", "b")], check=contains_ok, name="t")
    run(step("q"))
    d = json.loads(json.dumps(step.last_result.to_dict()))
    assert d["escalated"] is True and len(d["attempts"]) == 2


def test_constructor_validation():
    with pytest.raises(ValueError):
        cascade([], check=contains_ok, name="t")
    with pytest.raises(ValueError):
        cascade([Tier("a", "a")], check=contains_ok, name="t", labels=["x", "y"])


class StubBackend:
    name = "stub"
    model = "stub-model"

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    async def decide(self, state, questions):
        self.calls.append((state, questions))
        return {"model": self.model, "answers": self.answers}


def test_decision_check_state_and_pass():
    be = StubBackend({"ok": {"boolean": True, "confidence": 0.97}})
    check = decision_check(be, instructions="Is it right?", accept=0.9)
    passed, reason = run(check("the q", "the a"))
    assert passed and "band=accept" in reason
    assert be.calls[0][0] == "<input>the q</input><output>the a</output>"
    assert be.calls[0][1]["ok"]["instructions"] == "Is it right?"


def test_decision_check_fails_on_false_or_low_confidence():
    low = StubBackend({"ok": {"boolean": True, "confidence": 0.6}})
    false = StubBackend({"ok": {"boolean": False, "confidence": 0.99}})
    assert run(decision_check(low, instructions="?", accept=0.9)("i", "o"))[0] is False
    assert run(decision_check(false, instructions="?", accept=0.9)("i", "o"))[0] is False


def test_decision_check_boundary_confidence_passes():
    be = StubBackend({"ok": {"boolean": True, "confidence": 0.9}})
    assert run(decision_check(be, instructions="?", accept=0.9)("i", "o"))[0] is True


def test_decision_check_in_cascade_in_chain():
    be = StubBackend({"ok": {"boolean": True, "confidence": 0.99}})
    step = cascade(
        [Tier("answer", "cheap")],
        check=decision_check(be, instructions="?", accept=0.9),
        name="t",
    )
    chain = PromptChain(models=[], instructions=[step])
    assert run(chain.process_prompt_async("question")) == "answer"
    assert be.calls[0][0] == "<input>question</input><output>answer</output>"
