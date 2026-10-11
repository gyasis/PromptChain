"""Offline tests for promptchain.utils.decision_step (no model, no network).

They assert what goes ON THE WIRE (request body / argv / env), not just the parsed answer —
a test that a prompt-and-parse backend would also pass is not testing the primitive.
"""
import asyncio
import json

import pytest

from promptchain import PromptChain
from promptchain.utils import decision_step as ds
from promptchain.utils.decision_step import (
    CliBackend,
    DecisionError,
    SystemOneBackend,
    decision,
    hunch_backend,
    route_on,
)

DEPT = {
    "dept": {
        "type": "choice",
        "instructions": "Which team owns this ticket?",
        "criteria": {"billing": "charges, refunds", "technical": "bugs, errors"},
    }
}
URGENT = {"urgent": {"type": "boolean", "instructions": "Is this urgent?"}}


class StubBackend:
    """Returns a fixed reply and records every call."""

    name = "stub"
    model = "stub-model"

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    async def decide(self, state, questions):
        self.calls.append((state, questions))
        return {"model": self.model, "answers": self.answers}


def run(coro):
    return asyncio.run(coro)


# --- explicit model / backend (R-PC4) -------------------------------------------------


def test_backends_require_explicit_model():
    with pytest.raises(DecisionError):
        SystemOneBackend(model="")
    with pytest.raises(DecisionError):
        hunch_backend(model="")


def test_decision_requires_backend():
    with pytest.raises(DecisionError):
        decision(DEPT, backend=None, name="x")


def test_no_env_default_model(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_MODEL", "should-not-be-used")
    monkeypatch.setenv("HUNCH_MODEL", "should-not-be-used")
    assert SystemOneBackend(model="tev1:4b").model == "tev1:4b"
    _, env = hunch_backend(model="qwen2.5:3b").build_invocation(DEPT)
    assert env["HUNCH_MODEL"] == "qwen2.5:3b"


# --- question validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"q": {"type": "prose"}},
        {"q": {"type": "choice", "criteria": {"only": "one"}}},
        {"q": {"type": "score", "criteria": "low"}},
    ],
)
def test_malformed_questions_rejected_before_any_call(bad):
    with pytest.raises(DecisionError):
        decision(bad, backend=StubBackend({}), name="x")


# --- wire format --------------------------------------------------------------------


def test_systemone_request_body_and_noul_translation():
    b = SystemOneBackend(model="tev1:4b", base_url="http://lane/llm/ollama/")
    body = b.build_request("my ticket", {**DEPT, **URGENT})
    assert b.url == "http://lane/llm/ollama/v1/systemone"
    assert body["model"] == "tev1:4b"
    assert body["state"] == "my ticket"
    assert body["questions"]["urgent"]["type"] == "noul"  # endpoint spelling
    assert body["questions"]["dept"] == DEPT["dept"]
    assert URGENT["urgent"]["type"] == "boolean"  # caller's dict not mutated


def test_systemone_default_is_governed_lane_not_raw_ollama():
    assert ":8799" in SystemOneBackend(model="tev1:4b").url
    assert ":11434" not in SystemOneBackend(model="tev1:4b").url


def test_systemone_reply_noul_normalized(monkeypatch):
    b = SystemOneBackend(model="tev1:4b")
    sent = {}

    def fake_post(body):
        sent.update(body)
        return {"model": "tev1:4b", "answers": {"urgent": {"type": "noul", "noul": 0.2}}}

    monkeypatch.setattr(b, "_post", fake_post)
    reply = run(b.decide("x", URGENT))
    assert sent["questions"]["urgent"]["type"] == "noul"
    ans = reply["answers"]["urgent"]
    assert ans["type"] == "boolean" and ans["boolean"] is False and ans["probability"] == 0.2


def test_hunch_invocation_argv_and_mode_env():
    b = hunch_backend(model="mistral-small:22b", vote=6, timeout=60)
    argv, env = b.build_invocation(DEPT)
    assert argv[:2] == ["hunch", "-"]  # state goes on stdin, never argv
    assert argv[argv.index("-q") + 1] == json.dumps(DEPT)
    assert "--raw" in argv
    assert env == {"HUNCH_MODEL": "mistral-small:22b", "HUNCH_VOTE": "6"}
    assert b.name == "hunch-vote"


def test_cli_backend_missing_binary_is_a_clear_error():
    b = CliBackend(["no-such-decider-xyz"], model="m", model_env="M")
    with pytest.raises(DecisionError, match="not on PATH"):
        run(b.decide("x", DEPT))


# --- answer validation --------------------------------------------------------------


def test_out_of_set_choice_is_rejected():
    step = decision(DEPT, StubBackend({"dept": {"choice": "sales", "confidence": 0.99}}), "t")
    with pytest.raises(DecisionError, match="not one of"):
        run(step("ticket"))


def test_backend_error_answer_propagates():
    step = decision(DEPT, StubBackend({"dept": {"error": "no candidate label"}}), "t")
    with pytest.raises(DecisionError, match="no candidate label"):
        run(step("ticket"))


def test_missing_answer_is_rejected():
    step = decision({**DEPT, **URGENT}, StubBackend({"dept": {"choice": "billing",
                                                              "confidence": 0.9}}), "t")
    with pytest.raises(DecisionError, match="no answer for"):
        run(step("ticket"))


def test_boolean_confidence_derived_from_probability_when_absent():
    step = decision(URGENT, StubBackend({"urgent": {"probability": 0.5}}), "t")
    rec = json.loads(run(step("ticket")))
    ans = rec["answers"]["urgent"]
    assert ans["value"] is True and ans["confidence"] == 0.0  # 50/50 = no confidence


def test_record_keeps_state_and_metadata():
    stub = StubBackend({"dept": {"choice": "billing", "confidence": 0.95,
                                 "probabilities": {"billing": 0.97, "technical": 0.03}}})
    rec = json.loads(run(decision(DEPT, stub, "triage")("refund please")))
    assert rec["state"] == "refund please"
    assert rec["decision"] == "triage" and rec["backend"] == "stub"
    assert rec["answers"]["dept"]["value"] == "billing"
    assert stub.calls == [("refund please", DEPT)]


# --- routing ------------------------------------------------------------------------


def _record(value, confidence, state="S"):
    return json.dumps({"state": state, "answers": {"dept": {"value": value,
                                                            "confidence": confidence}}})


@pytest.mark.parametrize(
    "conf,expected", [(0.95, "accept"), (0.9, "accept"), (0.7, "review"), (0.3, "escalate")]
)
def test_route_bands(conf, expected):
    step = route_on("dept", accept=0.9, review=0.6,
                    on_accept=lambda s: f"accept:{s}",
                    on_review=lambda s: f"review:{s}",
                    on_escalate=lambda s: f"escalate:{s}")
    assert run(step(_record("billing", conf))) == f"{expected}:S"


def test_route_accept_mapping_routes_by_value_and_awaits_async_branches():
    async def tech(s):
        return f"tech:{s}"

    step = route_on("dept", accept=0.9, review=0.6,
                    on_accept={"billing": lambda s: f"bill:{s}", "technical": tech},
                    on_review=str, on_escalate=str)
    assert run(step(_record("technical", 0.99))) == "tech:S"
    with pytest.raises(DecisionError, match="no accept branch"):
        run(route_on("dept", accept=0.9, review=0.6, on_accept={"billing": str},
                     on_review=str, on_escalate=str)(_record("technical", 0.99)))


def test_route_rejects_bad_thresholds_and_bad_input():
    with pytest.raises(DecisionError):
        route_on("dept", accept=0.5, review=0.8, on_accept=str, on_review=str, on_escalate=str)
    step = route_on("dept", accept=0.9, review=0.6, on_accept=str, on_review=str,
                    on_escalate=str)
    with pytest.raises(DecisionError, match="must follow a decision step"):
        run(step("not json"))


# --- end to end in a real PromptChain -----------------------------------------------


def test_decision_and_route_compose_in_promptchain_without_models():
    stub = StubBackend({"dept": {"choice": "billing", "confidence": 0.97}})
    chain = PromptChain(
        models=[],
        instructions=[
            decision(DEPT, stub, "triage"),
            route_on("dept", accept=0.9, review=0.6,
                     on_accept={"billing": lambda s: f"BILLING<{s}>",
                                "technical": lambda s: f"TECH<{s}>"},
                     on_review=lambda s: "REVIEW", on_escalate=lambda s: "ESCALATE"),
        ],
    )
    assert run(chain.process_prompt_async("I was double charged")) == "BILLING<I was double charged>"
    assert stub.calls[0][0] == "I was double charged"


def test_band_helper():
    assert ds.band(0.9, 0.9, 0.6) == "accept"
    assert ds.band(0.6, 0.9, 0.6) == "review"
    assert ds.band(0.59, 0.9, 0.6) == "escalate"
