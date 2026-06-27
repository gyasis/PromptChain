"""RED (test-first) tests for promptchain.profiler.scoring (NOT YET IMPLEMENTED).

Pins the auto-scorer contract:
    score_response(item: ProbeItem, reply: dict) -> tuple[bool, dict]
where reply == {"text": str, "usage": {...}, "latency_ms": float}.
score_response dispatches on item.dimension and returns (correct, raw_values).

Each test STATES the (simple, deterministic) scoring rule the impl must mirror.
MUST fail now (scoring.py is a stub).
"""
from __future__ import annotations

from promptchain.profiler.scoring import score_response
from promptchain.profiler.item_bank import ProbeItem


def _reply(text, latency_ms=10.0):
    return {
        "text": text,
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        "latency_ms": latency_ms,
    }


# --------------------------------------------------------------------------- #
# instruction_following: correct iff reply["text"].strip() == meta["expect"]
# --------------------------------------------------------------------------- #
def test_instruction_following_exact_match():
    item = ProbeItem(
        item_id="if1", dimension="instruction_following", a=1.0, b=0.0, c=0.0,
        meta={"expect": "DONE"},
    )
    correct, raw = score_response(item, _reply("  DONE  "))
    assert correct is True
    assert isinstance(raw, dict)


def test_instruction_following_non_match():
    item = ProbeItem(
        item_id="if2", dimension="instruction_following", a=1.0, b=0.0, c=0.0,
        meta={"expect": "DONE"},
    )
    correct, _ = score_response(item, _reply("not done"))
    assert correct is False


# --------------------------------------------------------------------------- #
# tool_call_reliability: reply text must be a JSON object whose "tool" == meta["expect_tool"]
# --------------------------------------------------------------------------- #
def test_tool_call_reliability_valid_json_match():
    item = ProbeItem(
        item_id="tc1", dimension="tool_call_reliability", a=1.0, b=0.0, c=0.0,
        meta={"expect_tool": "search"},
    )
    correct, _ = score_response(item, _reply('{"tool": "search", "args": {}}'))
    assert correct is True


def test_tool_call_reliability_malformed_json():
    item = ProbeItem(
        item_id="tc2", dimension="tool_call_reliability", a=1.0, b=0.0, c=0.0,
        meta={"expect_tool": "search"},
    )
    correct, _ = score_response(item, _reply('{tool: search'))
    assert correct is False


# --------------------------------------------------------------------------- #
# format_sensitivity: correct iff reply["text"] is parseable JSON
# --------------------------------------------------------------------------- #
def test_format_sensitivity_valid_json():
    item = ProbeItem(
        item_id="fs1", dimension="format_sensitivity", a=1.0, b=0.0, c=0.0,
        meta={"format": "json"},
    )
    correct, _ = score_response(item, _reply('{"ok": true}'))
    assert correct is True


def test_format_sensitivity_invalid_json():
    item = ProbeItem(
        item_id="fs2", dimension="format_sensitivity", a=1.0, b=0.0, c=0.0,
        meta={"format": "json"},
    )
    correct, _ = score_response(item, _reply('not json at all'))
    assert correct is False


# --------------------------------------------------------------------------- #
# reasoning_depth: correct iff meta["answer"] token appears in reply["text"]
# --------------------------------------------------------------------------- #
def test_reasoning_depth_answer_present():
    item = ProbeItem(
        item_id="rd1", dimension="reasoning_depth", a=1.0, b=0.0, c=0.0,
        meta={"answer": "42"},
    )
    correct, _ = score_response(item, _reply("after reasoning the answer is 42."))
    assert correct is True


def test_reasoning_depth_answer_absent():
    item = ProbeItem(
        item_id="rd2", dimension="reasoning_depth", a=1.0, b=0.0, c=0.0,
        meta={"answer": "42"},
    )
    correct, _ = score_response(item, _reply("the answer is 7"))
    assert correct is False


# --------------------------------------------------------------------------- #
# latency_throughput: raw-value dimension — raw carries latency_ms; correct=True (no threshold)
# --------------------------------------------------------------------------- #
def test_latency_throughput_raw_carries_latency():
    item = ProbeItem(
        item_id="lt1", dimension="latency_throughput", a=1.0, b=0.0, c=0.0, meta={},
    )
    correct, raw = score_response(item, _reply("anything", latency_ms=123.4))
    assert correct is True
    assert raw["latency_ms"] == 123.4
