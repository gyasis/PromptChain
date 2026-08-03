"""AgenticStepProcessor must bound its inner ReAct tool loop.

Regression: `max_internal_steps` bounds REASONING iterations only. After tools
execute the loop `continue`s WITHOUT incrementing step_num, so a model that keeps
emitting tool calls runs unbounded in wall-clock. Observed in the wild: a tool the
model could not call correctly was retried 44 times at ~2-3s each, consuming a
20-minute job budget and producing nothing — while every logged internal iteration
still read "1/12", so the run looked healthy right up until the timeout.

These tests drive run_async with an llm_runner that ALWAYS returns a tool call —
i.e. a model that never converges — and assert termination.
"""

import asyncio
import pytest

from promptchain.utils.agentic_step_processor import AgenticStepProcessor


class _ToolCall:
    """Minimal LiteLLM-ish tool_call object."""
    def __init__(self, i):
        self.id = f"call_{i}"
        self.type = "function"
        self.function = type("F", (), {"name": "always_fails", "arguments": "{}"})()


class _Msg:
    """Assistant message that always requests another tool call."""
    def __init__(self, i):
        self.content = None
        self.tool_calls = [_ToolCall(i)]
        self.role = "assistant"


TOOLS = [{
    "type": "function",
    "function": {
        "name": "always_fails",
        "description": "a tool the model can never satisfy",
        "parameters": {"type": "object", "properties": {}},
    },
}]


def _run(max_tool_rounds, max_internal_steps=3):
    """Drive the processor with a never-converging model; return (result, calls)."""
    calls = {"n": 0}

    async def llm_runner(*a, **k):
        calls["n"] += 1
        return _Msg(calls["n"])

    async def tool_executor(tool_call):
        return "error: tool unavailable"

    proc = AgenticStepProcessor(
        objective="loop forever unless bounded",
        max_internal_steps=max_internal_steps,
        max_tool_rounds=max_tool_rounds,
    )
    result = asyncio.run(asyncio.wait_for(
        proc.run_async("go", TOOLS, llm_runner, tool_executor),
        timeout=30,          # the whole point: this must NOT be what stops it
    ))
    return result, calls["n"]


def test_terminates_when_model_never_converges():
    """Without the bound this never returns; with it, it must."""
    result, n = _run(max_tool_rounds=4)
    assert result is not None
    # bounded per step, across at most max_internal_steps steps, plus the
    # forced-final-answer call each step — generous ceiling, but FINITE.
    assert n < 40, f"expected a bounded number of LLM calls, got {n}"


def test_lower_bound_costs_fewer_calls():
    """The budget is real: a tighter bound must do strictly less work."""
    _, few = _run(max_tool_rounds=2)
    _, many = _run(max_tool_rounds=8)
    assert few < many, f"budget not enforced: {few} !< {many}"


def test_budget_floor_is_at_least_one():
    proc = AgenticStepProcessor(objective="x", max_tool_rounds=0)
    assert proc.max_tool_rounds == 1
    proc2 = AgenticStepProcessor(objective="x", max_tool_rounds=-5)
    assert proc2.max_tool_rounds == 1


def test_default_is_generous_enough_for_normal_work():
    """Default must not throttle legitimate multi-tool steps (observed max ~12)."""
    assert AgenticStepProcessor(objective="x").max_tool_rounds >= 30


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
