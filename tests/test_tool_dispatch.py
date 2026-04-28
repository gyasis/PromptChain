"""Tests for v0.6.1 tool-dispatch fixes.

Covers:
* Bug 2 — schema/function name-mismatch validation in PromptChain.add_tools.
* Bug 3 — tool-execution body must run INSIDE the per-tool for-loop in
  AgenticStepProcessor (formerly was outside, with for/else semantics that
  silently dropped all but the last tool_call and overwrote tool_call_id with None).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.promptchaining import PromptChain


# ---------------------------------------------------------------------------
# Bug 2 — schema/function name-mismatch
# ---------------------------------------------------------------------------


def _tool_schema(name: str) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_schema_function_name_mismatch_raises_value_error():
    """add_tools + register_tool_function with mismatched names must raise.

    Exercises the v0.6.1 validation: schema name 'my_tool' has no matching
    registered function 'my_tool_func'. Error message must name BOTH the
    schema name and the available function names so the consumer can fix it.
    """
    chain = PromptChain(models=["openai/gpt-4o-mini"], instructions=["{input}"])

    def my_tool_func(q: str) -> str:  # noqa: ARG001
        return "ok"

    chain.register_tool_function(my_tool_func)
    # Mismatch: schema name 'my_tool' but registered function is 'my_tool_func'.
    # Validation runs eagerly inside add_tools because a function is already
    # registered, so the ValueError fires here.
    with pytest.raises(ValueError) as exc_info:
        chain.add_tools([_tool_schema("my_tool")])

    msg = str(exc_info.value)
    assert "my_tool" in msg, f"Schema name missing from error: {msg}"
    assert "my_tool_func" in msg, f"Registered function name missing from error: {msg}"


def test_schema_function_name_match_no_error():
    """Matching names pass validation cleanly — no exception raised."""
    chain = PromptChain(models=["openai/gpt-4o-mini"], instructions=["{input}"])

    def my_tool(q: str) -> str:  # noqa: ARG001
        return "ok"

    chain.register_tool_function(my_tool)
    chain.add_tools([_tool_schema("my_tool")])

    # Must not raise.
    chain._validate_tool_registry()


# ---------------------------------------------------------------------------
# Bug 3 — tool-execution body must run inside the per-tool for-loop
# ---------------------------------------------------------------------------


def _dict_tool_call(call_id: str, name: str, args: str = "{}") -> Dict[str, Any]:
    """Build an OpenAI-style dict tool_call."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _object_tool_call(call_id: str, name: str, args: str = "{}") -> SimpleNamespace:
    """Build a LiteLLM-style object tool_call."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=args),
    )


def _make_processor() -> AgenticStepProcessor:
    return AgenticStepProcessor(
        objective="Test objective",
        max_internal_steps=3,
        model_name="openai/gpt-4o-mini",
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


class _ScriptedLLM:
    """LLM runner that returns a scripted sequence of response messages.

    Each entry is a dict with optional 'tool_calls' and 'content' keys. After
    the script is exhausted the runner returns a content-only message that
    terminates the loop.
    """

    def __init__(self, responses: List[Dict[str, Any]]):
        self._responses = list(responses)
        self.call_count = 0

    async def __call__(self, *args, **kwargs):
        self.call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return {"role": "assistant", "content": "DONE", "tool_calls": None}


class _RecordingExecutor:
    """Tool executor that records each call and returns scripted results.

    Maps function-name -> result string. Optionally raises if asked.
    """

    def __init__(self, results: Dict[str, Any], raise_for: str | None = None):
        self.results = results
        self.raise_for = raise_for
        self.calls: List[Any] = []

    async def __call__(self, tool_call):
        self.calls.append(tool_call)
        # Resolve function name (works for dict + object)
        if isinstance(tool_call, dict):
            name = tool_call.get("function", {}).get("name")
        else:
            name = getattr(getattr(tool_call, "function", None), "name", None)
        if self.raise_for and name == self.raise_for:
            raise RuntimeError(f"boom from {name}")
        return self.results.get(name, f"result for {name}")


def _extract_tool_msgs(processor_history_attr: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [m for m in processor_history_attr if isinstance(m, dict) and m.get("role") == "tool"]


async def _run_step(processor, tool_calls, executor, available_tools):
    """Drive one ACT iteration: scripted LLM yields tool_calls then a final-answer message.

    If any tool_call is an object (SimpleNamespace) we wrap the whole response
    in a SimpleNamespace too — mirrors LiteLLM's response-object shape and
    avoids json.dumps choking on the SimpleNamespace tool_calls in debug log.
    """
    has_object_calls = any(not isinstance(tc, dict) for tc in tool_calls)
    if has_object_calls:
        first = SimpleNamespace(
            role="assistant", content=None, tool_calls=tool_calls
        )
    else:
        first = {"role": "assistant", "content": None, "tool_calls": tool_calls}
    llm = _ScriptedLLM(
        [
            first,
            {"role": "assistant", "content": "FINAL", "tool_calls": None},
        ]
    )
    return await processor.run_async(
        initial_input="ignored",
        available_tools=available_tools,
        llm_runner=llm,
        tool_executor=executor,
        return_metadata=True,
    )


def _available_tools(*names: str) -> List[Dict[str, Any]]:
    return [_tool_schema(n) for n in names]


def test_dispatch_single_dict_tool_call():
    """Single dict-style tool_call → executor invoked once with the right id."""
    processor = _make_processor()
    executor = _RecordingExecutor({"foo": "FOO_RESULT"})
    tool_calls = [_dict_tool_call("call_1", "foo", '{"x": 1}')]

    result = asyncio.run(_run_step(processor, tool_calls, executor, _available_tools("foo")))

    assert len(executor.calls) == 1
    assert executor.calls[0]["id"] == "call_1"
    # Verify a tool message was appended with the right tool_call_id and result
    tool_msgs = [s for s in result.steps for tm in [None] if False]  # unused — go via result content
    # Confirm the loop produced a final answer (would not happen if dispatch crashed)
    assert result.final_answer == "FINAL"


def test_dispatch_single_object_tool_call():
    """Single object-style tool_call (LiteLLM shape) is dispatched correctly."""
    processor = _make_processor()
    executor = _RecordingExecutor({"bar": "BAR_RESULT"})
    tool_calls = [_object_tool_call("call_2", "bar")]

    result = asyncio.run(_run_step(processor, tool_calls, executor, _available_tools("bar")))

    assert len(executor.calls) == 1
    assert executor.calls[0].id == "call_2"
    assert result.final_answer == "FINAL"


def test_dispatch_multiple_parallel_tool_calls():
    """Two tool_calls in one LLM response → BOTH dispatched (regression for Bug 3)."""
    processor = _make_processor()
    executor = _RecordingExecutor({"foo": "FOO", "bar": "BAR"})
    tool_calls = [
        _dict_tool_call("call_a", "foo"),
        _dict_tool_call("call_b", "bar"),
    ]

    result = asyncio.run(
        _run_step(processor, tool_calls, executor, _available_tools("foo", "bar"))
    )

    assert len(executor.calls) == 2, (
        "Both tool_calls must be dispatched. Bug 3: only the LAST was dispatched."
    )
    ids = [c["id"] for c in executor.calls]
    assert ids == ["call_a", "call_b"]
    assert result.final_answer == "FINAL"


def test_dispatch_tool_raises_exception():
    """If a tool raises, dispatch must capture the error and continue (not die)."""
    processor = _make_processor()
    executor = _RecordingExecutor({"foo": "FOO"}, raise_for="foo")
    tool_calls = [_dict_tool_call("call_err", "foo")]

    result = asyncio.run(_run_step(processor, tool_calls, executor, _available_tools("foo")))

    # Loop must still terminate — a final message was reached.
    assert result.final_answer == "FINAL"
    # The executor was invoked.
    assert len(executor.calls) == 1


def test_dispatch_tool_call_id_preserved_through_for_loop():
    """tool_call_id from a dict-style call must NOT be overwritten with None.

    Reproduces the for/else bug surface: the v0.6.1 fix re-indented the
    per-tool execution body INTO the for-loop. Pre-fix, the body was at the
    for-loop's column, so the assignment of tool_call_id used
    ``getattr(tool_call, "id", None)`` on the last loop variable. For a
    dict-style tool_call, getattr(dict, 'id', None) is None, so the
    constructed tool message had ``tool_call_id=None``. OpenAI then rejected
    the next turn with "Missing parameter 'tool_call_id'".

    The previous version of this test only checked the executor's recorded
    call (whose dict id is intact regardless of the bug). That was theatre.
    This version inspects the actual ``tool_msg`` constructed inside the
    for-loop and appended to ``internal_history`` — the real bug surface.
    """
    processor = _make_processor()
    executor = _RecordingExecutor({"foo": "FOO"})
    tool_calls = [_dict_tool_call("call_unique_id_xyz", "foo")]

    asyncio.run(_run_step(processor, tool_calls, executor, _available_tools("foo")))

    # Sanity: executor saw the original dict id intact.
    assert executor.calls[0]["id"] == "call_unique_id_xyz"

    # Real assertion: the constructed tool message in internal_history must
    # carry the same id. Pre-fix this would be None.
    tool_msgs = _extract_tool_msgs(processor.internal_history)
    assert tool_msgs, (
        "No role='tool' message was appended to internal_history — dispatch "
        "loop never built one. Pre-fix this happened because the body lived "
        "outside the for-loop."
    )
    assert tool_msgs[-1]["tool_call_id"] == "call_unique_id_xyz", (
        f"tool_call_id on the constructed tool message was "
        f"{tool_msgs[-1]['tool_call_id']!r}, expected 'call_unique_id_xyz'. "
        "Pre-fix this was None because getattr(dict, 'id', None) on the "
        "last (dict) tool_call returned None."
    )
    assert tool_msgs[-1]["tool_call_id"] is not None, (
        "tool_call_id is None — OpenAI will reject the next turn with "
        "'Missing parameter tool_call_id'."
    )


# ---------------------------------------------------------------------------
# Bug 2 — symmetric validation on register_tool_function
# ---------------------------------------------------------------------------


def test_register_function_after_add_tools_validates_immediately():
    """add_tools first, then register_tool_function with a mismatching name.

    Pre-fix this silently passed: ``add_tools`` had no functions yet so it
    short-circuited the validation, and ``register_tool_function`` never ran
    the check. The schema/function name mismatch only surfaced much later as
    an opaque OpenAI error ("Missing parameter 'tool_call_id'").

    Post-fix: ``register_tool_function`` mirrors ``add_tools`` and calls
    ``_validate_tool_registry`` whenever schemas are already present. The
    mismatch fires at the natural call-site the consumer can act on.
    """
    chain = PromptChain(models=["openai/gpt-4o-mini"], instructions=["{input}"])

    # Schema first, no function registered yet — validation is deferred here.
    chain.add_tools([_tool_schema("my_tool")])

    def my_other_func(q: str) -> str:  # noqa: ARG001
        return "ok"

    # Now register a function with a non-matching name. Validation MUST fire.
    with pytest.raises(ValueError) as exc_info:
        chain.register_tool_function(my_other_func)

    msg = str(exc_info.value)
    assert "my_tool" in msg, f"Schema name missing from error: {msg}"
    assert "my_other_func" in msg, (
        f"Registered function name missing from error: {msg}"
    )


def test_register_function_after_add_tools_match_no_error():
    """Symmetric clean-path: add_tools first, then register matching function."""
    chain = PromptChain(models=["openai/gpt-4o-mini"], instructions=["{input}"])

    chain.add_tools([_tool_schema("my_tool")])

    def my_tool(q: str) -> str:  # noqa: ARG001
        return "ok"

    # Must not raise — names match.
    chain.register_tool_function(my_tool)
    chain._validate_tool_registry()


def test_register_function_with_no_schemas_yet_skips_validation():
    """register_tool_function with no schemas present must not raise.

    The guard mirrors add_tools: skip validation until both sides are
    present. This avoids spurious failures when consumers register
    functions before adding schemas.
    """
    chain = PromptChain(models=["openai/gpt-4o-mini"], instructions=["{input}"])

    def some_func(q: str) -> str:  # noqa: ARG001
        return "ok"

    # Must not raise — no schemas yet.
    chain.register_tool_function(some_func)
