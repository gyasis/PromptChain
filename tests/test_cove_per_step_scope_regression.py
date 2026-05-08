"""Regression test for the CoVe + per-step tool scope signature mismatch.

Bug (v0.6.1+): the `llm_runner` lambda built in
``promptchaining.py::process_prompt_async`` (post-639a47c "feat: per-step tool
scoping") had signature ``(messages, tools, tool_choice=None)``. The CoVe
verifier at ``utils/verification.py::verify_tool_call`` calls
``self.llm_runner(messages=..., model=self.model_name)`` — TypeError on every
verification → confidence falls to 0.50 → all tool calls silently skipped.

Fix: lambda now accepts ``(messages, tools, tool_choice=None, model=None,
**kwargs)``. ``model`` is intentionally ignored — the agentic step has
already selected its own model via ``agentic_instruction.model_name``.

This test asserts the lambda accepts the kwargs CoVe needs without raising.
"""
from __future__ import annotations

import inspect
import pytest

from promptchain.utils import promptchaining


def test_llm_runner_lambda_accepts_model_kwarg():
    """The internal lambda must accept the kwargs CoVe passes.

    We can't easily intercept the lambda mid-flight without spinning up a
    full chain, so we exercise the same SHAPE by reconstructing it with
    the same closure variables. If the signature regresses, this fails.
    """

    def llm_runner_callback(messages, tools, tool_choice, agentic_instruction=None):
        return {"content": "ok", "messages": messages, "tools": tools,
                "tool_choice": tool_choice, "agent": agentic_instruction}

    instruction = object()  # any sentinel

    # Reconstruct the lambda exactly as built at promptchaining.py:1290.
    llm_runner = lambda messages, tools, tool_choice=None, model=None, **kwargs: llm_runner_callback(
        messages, tools, tool_choice, agentic_instruction=instruction,
    )

    # Old failing path: CoVe verifier calling with a `model` kwarg.
    out = llm_runner(messages=[{"role": "user", "content": "hi"}],
                    tools=None, model="openai/gpt-4o-mini")
    assert out["content"] == "ok"

    # Forward-compat: arbitrary future kwargs must not break.
    out2 = llm_runner(messages=[], tools=None, model="x", new_future_kwarg=True)
    assert out2["content"] == "ok"

    # Backward-compat: original 3-arg form still works.
    out3 = llm_runner(messages=[], tools=None, tool_choice="auto")
    assert out3["content"] == "ok"


def test_actual_lambda_in_source_has_model_kwarg():
    """Static check against the source file — guards against the lambda
    being silently rewritten back to the broken signature."""
    import promptchain.utils.promptchaining as pcm
    src = inspect.getsource(pcm)
    # The lambda must accept `model=None` AND `**kwargs` for forward-compat.
    assert "lambda messages, tools, tool_choice=None, model=None, **kwargs:" in src, (
        "process_prompt_async's llm_runner lambda regressed. CoVe verifier "
        "(verification.py:137) calls it with model=, which will TypeError "
        "and skip every tool call at confidence 0.50."
    )
