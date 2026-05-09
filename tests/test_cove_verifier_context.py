"""Tests for `verifier_context` — per-step CoVe verification prompt context (Issue #4).

Background: after the v0.6.1 lambda-signature fix, CoVe ran cleanly but began
over-rejecting legitimate, hand-vetted tool calls in chains using per-step tool
scoping `(step, ["tool_a", "tool_b"])`. Root cause: the verifier sees only the
tool name + JSON schema + args — it never sees the AgenticStepProcessor's
objective. With no domain context it conservatively defaults to
``should_execute=False`` even at confidence=1.00.

Fix (Option A from PRD `promptchain_cove_verifier_context_2026-05-08`): a new
``verifier_context: str`` kwarg on ``AgenticStepProcessor`` is plumbed down to
``CoVeVerifier`` as ``extra_context``, where ``_build_verification_prompt``
appends it under a "## Step Context" heading.

These tests cover three behaviours:
    (a) empty verifier_context → identical prompt to current behaviour
    (b) populated verifier_context → flows into the verification prompt
    (c) AgenticStepProcessor wires the kwarg through to CoVeVerifier on lazy init
"""
from __future__ import annotations

from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.verification import CoVeVerifier


def _make_verifier(extra_context=None):
    """CoVeVerifier with a no-op llm_runner — we only inspect prompt-building, not LLM calls."""
    async def noop_runner(messages, model=None, **kwargs):
        return {"content": ""}
    return CoVeVerifier(
        llm_runner=noop_runner,
        model_name="openai/gpt-4o-mini",
        extra_context=extra_context,
    )


def test_empty_verifier_context_preserves_current_behaviour():
    """No extra_context → no '## Step Context' section in the verification prompt.

    Backward-compat guard: existing CoVe users (no v0.6.1 per-step scoping)
    must see an unchanged prompt.
    """
    verifier = _make_verifier(extra_context=None)
    prompt = verifier._build_verification_prompt(
        tool_name="my_tool",
        tool_args={"a": 1},
        tool_schema={"name": "my_tool"},
        context="some current context",
    )
    assert "## Step Context" not in prompt
    assert "my_tool" in prompt  # sanity — base prompt still built


def test_populated_verifier_context_flows_into_prompt():
    """Populated extra_context → appended under '## Step Context' heading."""
    ctx = (
        "You are part of REMEDIATE_PRE in a self-healing pipeline. "
        "All available fixes are pre-vetted and idempotent."
    )
    verifier = _make_verifier(extra_context=ctx)
    prompt = verifier._build_verification_prompt(
        tool_name="fix_set_hh_worktree_mode",
        tool_args={},
        tool_schema={"name": "fix_set_hh_worktree_mode"},
        context="some current context",
    )
    assert "## Step Context" in prompt
    assert ctx in prompt
    # Heading must come AFTER the JSON-only instruction so the verifier still
    # treats it as supplementary context (not a replacement for the checklist).
    assert prompt.index("VERIFICATION CHECKLIST") < prompt.index("## Step Context")


def test_agentic_step_processor_plumbs_verifier_context_to_cove_verifier():
    """The kwarg on AgenticStepProcessor must reach CoVeVerifier.extra_context.

    Exercises the lazy-init wiring in ``_ensure_cove_verifier`` end-to-end without
    spinning up a chain.
    """
    ctx = "Step-specific safety hint."
    asp = AgenticStepProcessor(
        objective="Demo objective.",
        model_name="openai/gpt-4o-mini",
        enable_cove=True,
        verifier_context=ctx,
    )
    assert asp.verifier_context == ctx
    assert asp.cove_verifier is None  # lazy

    async def noop_runner(messages, model=None, **kwargs):
        return {"content": ""}

    asp._ensure_cove_verifier(noop_runner)
    assert asp.cove_verifier is not None
    assert asp.cove_verifier.extra_context == ctx


def test_verifier_context_default_is_none_backward_compat():
    """Existing AgenticStepProcessor(enable_cove=True) callers must keep working
    without the new kwarg — extra_context defaults to None on the verifier."""
    asp = AgenticStepProcessor(
        objective="Demo.",
        model_name="openai/gpt-4o-mini",
        enable_cove=True,
    )
    assert asp.verifier_context is None

    async def noop_runner(messages, model=None, **kwargs):
        return {"content": ""}

    asp._ensure_cove_verifier(noop_runner)
    assert asp.cove_verifier.extra_context is None
