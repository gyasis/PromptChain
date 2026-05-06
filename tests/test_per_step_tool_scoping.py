"""Tests for per-step tool scoping (the (instruction, [tool_names]) tuple form).

Closes the gap originally documented in
docs/llms/issues/PROPOSAL_per_step_tool_scoping.md and
docs/llms/PROMPTCHAIN_FOR_LLMS.md §9 anti-pattern #14.

Behaviour contract:
- Bare instruction → step sees ALL chain-scoped tools (default).
- (instruction, []) → step sees ZERO tools.
- (instruction, ["a", "b"]) → step sees ONLY tools whose function.name is "a" or "b".
- model_instruction_count must unwrap tuples for slot-matching.
"""
from __future__ import annotations

from promptchain.utils.promptchaining import _unwrap_instruction, PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


# ---------- _unwrap_instruction unit tests ----------

def test_unwrap_bare_string():
    inst, scope = _unwrap_instruction("hello {input}")
    assert inst == "hello {input}"
    assert scope is None


def test_unwrap_bare_callable():
    def f(x: str) -> str: return x
    inst, scope = _unwrap_instruction(f)
    assert inst is f
    assert scope is None


def test_unwrap_bare_agentic():
    a = AgenticStepProcessor(objective="x", model_name="openai/gpt-4o-mini")
    inst, scope = _unwrap_instruction(a)
    assert inst is a
    assert scope is None


def test_unwrap_tuple_empty_scope():
    inst, scope = _unwrap_instruction(("hi", []))
    assert inst == "hi"
    assert scope == []


def test_unwrap_tuple_named_scope():
    inst, scope = _unwrap_instruction(("hi", ["a", "b"]))
    assert inst == "hi"
    assert scope == ["a", "b"]


def test_unwrap_non_scoped_tuple_passes_through():
    """A (str, str) tuple is NOT a scope — only (instruction, list) is."""
    weird = ("hello", "world")
    inst, scope = _unwrap_instruction(weird)
    assert inst is weird
    assert scope is None


# ---------- PromptChain construction with tuple-form instructions ----------

def test_construction_accepts_tuple_form():
    """Mixed bare + tuple instructions construct without error."""
    def f(x: str) -> str: return x
    # Two string instructions: one bare, one scoped. f is a function (no model slot).
    chain = PromptChain(
        models=["openai/gpt-4o-mini", "openai/gpt-4o"],
        instructions=["bare: {input}", ("scoped: {input}", []), f],
    )
    assert len(chain.instructions) == 3
    # Single-model auto-expansion case
    chain2 = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["bare: {input}", ("scoped: {input}", ["a"])],
    )
    assert len(chain2.models) == 2  # auto-expanded for 2 string instructions


def test_construction_model_count_validates_against_unwrapped():
    """Validator counts string instructions correctly when wrapped in tuples."""
    # 1 string-in-tuple + 1 function — needs exactly 1 model.
    def f(x: str) -> str: return x
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[("hi: {input}", []), f],
    )
    assert len(chain.models) == 1  # NOT auto-expanded — we already had exactly 1


def test_construction_agentic_in_tuple_does_not_consume_model_slot():
    """Wrapping an AgenticStepProcessor in a tuple doesn't change the model count."""
    a = AgenticStepProcessor(objective="x", model_name="openai/gpt-4o-mini")
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[("plain: {input}", []), (a, ["tool_a"])],
    )
    # Only 1 string instruction → 1 model needed (auto-expanded but stays 1)
    assert len(chain.models) == 1
