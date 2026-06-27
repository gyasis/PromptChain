"""TEST-FIRST (Constitution III) failing tests for the F3 generator (US1 acceptance).

Targets ``promptchain.prompts.model_dynamic.DynamicModelPromptGenerator`` (empty stub → RED).
Asserts SC-001/002/003/005/008 from ``specs/014-dynamic-prompt-layer/spec.md`` and the
``contracts/generator-api.md`` surface. Offline, deterministic, seeded profiles (no live model).
"""
from __future__ import annotations

import pytest

from promptchain.prompts import TUI_FOUNDATION_PROMPT, BasePromptBuilder  # noqa: F401
from promptchain.prompts.model_dynamic import DynamicModelPromptGenerator
from promptchain.profiler.jacket import CapabilityProfile, Jacket

# A stable verbatim substring of the static base (SC-003).
BASE_SENTENCE = "You are an expert software-engineering agent working in a terminal."
HARD_CAP = 1500

TOOLS = [{"function": {"name": "file_read", "description": "Read a file"}}]
OBJECTIVE = "Refactor the parser to handle empty input"


# --------------------------------------------------------------------------- #
# Seeded profiles + fake store
# --------------------------------------------------------------------------- #
def _strong_profile() -> CapabilityProfile:
    jacket = Jacket(
        tier="extended",
        budget_tokens=4000,
        mode="heavy-loop",
        spawn_temp=0.15,
        compress_at=0.60,
        max_turns=20,
        role="both",
        escalate=True,
    )
    return CapabilityProfile(
        model_id="strong/model",
        capability=0.85,
        recommended_tier="extended",
        budget_tokens=4000,
        jacket=jacket,
    )


def _tiny_profile() -> CapabilityProfile:
    jacket = Jacket(
        tier="tiny",
        budget_tokens=600,
        mode="single-shot+retry",
        spawn_temp=0.80,
        compress_at=0.60,
        max_turns=10,
        role="executor",
        escalate=False,
    )
    return CapabilityProfile(
        model_id="tiny/model",
        capability=0.20,
        recommended_tier="tiny",
        budget_tokens=600,
        jacket=jacket,
    )


def _null_jacket_profile() -> CapabilityProfile:
    # Profile exists, but jacket is None → fall back to recommended_tier + budget_tokens (SC-005).
    return CapabilityProfile(
        model_id="nojacket/model",
        capability=0.50,
        recommended_tier="standard",
        budget_tokens=1000,
        jacket=None,
    )


class FakeStore:
    """Minimal F2 profile-store stand-in exposing ``get_profile``."""

    def __init__(self, profiles):
        self._profiles = profiles

    def get_profile(self, model_id):
        return self._profiles.get(model_id)  # None for unknown ids


def _store() -> FakeStore:
    return FakeStore(
        {
            "strong/model": _strong_profile(),
            "tiny/model": _tiny_profile(),
            "nojacket/model": _null_jacket_profile(),
        }
    )


def _gen() -> DynamicModelPromptGenerator:
    return DynamicModelPromptGenerator(store=_store())


# --------------------------------------------------------------------------- #
# SC-003 — static base verbatim for BOTH models
# --------------------------------------------------------------------------- #
def test_static_base_verbatim_for_both_models():
    gen = _gen()
    strong = gen.generate(OBJECTIVE, TOOLS, model="strong/model")
    tiny = gen.generate(OBJECTIVE, TOOLS, model="tiny/model")
    assert BASE_SENTENCE in strong
    assert BASE_SENTENCE in tiny


def test_full_static_base_render_is_verbatim_substring():
    # SC-003 (strong form): the ENTIRE static-base render (not just the first
    # sentence) appears byte-identical in the F3 output — F3 must never rewrite or
    # mutate the foundation. Strong/tiny ids resolve to the `default` family
    # (identity framing), so the base render is a contiguous substring.
    from promptchain.prompts import DynamicTUIPromptGenerator

    base_render = DynamicTUIPromptGenerator().generate(OBJECTIVE, TOOLS)
    gen = _gen()
    for model in ("strong/model", "tiny/model"):
        out = gen.generate(OBJECTIVE, TOOLS, model=model)
        assert base_render in out, f"static base not verbatim for {model}"


# --------------------------------------------------------------------------- #
# SC-002 — per-model difference
# --------------------------------------------------------------------------- #
def test_per_model_prompts_differ():
    gen = _gen()
    strong = gen.generate(OBJECTIVE, TOOLS, model="strong/model")
    tiny = gen.generate(OBJECTIVE, TOOLS, model="tiny/model")
    assert strong != tiny


# --------------------------------------------------------------------------- #
# SC-001 — budget compliance
# --------------------------------------------------------------------------- #
def test_token_estimate_within_hard_cap_both_models():
    gen = _gen()
    strong_est = gen.get_token_estimate(OBJECTIVE, TOOLS)  # default builder model
    assert strong_est >= 0
    # Per-model estimates via fresh model-bound builders.
    strong = DynamicModelPromptGenerator(store=_store(), model="strong/model")
    tiny = DynamicModelPromptGenerator(store=_store(), model="tiny/model")
    se = strong.get_token_estimate(OBJECTIVE, TOOLS)
    te = tiny.get_token_estimate(OBJECTIVE, TOOLS)
    assert se <= HARD_CAP
    assert te <= HARD_CAP


def test_tiny_respects_tighter_budget_than_strong():
    from promptchain.prompts.budget import measure

    gen = _gen()
    strong = gen.generate(OBJECTIVE, TOOLS, model="strong/model")
    tiny = gen.generate(OBJECTIVE, TOOLS, model="tiny/model")
    assert measure(strong) <= HARD_CAP
    assert measure(tiny) <= HARD_CAP
    # The tiny jacket has a tighter budget → its prompt is no larger than the strong one.
    assert measure(tiny) <= measure(strong)


# --------------------------------------------------------------------------- #
# SC-005 — fallbacks never raise
# --------------------------------------------------------------------------- #
def test_null_jacket_generates_without_raising():
    gen = _gen()
    out = gen.generate(OBJECTIVE, TOOLS, model="nojacket/model")
    assert BASE_SENTENCE in out


def test_no_profile_generates_without_raising_default_core():
    gen = _gen()  # store returns None for unknown id
    out = gen.generate(OBJECTIVE, TOOLS, model="unknown/model-xyz")
    assert BASE_SENTENCE in out


# --------------------------------------------------------------------------- #
# SC-008 — determinism
# --------------------------------------------------------------------------- #
def test_generate_is_deterministic():
    gen = _gen()
    a = gen.generate(OBJECTIVE, TOOLS, model="strong/model")
    b = gen.generate(OBJECTIVE, TOOLS, model="strong/model")
    assert a == b


# --------------------------------------------------------------------------- #
# BasePromptBuilder conformance + empty-objective rejection
# --------------------------------------------------------------------------- #
def test_conforms_to_base_prompt_builder_surface():
    gen = _gen()
    # BasePromptBuilder is a non-runtime-checkable Protocol → assert the structural surface.
    assert callable(getattr(gen, "generate", None))
    assert callable(getattr(gen, "get_token_estimate", None))
    # Protocol positional call (objective, tools, context=None) must work.
    out = gen.generate(OBJECTIVE, TOOLS, None)
    assert isinstance(out, str) and out


def test_empty_objective_raises_value_error():
    gen = _gen()
    with pytest.raises(ValueError):
        gen.generate("", TOOLS, model="strong/model")
    with pytest.raises(ValueError):
        gen.generate("   ", TOOLS, model="strong/model")
