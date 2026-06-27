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
def test_static_base_verbatim_for_core_extended_only():
    # SC-003 holds for CORE/EXTENDED: the full foundation appears verbatim. TINY is
    # the sanctioned exception (decided 2026-06-27) — it SWAPS to a simpler base, so
    # the foundation's BASE_SENTENCE is intentionally absent for a TINY model.
    gen = _gen()
    strong = gen.generate(OBJECTIVE, TOOLS, model="strong/model")  # EXTENDED
    tiny = gen.generate(OBJECTIVE, TOOLS, model="tiny/model")      # TINY
    assert BASE_SENTENCE in strong
    assert BASE_SENTENCE not in tiny, "TINY must swap to the simpler base, not the full foundation"


def test_full_static_base_render_is_verbatim_substring_for_extended():
    # SC-003 (strong form) for CORE/EXTENDED: the ENTIRE foundation render appears
    # byte-identical in the F3 output — F3 must never rewrite/mutate the foundation
    # for these tiers. (TINY swaps the base entirely, so this does not apply there.)
    from promptchain.prompts import DynamicTUIPromptGenerator

    base_render = DynamicTUIPromptGenerator().generate(OBJECTIVE, TOOLS)
    out = _gen().generate(OBJECTIVE, TOOLS, model="strong/model")
    assert base_render in out, "foundation not verbatim for EXTENDED model"


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
# TINY tier — focusing directive in the never-dropped floor (SC-007 aid)
# --------------------------------------------------------------------------- #
def test_tiny_tier_uses_simpler_shorter_base():
    # TINY swaps to a short replacement base (Goose tiny_model_system). It must:
    # (a) carry the TINY focusing directive, (b) NOT carry the full foundation, and
    # (c) be MUCH shorter than the full foundation — the whole point (a weak model
    # is derailed by the big base; additive-only F3 measurably HURT it).
    from promptchain.prompts.tiers import TINY_DIRECTIVE
    from promptchain.prompts import DynamicTUIPromptGenerator
    from promptchain.prompts.budget import measure

    gen = _gen()
    tiny = gen.generate(OBJECTIVE, TOOLS, model="tiny/model")
    full_base = DynamicTUIPromptGenerator().generate(OBJECTIVE, TOOLS)
    assert TINY_DIRECTIVE in tiny
    assert BASE_SENTENCE not in tiny
    assert measure(tiny) < measure(full_base), "TINY base must be shorter than the full foundation"
    # The directive is TINY-only — a strong (EXTENDED) model keeps the full base.
    strong = gen.generate(OBJECTIVE, TOOLS, model="strong/model")
    assert TINY_DIRECTIVE not in strong


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
