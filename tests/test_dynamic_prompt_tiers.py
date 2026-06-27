"""TEST-FIRST (Constitution III) failing tests for F3 tier selection + optional modules.

Targets ``promptchain.prompts.tiers`` — currently an empty stub, so these tests are
RED until the module is implemented. Asserts the contract in
``specs/014-dynamic-prompt-layer/data-model.md`` (PromptTier, OptionalModule) and
``contracts/generator-api.md`` (select_tier, modules_for_tier).
"""
from __future__ import annotations

from enum import Enum

from promptchain.prompts.tiers import (
    OptionalModule,
    PromptTier,
    modules_for_tier,
    select_tier,
)
from promptchain.profiler.jacket import CapabilityProfile


# --------------------------------------------------------------------------- #
# PromptTier shape
# --------------------------------------------------------------------------- #
def test_prompt_tier_is_str_enum_with_members():
    assert issubclass(PromptTier, Enum)
    assert issubclass(PromptTier, str)
    assert {m.name for m in PromptTier} >= {"CORE", "EXTENDED", "TINY"}
    # str-enum: members compare/behave as strings
    assert isinstance(PromptTier.CORE, str)


# --------------------------------------------------------------------------- #
# OptionalModule dataclass shape
# --------------------------------------------------------------------------- #
def test_optional_module_fields():
    m = OptionalModule(key="examples", text="some text", drop_priority=1, tiers={PromptTier.EXTENDED})
    assert m.key == "examples"
    assert m.text == "some text"
    assert m.drop_priority == 1
    assert PromptTier.EXTENDED in m.tiers


# --------------------------------------------------------------------------- #
# select_tier — capability thresholds (D2)
# --------------------------------------------------------------------------- #
def test_select_tier_extended_band():
    assert select_tier(CapabilityProfile(model_id="m", capability=0.85)) is PromptTier.EXTENDED
    assert select_tier(CapabilityProfile(model_id="m", capability=0.66)) is PromptTier.EXTENDED


def test_select_tier_core_band():
    assert select_tier(CapabilityProfile(model_id="m", capability=0.50)) is PromptTier.CORE
    assert select_tier(CapabilityProfile(model_id="m", capability=0.33)) is PromptTier.CORE
    assert select_tier(CapabilityProfile(model_id="m", capability=0.65)) is PromptTier.CORE


def test_select_tier_tiny_band():
    assert select_tier(CapabilityProfile(model_id="m", capability=0.20)) is PromptTier.TINY
    assert select_tier(CapabilityProfile(model_id="m", capability=0.0)) is PromptTier.TINY


def test_select_tier_none_profile_defaults_core():
    assert select_tier(None) is PromptTier.CORE


# --------------------------------------------------------------------------- #
# modules_for_tier — per-tier module sets + drop ordering (D5)
# --------------------------------------------------------------------------- #
def test_modules_for_extended_includes_examples_and_extended_guidance():
    keys = {m.key for m in modules_for_tier(PromptTier.EXTENDED)}
    assert "examples" in keys
    assert "extended_guidance" in keys


def test_modules_for_core_is_leaner_no_examples():
    core_keys = {m.key for m in modules_for_tier(PromptTier.CORE)}
    assert "examples" not in core_keys


def test_modules_for_tiny_no_more_than_core():
    tiny = modules_for_tier(PromptTier.TINY)
    core = modules_for_tier(PromptTier.CORE)
    assert len(tiny) <= len(core)


def test_examples_dropped_before_extended_guidance():
    by_key = {m.key: m for m in modules_for_tier(PromptTier.EXTENDED)}
    assert by_key["examples"].drop_priority < by_key["extended_guidance"].drop_priority


def test_modules_for_tier_returns_list():
    assert isinstance(modules_for_tier(PromptTier.EXTENDED), list)
    assert all(isinstance(m, OptionalModule) for m in modules_for_tier(PromptTier.EXTENDED))
