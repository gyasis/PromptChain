"""TEST-FIRST (Constitution III) failing tests for F3 family adapter.

Targets ``promptchain.prompts.family`` (empty stub → RED). Asserts the contract in
``contracts/generator-api.md`` (family_of, adapt_format) and data-model FamilyAdapter:
adapt_format adjusts FORMAT ONLY — it must never drop or alter the base content (FR-003).
"""
from __future__ import annotations

import pytest

from promptchain.prompts.family import adapt_format, family_of


# --------------------------------------------------------------------------- #
# family_of — model id → family stem
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("anthropic/claude-3-5-sonnet", "anthropic"),
        ("claude-opus-4", "anthropic"),
        ("openai/gpt-4o", "openai"),
        ("gpt-4", "openai"),
        ("o1-preview", "openai"),
        ("gemini-2.0-flash", "google"),
        ("ollama/qwen3-coder:30b", "qwen"),
        ("ollama/llama3.2:1b", "llama"),
        ("some-unknown-model", "default"),
    ],
)
def test_family_of(model_id, expected):
    assert family_of(model_id) == expected


def test_family_of_returns_known_set():
    known = {"anthropic", "openai", "google", "qwen", "llama", "default"}
    for mid in ["claude-x", "gpt-x", "gemini-x", "qwen-x", "llama-x", "zzz"]:
        assert family_of(mid) in known


# --------------------------------------------------------------------------- #
# adapt_format — format-only normalization (never drops/alters base content)
# --------------------------------------------------------------------------- #
PARTS = [
    "You are an expert agent.",
    "Work loop: THINK, PLAN, ACT, OBSERVE.",
    "AVAILABLE TOOLS:\n- file_read: Read a file",
]


@pytest.mark.parametrize(
    "family",
    ["anthropic", "openai", "google", "qwen", "llama", "default", "totally-unknown-family"],
)
def test_adapt_format_preserves_every_part_verbatim(family):
    out = adapt_format(list(PARTS), family)
    assert isinstance(out, list)
    joined = "\n".join(out)
    for part in PARTS:
        assert part in joined, f"part dropped/altered under family={family!r}"


def test_adapt_format_default_is_identity():
    assert adapt_format(list(PARTS), "default") == PARTS


def test_adapt_format_unknown_family_is_format_safe():
    # Unknown families fall back to the agnostic core: content fully preserved.
    out = adapt_format(list(PARTS), "totally-unknown-family")
    joined = "\n".join(out)
    for part in PARTS:
        assert part in joined


def test_adapt_format_known_families_are_real_variants():
    # Per-family variants over the `default` fallback (PRD §7): a known family
    # produces a measurably DIFFERENT format than `default`, and two distinct
    # known families differ from each other — while still preserving content.
    base = adapt_format(list(PARTS), "default")
    anthropic = adapt_format(list(PARTS), "anthropic")
    openai = adapt_format(list(PARTS), "openai")
    assert anthropic != base, "known family must differ from the default fallback"
    assert openai != base
    assert anthropic != openai, "distinct families must produce distinct framing"
    # content still verbatim under the real variants
    for variant in (anthropic, openai):
        joined = "\n".join(variant)
        for part in PARTS:
            assert part in joined
