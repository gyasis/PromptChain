"""TEST-FIRST (Constitution III) failing tests for F3 US2 — the toolshim module +
generator wiring.

Targets ``promptchain.prompts.toolshim`` (currently an empty stub → RED) and the
shim/native branching in ``DynamicModelPromptGenerator.generate`` (not wired yet → RED).

Asserts US2 acceptance + SC-004 + FR-008/009/010 from
``specs/014-dynamic-prompt-layer/spec.md`` and the ``<tools>`` JSON-in-text shape from
``contracts/prompt-layout.md``. Offline, deterministic, seeded profiles (no live model).
"""
from __future__ import annotations

from promptchain.prompts.toolshim import (
    render_tools_block,
    resolve_tool_mode,
    serialize_history_plaintext,
)
from promptchain.prompts.model_dynamic import DynamicModelPromptGenerator
from promptchain.profiler.jacket import CapabilityProfile, Jacket

# Verbatim substring of the static base (parity floor — SC-003).
BASE_SENTENCE = "You are an expert software-engineering agent working in a terminal."

TOOLS = [
    {"function": {"name": "file_read", "description": "Read a file"}},
    {"function": {"name": "shell", "description": "Run a shell command"}},
]
OBJECTIVE = "Refactor the parser to handle empty input"


# --------------------------------------------------------------------------- #
# resolve_tool_mode (D4 / FR-010 default native)
# --------------------------------------------------------------------------- #
def _jacket(tool_mode):
    return Jacket(
        tier="tiny",
        budget_tokens=600,
        mode="single-shot+retry",
        spawn_temp=0.80,
        compress_at=0.60,
        max_turns=10,
        role="executor",
        tool_mode=tool_mode,
    )


def test_resolve_tool_mode_returns_shim_when_set():
    assert resolve_tool_mode(_jacket("shim_prompt")) == "shim_prompt"


def test_resolve_tool_mode_defaults_native_when_jacket_tool_mode_none():
    assert resolve_tool_mode(_jacket(None)) == "native"


def test_resolve_tool_mode_defaults_native_when_jacket_is_none():
    assert resolve_tool_mode(None) == "native"


# --------------------------------------------------------------------------- #
# render_tools_block (<tools> JSON-in-text protocol)
# --------------------------------------------------------------------------- #
def test_render_tools_block_has_tags_and_tool_names():
    block = render_tools_block(TOOLS)
    assert "<tools>" in block
    assert "</tools>" in block
    assert "file_read" in block
    assert "shell" in block


def test_render_tools_block_instructs_json_in_text_protocol():
    block = render_tools_block(TOOLS)
    # The JSON-in-text protocol shape per prompt-layout.md: {"tool": ..., "arguments": {...}}
    assert "tool" in block
    assert "arguments" in block


# --------------------------------------------------------------------------- #
# serialize_history_plaintext (FR-009 — re-serialize tool history as plain text)
# --------------------------------------------------------------------------- #
def _history():
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "file_read", "arguments": "{\"path\":\"x\"}"}}
            ],
        },
        {"role": "tool", "name": "file_read", "content": "file contents"},
    ]


def test_serialize_history_plaintext_is_readable_text():
    out = serialize_history_plaintext(_history())
    assert isinstance(out, str)
    assert "file_read" in out
    assert "file contents" in out
    # No native tool-call objects leaked as python reprs.
    assert "tool_calls" not in out


# --------------------------------------------------------------------------- #
# Generator wiring (SC-004 — shim vs native, mutually exclusive)
# --------------------------------------------------------------------------- #
def _shim_profile() -> CapabilityProfile:
    return CapabilityProfile(
        model_id="shim/model",
        capability=0.20,
        recommended_tier="tiny",
        budget_tokens=600,
        jacket=_jacket("shim_prompt"),
    )


def _native_profile() -> CapabilityProfile:
    return CapabilityProfile(
        model_id="native/model",
        capability=0.85,
        recommended_tier="extended",
        budget_tokens=4000,
        jacket=Jacket(
            tier="extended",
            budget_tokens=4000,
            mode="heavy-loop",
            spawn_temp=0.15,
            compress_at=0.60,
            max_turns=20,
            role="both",
            tool_mode="native",
        ),
    )


class FakeStore:
    """Minimal F2 profile-store stand-in exposing ``get_profile`` (mirrors
    tests/test_dynamic_prompt_generate.py)."""

    def __init__(self, profiles):
        self._profiles = profiles

    def get_profile(self, model_id):
        return self._profiles.get(model_id)


def _store() -> FakeStore:
    return FakeStore(
        {
            "shim/model": _shim_profile(),
            "native/model": _native_profile(),
        }
    )


# The static base legitimately contains its own `<tools>` GUIDANCE section, so the
# bare `<tools>` tag cannot discriminate native vs shim. Discriminate on the shim
# block's own distinctive content ("cannot call tools natively") so the static base
# stays VERBATIM (SC-003) — we never rewrite the foundation's tags.
SHIM_MARKER = "cannot call tools natively"


def test_generate_renders_tools_block_for_shim_model():
    gen = DynamicModelPromptGenerator(store=_store())
    out = gen.generate(OBJECTIVE, TOOLS, model="shim/model")
    assert SHIM_MARKER in out  # the <tools> JSON-in-text shim block is present
    # Parity floor intact (static base verbatim).
    assert BASE_SENTENCE in out


def test_generate_no_tools_block_for_native_model():
    gen = DynamicModelPromptGenerator(store=_store())
    out = gen.generate(OBJECTIVE, TOOLS, model="native/model")
    assert SHIM_MARKER not in out  # no shim block in native mode
    # Parity floor intact (static base verbatim).
    assert BASE_SENTENCE in out
