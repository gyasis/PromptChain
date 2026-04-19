"""Tests for the prompt-builder package (spec 011-agentic-prompt-builder).

Wave 2 skeleton — protocol shape assertions only. Later waves extend this
file with user-story specific tests.
"""

from typing import get_type_hints

from promptchain.prompts.base import BasePromptBuilder


def _custom_tools() -> list:
    """Four non-TUI tool schemas used by US1 tests."""
    return [
        {
            "type": "function",
            "function": {
                "name": "customer_lookup",
                "description": "Look up a customer record by ID.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "invoice_totals",
                "description": "Aggregate invoice totals for a customer.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_followup_email",
                "description": "Send a templated follow-up email.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "body"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "log_audit_event",
                "description": "Append an entry to the compliance audit log.",
                "parameters": {
                    "type": "object",
                    "properties": {"event": {"type": "string"}},
                    "required": ["event"],
                },
            },
        },
    ]


def test_protocol_has_required_methods() -> None:
    """BasePromptBuilder must expose `generate` and `get_token_estimate`."""
    assert hasattr(BasePromptBuilder, "generate")
    assert hasattr(BasePromptBuilder, "get_token_estimate")
    assert callable(getattr(BasePromptBuilder, "generate"))
    assert callable(getattr(BasePromptBuilder, "get_token_estimate"))


def test_protocol_structural_type_check() -> None:
    """A plain class exposing the right methods must satisfy the Protocol.

    Because `BasePromptBuilder` is a structural `typing.Protocol`, any class
    with matching method signatures should be accepted as a compatible
    implementation without explicit subclassing.
    """

    class _Dummy:
        def generate(self, objective, tools, context=None):
            return "prompt"

        def get_token_estimate(self, objective, tools):
            return 0

    dummy: BasePromptBuilder = _Dummy()  # type: ignore[assignment]
    assert dummy.generate("obj", []) == "prompt"
    assert dummy.get_token_estimate("obj", []) == 0

    # Protocol hints resolve without raising.
    hints = get_type_hints(BasePromptBuilder.generate)
    assert "return" in hints


def test_dynamic_renders_registered_tools() -> None:
    """DynamicPromptGenerator output must list every registered tool name."""
    from promptchain.prompts import DynamicPromptGenerator

    tools = _custom_tools()
    prompt = DynamicPromptGenerator().generate("ship the release", tools)

    assert "AVAILABLE TOOLS" in prompt
    for tool in tools:
        name = tool["function"]["name"]
        assert name in prompt, f"registered tool {name!r} missing from prompt"
