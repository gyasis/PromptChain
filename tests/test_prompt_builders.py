"""Tests for the prompt-builder package (spec 011-agentic-prompt-builder).

Wave 2 skeleton — protocol shape assertions only. Later waves extend this
file with user-story specific tests.
"""

from typing import get_type_hints

from promptchain.prompts.base import BasePromptBuilder


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
