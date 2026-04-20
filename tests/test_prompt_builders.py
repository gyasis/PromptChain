"""Tests for the prompt-builder package (spec 011-agentic-prompt-builder).

Wave 2 skeleton — protocol shape assertions only. Later waves extend this
file with user-story specific tests.
"""

from typing import Any, Dict, List, get_type_hints

import pytest

from promptchain.prompts.base import BasePromptBuilder


_TUI_ONLY_TOOL_NAMES = (
    "ripgrep_search",
    "file_read",
    "file_write",
    "file_edit",
    "terminal_execute",
    "list_directory",
    "create_directory",
)


def _tool(name: str, description: str = "") -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"Tool named {name}.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _make_tools(n: int) -> List[Dict[str, Any]]:
    """Return ``n`` distinct custom tool schemas for parametrized tests."""
    return [_tool(f"custom_tool_{i:02d}", f"Description for custom tool {i}.") for i in range(n)]


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


@pytest.mark.parametrize("n", [0, 1, 4, 10])
def test_dynamic_renders_registered_tools(n: int) -> None:
    """DynamicPromptGenerator output must list every registered tool name (SC-001).

    Parametrized (T064) over tool-count cases 0/1/4/10:
    - n=0: empty inventory must still render AVAILABLE TOOLS header with sentinel
    - n>=1: every registered tool name appears in the prompt body
    """
    from promptchain.prompts import DynamicPromptGenerator

    tools = _make_tools(n)
    prompt = DynamicPromptGenerator().generate("ship the release", tools)

    assert "AVAILABLE TOOLS" in prompt
    for tool in tools:
        name = tool["function"]["name"]
        assert name in prompt, f"registered tool {name!r} missing from prompt"


def test_dynamic_does_not_advertise_tui_only_tools() -> None:
    """No TUI-only tool names may appear when only custom tools are registered (T008)."""
    from promptchain.prompts import DynamicPromptGenerator

    tools = _make_tools(4)
    prompt = DynamicPromptGenerator().generate("ship the release", tools)

    for name in _TUI_ONLY_TOOL_NAMES:
        assert name not in prompt, (
            f"TUI-only tool {name!r} leaked into prompt without being registered"
        )


def test_dynamic_empty_tools_renders_sentinel() -> None:
    """Empty tool inventory must render a sentinel line, not a blank block (T009)."""
    from promptchain.prompts import DynamicPromptGenerator

    prompt = DynamicPromptGenerator().generate("ship the release", [])

    assert "AVAILABLE TOOLS" in prompt, "AVAILABLE TOOLS header missing"
    # Sentinel must be an explicit "no tools" line, never a blank block.
    lowered = prompt.lower()
    assert ("no tools" in lowered) or ("(none)" in lowered) or ("no tools registered" in lowered), (
        "empty inventory must render an explicit sentinel line, not a blank block"
    )
