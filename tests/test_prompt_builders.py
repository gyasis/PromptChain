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


def test_dynamic_standard_mode_omits_react_block() -> None:
    """Standard workflow_pattern must not emit any ReAct scaffold sections (T010)."""
    from promptchain.prompts import DynamicPromptGenerator

    prompt = DynamicPromptGenerator(workflow_pattern="standard").generate(
        "ship the release", _make_tools(3)
    )
    upper = prompt.upper()
    forbidden = ("REACT WORKFLOW", "STEP 1 - THINK", "STEP 2 - PLAN", "STEP 3 - ACT", "STEP 4 - OBSERVE")
    for marker in forbidden:
        assert marker not in upper, f"standard mode leaked ReAct marker: {marker!r}"


def test_dynamic_react_with_tasklist_tool_renders_scaffold() -> None:
    """ReAct mode with a task-list tool registered must render the full scaffold (T011)."""
    from promptchain.prompts import DynamicPromptGenerator

    tools = _make_tools(2) + [_tool("task_list_write_tool", "Write or update the task list.")]
    prompt = DynamicPromptGenerator(workflow_pattern="react").generate("ship the release", tools)

    upper = prompt.upper()
    # Full scaffold should be present when a task-list tool is registered.
    assert "THINK" in upper, "ReAct THINK step missing"
    assert "ACT" in upper, "ReAct ACT step missing"
    assert "OBSERVE" in upper, "ReAct OBSERVE step missing"
    # task-list tool name appears in the AVAILABLE TOOLS block.
    assert "task_list_write_tool" in prompt


def test_dynamic_react_without_tasklist_warns_and_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ReAct mode WITHOUT a task-list tool must log a warning and emit the
    minimal thought/action scaffold instead of the full ReAct scaffold (T012).
    """
    from promptchain.prompts import DynamicPromptGenerator

    tools = _make_tools(3)  # no task-list-writer in inventory
    with caplog.at_level("WARNING", logger="promptchain.prompts.dynamic"):
        prompt = DynamicPromptGenerator(workflow_pattern="react").generate(
            "ship the release", tools
        )

    # (a) A warning must have fired naming the trigger and the fallback.
    matching = [
        rec for rec in caplog.records
        if rec.levelname == "WARNING"
        and "react" in rec.getMessage().lower()
        and "task" in rec.getMessage().lower()
        and "fall" in rec.getMessage().lower()  # "Falling back"
    ]
    assert len(matching) == 1, (
        f"expected exactly 1 react/task-list/fallback warning, got "
        f"{[r.getMessage() for r in caplog.records]}"
    )

    # (b) Rendered prompt must be the MINIMAL scaffold, not the full one.
    assert "REASONING PATTERN (minimal)" in prompt, "minimal scaffold header missing"
    assert "THOUGHT" in prompt, "minimal scaffold THOUGHT marker missing"
    assert "ACTION" in prompt, "minimal scaffold ACTION marker missing"
    # Full-scaffold-only markers must NOT appear.
    assert "REASONING PATTERN (ReAct)" not in prompt, "full ReAct header leaked into fallback"
    assert "PLAN:" not in prompt, "full-scaffold PLAN step leaked into fallback"
    assert "OBSERVE:" not in prompt, "full-scaffold OBSERVE step leaked into fallback"


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
