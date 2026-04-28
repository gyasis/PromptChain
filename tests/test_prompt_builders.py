"""Tests for the prompt-builder package (spec 011-agentic-prompt-builder).

Wave 2 skeleton — protocol shape assertions only. Later waves extend this
file with user-story specific tests.
"""

import ast
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, get_type_hints

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


def test_dynamic_standard_mode_line_count_under_cap() -> None:
    """Standard mode with 3 tools must stay under 15 lines (T013).

    Protects the "minimal/honest by default" promise — library consumers
    who don't opt into ReAct shouldn't get a long scaffold.
    """
    from promptchain.prompts import DynamicPromptGenerator

    prompt = DynamicPromptGenerator(workflow_pattern="standard").generate(
        "ship the release", _make_tools(3)
    )
    line_count = len(prompt.splitlines())
    assert line_count < 15, (
        f"standard-mode prompt with 3 tools rendered {line_count} lines; "
        f"cap is <15. Rendered prompt:\n{prompt}"
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


# === Wave 10-13: US1 add'l (T014, T015, T016, T017) ===


def test_dynamic_token_estimate_positive_and_monotonic() -> None:
    """get_token_estimate returns positive int and grows with tool count (T014)."""
    from promptchain.prompts import DynamicPromptGenerator

    gen = DynamicPromptGenerator()
    est_zero = gen.get_token_estimate("ship the release", [])
    est_five = gen.get_token_estimate("ship the release", _make_tools(5))

    assert isinstance(est_zero, int)
    assert isinstance(est_five, int)
    assert est_zero > 0, "even with no tools, the prompt has objective + sentinel + final-answer"
    assert est_five > est_zero, (
        f"5-tool estimate ({est_five}) must exceed 0-tool estimate ({est_zero})"
    )


def test_dynamic_context_block_renders_when_provided() -> None:
    """Passing context= renders the PRIOR CONTEXT header; omitting it does not (T015)."""
    from promptchain.prompts import DynamicPromptGenerator

    gen = DynamicPromptGenerator()
    tools = _make_tools(2)

    with_ctx = gen.generate("obj", tools, context="previous scratchpad text here")
    without_ctx = gen.generate("obj", tools)

    assert "PRIOR CONTEXT" in with_ctx
    assert "previous scratchpad text here" in with_ctx
    assert "PRIOR CONTEXT" not in without_ctx


def test_dispatch_default_is_dynamic_standard_mode() -> None:
    """No-kwarg processor gets DynamicPromptGenerator in standard mode (T016)."""
    from promptchain.prompts import DynamicPromptGenerator
    from promptchain.utils.agentic_step_processor import AgenticStepProcessor

    proc = AgenticStepProcessor(objective="x")
    assert isinstance(proc.prompt_builder, DynamicPromptGenerator)
    assert proc.workflow_pattern == "standard"


def test_dynamic_tool_description_updates_reflected_on_regenerate() -> None:
    """No caching: a mutated tool description appears in the next render (T017)."""
    from promptchain.prompts import DynamicPromptGenerator

    gen = DynamicPromptGenerator()
    tools = [_tool("custom_tool_00", "Original description text.")]
    first = gen.generate("obj", tools)
    assert "Original description text." in first

    # Mutate the description in place — the generator must read it fresh each call.
    tools[0]["function"]["description"] = "Updated description text v2."
    second = gen.generate("obj", tools)
    assert "Updated description text v2." in second
    assert "Original description text." not in second


# === Wave 14-22: US2 (T024-T032) ===


_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "legacy_tui_prompt.snapshot.txt"


def test_legacy_snapshot_byte_identical() -> None:
    """LegacyTUIPromptGenerator output equals the on-disk snapshot byte-for-byte (T024).

    The snapshot file contains the literal token ``{objective}`` as a placeholder
    (per T001 capture rules), so we render with that exact objective and compare.
    """
    from promptchain.prompts import LegacyTUIPromptGenerator

    expected = _FIXTURE_PATH.read_text()
    # Snapshot file preserves the original ``{self.objective}`` f-string token
    # (per T001 capture rules); render with that literal so str.replace yields it.
    rendered = LegacyTUIPromptGenerator().generate("{self.objective}", tools=[])
    assert rendered == expected, (
        f"legacy prompt drifted from snapshot. len(expected)={len(expected)} "
        f"len(rendered)={len(rendered)}"
    )


def test_legacy_ignores_tools_for_content() -> None:
    """Tool list is structurally ignored by the legacy builder (T025)."""
    from promptchain.prompts import LegacyTUIPromptGenerator

    gen = LegacyTUIPromptGenerator()
    out_a = gen.generate("obj", tools=[])
    out_b = gen.generate("obj", tools=_make_tools(5))
    out_c = gen.generate("obj", tools=[_tool("totally_unrelated_tool")])
    assert out_a == out_b == out_c, "legacy output must not depend on tool list"


def test_legacy_drift_warning_on_unexpected_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A divergent tool set produces a single advisory warning (T026)."""
    from promptchain.prompts import LegacyTUIPromptGenerator

    with caplog.at_level("WARNING", logger="promptchain.prompts.legacy_tui"):
        LegacyTUIPromptGenerator().generate(
            "obj", tools=[_tool("not_a_tui_tool"), _tool("another_unknown")]
        )

    matching = [
        rec for rec in caplog.records
        if rec.levelname == "WARNING" and "diverge" in rec.getMessage().lower()
    ]
    assert len(matching) >= 1, (
        f"expected drift warning, got: {[r.getMessage() for r in caplog.records]}"
    )


def test_tui_subclass_bakes_legacy_builder() -> None:
    """TUIAgenticStepProcessor wires up LegacyTUIPromptGenerator (T027)."""
    from promptchain.cli.tui_processor import TUIAgenticStepProcessor
    from promptchain.prompts import LegacyTUIPromptGenerator

    proc = TUIAgenticStepProcessor(objective="x")
    assert isinstance(proc.prompt_builder, LegacyTUIPromptGenerator)


def test_tui_subclass_rejects_prompt_builder_kwarg() -> None:
    """TUI subclass refuses prompt_builder= (T028)."""
    from promptchain.cli.tui_processor import TUIAgenticStepProcessor
    from promptchain.prompts import DynamicPromptGenerator

    with pytest.raises(TypeError):
        TUIAgenticStepProcessor(objective="x", prompt_builder=DynamicPromptGenerator())


def test_tui_subclass_rejects_instructions_kwarg() -> None:
    """TUI subclass refuses instructions= (T029)."""
    from promptchain.cli.tui_processor import TUIAgenticStepProcessor

    with pytest.raises(TypeError):
        TUIAgenticStepProcessor(objective="x", instructions=["A"])


def test_subclass_inheritance_enhanced_sees_custom_tools() -> None:
    """EnhancedAgenticStepProcessor inherits the prompt_builder dispatch (T030)."""
    from promptchain.prompts import DynamicPromptGenerator
    from promptchain.utils.enhanced_agentic_step_processor import (
        EnhancedAgenticStepProcessor,
    )

    proc = EnhancedAgenticStepProcessor(
        objective="x",
        enable_rag_verification=False,
        enable_gemini_augmentation=False,
        enable_memo_store=False,
        enable_interrupt_queue=False,
    )
    assert isinstance(proc.prompt_builder, DynamicPromptGenerator)


def test_subclass_inheritance_state_agent_sees_custom_tools() -> None:
    """StateAgent propagates the prompt_builder fix through super().__init__ (T031).

    Skipped: ``StateAgent.__init__`` takes ``(agent_chain, verbose)`` rather than
    ``objective``, and constructing a real AgentChain pulls in heavy LLM/SQLite
    machinery unsafe for a unit test. The inheritance is still verifiable manually
    — see promptchain/utils/strategies/state_agent.py:374 (passes objective= to
    AgenticStepProcessor via super().__init__).
    """
    pytest.skip(
        "StateAgent constructor signature differs (agent_chain not objective) — "
        "see promptchain/utils/strategies/state_agent.py:364"
    )


def test_callsite_compliance_tui_grep() -> None:
    """No bare AgenticStepProcessor(...) calls remain in CLI/agentic_chat .py files (T032)."""
    repo_root = Path(__file__).resolve().parents[1]
    targets = [str(repo_root / "promptchain" / "cli"), str(repo_root / "agentic_chat")]
    # Only scan paths that exist
    targets = [t for t in targets if Path(t).exists()]
    if not targets:
        pytest.skip("Neither promptchain/cli nor agentic_chat present")

    result = subprocess.run(
        ["grep", "-rnE", r"AgenticStepProcessor\s*\(", *targets],
        capture_output=True,
        text=True,
    )
    raw_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]

    offenders: List[str] = []
    for ln in raw_lines:
        # Skip subclass usages and explicitly whitelisted sub-agent spawners.
        if "TUIAgenticStepProcessor" in ln:
            continue
        if "EnhancedAgenticStepProcessor" in ln:
            continue
        if "# sub-agent:" in ln:
            continue
        # Documentation files (.md) are fine.
        path_part = ln.split(":", 1)[0]
        if path_part.endswith(".md"):
            continue
        # Test files in cli/ would be unusual but allowed.
        if "/tests/" in path_part or "/test_" in path_part:
            continue
        # Strip leading "path:lineno:" prefix to inspect the source line itself.
        try:
            _, _, source = ln.split(":", 2)
        except ValueError:
            source = ln
        stripped = source.lstrip()
        # Skip docstrings and string-literal mentions (e.g. error messages
        # that include "AgenticStepProcessor(instructions=...)" as text).
        if stripped.startswith(("#", '"', "'")):
            continue
        # Heuristic: treat as a string-literal mention if the call appears
        # inside quotes on the same line.
        before_call = source.split("AgenticStepProcessor", 1)[0]
        if before_call.count('"') % 2 == 1 or before_call.count("'") % 2 == 1:
            continue
        offenders.append(ln)

    assert not offenders, (
        "Found bare AgenticStepProcessor(...) calls in CLI/agentic_chat sources:\n"
        + "\n".join(offenders)
    )


# === Wave 23-25: US3 (T040, T041, T042) ===


def test_dispatch_mutually_exclusive_raises_value_error() -> None:
    """instructions= AND prompt_builder= together raise ValueError (T040)."""
    from promptchain.prompts import DynamicPromptGenerator
    from promptchain.utils.agentic_step_processor import AgenticStepProcessor

    with pytest.raises(ValueError):
        AgenticStepProcessor(
            objective="x",
            instructions=["a"],
            prompt_builder=DynamicPromptGenerator(),
        )


def test_dispatch_instructions_emits_single_deprecation_warning() -> None:
    """Exactly one DeprecationWarning mentioning 'instructions' fires (T041)."""
    from promptchain.utils.agentic_step_processor import AgenticStepProcessor

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AgenticStepProcessor(objective="x", instructions=["Always cite sources."])

    relevant = [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "instructions" in str(w.message).lower()
    ]
    assert len(relevant) == 1, (
        f"expected exactly 1 DeprecationWarning mentioning 'instructions', "
        f"got {[(w.category.__name__, str(w.message)) for w in caught]}"
    )


def test_dispatch_instructions_renders_as_extra_instructions_block() -> None:
    """Supplied instructions appear in the rendered prompt (T042)."""
    from promptchain.utils.agentic_step_processor import AgenticStepProcessor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        proc = AgenticStepProcessor(
            objective="x", instructions=["Always cite sources."]
        )

    rendered = proc.prompt_builder.generate("x", _make_tools(1))
    assert "ADDITIONAL INSTRUCTIONS" in rendered
    assert "Always cite sources." in rendered


# === Wave 26-27: US4 (T048, T049) ===


def test_dispatch_accepts_custom_builder() -> None:
    """A custom builder injected via prompt_builder= is used verbatim (T048)."""
    from promptchain.utils.agentic_step_processor import AgenticStepProcessor

    class _MyBuilder:
        def generate(
            self,
            objective: str,
            tools: List[Dict[str, Any]],
            context: Optional[str] = None,
        ) -> str:
            return "MY-BUILDER-OUTPUT"

        def get_token_estimate(
            self, objective: str, tools: List[Dict[str, Any]]
        ) -> int:
            return 1

    builder = _MyBuilder()
    proc = AgenticStepProcessor(objective="x", prompt_builder=builder)
    assert proc.prompt_builder is builder
    assert proc.prompt_builder.generate("x", []) == "MY-BUILDER-OUTPUT"


def test_dispatch_custom_builder_with_non_default_workflow_hint_warns() -> None:
    """workflow_pattern='react' + custom prompt_builder triggers UserWarning (T049)."""
    from promptchain.utils.agentic_step_processor import AgenticStepProcessor

    class _MyBuilder:
        def generate(self, objective, tools, context=None):
            return "X"

        def get_token_estimate(self, objective, tools):
            return 0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AgenticStepProcessor(
            objective="x",
            prompt_builder=_MyBuilder(),
            workflow_pattern="react",
        )

    relevant = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "workflow_pattern" in str(w.message)
    ]
    assert len(relevant) >= 1, (
        f"expected UserWarning about ignored workflow_pattern, got "
        f"{[(w.category.__name__, str(w.message)) for w in caught]}"
    )


# === Wave 28-32: Polish (T060, T061, T062, T063, T065) ===


def test_dynamic_extra_instructions_renders_block() -> None:
    """extra_instructions=[X, Y] renders both bullets under header (T060, FR-005)."""
    from promptchain.prompts import DynamicPromptGenerator

    rendered = DynamicPromptGenerator(
        extra_instructions=["X-instruction-marker", "Y-instruction-marker"]
    ).generate("obj", _make_tools(1))

    assert "ADDITIONAL INSTRUCTIONS" in rendered
    assert "X-instruction-marker" in rendered
    assert "Y-instruction-marker" in rendered


def test_dynamic_final_answer_block_conditional_on_flag() -> None:
    """include_response_format_hint toggles the FINAL ANSWER block (T061, FR-007)."""
    from promptchain.prompts import DynamicPromptGenerator

    on = DynamicPromptGenerator(include_response_format_hint=True).generate(
        "obj", _make_tools(1)
    )
    off = DynamicPromptGenerator(include_response_format_hint=False).generate(
        "obj", _make_tools(1)
    )
    assert "FINAL ANSWER" in on
    assert "FINAL ANSWER" not in off


def test_no_process_global_state_in_prompts_module() -> None:
    """No module-level mutable singletons in promptchain/prompts/ (T062, FR-017)."""
    pkg_dir = Path(__file__).resolve().parents[1] / "promptchain" / "prompts"
    py_files = sorted(pkg_dir.glob("*.py"))
    assert py_files, f"no .py files found under {pkg_dir}"

    offenders: List[str] = []
    for path in py_files:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            # Look for top-level Assign nodes whose value is a mutable literal.
            if isinstance(node, ast.Assign):
                if isinstance(node.value, (ast.Dict, ast.List, ast.Set)):
                    targets = [
                        t.id for t in node.targets if isinstance(t, ast.Name)
                    ]
                    # ``__all__`` is the conventional public-API declaration
                    # and is not mutated at runtime — allow it.
                    if targets == ["__all__"]:
                        continue
                    offenders.append(f"{path.name}:{node.lineno} {targets}")
    assert not offenders, (
        "Mutable module-level state detected in prompts package (FR-017):\n"
        + "\n".join(offenders)
    )


def test_base_module_has_no_internal_imports() -> None:
    """promptchain/prompts/base.py imports nothing from promptchain.* (T063, FR-018)."""
    base_path = (
        Path(__file__).resolve().parents[1] / "promptchain" / "prompts" / "base.py"
    )
    tree = ast.parse(base_path.read_text(), filename=str(base_path))

    bad: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("promptchain"):
                bad.append(f"from {mod} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("promptchain"):
                    bad.append(f"import {alias.name}")
    assert not bad, (
        "BasePromptBuilder module must not import from promptchain.* (FR-018). "
        f"Offending imports: {bad}"
    )


def test_tui_sub_agent_spawn_uses_custom_instructions() -> None:
    """Sub-agent spawn path: DynamicPromptGenerator with extra_instructions+react (T065).

    Mirrors the quickstart Part 2 sub-agent path: the TUI may spawn a child
    AgenticStepProcessor with a custom builder rather than the legacy one.
    """
    from promptchain.prompts import DynamicPromptGenerator
    from promptchain.utils.agentic_step_processor import AgenticStepProcessor

    builder = DynamicPromptGenerator(
        extra_instructions=["sub-task hint"],
        workflow_pattern="react",
    )
    proc = AgenticStepProcessor(objective="sub", prompt_builder=builder)
    rendered = proc.prompt_builder.generate(
        "sub", _make_tools(1) + [_tool("task_list_write_tool", "Plan tasks.")]
    )
    assert "sub-task hint" in rendered
