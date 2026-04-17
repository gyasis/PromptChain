# Contract — `BasePromptBuilder` Protocol and Dispatch Semantics

**Feature**: 011-agentic-prompt-builder
**Kind**: Public API contract. Third parties may ship classes conforming to this Protocol and pass them to `AgenticStepProcessor(prompt_builder=...)`.

This document is the authoritative reference for the shape and semantics of the new prompt-builder surface. It covers (1) the Protocol itself, (2) the two shipped concrete builders, (3) the `AgenticStepProcessor.__init__` dispatch rules, and (4) the TUI subclass's constructor constraints.

---

## 1. `BasePromptBuilder` Protocol

```python
# promptchain/prompts/base.py
from typing import Any, Dict, List, Optional, Protocol

class BasePromptBuilder(Protocol):
    def generate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> str: ...

    def get_token_estimate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
    ) -> int: ...
```

### Preconditions

- `objective` is a non-empty string. Implementations MAY raise `ValueError` if an empty objective would produce a meaningless prompt, but MUST document that choice.
- `tools` is a list of OpenAI-format function-tool schemas (each a dict with keys `type`, `function` where `function` carries `name`, `description`, `parameters`). Empty list is valid and MUST be accepted.
- `context` is either `None` or a string. If a string, it is treated as prior scratchpad / step output and rendered verbatim as the PRIOR CONTEXT block.

### Postconditions

- `generate` returns a non-empty string suitable for direct use as the `content` of a `{"role": "system"}` message.
- `generate` does not mutate `tools`. Implementations that need a sorted copy MUST make their own.
- `generate` is deterministic: identical `(objective, tools, context)` inputs produce identical output.
- `get_token_estimate` returns a non-negative integer.

### Invariants

- No implementation may hold process-global mutable state that affects output between calls. Instance-local state is fine (for example a builder that caches a compiled template).
- `base.py` imports only from `typing` — zero internal PromptChain imports. This is enforced by lint check.

### Error behavior

- If an implementation cannot produce a prompt (for example a required tool shape is malformed), it MUST raise an exception derived from `ValueError` with a clear diagnostic. It MUST NOT silently return an empty string or a fallback that mismatches the caller's expectations.

---

## 2. `DynamicPromptGenerator` (shipped)

```python
# promptchain/prompts/dynamic.py
from typing import List, Literal, Optional

class DynamicPromptGenerator:
    def __init__(
        self,
        *,
        extra_instructions: Optional[List[str]] = None,
        workflow_pattern: Literal["standard", "react"] = "standard",
        include_response_format_hint: bool = True,
    ) -> None: ...

    def generate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> str: ...

    def get_token_estimate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
    ) -> int: ...
```

### Constructor contract

- `extra_instructions` — if provided and non-empty, each string is rendered as a bullet under an "ADDITIONAL INSTRUCTIONS:" header.
- `workflow_pattern` — must be exactly `"standard"` or `"react"`. Any other value raises `ValueError` at construction time.
- `include_response_format_hint` — when true, the generated prompt ends with the canonical final-answer block (require full tool-result content, no summarization).

### `generate` output contract

For `workflow_pattern="standard"`, the prompt consists of:

1. Objective line.
2. AVAILABLE TOOLS block (one line per tool, or an empty-inventory sentinel line if `tools` is empty).
3. ADDITIONAL INSTRUCTIONS block — only if `extra_instructions` was non-empty.
4. PRIOR CONTEXT block — only if `context is not None`.
5. FINAL ANSWER block — only if `include_response_format_hint=True` at construction.

For `workflow_pattern="react"`, a ReAct think/plan/act/observe scaffold is inserted between blocks 2 and 3. If `tools` does not contain any task-list-writer tool (heuristic: function name contains `task_list` OR matches the allowlist in the builder's module docstring), the builder emits a `logger.warning(...)` and falls back to a minimal thought/action scaffold instead.

### Negative assertions (what the output MUST NOT contain)

- No hardcoded TUI-only tool name unless that name is actually present in the supplied `tools` list.
- No reference to the legacy `{objective}`-only substitution pattern — the dynamic builder is not a port of the legacy prompt with variable substitution.

---

## 3. `LegacyTUIPromptGenerator` (shipped)

```python
# promptchain/prompts/legacy_tui.py
from typing import Any, Dict, List, Optional

class LegacyTUIPromptGenerator:
    def __init__(self) -> None: ...

    def generate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],  # deliberately ignored for content; inspected for drift-warning
        context: Optional[str] = None,  # appended as-is if provided
    ) -> str: ...

    def get_token_estimate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
    ) -> int: ...
```

### Constructor contract

Accepts no arguments. The legacy prompt is frozen.

### `generate` output contract

Returns the literal v0.5.0 hardcoded terminal-UI prompt with `{objective}` substituted to the argument. The `tools` argument does not influence the prompt content. If `context` is provided, it is appended as a PRIOR CONTEXT block after the frozen body (this is additive — the frozen body is unchanged).

### Drift-warning contract

On every `generate(...)` call:
- If `tools` is non-empty AND its tool-name set differs from `_EXPECTED_TUI_DEFAULT_TOOL_NAMES` (a module-level frozenset baseline), a single runtime warning is logged via `logger.warning(...)`. The warning names the unexpected tools so the caller can decide whether to switch to `DynamicPromptGenerator`.

### Snapshot contract

`LegacyTUIPromptGenerator().generate("test-objective", [])` must byte-equal `tests/fixtures/legacy_tui_prompt.snapshot.txt` after substituting `"test-objective"` for `{objective}`. Any unintentional diff is caught by `tests/test_prompt_builders.py::test_legacy_snapshot_byte_identical`.

---

## 4. `AgenticStepProcessor.__init__` dispatch contract

The modified constructor accepts, alongside all pre-existing arguments, these new optional kwargs:

```python
def __init__(
    self,
    objective: str,
    max_internal_steps: int = 5,
    # ... all existing kwargs preserved ...
    instructions: Optional[List[str]] = None,
    prompt_builder: Optional[BasePromptBuilder] = None,
    workflow_pattern: Literal["standard", "react"] = "standard",
) -> None: ...
```

### Dispatch table

| `instructions` | `prompt_builder` | `workflow_pattern` | Effect |
|----------------|------------------|--------------------|--------|
| non-None | non-None | any | `ValueError("...mutually exclusive...")` |
| non-None | None | any | `DeprecationWarning` emitted once; `self.prompt_builder = DynamicPromptGenerator(extra_instructions=instructions, workflow_pattern=workflow_pattern)` |
| None | non-None | default (`"standard"`) | `self.prompt_builder = prompt_builder` (no warning) |
| None | non-None | non-default (`"react"`) | `logger.warning(...)` that the hint is ignored because a custom builder was supplied; `self.prompt_builder = prompt_builder` |
| None | None | any | `self.prompt_builder = DynamicPromptGenerator(workflow_pattern=workflow_pattern)` |

### `run()` integration

Inside `run(...)`, wherever the old hardcoded f-string used to construct `system_prompt`, the new code is:

```python
system_prompt = self.prompt_builder.generate(
    objective=self.objective,
    tools=available_tools,
    context=(scratchpad_text if scratchpad_text else None),
)
```

All other `run()` logic is unchanged.

### Subclass behavior

`EnhancedAgenticStepProcessor` and `StateAgent` call `super().__init__(...)` with all relevant kwargs forwarded. They pick up the fix automatically. Any future subclass that overrides `run()` with its own inlined prompt construction MUST also call `self.prompt_builder.generate(...)` — this is documented in `quickstart.md` and in the processor's module docstring.

---

## 5. `TUIAgenticStepProcessor` constructor contract

```python
# promptchain/cli/tui_processor.py
from typing import Any
from promptchain.prompts import LegacyTUIPromptGenerator
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

class TUIAgenticStepProcessor(AgenticStepProcessor):
    def __init__(self, **kwargs: Any) -> None:
        if "prompt_builder" in kwargs:
            raise TypeError(...)
        if "instructions" in kwargs:
            raise TypeError(...)
        super().__init__(prompt_builder=LegacyTUIPromptGenerator(), **kwargs)
```

### Constructor contract

- Accepts every argument `AgenticStepProcessor.__init__` accepts, **except** `prompt_builder` and `instructions`.
- Supplying either rejected kwarg raises `TypeError` with a message pointing the caller at the base class if they need that flexibility.
- Always baked-in: `prompt_builder=LegacyTUIPromptGenerator()`.

### Intended use

TUI code paths that previously constructed `AgenticStepProcessor(...)` for the default agent now construct `TUIAgenticStepProcessor(...)` instead. For sub-agent spawning or specialized tasks, TUI code freely constructs the base `AgenticStepProcessor(...)` (with `instructions=` or `prompt_builder=` as appropriate), `EnhancedAgenticStepProcessor(...)`, `StateAgent(...)`, or any third-party subclass — all of which correctly receive the new dynamic default unless they explicitly opt into something else.

---

## 6. Backwards-compatibility surface

- The Protocol is new — no existing code imports it, so no migration concern.
- `DynamicPromptGenerator` and `LegacyTUIPromptGenerator` are new public classes in a new public module. Additive.
- `TUIAgenticStepProcessor` is a new subclass. Additive.
- `AgenticStepProcessor.__init__` gains three new optional kwargs (`instructions`, `prompt_builder`, `workflow_pattern`), all defaulting to None / `"standard"`. Pre-existing call sites that did not pass those kwargs see a behavior change only in the system-prompt content (dynamic instead of legacy hardcoded) — this is the intended fix and is documented as a breaking change in CHANGELOG.md.
- The `instructions=` kwarg is the only item on the restored-with-deprecation list. `DeprecationWarning` is emitted once per construction; the behavior produces a fully functional agent.

---

## 7. Contract tests (enumerated for tasks phase)

The following contract tests MUST exist and be green before release:

1. `test_protocol_conformance_dynamic` — `isinstance` check and signature check against `DynamicPromptGenerator`.
2. `test_protocol_conformance_legacy` — same against `LegacyTUIPromptGenerator`.
3. `test_protocol_conformance_custom` — a trivial in-test custom class must be accepted by `AgenticStepProcessor(prompt_builder=custom)` without error.
4. `test_legacy_snapshot_byte_identical` — legacy output matches `tests/fixtures/legacy_tui_prompt.snapshot.txt`.
5. `test_dynamic_renders_tools` — supplied tool names appear in the output; no TUI-only names appear.
6. `test_dynamic_empty_tools` — empty inventory produces a clean sentinel, not a blank block.
7. `test_dynamic_react_without_tasklist_warns` — `workflow_pattern="react"` + no task-list tool triggers `logger.warning` and falls back to minimal scaffold.
8. `test_dispatch_mutually_exclusive` — both `instructions=` and `prompt_builder=` raises `ValueError`.
9. `test_dispatch_instructions_deprecation` — `instructions=` alone emits `DeprecationWarning` (captured via `pytest.warns`).
10. `test_dispatch_default` — no args produces a dynamic builder in `"standard"` mode.
11. `test_tui_subclass_rejects_prompt_builder` — `TUIAgenticStepProcessor(prompt_builder=...)` raises `TypeError`.
12. `test_tui_subclass_rejects_instructions` — `TUIAgenticStepProcessor(instructions=...)` raises `TypeError`.
13. `test_subclass_inheritance_enhanced` — `EnhancedAgenticStepProcessor` with 4 custom tools produces dynamic output containing those tool names.
14. `test_subclass_inheritance_state_agent` — same for `StateAgent`.
15. `test_callsite_compliance_tui` — grep-based: no default-path TUI call site constructs bare `AgenticStepProcessor(...)`. (Sub-agent-spawning call sites are explicitly whitelisted in the test.)
16. `test_library_consumer_e2e` — PRD reproduction: 4 registered tools + multi-hop query → `accumulated_contexts_count > 0`.
