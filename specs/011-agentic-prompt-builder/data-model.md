# Data Model — Agentic Prompt Builder Decoupling

**Feature**: 011-agentic-prompt-builder
**Scope**: In-process strategy objects and their interaction with `AgenticStepProcessor`. No persistence.

This document enumerates the entities, their fields, their relationships, and the relevant state transitions on `AgenticStepProcessor.__init__` and on each call of `AgenticStepProcessor.run()`.

---

## Entity: `BasePromptBuilder` (Protocol)

**Module**: `promptchain/prompts/base.py`
**Kind**: `typing.Protocol` — structural type. No inheritance required of implementors.

### Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `generate` | `(self, objective: str, tools: List[Dict[str, Any]], context: Optional[str] = None) -> str` | Produce the final system-prompt string that the processor will send to the LLM as the `system` role message. |
| `get_token_estimate` | `(self, objective: str, tools: List[Dict[str, Any]]) -> int` | Return an integer estimated token count for the prompt the generator would produce. Used by callers for fail-fast guards before LLM invocation. |

### Invariants

1. `generate` is deterministic for a given `(objective, tools, context)` triple.
2. `generate` is pure — it must not mutate `tools`, must not read global state that could change between calls, and must not hold process-global side-effect registrations.
3. `get_token_estimate` is non-negative and O(n_tools) in time.
4. Any class implementing the Protocol must provide both methods. Python's Protocol machinery enforces structural conformance at type-check time; at runtime, missing methods surface as `AttributeError` at first call.

### Relationships

- `AgenticStepProcessor` holds exactly one `BasePromptBuilder` reference after construction. The reference is immutable for the processor's lifetime (unless the caller explicitly reassigns `processor.prompt_builder`, which is supported but not required).
- `BasePromptBuilder` has no knowledge of the processor, of the LLM, or of the reasoning loop. It only consumes objective + tool schemas + optional context and returns a string.

---

## Entity: `DynamicPromptGenerator`

**Module**: `promptchain/prompts/dynamic.py`
**Conforms to**: `BasePromptBuilder` Protocol.
**Role**: New default prompt builder for the library. Renders the actually-registered tool inventory.

### Constructor parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `extra_instructions` | `Optional[List[str]]` | `None` | Free-form lines appended as an additional-instructions block in the rendered prompt. Used by the `instructions=` deprecation shim. |
| `workflow_pattern` | `Literal["standard", "react"]` | `"standard"` | Controls whether the rendered prompt includes a ReAct-style think/plan/act/observe scaffold. |
| `include_response_format_hint` | `bool` | `True` | If true, appends the final-answer-requirements block requiring the model to include full tool-result content. |

### Output structure (rendered in this order)

```
1. Objective line:                 "Your goal is to achieve the following objective: {objective}"
2. AVAILABLE TOOLS block:          one line per tool: "- {name}: {description}"
                                   If no tools: "AVAILABLE TOOLS: (none registered — the agent should proceed without tool use)"
3. Workflow block (conditional):
   - if workflow_pattern == "react" AND a task-list writer tool is registered:
        full ReAct scaffold (think / plan / act / observe)
   - if workflow_pattern == "react" AND no task-list writer tool:
        emit warning via logger; render a minimal thought/action scaffold instead
   - if workflow_pattern == "standard":
        omit the block entirely
4. Extra-instructions block (conditional):
   - if extra_instructions provided:
        header "ADDITIONAL INSTRUCTIONS:" followed by the lines
5. Prior-context block (conditional):
   - if context argument provided to generate(...):
        header "PRIOR CONTEXT:" followed by the context string
6. Final-answer block (conditional):
   - if include_response_format_hint:
        canonical block instructing model to return full tool-result content verbatim
```

### State transitions

- Constructor fails fast with `ValueError` if `workflow_pattern` is not in `{"standard", "react"}`.
- Constructor does not warn; warnings about ReAct-without-task-list fire at `generate()` time (when the tool schema is known) rather than at construction (when it is not).

### Tool-detection heuristic

For the ReAct-mode conditional, "a task-list writer tool is registered" means a tool in the schema whose function name contains the substring `task_list` OR whose name exactly matches one of a small allowlist (documented in the dynamic.py docstring, initially `{"task_list_write_tool", "task_list_update_tool"}`). This heuristic is conservative and emits a runtime warning whenever it fires, so surprising matches are visible to the developer.

---

## Entity: `LegacyTUIPromptGenerator`

**Module**: `promptchain/prompts/legacy_tui.py`
**Conforms to**: `BasePromptBuilder` Protocol.
**Role**: Preserves v0.5.0 shipped-TUI prompt byte-for-byte. Only the TUI is expected to use it.

### Constructor parameters

None. The legacy prompt is frozen at a literal string pulled from the pre-change `agentic_step_processor.py:909-1013`.

### Output structure

Returns the exact pre-release hardcoded prompt with `{self.objective}` substituted to the supplied `objective` argument. The `tools` argument is **deliberately ignored** for the purpose of prompt content — the frozen inventory lists only the tools the TUI shipped with.

### Runtime warning behavior

On each call to `generate(objective, tools, context)`:
1. Compute the current tool-name set from `tools`.
2. If the set is non-empty and differs from the known-frozen-default set (hardcoded in the module — documented as `_EXPECTED_TUI_DEFAULT_TOOL_NAMES`), log a single runtime warning naming the unexpected tools.
3. Always return the literal frozen prompt regardless of the tool schema.

The warning helps TUI plugin authors understand that their extra tools won't appear in the prompt — they should switch to `DynamicPromptGenerator` if they need those tools surfaced.

### Snapshot testing

The literal prompt is stored redundantly in two places:
- Inside `legacy_tui.py` as the return value of `generate()`.
- In `tests/fixtures/legacy_tui_prompt.snapshot.txt` as the fixture.

A snapshot test (`tests/test_prompt_builders.py::test_legacy_snapshot_byte_identical`) asserts the two match. Incidental edits to either fail the test immediately.

---

## Entity: `Tool Schema`

**Shape**: `Dict[str, Any]` — OpenAI-format function-calling tool schema as already used throughout PromptChain.

**Fields consumed by builders**:
- `type` (str) — expected to be `"function"`.
- `function.name` (str) — tool name. Required.
- `function.description` (str) — human-readable description. Used by `DynamicPromptGenerator`. If missing or empty, the builder substitutes `"(no description provided)"` to avoid rendering a blank line.
- `function.parameters` (dict) — parameter schema. **Not consumed by prompt builders** in this release; builders summarize tools by name+description only. (Parameter schemas still reach the LLM via the `tools=` API parameter, so they remain visible to the model; the builders just don't duplicate them in the system prompt text.)

**Source**: `available_tools` list passed into `AgenticStepProcessor.run(...)`. The list is produced by the chain and reflects the tools registered at call time.

---

## Entity: `AgenticStepProcessor` (modified)

**Module**: `promptchain/utils/agentic_step_processor.py`
**Change scope**: `__init__` gains two new optional kwargs (`prompt_builder`, `instructions`) and one controlling hint (`workflow_pattern`). `run()` delegates system-prompt construction to `self.prompt_builder.generate(...)`.

### New / modified fields

| Field | Type | Initialized from | Purpose |
|-------|------|------------------|---------|
| `prompt_builder` | `BasePromptBuilder` | `__init__` dispatch (see below) | Injected strategy used to produce each system prompt. |

### Constructor dispatch state machine

On each `__init__` call:

```
inputs: (instructions, prompt_builder, workflow_pattern)

if instructions is not None and prompt_builder is not None:
    raise ValueError("`instructions=` and `prompt_builder=` are mutually exclusive. "
                     "Use `prompt_builder=DynamicPromptGenerator(extra_instructions=...)` "
                     "instead of `instructions=`.")

elif instructions is not None and prompt_builder is None:
    warnings.warn(
        "The `instructions=` parameter is deprecated. "
        "Migrate to `prompt_builder=DynamicPromptGenerator(extra_instructions=...)`.",
        DeprecationWarning,
        stacklevel=2,
    )
    self.prompt_builder = DynamicPromptGenerator(
        extra_instructions=list(instructions),
        workflow_pattern=workflow_pattern,
    )

elif instructions is None and prompt_builder is not None:
    if workflow_pattern != "standard":
        # User passed both a custom builder AND a workflow_pattern hint.
        # The hint only affects the shipped DynamicPromptGenerator, so warn.
        logger.warning(
            "workflow_pattern=%r is ignored because a custom prompt_builder was supplied; "
            "the hint only applies to the shipped DynamicPromptGenerator.",
            workflow_pattern,
        )
    self.prompt_builder = prompt_builder

else:  # both None
    self.prompt_builder = DynamicPromptGenerator(workflow_pattern=workflow_pattern)
```

### `run()` integration point

The hardcoded f-string at the old `agentic_step_processor.py:909-1013` is replaced with:

```python
system_prompt = self.prompt_builder.generate(
    objective=self.objective,
    tools=available_tools,
    context=(scratchpad_text if scratchpad_text else None),
)
system_message = {"role": "system", "content": system_prompt}
```

The rest of `run()` is unchanged — tool dispatch, step counting, success reporting, async behavior all remain identical.

### Behavior for subclasses

`EnhancedAgenticStepProcessor` and `StateAgent` both call `super().__init__(...)` and do not override `run()`'s system-prompt construction. They pick up the fix automatically.

---

## Entity: `TUIAgenticStepProcessor` (new subclass — Option A default path)

**Module**: `promptchain/cli/tui_processor.py`
**Inherits from**: `AgenticStepProcessor` (base class).
**Role**: Convenience subclass for the TUI's default agent path. Bakes in `prompt_builder=LegacyTUIPromptGenerator()` so TUI code reads naturally as `TUIAgenticStepProcessor(objective=..., ...)` without having to remember the opt-in.

### Constructor

```python
class TUIAgenticStepProcessor(AgenticStepProcessor):
    def __init__(self, **kwargs: Any) -> None:
        if "prompt_builder" in kwargs:
            raise TypeError(
                "TUIAgenticStepProcessor bakes in LegacyTUIPromptGenerator. "
                "If you need a different prompt builder, use the base "
                "AgenticStepProcessor directly and pass prompt_builder=... there."
            )
        if "instructions" in kwargs:
            raise TypeError(
                "TUIAgenticStepProcessor bakes in the legacy TUI prompt and does not "
                "accept the deprecated `instructions=` shim. Use the base "
                "AgenticStepProcessor directly if you need per-call instructions."
            )
        super().__init__(prompt_builder=LegacyTUIPromptGenerator(), **kwargs)
```

### Behavior

- Accepts every argument the base class accepts **except** `prompt_builder=` and `instructions=`, both of which are rejected with a clear `TypeError`. The rejection is deliberate: if a caller wanted a different prompt, they should be reaching for the base class, not this subclass.
- Inherits all runtime behavior (including the reasoning loop) from `AgenticStepProcessor` unchanged.

### Call-site adoption (Option A — default path)

The 10 known TUI-side default-path construction sites (listed in `research.md` §"Decision 7 — Call-site census") switch to calling `TUIAgenticStepProcessor(...)` instead of `AgenticStepProcessor(...)`.

### Option B — swappable processor class

The TUI is **not** locked into `TUIAgenticStepProcessor`. TUI code may freely construct, in any context and at any call site:

- `AgenticStepProcessor(objective=..., instructions=[...])` — vanilla processor with per-call instructions. Ideal for sub-agent spawning where the sub-agent's system prompt must reflect its own instructions rather than the legacy TUI default.
- `AgenticStepProcessor(objective=..., prompt_builder=CustomBuilder())` — vanilla processor with an explicit custom builder.
- `EnhancedAgenticStepProcessor(objective=..., ...)` — enhanced multi-hop variant. Inherits the dynamic default via `super().__init__`.
- `StateAgent(objective=..., ...)` — state-driven variant. Inherits the dynamic default via `super().__init__`.
- Any third-party subclass of `AgenticStepProcessor` — also inherits the dynamic default via `super().__init__`.

### Library-path sites

The two library-path sites (`orchestrator_supervisor.py:80`, `chain_executor.py:403`) continue to call `AgenticStepProcessor(...)` and correctly land on the new dynamic default. They do NOT adopt `TUIAgenticStepProcessor`.

---

## Relationship diagram

```
+--------------------------+               +----------------------------+
|                          |  constructs   |                            |
|   Library consumer /     +-------------->|   AgenticStepProcessor     |
|   TUI call site          |               |   (holds prompt_builder)   |
|                          |               +-------------+--------------+
+--------------------------+                             |
                                                          | .prompt_builder
                                                          v
                                           +----------------------------+
                                           |   BasePromptBuilder         |
                                           |   (Protocol)                |
                                           +-------------+--------------+
                                                          ^
                             implements                   | implements
                      +--------+-------------+            |     +-----------------------+
                      |                      |            |     |                       |
        +-------------+------------+   +-----+------------+---+ |
        | DynamicPromptGenerator   |   | LegacyTUIPromptGenerator|
        | (new default)            |   | (frozen v0.5.0 prompt) |
        +--------------------------+   +------------------------+
                      ^
                      | also constructed by
                      |
        +-------------+------------+
        | instructions= shim path  |
        +--------------------------+
```

---

## Test surface summary (informs Phase 2 task generation)

The data model above produces the following minimum test surface:

1. **Protocol conformance**: both `DynamicPromptGenerator` and `LegacyTUIPromptGenerator` satisfy `BasePromptBuilder`. Checked at type-check time via mypy and at runtime via `isinstance`/`callable` probes in `tests/test_prompt_builders.py`.
2. **Dynamic render contents**: `DynamicPromptGenerator().generate(...)` output contains every registered tool name, contains no hardcoded TUI-only tool name, and ends with the final-answer block.
3. **Dynamic workflow modes**: `standard` omits the workflow block; `react` includes it when a task-list tool is registered; `react` without task-list emits a warning and falls back.
4. **Dynamic extra-instructions**: output contains the supplied instructions inside the additional-instructions block.
5. **Legacy snapshot**: `LegacyTUIPromptGenerator().generate("x", [...])` output byte-equals `tests/fixtures/legacy_tui_prompt.snapshot.txt` after substituting `"x"` for `{objective}`.
6. **Legacy tool-drift warning**: passing an unexpected tools list logs a warning but returns the frozen output.
7. **Constructor dispatch**:
   - both `instructions=` and `prompt_builder=` → `ValueError`.
   - only `instructions=` → `DeprecationWarning` emitted, dynamic builder with `extra_instructions` installed.
   - only `prompt_builder=` + non-default `workflow_pattern` → `logger.warning` fires.
   - neither → default dynamic builder with `standard` mode.
8. **Subclass inheritance**: instantiating `EnhancedAgenticStepProcessor` and `StateAgent` with a chain of 4 custom tools produces a system prompt that names all 4 tools and none of the TUI-only names.
9. **Call-site compliance**: `grep -rE "AgenticStepProcessor\s*\(" promptchain/cli/ agentic_chat/` returns zero lines that are not inside `_build_tui_agent`.
10. **Library-consumer e2e**: the PRD reproduction with 4 registered tools produces a response with `accumulated_contexts_count > 0` and relevant retrieval calls.
