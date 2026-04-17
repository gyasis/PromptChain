# Product Requirements Document: AgenticStepProcessor Prompt & Tool Decoupling

**Version:** 1.0
**Date:** April 17, 2026
**Status:** Draft — pending speckit intake
**Owner:** PromptChain Core
**Tracking Issue:** https://github.com/gyasis/PromptChain/issues/2
**Target Release:** v0.6.0

---

## Executive Summary

`AgenticStepProcessor` embeds a ~100-line system prompt directly in its source (`agentic_step_processor.py:909-1013`) that names a fixed inventory of tools only available in the PromptChain TUI. Any library consumer — Athena-LightRAG, CDIA, custom RAG agents — gets an agent whose system prompt lies to the LLM about what tools are registered. Tools registered via `chain.add_tools()` are never mentioned; the prompt advertises tools the consumer never registered.

This PRD decouples agent *logic* (the reasoning loop) from agent *personality* (the system prompt) using the Strategy pattern. Library consumers get a library-correct default; the TUI preserves its current behavior by explicitly opting into the legacy prompt.

**Target Impact:**
- Library consumers' agents correctly reference the tools they registered — no hallucinated / ungrounded responses produced by prompt/tool mismatch
- Pre-v0.4.2 users whose `instructions=` code broke silently in v0.4.2 are restored with a deprecation-bridge path
- TUI (`promptchain` CLI) behavior unchanged
- Foundation for follow-up work: groundedness checks, TUI prompt modernization

---

## Problem Statement

### 1. The system prompt is a structural lie

`AgenticStepProcessor.run()` constructs the LLM system message from a hardcoded f-string with exactly one dynamic field: `{self.objective}`. The rest (~100 lines) is static text listing tools like `task_list_write_tool`, `ripgrep_search`, `file_read/write/edit`, `terminal_execute`, `list_directory`, `sandbox_*`, `mcp_gemini_*`. These tools only exist inside the TUI's default toolset. Library consumers who register their own tools via `chain.add_tools()` find those tools *never appear* in the prompt, while the prompt *still* advertises the TUI-only tools the consumer never registered.

### 2. The `instructions=` escape hatch was removed without replacement

In v0.4.2 the `instructions=` parameter on `AgenticStepProcessor.__init__` was removed. Pre-v0.4.2 code that passed a custom instruction list to steer the agent silently lost that guidance. No replacement API was added. A confirming comment exists in a downstream consumer (`athena-lightrag/agentic_lightrag.py:393`): `"NOTE: PromptChain v0.4.2 removed instructions parameter from AgenticStepProcessor"`.

### 3. Subclasses inherit the defect

Two existing subclasses — `EnhancedAgenticStepProcessor(AgenticStepProcessor)` and `StateAgent(AgenticStepProcessor)` — both call `super().__init__(...)` and inherit the hardcoded system prompt. Neither overrides it. The "enhanced" processor adds RAG verification, Gemini augmentation, memo stores, and interrupt queues; none of those features address the prompt/tool mismatch. `StateAgent` has no prompt logic at all.

### 4. Success reporting masks the failure mode

`execute_multi_hop_reasoning` returns `success: True` based solely on "did every reasoning step return a string?" — no groundedness or tool-invocation check. When the agent skips retrieval entirely and free-associates (which is what the mismatched prompt encourages), the caller receives `success: true` alongside a completely ungrounded response. There is no structural signal to the caller that the agent never actually consulted the registered tools.

### Reproduction

Athena-LightRAG, an MCP server for healthcare EHR analysis, registers four LightRAG query tools on its PromptChain. A multi-hop query asking about patient appointment / billing relationships returned `success: true`, `total_tokens_used: 257`, `execution_time: 39s`, `reasoning_steps: 1` ("Synthesize comprehensive insights..."), `accumulated_contexts_count: 1`, and **zero retrieval calls**. The response content was unrelated to the query or the domain — `gpt-4o-mini`, presented with a system prompt listing tools it couldn't call and no mention of the LightRAG tools it could, skipped tool use entirely and free-associated from training priors. This is the baseline failure mode any non-TUI consumer will encounter.

### Impact on Users

- **Library consumers** (Athena-LightRAG, CDIA, any custom domain agent): ungrounded, off-topic outputs returned with `success: true`. No structural way to detect the failure short of human review.
- **Pre-v0.4.2 users**: `instructions=` calls silently lost effect; agents behave differently from what their code describes.
- **TUI users**: unaffected today, but every unrelated feature change to `AgenticStepProcessor` risks incidentally breaking the embedded prompt.

### Scope Constraint

PromptChain ships a TUI (`promptchain` CLI) whose default agent configuration relies on the current hardcoded prompt + default toolset. **The TUI user experience must not change.** Any fix must be backwards compatible for TUI users at the behavioral level.

---

## Goals

1. **Library correctness by default** — `AgenticStepProcessor` instantiated with no prompt config renders the actually-registered tools into the system prompt.
2. **TUI preservation** — `promptchain` CLI behavior unchanged. The TUI explicitly opts into the existing hardcoded prompt via a named class (`LegacyTUIPromptGenerator`).
3. **Restored compatibility bridge** — `instructions=` parameter re-added as a deprecation shim that maps internally to the new Strategy, so pre-v0.4.2 code runs again.
4. **Type-safe Strategy pattern** — no magic string registries, no `ClassVar` globals, no auto-detection heuristics. Dependency injection only.
5. **Foundation for follow-ups** — the new structure enables (a) groundedness checks against registered tools, (b) TUI prompt modernization, (c) per-domain prompt builders — all as separate future work.

---

## Non-Goals

- Rewriting `AgenticStepProcessor.run()` loop logic.
- Adding groundedness / relevance checks to `execute_multi_hop_reasoning` (separate follow-up).
- Migrating the TUI off the legacy prompt (separate follow-up — tentative `011-tui-prompt-modernization`).
- Restructuring `EnhancedAgenticStepProcessor` or `StateAgent` (they inherit the fix automatically).
- Changing the OpenAI tool-calling schema format — tools continue to be passed as OpenAI-format dicts.

---

## Proposed Solution

Decouple **agent logic** (the reasoning loop in `run()`) from **agent personality** (the system prompt) using the Strategy pattern. Library-first default.

### New module: `promptchain/prompts/`

| File | Purpose |
|------|---------|
| `promptchain/prompts/__init__.py` | Re-exports for public surface |
| `promptchain/prompts/base.py` | `BasePromptBuilder` Protocol — zero internal deps (avoids circular imports) |
| `promptchain/prompts/dynamic.py` | `DynamicPromptGenerator` — renders the tools actually registered on the chain; becomes the new default |
| `promptchain/prompts/legacy_tui.py` | `LegacyTUIPromptGenerator` — returns the current 100-line hardcoded prompt verbatim; used by the TUI via explicit opt-in |

### `BasePromptBuilder` Protocol

```python
class BasePromptBuilder(Protocol):
    def generate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],        # OpenAI-format tool schemas from chain
        context: Optional[str] = None,       # Agent scratchpad / prior step output
    ) -> str: ...

    def get_token_estimate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
    ) -> int: ...                            # Fail-fast guard for 50+ tool bloat
```

### `DynamicPromptGenerator` (new default) output skeleton

```
Your goal is to achieve the following objective: {objective}

AVAILABLE TOOLS:
{one line per registered tool: "- {name}: {description}"}

{if workflow_pattern == "react" AND a task-list tool is registered:
    REACT WORKFLOW (Think/Plan/Act/Observe) block
 elif workflow_pattern == "react" AND no task-list tool:
    thought/action block + log warning
 else: omit block}

{if extra_instructions: ADDITIONAL INSTRUCTIONS block}

{if context: PRIOR CONTEXT block}

FINAL ANSWER REQUIREMENTS:
- Final answer must include full content from tool results.
- Do not summarize tool output into "I have explained ..." — include the content.
```

Constructor parameters: `extra_instructions: Optional[List[str]]`, `workflow_pattern: Literal["standard", "react"] = "standard"`, `include_response_format_hint: bool = True`.

### `LegacyTUIPromptGenerator` — TUI preservation

Returns the current `agentic_step_processor.py:909-1013` prompt verbatim, substituting only `{objective}`. Ignores the `tools` parameter (deliberately — the prompt text is frozen). A warning is logged when `len(tools) != EXPECTED_TUI_DEFAULT_COUNT` so TUI plugin authors who register extra tools at runtime know those tools are not surfaced in the frozen prompt.

### `AgenticStepProcessor.__init__` — additive changes

```python
def __init__(
    self,
    objective: str,
    max_internal_steps: int = 5,
    # ... all existing params unchanged ...
    instructions: Optional[List[str]] = None,               # RESTORED — compat shim, warns
    prompt_builder: Optional[BasePromptBuilder] = None,     # NEW — Strategy
    workflow_pattern: Literal["standard", "react"] = "standard",
):
```

Dispatch logic:

- Both `instructions=` and `prompt_builder=` provided → `ValueError("mutually exclusive")`.
- Only `instructions=` provided → emit `DeprecationWarning`, construct `DynamicPromptGenerator(extra_instructions=instructions, workflow_pattern=workflow_pattern)` internally.
- Only `prompt_builder=` provided → use it; log a warning if the caller also passed `workflow_pattern` (it will be ignored in favor of the custom builder's behavior).
- Neither provided → default to `DynamicPromptGenerator(workflow_pattern=workflow_pattern)`.

### `run()` integration point

The hardcoded f-string at `agentic_step_processor.py:909-1013` is replaced with:

```python
system_prompt = self.prompt_builder.generate(
    objective=self.objective,
    tools=available_tools,
    context=scratchpad_text if scratchpad_text else None,
)
```

### TUI migration

Every `AgenticStepProcessor(...)` call site inside `promptchain/cli/` and `agentic_chat/` must pass `prompt_builder=LegacyTUIPromptGenerator()` explicitly. If more than five call sites exist, introduce a shared `_build_tui_agent()` helper to DRY the opt-in.

**No `ClassVar`, no env var, no magic auto-detection.** Strategy instances are passed explicitly at construction.

---

## Compatibility Matrix

| Consumer | Pre-v0.6.0 Behavior | Post-v0.6.0 Behavior |
|----------|---------------------|----------------------|
| TUI (`promptchain` CLI) | Hardcoded SWE prompt | Same prompt — now via explicit `LegacyTUIPromptGenerator()` |
| Pre-v0.4.2 user with `instructions=[...]` | Broken since v0.4.2 (silent) | Works + `DeprecationWarning` pointing at `DynamicPromptGenerator` |
| Post-v0.4.2 library consumer (Athena-LightRAG, CDIA, custom RAG) | Ungrounded outputs; `success: true` on hallucinated content | `DynamicPromptGenerator` renders their registered tools honestly |
| New user passing `prompt_builder=CustomBuilder()` | N/A | Uses their custom builder |

---

## Acceptance Criteria

1. A library consumer instantiating `AgenticStepProcessor(objective="x")` on a chain with 4 custom tools and no additional config sees those 4 tool names rendered in the system prompt sent to the LLM. The system prompt does NOT contain `task_list_write_tool`, `ripgrep_search`, or any `mcp_gemini_*` reference.
2. A `LegacyTUIPromptGenerator().generate(objective="test", tools=[...])` output matches a frozen snapshot of the v0.5.0 prompt at `agentic_step_processor.py:909-1013`, modulo the `{objective}` substitution.
3. `AgenticStepProcessor(objective="x", instructions=["foo"])` emits a `DeprecationWarning`, completes construction, and produces a system prompt containing "foo" as an additional instruction.
4. `AgenticStepProcessor(objective="x", instructions=[...], prompt_builder=...)` raises `ValueError`.
5. Every `AgenticStepProcessor(...)` call site inside `promptchain/cli/` and `agentic_chat/` explicitly passes `prompt_builder=LegacyTUIPromptGenerator()` (directly or via a shared helper). `grep -rE "AgenticStepProcessor\s*\(" promptchain/cli/ agentic_chat/` shows zero sites without the opt-in.
6. The existing Athena-LightRAG multi-hop reproduction, re-run against the fixed library, returns a response containing content grounded in the LightRAG knowledge graph and non-zero retrieval calls — proving the default path no longer produces ungrounded outputs.
7. All pre-existing PromptChain unit and integration tests pass. Tests that string-matched on fragments of the old hardcoded prompt are updated to assert against the new dynamic-render behavior (when run against a library consumer) or the frozen legacy snapshot (when run against the TUI).
8. CHANGELOG.md documents the release as v0.6.0 with a "Changed (BREAKING)" entry explaining the default prompt behavior change and the `LegacyTUIPromptGenerator` opt-in for consumers who need the old default.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Circular import between `promptchain/prompts/` and `promptchain/utils/agentic_step_processor.py` | `prompts/base.py` contains only a `Protocol` definition with zero internal imports |
| `run()` has hidden string-matching logic that assumes specific keywords from the old hardcoded prompt | Pre-merge audit: `grep` `run()` for literal strings lifted from the old prompt; if any exist, either lift them into the new builders or refactor the `run()` logic to not depend on prompt content |
| Prompt bloat for small-context models (e.g., older gpt-3.5, small local models) when many tools are registered | `workflow_pattern="standard"` remains minimal (≈15 lines for 3 tools). `BasePromptBuilder.get_token_estimate()` lets callers fail fast before hitting the model |
| Test pollution from `DeprecationWarning` | Tests that exercise the `instructions=` compat path wrap in `pytest.warns(DeprecationWarning)` context manager |
| TUI has more `AgenticStepProcessor` instantiation sites than initial grep reveals | Centralize TUI opt-in through a `_build_tui_agent()` helper; document it at the top of `promptchain/cli/session_manager.py` |
| Snapshot of legacy prompt drifts as someone incidentally edits `legacy_tui.py` | Snapshot test in `tests/test_prompt_builders.py`; comment in `legacy_tui.py` pointing at the snapshot test |

---

## Rollout

1. Branch `010-agentic-prompt-builder` off `main`
2. Implementation + tests → PR → CI green
3. Tag `v0.6.0-rc1`
4. CHANGELOG.md entry:
   - **Added:** `prompt_builder=` parameter on `AgenticStepProcessor`; `BasePromptBuilder` / `DynamicPromptGenerator` / `LegacyTUIPromptGenerator` in `promptchain.prompts`
   - **Restored:** `instructions=` parameter (with `DeprecationWarning`, now backed by `DynamicPromptGenerator`)
   - **Changed (BREAKING):** default system prompt is now rendered from the tools actually registered on the chain. Consumers who need pre-v0.6.0 behavior must pass `prompt_builder=LegacyTUIPromptGenerator()` explicitly.
5. Release `v0.6.0`
6. Update downstream consumers known to be affected (Athena-LightRAG `agentic_lightrag.py:393` comment + the workaround that swallowed the v0.4.2 regression)
7. Open follow-up issues:
   - `011-tui-prompt-modernization` — migrate TUI off `LegacyTUIPromptGenerator` by parametrizing its blocks into `DynamicPromptGenerator` workflows
   - `012-groundedness-checks` — add a post-run check that `accumulated_contexts_count > 0` when tools are registered, and surface a warning (or failure) when an agent declared `success: true` without calling any registered tool

---

## Design Debate Record

This PRD is the converged output of a 5-round paired-programming design debate on 2026-04-17. Rejected alternatives:

- **Minimal bolt-on** (`instructions=` + `system_prompt_override=` with legacy default) — rejected for poor discoverability; library consumers upgrading would silently retain the bug.
- **Smart-Switch auto-detection** of "custom tool environment" via tool-name subset check — rejected. A legitimate Gemini-only RAG agent would be misclassified as a TUI environment by any tool-name subset heuristic.
- **ClassVar global default** swap at TUI import time — rejected for process-global mutation side effects (multi-agent / notebook pollution, test ordering dependencies).
- **String-based prompt registry** (`prompt_style="dynamic"`) — rejected for hiding dependencies, killing IDE autocomplete, and failing at runtime instead of type-check time.

The selected approach (Dynamic-default + TUI explicit opt-in via Strategy pattern) was the only candidate that scored positively across all five evaluation criteria: TUI preservation, library ergonomics, discoverability, architectural coherence, and diff size / test-surface risk.

---

## Estimated Effort

Single PR, ~4–6 hours of focused work. No infrastructure changes. No external dependencies. Additive API changes with one deliberate breaking change to the default system prompt — clearly flagged in CHANGELOG and gated behind an explicit opt-in for consumers who need the old behavior.
