# Research — Agentic Prompt Builder Decoupling

**Feature**: 011-agentic-prompt-builder
**Status**: Complete — no outstanding NEEDS CLARIFICATION markers
**Source inputs**: `spec.md`, `prd/agentic_step_processor_prompt_decoupling_prd.md`, and a 2026-04-17 five-round paired-programming design debate summarized in the PRD's Design Debate Record section.

## Purpose

Consolidate the design decisions behind the chosen implementation approach, name the rejected alternatives, and record any research-level findings that downstream phases (data-model, contracts, tasks) depend on.

---

## Decision 1 — Use Strategy pattern with explicit dependency injection

**Decision**: Expose a `BasePromptBuilder` Protocol and two concrete implementations (`DynamicPromptGenerator`, `LegacyTUIPromptGenerator`). `AgenticStepProcessor.__init__` accepts a `prompt_builder=` keyword argument. No process-global defaults, no environment-variable switches, no auto-detection.

**Rationale**:
- Makes the choice of prompt-generation behavior explicit at the point of object construction — the reader of the calling code can see what the agent will do without chasing class-level defaults.
- Allows `isinstance`/`issubclass`-free polymorphism via `typing.Protocol` (structural typing) so downstream consumers can ship their own builders without inheriting from a library class.
- Enables static analysis: IDE autocomplete, mypy verification, runtime catching of typos. A string-based registry ("dynamic", "legacy") would defer all of those checks to runtime.
- Isolates the legacy frozen prompt in a single literal-valued class so a snapshot test can enforce byte-for-byte stability.
- Zero global state — safe for notebook usage, parallel agent runs, and test suites that would otherwise suffer ordering dependencies.

**Alternatives considered and rejected** (transcribed from the PRD's Design Debate Record, with one-line summary rationale):

1. **Minimal bolt-on** (`instructions=` + `system_prompt_override=` with legacy default). Rejected — keeps the lie as the default, so library consumers who upgrade still ship broken agents unless they remember to override. Poor discoverability.
2. **Smart-switch auto-detection** based on tool-name subset check. Rejected — any heuristic that inspects tool names can misclassify legitimate non-TUI agents (for example a Gemini-only RAG agent) as "TUI environment" and serve them the legacy prompt.
3. **`ClassVar` global default** swap at TUI import time. Rejected — process-global mutation has side effects in multi-agent setups, notebooks, and test suites; creates ordering-dependent behavior that is hard to debug.
4. **String-based prompt registry** (`prompt_style="dynamic"`). Rejected — hides the dependency graph, kills IDE autocomplete, and fails at runtime on typos rather than at type-check time.

**Evidence this was the right call**: the selected approach was the only candidate that scored positively across all five evaluation criteria the design debate used: TUI preservation, library ergonomics, discoverability, architectural coherence, and diff size / test-surface risk.

---

## Decision 2 — Use `typing.Protocol` rather than an abstract base class

**Decision**: `BasePromptBuilder` is a `typing.Protocol` (PEP 544, available in Python 3.8+; `runtime_checkable` optional).

**Rationale**:
- Consumers can pass any object with the right method signatures without inheriting from a library class. This matches Python's duck-typing idiom.
- Zero runtime import burden on `promptchain/prompts/base.py` — the module imports only from `typing`, which eliminates the risk of circular imports with `promptchain/utils/agentic_step_processor.py`.
- Supports static type-checking via mypy without requiring callers to couple to a library ABC.
- If a future need arises for runtime isinstance checks (for example a debug-mode builder introspection), `@runtime_checkable` can be added additively without a behavior change.

**Alternatives considered and rejected**:
- **`abc.ABC` with `@abstractmethod`**. Rejected — introduces a runtime import dependency on the library class, forces inheritance, and complicates the import graph. Protocol gives the same compile-time guarantees with fewer constraints.
- **Plain duck typing (no Protocol, no ABC)**. Rejected — loses static type-checking support and IDE tooling hints for third-party authors implementing their own builder.

---

## Decision 3 — Ship two concrete builders at v0.6.0

**Decision**: The release includes exactly two concrete builder classes:
- `DynamicPromptGenerator` — new default, renders live tool schemas.
- `LegacyTUIPromptGenerator` — frozen v0.5.0 prompt, opted into explicitly by the TUI.

**Rationale**:
- Two concrete classes give snapshot testing a clean target: `LegacyTUIPromptGenerator` is a literal-string-returning class that a single `assert got == frozen_snapshot` can lock down.
- A single "dynamic + options" builder with a `legacy_mode=True` flag was considered and rejected — it would force every caller to read the class's internals to know which mode they were getting, and it would intermingle legacy-compat branches with the default code path.
- Future builders (research-mode, domain-specific scaffolds, non-English prompts) are additive: users provide their own class conforming to the Protocol. The library does not need to anticipate or ship them in this release.

**Alternatives considered and rejected**:
- **Single builder with mode flag**. Rejected per above — blurs the snapshot target and couples legacy-compat branches to the default code path.
- **Three builders** (dynamic / legacy / "custom template"). Rejected as premature — YAGNI. The Protocol surface already lets anyone ship a custom template without library changes.

---

## Decision 4 — Token estimation via `tiktoken` with length-based fallback

**Decision**: `BasePromptBuilder.get_token_estimate(objective, tools)` returns an integer token count. Implementation uses `tiktoken` (already a project dependency — `execution_history_manager.py` uses it) with a safe fallback to `len(rendered_prompt) // 4` if tiktoken is unavailable at import time.

**Rationale**:
- Matches the precedent set by `execution_history_manager.py`, keeping token-counting practices uniform across the library.
- Allows callers to fail fast before an oversize LLM call — satisfies the PRD's named risk about 50+ registered tools blowing the context budget on small-context models.
- Fallback path exists because `tiktoken` occasionally has install friction (C extension); the PRD's non-goal list explicitly rules out adding new dependencies, and the `// 4` approximation is honest about being an approximation.

**Alternatives considered and rejected**:
- **Always use `len(s) // 4`**. Rejected — loses accuracy that the rest of the library already has via tiktoken.
- **Require a model-specific tokenizer argument**. Rejected as overengineering — the estimate is a fail-fast guard, not a precision tool. Callers who need exact counts for a specific model can compute them themselves.

---

## Decision 5 — Ship `workflow_pattern` as a 2-mode hint at release

**Decision**: `DynamicPromptGenerator.__init__` accepts `workflow_pattern: Literal["standard", "react"] = "standard"`. Other patterns defer to future work.

**Rationale**:
- The PRD scopes the feature as "decouple prompt from processor" — not "ship every prompt style." YAGNI respected.
- `"standard"` is the minimal default for library consumers. `"react"` preserves a capability the old hardcoded prompt provided (think/plan/act/observe scaffold) for consumers who want it.
- The Protocol surface is the extension point: users who want a research-mode or domain-specific scaffold can ship their own builder. No library change needed.

**Alternatives considered and rejected**:
- **Ship `standard` only**. Rejected — would silently lose a capability (ReAct scaffolding) that existing consumers inherit from the old hardcoded prompt. Regression risk too high.
- **Ship `standard` + `react` + `research` + `domain`**. Rejected — speculative. No concrete consumer exists for the latter two today.

---

## Decision 6 — Subclass propagation via `super().__init__`

**Decision**: No source changes to `EnhancedAgenticStepProcessor` (at `promptchain/utils/enhanced_agentic_step_processor.py:863`) or `StateAgent` (at `promptchain/utils/strategies/state_agent.py:96`). Both already call `super().__init__(...)` and the fix propagates automatically.

**Rationale**:
- Confirmed by code inspection during Phase 0. Both subclasses delegate construction to `AgenticStepProcessor.__init__` and do not independently construct the system prompt.
- Zero source diff in subclass files reduces blast radius and simplifies review.

**Risk**: if a future subclass author overrides `run()` with their own inlined prompt-string construction, they lose the fix. Mitigation — contracts doc and `quickstart.md` both call out that the `run()` method's prompt-generation line is the integration point; any subclass overriding `run()` must also call `self.prompt_builder.generate(...)`.

---

## Decision 7 — Call-site census confirms FR-016 threshold

**Decision**: All terminal-UI-side `AgenticStepProcessor(...)` construction sites are routed through a shared helper at `promptchain/cli/_tui_agent_helper.py` called `_build_tui_agent(...)`. The helper centralizes the `prompt_builder=LegacyTUIPromptGenerator()` opt-in.

**Evidence from code inspection**:
- `promptchain/cli/tui/app.py:3062` — TUI agent construction (direct).
- `promptchain/cli/tui/app.py:3776` — TUI agent construction (direct).
- `promptchain/cli/config/yaml_translator.py:315` — YAML-driven TUI agent config.
- `agentic_chat/agentic_team_chat.py:60` — multi-agent demo (research_step).
- `agentic_chat/agentic_team_chat.py:150` — multi-agent demo (analysis_step).
- `agentic_chat/agentic_team_chat.py:210` — multi-agent demo (terminal_step).
- `agentic_chat/agentic_team_chat.py:526` — multi-agent demo (documentation_step).
- `agentic_chat/agentic_team_chat.py:566` — multi-agent demo (synthesis_step).
- `agentic_chat/agentic_team_chat.py:647` — multi-agent demo (coding_step).
- `agentic_chat/agentic_team_chat.py:1249` — multi-agent demo (orchestrator_step).

Ten sites — exceeds FR-016's threshold of five. Shared helper is therefore mandatory.

**Library-path sites** (will receive the new dynamic default automatically):
- `promptchain/utils/orchestrator_supervisor.py:80` — reasoning_engine in orchestrator.
- `promptchain/utils/chain_executor.py:403` — chain executor's internal processor.

These two sites should NOT be routed through `_build_tui_agent()`. They are library-internal and correctly land on the new default.

**Alternatives considered and rejected**:
- **Per-file inline `prompt_builder=LegacyTUIPromptGenerator()`**. Rejected — 10 duplicate call-site edits, future call sites easily drift off the opt-in.
- **Subclass the processor for TUI use**. Rejected — creates a new class hierarchy for a purely configurational concern. Helper function is simpler.

---

## Decision 8 — Deprecation cycle for `instructions=`

**Decision**: `instructions=` is re-added as an optional kwarg. When supplied, the constructor emits a single `DeprecationWarning` citing `DynamicPromptGenerator(extra_instructions=...)` as the replacement. Hard removal is deferred to a future release (no date committed in v0.6.0).

**Rationale**:
- Restores broken pre-v0.4.2 code immediately with one-step migration.
- One-release warn-before-remove cadence is standard for Python library compatibility.
- The shim is thin (one `warnings.warn(...)` call + construct a `DynamicPromptGenerator` with `extra_instructions=instructions`). Low maintenance cost until removal.

**Alternatives considered and rejected**:
- **Reject `instructions=` outright**. Rejected — leaves the broken v0.4.2 regression unresolved and forces affected users to do a bigger migration.
- **Remove the warning after one release**. Decided against for v0.6.0 — defer the removal to a future decision point. The warning stays as the primary signal in v0.6.0.

---

## Findings that influence downstream phases

- **Circular-import risk is real** — `promptchain/utils/agentic_step_processor.py` is one of the most-imported modules. Placing Protocol definitions in `promptchain/prompts/base.py` with *zero* internal imports is the cleanest solution.
- **Frozen snapshot must live in the test suite, not in the module** — if the snapshot lives inside `legacy_tui.py` as an assertion, accidental edits to that file become silent test-passes. The snapshot is stored under `tests/fixtures/legacy_tui_prompt.snapshot.txt` and the builder file's docstring points at it.
- **Test-file updates for pre-existing prompt string matches** — exactly three files (`tests/test_tao_loop.py`, `tests/test_verification_integration.py`, `tests/cli/integration/test_agentic_reasoning.py`) reference the old hardcoded text. Two assertion types found: (a) substring checks for tool names that now need to split by audience (TUI vs. library), (b) workflow-block matches that should assert against either the frozen snapshot or the dynamic render.
- **Enhanced processor uses `run()` override** — inspection shows `EnhancedAgenticStepProcessor.run()` in `promptchain/utils/enhanced_agentic_step_processor.py` does NOT reinline the system prompt — it augments the reasoning loop around it. The fix propagates cleanly. Confirmed during Phase 0 code walk.

## Outstanding NEEDS CLARIFICATION

None. All design questions that emerged during spec authoring were resolved during the 2026-04-17 design debate. The Phase 1 artifacts (data-model, contracts, quickstart) can be produced directly from this research without further user input.
