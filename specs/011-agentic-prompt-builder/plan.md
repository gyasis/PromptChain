# Implementation Plan: Agentic Prompt Builder Decoupling

**Branch**: `011-agentic-prompt-builder` | **Date**: 2026-04-17 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/011-agentic-prompt-builder/spec.md`
**Source PRD**: [../../prd/agentic_step_processor_prompt_decoupling_prd.md](../../prd/agentic_step_processor_prompt_decoupling_prd.md)
**Target Release**: v0.6.0
**Tracking Issue**: gyasis/PromptChain#2

## Summary

Introduce a new `promptchain/prompts/` module that decouples the hardcoded terminal-UI system prompt from `AgenticStepProcessor` using the Strategy pattern. The module ships two concrete builders — `DynamicPromptGenerator` (new default, renders the chain's actually-registered tools) and `LegacyTUIPromptGenerator` (frozen v0.5.0 prompt). Restore the removed `instructions=` constructor argument as a `DeprecationWarning`-emitting shim that maps to the dynamic builder. Replace the hardcoded f-string at `agentic_step_processor.py:909-1013` with a `self.prompt_builder.generate(...)` call.

The TUI adopts a dedicated subclass `TUIAgenticStepProcessor(AgenticStepProcessor)` in `promptchain/cli/tui_processor.py` that bakes in `prompt_builder=LegacyTUIPromptGenerator()` and forwards all other kwargs. The TUI's **default** construction sites (currently 10 — 3 in CLI, 7 in `agentic_chat/`) switch to that subclass so TUI code reads naturally as `TUIAgenticStepProcessor(...)`. Crucially the TUI is **not locked into a single processor class**:

- **Option A — default path**: `TUIAgenticStepProcessor(...)` for the TUI's baseline agent behavior.
- **Option B — swappable processor class**: TUI code is free to construct any of the other processor variants directly for specialized tasks — vanilla `AgenticStepProcessor(...)` for sub-agents with custom per-call instructions, `EnhancedAgenticStepProcessor(...)` for RAG / Gemini / memo-store workflows, `StateAgent(...)` for state-machine-driven flows, or a third-party custom subclass. Each of those, by virtue of `AgenticStepProcessor`'s new default dispatch, receives the library-correct `DynamicPromptGenerator` unless the caller explicitly passes `prompt_builder=` or `instructions=`.
- **Sub-agent spawning**: When an orchestrator agent inside the TUI launches a sub-agent, it uses vanilla `AgenticStepProcessor(objective=..., instructions=[...])` (or with a custom `prompt_builder=`) so the sub-agent's system prompt matches its actual per-call instructions rather than the frozen TUI default.

The subclass is the TUI's convenient opt-in for the legacy prompt, not a forced channel. Two library-path call sites (`orchestrator_supervisor.py`, `chain_executor.py`) drop to the new default. Two existing subclasses (`EnhancedAgenticStepProcessor`, `StateAgent`) inherit the fix automatically via `super().__init__`.

Because the subclass itself centralizes the legacy-prompt opt-in, the previously planned `_build_tui_agent()` helper is no longer needed and is dropped from the design.

## Technical Context

**Language/Version**: Python 3.10+ (Protocol requires 3.8+; repo CI runs 3.10 and 3.12; current active: 3.12.11)
**Primary Dependencies**: litellm (LLM calls, unchanged), tiktoken (token estimation, already in use for `execution_history_manager.py`), Textual/Rich (TUI layer, unchanged), standard library `warnings` and `typing.Protocol` for the new module
**Storage**: N/A (this feature ships in-process strategy objects with no persistence)
**Testing**: pytest with existing conventions under `tests/` — new `tests/test_prompt_builders.py` for the module itself, updates to 3 test files that string-matched the old prompt (`tests/test_tao_loop.py`, `tests/test_verification_integration.py`, `tests/cli/integration/test_agentic_reasoning.py`), plus `pytest.warns(DeprecationWarning)` wrappers for the compat-shim path
**Target Platform**: Linux / macOS / Windows Python runtime (identical to rest of PromptChain). No platform-specific code.
**Project Type**: Single project — Python library with TUI CLI. Uses existing `promptchain/` package layout; new sub-package `promptchain/prompts/` is additive.
**Performance Goals**: Dynamic builder output for 3 tools in `standard` workflow mode stays ≤ ~15 lines / ~200 tokens; for 10 tools ≤ ~40 lines / ~600 tokens. `get_token_estimate()` must return in O(n_tools) time with negligible overhead vs. the LLM call it guards. No async path — prompt generation is synchronous and fits inside the existing async reasoning loop.
**Constraints**: (a) zero net new runtime dependencies — only stdlib + already-installed libraries; (b) no circular imports between `promptchain/prompts/` and `promptchain/utils/agentic_step_processor.py`; (c) no process-global mutable state (no `ClassVar` defaults, no env-var switches, no auto-detection); (d) legacy prompt output byte-identical to v0.5.0 modulo `{objective}` substitution; (e) single deprecation cycle — warn in v0.6.0, hard removal decision deferred.
**Scale/Scope**: 12 direct `AgenticStepProcessor(...)` call sites (2 TUI + 1 YAML translator + 7 agentic_chat + 2 library). 2 existing subclasses (`EnhancedAgenticStepProcessor`, `StateAgent`) inherit the fix. 1 new subclass (`TUIAgenticStepProcessor`) added for the TUI's default path. ~100 lines of hardcoded prompt to lift out. New module estimated at ~350 LOC total: `base.py` ~40 LOC, `dynamic.py` ~180 LOC, `legacy_tui.py` ~140 LOC, `__init__.py` ~10 LOC. New `tui_processor.py` ~40 LOC. Test file ~300 LOC. Total delta including 10 TUI call-site edits: ~800 LOC added, ~110 LOC removed.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Evaluating against the 7 core principles in `.specify/memory/constitution.md`:

| Principle | Compliance | Evidence |
|-----------|------------|----------|
| **I. Library-First Architecture** | ✅ PASS | `promptchain/prompts/` is a standalone, self-contained sub-package. `base.py` has zero internal imports (only `typing`). Builders are independently testable without a chain instance. Purpose is concrete (not organizational): expose the Strategy surface. |
| **II. Observable Systems** | ✅ PASS | Builders log their selection at construction time. `DynamicPromptGenerator` logs a warning when `workflow_pattern="react"` but no task-list tool is registered. `LegacyTUIPromptGenerator` logs a warning when the tool inventory drifts from the frozen default. These integrate with the existing `logger = logging.getLogger(__name__)` pattern. No private-attribute access required; everything inspectable through `processor.prompt_builder`. |
| **III. Test-First Development** | ✅ PASS | Phase 1 produces `tests/test_prompt_builders.py` *before* any implementation in `promptchain/prompts/`. The test surface is enumerated in `quickstart.md` with concrete red-state expectations: frozen snapshot assertion, dynamic-render assertion, deprecation-warning capture, `ValueError` on mutually-exclusive args. Constitution's red-green-refactor cycle applies cleanly. |
| **IV. Integration Testing** | ✅ PASS | Contracts file enumerates the builder Protocol as a first-class public API subject to contract tests. Inter-component tests cover processor ↔ builder ↔ chain. Subclass tests (`EnhancedAgenticStepProcessor`, `StateAgent`) prove the fix propagates through `super().__init__`. One end-to-end integration test mirrors the PRD reproduction (four custom tools, multi-hop question, non-zero retrieval calls). |
| **V. Token Economy & Performance** | ✅ PASS | `BasePromptBuilder.get_token_estimate()` is a first-class method, letting callers fail fast before an oversize LLM call. `DynamicPromptGenerator` default `standard` mode stays minimal (~15 lines for 3 tools). The new default is *cheaper* than the status quo for library consumers (who today ship the ~100-line legacy prompt needlessly). No regression for TUI users (they get the same prompt they had). |
| **VI. Async-First Design** | ✅ PASS (N/A at implementation level) | Prompt generation is pure synchronous string construction and fits cleanly inside the existing async `run()` method. No async/sync pair is needed for the builders themselves. The processor's existing async/sync dual interface is preserved unchanged. |
| **VII. Simplicity & Maintainability** | ✅ PASS | Two concrete builders, one protocol, one constructor-dispatch method — nothing more. No registry, no auto-detection, no env vars. YAGNI respected: no `React`/`Standard`/`Domain`/`Research` variant explosion shipped speculatively; only `workflow_pattern` hint with 2 modes. Explicit injection at construction time. |

**Post-Phase-1 re-evaluation**: See §"Post-Design Constitution Re-Check" at the end of this document.

**Gate status**: ALL PASS. No violations to track.

## Project Structure

### Documentation (this feature)

```text
specs/011-agentic-prompt-builder/
├── plan.md              # This file
├── research.md          # Phase 0 — decisions + alternatives
├── data-model.md        # Phase 1 — entities (BasePromptBuilder, DynamicPromptGenerator, LegacyTUIPromptGenerator, etc.)
├── quickstart.md        # Phase 1 — how a library consumer and TUI maintainer each use the new API
├── contracts/
│   └── prompt_builder_protocol.md   # Phase 1 — formal contract for the BasePromptBuilder Protocol
├── checklists/
│   └── requirements.md  # (already created by /speckit.specify)
└── tasks.md             # (will be created by /speckit.tasks)
```

### Source Code (repository root)

```text
promptchain/
├── prompts/                                 # NEW — Strategy pattern surface
│   ├── __init__.py                           # Re-exports BasePromptBuilder, DynamicPromptGenerator, LegacyTUIPromptGenerator
│   ├── base.py                               # BasePromptBuilder Protocol. ZERO internal imports.
│   ├── dynamic.py                            # DynamicPromptGenerator — new default
│   └── legacy_tui.py                         # LegacyTUIPromptGenerator — frozen v0.5.0 prompt
├── utils/
│   ├── agentic_step_processor.py             # MODIFIED — __init__ accepts prompt_builder/instructions; run() calls self.prompt_builder.generate()
│   ├── enhanced_agentic_step_processor.py    # UNCHANGED source — inherits fix via super().__init__
│   └── strategies/
│       └── state_agent.py                    # UNCHANGED source — inherits fix via super().__init__
└── cli/
    ├── tui_processor.py                      # NEW — TUIAgenticStepProcessor subclass (Option A default path)
    ├── tui/app.py                            # MODIFIED — default sites use TUIAgenticStepProcessor; sub-agent spawn sites keep vanilla AgenticStepProcessor
    └── config/yaml_translator.py             # MODIFIED — line 315 uses TUIAgenticStepProcessor

agentic_chat/
└── agentic_team_chat.py                      # MODIFIED — 7 default construction sites use TUIAgenticStepProcessor; any sub-agent spawning remains free to use vanilla/Enhanced/StateAgent

tests/
├── test_prompt_builders.py                   # NEW — unit + contract tests for the new module
├── integration/
│   └── test_011_library_consumer_flow.py     # NEW — end-to-end: custom tools → dynamic prompt → non-zero retrieval
├── test_tao_loop.py                          # MODIFIED — assertions updated to reference dynamic/legacy split
├── test_verification_integration.py          # MODIFIED — same
└── cli/integration/
    └── test_agentic_reasoning.py             # MODIFIED — same

CHANGELOG.md                                  # MODIFIED — add v0.6.0 entry (Added + Restored + Changed BREAKING)
```

**Structure Decision**: **Option 1 — Single project (default)**. The feature is purely an additive sub-package inside an existing Python library. No backend/frontend split, no mobile component. The new `promptchain/prompts/` directory slots in next to `promptchain/utils/` with identical conventions. The TUI's default-path opt-in is centralized through a new subclass at `promptchain/cli/tui_processor.py::TUIAgenticStepProcessor` rather than a factory helper — this gives TUI code a natural constructor call and leaves it free to also construct vanilla `AgenticStepProcessor`, `EnhancedAgenticStepProcessor`, `StateAgent`, or custom subclasses for specialized workflows (Option B). Tests follow the existing `tests/` structure — a new unit/contract file for the module itself, a new integration file for the PRD reproduction, and minimal edits to three existing tests that string-matched the old hardcoded prompt.

## Phase 0 — Research Output

See [research.md](research.md) for the full research record. Decisions, in brief:

1. **Strategy pattern with explicit dependency injection** chosen over four alternatives (minimal bolt-on `system_prompt_override=`, smart-switch auto-detection, `ClassVar` global default, string-based registry). Rationale and rejected alternatives documented in the PRD's Design Debate Record and transcribed verbatim into `research.md`.
2. **`typing.Protocol` chosen for `BasePromptBuilder`** over abstract base class — enables structural typing so consumers can pass any object with the right methods without inheriting a library class. Avoids the import cycle risk of an ABC.
3. **Two concrete builders at release** — `DynamicPromptGenerator` and `LegacyTUIPromptGenerator` — chosen over a single "dynamic + options" builder with a `legacy_mode=True` flag. Keeps the legacy prompt literal-frozen (easier snapshot testing) and keeps the dynamic builder free of legacy-compat branches.
4. **Token estimation via `tiktoken`** (already a project dep) with a graceful fallback to `len(s) // 4` if tiktoken is not available at import time. Matches the existing `execution_history_manager.py` pattern.
5. **Single `workflow_pattern` enum with 2 modes at release** — `"standard"` and `"react"`. Other patterns (research, domain-specific) defer to future work; `BasePromptBuilder` being a Protocol makes those additive.
6. **Subclass behavior confirmed** — both `EnhancedAgenticStepProcessor` and `StateAgent` call `super().__init__(...)` and take no independent action on system-prompt construction. They inherit the fix with zero source changes.
7. **Call-site census: 10 TUI-style sites + 2 library-path sites + 2 subclasses**. Ten exceeds the FR-016 threshold of five, so the shared `_build_tui_agent()` helper is mandatory.

**NEEDS CLARIFICATION**: None. All unknowns from the PRD were resolved during the design debate that produced the spec. The implementation plan can proceed directly to Phase 1.

## Phase 1 — Design & Contracts

Outputs produced in this phase (see companion files):

- **[data-model.md](data-model.md)** — entity model for `BasePromptBuilder`, `DynamicPromptGenerator`, `LegacyTUIPromptGenerator`, tool-schema shape, processor dispatch state machine.
- **[contracts/prompt_builder_protocol.md](contracts/prompt_builder_protocol.md)** — formal contract: method signatures, pre/postconditions, invariants, error behavior. This is the public API surface callers can rely on.
- **[quickstart.md](quickstart.md)** — one-screen walkthrough for two audiences: (a) library consumer migrating off the implicit-TUI-default behavior, (b) TUI maintainer adding a new `AgenticStepProcessor(...)` call site.
- **Agent context file update** — `.specify/scripts/bash/update-agent-context.sh claude` will append a "Recent Changes" entry to the project `CLAUDE.md` noting the new `promptchain/prompts/` module and the behavior change at default.

### Post-Design Constitution Re-Check

Re-evaluating the 7 principles after the Phase 1 artifacts exist:

| Principle | Status | Evidence from Phase 1 artifacts |
|-----------|--------|---------------------------------|
| I. Library-First | ✅ PASS | `data-model.md` confirms `base.py` imports only `typing`. `contracts/` document positions the Protocol as independent, composable, and testable without a chain. |
| II. Observable Systems | ✅ PASS | `contracts/` enumerate the exact warning points (`DynamicPromptGenerator.__init__` in `react` mode without task-list tool; `LegacyTUIPromptGenerator.generate` when tools drift from the frozen inventory). Both integrate with stdlib `logging`. |
| III. Test-First | ✅ PASS | `quickstart.md` opens with the test surface (frozen snapshot, dynamic render, deprecation warning, mutual-exclusion ValueError, library-consumer e2e). Implementation tasks will be produced in Phase 2 (`/speckit.tasks`) with tests scheduled ahead of each implementation pair. |
| IV. Integration Testing | ✅ PASS | `contracts/` define the contract test; `quickstart.md` defines the e2e test; `data-model.md` defines the processor-dispatch state machine that is the integration-test target. |
| V. Token Economy | ✅ PASS | `contracts/` require `get_token_estimate()` to be O(n_tools). `data-model.md` constrains the default `standard` render to ~15 lines for 3 tools. |
| VI. Async-First | ✅ PASS (N/A at this layer) | Builders are synchronous pure-string producers. They live cleanly inside the already-async `run()` method. |
| VII. Simplicity | ✅ PASS | Two concrete builders, one Protocol, one helper, one constructor dispatch. Nothing speculative. `research.md` documents what was explicitly *not* shipped. |

**Post-design gate status**: ALL PASS. No new violations introduced by the design artifacts.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations. The Complexity Tracking table is intentionally empty — every design decision traced back cleanly to one of the 7 core principles. The deliberate non-decisions (no auto-detection, no registry, no global default, no speculative workflow modes) are recorded in `research.md` under "Rejected alternatives" for future reference.
