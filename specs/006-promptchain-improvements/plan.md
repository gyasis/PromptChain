# Implementation Plan: PromptChain Comprehensive Improvements

**Branch**: `006-promptchain-improvements` | **Date**: 2026-02-24 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/006-promptchain-improvements/spec.md`

---

## Summary

Fix six production bugs (Gemini MCP parameter mismatches, TUI event-loop crashes,
JSON parser data loss, MLflow shutdown hang, config re-reads, verification cache
corruption) and add four major capability areas: context distillation, semantic
memo store, real-time interrupt steering, and non-blocking async agent execution.
Most building blocks already exist in the codebase; this plan wires them together
and fills the gaps.

---

## Technical Context

**Language/Version**: Python 3.10.15
**Primary Dependencies**: litellm 1.70.2, tiktoken 0.9.0, Textual (TUI), Rich 13.8+,
asyncio (stdlib), SQLite 3 (stdlib), numpy (MemoStore embeddings)
**Storage**: SQLite (memo store at `~/.promptchain/memos.db`, existing session DB),
in-memory dicts for micro-checkpoints
**Testing**: pytest — unit, integration, e2e test structure under `tests/`
**Target Platform**: Linux server / developer workstation (Python 3.10+)
**Project Type**: Single Python library with CLI/TUI frontend
**Performance Goals**: TUI responds in <100 ms during active LLM task; interrupt
acknowledged within 2 s of next thought cycle boundary; <5% overhead per agent
in two-agent concurrent scenario
**Constraints**: Backward-compatible sync APIs; no new required external dependencies
(numpy already present); async-first internally with sync wrappers
**Scale/Scope**: Individual developer sessions; single-host multi-agent execution
(2–10 concurrent agents)

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Library-First Architecture | PASS | All new components are self-contained (InterruptQueue, MemoStore, PubSubBus, AsyncAgentInbox, JanitorAgent) with clear boundaries and independent testability |
| II. Observable Systems | PASS | All new components emit structured events via PubSubBus topics; JanitorAgent logs compression events; interrupt handling logs all state transitions |
| III. Test-First Development | PASS | TDD mandatory — tests written first for all changes; test files listed in Phase 2 |
| IV. Integration Testing | PASS | Integration tests for component wiring: bug fixes, steering flow, async concurrency |
| V. Token Economy & Performance | PASS | Core motivation of FR-007/FR-010; distillation targets 30%+ token reduction |
| VI. Async-First Design | PASS | All new APIs are async-primary with sync wrappers; LLM calls use `acompletion` |
| VII. Simplicity & Maintainability | PASS | Leverages existing implementations (InterruptQueue, MemoStore, ContextDistiller already built); wiring, not greenfield |

**Complexity Tracking**: No violations. No justified exceptions required.

---

## Project Structure

### Documentation (this feature)

```text
specs/006-promptchain-improvements/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   ├── bug-fixes.md
│   ├── context-memory.md
│   ├── steering.md
│   └── async-execution.md
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
promptchain/
├── utils/
│   ├── enhanced_agentic_step_processor.py  # MODIFY: Gemini params, deep copy, distiller/memo/interrupt wiring
│   ├── agentic_step_processor.py           # MODIFY: LLM calls to acompletion, interrupt wiring
│   ├── execution_history_manager.py        # MODIFY: ContextDistiller wiring (already exists at line 519)
│   ├── json_output_parser.py               # MODIFY: Graceful fallback in extract()
│   ├── interrupt_queue.py                  # EXISTS: No changes needed
│   ├── memo_store.py                       # EXISTS: No changes needed
│   ├── checkpoint_manager.py              # EXISTS: MicroCheckpoint addition
│   ├── async_agent_inbox.py               # NEW: AsyncAgentInbox + InboxMessage
│   └── janitor_agent.py                   # NEW: JanitorAgent background compression
├── observability/
│   ├── queue.py                            # MODIFY: shutdown() uses flush() with bounded timeout
│   └── config.py                           # AUDIT: confirm all access routes through cache
└── cli/
    ├── utils/
    │   └── event_loop_manager.py           # EXISTS: Adopt in all pattern command handlers
    ├── communication/
    │   └── message_bus.py                  # MODIFY: Add PubSubBus class + GlobalOverrideSignal
    └── tui/
        └── app.py                          # MODIFY: Wire interrupt command handler

tests/
├── unit/
│   ├── test_interrupt_queue_integration.py  # NEW
│   ├── test_memo_store_integration.py       # NEW
│   ├── test_context_distiller_wiring.py     # NEW
│   ├── test_pubsub_bus.py                   # NEW
│   ├── test_async_agent_inbox.py            # NEW
│   └── test_janitor_agent.py               # NEW
└── integration/
    ├── test_006_bug_fixes.py               # NEW
    └── test_006_steering_flow.py           # NEW
```

**Structure Decision**: Single project layout, existing source tree. No new
top-level directories. New files are isolated to their natural home within
`promptchain/utils/` and `promptchain/cli/communication/`.

---

## Phase 0: Research Output

See [research.md](research.md) — all NEEDS CLARIFICATION items resolved.

**Key findings**:
- Six of the eight required components are already implemented; work is wiring/integration
- Backward-compatible parameter injection via `Optional` defaults
- No new external dependencies required

---

## Phase 1: Design Output

- [data-model.md](data-model.md) — Entity schemas, relationships, validation rules
- [contracts/bug-fixes.md](contracts/bug-fixes.md) — FR-001 to FR-006
- [contracts/context-memory.md](contracts/context-memory.md) — FR-007 to FR-010
- [contracts/steering.md](contracts/steering.md) — FR-011 to FR-014
- [contracts/async-execution.md](contracts/async-execution.md) — FR-015 to FR-017
- [quickstart.md](quickstart.md) — Developer quick-start for all new features

---

## Implementation Phases (for tasks.md)

### Phase 1: Bug Fixes (P0/P1) — Zero Breakage
Tasks: Fix Gemini params, event loop adoption, JSON fallback, MLflow shutdown,
config cache audit, verification deep copy.
All changes are targeted one-line to five-line fixes with high test coverage.

### Phase 2: Context & Memory
Tasks: Wire `ContextDistiller` into step processor, wire `MemoStore` into step
processor, implement `JanitorAgent`.
Components exist; work is integration + tests.

### Phase 3: Real-Time Steering
Tasks: Wire `InterruptQueue` into step processor thought cycle, TUI interrupt
command handler, `MicroCheckpoint` save/rewind, `GlobalOverrideSignal` on bus.

### Phase 4: Non-Blocking Async Execution
Tasks: `AsyncAgentInbox` implementation, `PubSubBus` implementation, refactor
LLM calls to `acompletion`, integration tests for concurrency.

### Phase 5: Validation & Success Criteria
Tasks: Integration test suite for all SCs (SC-001 to SC-008), performance
benchmarks, documentation updates.
