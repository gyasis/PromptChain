---
description: "Task list for 006-promptchain-improvements"
---

# Tasks: PromptChain Comprehensive Improvements

**Input**: Design documents from `/specs/006-promptchain-improvements/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓, quickstart.md ✓

**Tests**: Constitution III (TDD) is mandatory — tests are written FIRST and must FAIL before implementation.

**Organization**: Tasks grouped by user story. Most building blocks exist; work is wiring, integration, and gap-filling.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no blocking dependencies)
- **[Story]**: Which user story this task belongs to (US1–US4)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Branch, test structure, and shared test utilities for all stories.

- [x] T001 Create feature branch `006-promptchain-improvements` from `main`
- [x] T002 [P] Create `tests/unit/` subdirectory stubs: `test_interrupt_queue_integration.py`, `test_memo_store_integration.py`, `test_context_distiller_wiring.py`, `test_pubsub_bus.py`, `test_async_agent_inbox.py`, `test_janitor_agent.py`
- [x] T003 [P] Create `tests/integration/test_006_bug_fixes.py` and `tests/integration/test_006_steering_flow.py` with empty test class skeletons

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Cross-cutting fixes and imports that every user story depends on.

**⚠️ CRITICAL**: No user story work begins until this phase is complete.

- [x] T004 Audit `promptchain/observability/config.py` — confirm ALL public accessors (`get_observability_config`, etc.) route through `_load_yaml_config()` cache; add missing routes where found
- [x] T005 [P] Fix `promptchain/observability/queue.py` `shutdown()` method (lines ~172–188): call `self.flush(timeout=timeout)` before `self.worker.join(timeout=timeout)` to guarantee bounded shutdown per FR-004
- [x] T006 [P] Add `copy.deepcopy()` return in `promptchain/utils/enhanced_agentic_step_processor.py` line ~190 (`verify_logic` cache retrieval) to prevent cache corruption per FR-006 / BUG-009
- [x] T007 Harden `promptchain/utils/json_output_parser.py` `extract()` top-level except block (line ~92): catch all `Exception`, log warning with raw string, return configured `default` — never propagate per FR-003

**Checkpoint**: Foundational fixes merged. All four P1/P0 non-Gemini bugs resolved.

---

## Phase 3: User Story 1 — Critical Bug Fixes: Stability & Correctness (Priority: P1) 🎯 MVP

**Goal**: Gemini MCP tools work, TUI pattern commands don't crash on event loop, JSON parser never silently loses data, MLflow queue shuts down cleanly, config cache eliminates redundant reads, verification results are mutation-safe.

**Independent Test**: Run `pytest tests/integration/test_006_bug_fixes.py` — all scenarios pass with zero errors. Exercise `gemini_debug`, `gemini_brainstorm`, `ask_gemini` with mocked MCP server; run a LightRAG pattern command from TUI; feed malformed JSON to parser; trigger MLflow shutdown with unresponsive server; call config twice with no file change; modify a returned verification result and confirm cache unchanged.

### Tests for User Story 1 (TDD — write FIRST, ensure FAIL before implementation)

- [x] T008 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_gemini_debug_correct_params` — mock MCP call, assert `error_message` key present, `error_context` absent
- [x] T009 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_gemini_brainstorm_no_num_ideas` — mock MCP, assert `num_ideas` key absent from call args
- [x] T010 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_ask_gemini_prompt_param` — mock MCP, assert `prompt` key present, `question` absent
- [x] T011 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_event_loop_no_crash_in_tui_context` — simulate Textual running loop, invoke pattern command handler, assert no `RuntimeError`
- [x] T012 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_json_parser_malformed_returns_default` — feed invalid JSON to `JSONOutputParser.extract()`, assert default returned and no exception raised
- [x] T013 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_mlflow_shutdown_bounded` — mock unresponsive queue, call `shutdown(timeout=2.0)`, assert returns within 3 s
- [x] T014 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_config_cache_no_disk_read_on_second_call` — call `get_observability_config()` twice, assert file open count == 1
- [x] T015 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_verification_result_deep_copy` — retrieve cached result, mutate it, re-retrieve, assert original cache entry unchanged

### Implementation for User Story 1

- [x] T016 [US1] Fix `promptchain/utils/enhanced_agentic_step_processor.py` line ~645: change `error_context` → `error_message` in `gemini_debug` tool call arguments per FR-001 / BUG-017
- [x] T017 [US1] Fix `promptchain/utils/enhanced_agentic_step_processor.py` line ~564: remove `num_ideas` parameter from `gemini_brainstorm` tool call arguments per FR-001 / BUG-018
- [x] T018 [US1] Fix `promptchain/utils/enhanced_agentic_step_processor.py` line ~575: change `question` → `prompt` in `ask_gemini` tool call arguments per FR-001 / BUG-019
- [x] T019 [US1] Audit all TUI pattern command handlers in `promptchain/cli/tui/app.py` and any files under `promptchain/cli/` that call `asyncio.run()`: replace with `run_async_in_context()` from `promptchain/cli/utils/event_loop_manager.py` per FR-002 / BUG-001

**Checkpoint**: `pytest tests/integration/test_006_bug_fixes.py` — all 8 tests green. SC-001 satisfied.

---

## Phase 4: User Story 2 — Conversational Longevity: Context & Memory (Priority: P2)

**Goal**: Sessions running 3× beyond previous context limit maintain coherent agent behaviour. Context is automatically distilled at 70% token usage. Learned facts persist across sessions and are semantically retrieved on new sessions.

**Independent Test**: Run a session long enough to exceed 70% of `max_tokens`; confirm a "Current State of Knowledge" summary entry appears in history. Start fresh session, verify relevant memo from previous session is injected into context. Run `pytest tests/unit/test_context_distiller_wiring.py tests/unit/test_memo_store_integration.py tests/unit/test_janitor_agent.py`.

### Tests for User Story 2 (TDD — write FIRST, ensure FAIL before implementation)

- [x] T020 [P] [US2] Write test in `tests/unit/test_context_distiller_wiring.py`: `test_distiller_triggered_at_threshold` — build `ExecutionHistoryManager` at 75% token usage, instantiate `AgenticStepProcessor` with `context_distiller`, call one step, assert `ContextDistiller.distill()` was called
- [x] T021 [P] [US2] Write test in `tests/unit/test_context_distiller_wiring.py`: `test_distiller_not_triggered_below_threshold` — history at 50%, assert `distill()` NOT called
- [x] T022 [P] [US2] Write test in `tests/unit/test_context_distiller_wiring.py`: `test_distiller_llm_failure_leaves_history_unchanged` — mock LLM call raising exception, assert history unmodified after distill attempt
- [x] T023 [P] [US2] Write test in `tests/unit/test_memo_store_integration.py`: `test_memo_injected_into_context_before_llm_call` — store one memo, instantiate `AgenticStepProcessor` with `memo_store`, run step, assert memo content appears in captured context string
- [x] T024 [P] [US2] Write test in `tests/unit/test_memo_store_integration.py`: `test_successful_task_stored_as_memo` — run step that completes successfully, assert `MemoStore.store_memo()` called with `outcome="success"`
- [x] T025 [P] [US2] Write test in `tests/unit/test_janitor_agent.py`: `test_janitor_compresses_at_threshold` — set compression_threshold=0.5, fill history to 60%, start janitor, wait 2× check_interval, assert `distill()` called
- [x] T026 [P] [US2] Write test in `tests/unit/test_janitor_agent.py`: `test_janitor_stop_cancels_task` — start janitor, await stop(), assert background task is cancelled within 5 s

### Implementation for User Story 2

- [x] T027 [US2] Wire `ContextDistiller` into `promptchain/utils/enhanced_agentic_step_processor.py`: add `context_distiller: Optional[ContextDistiller] = None` parameter to `__init__()`, call `should_distill()` / `await distill()` at start of each thought cycle per FR-007
- [x] T028 [US2] Wire `MemoStore` into `promptchain/utils/enhanced_agentic_step_processor.py`: add `memo_store: Optional[MemoStore] = None` parameter to `__init__()`, call `inject_relevant_memos()` before LLM context build, call `store_memo()` on task completion per FR-008/FR-009
- [x] T029 [US2] Create `promptchain/utils/janitor_agent.py`: implement `JanitorAgent` class with `start()`, `stop()`, `_monitor_loop()` as specified in `contracts/context-memory.md`; uses `asyncio.Task` for non-blocking background monitoring per FR-010

**Checkpoint**: Context distillation fires automatically. Memos persist and are retrieved. `pytest tests/unit/test_context_distiller_wiring.py tests/unit/test_memo_store_integration.py tests/unit/test_janitor_agent.py` — all green. SC-002, SC-003, SC-004, SC-008 measurably improved.

---

## Phase 5: User Story 3 — Real-Time User Steering: Interrupt & Override (Priority: P3)

**Goal**: Users can send interrupt signals mid-execution. Agent acknowledges interrupt at next thought cycle boundary. Micro-checkpoints enable partial rewind. Global override replaces active prompt.

**Independent Test**: Start long agentic task, submit `InterruptType.STEERING` mid-run, verify agent context includes interrupt message in next LLM call. Run `pytest tests/unit/test_interrupt_queue_integration.py tests/integration/test_006_steering_flow.py`.

### Tests for User Story 3 (TDD — write FIRST, ensure FAIL before implementation)

- [x] T030 [P] [US3] Write test in `tests/unit/test_interrupt_queue_integration.py`: `test_interrupt_checked_each_thought_cycle` — submit steering interrupt, run 3-step agentic task, assert interrupt processed by step 2
- [x] T031 [P] [US3] Write test in `tests/unit/test_interrupt_queue_integration.py`: `test_abort_interrupt_halts_execution` — submit `ABORT` interrupt, assert `AgenticStepResult.status == "aborted"` returned without further LLM calls
- [x] T032 [P] [US3] Write test in `tests/unit/test_interrupt_queue_integration.py`: `test_steering_message_injected_into_context` — submit steering interrupt, assert interrupt text present in messages passed to LLM
- [x] T033 [P] [US3] Write test in `tests/integration/test_006_steering_flow.py`: `test_micro_checkpoint_saved_after_tool_call` — run step with one tool call, assert `_micro_checkpoints` dict has one entry with `tool_call_index=0`
- [x] T034 [P] [US3] Write test in `tests/integration/test_006_steering_flow.py`: `test_rewind_to_checkpoint_restores_state` — save checkpoint, modify conversation, rewind, assert conversation matches checkpoint snapshot
- [x] T035 [P] [US3] Write test in `tests/integration/test_006_steering_flow.py`: `test_global_override_replaces_prompt` — subscribe `AgenticStepProcessor` to `"agent.global_override"` topic, publish override, assert active prompt updated at next cycle
- [x] T036 [P] [US3] Write test in `tests/integration/test_006_steering_flow.py`: `test_tui_interrupt_command_enqueues_without_blocking` — call `handle_interrupt_command("steering", "msg")`, assert enqueued, assert returns < 10 ms

### Implementation for User Story 3

- [x] T037 [US3] Wire `InterruptQueue` + `InterruptHandler` into `promptchain/utils/enhanced_agentic_step_processor.py`: add `interrupt_queue: Optional[InterruptQueue] = None` to `__init__()`, call `check_and_handle_interrupt()` at start of each thought cycle, handle abort/steering/correction/clarification actions per FR-011
- [x] T038 [US3] Add `MicroCheckpoint` dataclass to `promptchain/utils/checkpoint_manager.py` per `data-model.md` schema; add `_save_micro_checkpoint()` and `rewind_to_last_checkpoint()` methods to `EnhancedAgenticStepProcessor` in `promptchain/utils/enhanced_agentic_step_processor.py`; call save after each successful tool call per FR-013
- [x] T039 [US3] Add `send_global_override()` method and `"agent.global_override"` topic handling to `promptchain/cli/communication/message_bus.py`; wire `_handle_override()` subscription into `AgenticStepProcessor` per FR-014
- [x] T040 [US3] Add `handle_interrupt_command()` to `promptchain/cli/tui/app.py` TUI input handler: non-blocking `submit_interrupt()` call to global `InterruptQueue` per FR-012

**Checkpoint**: Interrupt steering works end-to-end. Micro-checkpoints save and rewind. Override signal propagates. SC-005 satisfied (interrupt ack within 2 s).

---

## Phase 6: User Story 4 — Non-Blocking Async Agent Flows (Priority: P4)

**Goal**: Two agents run simultaneously without blocking each other's I/O. PubSubBus delivers events concurrently to all subscribers. TUI stays responsive (<100 ms) during active LLM calls.

**Independent Test**: Run two `AsyncAgentInbox` agents in parallel tasks; verify neither's receive time increases more than 5% vs single-agent. Subscribe two callbacks to same topic; publish; verify both fired concurrently. Run `pytest tests/unit/test_async_agent_inbox.py tests/unit/test_pubsub_bus.py`.

### Tests for User Story 4 (TDD — write FIRST, ensure FAIL before implementation)

- [x] T041 [P] [US4] Write test in `tests/unit/test_async_agent_inbox.py`: `test_send_receive_normal_priority` — send `InboxMessage(priority=1, topic="task", payload="x")`, receive, assert payload matches
- [x] T042 [P] [US4] Write test in `tests/unit/test_async_agent_inbox.py`: `test_priority_ordering` — send priority-2 then priority-0 messages, assert priority-0 received first
- [x] T043 [P] [US4] Write test in `tests/unit/test_async_agent_inbox.py`: `test_try_receive_returns_none_when_empty` — call `try_receive()` on empty inbox, assert `None` returned without blocking
- [x] T044 [P] [US4] Write test in `tests/unit/test_pubsub_bus.py`: `test_publish_triggers_all_subscribers_concurrently` — subscribe 3 callbacks with `asyncio.Event` flags, publish, assert all 3 flags set
- [x] T045 [P] [US4] Write test in `tests/unit/test_pubsub_bus.py`: `test_subscriber_exception_does_not_propagate` — subscriber raises exception, publish still returns without error
- [x] T046 [P] [US4] Write test in `tests/unit/test_pubsub_bus.py`: `test_unsubscribe_removes_callback` — subscribe, unsubscribe, publish, assert callback NOT called
- [x] T047 [P] [US4] Write test in `tests/unit/test_pubsub_bus.py`: `test_publish_sync_works_from_non_async_context` — call `publish_sync()` from plain function, assert subscriber received payload

### Implementation for User Story 4

- [x] T048 [US4] Create `promptchain/utils/async_agent_inbox.py`: implement `InboxMessage` dataclass and `AsyncAgentInbox` class using `asyncio.PriorityQueue` per `contracts/async-execution.md` and `data-model.md`
- [x] T049 [US4] Add `PubSubBus` class to `promptchain/cli/communication/message_bus.py`: `subscribe()`, `unsubscribe()`, `publish()` (async, `asyncio.gather` fan-out, per-subscriber error isolation), `publish_sync()` wrapper per FR-017 / `contracts/async-execution.md`
- [x] T050 [US4] Refactor LLM calls in `promptchain/utils/agentic_step_processor.py` and `promptchain/utils/enhanced_agentic_step_processor.py`: replace any `litellm.completion()` calls in async contexts with `await litellm.acompletion()` per FR-015
- [x] T051 [US4] Add optional `inbox: Optional[AsyncAgentInbox] = None` parameter to `EnhancedAgenticStepProcessor.__init__()` in `promptchain/utils/enhanced_agentic_step_processor.py`; integrate inbox message polling alongside interrupt queue check per FR-016

**Checkpoint**: Two-agent parallel scenario runs with <5% overhead. PubSubBus fan-out confirmed. SC-006 and SC-007 validated.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Validation of all success criteria, documentation, and cleanup.

- [x] T052 [P] Run `pytest tests/integration/test_006_bug_fixes.py tests/integration/test_006_steering_flow.py tests/unit/ -v` and confirm all tests green; record results in `specs/006-promptchain-improvements/test-results.md`
- [x] T053 [P] Validate SC-001: exercise Gemini tools, TUI event loop, JSON parser with integration tests; confirm zero errors
- [x] T054 [P] Validate SC-004: run token consumption baseline (truncation-only) vs distillation path; confirm ≥30% reduction documented in `specs/006-promptchain-improvements/test-results.md`
- [x] T055 [P] Validate SC-005: time interrupt submission to LLM context inclusion; confirm ≤2 s acknowledgment in `tests/integration/test_006_steering_flow.py::test_interrupt_ack_latency`
- [x] T056 [P] Validate SC-006: two-agent parallel benchmark; confirm <5% I/O overhead; add `tests/integration/test_006_concurrency.py::test_two_agent_overhead`
- [x] T057 Update `CLAUDE.md` "Recent Changes" section to record 006-promptchain-improvements as complete
- [x] T058 [P] Update `promptchain/__init__.py` exports to expose `AsyncAgentInbox`, `PubSubBus`, `JanitorAgent`, `MemoStore`, `InterruptQueue` at top-level package per library-first principle
- [x] T059 [P] Run `black . && isort . && flake8 promptchain/` and fix any linting issues in new/modified files
- [x] T060 Run `mypy promptchain/utils/async_agent_inbox.py promptchain/utils/janitor_agent.py promptchain/cli/communication/message_bus.py` and resolve all type errors

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — BLOCKS all user stories
- **US1 Bug Fixes (Phase 3)**: Depends on Phase 2 — highest priority, tackle first
- **US2 Context & Memory (Phase 4)**: Depends on Phase 2 — independent of US1 (different files)
- **US3 Steering (Phase 5)**: Depends on Phase 2 — lightly depends on US1 (Gemini tools must work to test full flows); can start in parallel
- **US4 Async Execution (Phase 6)**: Depends on Phase 2 — independent of US1–US3 for core implementation; benefits from US3 interrupt infrastructure
- **Polish (Phase 7)**: Depends on all user story phases complete

### User Story Dependencies

- **US1 (P1)**: Starts after Phase 2. No inter-story dependencies.
- **US2 (P2)**: Starts after Phase 2. No inter-story dependencies (wires to existing `AgenticStepProcessor`).
- **US3 (P3)**: Starts after Phase 2. Assumes `AgenticStepProcessor` exists; benefits from US1 Gemini fix being complete for full TUI test coverage.
- **US4 (P4)**: Starts after Phase 2. Fully independent of US1–US3 at implementation level; `PubSubBus` standard topics referenced by US3 but not required for US3 core functionality.

### Within Each User Story

1. Write tests FIRST → confirm FAIL
2. Implement → confirm tests PASS
3. Refactor → tests still PASS
4. Checkpoint before moving to next story

### Parallel Opportunities

- T004, T005, T006, T007 — Phase 2 tasks can all run in parallel (different files)
- T008–T015 — All US1 test-writing tasks are parallel (same file, but disjoint test functions)
- T016, T017, T018 — US1 Gemini fixes are parallel (same file, different lines)
- T020–T026 — All US2 test-writing tasks are parallel
- T027, T028 — US2 implementation tasks share `enhanced_agentic_step_processor.py`; T029 is fully parallel (new file)
- T030–T036 — All US3 test-writing tasks are parallel
- T038, T039, T040 — US3 implementation tasks are parallel (different files); T037 depends on T038 for checkpoint data model
- T041–T047 — All US4 test-writing tasks are parallel
- T048, T049 — US4 new-file tasks are parallel; T050, T051 share step processor file
- T052–T060 — All Polish tasks marked [P] are parallel

---

## Parallel Example: User Story 1 (Bug Fixes)

```bash
# Wave 1 — write all US1 tests in parallel (disjoint test functions in same file):
Task: "T008 — test_gemini_debug_correct_params"
Task: "T009 — test_gemini_brainstorm_no_num_ideas"
Task: "T010 — test_ask_gemini_prompt_param"
Task: "T011 — test_event_loop_no_crash_in_tui_context"
Task: "T012 — test_json_parser_malformed_returns_default"
Task: "T013 — test_mlflow_shutdown_bounded"
Task: "T014 — test_config_cache_no_disk_read_on_second_call"
Task: "T015 — test_verification_result_deep_copy"

# Wave 2 — implement fixes in parallel (different lines / different files):
Task: "T016 — Fix gemini_debug error_context → error_message"
Task: "T017 — Fix gemini_brainstorm remove num_ideas"
Task: "T018 — Fix ask_gemini question → prompt"
Task: "T019 — Fix TUI event loop handlers"
```

## Parallel Example: User Story 4 (Async)

```bash
# Wave 1 — write tests in parallel:
Task: "T041–T043 — AsyncAgentInbox tests"
Task: "T044–T047 — PubSubBus tests"

# Wave 2 — implement in parallel (new files):
Task: "T048 — Create async_agent_inbox.py"
Task: "T049 — Add PubSubBus to message_bus.py"
Task: "T050 — Refactor acompletion in step processors"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T003)
2. Complete Phase 2: Foundational (T004–T007)
3. Complete Phase 3: US1 Bug Fixes (T008–T019)
4. **STOP and VALIDATE**: `pytest tests/integration/test_006_bug_fixes.py` — all green
5. Deploy / demo: Gemini tools work, no crashes, no data loss

### Incremental Delivery

1. Setup + Foundational → infrastructure ready
2. US1 Bug Fixes → zero-error Gemini calls, stable TUI (SC-001) **← MVP**
3. US2 Context & Memory → sessions 3× longer (SC-002, SC-003, SC-004, SC-008)
4. US3 Steering → interrupt control, micro-checkpoints (SC-005)
5. US4 Async Execution → concurrent agents, pub/sub (SC-006, SC-007)
6. Polish → docs, exports, linting, mypy

### Parallel Team Strategy

With 4 developers after Phase 2 completes:

- Developer A: US1 Bug Fixes (Phase 3)
- Developer B: US2 Context & Memory (Phase 4)
- Developer C: US3 Real-Time Steering (Phase 5)
- Developer D: US4 Async Execution (Phase 6)

All four stories can proceed in parallel — they modify largely different files.

---

## Notes

- [P] tasks = different files or disjoint changes, no blocking inter-dependencies
- [Story] label maps each task to its user story for traceability
- Constitution III TDD is NON-NEGOTIABLE: write test → confirm RED → implement → confirm GREEN → refactor
- Most components are already implemented (`InterruptQueue`, `MemoStore`, `ContextDistiller`); verify correct wiring before writing new code
- Commit after each checkpoint with message: `fix(006)/feat(006): <description>`
- Do not modify `tests/` logic to force passes — fix the implementation
