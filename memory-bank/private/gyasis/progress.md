# Progress

**Last Updated**: 2026-02-25 06:23:03

## Overall Progress
- Total Tasks: 62
- Completed: 40 ✅
- Pending: 22 ⏳
- Progress: 64%

## Task Breakdown
- [x] T001 Create feature branch `006-promptchain-improvements` from `main`
- [x] T002 [P] Create `tests/unit/` subdirectory stubs: `test_interrupt_queue_integration.py`, `test_memo_store_integration.py`, `test_context_distiller_wiring.py`, `test_pubsub_bus.py`, `test_async_agent_inbox.py`, `test_janitor_agent.py`
- [x] T003 [P] Create `tests/integration/test_006_bug_fixes.py` and `tests/integration/test_006_steering_flow.py` with empty test class skeletons
- [x] T004 Audit `promptchain/observability/config.py` — confirm ALL public accessors (`get_observability_config`, etc.) route through `_load_yaml_config()` cache; add missing routes where found
- [x] T005 [P] Fix `promptchain/observability/queue.py` `shutdown()` method (lines ~172–188): call `self.flush(timeout=timeout)` before `self.worker.join(timeout=timeout)` to guarantee bounded shutdown per FR-004
- [x] T006 [P] Add `copy.deepcopy()` return in `promptchain/utils/enhanced_agentic_step_processor.py` line ~190 (`verify_logic` cache retrieval) to prevent cache corruption per FR-006 / BUG-009
- [x] T007 Harden `promptchain/utils/json_output_parser.py` `extract()` top-level except block (line ~92): catch all `Exception`, log warning with raw string, return configured `default` — never propagate per FR-003
- [x] T008 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_gemini_debug_correct_params` — mock MCP call, assert `error_message` key present, `error_context` absent
- [x] T009 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_gemini_brainstorm_no_num_ideas` — mock MCP, assert `num_ideas` key absent from call args
- [x] T010 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_ask_gemini_prompt_param` — mock MCP, assert `prompt` key present, `question` absent
- [x] T011 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_event_loop_no_crash_in_tui_context` — simulate Textual running loop, invoke pattern command handler, assert no `RuntimeError`
- [x] T012 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_json_parser_malformed_returns_default` — feed invalid JSON to `JSONOutputParser.extract()`, assert default returned and no exception raised
- [x] T013 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_mlflow_shutdown_bounded` — mock unresponsive queue, call `shutdown(timeout=2.0)`, assert returns within 3 s
- [x] T014 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_config_cache_no_disk_read_on_second_call` — call `get_observability_config()` twice, assert file open count == 1
- [x] T015 [P] [US1] Write test in `tests/integration/test_006_bug_fixes.py`: `test_verification_result_deep_copy` — retrieve cached result, mutate it, re-retrieve, assert original cache entry unchanged
- [x] T016 [US1] Fix `promptchain/utils/enhanced_agentic_step_processor.py` line ~645: change `error_context` → `error_message` in `gemini_debug` tool call arguments per FR-001 / BUG-017
- [x] T017 [US1] Fix `promptchain/utils/enhanced_agentic_step_processor.py` line ~564: remove `num_ideas` parameter from `gemini_brainstorm` tool call arguments per FR-001 / BUG-018
- [x] T018 [US1] Fix `promptchain/utils/enhanced_agentic_step_processor.py` line ~575: change `question` → `prompt` in `ask_gemini` tool call arguments per FR-001 / BUG-019
- [x] T019 [US1] Audit all TUI pattern command handlers in `promptchain/cli/tui/app.py` and any files under `promptchain/cli/` that call `asyncio.run()`: replace with `run_async_in_context()` from `promptchain/cli/utils/event_loop_manager.py` per FR-002 / BUG-001
- [x] T020 [P] [US2] Write test in `tests/unit/test_context_distiller_wiring.py`: `test_distiller_triggered_at_threshold` — build `ExecutionHistoryManager` at 75% token usage, instantiate `AgenticStepProcessor` with `context_distiller`, call one step, assert `ContextDistiller.distill()` was called
- [x] T021 [P] [US2] Write test in `tests/unit/test_context_distiller_wiring.py`: `test_distiller_not_triggered_below_threshold` — history at 50%, assert `distill()` NOT called
- [x] T022 [P] [US2] Write test in `tests/unit/test_context_distiller_wiring.py`: `test_distiller_llm_failure_leaves_history_unchanged` — mock LLM call raising exception, assert history unmodified after distill attempt
- [x] T023 [P] [US2] Write test in `tests/unit/test_memo_store_integration.py`: `test_memo_injected_into_context_before_llm_call` — store one memo, instantiate `AgenticStepProcessor` with `memo_store`, run step, assert memo content appears in captured context string
- [x] T024 [P] [US2] Write test in `tests/unit/test_memo_store_integration.py`: `test_successful_task_stored_as_memo` — run step that completes successfully, assert `MemoStore.store_memo()` called with `outcome="success"`
- [x] T025 [P] [US2] Write test in `tests/unit/test_janitor_agent.py`: `test_janitor_compresses_at_threshold` — set compression_threshold=0.5, fill history to 60%, start janitor, wait 2× check_interval, assert `distill()` called
- [x] T026 [P] [US2] Write test in `tests/unit/test_janitor_agent.py`: `test_janitor_stop_cancels_task` — start janitor, await stop(), assert background task is cancelled within 5 s
- [ ] T027 [US2] Wire `ContextDistiller` into `promptchain/utils/enhanced_agentic_step_processor.py`: add `context_distiller: Optional[ContextDistiller] = None` parameter to `__init__()`, call `should_distill()` / `await distill()` at start of each thought cycle per FR-007
- [ ] T028 [US2] Wire `MemoStore` into `promptchain/utils/enhanced_agentic_step_processor.py`: add `memo_store: Optional[MemoStore] = None` parameter to `__init__()`, call `inject_relevant_memos()` before LLM context build, call `store_memo()` on task completion per FR-008/FR-009
- [ ] T029 [US2] Create `promptchain/utils/janitor_agent.py`: implement `JanitorAgent` class with `start()`, `stop()`, `_monitor_loop()` as specified in `contracts/context-memory.md`; uses `asyncio.Task` for non-blocking background monitoring per FR-010
- [x] T030 [P] [US3] Write test in `tests/unit/test_interrupt_queue_integration.py`: `test_interrupt_checked_each_thought_cycle` — submit steering interrupt, run 3-step agentic task, assert interrupt processed by step 2
- [x] T031 [P] [US3] Write test in `tests/unit/test_interrupt_queue_integration.py`: `test_abort_interrupt_halts_execution` — submit `ABORT` interrupt, assert `AgenticStepResult.status == "aborted"` returned without further LLM calls
- [x] T032 [P] [US3] Write test in `tests/unit/test_interrupt_queue_integration.py`: `test_steering_message_injected_into_context` — submit steering interrupt, assert interrupt text present in messages passed to LLM
- [x] T033 [P] [US3] Write test in `tests/integration/test_006_steering_flow.py`: `test_micro_checkpoint_saved_after_tool_call` — run step with one tool call, assert `_micro_checkpoints` dict has one entry with `tool_call_index=0`
- [x] T034 [P] [US3] Write test in `tests/integration/test_006_steering_flow.py`: `test_rewind_to_checkpoint_restores_state` — save checkpoint, modify conversation, rewind, assert conversation matches checkpoint snapshot
- [x] T035 [P] [US3] Write test in `tests/integration/test_006_steering_flow.py`: `test_global_override_replaces_prompt` — subscribe `AgenticStepProcessor` to `"agent.global_override"` topic, publish override, assert active prompt updated at next cycle
- [x] T036 [P] [US3] Write test in `tests/integration/test_006_steering_flow.py`: `test_tui_interrupt_command_enqueues_without_blocking` — call `handle_interrupt_command("steering", "msg")`, assert enqueued, assert returns < 10 ms
- [ ] T037 [US3] Wire `InterruptQueue` + `InterruptHandler` into `promptchain/utils/enhanced_agentic_step_processor.py`: add `interrupt_queue: Optional[InterruptQueue] = None` to `__init__()`, call `check_and_handle_interrupt()` at start of each thought cycle, handle abort/steering/correction/clarification actions per FR-011
- [ ] T038 [US3] Add `MicroCheckpoint` dataclass to `promptchain/utils/checkpoint_manager.py` per `data-model.md` schema; add `_save_micro_checkpoint()` and `rewind_to_last_checkpoint()` methods to `EnhancedAgenticStepProcessor` in `promptchain/utils/enhanced_agentic_step_processor.py`; call save after each successful tool call per FR-013
- [ ] T039 [US3] Add `send_global_override()` method and `"agent.global_override"` topic handling to `promptchain/cli/communication/message_bus.py`; wire `_handle_override()` subscription into `AgenticStepProcessor` per FR-014
- [ ] T040 [US3] Add `handle_interrupt_command()` to `promptchain/cli/tui/app.py` TUI input handler: non-blocking `submit_interrupt()` call to global `InterruptQueue` per FR-012
- [x] T041 [P] [US4] Write test in `tests/unit/test_async_agent_inbox.py`: `test_send_receive_normal_priority` — send `InboxMessage(priority=1, topic="task", payload="x")`, receive, assert payload matches
- [x] T042 [P] [US4] Write test in `tests/unit/test_async_agent_inbox.py`: `test_priority_ordering` — send priority-2 then priority-0 messages, assert priority-0 received first
- [x] T043 [P] [US4] Write test in `tests/unit/test_async_agent_inbox.py`: `test_try_receive_returns_none_when_empty` — call `try_receive()` on empty inbox, assert `None` returned without blocking
- [x] T044 [P] [US4] Write test in `tests/unit/test_pubsub_bus.py`: `test_publish_triggers_all_subscribers_concurrently` — subscribe 3 callbacks with `asyncio.Event` flags, publish, assert all 3 flags set
- [x] T045 [P] [US4] Write test in `tests/unit/test_pubsub_bus.py`: `test_subscriber_exception_does_not_propagate` — subscriber raises exception, publish still returns without error
- [x] T046 [P] [US4] Write test in `tests/unit/test_pubsub_bus.py`: `test_unsubscribe_removes_callback` — subscribe, unsubscribe, publish, assert callback NOT called
- [x] T047 [P] [US4] Write test in `tests/unit/test_pubsub_bus.py`: `test_publish_sync_works_from_non_async_context` — call `publish_sync()` from plain function, assert subscriber received payload
- [ ] T048 [US4] Create `promptchain/utils/async_agent_inbox.py`: implement `InboxMessage` dataclass and `AsyncAgentInbox` class using `asyncio.PriorityQueue` per `contracts/async-execution.md` and `data-model.md`
- [ ] T049 [US4] Add `PubSubBus` class to `promptchain/cli/communication/message_bus.py`: `subscribe()`, `unsubscribe()`, `publish()` (async, `asyncio.gather` fan-out, per-subscriber error isolation), `publish_sync()` wrapper per FR-017 / `contracts/async-execution.md`
- [ ] T050 [US4] Refactor LLM calls in `promptchain/utils/agentic_step_processor.py` and `promptchain/utils/enhanced_agentic_step_processor.py`: replace any `litellm.completion()` calls in async contexts with `await litellm.acompletion()` per FR-015
- [ ] T051 [US4] Add optional `inbox: Optional[AsyncAgentInbox] = None` parameter to `EnhancedAgenticStepProcessor.__init__()` in `promptchain/utils/enhanced_agentic_step_processor.py`; integrate inbox message polling alongside interrupt queue check per FR-016
- [ ] T052 [P] Run `pytest tests/integration/test_006_bug_fixes.py tests/integration/test_006_steering_flow.py tests/unit/ -v` and confirm all tests green; record results in `specs/006-promptchain-improvements/test-results.md`
- [ ] T053 [P] Validate SC-001: exercise Gemini tools, TUI event loop, JSON parser with integration tests; confirm zero errors
- [ ] T054 [P] Validate SC-004: run token consumption baseline (truncation-only) vs distillation path; confirm ≥30% reduction documented in `specs/006-promptchain-improvements/test-results.md`
- [ ] T055 [P] Validate SC-005: time interrupt submission to LLM context inclusion; confirm ≤2 s acknowledgment in `tests/integration/test_006_steering_flow.py::test_interrupt_ack_latency`
- [ ] T056 [P] Validate SC-006: two-agent parallel benchmark; confirm <5% I/O overhead; add `tests/integration/test_006_concurrency.py::test_two_agent_overhead`
- [ ] T057 Update `CLAUDE.md` "Recent Changes" section to record 006-promptchain-improvements as complete
- [ ] T058 [P] Update `promptchain/__init__.py` exports to expose `AsyncAgentInbox`, `PubSubBus`, `JanitorAgent`, `MemoStore`, `InterruptQueue` at top-level package per library-first principle
- [ ] T059 [P] Run `black . && isort . && flake8 promptchain/` and fix any linting issues in new/modified files
- [ ] T060 Run `mypy promptchain/utils/async_agent_inbox.py promptchain/utils/janitor_agent.py promptchain/cli/communication/message_bus.py` and resolve all type errors
- [P] tasks = different files or disjoint changes, no blocking inter-dependencies
- [Story] label maps each task to its user story for traceability

## Recent Milestones
9837ff2 [MILESTONE] Dev-kid initialized
