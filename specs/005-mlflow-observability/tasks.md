# Tasks: MLflow Observability Package

**Input**: Design documents from `/specs/005-mlflow-observability/`
**Prerequisites**: plan.md (at `/home/gyasis/.claude/plans/optimized-wishing-donut.md`), spec.md

**Organization**: Tasks follow the 9-wave execution strategy from the implementation plan, with parallel waves for maximum efficiency. Each wave maps to specific user stories for independent testing and delivery.

## Format: `[ID] [P?] [W#] [Story] [Agent] [Files] Description`

- **[P]**: Can run in parallel within wave (different files, no dependencies)
- **[W#]**: Wave number (W1-W9) for parallel execution grouping
- **[Story]**: User story reference (US1-US5, FOUNDATION for shared infrastructure)
- **[Agent]**: Recommended agent type (python-pro, test-automator, content-marketer)
- **[Files]**: File ownership for conflict prevention

## Wave Strategy

- **Waves 1-3**: Foundation (blocking prerequisite for all user stories)
- **Waves 4-6**: User story implementations (can proceed in parallel after foundation)
- **Waves 7-9**: Integration, testing, documentation (sequential finalization)

---

## Phase 1: Core Infrastructure (Wave 1)

**Purpose**: Create the observability package foundation with ghost decorator pattern and MLflow adapter

**⚠️ CRITICAL**: These files form the zero-overhead foundation that enables US1 (basic tracking) and US4 (disable for production)

**Wave 1** - All tasks can run in parallel (4 agents, different files):

- [x] T001 [P] [W1] [FOUNDATION] [python-pro] [promptchain/observability/config.py] Create config.py with environment variable handling (PROMPTCHAIN_MLFLOW_ENABLED, MLFLOW_TRACKING_URI, PROMPTCHAIN_MLFLOW_EXPERIMENT, PROMPTCHAIN_MLFLOW_BACKGROUND) - 60 lines ✅ Created 117 lines
- [x] T002 [P] [W1] [FOUNDATION] [python-pro] [promptchain/observability/ghost.py] Create ghost.py with import-time check pattern for zero-overhead ghost decorators - 80 lines ✅ Created 109 lines
- [x] T003 [P] [W1] [FOUNDATION] [python-pro] [promptchain/observability/mlflow_adapter.py] Create mlflow_adapter.py with MLflow API wrapper, error handling, and graceful degradation - 150 lines ✅ Created 367 lines
- [x] T004 [P] [W1] [FOUNDATION] [python-pro] [promptchain/observability/__init__.py] Create __init__.py with public API exports and graceful fallback - 30 lines ✅ Created 30 lines

**Checkpoint**: Core infrastructure ready - config, ghost pattern, MLflow adapter available

---

## Phase 2: Tracking Infrastructure (Wave 2)

**Purpose**: Implement async-safe tracking context and background queue for non-blocking operations

**Dependencies**: Wave 1 complete (imports config.py, ghost.py)

**Wave 2** - All tasks can run in parallel (3 agents, different files):

- [x] T005 [P] [W2] [FOUNDATION] [python-pro] [promptchain/observability/context.py] Create context.py with ContextVars for async-safe nested MLflow run tracking - 60 lines ✅ Created 123 lines
- [x] T006 [P] [W2] [FOUNDATION] [python-pro] [promptchain/observability/queue.py] Create queue.py with background thread-safe queue for non-blocking MLflow API calls - 100 lines ✅ Created 209 lines
- [x] T007 [P] [W2] [FOUNDATION] [python-pro] [promptchain/observability/extractors.py] Create extractors.py with smart argument extraction using inspect module for automatic parameter capture - 120 lines ✅ Created 281 lines

**Checkpoint**: Tracking infrastructure ready - ContextVars, background queue, extractors available

---

## Phase 3: Decorator Implementation (Wave 3)

**Purpose**: Implement all tracking decorators using the ghost pattern and tracking infrastructure

**Dependencies**: Waves 1 and 2 complete (imports all foundation modules)

**Wave 3** - Sequential task (1 agent, depends on all previous infrastructure):

- [x] T008 [W3] [FOUNDATION] [python-pro] [promptchain/observability/decorators.py] Create decorators.py implementing @track_llm_call, @track_task, @track_routing, @track_session, @track_tool, init_mlflow(), shutdown_mlflow() with ghost pattern integration - 400 lines ✅ Created 731 lines

**Checkpoint**: Foundation complete - all decorators ready for integration. User story implementation can now begin.

---

## Phase 4: User Story 1 - Enable Basic MLflow Tracking (Priority: P1) 🎯 MVP

**Goal**: Enable LLM call tracking with model name, tokens, and execution time without code modifications

**Independent Test**: Set PROMPTCHAIN_MLFLOW_ENABLED=true, run CLI with LLM calls, verify metrics appear in MLflow UI within 5 seconds

**Dependencies**: Foundation complete (Wave 3)

**Wave 4** - All tasks can run in parallel (5 agents, different files):

### Implementation for User Story 1

- [x] T009 [P] [W4] [US1] [python-pro] [promptchain/utils/promptchaining.py] Add @track_llm_call decorator to run_model_async() at line 1833 with model_param="model_name", extract_args=["temperature", "max_tokens"] ✅ Import line 14, decorator lines 1834-1837
- [x] T010 [P] [W4] [US1] [python-pro] [promptchain/utils/dynamic_chain_builder.py] Add @track_llm_call decorator to _run_model_step_async() at line 392 ✅ Import line 13, decorator lines 393-396
- [x] T011 [P] [W4] [US1] [python-pro] [promptchain/utils/history_summarizer.py] Add @track_llm_call decorator to summarize_history() at line 149 ✅ Import line 17, decorator lines 151-154
- [x] T012 [P] [W4] [US1] [python-pro] [promptchain/integrations/lightrag/multi_hop.py] Add @track_llm_call decorator to _decompose_question() at line 246 ✅ Import line 13, decorator lines 247-250
- [x] T013 [P] [W4] [US1] [python-pro] [promptchain/integrations/lightrag/branching.py] Add @track_llm_call decorator to _judge_hypotheses() at line 379 ✅ Import line 28, decorator lines 380-383

**Checkpoint**: User Story 1 (LLM tracking) functional - can track all LLM calls with tokens, model, and timing

---

## Phase 5: User Story 2 - Track Task Operations (Priority: P2)

**Goal**: Track task creation, updates, and state changes for workflow analysis

**Independent Test**: Create task list, transition tasks through states, verify operations logged in MLflow with operation types and durations

**Dependencies**: Foundation complete (Wave 3) - can run in parallel with Phase 4

**Wave 5** - All tasks can run in parallel (3 agents, different files):

### Implementation for User Story 2

- [x] T014 [P] [W5] [US2] [python-pro] [promptchain/cli/models/task_list.py] Add @track_task decorators to 5 methods: create_list() at line 224 (CREATE), update_list() at line 246 (UPDATE), mark_task_in_progress() at line 261 (STATE_CHANGE), mark_task_completed() at line 275 (STATE_CHANGE), add_task() at line 141 (CREATE) ✅ Import line 18, decorators added at lines 143, 227, 250, 266, 281
- [x] T015 [P] [W5] [US2] [python-pro] [promptchain/cli/session_manager.py] Add @track_task decorators to 2 methods: create_task() at line 1648, update_task_status() at line 1783 ✅ Import line 17, decorators added at lines 1649, 1785
- [x] T016 [P] [W5] [US2] [python-pro] [promptchain/cli/tools/library/task_list_tool.py] Add @track_task decorator to task_list_write() at line 31 ✅ Import line 11, decorator added at line 32

**Checkpoint**: User Story 2 (task tracking) functional - can track all task operations with types and timings

---

## Phase 6: User Story 3 - Monitor Agent Routing (Priority: P3)

**Goal**: Track agent routing decisions with selected agent, strategy, and confidence

**Independent Test**: Run multi-agent session with router mode, verify routing decisions logged with agent names and strategies

**Dependencies**: Foundation complete (Wave 3) - can run in parallel with Phases 4-5

**Wave 6** - All tasks can run in parallel (3 agents, different files):

### Implementation for User Story 3

- [x] T017 [P] [W6] [US3] [python-pro] [promptchain/utils/agent_chain.py] Add @track_routing decorators to 4 methods: _route_to_agent() at line 2025, _simple_router() at line 935, _parse_decision() at line 830, run_chat_turn_async() at line 1833 ✅ Import line 20, decorators added at lines 831, 938, 2029, 1837
- [x] T018 [P] [W6] [US3] [python-pro] [promptchain/utils/strategies/single_dispatch_strategy.py] Add @track_routing decorator to execute_single_dispatch_strategy_async() at line 11 ✅ Import line 5, decorator added at line 12
- [x] T019 [P] [W6] [US3] [python-pro] [promptchain/utils/strategies/static_plan_strategy.py] Add @track_routing decorator to execute_static_plan_strategy_async() at line 10 ✅ Import line 4, decorator added at line 11

**Checkpoint**: User Story 3 (routing tracking) functional - can track all routing decisions with agents and strategies

---

## Phase 7: Session Lifecycle Integration (Wave 7)

**Purpose**: Add MLflow session initialization and shutdown hooks to CLI main entry point

**Dependencies**: Waves 4-6 complete (ensures all decorators are integrated before session management)

**Wave 7** - Sequential task (1 agent, modifies main.py after all other integrations):

- [x] T020 [W7] [US1] [python-pro] [promptchain/cli/main.py] Add @track_session decorator to main(), add init_mlflow() at start and shutdown_mlflow() in finally block for session lifecycle management ✅ Import line 19, decorator line 197, init_mlflow line 200, shutdown_mlflow in finally block line 250, also fixed observability/__init__.py to import from decorators module

**Checkpoint**: Session lifecycle integrated - MLflow initializes on CLI start, shuts down on exit

---

## Phase 8: Testing & Validation (Wave 8)

**Purpose**: Verify all user stories work independently and meet success criteria

**Dependencies**: Waves 4-7 complete (all implementation done)

**Wave 8** - All tests can run in parallel (3 agents, different test files):

### Unit Tests (Verify Technical Requirements)

- [x] T021 [P] [W8] [US4] [test-automator] [tests/test_observability_unit.py] ✅ **COMPLETE** - Created unit tests (683 lines, 20 tests): test_ghost_decorator_zero_overhead() with 1M iteration benchmark showing <0.1% overhead, test_context_vars_nested_runs() verifying async isolation, test_background_queue() verifying 100+ metrics/second throughput, test_config_parsing() with 6 env var tests, test_graceful_degradation() with 4 MLflow unavailability tests, test_exception_handling() with 2 error propagation tests - All tests pass, ruff formatting complete

### Integration Tests (Verify User Stories Work End-to-End)

- [x] T022 [P] [W8] [US1,US2,US3] [test-automator] [tests/test_observability_integration.py] ✅ **COMPLETE** - Created integration tests (743 lines, 13 tests): test_llm_call_tracking() with US1 acceptance scenarios, test_task_tracking() verifying CREATE/UPDATE/STATE_CHANGE operations, test_routing_tracking() with agent selection logging, test_nested_runs_hierarchy() with parent-child validation, test_server_reconnection() with buffered metrics, test_concurrent_tracking() with parallel execution - All tests pass, includes bug fix (added is_available() to mlflow_adapter.py), ruff formatting complete

### Performance Tests (Verify Success Criteria)

- [x] T023 [P] [W8] [US1,US4] [test-automator] [tests/test_observability_performance.py] ✅ **COMPLETE** - Created performance benchmarks (~650 lines, 7 tests): benchmark_disabled_overhead() with 1M iterations verifying SC-002 (<0.1% overhead), benchmark_enabled_overhead() with 10k iterations verifying SC-003 (<5ms per operation), benchmark_queue_throughput() verifying SC-010 (≥100 metrics/second), benchmark_startup_time() verifying SC-001 (<5 seconds), benchmark_concurrent_operations() for parallel execution stress test - All benchmarks pass success criteria, ruff formatting complete

**Checkpoint**: All tests pass - user stories verified independently, performance criteria met

---

## Phase 9: Documentation & Polish (Wave 9)

**Purpose**: Document usage, configuration, and removal procedures for users

**Dependencies**: Wave 8 complete (all functionality tested and working)

**Wave 9** - Sequential task (1 agent, writes documentation after everything is working):

- [x] T024 [W9] [US1,US4,US5] [content-marketer] [docs/observability_guide.md, README.md] ✅ **COMPLETE** - Created comprehensive observability guide (1,227 lines, 34KB): configuration instructions (environment variables + .promptchain.yml), usage examples with 4-tier tracking coverage (21 integration points), performance characteristics (benchmarked <0.1% overhead disabled, <5ms enabled), complete 3-step removal instructions, troubleshooting guide (server unavailability, connection issues, performance problems, common errors), advanced topics (custom metrics, decorator extensions, multi-environment setup), full API reference. Updated README.md with "MLflow Observability & Tracking (NEW)" section including quick start, features summary, visual hierarchy, and link to full guide

**Checkpoint**: Documentation complete - users can configure, use, and remove observability package

---

## Dependencies & Execution Order

### Wave Dependencies

```
Wave 1 (Core Infrastructure)
  ↓
Wave 2 (Tracking Infrastructure) - depends on Wave 1
  ↓
Wave 3 (Decorators) - depends on Waves 1 + 2
  ↓
┌─────────────────────────────────────┐
│ FOUNDATION COMPLETE - User stories  │
│ can now proceed in parallel         │
└─────────────────────────────────────┘
  ↓
Wave 4 (US1: LLM Tracking) ──┐
Wave 5 (US2: Task Tracking) ─┼─→ All depend on Wave 3, run in parallel
Wave 6 (US3: Routing) ───────┘
  ↓
Wave 7 (Session Lifecycle) - depends on Waves 4-6
  ↓
Wave 8 (Testing) - depends on Wave 7
  ↓
Wave 9 (Documentation) - depends on Wave 8
```

### User Story Dependencies

- **US1 (P1) - Enable Basic Tracking**: Waves 1-3 (foundation) + Wave 4 (LLM integration) + Wave 7 (session lifecycle)
- **US2 (P2) - Track Task Operations**: Waves 1-3 (foundation) + Wave 5 (task integration)
- **US3 (P3) - Monitor Agent Routing**: Waves 1-3 (foundation) + Wave 6 (routing integration)
- **US4 (P1) - Disable for Production**: Waves 1-3 (foundation) - ghost pattern is in core infrastructure
- **US5 (P2) - Easy Package Removal**: All waves - verifies clean removal after full implementation

### Parallel Opportunities

**Wave 1**: 4 tasks in parallel (T001-T004)
**Wave 2**: 3 tasks in parallel (T005-T007)
**Wave 3**: 1 sequential task (T008)
**Wave 4**: 5 tasks in parallel (T009-T013)
**Wave 5**: 3 tasks in parallel (T014-T016)
**Wave 6**: 3 tasks in parallel (T017-T019)
**Wave 7**: 1 sequential task (T020)
**Wave 8**: 3 tasks in parallel (T021-T023)
**Wave 9**: 1 sequential task (T024)

**Total Parallelization**: 18 of 24 tasks (75%) can run in parallel

---

## Implementation Strategy

### MVP First (US1 + US4 Only)

1. Complete Waves 1-3: Foundation (CRITICAL - blocks all stories)
2. Complete Wave 4: US1 LLM tracking
3. Complete Wave 7: Session lifecycle
4. **STOP and VALIDATE**: Test LLM tracking independently, verify zero overhead when disabled
5. Deploy/demo US1 + US4 as MVP

### Incremental Delivery

1. Complete Waves 1-3 → Foundation ready (US4 works: can disable with zero overhead)
2. Add Wave 4 → US1 complete (LLM tracking works)
3. Add Wave 5 → US2 complete (Task tracking works)
4. Add Wave 6 → US3 complete (Routing tracking works)
5. Add Waves 7-9 → Full integration, testing, documentation

### Parallel Team Strategy

With 3 developers after foundation complete:

1. **Team**: Completes Waves 1-3 together (foundation)
2. **Once foundation done**:
   - **Developer A**: Wave 4 (US1 - LLM tracking)
   - **Developer B**: Wave 5 (US2 - Task tracking)
   - **Developer C**: Wave 6 (US3 - Routing tracking)
3. **Merge**: Wave 7 (session lifecycle) after Waves 4-6 complete
4. **Team**: Waves 8-9 together (testing + docs)

---

## File Ownership by Wave

### Wave 1 (4 files)
- T001 → `promptchain/observability/config.py`
- T002 → `promptchain/observability/ghost.py`
- T003 → `promptchain/observability/mlflow_adapter.py`
- T004 → `promptchain/observability/__init__.py`

### Wave 2 (3 files)
- T005 → `promptchain/observability/context.py`
- T006 → `promptchain/observability/queue.py`
- T007 → `promptchain/observability/extractors.py`

### Wave 3 (1 file)
- T008 → `promptchain/observability/decorators.py`

### Wave 4 (5 files)
- T009 → `promptchain/utils/promptchaining.py`
- T010 → `promptchain/utils/dynamic_chain_builder.py`
- T011 → `promptchain/utils/history_summarizer.py`
- T012 → `promptchain/integrations/lightrag/multi_hop.py`
- T013 → `promptchain/integrations/lightrag/branching.py`

### Wave 5 (3 files)
- T014 → `promptchain/cli/models/task_list.py`
- T015 → `promptchain/cli/session_manager.py`
- T016 → `promptchain/cli/tools/library/task_list_tool.py`

### Wave 6 (3 files)
- T017 → `promptchain/utils/agent_chain.py`
- T018 → `promptchain/utils/strategies/single_dispatch_strategy.py`
- T019 → `promptchain/utils/strategies/static_plan_strategy.py`

### Wave 7 (1 file)
- T020 → `promptchain/cli/main.py`

### Wave 8 (3 files)
- T021 → `tests/test_observability_unit.py`
- T022 → `tests/test_observability_integration.py`
- T023 → `tests/test_observability_performance.py`

### Wave 9 (2 files)
- T024 → `docs/observability_guide.md`, `README.md`

**Total**: 8 new files, 15 modified files, 24 tasks

---

## Notes

- **[P] markers**: Tasks can run in parallel within their wave (different files, no conflicts)
- **[W#] markers**: Wave number for orchestration (waves run sequentially, tasks within wave run in parallel)
- **[Story] markers**: User story mapping for independent testing and validation
- **[Agent] markers**: Recommended specialized agent type for task execution
- **File ownership**: Each task has exclusive write access to assigned files during execution
- **Foundation first**: Waves 1-3 MUST complete before any user story implementation
- **Independent stories**: US1, US2, US3 can be developed and tested independently after foundation
- **Performance criteria**: All success criteria (SC-001 through SC-012) verified in Wave 8 tests
- **Ghost pattern**: US4 (disable for production) is built into foundation (Wave 2: ghost.py)
- **Easy removal**: US5 verified by clean deletion of 24 task outputs (8 new files + 34 lines of imports)
