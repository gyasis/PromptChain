# Tasks: CLI Multi-Agent Communication Architecture

**Input**: Design documents from `/specs/003-multi-agent-communication/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Constitution mandates TDD (Principle III) - tests included for each user story.

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

## Orchestration Protocol Reference

**See**: `CLAUDE.md` → "Development Orchestration Protocol" (Constitution Principle VIII)

This document follows the Distributed Execution with File Locking protocol:
- Pre-phase wave analysis before each phase
- File ownership assignments for parallel safety
- Checkpoint sync after each phase (memory-bank + git)

## Format: `[ID] [P?] [Wn?] [Story] [agent-type?] [file-ownership?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Wn]**: Wave number within phase (W1, W2, etc.)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- **[agent-type]**: Recommended subagent_type for Task tool
- **[file-ownership]**: Files this task owns exclusively
- Include exact file paths in descriptions

## Path Conventions

Based on plan.md structure:
- **Source**: `promptchain/cli/` (extend existing)
- **Tests**: `tests/cli/` (extend existing)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and shared components

### Wave Analysis

| Wave | Tasks | Agents | Dependencies |
|------|-------|--------|--------------|
| W1 | T001, T002 | 2 | None |
| W2 | T003, T004, T005, T006 | 4 | T002 (init exists) |
| W3 | T007 | 1 | T003-T006 (models exist) |

### File Ownership Map

| File | Owner Task | Status |
|------|------------|--------|
| promptchain/cli/schema.sql | T001 | Available |
| promptchain/cli/communication/__init__.py | T002 | Available |
| promptchain/cli/models/message.py | T003 | Available |
| promptchain/cli/models/task.py | T004 | Available |
| promptchain/cli/models/blackboard.py | T005 | Available |
| promptchain/cli/models/workflow.py | T006 | Available |
| promptchain/cli/models/__init__.py | T007 | Available |

### Tasks

**Wave 1** (parallel - 2 agents):
- [ ] T001 [W1] [sql-pro] [schema.sql] Add V3 SQLite schema tables (task_queue, blackboard, workflow_state) in promptchain/cli/schema.sql
- [ ] T002 [P] [W1] [python-pro] [communication/__init__.py] Create communication module init in promptchain/cli/communication/__init__.py

**Wave 2** (parallel - 4 agents, after W1):
- [ ] T003 [P] [W2] [python-pro] [models/message.py] Add Message and MessageType dataclasses in promptchain/cli/models/message.py
- [ ] T004 [P] [W2] [python-pro] [models/task.py] Add Task dataclass in promptchain/cli/models/task.py
- [ ] T005 [P] [W2] [python-pro] [models/blackboard.py] Add BlackboardEntry dataclass in promptchain/cli/models/blackboard.py
- [ ] T006 [P] [W2] [python-pro] [models/workflow.py] Add WorkflowState and WorkflowStage in promptchain/cli/models/workflow.py

**Wave 3** (sequential - requires W2):
- [ ] T007 [W3] [python-pro] [models/__init__.py] Update promptchain/cli/models/__init__.py to export new models

**CHECKPOINT**: After T007, spawn memory-bank-keeper + git-version-manager for Phase 1 sync

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story implementation

**CRITICAL**: No user story work can begin until this phase is complete

### Wave Analysis

| Wave | Tasks | Agents | Dependencies |
|------|-------|--------|--------------|
| W1 | T008 | 1 | Phase 1 complete (schema exists) |
| W2 | T009, T010, T011, T012, T014 | 5 | T008 (migration ready) |
| W3 | T013 | 1 | T012 (ToolMetadata extended) |

### File Ownership Map

| File | Owner Task | Status |
|------|------------|--------|
| promptchain/cli/session_manager.py | T008, T009, T010, T011 | Shared (sequential W1-W2) |
| promptchain/cli/tools/registry.py | T012, T013 | Shared (sequential W2-W3) |
| tests/cli/communication/ | T014 | Available |
| tests/cli/e2e/ | T014 | Available |

### Tasks

**Wave 1** (sequential - foundation):
- [ ] T008 [W1] [python-pro] [session_manager.py] Implement V3 schema migration in promptchain/cli/session_manager.py (method: _check_and_migrate_v3)

**Wave 2** (parallel - 5 agents, after W1):
- [ ] T009 [P] [W2] [python-pro] [session_manager.py:blackboard] Add blackboard CRUD methods to SessionManager in promptchain/cli/session_manager.py
- [ ] T010 [P] [W2] [python-pro] [session_manager.py:task_queue] Add task_queue CRUD methods to SessionManager in promptchain/cli/session_manager.py
- [ ] T011 [P] [W2] [python-pro] [session_manager.py:workflow] Add workflow_state CRUD methods to SessionManager in promptchain/cli/session_manager.py
- [ ] T012 [P] [W2] [python-pro] [tools/registry.py:metadata] Extend ToolMetadata with allowed_agents and capabilities fields in promptchain/cli/tools/registry.py
- [ ] T014 [P] [W2] [test-automator] [tests/cli/] Create test directory structure: tests/cli/communication/, tests/cli/e2e/

**Wave 3** (sequential - requires T012):
- [ ] T013 [W3] [python-pro] [tools/registry.py:discover] Add discover_capabilities() method to ToolRegistry in promptchain/cli/tools/registry.py

**CHECKPOINT**: Foundation ready - spawn memory-bank-keeper + git-version-manager, then user story implementation can begin

---

## Phase 3: User Story 1 - Agent Capability Discovery (Priority: P1)

**Goal**: Enable agents to discover what capabilities other agents have for intelligent routing

**Independent Test**: Register tools with `allowed_agents` and `capabilities`, call `discover_capabilities()` to verify filtering

### Wave Analysis

| Wave | Tasks | Agents | Dependencies |
|------|-------|--------|--------------|
| W1 | T015, T016, T017 | 3 | Phase 2 complete |
| W2 | T018, T019 | 1 | Tests exist (TDD) |
| W3 | T020, T021 | 1 | T018, T019 (decorator ready) |
| W4 | T022, T023, T024, T025 | 4 | T020 (discover_capabilities ready) |
| W5 | T026 | 1 | All implementation done |

### File Ownership Map

| File | Owner Task | Status |
|------|------------|--------|
| tests/cli/tools/test_registry.py | T015, T016 | Shared |
| tests/cli/integration/test_capability_discovery.py | T017 | Available |
| promptchain/cli/tools/registry.py | T018, T019, T020 | Sequential |
| promptchain/cli/tools/library/registration.py | T022, T023, T024, T025 | Sections parallel |
| promptchain/cli/command_handler.py | T026 | Available |

### Tests for User Story 1

**Wave 1** (parallel - 3 agents, TDD first):
- [ ] T015 [P] [W1] [US1] [test-automator] [test_registry.py:metadata] Unit test for ToolMetadata extensions in tests/cli/tools/test_registry.py
- [ ] T016 [P] [W1] [US1] [test-automator] [test_registry.py:discover] Unit test for discover_capabilities() in tests/cli/tools/test_registry.py
- [ ] T017 [P] [W1] [US1] [test-automator] [test_capability_discovery.py] Integration test for capability discovery in tests/cli/integration/test_capability_discovery.py

### Implementation for User Story 1

**Wave 2** (sequential - decorator changes):
- [ ] T018 [W2] [US1] [python-pro] [tools/registry.py] Update @registry.register decorator to accept allowed_agents param in promptchain/cli/tools/registry.py
- [ ] T019 [W2] [US1] [python-pro] [tools/registry.py] Update @registry.register decorator to accept capabilities param in promptchain/cli/tools/registry.py

**Wave 3** (sequential - after decorator):
- [ ] T020 [W3] [US1] [python-pro] [tools/registry.py] Implement discover_capabilities(agent_name, capability_filter) in promptchain/cli/tools/registry.py
- [ ] T021 [W3] [US1] [test-automator] [test_registry.py] Add backward compatibility test - tools without allowed_agents work for all agents

**Wave 4** (parallel - 4 agents, tool tagging):
- [ ] T022 [P] [W4] [US1] [python-pro] [registration.py:filesystem] Tag filesystem tools with capabilities in promptchain/cli/tools/library/registration.py
- [ ] T023 [P] [W4] [US1] [python-pro] [registration.py:session] Tag session tools with capabilities in promptchain/cli/tools/library/registration.py
- [ ] T024 [P] [W4] [US1] [python-pro] [registration.py:shell] Tag shell tools with capabilities in promptchain/cli/tools/library/registration.py
- [ ] T025 [P] [W4] [US1] [python-pro] [registration.py:context] Tag context tools with capabilities in promptchain/cli/tools/library/registration.py

**Wave 5** (sequential - CLI integration):
- [ ] T026 [W5] [US1] [python-pro] [command_handler.py] Add /capabilities CLI command in promptchain/cli/command_handler.py

**CHECKPOINT**: Capability discovery fully functional - spawn memory-bank-keeper + git-version-manager

---

## Phase 4: User Story 2 - Task Delegation Between Agents (Priority: P1)

**Goal**: Enable supervisor agents to delegate tasks to worker agents with status tracking

**Independent Test**: Call `delegate_task()`, verify task in SQLite with status transitions

### Wave Analysis

| Wave | Tasks | Agents | Dependencies |
|------|-------|--------|--------------|
| W1 | T027, T028, T029 | 3 | Phase 2 complete |
| W2 | T030, T031, T032 | 1 | Tests exist (TDD) |
| W3 | T033, T034, T035, T036 | 2 | Core tools ready |

### File Ownership Map

| File | Owner Task | Status |
|------|------------|--------|
| tests/cli/tools/test_delegation.py | T027, T028 | Shared |
| tests/cli/integration/test_task_delegation.py | T029 | Available |
| promptchain/cli/tools/library/delegation_tools.py | T030, T031, T032, T035, T036 | Sequential |
| promptchain/cli/tools/library/__init__.py | T033 | Available |
| promptchain/cli/command_handler.py | T034 | Available |

### Tests for User Story 2

**Wave 1** (parallel - 3 agents, TDD first):
- [ ] T027 [P] [W1] [US2] [test-automator] [test_delegation.py:model] Unit test for Task model in tests/cli/tools/test_delegation.py
- [ ] T028 [P] [W1] [US2] [test-automator] [test_delegation.py:tool] Unit test for delegate_task tool in tests/cli/tools/test_delegation.py
- [ ] T029 [P] [W1] [US2] [test-automator] [test_task_delegation.py] Integration test for task status transitions in tests/cli/integration/test_task_delegation.py

### Implementation for User Story 2

**Wave 2** (sequential - core delegation):
- [ ] T030 [W2] [US2] [python-pro] [delegation_tools.py] Implement delegate_task() tool in promptchain/cli/tools/library/delegation_tools.py
- [ ] T031 [W2] [US2] [python-pro] [delegation_tools.py] Implement get_pending_tasks() helper in promptchain/cli/tools/library/delegation_tools.py
- [ ] T032 [W2] [US2] [python-pro] [delegation_tools.py] Implement update_task_status() helper in promptchain/cli/tools/library/delegation_tools.py

**Wave 3** (parallel - 2 agents, integration):
- [ ] T033 [P] [W3] [US2] [python-pro] [library/__init__.py] Register delegation tools in promptchain/cli/tools/library/__init__.py
- [ ] T034 [P] [W3] [US2] [python-pro] [command_handler.py] Add /tasks CLI command in promptchain/cli/command_handler.py
- [ ] T035 [W3] [US2] [python-pro] [delegation_tools.py] Add task validation (target_agent != source_agent) in delegation_tools.py
- [ ] T036 [W3] [US2] [python-pro] [delegation_tools.py] Add activity logger integration for task events in delegation_tools.py

**CHECKPOINT**: Task delegation fully functional - spawn memory-bank-keeper + git-version-manager

---

## Phase 5: User Story 3 - Blackboard Data Sharing (Priority: P1)

**Goal**: Enable agents to share data via blackboard pattern without direct communication

**Independent Test**: Call `write_to_blackboard()` and `read_from_blackboard()`, verify persistence

### Wave Analysis

| Wave | Tasks | Agents | Dependencies |
|------|-------|--------|--------------|
| W1 | T037, T038, T039, T040 | 4 | Phase 2 complete |
| W2 | T041, T042, T043, T044 | 1 | Tests exist (TDD) |
| W3 | T045, T046, T047 | 2 | Core tools ready |

### File Ownership Map

| File | Owner Task | Status |
|------|------------|--------|
| tests/cli/tools/test_blackboard.py | T037, T038, T039 | Shared |
| tests/cli/integration/test_blackboard.py | T040 | Available |
| promptchain/cli/tools/library/blackboard_tools.py | T041, T042, T043, T044, T047 | Sequential |
| promptchain/cli/tools/library/__init__.py | T045 | Available |
| promptchain/cli/command_handler.py | T046 | Available |

### Tests for User Story 3

**Wave 1** (parallel - 4 agents, TDD first):
- [ ] T037 [P] [W1] [US3] [test-automator] [test_blackboard.py:model] Unit test for BlackboardEntry model in tests/cli/tools/test_blackboard.py
- [ ] T038 [P] [W1] [US3] [test-automator] [test_blackboard.py:write] Unit test for write_to_blackboard tool in tests/cli/tools/test_blackboard.py
- [ ] T039 [P] [W1] [US3] [test-automator] [test_blackboard.py:read] Unit test for read_from_blackboard tool in tests/cli/tools/test_blackboard.py
- [ ] T040 [P] [W1] [US3] [test-automator] [test_blackboard.py:integration] Integration test for blackboard operations in tests/cli/integration/test_blackboard.py

### Implementation for User Story 3

**Wave 2** (sequential - core blackboard):
- [ ] T041 [W2] [US3] [python-pro] [blackboard_tools.py] Implement write_to_blackboard() tool in promptchain/cli/tools/library/blackboard_tools.py
- [ ] T042 [W2] [US3] [python-pro] [blackboard_tools.py] Implement read_from_blackboard() tool in promptchain/cli/tools/library/blackboard_tools.py
- [ ] T043 [W2] [US3] [python-pro] [blackboard_tools.py] Implement list_blackboard_keys() tool in promptchain/cli/tools/library/blackboard_tools.py
- [ ] T044 [W2] [US3] [python-pro] [blackboard_tools.py] Add upsert logic with version incrementing in blackboard_tools.py

**Wave 3** (parallel - 2 agents, integration):
- [ ] T045 [P] [W3] [US3] [python-pro] [library/__init__.py] Register blackboard tools in promptchain/cli/tools/library/__init__.py
- [ ] T046 [P] [W3] [US3] [python-pro] [command_handler.py] Add /blackboard CLI command in promptchain/cli/command_handler.py
- [ ] T047 [W3] [US3] [python-pro] [blackboard_tools.py] Add activity logger integration for blackboard events

**CHECKPOINT**: Blackboard sharing fully functional - spawn memory-bank-keeper + git-version-manager

---

## Phase 6: User Story 4 - Agent-to-Agent Messaging (Priority: P2)

**Goal**: Enable direct typed messaging between agents for real-time coordination

**Independent Test**: Register `@cli_communication_handler`, send messages, verify handler invocation

### Wave Analysis

| Wave | Tasks | Agents | Dependencies |
|------|-------|--------|--------------|
| W1 | T048, T049, T050, T051 | 4 | Phase 2 complete |
| W2 | T052, T053 | 1 | Tests exist (TDD) |
| W3 | T054, T055, T056, T057, T058 | 1 | Handlers ready |
| W4 | T059 | 1 | All implementation done |

### File Ownership Map

| File | Owner Task | Status |
|------|------------|--------|
| tests/cli/communication/test_handlers.py | T048, T049, T051 | Shared |
| tests/cli/communication/test_message_bus.py | T050 | Available |
| promptchain/cli/communication/handlers.py | T052, T053 | Sequential |
| promptchain/cli/communication/message_bus.py | T054, T055, T056, T057, T058 | Sequential |
| promptchain/cli/communication/__init__.py | T059 | Available |

### Tests for User Story 4

**Wave 1** (parallel - 4 agents, TDD first):
- [ ] T048 [P] [W1] [US4] [test-automator] [test_handlers.py:message] Unit test for Message dataclass in tests/cli/communication/test_handlers.py
- [ ] T049 [P] [W1] [US4] [test-automator] [test_handlers.py:decorator] Unit test for @cli_communication_handler decorator in tests/cli/communication/test_handlers.py
- [ ] T050 [P] [W1] [US4] [test-automator] [test_message_bus.py] Unit test for message_bus send/broadcast in tests/cli/communication/test_message_bus.py
- [ ] T051 [P] [W1] [US4] [test-automator] [test_handlers.py:filter] Integration test for handler filtering in tests/cli/communication/test_handlers.py

### Implementation for User Story 4

**Wave 2** (sequential - handlers):
- [ ] T052 [W2] [US4] [python-pro] [handlers.py] Implement @cli_communication_handler decorator in promptchain/cli/communication/handlers.py
- [ ] T053 [W2] [US4] [python-pro] [handlers.py] Add handler filtering logic (sender, receiver, type) in handlers.py

**Wave 3** (sequential - message bus):
- [ ] T054 [W3] [US4] [python-pro] [message_bus.py] Implement MessageBus class with send() in promptchain/cli/communication/message_bus.py
- [ ] T055 [W3] [US4] [python-pro] [message_bus.py] Implement broadcast() method in message_bus.py
- [ ] T056 [W3] [US4] [python-pro] [message_bus.py] Add handler registry for message routing in message_bus.py
- [ ] T057 [W3] [US4] [python-pro] [message_bus.py] Integrate activity logger for message events in message_bus.py
- [ ] T058 [W3] [US4] [python-pro] [message_bus.py] Add fail-safe error handling (log and continue) in message_bus.py

**Wave 4** (sequential - exports):
- [ ] T059 [W4] [US4] [python-pro] [communication/__init__.py] Export communication module in promptchain/cli/communication/__init__.py

**CHECKPOINT**: Messaging fully functional - spawn memory-bank-keeper + git-version-manager

---

## Phase 7: User Story 5 - Workflow State Tracking (Priority: P2)

**Goal**: Track workflow progress across agent interactions for monitoring and debugging

**Independent Test**: Create workflow, transition stages, verify state via `/workflow` command

### Wave Analysis

| Wave | Tasks | Agents | Dependencies |
|------|-------|--------|--------------|
| W1 | T060, T061, T062 | 3 | Phase 2 complete |
| W2 | T063, T064, T065, T066 | 1 | Tests exist (TDD) |
| W3 | T067, T068, T069 | 2 | Core methods ready |

### File Ownership Map

| File | Owner Task | Status |
|------|------------|--------|
| tests/cli/integration/test_workflow.py | T060, T061, T062 | Shared |
| promptchain/cli/session_manager.py | T063, T064, T065, T066, T069 | Sequential |
| promptchain/cli/command_handler.py | T067 | Available |
| promptchain/cli/tools/library/delegation_tools.py | T068 | Available (callback) |

### Tests for User Story 5

**Wave 1** (parallel - 3 agents, TDD first):
- [ ] T060 [P] [W1] [US5] [test-automator] [test_workflow.py:model] Unit test for WorkflowState model in tests/cli/integration/test_workflow.py
- [ ] T061 [P] [W1] [US5] [test-automator] [test_workflow.py:transitions] Unit test for workflow stage transitions in tests/cli/integration/test_workflow.py
- [ ] T062 [P] [W1] [US5] [test-automator] [test_workflow.py:persistence] Integration test for workflow persistence in tests/cli/integration/test_workflow.py

### Implementation for User Story 5

**Wave 2** (sequential - core workflow):
- [ ] T063 [W2] [US5] [python-pro] [session_manager.py] Implement create_workflow() in promptchain/cli/session_manager.py
- [ ] T064 [W2] [US5] [python-pro] [session_manager.py] Implement update_workflow_stage() in session_manager.py
- [ ] T065 [W2] [US5] [python-pro] [session_manager.py] Implement add_completed_task() in session_manager.py
- [ ] T066 [W2] [US5] [python-pro] [session_manager.py] Implement get_workflow_state() in session_manager.py

**Wave 3** (parallel - 2 agents, integration):
- [ ] T067 [P] [W3] [US5] [python-pro] [command_handler.py] Add /workflow CLI command in promptchain/cli/command_handler.py
- [ ] T068 [P] [W3] [US5] [python-pro] [delegation_tools.py] Integrate workflow updates with task completion callbacks
- [ ] T069 [W3] [US5] [python-pro] [session_manager.py] Add workflow state restoration on session load

**CHECKPOINT**: Workflow tracking fully functional - spawn memory-bank-keeper + git-version-manager

---

## Phase 8: User Story 6 - Help Request Protocol (Priority: P3)

**Goal**: Enable stuck agents to request help from capable agents automatically

**Independent Test**: Call `request_help()`, verify routing to capable agent

### Wave Analysis

| Wave | Tasks | Agents | Dependencies |
|------|-------|--------|--------------|
| W1 | T070, T071 | 2 | US1 + US2 complete |
| W2 | T072, T073, T074 | 1 | Tests exist (TDD) |
| W3 | T075 | 1 | Implementation done |

### File Ownership Map

| File | Owner Task | Status |
|------|------------|--------|
| tests/cli/tools/test_delegation.py | T070 | Available (append) |
| tests/cli/integration/test_help_request.py | T071 | Available |
| promptchain/cli/tools/library/delegation_tools.py | T072, T073, T074 | Sequential |
| promptchain/cli/tools/library/__init__.py | T075 | Available |

### Tests for User Story 6

**Wave 1** (parallel - 2 agents, TDD first):
- [ ] T070 [P] [W1] [US6] [test-automator] [test_delegation.py:help] Unit test for request_help tool in tests/cli/tools/test_delegation.py
- [ ] T071 [P] [W1] [US6] [test-automator] [test_help_request.py] Integration test for help routing in tests/cli/integration/test_help_request.py

### Implementation for User Story 6

**Wave 2** (sequential - help system):
- [ ] T072 [W2] [US6] [python-pro] [delegation_tools.py] Implement request_help() tool in promptchain/cli/tools/library/delegation_tools.py
- [ ] T073 [W2] [US6] [python-pro] [delegation_tools.py] Add capability matching logic for help routing in delegation_tools.py
- [ ] T074 [W2] [US6] [python-pro] [delegation_tools.py] Add broadcast fallback when no matching capability found

**Wave 3** (sequential - registration):
- [ ] T075 [W3] [US6] [python-pro] [library/__init__.py] Register request_help tool in library/__init__.py

**CHECKPOINT**: Help request fully functional - spawn memory-bank-keeper + git-version-manager

---

## Phase 9: User Story 7 - Mental Models Integration (Priority: P1) **CRITICAL**

**Goal**: Enable agents to select and apply structured reasoning frameworks during task execution

**Source**: `/docs/agent_communication/thoughtbox_mental_models_integration.md`

**Key Concept**: Mental models are **process scaffolds** that tell agents **HOW to think** about problems, not **WHAT to think**. They enable:
1. **Discovery** - Find relevant models based on task characteristics
2. **Selection** - Automatically select appropriate reasoning framework
3. **Application** - Apply model's process during task execution
4. **Switching** - Change models as task evolves

### Wave Analysis

| Wave | Tasks | Agents | Dependencies |
|------|-------|--------|--------------|
| W1 | T076-T081 | 6 | Phase 2 complete |
| W2 | T082, T083, T084, T085, T086, T087, T088 | 1 | Tests exist (TDD) |
| W3 | T089-T103 | 15 | Registry ready - **MEGA PARALLEL** |
| W4 | T104, T105, T106 | 1 | Models populated |
| W5 | T107, T108 | 1 | Selector ready |
| W6 | T109-T115 | 1 | Applicator ready |
| W7 | T116, T117, T118 | 1 | AgentChain integration |
| W8 | T119, T120 | 2 | All implementation done |

### File Ownership Map

| File | Owner Task | Status |
|------|------------|--------|
| tests/cli/tools/test_mental_models.py | T076, T077, T078, T079 | Shared |
| tests/cli/integration/test_mental_models.py | T080, T081 | Shared |
| promptchain/utils/mental_models.py | T082-T108 | Sequential (registry → models → selector → applicator) |
| promptchain/utils/agent_chain.py | T109-T118 | Sequential |
| promptchain/cli/command_handler.py | T119 | Available |
| promptchain/cli/tools/library/registration.py | T120 | Available |

### Tests for User Story 7

**Wave 1** (parallel - 6 agents, TDD first):
- [ ] T076 [P] [W1] [US7] [test-automator] [test_mental_models.py:model] Unit test for MentalModel dataclass in tests/cli/tools/test_mental_models.py
- [ ] T077 [P] [W1] [US7] [test-automator] [test_mental_models.py:registry] Unit test for MentalModelRegistry in tests/cli/tools/test_mental_models.py
- [ ] T078 [P] [W1] [US7] [test-automator] [test_mental_models.py:selector] Unit test for MentalModelSelector in tests/cli/tools/test_mental_models.py
- [ ] T079 [P] [W1] [US7] [test-automator] [test_mental_models.py:find] Unit test for find_models_for_task() in tests/cli/tools/test_mental_models.py
- [ ] T080 [P] [W1] [US7] [test-automator] [test_mental_models_int.py:select] Integration test for mental model selection in tests/cli/integration/test_mental_models.py
- [ ] T081 [P] [W1] [US7] [test-automator] [test_mental_models_int.py:apply] Integration test for model application in tests/cli/integration/test_mental_models.py

### Implementation for User Story 7

**Wave 2** (sequential - Core Registry):
- [ ] T082 [W2] [US7] [python-pro] [mental_models.py:dataclass] Create MentalModel dataclass in promptchain/utils/mental_models.py
- [ ] T083 [W2] [US7] [python-pro] [mental_models.py:tag] Create Tag dataclass in promptchain/utils/mental_models.py
- [ ] T084 [W2] [US7] [python-pro] [mental_models.py:registry] Implement MentalModelRegistry class in promptchain/utils/mental_models.py
- [ ] T085 [W2] [US7] [python-pro] [mental_models.py:get] Implement get_model() method in mental_models.py
- [ ] T086 [W2] [US7] [python-pro] [mental_models.py:list] Implement list_models() method in mental_models.py
- [ ] T087 [W2] [US7] [python-pro] [mental_models.py:tags] Implement list_tags() method in mental_models.py
- [ ] T088 [W2] [US7] [python-pro] [mental_models.py:find] Implement find_models_for_task() method in mental_models.py

**Wave 3** (MEGA PARALLEL - 15 agents for 15 models):
> **NOTE**: This is the largest parallel opportunity in the project. All 15 mental models can be implemented simultaneously as they are independent data structures.

- [ ] T089 [P] [W3] [US7] [python-pro] [mental_models.py:rubber-duck] Implement rubber-duck process prompt (debugging, communication)
- [ ] T090 [P] [W3] [US7] [python-pro] [mental_models.py:five-whys] Implement five-whys process prompt (debugging, validation)
- [ ] T091 [P] [W3] [US7] [python-pro] [mental_models.py:pre-mortem] Implement pre-mortem process prompt (risk-analysis, planning)
- [ ] T092 [P] [W3] [US7] [python-pro] [mental_models.py:assumption] Implement assumption-surfacing process prompt (validation, planning)
- [ ] T093 [P] [W3] [US7] [python-pro] [mental_models.py:steelman] Implement steelmanning process prompt (decision-making, validation)
- [ ] T094 [P] [W3] [US7] [python-pro] [mental_models.py:trade-off] Implement trade-off-matrix process prompt (decision-making, prioritization)
- [ ] T095 [P] [W3] [US7] [python-pro] [mental_models.py:fermi] Implement fermi-estimation process prompt (estimation)
- [ ] T096 [P] [W3] [US7] [python-pro] [mental_models.py:abstraction] Implement abstraction-laddering process prompt (architecture, communication)
- [ ] T097 [P] [W3] [US7] [python-pro] [mental_models.py:decomposition] Implement decomposition process prompt (planning, architecture)
- [ ] T098 [P] [W3] [US7] [python-pro] [mental_models.py:adversarial] Implement adversarial-thinking process prompt (risk-analysis, validation)
- [ ] T099 [P] [W3] [US7] [python-pro] [mental_models.py:opportunity] Implement opportunity-cost process prompt (decision-making, prioritization)
- [ ] T100 [P] [W3] [US7] [python-pro] [mental_models.py:constraint] Implement constraint-relaxation process prompt (planning, architecture)
- [ ] T101 [P] [W3] [US7] [python-pro] [mental_models.py:time-horizon] Implement time-horizon-shifting process prompt (planning, decision-making)
- [ ] T102 [P] [W3] [US7] [python-pro] [mental_models.py:impact-effort] Implement impact-effort-grid process prompt (prioritization)
- [ ] T103 [P] [W3] [US7] [python-pro] [mental_models.py:inversion] Implement inversion process prompt (risk-analysis, planning)

**Wave 4** (sequential - Model Selector):
- [ ] T104 [W4] [US7] [python-pro] [mental_models.py:selector] Implement MentalModelSelector class in promptchain/utils/mental_models.py
- [ ] T105 [W4] [US7] [python-pro] [mental_models.py:select] Implement select_model() async method in mental_models.py
- [ ] T106 [W4] [US7] [python-pro] [mental_models.py:prompt] Implement _build_selection_prompt() in mental_models.py

**Wave 5** (sequential - Model Applicator):
- [ ] T107 [W5] [US7] [python-pro] [mental_models.py:applicator] Implement MentalModelApplicator class in promptchain/utils/mental_models.py
- [ ] T108 [W5] [US7] [python-pro] [mental_models.py:apply] Implement apply_model() async method in mental_models.py

**Wave 6** (sequential - AgentChain Integration):
- [ ] T109 [W6] [US7] [python-pro] [agent_chain.py:param] Add enable_mental_models parameter to AgentChain in promptchain/utils/agent_chain.py
- [ ] T110 [W6] [US7] [python-pro] [agent_chain.py:register] Add _register_mental_model_tools() method to AgentChain
- [ ] T111 [W6] [US7] [python-pro] [agent_chain.py:create] Add _create_mental_model_tools() method to AgentChain
- [ ] T112 [W6] [US7] [python-pro] [agent_chain.py:select_schema] Implement select_mental_model tool schema in agent_chain.py
- [ ] T113 [W6] [US7] [python-pro] [agent_chain.py:get_schema] Implement get_mental_model tool schema in agent_chain.py
- [ ] T114 [W6] [US7] [python-pro] [agent_chain.py:list_schema] Implement list_mental_models tool schema in agent_chain.py
- [ ] T115 [W6] [US7] [python-pro] [agent_chain.py:handle] Implement _handle_mental_model_tool() async method in agent_chain.py

**Wave 7** (sequential - Auto-Selection):
- [ ] T116 [W7] [US7] [python-pro] [agent_chain.py:auto] Add auto-select mental model logic in process_input() in agent_chain.py
- [ ] T117 [W7] [US7] [python-pro] [agent_chain.py:enhance] Enhance user input with mental model guidance when model selected
- [ ] T118 [W7] [US7] [python-pro] [agent_chain.py:context] Add mental model context to conversation history

**Wave 8** (parallel - 2 agents, CLI Integration):
- [ ] T119 [P] [W8] [US7] [python-pro] [command_handler.py] Add /mentalmodels CLI command in promptchain/cli/command_handler.py
- [ ] T120 [P] [W8] [US7] [python-pro] [registration.py] Add mental model tools to CLI tool registry in promptchain/cli/tools/library/registration.py

**CHECKPOINT**: Mental models fully functional - spawn memory-bank-keeper + git-version-manager

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Integration, documentation, and validation across all stories

### Wave Analysis

| Wave | Tasks | Agents | Dependencies |
|------|-------|--------|--------------|
| W1 | T121, T122, T123, T124, T125, T129, T130 | 7 | All US complete |
| W2 | T126, T127, T128 | 1 | All tests pass |

### File Ownership Map

| File | Owner Task | Status |
|------|------------|--------|
| tests/cli/e2e/test_multi_agent.py | T121 | Available |
| tests/cli/performance/test_communication.py | T122 | Available |
| tests/cli/performance/test_blackboard.py | T123 | Available |
| tests/cli/ (validation) | T124 | Read-only |
| promptchain/cli/__init__.py | T125 | Available |
| specs/003-multi-agent-communication/quickstart.md | T126 | Read-only |
| specs/003-multi-agent-communication/checklists/ | T127 | Available |
| docs/success-criteria-validation.md | T128 | Available |
| tests/cli/e2e/test_mental_models_e2e.py | T129 | Available |
| tests/cli/performance/test_mental_models.py | T130 | Available |

### Tasks

**Wave 1** (parallel - 7 agents, testing/exports):
- [ ] T121 [P] [W1] [test-automator] [test_multi_agent.py] E2E test for complete multi-agent workflow in tests/cli/e2e/test_multi_agent.py
- [ ] T122 [P] [W1] [test-automator] [test_communication.py] Performance test for communication < 10ms in tests/cli/performance/test_communication.py
- [ ] T123 [P] [W1] [test-automator] [test_blackboard.py] Performance test for blackboard < 5ms in tests/cli/performance/test_blackboard.py
- [ ] T124 [W1] [test-automator] [tests/cli/] Backward compatibility test - existing CLI tests pass in tests/cli/
- [ ] T125 [P] [W1] [python-pro] [cli/__init__.py] Update promptchain/cli/__init__.py exports for new modules
- [ ] T129 [P] [W1] [test-automator] [test_mental_models_e2e.py] E2E test for mental models integration with multi-agent workflow in tests/cli/e2e/test_mental_models_e2e.py
- [ ] T130 [P] [W1] [test-automator] [test_mental_models.py] Performance test for mental model selection < 100ms in tests/cli/performance/test_mental_models.py

**Wave 2** (sequential - validation and documentation):
- [ ] T126 [W2] [general-purpose] Run quickstart.md validation scenarios manually
- [ ] T127 [W2] [general-purpose] [checklists/requirements.md] Update checklists/requirements.md with completion status
- [ ] T128 [W2] [general-purpose] Validate all 15 success criteria (SC-001 to SC-015)

**FINAL CHECKPOINT**: Project complete - spawn memory-bank-keeper + git-version-manager for release

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 - BLOCKS all user stories
- **US1 Capability Discovery (Phase 3)**: Depends on Phase 2
- **US2 Task Delegation (Phase 4)**: Depends on Phase 2 (can run parallel to US1)
- **US3 Blackboard (Phase 5)**: Depends on Phase 2 (can run parallel to US1, US2)
- **US4 Messaging (Phase 6)**: Depends on Phase 2 (can run parallel to US1-3)
- **US5 Workflow (Phase 7)**: Depends on Phase 2, integrates with US2
- **US6 Help Request (Phase 8)**: Depends on US1 (capability discovery) and US2 (delegation)
- **US7 Mental Models (Phase 9)**: Can run parallel to US1-US6 (integrates with AgentChain)
- **Polish (Phase 10)**: Depends on all user stories

### User Story Dependencies

```
Phase 2 (Foundational)
    ├─> US1 (Capability Discovery) ─┬─> US6 (Help Request)
    ├─> US2 (Task Delegation) ──────┤
    ├─> US3 (Blackboard) ───────────┤
    ├─> US4 (Messaging) ────────────┤
    ├─> US5 (Workflow) → integrates with US2
    │                               │
    └─> US7 (Mental Models) ────────┤  **CRITICAL for agent behavior**
                                    │
                                    v
                             Phase 10 (Polish)
```

### Parallel Opportunities

**Within Phase 1** (all [P] tasks):
- T002, T003, T004, T005, T006 can run in parallel

**Within Phase 2**:
- T012, T014 can run in parallel

**User Stories can run in parallel after Phase 2**:
- US1, US2, US3, US4 can all start simultaneously
- US5 can start after US2 basics complete
- US6 depends on US1 + US2

**Within each User Story**:
- All tests ([P]) can run in parallel
- All tool tagging tasks ([P]) can run in parallel

---

## Parallel Example: Phase 1 Setup

```bash
# Launch all model creation tasks together:
Task: T003 "Add Message and MessageType dataclasses"
Task: T004 "Add Task dataclass"
Task: T005 "Add BlackboardEntry dataclass"
Task: T006 "Add WorkflowState and WorkflowStage"
```

## Parallel Example: User Story 1 Tests

```bash
# Launch all US1 tests together:
Task: T015 "Unit test for ToolMetadata extensions"
Task: T016 "Unit test for discover_capabilities()"
Task: T017 "Integration test for capability discovery"
```

## Parallel Example: Tool Tagging

```bash
# Launch all tagging tasks together:
Task: T022 "Tag filesystem tools with capabilities"
Task: T023 "Tag session tools with capabilities"
Task: T024 "Tag shell tools with capabilities"
Task: T025 "Tag context tools with capabilities"
```

---

## Implementation Strategy

### MVP First (User Stories 1-3 Only)

1. Complete Phase 1: Setup (T001-T007)
2. Complete Phase 2: Foundational (T008-T014)
3. Complete Phase 3: US1 Capability Discovery (T015-T026)
4. **STOP and VALIDATE**: Test capability discovery independently
5. Complete Phase 4: US2 Task Delegation (T027-T036)
6. **STOP and VALIDATE**: Test task delegation independently
7. Complete Phase 5: US3 Blackboard (T037-T047)
8. **STOP and VALIDATE**: Test blackboard independently
9. **MVP COMPLETE**: Core multi-agent patterns working

### Full Implementation

10. Complete Phase 6: US4 Messaging (T048-T059)
11. Complete Phase 7: US5 Workflow (T060-T069)
12. Complete Phase 8: US6 Help Request (T070-T075)
13. Complete Phase 9: Polish (T076-T083)

### Parallel Team Strategy

With 3 developers after Phase 2:
- Developer A: US1 (Capability Discovery) → US6 (Help Request)
- Developer B: US2 (Task Delegation) → US5 (Workflow)
- Developer C: US3 (Blackboard) → US4 (Messaging)

---

## Summary

| Phase | Tasks | Waves | Max Parallel Agents | Story | Priority |
|-------|-------|-------|---------------------|-------|----------|
| 1. Setup | T001-T007 | 3 | 4 | - | - |
| 2. Foundational | T008-T014 | 3 | 5 | - | - |
| 3. US1 Capability | T015-T026 | 5 | 4 | US1 | P1 |
| 4. US2 Delegation | T027-T036 | 3 | 3 | US2 | P1 |
| 5. US3 Blackboard | T037-T047 | 3 | 4 | US3 | P1 |
| 6. US4 Messaging | T048-T059 | 4 | 4 | US4 | P2 |
| 7. US5 Workflow | T060-T069 | 3 | 3 | US5 | P2 |
| 8. US6 Help | T070-T075 | 3 | 2 | US6 | P3 |
| 9. US7 Mental Models | T076-T120 | 8 | **15** (mega-parallel) | US7 | **P1 CRITICAL** |
| 10. Polish | T121-T130 | 2 | 7 | - | - |

**Total**: 130 tasks across 37 waves
**Per User Story**: US1=12, US2=10, US3=11, US4=12, US5=10, US6=6, US7=45
**Parallel Opportunities**: 55+ tasks can run in parallel
**Peak Parallelization**: Wave 3 of Phase 9 (15 agents for 15 mental models)
**MVP Scope**: Phases 1-5 + Phase 9 (Mental Models) - delivers core functionality with intelligent reasoning

### Mental Models Critical Integration

The mental models integration (US7) is **CRITICAL** because it determines **how agents think and reason**:

| Mental Model | Tags | Use Case |
|--------------|------|----------|
| rubber-duck | debugging, communication | Explaining problems step-by-step |
| five-whys | debugging, validation | Root cause analysis |
| pre-mortem | risk-analysis, planning | Identifying failure points |
| assumption-surfacing | validation, planning | Validating hidden assumptions |
| steelmanning | decision-making, validation | Balanced decision making |
| trade-off-matrix | decision-making, prioritization | Evaluating competing concerns |
| fermi-estimation | estimation | Order-of-magnitude estimates |
| abstraction-laddering | architecture, communication | Finding right abstraction level |
| decomposition | planning, architecture | Breaking down complexity |
| adversarial-thinking | risk-analysis, validation | Security and edge case analysis |
| opportunity-cost | decision-making, prioritization | Resource allocation |
| constraint-relaxation | planning, architecture | Exploring solution space |
| time-horizon-shifting | planning, decision-making | Multi-scale decision evaluation |
| impact-effort-grid | prioritization | Task prioritization |
| inversion | risk-analysis, planning | Avoiding paths to failure |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- Each user story independently completable and testable
- TDD enforced: Tests written first, must fail before implementation
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
