# Feature Specification: MLflow Observability Package

**Feature Branch**: `005-mlflow-observability`
**Created**: 2026-01-06
**Status**: Draft
**Input**: User description: "MLflow observability package with decorator-based tracking for LLM calls, task operations, and agent routing"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Enable Basic MLflow Tracking (Priority: P1)

As a PromptChain developer, I want to enable MLflow tracking for my CLI sessions so that I can monitor LLM API usage, costs, and performance metrics in the MLflow UI without modifying any existing code.

**Why this priority**: This is the core value proposition - enabling observability with zero code changes to existing applications. Without this, the entire feature has no value.

**Independent Test**: Can be fully tested by setting `PROMPTCHAIN_MLFLOW_ENABLED=true`, running a CLI session with LLM calls, and verifying metrics appear in MLflow UI. Delivers immediate value by providing visibility into LLM usage.

**Acceptance Scenarios**:

1. **Given** MLflow server is running at localhost:5000, **When** I set `PROMPTCHAIN_MLFLOW_ENABLED=true` and run a PromptChain CLI session with LLM calls, **Then** I see a new experiment run in MLflow UI with model name, token counts, and execution time
2. **Given** MLflow tracking is enabled, **When** I execute multiple LLM calls in sequence, **Then** all calls are logged as nested runs under the session parent run with accurate timestamps
3. **Given** MLflow server is unavailable, **When** I run a CLI session with tracking enabled, **Then** the CLI continues to work normally and logs a warning about MLflow connectivity

---

### User Story 2 - Track Task Operations (Priority: P2)

As a PromptChain developer using the task management system, I want to automatically track task creation, updates, and state changes in MLflow so that I can analyze task completion patterns and identify bottlenecks in my agent workflows.

**Why this priority**: Task tracking provides operational insights for multi-step agent workflows. This is the next most valuable metric after LLM calls because it shows agent orchestration efficiency.

**Independent Test**: Can be fully tested by running CLI sessions that create and complete tasks, then verifying task operations appear as separate runs in MLflow with operation types, durations, and success/failure status. Delivers value for workflow optimization.

**Acceptance Scenarios**:

1. **Given** MLflow tracking is enabled and a CLI session is active, **When** a task list is created with 3 tasks, **Then** MLflow logs a CREATE operation with task count and objective metadata
2. **Given** a task list exists with pending tasks, **When** tasks transition from pending → in_progress → completed, **Then** each state change is logged with operation type STATE_CHANGE and transition duration
3. **Given** multiple tasks execute in parallel, **When** tasks complete at different times, **Then** MLflow captures accurate per-task execution time and success/failure status

---

### User Story 3 - Monitor Agent Routing Decisions (Priority: P3)

As a PromptChain developer using multi-agent systems, I want to track which agents are selected by the router and why, so that I can optimize my agent selection logic and understand agent utilization patterns.

**Why this priority**: Agent routing insights help optimize multi-agent systems. This is valuable but less critical than basic LLM tracking and task monitoring because it applies only to multi-agent workflows.

**Independent Test**: Can be fully tested by running multi-agent CLI sessions with different routing strategies, then verifying routing decisions appear in MLflow with selected agent, routing strategy, and decision confidence. Delivers value for agent orchestration tuning.

**Acceptance Scenarios**:

1. **Given** a multi-agent system with router mode enabled, **When** the router selects an agent for a user query, **Then** MLflow logs the routing decision with selected agent name, routing strategy used, and confidence score (if available)
2. **Given** routing decisions are being tracked, **When** I review MLflow runs for a session, **Then** I can see the sequence of agent selections and identify which agents handle which types of queries
3. **Given** a routing decision fails or times out, **When** the system falls back to a default agent, **Then** MLflow logs the fallback event with the reason for failure

---

### User Story 4 - Disable Tracking for Production (Priority: P1)

As a PromptChain developer deploying to production, I want to completely disable MLflow tracking with zero performance overhead so that I can avoid tracking costs and latency in production environments.

**Why this priority**: Production safety is critical. This ensures the observability feature doesn't become a liability in production. Ranked P1 because it's a hard requirement for production adoption.

**Independent Test**: Can be fully tested by running performance benchmarks with tracking disabled vs. a baseline version without the observability package, verifying <0.1% overhead. Delivers value by making the feature safe for all environments.

**Acceptance Scenarios**:

1. **Given** `PROMPTCHAIN_MLFLOW_ENABLED` is not set or set to false, **When** I run a CLI session with LLM calls, **Then** MLflow decorators return the original function unchanged with zero wrapper overhead
2. **Given** MLflow tracking is disabled, **When** I benchmark 1 million function calls, **Then** performance overhead is <0.1% compared to a version without decorators
3. **Given** MLflow package is not installed, **When** I run a CLI session, **Then** the application works normally without import errors or warnings

---

### User Story 5 - Easy Package Removal (Priority: P2)

As a PromptChain maintainer, I want to remove the entire observability package in 3 simple steps (delete decorator imports, delete package directory, optional setup.py update) so that I can quickly remove the feature if it becomes problematic or unused.

**Why this priority**: Maintainability is important for long-term project health. This ensures the observability feature doesn't become technical debt if requirements change.

**Independent Test**: Can be fully tested by following the 3-step removal process, then verifying all tests pass and the application runs normally. Delivers value by providing an escape hatch.

**Acceptance Scenarios**:

1. **Given** the observability package is installed, **When** I delete all `@track_*` decorator imports (34 lines across 15 files), delete `promptchain/observability/` directory, and optionally remove from setup.py, **Then** all tests pass and the CLI works normally
2. **Given** the package has been removed, **When** I run the full test suite, **Then** no import errors occur and no tests reference removed observability code
3. **Given** I want to re-enable observability later, **When** I restore the package directory and decorator imports, **Then** tracking resumes working with no additional configuration

---

### Edge Cases

- What happens when MLflow server is running but becomes unavailable mid-session? → Application continues normally, logs warning, and buffers metrics in background queue until connection is restored or queue is full
- What happens when background queue fills up (too many metrics, slow MLflow server)? → Queue drops oldest entries and logs warning, preventing memory overflow
- What happens when MLflow is enabled but server URL is invalid? → Application starts normally, logs error on first tracking attempt, and disables tracking for remainder of session
- What happens when decorators are applied to functions with nested MLflow runs (session → LLM call)? → ContextVars ensure nested runs are correctly tracked under parent run, avoiding MLflow's thread-based run stack issues in async TUI
- What happens when a tracked function raises an exception? → Exception is logged to MLflow as run failure with stack trace, then re-raised to preserve original behavior
- What happens when user switches between multiple CLI sessions rapidly? → Each session gets its own MLflow run context via ContextVars, preventing run ID conflicts
- What happens if user tries to track a very long-running function (hours)? → MLflow run remains active, heartbeat mechanism prevents timeout, and final metrics are logged on completion
- What happens when tracking is enabled but no `MLFLOW_TRACKING_URI` is set? → System uses default `http://localhost:5000` and logs info message about using default server

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a configuration mechanism to enable/disable observability via environment variable `PROMPTCHAIN_MLFLOW_ENABLED` with default value `false`
- **FR-002**: System MUST use import-time check (not runtime) to determine if tracking is enabled, ensuring zero overhead when disabled
- **FR-003**: System MUST provide decorators (`@track_llm_call`, `@track_task`, `@track_routing`, `@track_session`) that can be applied to existing functions without modifying function logic
- **FR-004**: System MUST track LLM calls with model name, execution time, token counts (prompt/completion/total), temperature, max_tokens, and other parameters
- **FR-005**: System MUST track task operations with operation type (CREATE, UPDATE, STATE_CHANGE), execution time, success/failure status, and task metadata
- **FR-006**: System MUST track agent routing decisions with selected agent name, routing strategy, confidence score (if available), and execution time
- **FR-007**: System MUST support nested MLflow runs (session → LLM call → tool call) using ContextVars for async-safe run tracking
- **FR-008**: System MUST process MLflow API calls in background thread/queue to prevent blocking the TUI (target: <5ms overhead per tracked operation)
- **FR-009**: System MUST gracefully degrade when MLflow server is unavailable - log warning and continue execution
- **FR-010**: System MUST gracefully degrade when MLflow package is not installed - decorators become ghost decorators (return original function)
- **FR-011**: System MUST support configuration via environment variables (`PROMPTCHAIN_MLFLOW_ENABLED`, `MLFLOW_TRACKING_URI`, `PROMPTCHAIN_MLFLOW_EXPERIMENT`, `PROMPTCHAIN_MLFLOW_BACKGROUND`)
- **FR-012**: System MUST support optional configuration via `.promptchain.yml` file for persistent settings
- **FR-013**: System MUST provide `init_mlflow()` and `shutdown_mlflow()` functions for session lifecycle management
- **FR-014**: System MUST ensure package can be completely removed by deleting decorator imports (34 lines across 15 files) and `promptchain/observability/` directory
- **FR-015**: System MUST automatically extract function parameters for MLflow tags using smart argument extraction (inspect function signatures)
- **FR-016**: System MUST handle exceptions in tracked functions by logging to MLflow as run failure, then re-raising exception
- **FR-017**: System MUST provide performance impact <0.1% when tracking is disabled (verified via 1M iteration benchmark)
- **FR-018**: System MUST provide performance impact <5ms per operation when tracking is enabled with background queue

### Key Entities *(include if feature involves data)*

- **MLflow Run**: Represents a single execution of a tracked operation (session, LLM call, task operation, routing decision). Contains metrics (execution_time, tokens), parameters (model, temperature), and tags (operation_type, agent_name).

- **MLflow Experiment**: Represents a collection of runs grouped by purpose. Default experiment is `promptchain-cli`, but can be configured via `PROMPTCHAIN_MLFLOW_EXPERIMENT`.

- **Tracking Context**: Represents the current MLflow run context stored in ContextVars. Supports nested runs (session → LLM call → tool call) in async environments.

- **Background Queue**: Represents a thread-safe queue for non-blocking MLflow API calls. Processes metrics in background to prevent TUI blocking.

- **Ghost Decorator**: Represents a decorator that returns the original function unchanged when tracking is disabled. Ensures zero overhead by checking `PROMPTCHAIN_MLFLOW_ENABLED` at import time.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Developers can enable MLflow tracking by setting one environment variable (`PROMPTCHAIN_MLFLOW_ENABLED=true`) and see metrics in MLflow UI within 5 seconds of starting a session
- **SC-002**: System exhibits <0.1% performance overhead when tracking is disabled, verified by running 1 million function call iterations and comparing execution time to baseline
- **SC-003**: System exhibits <5ms overhead per tracked operation when tracking is enabled with background queue, verified by benchmarking 10,000 LLM calls
- **SC-004**: CLI application continues to function normally when MLflow server is unavailable, with less than 2 seconds total delay for logging warnings
- **SC-005**: Developers can completely remove the observability package in under 5 minutes by following the 3-step removal process (delete imports, delete directory, optional setup.py update)
- **SC-006**: All LLM calls (5 core functions) are automatically tracked with accurate token counts and model names without modifying function implementations
- **SC-007**: All task operations (8 core functions) are automatically tracked with operation types and success/failure status without modifying function implementations
- **SC-008**: All agent routing decisions (6 core functions) are automatically tracked with selected agent and routing strategy without modifying function implementations
- **SC-009**: Nested MLflow runs (session → LLM → tool) are correctly hierarchical in MLflow UI with accurate parent-child relationships
- **SC-010**: Background queue processes at least 100 metrics per second without blocking the TUI, verified by load testing with rapid LLM calls
- **SC-011**: System successfully handles MLflow server reconnection after temporary unavailability, with buffered metrics flushed within 10 seconds of reconnection
- **SC-012**: Developers can configure all MLflow settings via environment variables without touching code, verified by testing all configuration scenarios
