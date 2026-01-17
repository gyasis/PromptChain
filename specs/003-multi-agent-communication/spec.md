# Feature Specification: CLI Multi-Agent Communication Architecture

**Feature Branch**: `003-multi-agent-communication`
**Created**: 2025-11-27
**Status**: Draft
**Input**: PRD document at `docs/agent_communication/PRD_003_multi_agent_communication.md`
**Pattern Coverage**: See `/docs/agent_communication/14_agentic_patterns_gap_analysis.md`

## Pattern Coverage Summary

This spec targets **8/14 production-grade agentic patterns (57% coverage)**:

| Pattern | Status | User Story |
|---------|--------|------------|
| 1. Parallel Tool Processing | ✅ Already exists | - |
| 6. Competitive Agent Ensembles | ✅ Enhanced | US4 |
| 7. Hierarchical Agent Teams | ✅ Full support | US1, US2 |
| 8. Blackboard Collaboration | ✅ New | US3 |
| 10. Redundant Execution | ✅ Enabled | US4 |
| 11. Agent Assembly Line | ✅ Already exists | - |
| 12. Parallel Context Pre-processing | ✅ Enabled | US4 |
| 3. Parallel Evaluation | ✅ Enabled | US4 |

**Not in scope** (require additional infrastructure beyond communication):
- Branching Thoughts, Parallel Query Expansion, Sharded Retrieval, Multi-Hop Retrieval, Hybrid Search Fusion, Speculative Execution

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Agent Capability Discovery (Priority: P1)

As a developer using multi-agent workflows, I want agents to discover what capabilities other agents have so that work can be intelligently routed to the most qualified agent.

**Why this priority**: Capability discovery is foundational - without knowing what agents can do, no intelligent routing, delegation, or collaboration is possible. This enables the 57% pattern coverage target.

**Independent Test**: Can be fully tested by registering tools with `allowed_agents` and `capabilities` parameters, then calling `discover_capabilities(agent_name)` to verify correct tool filtering.

**Acceptance Scenarios**:

1. **Given** a tool registered with `allowed_agents=["DataAnalyst"]`, **When** `discover_capabilities("DataAnalyst")` is called, **Then** that tool is included in the returned list
2. **Given** a tool registered with `allowed_agents=["Writer"]`, **When** `discover_capabilities("DataAnalyst")` is called, **Then** that tool is NOT included in the returned list
3. **Given** a tool registered without `allowed_agents` parameter, **When** `discover_capabilities()` is called for any agent, **Then** that tool is available to all agents (backward compatible)
4. **Given** a tool with `capabilities=["data_processing", "statistics"]`, **When** querying capabilities by semantic tag, **Then** the tool is discoverable by capability name

---

### User Story 2 - Task Delegation Between Agents (Priority: P1)

As a supervisor agent, I want to delegate tasks to specialized worker agents so that complex workflows can be decomposed and executed by the most appropriate agent.

**Why this priority**: Task delegation enables hierarchical agent teams and is critical for complex multi-step workflows. This directly enables 4 of the 8 target patterns.

**Independent Test**: Can be fully tested by calling `delegate_task()` tool and verifying task appears in SQLite task_queue with correct status tracking through pending -> in_progress -> completed states.

**Acceptance Scenarios**:

1. **Given** a running session with multiple agents, **When** Agent A calls `delegate_task(description="Analyze data", target_agent="DataAnalyst")`, **Then** a new task record is created in task_queue with status "pending"
2. **Given** a pending delegated task, **When** the target agent picks up the task, **Then** status changes to "in_progress"
3. **Given** an in_progress task, **When** the target agent completes work, **Then** status changes to "completed" with `completed_at` timestamp
4. **Given** a delegated task, **When** task queue is queried, **Then** all task metadata (source_agent, target_agent, priority, context) is retrievable

---

### User Story 3 - Blackboard Data Sharing (Priority: P1)

As an agent in a collaborative workflow, I want to write data to a shared blackboard space so that other agents can read and build upon my work without direct communication.

**Why this priority**: Blackboard pattern enables asynchronous collaboration and is essential for parallel agent execution. This is the foundation for 3 critical collaboration patterns.

**Independent Test**: Can be fully tested by calling `write_to_blackboard(key, value)` and `read_from_blackboard(key)` tools, verifying data persistence in SQLite blackboard table.

**Acceptance Scenarios**:

1. **Given** an active session, **When** Agent A calls `write_to_blackboard("analysis_result", {"score": 95})`, **Then** data is persisted in blackboard table with `written_by="Agent A"`
2. **Given** data written by Agent A, **When** Agent B calls `read_from_blackboard("analysis_result")`, **Then** Agent B receives the stored value
3. **Given** an existing blackboard entry, **When** the same key is written again, **Then** the value is updated and version is incremented
4. **Given** concurrent blackboard access, **When** two agents write simultaneously, **Then** SQLite locking prevents data corruption

---

### User Story 4 - Agent-to-Agent Messaging (Priority: P2)

As an agent, I want to send typed messages to other agents so that I can request information, send responses, or broadcast status updates.

**Why this priority**: Direct messaging enables real-time coordination but is less critical than async patterns (blackboard, delegation) which can work without it. Still needed for competitive ensembles and redundant execution patterns.

**Independent Test**: Can be fully tested by registering a `@cli_communication_handler` and sending messages via the message bus, verifying handler receives filtered messages.

**Acceptance Scenarios**:

1. **Given** Agent B has a registered handler for `type="request"`, **When** Agent A sends a message with `type="request"` to Agent B, **Then** Agent B's handler is invoked
2. **Given** a message handler filtering by `sender="AgentA"`, **When** Agent A sends a message, **Then** the handler processes it; **When** Agent C sends a message, **Then** the handler ignores it
3. **Given** any message sent, **When** activity log is checked, **Then** message details are captured for debugging
4. **Given** no registered handler, **When** a message is sent, **Then** system operates normally (backward compatible)

---

### User Story 5 - Workflow State Tracking (Priority: P2)

As a developer, I want to track workflow progress across agent interactions so that I can monitor, debug, and resume complex multi-step operations.

**Why this priority**: Workflow tracking enhances observability but workflows can execute without explicit state tracking. Most valuable for debugging and long-running operations.

**Independent Test**: Can be fully tested by creating a workflow, transitioning through stages (planning -> execution -> review -> complete), and querying state via CLI commands.

**Acceptance Scenarios**:

1. **Given** a new multi-agent workflow starts, **When** workflow state is queried, **Then** stage is "planning" with list of involved agents
2. **Given** a workflow in execution, **When** a task completes, **Then** `completed_tasks` list is updated automatically
3. **Given** a workflow, **When** `/workflow` CLI command is invoked, **Then** current stage, progress, and task status are displayed
4. **Given** a session with workflow state, **When** session is reloaded, **Then** workflow state is restored from SQLite

---

### User Story 6 - Help Request Protocol (Priority: P3)

As an agent that is stuck, I want to request help from capable agents so that I can overcome blockers without human intervention.

**Why this priority**: Help requests are an optimization for autonomous operation but agents can function without them by delegating tasks or returning errors. Nice-to-have for fully autonomous workflows.

**Independent Test**: Can be fully tested by calling `request_help(problem_description)` and verifying the system identifies and routes to a capable agent based on registered capabilities.

**Acceptance Scenarios**:

1. **Given** an agent stuck on a data processing problem, **When** `request_help("Cannot parse JSON")` is called, **Then** system routes request to agent with "data_processing" capability
2. **Given** no agent has matching capability, **When** help is requested, **Then** system returns appropriate error or broadcasts to all agents
3. **Given** a help request, **When** a helper agent responds, **Then** response is routed back to original requesting agent

---

### User Story 7 - Mental Models Integration (Priority: P1) **CRITICAL**

As an agent executing tasks, I want to discover and apply structured reasoning frameworks (mental models) so that I can think systematically about problems and produce higher quality results.

**Why this priority**: Mental models determine **HOW agents think** about problems. Without structured reasoning frameworks, agents may use inconsistent or suboptimal reasoning approaches. This is **critical** for agent behavior quality.

**Source Document**: `/docs/agent_communication/thoughtbox_mental_models_integration.md`

**Independent Test**: Can be fully tested by:
1. Initializing MentalModelRegistry with 15 models
2. Calling `find_models_for_task()` with task description to get candidate models
3. Calling `select_model()` to have LLM choose appropriate model
4. Applying model's process prompt to task execution
5. Verifying agent uses model's reasoning framework

**Acceptance Scenarios**:

1. **Given** a task description "debug why API returns 500 errors", **When** `find_models_for_task()` is called, **Then** `rubber-duck` and `five-whys` models are returned (debugging tags)
2. **Given** a task "plan a complex project", **When** `find_models_for_task()` is called, **Then** `decomposition`, `pre-mortem`, `constraint-relaxation` models are returned (planning tags)
3. **Given** candidate models, **When** `select_model()` is called, **Then** LLM selects most appropriate model based on task context
4. **Given** a selected model, **When** model is applied, **Then** agent uses model's step-by-step reasoning process
5. **Given** AgentChain with `enable_mental_models=True`, **When** processing input, **Then** system auto-selects and applies appropriate mental model
6. **Given** an agent, **When** calling `list_mental_models()` tool, **Then** all 15 models are returned with tags
7. **Given** an agent, **When** calling `get_mental_model("five-whys")`, **Then** full process prompt is returned
8. **Given** an agent, **When** calling `select_mental_model(task_description)`, **Then** appropriate model name is returned

**15 Mental Models**:

| Model | Tags | Purpose |
|-------|------|---------|
| rubber-duck | debugging, communication | Explain problems step-by-step |
| five-whys | debugging, validation | Root cause analysis |
| pre-mortem | risk-analysis, planning | Identify failure points before starting |
| assumption-surfacing | validation, planning | Validate hidden assumptions |
| steelmanning | decision-making, validation | Present strongest opposing views |
| trade-off-matrix | decision-making, prioritization | Map competing concerns |
| fermi-estimation | estimation | Order-of-magnitude estimates |
| abstraction-laddering | architecture, communication | Find right abstraction level |
| decomposition | planning, architecture | Break down complexity |
| adversarial-thinking | risk-analysis, validation | Security/edge case analysis |
| opportunity-cost | decision-making, prioritization | Resource allocation |
| constraint-relaxation | planning, architecture | Explore solution space |
| time-horizon-shifting | planning, decision-making | Multi-scale evaluation |
| impact-effort-grid | prioritization | Task prioritization |
| inversion | risk-analysis, planning | Avoid paths to failure |

---

### Edge Cases

- What happens when a delegated task's target agent doesn't exist? System must reject with clear error message.
- How does system handle blackboard key collisions from concurrent writes? SQLite UNIQUE constraint + locking handles this.
- What happens when message handler throws an exception? Error is logged, message delivery marked as failed, system continues.
- How does system behave when workflow state table doesn't exist (migration scenario)? Auto-migrate on first access.
- What happens when task queue exceeds reasonable size? Consider implementing TTL or archival in future phase.

## Requirements *(mandatory)*

### Functional Requirements

**Agent Capability Registry (FR-001 to FR-005)**:
- **FR-001**: ToolRegistry MUST support `allowed_agents` parameter to restrict tool access to specific agents
- **FR-002**: ToolRegistry MUST support `capabilities` parameter for semantic capability tagging
- **FR-003**: System MUST provide `discover_capabilities(agent_name)` method returning available tools for an agent
- **FR-004**: Existing tools without `allowed_agents` MUST default to all-agent access (backward compatibility)
- **FR-005**: All 19 existing library tools MUST be tagged with appropriate capabilities

**Task Delegation Protocol (FR-006 to FR-010)**:
- **FR-006**: System MUST provide `delegate_task` tool in CLI tool registry
- **FR-007**: System MUST provide `request_help` tool for stuck-agent assistance
- **FR-008**: Task queue MUST be persisted in session SQLite database
- **FR-009**: System MUST track task status (pending, in_progress, completed, failed)
- **FR-010**: Tasks MUST include source_agent, target_agent, priority, context, and timestamps

**Blackboard Collaboration (FR-011 to FR-015)**:
- **FR-011**: System MUST provide `write_to_blackboard(key, value)` tool
- **FR-012**: System MUST provide `read_from_blackboard(key)` tool
- **FR-013**: System MUST provide `list_blackboard_keys()` tool
- **FR-014**: Blackboard data MUST be persisted in session SQLite database
- **FR-015**: System MUST handle concurrent blackboard access with SQLite locking

**Agent Communication Bus (FR-016 to FR-020)**:
- **FR-016**: System MUST support `@cli_communication_handler` decorator for message handlers
- **FR-017**: Handlers MUST support filtering by sender, receiver, and message type
- **FR-018**: System MUST support message types: request, response, broadcast, delegation, status
- **FR-019**: Activity logger MUST capture all communication for debugging
- **FR-020**: Communication MUST be backward compatible - existing code works without handlers

**Workflow State Management (FR-021 to FR-025)**:
- **FR-021**: System MUST track workflow stages: planning, execution, review, complete
- **FR-022**: AgentChain callbacks MUST update workflow state on events
- **FR-023**: System MUST provide CLI commands for workflow inspection (`/workflow`)
- **FR-024**: Workflow state MUST be persisted in session SQLite database
- **FR-025**: Workflow state MUST include agents_involved, completed_tasks, current_task, context

**Mental Models Integration (FR-026 to FR-036)** **CRITICAL**:
- **FR-026**: System MUST implement MentalModelRegistry with 15 pre-defined mental models
- **FR-027**: Each mental model MUST include: name, title, description, tags, process_prompt, examples, pitfalls
- **FR-028**: System MUST support 9 tag categories: debugging, planning, decision-making, risk-analysis, estimation, prioritization, communication, architecture, validation
- **FR-029**: System MUST provide `find_models_for_task()` method using keyword-to-tag mapping
- **FR-030**: System MUST implement MentalModelSelector with LLM-based model selection
- **FR-031**: AgentChain MUST support `enable_mental_models=True` parameter
- **FR-032**: When enabled, agents MUST have access to tools: `select_mental_model`, `get_mental_model`, `list_mental_models`
- **FR-033**: System MUST support auto-selection of mental model in `process_input()` when enabled
- **FR-034**: Mental model selection MUST complete in < 100ms (LLM call overhead)
- **FR-035**: CLI MUST provide `/mentalmodels` command for model discovery
- **FR-036**: Mental model tools MUST be registered in CLI tool registry

**Non-Functional Requirements (FR-037 to FR-041)**:
- **FR-037**: All existing CLI commands MUST continue to work unchanged
- **FR-038**: Existing sessions MUST load successfully with auto-migration of new tables
- **FR-039**: Communication overhead MUST be < 10ms per message
- **FR-040**: Blackboard read/write MUST complete in < 5ms for typical operations
- **FR-041**: System MUST maintain 90% token optimization achieved in Phase 11

### Key Entities

- **Task**: Represents a delegated unit of work with task_id, description, source_agent, target_agent, priority (low/medium/high), status (pending/in_progress/completed/failed), context (JSON), created_at, completed_at
- **BlackboardEntry**: Key-value storage entry with key, value (JSON serializable), written_by (agent name), written_at (timestamp), version (integer for optimistic locking)
- **WorkflowState**: Tracks multi-agent workflow with workflow_id, stage (planning/execution/review/complete), agents_involved (list), completed_tasks (list), current_task, context (JSON), started_at, updated_at
- **Message**: Communication unit with sender, receiver, type (request/response/broadcast/delegation/status), payload (dict), timestamp
- **ToolMetadata**: Extended tool registration with name, category, description, parameters, allowed_agents (list), capabilities (semantic tags)
- **MentalModel**: Reasoning framework with name, title, description, tags (list of categories), process_prompt (detailed reasoning steps), examples (list), pitfalls (list) - used to guide agent reasoning
- **Tag**: Mental model category with name and description - 9 categories total (debugging, planning, decision-making, risk-analysis, estimation, prioritization, communication, architecture, validation)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Pattern coverage increases from 14% (2/14) to 57% (8/14) of production-grade agentic patterns
- **SC-002**: All 6 newly fully supported patterns pass integration tests: Blackboard Collaboration, Hierarchical Teams, Competitive Ensembles, Parallel Evaluation, Redundant Execution, Context Pre-processing
- **SC-003**: 100% backward compatibility - all existing CLI tests pass without modification
- **SC-004**: Communication overhead measured at < 10ms per message (p95)
- **SC-005**: Blackboard operations complete in < 5ms for entries under 1KB (p95)
- **SC-006**: No regression in existing CLI response time (< 5% increase)
- **SC-007**: All 19 existing library tools tagged with capabilities and agent restrictions where appropriate
- **SC-008**: SQLite schema migrations run successfully on existing sessions
- **SC-009**: End-to-end multi-agent workflow completes successfully: capability discovery -> task delegation -> blackboard collaboration -> workflow completion
- **SC-010**: Activity log captures 100% of inter-agent communications for debugging

**Mental Models Success Criteria (SC-011 to SC-015)** **CRITICAL**:
- **SC-011**: All 15 mental models implemented with complete process prompts, examples, and pitfalls
- **SC-012**: `find_models_for_task()` returns relevant models for all 9 tag categories
- **SC-013**: AgentChain with `enable_mental_models=True` auto-selects appropriate model 80%+ of time
- **SC-014**: Mental model selection (LLM call) completes in < 100ms (p95)
- **SC-015**: Agent reasoning quality measurably improves when mental models are applied (manual verification of reasoning structure)
