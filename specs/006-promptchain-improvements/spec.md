# Feature Specification: PromptChain Comprehensive Improvement Roadmap

**Feature Branch**: `006-promptchain-improvements`
**Created**: 2026-02-24
**Status**: Draft
**Input**: User description: "IMPROVEMENT_ROADMAP.md"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Critical Bug Fixes: Stability & Correctness (Priority: P1)

As a developer using PromptChain, I need the core functionality to work reliably without crashes, data loss, or silent failures so that I can build dependable applications.

**Why this priority**: P0/P1 bugs actively break existing features — Gemini augmentation is completely non-functional, the TUI crashes on pattern commands, and malformed LLM JSON silently loses data. These block all downstream improvements.

**Independent Test**: Can be tested by exercising Gemini MCP tool calls, running LightRAG pattern commands from the TUI, and feeding malformed JSON to the parser. Each fix delivers immediate, verifiable stability.

**Acceptance Scenarios**:

1. **Given** an agent uses `gemini_debug`, `gemini_brainstorm`, or `ask_gemini` tools, **When** those tools are called, **Then** they execute successfully with correct parameters and return valid results.
2. **Given** a user invokes a LightRAG pattern command from the TUI, **When** the command executes, **Then** it completes without crashing due to event loop conflicts.
3. **Given** an LLM returns malformed JSON, **When** the parser processes it, **Then** a graceful fallback is applied and no data is silently lost.
4. **Given** the system shuts down while the MLflow queue is active, **When** the MLflow server is unresponsive, **Then** shutdown completes within a bounded timeout rather than hanging indefinitely.
5. **Given** the config is read repeatedly, **When** the file has not changed, **Then** the cached value is returned without disk I/O on each access.
6. **Given** a verification result is cached, **When** that result is modified post-retrieval, **Then** the cached entry remains unchanged.

---

### User Story 2 - Conversational Longevity: Context & Memory (Priority: P2)

As a developer running long multi-agent sessions, I need the system to intelligently manage context and persist learned knowledge across sessions so that agents remain coherent and useful as conversations grow.

**Why this priority**: Token truncation silently degrades agent quality over time. Context distillation and a semantic memo store directly extend how long agents reason effectively without losing important state.

**Independent Test**: Can be tested by running a session long enough to approach the context limit and verifying a coherent summary is produced rather than abrupt truncation. Memo store can be tested independently by writing and retrieving memos across separate sessions.

**Acceptance Scenarios**:

1. **Given** a session approaches 70% of the token limit, **When** new messages arrive, **Then** the system automatically generates a "Current State of Knowledge" summary replacing older messages while preserving essential context.
2. **Given** an agent learns a useful fact during a session, **When** a new session starts, **Then** relevant memos are retrieved and injected into the agent's context automatically.
3. **Given** a long-running session with many tool calls, **When** background compression runs, **Then** token usage is measurably reduced compared to truncation-only approaches.

---

### User Story 3 - Real-Time User Steering: Interrupt & Override (Priority: P3)

As a user interacting with long-running agentic tasks in the TUI, I need to send interrupt signals or pivot instructions mid-execution so that I can correct mistakes and guide agents without waiting for full completion.

**Why this priority**: The inability to interrupt agents mid-thought forces users to wait for wrong-direction execution to finish or kill the process entirely. Real-time steering dramatically improves control and development velocity.

**Independent Test**: Can be tested by triggering a long agentic task, sending an interrupt command mid-execution, and verifying the agent acknowledges the interrupt at the next thought cycle boundary.

**Acceptance Scenarios**:

1. **Given** an agent is mid-execution on a long task, **When** the user sends an interrupt signal, **Then** the agent pauses at the next thought cycle boundary and presents a summary of work done so far.
2. **Given** an agent has saved a checkpoint after a tool call, **When** the user sends a prompt override, **Then** the agent adopts the new direction from the most recent checkpoint without restarting from scratch.
3. **Given** a user sends "stop and summarize", **When** the agent receives this interrupt, **Then** it halts further steps and produces a coherent progress summary.

---

### User Story 4 - Non-Blocking Async Agent Flows (Priority: P4)

As a developer orchestrating multiple agents simultaneously, I need agents to operate concurrently without blocking each other so that multi-agent workflows are efficient and the TUI remains responsive.

**Why this priority**: Current blocking execution prevents true multi-agent concurrency and degrades TUI responsiveness. The async actor pattern unlocks parallel agent execution and flexible pub/sub pipelines.

**Independent Test**: Can be tested by running two agents simultaneously and verifying neither blocks the other's I/O operations. TUI responsiveness can be independently validated during any active long-running LLM call.

**Acceptance Scenarios**:

1. **Given** two agents are running simultaneously, **When** one agent waits on an LLM response, **Then** the other agent continues processing without being blocked.
2. **Given** an event fires that multiple agents have subscribed to, **When** that event is published, **Then** all subscribed agents are triggered concurrently rather than sequentially.
3. **Given** a long-running agent task is active, **When** the user interacts with the TUI, **Then** the TUI remains responsive with no perceptible lag.

---

### Edge Cases

- What happens when context distillation runs but the LLM call to generate the summary fails?
- How does the system handle a memo store that grows unboundedly over many sessions?
- What happens when an interrupt signal arrives at the exact same time as an agent finishes its last step?
- How does the pub/sub pipeline behave when a subscriber agent crashes mid-message?
- What happens when a hot-swapped prompt is incompatible with the current agent's tool state?
- How does async execution handle an agent inbox that fills faster than it can be consumed?

## Requirements *(mandatory)*

### Functional Requirements

**Bug Fixes (P0 — Immediate)**

- **FR-001**: The system MUST call Gemini MCP tools (`gemini_debug`, `gemini_brainstorm`, `ask_gemini`) with parameter names that exactly match the MCP tool signatures.
- **FR-002**: The system MUST handle async event loop conflicts across all TUI pattern commands without crashing.
- **FR-003**: The JSON output parser MUST handle malformed LLM responses gracefully with a documented fallback rather than raising an unhandled exception.

**Bug Fixes (P1 — Production Stability)**

- **FR-004**: The MLflow observability queue MUST respect a bounded shutdown timeout and not hang indefinitely when the upstream server is unresponsive.
- **FR-005**: The observability config MUST be cached after first load, with re-reads triggered only when the underlying file changes.
- **FR-006**: Verification results retrieved from cache MUST be deep-copied before modification to prevent cache corruption.

**Context & Memory**

- **FR-007**: The execution history manager MUST automatically distill context into a summary when token usage reaches 70% of the configured limit.
- **FR-008**: The system MUST provide a persistent memo store that saves and retrieves key facts, patterns, and lessons learned across separate sessions.
- **FR-009**: The memo store MUST support semantic search so that only contextually relevant memos are injected into agent context.
- **FR-010**: A background compression process MUST monitor token usage and apply lossy summarization at configurable thresholds without blocking primary agent execution.

**Real-Time Steering**

- **FR-011**: Each agentic step processor MUST expose an interrupt queue that is checked at the start of every thought cycle.
- **FR-012**: The TUI input handler MUST be able to enqueue interrupt signals to any active agent without requiring a full stop.
- **FR-013**: The system MUST save a micro-checkpoint after each tool call, enabling rewind to the last valid checkpoint on receiving a redirect signal.
- **FR-014**: The message bus MUST support a global override signal that replaces the active prompt mid-execution.

**Non-Blocking Async Execution**

- **FR-015**: Agent execution MUST yield control during LLM I/O operations rather than blocking the event loop.
- **FR-016**: Each agent MUST have a priority-ordered inbox for receiving messages without blocking other agents.
- **FR-017**: The pipeline system MUST support topic-based subscriptions where multiple agents subscribe to the same event topic and are triggered concurrently.

### Key Entities

- **MemoStore**: Persistent, session-spanning storage of facts, lessons, and patterns learned by agents. Supports semantic retrieval by relevance to current context.
- **ContextDistiller**: Component that monitors token usage and generates compressed "state of knowledge" summaries to replace older conversation history.
- **InterruptQueue**: Per-agent queue that accepts user interrupt signals, checked at the start of each thought cycle.
- **Micro-checkpoint**: Lightweight snapshot of agent state saved after each tool call, enabling partial rewind without full restart.
- **PubSubBus**: Extended message bus supporting named topics, multi-subscriber fan-out, and concurrent subscriber triggering.
- **AsyncAgentInbox**: Priority-ordered, non-blocking message queue per agent enabling concurrent multi-agent execution.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All P0 bugs are resolved — Gemini MCP tool calls, TUI event loop crashes, and JSON parse failures produce zero errors across a standard integration test suite.
- **SC-002**: Sessions running 3x beyond the previous context limit maintain coherent agent behavior, with task completion rate remaining above 80%.
- **SC-003**: Memo retrieval across sessions surfaces at least one relevant memo for 70% of follow-up queries on previously discussed topics.
- **SC-004**: Token consumption per equivalent task is reduced by at least 30% after context distillation compared to a truncation-only baseline.
- **SC-005**: Users can send an interrupt and receive agent acknowledgment within 2 seconds of the next thought cycle boundary.
- **SC-006**: When two agents run simultaneously, neither introduces more than 5% overhead to the other's I/O operation time versus a single-agent baseline.
- **SC-007**: The TUI responds to user input in under 100ms during any active long-running agent task.
- **SC-008**: Average session duration before context degradation (measured by task failure rate increase) improves by at least 2x compared to the current baseline.

## Assumptions

- Existing SQLite infrastructure (used for session caching) will serve as the backend for both the MemoStore and micro-checkpoints without requiring a new database system.
- Context distillation will use an LLM call (small, low-cost model) to generate summaries; this cost is acceptable given the token savings achieved.
- The async refactor will be backward-compatible — existing sync interfaces will continue to work via wrapper methods.
- Pub/Sub topics will be string-keyed; no schema enforcement on message payloads is required for the initial implementation.
- The interrupt queue will not support priority interrupts in the first iteration; all interrupts are processed FIFO.
- Micro-checkpoints are ephemeral (not persisted to disk) in the first iteration; they exist only for the duration of the current session.
