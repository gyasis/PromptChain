# PRD: CLI Multi-Agent Communication Architecture
## Feature ID: 003-cli-multi-agent-communication

**Version**: 1.0
**Status**: Draft
**Authors**: Claude + Gemini (Collaborative Analysis)
**Date**: 2025-11-27
**Parent Branch**: 002-cli-orchestration
**Target Branch**: 003-cli-multi-agent-communication

---

## 1. Overview

### 1.1 Problem Statement
PromptChain CLI currently supports only 14% (2/14) of production-grade agentic patterns. Agents cannot:
- Communicate directly with each other
- Discover capabilities of other agents
- Delegate tasks autonomously
- Share data via blackboard patterns
- Track workflow state across interactions

### 1.2 Proposed Solution
Implement a **HYBRID EXTEND-FIRST** architecture that:
- Extends existing CLI mechanisms (ToolRegistry, AgentChain, Session Manager)
- Adds targeted new CLI-specific modules only where extension is infeasible
- Increases pattern coverage from 14% to 57%

### 1.3 Success Metrics
| Metric | Current | Target |
|--------|---------|--------|
| Fully Supported Patterns | 2 (14%) | 8 (57%) |
| Partially Supported Patterns | 2 (14%) | 4 (29%) |
| Not Supported Patterns | 10 (72%) | 2 (14%) |

---

## 2. Requirements

### 2.1 Functional Requirements

#### FR-001: Agent Communication Bus
**Priority**: P0 (Critical)
**Description**: Enable direct agent-to-agent messaging via typed message handlers

**Acceptance Criteria**:
- [ ] Agents can send typed messages (request, response, broadcast)
- [ ] CLI decorator `@cli_communication_handler` filters messages by sender/receiver/type
- [ ] Activity logger captures all communication for debugging
- [ ] Backward compatible - existing code works without handlers

#### FR-002: Agent Capability Registry
**Priority**: P0 (Critical)
**Description**: Enable agents to discover capabilities of other agents

**Acceptance Criteria**:
- [ ] ToolRegistry supports `allowed_agents` parameter
- [ ] ToolRegistry supports `capabilities` semantic tags
- [ ] `discover_capabilities(agent_name)` method available
- [ ] Existing tools default to all-agent access

#### FR-003: Task Delegation Protocol
**Priority**: P1 (High)
**Description**: Enable agents to delegate tasks to other agents

**Acceptance Criteria**:
- [ ] `delegate_task` tool registered in CLI tool registry
- [ ] `request_help` tool for stuck-agent assistance
- [ ] Task queue persisted in session SQLite
- [ ] Task status tracking (pending, in_progress, completed, failed)

#### FR-004: Blackboard Collaboration
**Priority**: P1 (High)
**Description**: Enable shared data space for agent collaboration

**Acceptance Criteria**:
- [ ] `write_to_blackboard(key, value)` tool available
- [ ] `read_from_blackboard(key)` tool available
- [ ] Blackboard data persisted in session SQLite
- [ ] Concurrent access handled with locking

#### FR-005: Workflow State Management
**Priority**: P2 (Medium)
**Description**: Track workflow progress across agent interactions

**Acceptance Criteria**:
- [ ] Workflow stages tracked (planning, execution, review, complete)
- [ ] AgentChain callbacks update state on events
- [ ] CLI commands for workflow inspection
- [ ] State persisted in session SQLite

### 2.2 Non-Functional Requirements

#### NFR-001: Backward Compatibility
- All existing CLI commands must continue to work unchanged
- Existing sessions must load successfully (auto-migration)
- Existing tool registrations must work without modification

#### NFR-002: Performance
- Communication overhead < 10ms per message
- Blackboard read/write < 5ms for typical operations
- No regression in existing CLI response time

#### NFR-003: Token Efficiency
- Maintain 90% token optimization achieved in Phase 11
- Communication metadata minimal in LLM context

---

## 3. Architecture

### 3.1 Design Principles
1. **EXTEND over BUILD** where infrastructure exists
2. **CLI-specific modules** separate from core library
3. **Decorator-based** approach for communication handlers
4. **SQLite-based** persistence (existing pattern)
5. **Tool registry extension** for capabilities (not separate registry)

### 3.2 Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Layer                                 │
├─────────────────────────────────────────────────────────────┤
│  Commands: /capabilities, /tasks, /blackboard, /workflow     │
├─────────────────────────────────────────────────────────────┤
│                    Tools Layer                               │
├─────────────────────────────────────────────────────────────┤
│  delegation_tools.py  │  blackboard_tools.py                │
│  - delegate_task      │  - write_to_blackboard              │
│  - request_help       │  - read_from_blackboard             │
├─────────────────────────────────────────────────────────────┤
│                 Communication Layer                          │
├─────────────────────────────────────────────────────────────┤
│  handlers.py          │  message_bus.py                     │
│  - @cli_handler       │  - send_message()                   │
│  - filter by type     │  - broadcast()                      │
├─────────────────────────────────────────────────────────────┤
│                 Foundation Layer                             │
├─────────────────────────────────────────────────────────────┤
│  registry.py (ext)    │  session_manager.py (ext)           │
│  - allowed_agents     │  - blackboard tables                │
│  - capabilities       │  - task_queue tables                │
│  - discover_caps()    │  - workflow_state tables            │
├─────────────────────────────────────────────────────────────┤
│                 Existing Infrastructure                      │
├─────────────────────────────────────────────────────────────┤
│  AgentChain           │  ToolRegistry        │  SQLite      │
│  - callbacks          │  - categories        │  - sessions  │
│  - router strategies  │  - validation        │  - history   │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Gap-to-Solution Mapping

| Gap | Decision | Extend % | Build % |
|-----|----------|----------|---------|
| AgentCommunicationBus | Extend AgentChain + NEW decorators | 60% | 40% |
| AgentCapabilityRegistry | Extend ToolRegistry | 90% | 10% |
| TaskDelegation | Extend callbacks + NEW tools | 50% | 50% |
| Blackboard | Extend history + NEW tools | 40% | 60% |
| WorkflowState | Extend Session Manager | 80% | 20% |

---

## 4. Technical Specifications

### 4.1 Gap 1: AgentCommunicationBus

**Files to Modify**:
- `promptchain/cli/command_handler.py` - Add communication commands

**Files to Create**:
- `promptchain/cli/communication/__init__.py`
- `promptchain/cli/communication/handlers.py`
- `promptchain/cli/communication/message_bus.py`

**API Design**:
```python
# Decorator for message handlers
@cli_communication_handler(sender="AgentA", type="request")
def handle_request(message: Dict, sender: str, receiver: str) -> Dict:
    """Handle and optionally modify messages."""
    return message  # Return modified or original message

# Message types
class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    DELEGATION = "delegation"
    STATUS = "status"
```

### 4.2 Gap 2: AgentCapabilityRegistry

**Files to Modify**:
- `promptchain/cli/tools/registry.py`
- `promptchain/cli/tools/library/registration.py`

**API Design**:
```python
@registry.register(
    category="analysis",
    description="Analyze data files",
    allowed_agents=["DataAnalyst", "Supervisor"],  # NEW
    capabilities=["data_processing", "statistics"],  # NEW
    parameters={...}
)
def analyze_data(path: str) -> str:
    ...

# New method
def discover_capabilities(agent_name: str) -> List[ToolMetadata]:
    """Return tools available to specific agent."""
    ...
```

### 4.3 Gap 3: TaskDelegation Protocol

**Files to Modify**:
- `promptchain/cli/session_manager.py` - Add task_queue table
- `promptchain/utils/strategies/dynamic_decomposition_strategy.py`

**Files to Create**:
- `promptchain/cli/tools/library/delegation_tools.py`

**API Design**:
```python
# Task structure
@dataclass
class Task:
    task_id: str
    description: str
    source_agent: str
    target_agent: str
    priority: str  # "low", "medium", "high"
    status: str    # "pending", "in_progress", "completed", "failed"
    context: Dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime]

# Tools
async def delegate_task(
    task_description: str,
    target_agent: str,
    priority: str = "medium",
    context: Optional[Dict] = None
) -> str:
    """Delegate task to another agent."""
    ...

async def request_help(
    problem_description: str,
    context: Optional[Dict] = None
) -> str:
    """Request help from capable agents."""
    ...
```

### 4.4 Gap 4: Blackboard Collaboration

**Files to Modify**:
- `promptchain/cli/session_manager.py` - Add blackboard table
- `promptchain/cli/models/session.py`

**Files to Create**:
- `promptchain/cli/tools/library/blackboard_tools.py`

**API Design**:
```python
# Blackboard entry
@dataclass
class BlackboardEntry:
    key: str
    value: Any  # JSON serializable
    written_by: str
    written_at: datetime
    version: int

# Tools
def write_to_blackboard(key: str, value: Any) -> str:
    """Write data to shared blackboard."""
    ...

def read_from_blackboard(key: str) -> Any:
    """Read data from shared blackboard."""
    ...

def list_blackboard_keys() -> List[str]:
    """List all keys on blackboard."""
    ...
```

### 4.5 Gap 5: WorkflowState Management

**Files to Modify**:
- `promptchain/cli/session_manager.py` - Add workflow_state table
- `promptchain/cli/command_handler.py` - Add workflow commands
- `promptchain/cli/models/session.py`

**Files to Create**:
- `promptchain/cli/models/workflow.py`

**API Design**:
```python
# Workflow state
@dataclass
class WorkflowState:
    workflow_id: str
    stage: str  # "planning", "execution", "review", "complete"
    agents_involved: List[str]
    completed_tasks: List[str]
    current_task: Optional[str]
    context: Dict[str, Any]
    started_at: datetime
    updated_at: datetime

# Stages enum
class WorkflowStage(Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    REVIEW = "review"
    COMPLETE = "complete"
```

---

## 5. SQLite Schema Extensions

### 5.1 Blackboard Table
```sql
CREATE TABLE IF NOT EXISTS blackboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,  -- JSON serialized
    written_by TEXT NOT NULL,
    written_at TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    UNIQUE(session_id, key)
);
```

### 5.2 Task Queue Table
```sql
CREATE TABLE IF NOT EXISTS task_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'pending',
    context TEXT,  -- JSON serialized
    created_at TEXT NOT NULL,
    completed_at TEXT
);
```

### 5.3 Workflow State Table
```sql
CREATE TABLE IF NOT EXISTS workflow_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL UNIQUE,
    stage TEXT DEFAULT 'planning',
    agents_involved TEXT,  -- JSON array
    completed_tasks TEXT,  -- JSON array
    current_task TEXT,
    context TEXT,  -- JSON serialized
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

---

## 6. Implementation Plan

### 6.1 Implementation Order

```
Foundation Layer → Communication Layer → Tools Layer → CLI Commands
      ↓                   ↓                  ↓              ↓
   Tables            Handlers           Functions         UI
```

### 6.2 Work Streams (Parallel)

| Stream | Components | Dependencies |
|--------|------------|--------------|
| **A: Schema** | SQLite tables for blackboard, task_queue, workflow | None |
| **B: Registry** | ToolRegistry extensions | None |
| **C: Communication** | Message bus, handlers | Stream A |
| **D: Tools** | Delegation, blackboard tools | Streams A, B |
| **E: Commands** | CLI commands | Streams C, D |

### 6.3 Testing Strategy

| Component | Test Type | Location |
|-----------|-----------|----------|
| Registry extensions | Unit | `tests/cli/tools/test_registry.py` |
| Communication handlers | Unit | `tests/cli/communication/test_handlers.py` |
| Delegation tools | Integration | `tests/cli/tools/test_delegation.py` |
| Blackboard tools | Integration | `tests/cli/tools/test_blackboard.py` |
| Workflow state | Integration | `tests/cli/integration/test_workflow.py` |
| End-to-end | E2E | `tests/cli/e2e/test_multi_agent.py` |

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking backward compatibility | Medium | High | Comprehensive regression tests |
| Performance degradation | Low | Medium | Benchmark before/after |
| Concurrent access issues | Medium | High | SQLite locking, transaction isolation |
| Complexity explosion | Medium | Medium | Strict interface boundaries |

---

## 8. Patterns Enabled

### 8.1 Newly Fully Supported
1. **Blackboard Collaboration** - Shared workspace via blackboard tools
2. **Hierarchical Agent Teams** - Task delegation + capability discovery
3. **Competitive Agent Ensembles** - Result comparison via communication
4. **Parallel Evaluation** - Multi-agent evaluation coordination
5. **Redundant Execution** - Monitoring + cancellation via bus
6. **Context Pre-processing** - Parallel filtering agents

### 8.2 Enhanced Partial Support
1. **Branching Thoughts** - Hypothesis sharing via blackboard
2. **Parallel Query Expansion** - Query agent coordination
3. **Sharded Retrieval** - Retrieval coordination via bus
4. **Multi-Hop Retrieval** - Sub-question coordination

---

## 9. Open Questions

1. **Q**: Should communication handlers be async by default?
   - **Proposed**: Yes, to align with AgentChain's async execution

2. **Q**: How to handle message delivery failures?
   - **Proposed**: Log error, continue execution, expose failure in workflow state

3. **Q**: Should blackboard support expiration/TTL?
   - **Proposed**: Defer to Phase 2, manual cleanup initially

---

## 10. Appendix

### A. Related Documents
- [gap_analysis_and_solution_mapping.md](./gap_analysis_and_solution_mapping.md)
- [14_agentic_patterns_gap_analysis.md](./14_agentic_patterns_gap_analysis.md)
- [multi_agent_task_network_concepts.md](./multi_agent_task_network_concepts.md)
- [agent_interaction_design.md](./agent_interaction_design.md)

### B. File Change Summary

**Files to Modify (6)**:
| File | Changes |
|------|---------|
| `promptchain/cli/tools/registry.py` | Add `allowed_agents`, `capabilities`, `discover_capabilities()` |
| `promptchain/cli/session_manager.py` | Add blackboard, task_queue, workflow tables |
| `promptchain/cli/command_handler.py` | Add `/capabilities`, `/tasks`, `/blackboard`, `/workflow` |
| `promptchain/cli/models/session.py` | Add BlackboardEntry, Task, WorkflowState models |
| `promptchain/cli/tools/library/registration.py` | Add capability tags to 19 existing tools |
| `promptchain/utils/strategies/dynamic_decomposition_strategy.py` | Add agent-initiated delegation |

**Files to Create (6)**:
| File | Purpose |
|------|---------|
| `promptchain/cli/communication/__init__.py` | Communication module exports |
| `promptchain/cli/communication/handlers.py` | `@cli_communication_handler` decorator |
| `promptchain/cli/communication/message_bus.py` | Message routing and dispatch |
| `promptchain/cli/tools/library/delegation_tools.py` | `delegate_task`, `request_help` tools |
| `promptchain/cli/tools/library/blackboard_tools.py` | `write_to_blackboard`, `read_from_blackboard` |
| `promptchain/cli/models/workflow.py` | WorkflowState, WorkflowStage models |

### C. Collaborative Analysis Attribution
This PRD was developed through collaborative analysis between:
- **Claude** (Anthropic) - Architecture analysis, codebase exploration
- **Gemini** (Google) - Pattern recommendations, extend vs. build trade-offs

Both AI systems agreed on the HYBRID EXTEND-FIRST approach documented herein.
