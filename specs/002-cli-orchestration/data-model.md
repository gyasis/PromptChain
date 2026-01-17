# Data Model: CLI Orchestration Integration

**Feature**: 002-cli-orchestration | **Date**: 2025-11-18
**Phase**: 1 - Data Models & Contracts

## Overview

This document defines the data structures for CLI orchestration integration. All models extend existing PromptChain CLI models to add orchestration capabilities while maintaining backward compatibility.

## Core Entities

### 1. AgentConfig (Extended)

**Purpose**: Configuration for individual agents within AgentChain orchestrator

**Location**: `promptchain/cli/models/agent_config.py`

**Fields**:

```python
@dataclass
class AgentConfig:
    """Extended agent configuration with instruction chains and history settings"""

    # Existing fields (v1 schema)
    name: str
    model: str  # LiteLLM model identifier
    created_at: datetime
    updated_at: datetime

    # NEW: Orchestration fields (v2 schema)
    description: str  # Used by router for agent selection
    instruction_chain: List[Union[str, Dict[str, Any]]]  # Strings, function refs, or AgenticStep configs
    tools: List[str] = field(default_factory=list)  # MCP tool names or local function names
    history_config: Optional[HistoryConfig] = None

    # Computed properties
    @property
    def is_terminal_agent(self) -> bool:
        """Terminal agents have disabled history for token efficiency"""
        return self.history_config and not self.history_config.enabled
```

**Validation Rules**:
- `name` must be unique within session (1-50 characters, alphanumeric + underscore/dash)
- `model` must be valid LiteLLM model string
- `instruction_chain` must contain at least one instruction
- `tools` references must exist (either MCP tools or registered functions)
- `description` required for multi-agent sessions (router needs it for selection)

**State Transitions**:
- Created → Active (when first message processed)
- Active → Suspended (when different agent selected)
- Suspended → Active (when re-selected by router)
- Active → Deleted (when user executes `/agent delete`)

**Relationships**:
- Belongs to Session (one-to-many: session has multiple agents)
- References HistoryConfig (one-to-one: agent has config)
- References MCP tools via `tools` list (many-to-many: agent can use multiple tools)

---

### 2. HistoryConfig (New)

**Purpose**: Per-agent history management configuration

**Location**: `promptchain/cli/models/agent_config.py` (nested within AgentConfig)

**Fields**:

```python
@dataclass
class HistoryConfig:
    """Token-efficient history configuration for individual agents"""

    enabled: bool = True
    max_tokens: int = 4000  # Token limit for conversation history
    max_entries: int = 20  # Entry limit (messages, tool calls, etc.)
    truncation_strategy: Literal["oldest_first", "keep_last"] = "oldest_first"

    # Optional filters
    include_types: Optional[List[str]] = None  # ["user_input", "agent_output", "tool_call"]
    exclude_sources: Optional[List[str]] = None  # ["system", "debug"]
```

**Validation Rules**:
- `max_tokens` range: 100-16000 (model context limits)
- `max_entries` range: 1-100
- `truncation_strategy` must be valid enum value
- `include_types` if specified must be subset of valid entry types
- `exclude_sources` if specified must be valid source identifiers

**Default Values by Agent Type**:
- Terminal: `enabled=False` (saves ~60% tokens)
- Coder: `max_tokens=4000, truncation_strategy="keep_last"`
- Researcher: `max_tokens=8000, truncation_strategy="oldest_first"`
- Analyst: `max_tokens=6000, truncation_strategy="oldest_first"`

---

### 3. Session (Extended)

**Purpose**: Session state including agents, MCP servers, and workflow state

**Location**: `promptchain/cli/models/session.py`

**Fields**:

```python
@dataclass
class Session:
    """Extended session model with orchestration support"""

    # Existing fields (v1 schema)
    session_id: str  # UUID
    created_at: datetime
    updated_at: datetime
    working_directory: str

    # NEW: Orchestration fields (v2 schema)
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    active_agent_name: Optional[str] = None  # Currently selected agent
    mcp_servers: List[MCPServerConfig] = field(default_factory=list)
    workflow_state: Optional[WorkflowState] = None
    orchestration_config: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    schema_version: str = "2.0"  # For migration tracking
```

**Validation Rules**:
- `session_id` must be valid UUID4
- `agents` must contain at least one agent (or auto-create default)
- `active_agent_name` if set must reference existing agent in `agents` dict
- `working_directory` must be valid accessible path
- `schema_version` must be semantic version string

**State Transitions**:
- Created → Active (when first message sent)
- Active → Saved (when auto-save triggers or user executes `/session save`)
- Active → Closed (when user exits CLI)
- Closed → Resumed (when user restarts session with same ID)

**Relationships**:
- Has many AgentConfig (one-to-many)
- Has many MCPServerConfig (one-to-many)
- Has one WorkflowState (one-to-one, optional)
- Has one OrchestrationConfig (one-to-one)

---

### 4. OrchestrationConfig (New)

**Purpose**: AgentChain execution mode and router settings

**Location**: `promptchain/cli/models/session.py` (nested within Session)

**Fields**:

```python
@dataclass
class OrchestrationConfig:
    """AgentChain orchestration settings"""

    execution_mode: Literal["router", "pipeline", "round-robin", "broadcast"] = "router"
    default_agent: Optional[str] = None  # Fallback if routing fails
    router_config: Optional[RouterConfig] = None  # Required if execution_mode="router"
    auto_include_history: bool = True  # Global history inclusion setting
```

**Validation Rules**:
- `execution_mode` must be valid enum value
- `default_agent` if set must reference existing agent
- `router_config` required when `execution_mode="router"`

---

### 5. RouterConfig (New)

**Purpose**: AgentChain router decision-making configuration

**Location**: `promptchain/cli/models/session.py` (nested within OrchestrationConfig)

**Fields**:

```python
@dataclass
class RouterConfig:
    """Router decision prompt and model configuration"""

    model: str = "openai/gpt-4o-mini"  # Fast model for routing decisions
    decision_prompt_template: str = DEFAULT_ROUTER_PROMPT
    timeout_seconds: int = 10  # Routing decision timeout
```

**Default Prompt Template**:
```
User query: {user_input}

Available agents:
{agent_details}

Choose the most appropriate agent based on query type and agent specialization.
Return JSON: {"chosen_agent": "agent_name", "refined_query": "optional"}
```

---

### 6. MCPServerConfig (New)

**Purpose**: MCP server connection configuration

**Location**: `promptchain/cli/models/session.py`

**Fields**:

```python
@dataclass
class MCPServerConfig:
    """MCP server connection settings"""

    id: str  # Unique identifier (e.g., "filesystem", "web_search")
    type: Literal["stdio", "http"]
    auto_connect: bool = True

    # stdio-specific fields
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)

    # http-specific fields
    url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

    # Connection state
    status: Literal["connected", "disconnected", "error"] = "disconnected"
    error_message: Optional[str] = None
    last_connected: Optional[datetime] = None
    discovered_tools: List[str] = field(default_factory=list)  # Tool names
```

**Validation Rules**:
- `id` must be unique within session
- `type="stdio"` requires `command` to be set
- `type="http"` requires `url` to be set
- `url` if set must be valid URL format
- `status` updates automatically based on connection health

**State Transitions**:
- Disconnected → Connected (successful server handshake)
- Connected → Disconnected (graceful shutdown)
- Connected → Error (connection failure, timeout)
- Error → Connected (retry successful)

---

### 7. WorkflowState (New)

**Purpose**: Multi-session workflow progress tracking

**Location**: `promptchain/cli/models/workflow.py`

**Fields**:

```python
@dataclass
class WorkflowState:
    """Persistent workflow state across sessions"""

    id: str  # UUID
    session_id: str  # Parent session
    objective: str  # User-defined workflow goal
    created_at: datetime
    updated_at: datetime
    status: Literal["active", "paused", "completed"] = "active"

    steps: List[WorkflowStep] = field(default_factory=list)
    current_step_index: int = 0

    # Metadata
    total_steps: int = field(default=0)
    completed_steps: int = field(default=0)
    estimated_remaining_steps: int = field(default=0)
```

**Validation Rules**:
- `objective` must be non-empty (10-500 characters)
- `current_step_index` must be valid index in `steps` list
- `completed_steps` must equal count of completed steps in `steps` list
- `total_steps` must equal length of `steps` list

**State Transitions**:
- Active → Paused (when user exits session mid-workflow)
- Paused → Active (when user resumes workflow)
- Active → Completed (when all steps marked complete)

**Relationships**:
- Belongs to Session (one-to-one: session has optional workflow)
- Has many WorkflowStep (one-to-many)

---

### 8. WorkflowStep (New)

**Purpose**: Individual step within multi-session workflow

**Location**: `promptchain/cli/models/workflow.py` (nested within WorkflowState)

**Fields**:

```python
@dataclass
class WorkflowStep:
    """Individual workflow step with completion tracking"""

    step_number: int  # 1-indexed step position
    description: str  # What needs to be accomplished
    status: Literal["pending", "in_progress", "completed"] = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result_summary: Optional[str] = None  # Brief outcome description
```

**Validation Rules**:
- `step_number` must be positive integer
- `description` must be non-empty (10-200 characters)
- `completed_at` only set when `status="completed"`
- `result_summary` optional even when completed

**State Transitions**:
- Pending → In Progress (when agent begins working on step)
- In Progress → Completed (when step objective achieved)
- Completed → (terminal state, no further transitions)

---

## YAML Configuration Schema

### 9. YAMLConfig (New)

**Purpose**: Declarative agent and orchestration configuration via YAML files

**Location**: `promptchain/cli/config/yaml_config.py`

**Schema Structure**:

```python
@dataclass
class YAMLConfig:
    """Parsed and validated YAML configuration"""

    mcp_servers: List[MCPServerConfig] = field(default_factory=list)
    agents: Dict[str, YAMLAgentConfig] = field(default_factory=dict)
    orchestration: YAMLOrchestrationConfig = field(default_factory=YAMLOrchestrationConfig)
    session: YAMLSessionConfig = field(default_factory=YAMLSessionConfig)
    preferences: YAMLPreferencesConfig = field(default_factory=YAMLPreferencesConfig)

    @classmethod
    def from_file(cls, path: str) -> 'YAMLConfig':
        """Load and validate YAML config from file"""
        # Implementation in YAMLConfigTranslator
```

### 10. YAML Agent Config

```python
@dataclass
class YAMLAgentConfig:
    """Agent configuration from YAML (before translation to AgentConfig)"""

    model: str
    description: str
    instruction_chain: List[Union[str, Dict[str, Any]]]
    tools: List[str] = field(default_factory=list)
    history: Optional[Dict[str, Any]] = None  # Translated to HistoryConfig
```

---

## Data Flow

### Agent Creation Flow

```
User Command
    ↓
/agent create researcher --model gpt-4 --description "Research specialist"
    ↓
CommandHandler.handle_agent_create()
    ↓
AgentConfig(
    name="researcher",
    model="gpt-4",
    description="Research specialist",
    instruction_chain=[...],
    history_config=HistoryConfig(max_tokens=8000)
)
    ↓
SessionManager.add_agent(config)
    ↓
Session.agents["researcher"] = config
    ↓
SQLite persistence
```

### Agent Routing Flow

```
User Message
    ↓
TUI App receives input
    ↓
AgentChain.process_message(input)
    ↓
Router evaluates:
  - user_input
  - agent_descriptions
  - conversation_history (if auto_include_history=True)
    ↓
Router selects agent based on decision prompt
    ↓
Selected agent (PromptChain instance) processes message
    ↓
Response returned to user
    ↓
ExecutionHistoryManager records:
  - user_input
  - agent_selection_decision
  - agent_output
  - tool_calls (if any)
```

### MCP Server Lifecycle

```
Session Initialization
    ↓
Load MCPServerConfig list from session
    ↓
For each server with auto_connect=True:
    MCPHelper.connect(server_config)
    ↓
    Server handshake
    ↓
    Tool discovery
    ↓
    Register tools with AgentChain
    ↓
    Update server.status = "connected"
    Update server.discovered_tools = [...]
    ↓
Session Ready (agents can call MCP tools)
```

### Workflow State Management

```
/workflow create "Research and implement feature X"
    ↓
WorkflowState created with:
  - objective
  - steps: [] (empty initially)
  - status: "active"
    ↓
User works through steps, CLI tracks progress
    ↓
User exits session
    ↓
WorkflowState persisted to SQLite
  - status: "paused"
    ↓
User resumes session
    ↓
WorkflowState loaded
  - Display progress summary
  - Resume from current_step_index
    ↓
User completes final step
    ↓
WorkflowState.status = "completed"
```

---

## Storage Schema

### SQLite Tables

**sessions** (extended):
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    working_directory TEXT,
    orchestration_config JSON,  -- OrchestrationConfig serialized
    schema_version TEXT DEFAULT '2.0'
);
```

**agents**:
```sql
CREATE TABLE agents (
    agent_id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(session_id),
    name TEXT UNIQUE,
    model TEXT,
    description TEXT,
    instruction_chain JSON,
    tools JSON,  -- List[str]
    history_config JSON,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

**mcp_servers**:
```sql
CREATE TABLE mcp_servers (
    server_id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(session_id),
    server_config JSON,  -- MCPServerConfig serialized
    status TEXT,
    last_connected TIMESTAMP
);
```

**workflow_states**:
```sql
CREATE TABLE workflow_states (
    workflow_id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(session_id) UNIQUE,  -- One workflow per session
    objective TEXT,
    steps JSON,  -- List[WorkflowStep] serialized
    current_step_index INTEGER,
    status TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

---

## Migration Strategy

### V1 → V2 Migration

**Backward Compatibility**:
```python
def migrate_session_v1_to_v2(v1_session: dict) -> Session:
    """Migrate existing V1 session to V2 schema"""
    return Session(
        # Preserve existing fields
        session_id=v1_session["session_id"],
        created_at=v1_session["created_at"],
        updated_at=v1_session["updated_at"],
        working_directory=v1_session.get("working_directory", os.getcwd()),

        # Add V2 fields with defaults
        agents={
            "default": AgentConfig(
                name="default",
                model="openai/gpt-4",
                description="Default conversational agent",
                instruction_chain=["{input}"],
                history_config=HistoryConfig(max_tokens=4000)
            )
        },
        active_agent_name="default",
        mcp_servers=DEFAULT_MCP_SERVERS,
        workflow_state=None,
        orchestration_config=OrchestrationConfig(),
        schema_version="2.0"
    )
```

**Data Preservation**:
- Existing conversation history preserved in JSONL files (unchanged format)
- Agent configurations from V1 become single "default" agent in V2
- No data loss during migration
- Migration automatic on first V2 CLI launch

---

## Validation & Constraints

### Cross-Entity Validation

- Agent references in `Session.active_agent_name` must exist in `Session.agents`
- Tool references in `AgentConfig.tools` must exist in MCP discovered tools or registered functions
- MCP server IDs must be unique within session
- Workflow steps must have sequential `step_number` values (1, 2, 3, ...)
- Router config required when `OrchestrationConfig.execution_mode = "router"`

### Business Logic Constraints

- Sessions must have at least one agent (auto-create "default" if none exist)
- Active agent must be set before processing messages
- MCP servers cannot be removed if agents reference their tools
- Workflow cannot be deleted while status is "active"
- Agent names must be unique within session (case-insensitive)

---

## Summary

Data model extends existing PromptChain CLI structures with:
- **AgentConfig**: Multi-agent support with instruction chains and history configs
- **Session**: Orchestration settings, MCP servers, workflow state
- **WorkflowState**: Multi-session objective tracking
- **MCPServerConfig**: External tool server management
- **YAML Configuration**: Declarative setup via config files

All models support backward compatibility via schema versioning and automatic migration.
