# Agent Activity Logging Design

**Date**: 2025-11-20
**Status**: 🎯 DESIGN PHASE
**Priority**: HIGH - User Requirement

## Problem Statement

Users need a comprehensive, searchable log of ALL agent interactions and reasoning steps that:

1. **Captures Everything**: Every agent input, output, reasoning step, tool call, and decision
2. **Independent from Chat History**: Works regardless of token limits or chat history truncation
3. **Searchable**: Agents can grep/search logs without loading full conversation history
4. **Navigable**: Each log entry includes agent name, timestamp, and full content for easy tracking
5. **Non-Intrusive**: Doesn't flood chat history or consume excessive tokens
6. **Integrates with Existing Systems**: Works alongside RunLogger and ExecutionHistoryManager

### User Quote

> "now where is another concern we still need a tmp log or something to capture ALL HISTORY AND CONTENT WHERE INTER AGENTS IN STEP OR AGENTS.. THIS WAY IF ANY AGENT NEED TO THEY access the logs without the full logs always going to the chat history do you understand the concept, this should go hand and hand with our history and observability methods but never though where are limiting the chat history the agents should still be able to search with a grep or something the log to see each step and output of agents step process or some agent, this is the most robust way of collecting everything without flooding token space....understand....get a robust way of navigating the logging file which does get different log with the agent name and a time stamp to get the way to navigate and track through the document"

## Existing Infrastructure Analysis

### Current Logging Systems

#### 1. **RunLogger** (`promptchain/utils/logging_utils.py`)

**Purpose**: Event-based logging to console and JSONL files

**Capabilities**:
- Console logging with log levels (INFO, WARNING, ERROR)
- JSONL file logging (one JSON object per line)
- Event-based structure (`{"event": "...", "timestamp": "...", ...}`)
- Session-based logging (all logs to single file if `session_filename` provided)

**Limitations**:
- Generic event logging (not agent-activity specific)
- No built-in search/grep interface
- No structured agent interaction tracking
- Limited metadata (no agent chains, reasoning steps)

**Current Usage in AgentChain**:
- 100+ `self.logger.log_run()` calls throughout agent_chain.py
- Events: `router_decision`, `router_fallback`, `chat_started`, `agent_running`, etc.
- Logs to session-specific files when `log_dir` provided

#### 2. **ExecutionHistoryManager** (`promptchain/utils/execution_history_manager.py`)

**Purpose**: Token-aware conversation history management

**Capabilities**:
- Structured entry types (`user_input`, `agent_output`, `tool_call`, `tool_result`, etc.)
- Token counting with tiktoken
- Automatic truncation (`oldest_first` strategy)
- Filtering by type, source, timestamp
- Formatted output (`chat`, `full_json`, `content_only`)

**Limitations**:
- Designed for in-memory context management (not persistent)
- Truncates old entries to stay under token limits
- Not searchable without loading into memory
- No agent-specific indexing
- No reasoning chain tracking

#### 3. **Session Manager** (`promptchain/cli/session_manager.py`)

**Purpose**: SQLite-based session persistence for CLI

**Capabilities**:
- SQLite database for session metadata
- JSONL files for conversation history
- Session save/load/list/delete

**Current Schema**:
```sql
CREATE TABLE sessions (
    session_name TEXT PRIMARY KEY,
    created_at TEXT,
    updated_at TEXT,
    working_directory TEXT
)

CREATE TABLE agents (
    id INTEGER PRIMARY KEY,
    session_name TEXT,
    agent_name TEXT,
    model TEXT,
    description TEXT,
    created_at TEXT,
    FOREIGN KEY (session_name) REFERENCES sessions(session_name)
)

CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    session_name TEXT,
    agent_name TEXT,
    role TEXT,
    content TEXT,
    timestamp TEXT,
    FOREIGN KEY (session_name) REFERENCES sessions(session_name)
)
```

**Limitations**:
- Message storage is basic (role, content, timestamp)
- No reasoning step tracking
- No tool call tracking
- No agent interaction metadata
- Not optimized for searching agent activities

## Proposed Solution: Comprehensive Agent Activity Logger

### Design Goals

1. **Complete Activity Capture**: Log every agent interaction, reasoning step, tool call, decision
2. **Persistent Storage**: JSONL files + SQLite for fast querying
3. **Searchable Interface**: grep-friendly text logs + SQL queries for structured data
4. **Agent-Centric Indexing**: Track activities by agent, session, timestamp, interaction chain
5. **Non-Intrusive**: Independent from chat history, doesn't affect token usage
6. **Backwards Compatible**: Integrates with existing RunLogger, ExecutionHistoryManager, SessionManager

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Agent Activity Logger                      │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │  ActivityLogger  │  │  ActivitySearcher│               │
│  │                  │  │                  │               │
│  │  - Capture       │  │  - Grep logs     │               │
│  │  - Persist       │  │  - SQL queries   │               │
│  │  - Index         │  │  - Filter        │               │
│  └──────────────────┘  └──────────────────┘               │
│           │                      │                          │
│           ▼                      ▼                          │
│  ┌────────────────────────────────────────┐                │
│  │         Storage Layer                  │                │
│  │                                        │                │
│  │  ┌─────────────┐  ┌─────────────────┐│                │
│  │  │ JSONL Files │  │ SQLite Database ││                │
│  │  │             │  │                 ││                │
│  │  │ - Full logs │  │ - Fast queries  ││                │
│  │  │ - Grep-able │  │ - Indexes       ││                │
│  │  └─────────────┘  └─────────────────┘│                │
│  └────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
           ▲                               ▲
           │                               │
    ┌──────┴────────┐             ┌───────┴────────┐
    │  AgentChain   │             │  CLI Commands  │
    │               │             │                │
    │  - Log events │             │  /log search   │
    │  - Track      │             │  /log show     │
    │    reasoning  │             │  /log grep     │
    └───────────────┘             └────────────────┘
```

### Data Model

#### Activity Entry Structure (JSONL)

```json
{
  "activity_id": "act_20251120_123456_abc123",
  "session_name": "my-project",
  "timestamp": "2025-11-20T12:34:56.789123",
  "activity_type": "agent_input|agent_output|tool_call|tool_result|reasoning_step|router_decision|error",
  "agent_name": "researcher",
  "agent_model": "gpt-4",
  "parent_activity_id": "act_20251120_123455_xyz789",  // For tracking chains
  "interaction_chain_id": "chain_20251120_123450_root",  // Root interaction
  "depth_level": 2,  // Nesting depth (0 = root, 1 = first agent, 2 = nested reasoning)
  "content": {
    "input": "User question or previous agent output",
    "output": "Agent response or tool result",
    "metadata": {
      "tokens_used": 1234,
      "duration_ms": 567,
      "temperature": 0.7,
      "reasoning_steps": 3,
      "tool_calls": ["search_files", "read_document"]
    }
  },
  "context": {
    "mode": "router",
    "routing_decision": "complex_llm_router",
    "fallback_count": 0
  },
  "searchable_text": "Full concatenated text for grep: User question... Agent response...",
  "tags": ["research", "multi-hop", "successful"]
}
```

#### SQLite Schema Extension

```sql
-- Add to existing sessions.db
CREATE TABLE IF NOT EXISTS agent_activities (
    activity_id TEXT PRIMARY KEY,
    session_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    agent_name TEXT,
    agent_model TEXT,
    parent_activity_id TEXT,
    interaction_chain_id TEXT NOT NULL,
    depth_level INTEGER DEFAULT 0,
    content_preview TEXT,  -- First 200 chars for quick view
    full_log_path TEXT,    -- Path to JSONL file with complete entry
    tags TEXT,             -- Comma-separated for basic searching
    FOREIGN KEY (session_name) REFERENCES sessions(session_name),
    FOREIGN KEY (parent_activity_id) REFERENCES agent_activities(activity_id)
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_activities_session ON agent_activities(session_name);
CREATE INDEX IF NOT EXISTS idx_activities_timestamp ON agent_activities(timestamp);
CREATE INDEX IF NOT EXISTS idx_activities_type ON agent_activities(activity_type);
CREATE INDEX IF NOT EXISTS idx_activities_agent ON agent_activities(agent_name);
CREATE INDEX IF NOT EXISTS idx_activities_chain ON agent_activities(interaction_chain_id);
CREATE INDEX IF NOT EXISTS idx_activities_parent ON agent_activities(parent_activity_id);

-- Activity chains tracking
CREATE TABLE IF NOT EXISTS interaction_chains (
    chain_id TEXT PRIMARY KEY,
    session_name TEXT NOT NULL,
    root_activity_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    total_activities INTEGER DEFAULT 0,
    total_agents_involved INTEGER DEFAULT 0,
    max_depth_level INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active',  -- active|completed|error
    FOREIGN KEY (session_name) REFERENCES sessions(session_name),
    FOREIGN KEY (root_activity_id) REFERENCES agent_activities(activity_id)
);

CREATE INDEX IF NOT EXISTS idx_chains_session ON interaction_chains(session_name);
CREATE INDEX IF NOT EXISTS idx_chains_status ON interaction_chains(status);
```

### Component Design

#### 1. ActivityLogger Class

**Location**: `promptchain/cli/activity_logger.py`

**Purpose**: Capture and persist all agent activities

```python
class ActivityLogger:
    """Comprehensive agent activity logging system.

    Captures ALL agent interactions, reasoning steps, tool calls, and decisions
    to persistent storage (JSONL + SQLite) for searchability without affecting
    chat history or token usage.
    """

    def __init__(
        self,
        session_name: str,
        log_dir: Path,
        db_path: Path,
        enable_console: bool = False
    ):
        """Initialize activity logger.

        Args:
            session_name: Current session identifier
            log_dir: Directory for JSONL activity logs
            db_path: Path to SQLite database
            enable_console: Whether to also log to console (default: False)
        """
        self.session_name = session_name
        self.log_dir = Path(log_dir)
        self.db_path = Path(db_path)
        self.enable_console = enable_console

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current interaction chain tracking
        self.current_chain_id: Optional[str] = None
        self.current_parent_id: Optional[str] = None
        self.current_depth: int = 0

        # Initialize database
        self._init_database()

    def start_interaction_chain(self, root_activity_id: str) -> str:
        """Start a new interaction chain (root user input)."""

    def log_activity(
        self,
        activity_type: str,
        agent_name: Optional[str],
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Log a single activity entry."""

    def end_interaction_chain(self, status: str = "completed"):
        """Mark current interaction chain as completed."""

    def get_chain_activities(self, chain_id: str) -> List[Dict[str, Any]]:
        """Get all activities in a specific interaction chain."""

    def get_agent_activities(
        self,
        agent_name: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent activities for a specific agent."""
```

#### 2. ActivitySearcher Class

**Location**: `promptchain/cli/activity_searcher.py`

**Purpose**: Provide grep/SQL search interface for logged activities

```python
class ActivitySearcher:
    """Search and query interface for agent activity logs."""

    def __init__(self, session_name: str, log_dir: Path, db_path: Path):
        """Initialize activity searcher."""

    def grep_logs(
        self,
        pattern: str,
        agent_name: Optional[str] = None,
        activity_type: Optional[str] = None,
        since: Optional[datetime] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """Grep through activity logs with filters."""

    def sql_query(
        self,
        query: str,
        params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute custom SQL query on activity database."""

    def get_interaction_chain(
        self,
        chain_id: str,
        include_nested: bool = True
    ) -> Dict[str, Any]:
        """Get complete interaction chain with nested structure."""

    def find_reasoning_chains(
        self,
        agent_name: str,
        min_depth: int = 2,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find complex reasoning chains (multi-hop) by agent."""

    def get_statistics(self) -> Dict[str, Any]:
        """Get activity statistics for current session."""
```

### Integration Points

#### 1. AgentChain Integration

Modify `promptchain/utils/agent_chain.py`:

```python
class AgentChain:
    def __init__(
        self,
        # ... existing params ...
        activity_logger: Optional[ActivityLogger] = None
    ):
        self.activity_logger = activity_logger

    async def process_input_async(self, user_input: str, ...):
        # Start new interaction chain
        if self.activity_logger:
            chain_id = self.activity_logger.start_interaction_chain(
                root_activity_id=self._generate_activity_id()
            )
            self.activity_logger.log_activity(
                activity_type="user_input",
                agent_name=None,
                content={"input": user_input},
                tags=["root_interaction"]
            )

        # Process with existing logic...
        # Log each agent call, router decision, tool use

        # End interaction chain
        if self.activity_logger:
            self.activity_logger.end_interaction_chain(status="completed")
```

#### 2. CLI Command Integration

Add new `/log` commands to `promptchain/cli/command_handler.py`:

```python
# /log search <pattern> [--agent <name>] [--type <type>] [--since <date>]
# /log show <activity_id> [--chain]
# /log grep <pattern> [--context 3]
# /log chains [--active] [--errors]
# /log stats [--agent <name>]
# /log export <chain_id> [--format json|text]
```

#### 3. TUI Integration

Add log viewer to `promptchain/cli/tui/app.py`:

```python
# New widget: ActivityLogViewer
# - Real-time activity stream
# - Search interface
# - Chain navigation
# - Agent filter
```

### File Structure

```
~/.promptchain/
├── sessions/
│   ├── sessions.db (extended with agent_activities tables)
│   ├── <session-name>/
│   │   ├── history.jsonl (existing conversation history)
│   │   └── activities/
│   │       ├── activities.jsonl (comprehensive activity log)
│   │       └── chains/
│   │           ├── chain_<id>.jsonl (individual chain logs)
│   │           └── ... (one file per interaction chain)
```

### Activity Types

1. **`user_input`**: Root user question/request
2. **`agent_input`**: Input passed to an agent (may be refined query)
3. **`agent_output`**: Response from agent
4. **`tool_call`**: Agent calling a tool/function
5. **`tool_result`**: Result returned from tool
6. **`reasoning_step`**: Internal reasoning step (AgenticStepProcessor)
7. **`router_decision`**: Router selecting which agent to use
8. **`router_fallback`**: Router falling back to default agent
9. **`error`**: Error during processing
10. **`system_message`**: System-level messages

### Search Examples

#### Grep Interface

```bash
# Find all research agent reasoning steps
/log grep "research" --agent researcher --type reasoning_step

# Find errors in last hour
/log grep "error" --since "1 hour ago"

# Find tool calls with "database" in context
/log grep "database" --type tool_call

# Show activity with context
/log show act_20251120_123456_abc123 --chain
```

#### SQL Interface

```bash
# Count activities by agent
/log sql "SELECT agent_name, COUNT(*) FROM agent_activities GROUP BY agent_name"

# Find longest reasoning chains
/log sql "SELECT chain_id, max_depth_level FROM interaction_chains ORDER BY max_depth_level DESC LIMIT 10"

# Get failed interactions
/log sql "SELECT * FROM interaction_chains WHERE status = 'error'"
```

## Implementation Plan

### Phase 1: Core Infrastructure (1-2 days)

**Tasks**:
1. Create `ActivityLogger` class with JSONL persistence
2. Extend SQLite schema in SessionManager
3. Create `ActivitySearcher` class with grep functionality
4. Write comprehensive unit tests

**Deliverables**:
- `promptchain/cli/activity_logger.py` (300+ lines)
- `promptchain/cli/activity_searcher.py` (200+ lines)
- Extended `promptchain/cli/session_manager.py` (add activity tables)
- `tests/cli/unit/test_activity_logger.py` (150+ lines)

### Phase 2: AgentChain Integration (1 day)

**Tasks**:
1. Integrate ActivityLogger into AgentChain initialization
2. Add activity logging to all agent interactions
3. Track interaction chains (start/end)
4. Add reasoning step logging to AgenticStepProcessor

**Deliverables**:
- Modified `promptchain/utils/agent_chain.py` (add logging calls)
- Modified `promptchain/utils/agentic_step_processor.py` (add step logging)
- Integration tests

### Phase 3: CLI Commands (1 day)

**Tasks**:
1. Implement `/log` command family
2. Add search filters (agent, type, date)
3. Add chain navigation commands
4. Add export functionality

**Deliverables**:
- Extended `promptchain/cli/command_handler.py`
- Command tests

### Phase 4: TUI Integration (1 day)

**Tasks**:
1. Create ActivityLogViewer widget
2. Add real-time activity stream
3. Add search interface
4. Add chain visualization

**Deliverables**:
- `promptchain/cli/tui/activity_log_viewer.py`
- TUI integration tests

### Phase 5: Documentation & Testing (1 day)

**Tasks**:
1. Write user documentation
2. Create example workflows
3. Performance testing
4. Complete test coverage

**Deliverables**:
- `docs/cli/ACTIVITY_LOGGING_GUIDE.md`
- `docs/cli/ACTIVITY_LOGGING_EXAMPLES.md`
- Performance benchmarks

## Benefits

### For Users

1. **Complete Transparency**: See exactly what every agent does, step-by-step
2. **Debugging Power**: Grep through logs to find specific interactions, errors, reasoning chains
3. **Performance Analysis**: Identify slow agents, excessive reasoning loops
4. **Audit Trail**: Complete record of all agent activities for compliance/review
5. **Learning**: Understand how multi-agent systems make decisions

### For Agents

1. **Context Without Token Cost**: Agents can search logs without loading full history into context
2. **Pattern Discovery**: Find similar past interactions for reference
3. **Error Investigation**: Review previous errors and solutions
4. **Tool Usage Patterns**: Analyze which tools work best for which tasks

### For Developers

1. **Debugging**: Complete activity trail for troubleshooting
2. **Optimization**: Identify bottlenecks and inefficient patterns
3. **Testing**: Verify agent behavior across complex scenarios
4. **Metrics**: Track agent performance, success rates, token usage

## Performance Considerations

### Storage

- **JSONL Files**: ~1-2KB per activity entry
- **SQLite Index**: ~100 bytes per activity entry
- **Expected Volume**: ~1000 activities per hour of active use = ~2MB/hour

### Query Performance

- **SQLite Indexes**: O(log n) lookups on agent_name, timestamp, chain_id
- **Grep Search**: O(n) but only scans JSONL files (not loaded in memory)
- **Chain Reconstruction**: O(depth) where depth is typically 3-5 levels

### Optimization Strategies

1. **Batch Writes**: Buffer activities and write in batches every 5 seconds
2. **Lazy Loading**: Only load full content when requested (store preview in SQLite)
3. **Compression**: Optionally gzip old JSONL files (>1 week old)
4. **Archiving**: Move old activities to separate database after 30 days

## Security & Privacy

1. **Local Storage**: All logs stored locally in `~/.promptchain/`
2. **No External Transmission**: Activity logs never leave user's machine
3. **Sanitization**: Option to exclude sensitive content from logs
4. **Encryption**: Optional SQLite encryption for sensitive sessions

## Known Limitations

1. **Storage Growth**: Logs will grow over time (but compressible)
2. **Grep Performance**: Linear scan for text search (but fast enough for typical usage)
3. **No Real-Time Streaming**: TUI updates every second (not instant)
4. **Single Session Focus**: Designed for per-session logging (not cross-session queries)

## Future Enhancements

1. **Cross-Session Search**: Query across all sessions
2. **Visualization**: Interactive chain visualization in TUI
3. **Export Formats**: Export to CSV, markdown, PDF
4. **Metrics Dashboard**: Real-time agent performance metrics
5. **Pattern Recognition**: ML-based pattern detection in agent interactions
6. **Collaboration**: Share activity logs (with privacy controls)

## Alternatives Considered

### Alternative 1: Extend RunLogger

**Pros**: Builds on existing infrastructure
**Cons**: RunLogger is generic event logging, not structured for agent activities

**Decision**: Create separate ActivityLogger that complements RunLogger

### Alternative 2: Use ExecutionHistoryManager

**Pros**: Already structured for agent interactions
**Cons**: Designed for in-memory context, truncates old data, no persistence

**Decision**: ActivityLogger provides persistent, searchable storage that ExecutionHistoryManager cannot

### Alternative 3: Separate Log Files per Agent

**Pros**: Easy to find agent-specific logs
**Cons**: Hard to track interaction chains, no unified search

**Decision**: Unified log with agent indexing provides better searchability

## Conclusion

The Comprehensive Agent Activity Logger fills a critical gap in the CLI's observability infrastructure:

✅ **Captures everything** without affecting token usage
✅ **Searchable** via grep and SQL queries
✅ **Navigable** with interaction chain tracking
✅ **Integrates seamlessly** with existing systems
✅ **Production-ready** with proper indexing and performance optimization

This system enables users to understand exactly what their multi-agent systems are doing, debug complex interactions, and search through complete activity history without loading everything into memory.

---

*Agent Activity Logging Design | 2025-11-20 | CLI Enhancement*
