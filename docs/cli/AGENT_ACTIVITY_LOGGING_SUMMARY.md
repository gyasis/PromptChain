# Agent Activity Logging - Implementation Summary

**Date**: 2025-11-20
**Status**: ✅ PHASE 1 COMPLETE (Core Infrastructure)
**Next Phase**: AgentChain Integration & CLI Commands

## What Was Accomplished

### 1. Core Infrastructure ✅

#### **ActivityLogger Class** (`promptchain/cli/activity_logger.py`)

**Purpose**: Comprehensive agent activity logging system that captures ALL interactions

**Key Features**:
- JSONL file logging (grep-able, one JSON per line)
- SQLite database indexing (fast SQL queries)
- Interaction chain tracking (parent-child relationships)
- Depth tracking (nested agent reasoning)
- Activity buffering (batch writes for performance)
- Context manager support (automatic cleanup)

**Core Methods**:
```python
class ActivityLogger:
    def start_interaction_chain(root_activity_id=None) -> str
    def log_activity(activity_type, agent_name, content, ...) -> str
    def end_interaction_chain(status="completed")
    def increase_depth() / decrease_depth()
    def get_chain_activities(chain_id, include_content=True) -> List[Dict]
    def get_agent_activities(agent_name, limit=100) -> List[Dict]
```

**Activity Types Supported**:
- `user_input`: Root user question/request
- `agent_input`: Input passed to agent (refined query)
- `agent_output`: Response from agent
- `tool_call`: Agent calling a tool/function
- `tool_result`: Result returned from tool
- `reasoning_step`: Internal reasoning step
- `router_decision`: Router selecting agent
- `router_fallback`: Router falling back to default
- `error`: Error during processing
- `system_message`: System-level messages

**Data Structure** (JSONL + SQLite):
```json
{
  "activity_id": "act_20251120_123456_abc123",
  "session_name": "my-project",
  "timestamp": "2025-11-20T12:34:56.789123",
  "activity_type": "agent_output",
  "agent_name": "researcher",
  "agent_model": "gpt-4",
  "parent_activity_id": "act_20251120_123455_xyz789",
  "interaction_chain_id": "chain_20251120_123450_root",
  "depth_level": 2,
  "content": {...},
  "metadata": {...},
  "searchable_text": "Full text for grep",
  "tags": ["research", "successful"]
}
```

#### **ActivitySearcher Class** (`promptchain/cli/activity_searcher.py`)

**Purpose**: Search and query interface for logged activities

**Key Features**:
- Grep search (uses ripgrep if available, Python fallback)
- SQL queries (direct database access)
- Time range filtering
- Agent/type filtering
- Chain reconstruction with nesting
- Statistics and analytics

**Core Methods**:
```python
class ActivitySearcher:
    def grep_logs(pattern, agent_name=None, activity_type=None, ...) -> List[Dict]
    def sql_query(query, params=None) -> List[Dict]
    def get_interaction_chain(chain_id, include_nested=True) -> Dict
    def find_reasoning_chains(agent_name=None, min_depth=2) -> List[Dict]
    def get_statistics() -> Dict
    def search_by_timerange(start_time, end_time=None) -> List[Dict]
```

**Search Examples**:
```python
# Grep search
results = searcher.grep_logs(
    pattern="database",
    agent_name="researcher",
    activity_type="reasoning_step",
    since=datetime.now() - timedelta(hours=1),
    max_results=100
)

# SQL query
results = searcher.sql_query("""
    SELECT agent_name, COUNT(*) as count
    FROM agent_activities
    WHERE session_name = ?
    GROUP BY agent_name
""", ("my-session",))

# Get statistics
stats = searcher.get_statistics()
# Returns: total_activities, total_chains, activities_by_type, etc.
```

### 2. Database Schema ✅

**SQLite Tables Created**:

#### `agent_activities` Table
```sql
CREATE TABLE agent_activities (
    activity_id TEXT PRIMARY KEY,
    session_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    agent_name TEXT,
    agent_model TEXT,
    parent_activity_id TEXT,
    interaction_chain_id TEXT NOT NULL,
    depth_level INTEGER DEFAULT 0,
    content_preview TEXT,
    full_log_path TEXT,
    tags TEXT,
    FOREIGN KEY (parent_activity_id) REFERENCES agent_activities(activity_id)
);
```

**Indexes** (for fast queries):
- `idx_activities_session` (session_name)
- `idx_activities_timestamp` (timestamp)
- `idx_activities_type` (activity_type)
- `idx_activities_agent` (agent_name)
- `idx_activities_chain` (interaction_chain_id)
- `idx_activities_parent` (parent_activity_id)

#### `interaction_chains` Table
```sql
CREATE TABLE interaction_chains (
    chain_id TEXT PRIMARY KEY,
    session_name TEXT NOT NULL,
    root_activity_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    total_activities INTEGER DEFAULT 0,
    total_agents_involved INTEGER DEFAULT 0,
    max_depth_level INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active',
    FOREIGN KEY (root_activity_id) REFERENCES agent_activities(activity_id)
);
```

**Indexes**:
- `idx_chains_session` (session_name)
- `idx_chains_status` (status)

### 3. Testing ✅

Created comprehensive test script (`test_activity_logger.py`)

**Test Coverage**:
- ✅ Test 1: Basic activity logging (7 different activity types)
- ✅ Test 2: Chain retrieval with full content
- ✅ Test 3: Agent-specific activity retrieval
- ✅ Test 4: Search functionality (grep + filters)
- ✅ Test 5: SQL queries (counts, grouping, chains)
- ✅ Test 6: Statistics (totals, breakdowns, averages)
- ✅ Test 7: Context manager (auto-cleanup)

**Test Results**: ALL TESTS PASSING ✅

```
Test 1: Basic Activity Logging - ✓ PASSED
Test 2: Chain Retrieval - ✓ PASSED
Test 3: Agent Activity Retrieval - ✓ PASSED
Test 4: Search Functionality - ✓ PASSED
Test 5: SQL Queries - ✓ PASSED
Test 6: Statistics - ✓ PASSED
Test 7: Context Manager - ✓ PASSED
```

### 4. File Structure ✅

**Created Files**:
1. `promptchain/cli/activity_logger.py` (449 lines)
2. `promptchain/cli/activity_searcher.py` (490 lines)
3. `test_activity_logger.py` (520 lines)
4. `docs/cli/AGENT_ACTIVITY_LOGGING_DESIGN.md` (900+ lines)

**Storage Structure**:
```
~/.promptchain/
├── sessions/
│   ├── sessions.db (existing, now extended with activity tables)
│   ├── <session-name>/
│   │   ├── history.jsonl (existing conversation history)
│   │   └── activities/
│   │       ├── activities.jsonl (NEW: comprehensive activity log)
│   │       └── chains/ (NEW: optional per-chain logs)
```

## Technical Implementation Details

### Activity Logging Flow

```
User Input → ActivityLogger.start_interaction_chain()
    ↓
ActivityLogger.log_activity("user_input", ...)
    ↓
Agent Processes → ActivityLogger.log_activity("agent_input", ...)
    ↓
[Nested Reasoning]
    logger.increase_depth()
    ActivityLogger.log_activity("reasoning_step", ...)
    ActivityLogger.log_activity("tool_call", ...)
    ActivityLogger.log_activity("tool_result", ...)
    logger.decrease_depth()
    ↓
ActivityLogger.log_activity("agent_output", ...)
    ↓
ActivityLogger.end_interaction_chain()
```

### Dual Storage Strategy

**Why Both JSONL and SQLite?**

1. **JSONL Files** (`activities.jsonl`):
   - Complete activity records with full content
   - Grep-able text search (fast for pattern matching)
   - Human-readable format
   - Easy backup/transfer

2. **SQLite Database**:
   - Fast indexed queries (by agent, type, time, chain)
   - Relationship tracking (parent-child, chains)
   - Aggregations and statistics
   - Content previews (first 200 chars)

**Performance**:
- JSONL append: O(1) - just write to end of file
- SQLite insert: O(log n) - indexed insertion
- Grep search: O(n) - scans file (but fast enough for typical use)
- SQL query: O(log n) - index lookup

### Interaction Chain Tracking

**Concept**: Every user interaction creates a "chain" of related activities

**Example Chain Structure**:
```
chain_20251120_123450_root (depth 0)
├── act_..._001: user_input (depth 0)
├── act_..._002: agent_input (depth 0)
│   ├── act_..._003: reasoning_step (depth 1)
│   ├── act_..._004: tool_call (depth 1)
│   │   └── act_..._005: tool_result (depth 2)
│   └── act_..._006: reasoning_step (depth 1)
└── act_..._007: agent_output (depth 0)
```

**Benefits**:
- Track complete conversation threads
- Understand agent reasoning patterns
- Identify performance bottlenecks
- Reconstruct exact execution flow

### Search Capabilities

#### Grep Search
```python
# Find all database-related activities
results = searcher.grep_logs(pattern="database")

# Find errors from specific agent
results = searcher.grep_logs(
    pattern="error",
    agent_name="researcher",
    activity_type="error"
)

# Find recent activities
results = searcher.grep_logs(
    pattern=".*",
    since=datetime.now() - timedelta(hours=1)
)
```

#### SQL Queries
```python
# Count activities by agent
results = searcher.sql_query("""
    SELECT agent_name, COUNT(*) as count
    FROM agent_activities
    WHERE session_name = ?
    GROUP BY agent_name
""", ("my-session",))

# Find longest reasoning chains
results = searcher.sql_query("""
    SELECT chain_id, max_depth_level
    FROM interaction_chains
    WHERE session_name = ?
    ORDER BY max_depth_level DESC
    LIMIT 10
""", ("my-session",))

# Get failed interactions
results = searcher.sql_query("""
    SELECT * FROM interaction_chains
    WHERE status = 'error'
""")
```

## Integration Points (Phase 2 - Not Yet Implemented)

### 1. AgentChain Integration

**Planned Modification** (`promptchain/utils/agent_chain.py`):

```python
class AgentChain:
    def __init__(
        self,
        # ... existing params ...
        activity_logger: Optional[ActivityLogger] = None
    ):
        self.activity_logger = activity_logger

    async def process_input_async(self, user_input: str, ...):
        # Start chain
        if self.activity_logger:
            self.activity_logger.start_interaction_chain()
            self.activity_logger.log_activity(
                "user_input", None, {"input": user_input}
            )

        # Process... (existing logic)
        # Log each agent call, router decision, tool use

        # End chain
        if self.activity_logger:
            self.activity_logger.end_interaction_chain()
```

### 2. CLI Commands Integration

**Planned Commands** (`promptchain/cli/command_handler.py`):

```python
# /log search <pattern> [--agent <name>] [--type <type>] [--since <date>]
# /log show <activity_id> [--chain]
# /log grep <pattern> [--context 3]
# /log chains [--active] [--errors]
# /log stats [--agent <name>]
# /log export <chain_id> [--format json|text]
```

### 3. SessionManager Integration

**Planned Modification** (`promptchain/cli/session_manager.py`):

- Initialize ActivityLogger when loading session
- Pass to AgentChain initialization
- Clean up old activity logs (optional archiving)

## Usage Examples

### Basic Usage

```python
from promptchain.cli.activity_logger import ActivityLogger
from pathlib import Path

# Initialize logger
logger = ActivityLogger(
    session_name="my-project",
    log_dir=Path.home() / ".promptchain" / "sessions" / "my-project" / "activities",
    db_path=Path.home() / ".promptchain" / "sessions" / "sessions.db"
)

# Start interaction chain
chain_id = logger.start_interaction_chain()

# Log user input
logger.log_activity(
    activity_type="user_input",
    agent_name=None,
    content={"input": "Explain machine learning"}
)

# Log agent processing
logger.log_activity(
    activity_type="agent_output",
    agent_name="researcher",
    agent_model="gpt-4",
    content={"output": "Machine learning is..."},
    metadata={"tokens_used": 150}
)

# End chain
logger.end_interaction_chain(status="completed")
```

### Search Usage

```python
from promptchain.cli.activity_searcher import ActivitySearcher
from datetime import datetime, timedelta

# Initialize searcher
searcher = ActivitySearcher(
    session_name="my-project",
    log_dir=Path.home() / ".promptchain" / "sessions" / "my-project" / "activities",
    db_path=Path.home() / ".promptchain" / "sessions" / "sessions.db"
)

# Grep search
results = searcher.grep_logs(
    pattern="machine learning",
    agent_name="researcher",
    since=datetime.now() - timedelta(hours=1),
    max_results=10
)

# SQL query
agent_stats = searcher.sql_query("""
    SELECT agent_name, COUNT(*) as count, AVG(depth_level) as avg_depth
    FROM agent_activities
    WHERE session_name = ?
    GROUP BY agent_name
""", ("my-project",))

# Get statistics
stats = searcher.get_statistics()
print(f"Total activities: {stats['total_activities']}")
print(f"Total chains: {stats['total_chains']}")
print(f"Average depth: {stats['avg_chain_depth']}")
```

### Context Manager Usage

```python
# Auto-cleanup with context manager
with ActivityLogger(session_name="test", log_dir=..., db_path=...) as logger:
    chain_id = logger.start_interaction_chain()
    logger.log_activity("user_input", None, {"input": "Test"})
    logger.log_activity("agent_output", "researcher", {"output": "Response"})
    # Chain automatically ends when context exits
    # Status set to "error" if exception occurs
```

## Benefits Delivered

### For Users

✅ **Complete Transparency**: See exactly what every agent does, step-by-step
✅ **Debugging Power**: Grep through logs to find specific interactions, errors, reasoning chains
✅ **Performance Analysis**: Identify slow agents, excessive reasoning loops
✅ **Audit Trail**: Complete record of all agent activities for compliance/review
✅ **Learning**: Understand how multi-agent systems make decisions

### For Agents

✅ **Context Without Token Cost**: Search logs without loading full history into context
✅ **Pattern Discovery**: Find similar past interactions for reference
✅ **Error Investigation**: Review previous errors and solutions
✅ **Tool Usage Patterns**: Analyze which tools work best for which tasks

### For Developers

✅ **Debugging**: Complete activity trail for troubleshooting
✅ **Optimization**: Identify bottlenecks and inefficient patterns
✅ **Testing**: Verify agent behavior across complex scenarios
✅ **Metrics**: Track agent performance, success rates, token usage

## Performance Considerations

### Storage Estimates

- **JSONL Entry**: ~1-2KB per activity
- **SQLite Row**: ~100 bytes per activity
- **Expected Volume**: ~1000 activities per hour = ~2MB/hour
- **30-Day Session**: ~1.4GB (highly compressible)

### Query Performance

- **Grep Search**: O(n) but fast (ripgrep processes ~2GB/sec)
- **SQL Queries**: O(log n) with indexes
- **Chain Reconstruction**: O(depth) - typically 3-5 levels
- **Statistics**: Cached aggregations, millisecond response

### Optimization Strategies

1. **Batch Writes**: Buffer activities and write every 5 seconds (implemented)
2. **Lazy Loading**: Only load full content when requested (implemented)
3. **Compression**: Gzip old JSONL files (>1 week old) (planned Phase 3)
4. **Archiving**: Move old activities to separate DB after 30 days (planned Phase 3)

## Known Limitations

1. **Storage Growth**: Logs grow over time (but compressible ~10:1 ratio)
2. **Grep Performance**: Linear scan for text search (but ripgrep is very fast)
3. **No Real-Time Streaming**: Not designed for real-time log streaming (yet)
4. **Single Session Focus**: Optimized for per-session queries, not cross-session

## Next Steps (Phase 2)

### Required Tasks

1. **AgentChain Integration** (1 day):
   - Add `activity_logger` parameter to AgentChain.__init__
   - Log all agent interactions (input, output, router decisions)
   - Track reasoning steps in AgenticStepProcessor
   - Log tool calls and results

2. **SessionManager Integration** (0.5 days):
   - Initialize ActivityLogger when loading session
   - Pass to AgentChain initialization
   - Add cleanup methods for old logs

3. **CLI Commands** (1 day):
   - Implement `/log` command family
   - Add search/grep interface
   - Add chain navigation
   - Add statistics display

4. **TUI Integration** (1 day):
   - Create ActivityLogViewer widget (optional)
   - Real-time activity stream
   - Search interface

5. **Documentation** (0.5 days):
   - User guide for `/log` commands
   - Examples and workflows
   - Performance tuning guide

**Total Estimated Time**: 4 days

## Alternatives Considered

### Why Not Just Extend RunLogger?

**RunLogger** is event-based generic logging:
- No structured activity tracking
- No parent-child relationships
- No interaction chain concept
- No specialized search interface

**ActivityLogger** is activity-centric:
- Structured agent interactions
- Chain tracking with depth
- Rich search and query
- Agent-specific indexing

**Decision**: Keep both - RunLogger for events, ActivityLogger for activities

### Why Not Use ExecutionHistoryManager?

**ExecutionHistoryManager** is in-memory context management:
- Designed for LLM context windows
- Truncates old data
- No persistence
- No search capabilities

**ActivityLogger** is persistent audit trail:
- Complete history preserved
- SQLite + JSONL storage
- Grep and SQL search
- Never truncates

**Decision**: Both serve different purposes - ExecutionHistoryManager for context, ActivityLogger for audit trail

## Conclusion

✅ **Phase 1 Complete**: Core infrastructure production-ready

**Delivered**:
- ActivityLogger class (449 lines, fully functional)
- ActivitySearcher class (490 lines, grep + SQL)
- Comprehensive test suite (ALL PASSING)
- SQLite schema with optimized indexes
- Complete design documentation

**Ready for Phase 2**: Integration with AgentChain, SessionManager, and CLI commands

This system solves the user's requirement for **comprehensive, searchable agent activity logging** that works independently from chat history and token limits, enabling both users and agents to find specific interactions without loading everything into memory.

---

*Agent Activity Logging Summary | 2025-11-20 | CLI Enhancement Phase 1*
