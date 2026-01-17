# Research: Multi-Agent Communication Architecture

**Feature**: 003-multi-agent-communication
**Date**: 2025-11-27
**Status**: Complete

## Research Questions Resolved

### Q1: How should ToolRegistry be extended for agent-specific capabilities?

**Decision**: Add two new optional parameters to the existing `@registry.register()` decorator:
- `allowed_agents: Optional[List[str]]` - Restricts tool access to specific agents
- `capabilities: Optional[List[str]]` - Semantic tags for capability discovery

**Rationale**:
- Leverages existing decorator infrastructure (no new registration pattern needed)
- Backward compatible - tools without these params remain available to all agents
- Follows OpenAI function calling schema pattern already established

**Alternatives Considered**:
- Separate AgentCapabilityRegistry class - Rejected: duplicates existing infrastructure
- Agent-tool mapping table in SQLite - Rejected: adds complexity, slower lookups
- Tool inheritance/composition - Rejected: over-engineered for current needs

**Code Pattern**:
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
```

### Q2: What message types are needed for agent communication?

**Decision**: Five message types cover all identified use cases:
1. `REQUEST` - Agent asking another agent to perform action
2. `RESPONSE` - Reply to a request
3. `BROADCAST` - Message to all agents
4. `DELEGATION` - Task delegation with context
5. `STATUS` - Status updates (progress, completion, errors)

**Rationale**:
- Maps directly to supported agentic patterns (hierarchical, competitive, parallel)
- Minimal set avoids message type proliferation
- Each type has clear semantics for handler filtering

**Alternatives Considered**:
- Single generic "message" type - Rejected: loses semantic filtering capability
- Event-based (emit/subscribe) - Rejected: adds complexity without clear benefit
- Full pubsub system - Rejected: over-engineered for current scale

### Q3: How should blackboard data be persisted?

**Decision**: SQLite table in existing sessions.db with JSON serialization:

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

**Rationale**:
- Single database file simplifies backup/restore
- Existing session management handles connection lifecycle
- JSON serialization handles any Python object (with json.dumps/loads)
- Version field enables optimistic locking if needed later

**Alternatives Considered**:
- Separate SQLite file per session - Rejected: complicates session management
- In-memory dict with periodic flush - Rejected: data loss risk on crash
- Redis/external store - Rejected: adds dependency, deployment complexity

### Q4: How should task delegation track state?

**Decision**: Task state machine with 4 states:
```
pending → in_progress → completed
           ↓
         failed
```

**Rationale**:
- Minimal state machine covers all delegation scenarios
- Clear transitions: only forward or to failed
- No need for "cancelled" state initially (can be added later)

**State Transitions**:
- `pending` → `in_progress`: Target agent picks up task
- `in_progress` → `completed`: Task finished successfully
- `in_progress` → `failed`: Task encountered error
- `pending` → `failed`: Task rejected by target agent

### Q5: Should communication handlers be async?

**Decision**: Yes, async by default with sync wrapper pattern.

**Rationale**:
- Aligns with existing async-first design (Constitution VI)
- AgentChain and PromptChain already use async internally
- Enables non-blocking message processing
- Sync wrapper (`asyncio.run()`) maintains backward compatibility

**Pattern**:
```python
@cli_communication_handler(type="request")
async def handle_request(message: Dict, sender: str, receiver: str) -> Dict:
    # Async processing
    return message

# Sync wrapper for external use
def handle_request_sync(...):
    return asyncio.run(handle_request(...))
```

### Q6: How to handle message delivery failures?

**Decision**: Log error, continue execution, expose failure in activity log.

**Rationale**:
- Message delivery should not block agent execution
- Activity logger already captures all events for debugging
- Failed messages can be retried by sending agent if needed
- No automatic retry (simplicity over complexity)

**Error Handling Flow**:
1. Handler throws exception → catch and log
2. Mark message as "failed" in activity log
3. Continue system execution
4. Expose failure count in `/workflow` command output

### Q7: What capability categories should be pre-defined?

**Decision**: Use semantic tags rather than fixed categories. Suggested initial tags for 19 existing tools:

| Tool Group | Suggested Capabilities |
|------------|----------------------|
| Filesystem tools | `file_read`, `file_write`, `file_search` |
| Shell tools | `shell_execute`, `environment` |
| Session tools | `session_management`, `history` |
| Context tools | `context_management`, `memory` |
| Analysis tools | `data_processing`, `parsing` |

**Rationale**:
- Semantic tags more flexible than hierarchical categories
- Tools can have multiple capabilities
- New capabilities can be added without schema changes
- Discovery by capability enables intelligent routing

## Technology Best Practices

### SQLite Concurrency

**Best Practice**: Use WAL mode for concurrent read/write access.

```python
conn.execute("PRAGMA journal_mode=WAL")
```

**Applied**: Already used in existing session_manager.py

### Decorator Pattern for Handlers

**Best Practice**: Use `functools.wraps` to preserve function metadata.

```python
def cli_communication_handler(sender=None, type=None):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Filter logic
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

### JSON Serialization for Complex Types

**Best Practice**: Use custom encoder for datetime and dataclass objects.

```python
class BlackboardEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        return super().default(obj)
```

## Integration Patterns

### ToolRegistry Extension Pattern

**Existing Code** (`registry.py:162`):
```python
@dataclass
class ToolMetadata:
    name: str
    category: Union[ToolCategory, str]
    description: str
    parameters: Dict[str, ParameterSchema]
    function: Callable
    tags: Set[str] = field(default_factory=set)
    examples: List[str] = field(default_factory=list)
```

**Extension** (minimal addition):
```python
@dataclass
class ToolMetadata:
    # ... existing fields ...
    allowed_agents: Optional[List[str]] = None  # NEW
    capabilities: List[str] = field(default_factory=list)  # NEW
```

### SessionManager Table Addition Pattern

**Existing Migration Flow** (`session_manager.py:93`):
```python
def _check_and_migrate_v2(self):
    current_version = self.get_schema_version()
    # Version check and migration
```

**Extension**: Add V3 migration check for new tables.

## Summary

All research questions resolved. No NEEDS CLARIFICATION items remaining.

**Key Decisions**:
1. Extend ToolRegistry with `allowed_agents` and `capabilities` parameters
2. Five message types: REQUEST, RESPONSE, BROADCAST, DELEGATION, STATUS
3. SQLite blackboard table with JSON serialization
4. 4-state task delegation: pending → in_progress → completed/failed
5. Async-first handlers with sync wrappers
6. Fail-safe message delivery (log and continue)
7. Semantic capability tags over fixed categories
