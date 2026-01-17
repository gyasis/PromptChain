# Data Model: Multi-Agent Communication

**Feature**: 003-multi-agent-communication
**Date**: 2025-11-27

## Entities

### Task

Represents a delegated unit of work between agents.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| task_id | str | PK, UUID | Unique task identifier |
| session_id | str | FK → sessions | Parent session |
| description | str | NOT NULL | Human-readable task description |
| source_agent | str | NOT NULL | Agent that created the delegation |
| target_agent | str | NOT NULL | Agent assigned to execute |
| priority | str | ENUM(low, medium, high) | Execution priority |
| status | str | ENUM(pending, in_progress, completed, failed) | Current state |
| context | JSON | NULLABLE | Additional context data |
| created_at | datetime | NOT NULL | Task creation timestamp |
| completed_at | datetime | NULLABLE | Task completion timestamp |

**State Transitions**:
```
pending → in_progress → completed
         ↘         ↗
           failed
```

**Validation Rules**:
- `target_agent` must be different from `source_agent`
- `priority` defaults to "medium"
- `status` defaults to "pending"
- `completed_at` only set when status is "completed" or "failed"

---

### BlackboardEntry

Key-value storage for agent collaboration.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | int | PK, AUTO | Internal ID |
| session_id | str | FK → sessions | Parent session |
| key | str | NOT NULL | Unique key within session |
| value | JSON | NOT NULL | Stored value (any JSON-serializable) |
| written_by | str | NOT NULL | Agent that wrote the value |
| written_at | datetime | NOT NULL | Write timestamp |
| version | int | DEFAULT 1 | Optimistic locking version |

**Constraints**:
- UNIQUE(session_id, key) - One value per key per session
- `version` increments on each update

**Validation Rules**:
- `key` must be non-empty string
- `value` must be JSON-serializable
- `written_by` must be a valid agent name

---

### WorkflowState

Tracks multi-agent workflow progress.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | int | PK, AUTO | Internal ID |
| session_id | str | FK → sessions | Parent session |
| workflow_id | str | UNIQUE | Unique workflow identifier |
| stage | str | ENUM | Current workflow stage |
| agents_involved | JSON | NOT NULL | List of participating agents |
| completed_tasks | JSON | DEFAULT [] | List of completed task IDs |
| current_task | str | NULLABLE | Currently executing task ID |
| context | JSON | NULLABLE | Workflow-level context |
| started_at | datetime | NOT NULL | Workflow start timestamp |
| updated_at | datetime | NOT NULL | Last update timestamp |

**Stage Values** (enum WorkflowStage):
- `planning` - Initial phase, defining work
- `execution` - Tasks being executed
- `review` - Results being reviewed
- `complete` - Workflow finished

**State Transitions**:
```
planning → execution → review → complete
            ↑____↓
```

---

### Message (In-Memory)

Communication unit between agents. Not persisted to SQLite (activity log handles this).

| Field | Type | Description |
|-------|------|-------------|
| message_id | str | UUID |
| sender | str | Source agent name |
| receiver | str | Target agent name (or "*" for broadcast) |
| type | MessageType | REQUEST, RESPONSE, BROADCAST, DELEGATION, STATUS |
| payload | Dict | Message content |
| timestamp | datetime | Message creation time |

**MessageType Enum**:
```python
class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    DELEGATION = "delegation"
    STATUS = "status"
```

---

### ToolMetadata (Extended)

Existing entity with new fields for agent capabilities.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| name | str | PK | Tool identifier |
| category | ToolCategory | ENUM | Tool category |
| description | str | NOT NULL | Tool description |
| parameters | Dict | NOT NULL | Parameter schemas |
| function | Callable | NOT NULL | Implementation |
| tags | Set[str] | DEFAULT {} | Discovery tags |
| examples | List[str] | DEFAULT [] | Usage examples |
| **allowed_agents** | List[str] | NULLABLE, NEW | Agents allowed to use tool |
| **capabilities** | List[str] | DEFAULT [], NEW | Semantic capability tags |

**New Fields**:
- `allowed_agents`: If None, tool available to all agents. If list, only named agents can use.
- `capabilities`: Semantic tags like "data_processing", "file_read", etc.

## Relationships

```
Session (1) ──────< Task (N)
    │
    ├──────< BlackboardEntry (N)
    │
    └──────< WorkflowState (N)

WorkflowState (1) ──────< Task (N, via completed_tasks)

Agent ──────< Task (N, as source_agent or target_agent)
    │
    └──────< BlackboardEntry (N, as written_by)

ToolMetadata ──────< Agent (N:M, via allowed_agents)
```

## SQLite Schema

```sql
-- Task Queue Table (V3 migration)
CREATE TABLE IF NOT EXISTS task_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    target_agent TEXT NOT NULL,
    priority TEXT DEFAULT 'medium' CHECK(priority IN ('low', 'medium', 'high')),
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'in_progress', 'completed', 'failed')),
    context TEXT,  -- JSON serialized
    created_at TEXT NOT NULL,
    completed_at TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_task_queue_session ON task_queue(session_id);
CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(status);
CREATE INDEX IF NOT EXISTS idx_task_queue_target ON task_queue(target_agent);

-- Blackboard Table (V3 migration)
CREATE TABLE IF NOT EXISTS blackboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,  -- JSON serialized
    written_by TEXT NOT NULL,
    written_at TEXT NOT NULL,
    version INTEGER DEFAULT 1,
    UNIQUE(session_id, key),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_blackboard_session ON blackboard(session_id);

-- Workflow State Table (V3 migration)
CREATE TABLE IF NOT EXISTS workflow_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL UNIQUE,
    stage TEXT DEFAULT 'planning' CHECK(stage IN ('planning', 'execution', 'review', 'complete')),
    agents_involved TEXT NOT NULL,  -- JSON array
    completed_tasks TEXT DEFAULT '[]',  -- JSON array
    current_task TEXT,
    context TEXT,  -- JSON serialized
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_workflow_session ON workflow_state(session_id);
```

## Data Access Patterns

### Task Delegation

```python
# Create task
INSERT INTO task_queue (session_id, task_id, description, source_agent,
                        target_agent, priority, context, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))

# Get pending tasks for agent
SELECT * FROM task_queue
WHERE session_id = ? AND target_agent = ? AND status = 'pending'
ORDER BY CASE priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END

# Update task status
UPDATE task_queue SET status = ?, completed_at = datetime('now')
WHERE task_id = ?
```

### Blackboard Operations

```python
# Write (upsert)
INSERT INTO blackboard (session_id, key, value, written_by, written_at, version)
VALUES (?, ?, ?, ?, datetime('now'), 1)
ON CONFLICT(session_id, key) DO UPDATE SET
    value = excluded.value,
    written_by = excluded.written_by,
    written_at = excluded.written_at,
    version = version + 1

# Read
SELECT value, written_by, written_at, version FROM blackboard
WHERE session_id = ? AND key = ?

# List keys
SELECT key FROM blackboard WHERE session_id = ?
```

### Workflow State

```python
# Create workflow
INSERT INTO workflow_state (session_id, workflow_id, agents_involved,
                           started_at, updated_at)
VALUES (?, ?, ?, datetime('now'), datetime('now'))

# Update stage
UPDATE workflow_state SET stage = ?, updated_at = datetime('now')
WHERE workflow_id = ?

# Add completed task
UPDATE workflow_state
SET completed_tasks = json_insert(completed_tasks, '$[#]', ?),
    updated_at = datetime('now')
WHERE workflow_id = ?
```
