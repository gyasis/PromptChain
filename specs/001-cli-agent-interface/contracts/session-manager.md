# Contract: SessionManager

**Component**: `promptchain.cli.session_manager.SessionManager`
**Purpose**: Manages session lifecycle, persistence, and state transitions
**Integration Points**: SQLite database, ExecutionHistoryManager, AgentChain

---

## Public API Contract

### Class Definition

```python
class SessionManager:
    """
    Manages CLI session lifecycle and persistence.

    Responsibilities:
    - Create, load, save, delete sessions
    - Manage session state transitions
    - Integrate with ExecutionHistoryManager
    - Handle auto-save triggers
    """

    def __init__(
        self,
        db_path: Path,
        sessions_dir: Path,
        default_model: str = "openai/gpt-4"
    ):
        """
        Initialize session manager.

        Args:
            db_path: Path to SQLite database file
            sessions_dir: Directory for session data (JSONL logs)
            default_model: Default LLM model for new sessions
        """
```

---

### Method: create_session

**Purpose**: Create a new session with initial configuration

**Signature**:
```python
async def create_session(
    self,
    name: str,
    working_directory: Path | None = None,
    default_model: str | None = None
) -> Session:
    """
    Create a new session.

    Args:
        name: Unique session name (1-64 chars, alphanumeric+dashes)
        working_directory: Session working dir (default: current directory)
        default_model: LLM model for default agent (default: from config)

    Returns:
        Session: Created session object

    Raises:
        SessionExistsError: If session name already exists
        ValidationError: If name format invalid or working_dir inaccessible
    """
```

**Contract Guarantees**:
- Session name uniqueness enforced (database UNIQUE constraint)
- Working directory validated (exists, readable)
- Session ID generated as UUID v4
- Created session is in "Active" state
- ExecutionHistoryManager initialized with default settings
- Session entry created in SQLite database
- Session directory created: `sessions_dir/{session_id}/`

**Performance**: <100ms for session creation (SC-001 budget)

**Example**:
```python
manager = SessionManager(db_path, sessions_dir)
session = await manager.create_session(
    name="my-project",
    working_directory=Path("/home/user/projects/my-project")
)
# session.id = "uuid-..."
# session.name = "my-project"
# session.state = "Active"
```

---

### Method: load_session

**Purpose**: Load existing session by name

**Signature**:
```python
async def load_session(self, name: str) -> Session:
    """
    Load session by name.

    Args:
        name: Session name to load

    Returns:
        Session: Loaded session object with full conversation history

    Raises:
        SessionNotFoundError: If session name doesn't exist
        SessionCorruptedError: If session data is corrupted or invalid
    """
```

**Contract Guarantees**:
- Session metadata loaded from SQLite
- Agents loaded and associated with session
- Conversation history loaded from `history.jsonl`
- ExecutionHistoryManager reconstructed with conversation history
- `session.last_accessed` updated to current timestamp
- Session transitioned to "Active" state

**Performance**: <3s for session load (SC-003)

**Example**:
```python
session = await manager.load_session("my-project")
# session.messages = [Message(...), Message(...), ...]
# session.agents = {"coding": Agent(...), "research": Agent(...)}
```

---

### Method: save_session

**Purpose**: Persist session state to database and files

**Signature**:
```python
async def save_session(self, session: Session) -> None:
    """
    Save session to database and files.

    Args:
        session: Session object to persist

    Raises:
        SessionSaveError: If save operation fails (disk full, permissions)
    """
```

**Contract Guarantees**:
- Session metadata written to SQLite (atomic transaction)
- Agents written to SQLite agents table
- Conversation history appended to `history.jsonl` (incremental)
- `session.last_accessed` updated
- Auto-save timestamp recorded
- Session remains in "Active" state after save

**Performance**: <2s for session save (SC-003)

**Atomicity**: Database transaction ensures metadata consistency

**Example**:
```python
session.add_message(Message(role="user", content="Hello"))
await manager.save_session(session)
# Session persisted to disk, can be resumed later
```

---

### Method: list_sessions

**Purpose**: Retrieve list of all sessions with metadata

**Signature**:
```python
async def list_sessions(
    self,
    limit: int = 100,
    sort_by: str = "last_accessed"
) -> list[SessionMetadata]:
    """
    List sessions with metadata.

    Args:
        limit: Maximum number of sessions to return
        sort_by: Sort field ("last_accessed", "created_at", "name")

    Returns:
        list[SessionMetadata]: Session summaries (id, name, timestamps, agent_count)
    """
```

**Contract Guarantees**:
- Sessions sorted by specified field (descending for timestamps)
- Agent count included (efficient JOIN query)
- No conversation history loaded (metadata only)
- Fast response for large session counts

**Performance**: <500ms for 1000+ sessions

**Example**:
```python
sessions = await manager.list_sessions(limit=10)
# [
#   SessionMetadata(name="my-project", last_accessed=..., agent_count=3),
#   SessionMetadata(name="research", last_accessed=..., agent_count=1),
#   ...
# ]
```

---

### Method: delete_session

**Purpose**: Permanently remove session

**Signature**:
```python
async def delete_session(self, name: str) -> None:
    """
    Delete session permanently.

    Args:
        name: Session name to delete

    Raises:
        SessionNotFoundError: If session doesn't exist
    """
```

**Contract Guarantees**:
- Session removed from SQLite (CASCADE deletes agents)
- Session directory deleted: `sessions_dir/{session_id}/`
- All files removed: `history.jsonl`, logs, etc.
- Deletion is atomic (transaction)
- No orphaned data remains

**Safety**: Requires confirmation in CLI (separate concern)

**Example**:
```python
await manager.delete_session("old-project")
# Session completely removed from system
```

---

### Method: auto_save_if_needed

**Purpose**: Check if auto-save should trigger and execute if needed

**Signature**:
```python
async def auto_save_if_needed(self, session: Session) -> bool:
    """
    Auto-save session if conditions met.

    Args:
        session: Active session to potentially save

    Returns:
        bool: True if auto-save was performed

    Triggers:
    - Every 5 messages since last save (SC-007)
    - Every 2 minutes since last save (SC-007)
    - Before agent switch
    """
```

**Contract Guarantees**:
- Non-blocking (async)
- Triggers based on message count OR time interval
- Updates auto-save timestamp
- Gracefully handles save failures (logs error, doesn't crash)

**Example**:
```python
# Called by CLI event loop
was_saved = await manager.auto_save_if_needed(session)
if was_saved:
    logger.info("Auto-save completed")
```

---

## Integration Contracts

### With ExecutionHistoryManager

**Initialization**:
```python
# SessionManager creates ExecutionHistoryManager for each session
history_manager = ExecutionHistoryManager(
    max_tokens=8000,
    max_entries=100,
    truncation_strategy="oldest_first"
)
session.history_manager = history_manager
```

**Conversation Tracking**:
```python
# SessionManager delegates conversation tracking
history_manager.add_entry("user_input", message.content, source="user")
history_manager.add_entry("agent_output", response, source=agent_name)
```

---

### With SQLite Database

**Database Schema**:
```sql
-- Managed by SessionManager
CREATE TABLE sessions (...);
CREATE TABLE agents (...);
```

**Transaction Patterns**:
```python
# Atomic session save
async with self.db_connection() as conn:
    conn.execute("BEGIN TRANSACTION")
    try:
        # Update sessions table
        conn.execute("UPDATE sessions SET last_accessed = ? WHERE id = ?", ...)
        # Update agents table
        conn.executemany("INSERT OR REPLACE INTO agents ...", ...)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
```

---

## Error Handling Contracts

### SessionExistsError

```python
class SessionExistsError(Exception):
    """Raised when creating session with duplicate name."""
    def __init__(self, name: str):
        super().__init__(f"Session '{name}' already exists")
```

### SessionNotFoundError

```python
class SessionNotFoundError(Exception):
    """Raised when loading non-existent session."""
    def __init__(self, name: str):
        super().__init__(f"Session '{name}' not found")
```

### SessionCorruptedError

```python
class SessionCorruptedError(Exception):
    """Raised when session data is invalid or corrupted."""
    def __init__(self, name: str, reason: str):
        super().__init__(f"Session '{name}' is corrupted: {reason}")
```

---

## Testing Contract

### Contract Tests

```python
# tests/cli/contract/test_session_manager_contract.py

async def test_create_session_uniqueness():
    """Contract: Session names must be unique."""
    manager = SessionManager(...)
    await manager.create_session("project")

    with pytest.raises(SessionExistsError):
        await manager.create_session("project")  # Duplicate

async def test_save_and_load_roundtrip():
    """Contract: Save/load must preserve session state."""
    manager = SessionManager(...)
    session = await manager.create_session("test")
    session.add_message(Message(role="user", content="Hello"))

    await manager.save_session(session)
    loaded = await manager.load_session("test")

    assert loaded.name == session.name
    assert len(loaded.messages) == 1
    assert loaded.messages[0].content == "Hello"

async def test_performance_load_session():
    """Contract: Session load must complete in <3s (SC-003)."""
    manager = SessionManager(...)
    # Create session with 100 messages
    session = await create_large_session(message_count=100)
    await manager.save_session(session)

    start = time.perf_counter()
    loaded = await manager.load_session(session.name)
    duration = time.perf_counter() - start

    assert duration < 3.0, f"Load took {duration}s, exceeds 3s limit"
```

---

## Backward Compatibility

**Version 1.0 (Initial)**:
- All APIs defined above are stable
- Database schema version tracked
- Future schema changes via migrations
- No breaking changes to public API without major version bump

**Extensibility Points**:
- `Session.metadata` dict for custom data
- Migration system for schema evolution
- Plugin hooks (future: FR-EXT-007)
