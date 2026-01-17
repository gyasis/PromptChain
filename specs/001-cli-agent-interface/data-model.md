# Phase 1: Data Model & Entity Design

**Feature**: PromptChain CLI Agent Interface
**Branch**: 001-cli-agent-interface
**Date**: 2025-11-16

## Overview

This document defines the core entities for the PromptChain CLI, their attributes, relationships, validation rules, and state transitions. All entities are designed to integrate seamlessly with existing PromptChain components (AgentChain, ExecutionHistoryManager, MCPHelper).

---

## Entity: Session

**Purpose**: Represents a persistent conversation instance with unique identity, conversation history, and agent configurations.

### Attributes

| Attribute | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| `id` | str (UUID) | Primary key, immutable | Unique session identifier |
| `name` | str | Unique, 1-64 chars, alphanumeric+dashes | User-friendly session name |
| `created_at` | float (timestamp) | Immutable, auto-set | Unix timestamp of creation |
| `last_accessed` | float (timestamp) | Auto-updated | Unix timestamp of last interaction |
| `working_directory` | Path | Must exist, readable | Directory context for session |
| `active_agent` | str \| None | Foreign key to Agent.name | Currently active agent (null = default) |
| `default_model` | str | Valid LiteLLM model string | Model for default agent |
| `auto_save_enabled` | bool | Default: True | Enable periodic auto-save |
| `auto_save_interval` | int | Seconds, 60-600 | Auto-save frequency |
| `metadata` | dict | JSON-serializable | Extensible metadata storage |

### Validation Rules

```python
def validate_session_name(name: str) -> bool:
    """Session name: 1-64 chars, alphanumeric, dashes, underscores."""
    pattern = r'^[a-zA-Z0-9_-]{1,64}$'
    return bool(re.match(pattern, name))

def validate_working_directory(path: Path) -> bool:
    """Working directory must exist and be readable."""
    return path.exists() and path.is_dir() and os.access(path, os.R_OK)

def validate_model_string(model: str) -> bool:
    """Model string: provider/model-name format."""
    pattern = r'^[a-z]+/[a-zA-Z0-9_.-]+$'
    return bool(re.match(pattern, model))
```

### State Transitions

```
[Created] --load()--> [Active] --save()--> [Persisted]
   |                     |
   |                     +--auto_save()--> [Persisted]
   |                     |
   +--resume()---------> [Active]
   |
   +--delete()---------> [Deleted]

States:
- Created: Session initialized but not yet started
- Active: Session running, accepting user input
- Persisted: Session saved to SQLite + JSONL files
- Deleted: Session removed from database
```

### Relationships

- **One-to-Many with Agent**: Session has 0+ agents
- **One-to-Many with Message**: Session has 0+ messages (conversation history)
- **One-to-One with ExecutionHistoryManager**: Session creates one history manager

### Storage Schema (SQLite)

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL,
    working_directory TEXT NOT NULL,
    active_agent TEXT,
    default_model TEXT NOT NULL DEFAULT 'openai/gpt-4',
    auto_save_enabled INTEGER NOT NULL DEFAULT 1,
    auto_save_interval INTEGER NOT NULL DEFAULT 120,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_sessions_name ON sessions(name);
CREATE INDEX idx_sessions_last_accessed ON sessions(last_accessed DESC);
```

---

## Entity: Agent

**Purpose**: Represents an AI agent configuration with model selection, system prompt, and capabilities.

### Attributes

| Attribute | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| `session_id` | str (UUID) | Foreign key to Session.id | Owning session |
| `name` | str | Unique per session, 1-32 chars | Agent identifier |
| `description` | str | 0-256 chars | Human-readable description |
| `model_name` | str | Valid LiteLLM model string | LLM model for this agent |
| `system_prompt` | str \| None | Optional custom system prompt | Agent instructions |
| `created_at` | float | Auto-set | Creation timestamp |
| `usage_count` | int | Non-negative, default: 0 | Number of times agent used |
| `last_used` | float \| None | Timestamp or null | Last invocation time |

### Validation Rules

```python
def validate_agent_name(name: str) -> bool:
    """Agent name: 1-32 chars, alphanumeric, dashes, underscores."""
    pattern = r'^[a-zA-Z0-9_-]{1,32}$'
    return bool(re.match(pattern, name))

def validate_description(desc: str) -> bool:
    """Description: max 256 characters."""
    return len(desc) <= 256
```

### State Transitions

```
[Configured] --create()--> [Inactive]
                              |
                              +--use()--> [Active]
                              |            |
                              |            +--switch()--> [Inactive]
                              |            |
                              |            +--delete()--> [Deleted]
                              |
                              +--delete()--> [Deleted]

States:
- Configured: Agent parameters defined but not created
- Inactive: Agent exists but not currently selected
- Active: Agent currently handling user messages
- Deleted: Agent removed from session
```

### Relationships

- **Many-to-One with Session**: Agent belongs to one session
- **One-to-One with PromptChain**: Each agent wraps one PromptChain instance

### Storage Schema (SQLite)

```sql
CREATE TABLE agents (
    session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    model_name TEXT NOT NULL,
    system_prompt TEXT,
    created_at REAL NOT NULL,
    usage_count INTEGER NOT NULL DEFAULT 0,
    last_used REAL,
    PRIMARY KEY (session_id, name),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX idx_agents_session ON agents(session_id);
```

---

## Entity: Message

**Purpose**: Represents a single exchange in the conversation with role, content, metadata.

### Attributes

| Attribute | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| `id` | str (UUID) | Primary key | Unique message identifier |
| `session_id` | str (UUID) | Foreign key to Session.id | Owning session |
| `role` | str | Enum: user, assistant, system | Message role |
| `content` | str | Required, 1+ chars | Message text content |
| `agent_name` | str \| None | For assistant messages | Agent that generated response |
| `timestamp` | float | Auto-set | Message creation time |
| `file_references` | list[str] | JSON array of file paths | Files mentioned with @syntax |
| `command_executed` | str \| None | Shell command if !syntax used | Executed command |
| `metadata` | dict | JSON-serializable | Tool calls, timing, errors |

### Validation Rules

```python
def validate_role(role: str) -> bool:
    """Role must be user, assistant, or system."""
    return role in {'user', 'assistant', 'system'}

def validate_content(content: str) -> bool:
    """Content must be non-empty."""
    return len(content.strip()) > 0
```

### State Transitions

```
[Composed] --send()--> [Pending] --process()--> [Completed]
                                      |
                                      +--error()--> [Failed]

States:
- Composed: User is typing message (not yet sent)
- Pending: Message sent, awaiting agent response
- Completed: Agent response received and displayed
- Failed: Error occurred during processing
```

### Relationships

- **Many-to-One with Session**: Message belongs to one session
- **Many-to-Many with FileReference**: Message can reference 0+ files

### Storage Format (JSONL)

Messages stored in `~/.promptchain/sessions/{session_id}/history.jsonl`:

```jsonl
{"id": "uuid-1", "role": "user", "content": "Analyze @src/main.py", "timestamp": 1700000000.0, "file_references": ["src/main.py"]}
{"id": "uuid-2", "role": "assistant", "agent_name": "coding", "content": "The main.py file...", "timestamp": 1700000005.0, "metadata": {"model": "openai/gpt-4", "tokens": 1250}}
{"id": "uuid-3", "role": "user", "content": "!git status", "timestamp": 1700000010.0, "command_executed": "git status"}
```

---

## Entity: FileReference

**Purpose**: Represents a file or directory referenced in conversation context.

### Attributes

| Attribute | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| `path` | Path | Must exist, absolute path | File/directory absolute path |
| `content` | str \| None | For files only | File content snapshot |
| `is_directory` | bool | Computed from path | True if directory reference |
| `size_bytes` | int | Non-negative | File size in bytes |
| `modified_at` | float | File mtime | Last modification timestamp |
| `truncated` | bool | Default: False | True if content was truncated |
| `preview_lines` | int \| None | For large files | Number of lines in preview |

### Validation Rules

```python
def validate_file_reference(path: Path, max_size: int = 100 * 1024 * 1024) -> bool:
    """File must exist, be readable, and under size limit."""
    if not path.exists():
        raise FileNotFoundError(f"Referenced file not found: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Cannot read file: {path}")
    if path.is_file() and path.stat().st_size > max_size:
        # Allow but mark as truncated
        return True
    return True
```

### Content Handling Strategy

```python
def load_file_content(path: Path, max_size: int = 100 * 1024) -> tuple[str, bool]:
    """
    Load file content with intelligent truncation.

    Returns: (content, truncated)
    """
    file_size = path.stat().st_size

    if file_size <= max_size:
        # Small file: read entirely
        return path.read_text(), False

    # Large file: preview (first 500 + last 100 lines)
    lines = []
    with path.open('r') as f:
        lines.extend(itertools.islice(f, 500))  # First 500 lines
        lines.append(f"\n... [truncated {file_size - max_size} bytes] ...\n")

        # Seek to near end for last 100 lines
        f.seek(max(0, file_size - 10000))
        tail_lines = f.readlines()
        lines.extend(tail_lines[-100:])

    return ''.join(lines), True
```

### Relationships

- **Many-to-Many with Message**: File can be referenced by multiple messages
- **No persistence**: FileReference is ephemeral, reconstructed on session resume

---

## Entity: Command

**Purpose**: Represents a slash command execution with result/error state.

### Attributes

| Attribute | Type | Constraints | Description |
|-----------|------|-------------|-------------|
| `command_name` | str | Enum (see below) | Command identifier |
| `arguments` | list[str] | Optional | Command arguments |
| `executed_at` | float | Auto-set | Execution timestamp |
| `result` | str \| None | Success message | Command output |
| `error` | str \| None | Error message if failed | Error details |
| `status` | str | Enum: success, error | Execution status |

### Supported Commands (MVP)

```python
SLASH_COMMANDS = {
    '/agent': {
        'subcommands': ['create', 'list', 'use', 'delete'],
        'description': 'Manage agents in session'
    },
    '/session': {
        'subcommands': ['save', 'list', 'delete'],
        'description': 'Manage sessions'
    },
    '/help': {
        'subcommands': ['commands'],
        'description': 'Show help documentation'
    },
    '/exit': {
        'description': 'Exit session'
    },
    '/status': {
        'description': 'Show session status'
    }
}
```

### Validation Rules

```python
def validate_command(command_name: str, arguments: list[str]) -> bool:
    """Validate command syntax and arguments."""
    if command_name not in SLASH_COMMANDS:
        raise CommandNotFoundError(f"Unknown command: {command_name}")

    cmd_spec = SLASH_COMMANDS[command_name]
    if 'subcommands' in cmd_spec:
        if not arguments:
            raise InvalidCommandError(f"{command_name} requires subcommand")
        if arguments[0] not in cmd_spec['subcommands']:
            raise InvalidCommandError(f"Unknown subcommand: {arguments[0]}")

    return True
```

### State Transitions

```
[Typed] --parse()--> [Validated] --execute()--> [Completed]
                          |                         |
                          |                         +--[Success]
                          |                         |
                          +--error()--------------> [Failed]

States:
- Typed: User typed slash command
- Validated: Command syntax validated
- Completed: Command executed successfully
- Failed: Command validation or execution failed
```

---

## Entity Relationships Diagram

```
Session (1) ─────< (M) Agent
   |                     |
   |                     └──> (1:1) PromptChain instance
   |
   ├─────< (M) Message
   |          |
   |          └──> (M:M) FileReference
   |
   └──> (1:1) ExecutionHistoryManager
```

---

## Data Flow Example: User Creates Agent and Sends Message

```
1. User types: /agent create coding --model openai/gpt-4
   └─> Command entity validated
   └─> Agent entity created (session_id, name="coding", model="openai/gpt-4")
   └─> PromptChain instance instantiated with model
   └─> Agent saved to SQLite agents table

2. User types: /agent use coding
   └─> Session.active_agent updated to "coding"
   └─> SQLite sessions table updated

3. User types: Analyze @src/main.py
   └─> Message entity created (role="user", content="Analyze @src/main.py")
   └─> FileReference entity created (path="src/main.py", content=<file content>)
   └─> Message.file_references = ["src/main.py"]
   └─> Message appended to history.jsonl

4. Agent processes message
   └─> PromptChain.process_prompt(content + file_content)
   └─> Message entity created (role="assistant", agent_name="coding", content=<response>)
   └─> ExecutionHistoryManager.add_entry("agent_output", response, source="coding")
   └─> Message appended to history.jsonl

5. Auto-save triggered (5 messages OR 2 minutes)
   └─> Session.last_accessed updated
   └─> SQLite sessions/agents tables updated
   └─> history.jsonl flushed to disk
```

---

## Data Validation & Integrity

### Constraints Enforcement

**Database Level** (SQLite):
- Primary keys prevent duplicate sessions/agents
- Foreign keys ensure referential integrity (CASCADE on delete)
- Unique constraints on session names
- NOT NULL constraints on required fields

**Application Level** (Python):
- Pydantic models for runtime validation (optional dependency)
- Validation functions for business rules (regex patterns, file existence)
- Graceful error handling with user-friendly messages

**Example Validation Layer**:
```python
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class AgentConfig:
    """Agent configuration with validation."""
    session_id: str
    name: str
    model_name: str
    description: str = ""

    def __post_init__(self):
        if not validate_agent_name(self.name):
            raise ValueError(f"Invalid agent name: {self.name}")
        if not validate_model_string(self.model_name):
            raise ValueError(f"Invalid model string: {self.model_name}")
        if not validate_description(self.description):
            raise ValueError("Description too long (max 256 chars)")
```

---

## Migration Strategy

### Initial Schema (v1.0)

```sql
-- Version tracking
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at REAL NOT NULL
);

INSERT INTO schema_version (version, applied_at) VALUES (1, CURRENT_TIMESTAMP);

-- Sessions table
CREATE TABLE sessions (...);  -- As defined above

-- Agents table
CREATE TABLE agents (...);    -- As defined above
```

### Future Migrations

```python
# Example migration for adding plugin support
def migrate_v1_to_v2(conn: sqlite3.Connection):
    """Add plugins table for extension support (FR-EXT-001)."""
    conn.execute('''
        CREATE TABLE plugins (
            session_id TEXT NOT NULL,
            plugin_name TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            config_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (session_id, plugin_name),
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    ''')
    conn.execute("UPDATE schema_version SET version = 2")
```

---

## Performance Considerations

### Indexing Strategy

```sql
-- Session lookups by name (most common query)
CREATE INDEX idx_sessions_name ON sessions(name);

-- Session list ordered by last accessed
CREATE INDEX idx_sessions_last_accessed ON sessions(last_accessed DESC);

-- Agent lookups by session
CREATE INDEX idx_agents_session ON agents(session_id);
```

### Query Optimization

```python
# Efficient session list query (SC-003: <3s resume)
def list_sessions(limit: int = 100) -> list[Session]:
    """List recent sessions with agent counts."""
    query = """
        SELECT s.id, s.name, s.last_accessed,
               COUNT(a.name) as agent_count
        FROM sessions s
        LEFT JOIN agents a ON s.id = a.session_id
        GROUP BY s.id
        ORDER BY s.last_accessed DESC
        LIMIT ?
    """
    return execute_query(query, (limit,))
```

---

## Testing Strategy for Data Models

### Unit Tests

```python
# tests/cli/unit/test_session_models.py
def test_session_name_validation():
    """Test session name validation rules."""
    assert validate_session_name("my-project")
    assert validate_session_name("project_123")
    assert not validate_session_name("invalid name!")  # No spaces
    assert not validate_session_name("a" * 65)  # Too long

def test_agent_model_validation():
    """Test agent model string validation."""
    assert validate_model_string("openai/gpt-4")
    assert validate_model_string("anthropic/claude-3-opus-20240229")
    assert validate_model_string("ollama/llama2")
    assert not validate_model_string("gpt-4")  # Missing provider
```

### Integration Tests

```python
# tests/cli/integration/test_session_persistence.py
async def test_session_save_and_resume():
    """Test session persistence round-trip (SC-003)."""
    # Create session with agents and messages
    session = Session(name="test-session", working_directory=Path.cwd())
    session.create_agent("coding", "openai/gpt-4")
    session.add_message(Message(role="user", content="Hello"))

    # Save session
    start = time.perf_counter()
    await session_manager.save(session)
    save_time = time.perf_counter() - start
    assert save_time < 2.0, "Session save exceeds 2s (SC-003)"

    # Resume session
    start = time.perf_counter()
    resumed = await session_manager.resume("test-session")
    resume_time = time.perf_counter() - start
    assert resume_time < 3.0, "Session resume exceeds 3s (SC-003)"

    # Verify data integrity
    assert resumed.name == "test-session"
    assert len(resumed.agents) == 1
    assert resumed.agents[0].name == "coding"
    assert len(resumed.messages) == 1
```

---

**Data Model Phase Complete. Ready for contracts generation.**
