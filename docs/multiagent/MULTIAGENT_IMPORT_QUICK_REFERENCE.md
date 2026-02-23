# Multi-Agent Communication Import Quick Reference

## Running Import Validation

```bash
# Validate all multi-agent communication imports
python -m pytest tests/cli/unit/test_multiagent_imports.py -v

# Expected: 23 passed, 2 warnings (Pydantic deprecations from existing codebase)
```

## Import Cheat Sheet

### Models

```python
# Task Delegation (FR-006 to FR-010)
from promptchain.cli.models.task import Task, TaskPriority, TaskStatus

# Blackboard Collaboration (FR-011 to FR-015)
from promptchain.cli.models.blackboard import BlackboardEntry

# Mental Models (US7 Integration)
from promptchain.cli.models.mental_models import (
    SpecializationType,      # Enum of agent types
    AgentSpecialization,     # Single specialization
    MentalModel,             # Complete mental model
    MentalModelManager,      # Manager class
    create_default_model     # Factory function
)

# Workflow State (FR-021 to FR-025)
from promptchain.cli.models.workflow import WorkflowState, WorkflowStage
```

### Communication

```python
# Message Handlers (FR-016 to FR-020)
from promptchain.cli.communication.handlers import (
    MessageType,                 # request, response, broadcast, etc.
    cli_communication_handler,   # Main handler function
    HandlerRegistry              # Handler registration system
)

# Message Bus
from promptchain.cli.communication.message_bus import Message, MessageBus
```

### Tool Registry

```python
# Core Registry System
from promptchain.cli.tools.registry import (
    ToolRegistry,       # Main registry class
    ToolCategory,       # Enum of categories
    ToolMetadata,       # Tool specification
    ParameterSchema     # Parameter definitions
)

# Create and use registry
registry = ToolRegistry()

# Register a tool
@registry.register(
    category="collaboration",
    description="Delegate a task to another agent",
    parameters={
        "task": {"type": "string", "required": True, "description": "Task description"}
    },
    capabilities=["task_delegation"]
)
def my_tool(task: str) -> str:
    return f"Delegated: {task}"
```

### Delegation Tools

```python
# Import from library package (NO _tool suffix)
from promptchain.cli.tools.library import (
    delegate_task,           # Delegate task to another agent
    get_pending_tasks,       # Get tasks assigned to this agent
    update_task_status,      # Update task status
    set_delegation_session_manager  # Set session manager (internal)
)

# Usage example
result = delegate_task(
    assignee="analyst-agent",
    description="Analyze the data",
    priority="high"
)
```

### Blackboard Tools

```python
# Import from library package (NO _tool suffix)
from promptchain.cli.tools.library import (
    write_to_blackboard,     # Write data to blackboard
    read_from_blackboard,    # Read data from blackboard
    list_blackboard_keys,    # List available keys
    delete_blackboard_entry, # Delete entry
    set_blackboard_session_manager  # Set session manager (internal)
)

# Usage example
write_to_blackboard(
    key="analysis_results",
    value={"score": 0.95, "confidence": "high"}
)
```

### Mental Model Tools

```python
# Import from library package (WITH _tool suffix)
from promptchain.cli.tools.library import (
    get_my_capabilities_tool,      # Query own capabilities
    discover_capable_agents_tool,  # Find agents with specific capabilities
    update_specialization_tool,    # Update specialization
    record_task_experience_tool,   # Record task completion
    share_capabilities_tool        # Share capabilities with other agents
)

# Usage example
capabilities = get_my_capabilities_tool()
# Returns: {"specializations": [...], "competencies": {...}}
```

### Command Handler Integration

```python
# Import command handler
from promptchain.cli.command_handler import CommandHandler

# Import session manager
from promptchain.cli.session_manager import SessionManager

# Create session manager
session_manager = SessionManager(sessions_dir="/path/to/sessions")

# Set session manager for tools (required for tool functionality)
from promptchain.cli.tools.library import (
    set_delegation_session_manager,
    set_blackboard_session_manager
)
set_delegation_session_manager(session_manager)
set_blackboard_session_manager(session_manager)
```

### CLI Entry Points

```python
# Import CLI main module
from promptchain.cli import main

# Import TUI application
from promptchain.cli.tui.app import PromptChainApp

# Create TUI app
app = PromptChainApp(
    session_name="my-session",
    sessions_dir="/path/to/sessions"
)
```

## Database Schema Integration

### Required Tables

```sql
-- Task delegation
CREATE TABLE IF NOT EXISTS task_queue (...);

-- Shared data storage
CREATE TABLE IF NOT EXISTS blackboard (...);

-- Agent metadata (includes mental models in metadata_json column)
CREATE TABLE IF NOT EXISTS agents (...);

-- Workflow tracking
CREATE TABLE IF NOT EXISTS workflow_state (...);

-- Agent communication logs
CREATE TABLE IF NOT EXISTS message_log (...);
```

### Accessing Database

```python
# Through SessionManager
session_manager = SessionManager(sessions_dir="/path/to/sessions")
conn = session_manager.get_connection(session_id="session-123")

# Query tasks
cursor = conn.execute("SELECT * FROM task_queue WHERE session_id = ?", (session_id,))
tasks = cursor.fetchall()
```

## Tool Naming Conventions

**IMPORTANT**: Tool naming varies by module:

1. **Delegation tools**: NO `_tool` suffix
   - `delegate_task`, `get_pending_tasks`, `update_task_status`

2. **Blackboard tools**: NO `_tool` suffix
   - `write_to_blackboard`, `read_from_blackboard`, `list_blackboard_keys`

3. **Mental model tools**: WITH `_tool` suffix
   - `get_my_capabilities_tool`, `discover_capable_agents_tool`

## Avoiding Circular Imports

The module structure prevents circular imports:

```python
# Good: Import models independently
from promptchain.cli.models import task
from promptchain.cli.models import blackboard
from promptchain.cli.models import mental_models

# Good: Import tools independently
from promptchain.cli.tools.library import delegation_tools
from promptchain.cli.tools.library import blackboard_tools

# Good: Import communication independently
from promptchain.cli.communication import handlers
from promptchain.cli.communication import message_bus
```

## Test-Driven Validation

Before using any component, verify imports work:

```python
import pytest

def test_my_component():
    """Verify my component imports correctly."""
    from promptchain.cli.models.task import Task

    task = Task(
        task_id="test-1",
        assignee="test-agent",
        description="Test task"
    )
    assert task is not None
```

## Troubleshooting

### Import Errors

If you get import errors:

1. **Check naming**: Mental model tools have `_tool` suffix, others don't
2. **Verify path**: Use `promptchain.cli.tools.library`, not `.library.mental_model_tools`
3. **Check dependencies**: Some imports may fail gracefully if dependencies missing

### Missing Session Manager

Tools require session manager to be set:

```python
from promptchain.cli.tools.library import set_delegation_session_manager
from promptchain.cli.session_manager import SessionManager

session_manager = SessionManager(sessions_dir="/path/to/sessions")
set_delegation_session_manager(session_manager)
```

### Database Errors

If database operations fail:

1. **Check schema**: Ensure all tables exist
2. **Verify permissions**: Check file permissions on sessions directory
3. **Check session ID**: Ensure valid session ID is provided

## Quick Start Example

```python
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.tools.library import (
    delegate_task,
    write_to_blackboard,
    get_my_capabilities_tool,
    set_delegation_session_manager,
    set_blackboard_session_manager
)

# Initialize session manager
session_manager = SessionManager(sessions_dir="/tmp/sessions")
session_id = session_manager.create_session("test-session")

# Configure tools
set_delegation_session_manager(session_manager)
set_blackboard_session_manager(session_manager)

# Use tools
delegate_task(assignee="worker", description="Process data")
write_to_blackboard(key="status", value="processing")
capabilities = get_my_capabilities_tool()

print(f"Capabilities: {capabilities}")
```

## Additional Resources

- **Full Import Validation**: See `IMPORT_VALIDATION_SUMMARY.md`
- **Test Suite**: `tests/cli/unit/test_multiagent_imports.py`
- **Schema Definition**: `promptchain/cli/schema.sql`
- **Tool Documentation**: See individual tool module docstrings

## Validation Command

Always run before committing changes:

```bash
python -m pytest tests/cli/unit/test_multiagent_imports.py -v
```

Expected: **23 passed, 2 warnings** (warnings are from existing Pydantic deprecations)
