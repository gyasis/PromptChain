"""
Blackboard Tools - Shared data space for inter-agent communication

This module implements US3 - Blackboard Data Sharing (T041-T047).
Provides tools for agents to read/write shared data asynchronously.

The blackboard pattern enables:
- Asynchronous data sharing between agents
- Persistent state across agent invocations
- Version tracking for concurrent modifications
- Audit trail of who wrote what

Each entry is versioned and tracks the writing agent for accountability.
"""

from typing import Any, Optional
import json

from promptchain.cli.tools import registry, ToolCategory

# Module-level session manager holder
# This is set by the CLI initialization code
_session_manager = None


def set_session_manager(sm) -> None:
    """
    Set the session manager instance for this module.

    Must be called during CLI initialization before any tools are used.

    Args:
        sm: SessionManager instance with blackboard support
    """
    global _session_manager
    _session_manager = sm


def get_session_manager():
    """
    Get the current session manager instance.

    Returns:
        SessionManager instance

    Raises:
        RuntimeError: If session manager not initialized
    """
    if _session_manager is None:
        raise RuntimeError(
            "Session manager not initialized. Call set_session_manager first."
        )
    return _session_manager


@registry.register(
    category=ToolCategory.AGENT,
    description="""Write a key-value entry to the shared blackboard.

USE WHEN:
- Sharing data between agents asynchronously
- Storing intermediate results for other agents
- Publishing state that multiple agents need access to

BEHAVIOR:
- If key exists, value is updated (upsert) with version increment
- Value can be any JSON-serializable data
- written_by tracks which agent last modified the entry

EXAMPLES:
- write_to_blackboard("analysis_results", {"score": 0.95}, "analyzer_agent")
- write_to_blackboard("current_step", 3, "orchestrator")
- write_to_blackboard("shared_context", ["item1", "item2"], "coordinator")
""",
    parameters={
        "key": {
            "type": "string",
            "required": True,
            "description": "Unique key for the entry"
        },
        "value": {
            "type": "object",
            "required": True,
            "description": "Value to store (JSON-serializable: dict, list, str, number, bool)"
        },
        "written_by": {
            "type": "string",
            "required": True,
            "description": "Agent name writing the entry"
        }
    },
    tags=["blackboard", "write", "share", "data"],
    capabilities=["blackboard_write", "data_sharing"]
)
def write_to_blackboard(key: str, value: Any, written_by: str) -> str:
    """
    Write a key-value entry to the shared blackboard.

    Args:
        key: Unique key for the entry
        value: Any JSON-serializable value
        written_by: Agent name writing the entry

    Returns:
        Success message with key, writer, and version

    Raises:
        RuntimeError: If session manager not initialized
    """
    if not key.strip():
        return "Error: Key cannot be empty"

    if not written_by.strip():
        return "Error: written_by cannot be empty"

    sm = get_session_manager()
    entry = sm.write_blackboard(key=key, value=value, written_by=written_by)

    return f"Blackboard entry '{key}' written by {written_by} (v{entry.version})"


@registry.register(
    category=ToolCategory.AGENT,
    description="""Read a value from the shared blackboard by key.

USE WHEN:
- Retrieving data shared by another agent
- Checking current state of shared variables
- Getting intermediate results from other agents

RETURNS:
- Entry value with metadata (version, writer)
- Error message if key not found

EXAMPLES:
- read_from_blackboard("analysis_results")
- read_from_blackboard("current_step")
- read_from_blackboard("shared_context")
""",
    parameters={
        "key": {
            "type": "string",
            "required": True,
            "description": "Key to read"
        }
    },
    tags=["blackboard", "read", "share", "data"],
    capabilities=["blackboard_read", "data_sharing"]
)
def read_from_blackboard(key: str) -> str:
    """
    Read a value from the shared blackboard.

    Args:
        key: Key to read

    Returns:
        Formatted string with entry value and metadata,
        or error message if key not found

    Raises:
        RuntimeError: If session manager not initialized
    """
    sm = get_session_manager()
    entry = sm.read_blackboard(key)

    if entry is None:
        return f"No entry found for key '{key}'"

    # Format value nicely for dicts/lists, otherwise use str()
    if isinstance(entry.value, (dict, list)):
        value_str = json.dumps(entry.value, indent=2)
    else:
        value_str = str(entry.value)

    return (
        f"Blackboard['{key}'] (v{entry.version}, by {entry.written_by}):\n"
        f"{value_str}"
    )


@registry.register(
    category=ToolCategory.AGENT,
    description="""List all keys currently on the blackboard.

USE WHEN:
- Discovering what data is available
- Checking if specific keys exist
- Auditing shared state

RETURNS:
- Comma-separated list of all keys
- Message if blackboard is empty

EXAMPLES:
- list_blackboard_keys() -> "analysis_results, current_step, shared_context"
""",
    parameters={},
    tags=["blackboard", "list", "query"],
    capabilities=["blackboard_read", "data_sharing"]
)
def list_blackboard_keys() -> str:
    """
    List all keys currently on the blackboard.

    Returns:
        Comma-separated list of keys or empty message

    Raises:
        RuntimeError: If session manager not initialized
    """
    sm = get_session_manager()
    keys = sm.list_blackboard_keys()

    if not keys:
        return "Blackboard is empty"

    return f"Blackboard keys ({len(keys)}): {', '.join(sorted(keys))}"


@registry.register(
    category=ToolCategory.AGENT,
    description="""Delete an entry from the blackboard.

USE WHEN:
- Cleaning up temporary data
- Removing obsolete state
- Resetting shared variables

BEHAVIOR:
- Permanently removes the entry
- Returns success/failure message
- Safe to call on non-existent keys

EXAMPLES:
- delete_blackboard_entry("temp_results")
- delete_blackboard_entry("old_context")
""",
    parameters={
        "key": {
            "type": "string",
            "required": True,
            "description": "Key to delete"
        }
    },
    tags=["blackboard", "delete", "cleanup"],
    capabilities=["blackboard_write", "data_sharing"]
)
def delete_blackboard_entry(key: str) -> str:
    """
    Delete an entry from the blackboard.

    Args:
        key: Key to delete

    Returns:
        Success message if deleted, or not found message

    Raises:
        RuntimeError: If session manager not initialized
    """
    sm = get_session_manager()
    deleted = sm.delete_blackboard_entry(key)

    if deleted:
        return f"Blackboard entry '{key}' deleted"

    return f"No entry found for key '{key}'"


__all__ = [
    "write_to_blackboard",
    "read_from_blackboard",
    "list_blackboard_keys",
    "delete_blackboard_entry",
    "set_session_manager",
    "get_session_manager"
]
