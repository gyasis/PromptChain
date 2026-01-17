"""
Library tool registration package for PromptChain CLI.

This package provides CLI tool registry integration for PromptChain's
core library tools (FileOperations, RipgrepSearcher, TerminalTool) and
multi-agent coordination tools (delegation, blackboard, mental models).

Auto-registers 19+ tools when imported:
- 12 file operation tools
- 1 code search tool (ripgrep)
- 1 terminal execution tool
- 4 delegation tools (delegate_task, get_pending_tasks, update_task_status, set_session_manager)
- 5 blackboard tools (write_to_blackboard, read_from_blackboard, list_blackboard_keys, delete_blackboard_entry, set_session_manager)
- 5 mental model tools (get_my_capabilities, discover_capable_agents, update_specialization, record_task_experience, share_capabilities)
"""

# Import and auto-register CLI tools
# This triggers @registry.register() decorators when package is imported
try:
    from . import registration  # noqa: F401
except ImportError:
    # Graceful fallback if dependencies are not available
    pass

# Import delegation tools
try:
    from .delegation_tools import (
        delegate_task,
        get_pending_tasks,
        update_task_status,
        set_session_manager as set_delegation_session_manager,
    )
except ImportError:
    delegate_task = None
    get_pending_tasks = None
    update_task_status = None
    set_delegation_session_manager = None

# Import blackboard tools
try:
    from .blackboard_tools import (
        write_to_blackboard,
        read_from_blackboard,
        list_blackboard_keys,
        delete_blackboard_entry,
        set_session_manager as set_blackboard_session_manager,
    )
except ImportError:
    write_to_blackboard = None
    read_from_blackboard = None
    list_blackboard_keys = None
    delete_blackboard_entry = None
    set_blackboard_session_manager = None

# Import mental model tools
try:
    from .mental_model_tools import (
        get_my_capabilities_tool,
        discover_capable_agents_tool,
        update_specialization_tool,
        record_task_experience_tool,
        share_capabilities_tool,
    )
except ImportError:
    get_my_capabilities_tool = None
    discover_capable_agents_tool = None
    update_specialization_tool = None
    record_task_experience_tool = None
    share_capabilities_tool = None

__all__ = [
    "registration",
    # Delegation tools
    "delegate_task",
    "get_pending_tasks",
    "update_task_status",
    "set_delegation_session_manager",
    # Blackboard tools
    "write_to_blackboard",
    "read_from_blackboard",
    "list_blackboard_keys",
    "delete_blackboard_entry",
    "set_blackboard_session_manager",
    # Mental model tools
    "get_my_capabilities_tool",
    "discover_capable_agents_tool",
    "update_specialization_tool",
    "record_task_experience_tool",
    "share_capabilities_tool",
]
__version__ = "0.1.0"
