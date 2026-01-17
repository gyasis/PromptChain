"""Task delegation tools for agent-to-agent task assignment.

This module provides tools for agents to delegate tasks to other agents,
query pending tasks, and update task status. Implements US2 - Task Delegation
Between Agents (T030-T036).

DESIGN PRINCIPLES:
- Session manager injection pattern (avoids circular imports)
- No self-delegation validation
- Priority-aware task creation
- Context preservation across delegation
- Comprehensive error handling

USAGE:
    from promptchain.cli.tools.library import delegation_tools
    delegation_tools.set_session_manager(session_manager)

    # Delegate task
    result = delegation_tools.delegate_task(
        description="Review code changes",
        target_agent="code_reviewer",
        source_agent="main_agent",
        priority="high"
    )
"""

from typing import Dict, Any, List, Optional
import json
import uuid
from promptchain.cli.tools import registry, ToolCategory
from promptchain.cli.models import Task, TaskStatus, TaskPriority

# Module-level session manager holder (injected at runtime)
_session_manager = None


def set_session_manager(sm):
    """Inject session manager dependency.

    Args:
        sm: SessionManager instance to use for task operations

    DESIGN NOTE:
    - Avoids circular import issues
    - Allows lazy initialization
    - Enables testing with mock session managers
    """
    global _session_manager
    _session_manager = sm


def get_session_manager():
    """Get the injected session manager.

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
    description="""Delegate a task to another agent for execution.

USE WHEN:
- Breaking down complex work into subtasks
- Assigning specialized work to capable agents
- Need another agent to perform specific operations

VALIDATION:
- target_agent cannot equal source_agent (no self-delegation)
- description must be non-empty
- priority defaults to "medium"

EXAMPLE:
    delegate_task(
        description="Analyze system logs for errors",
        target_agent="log_analyzer",
        source_agent="orchestrator",
        priority="high",
        context={"log_path": "/var/logs/app.log"}
    )

RETURNS:
    Success: "Task delegated: <task_id> to <target_agent> (priority: <priority>)"
    Error: "Error: <error_message>"
""",
    parameters={
        "description": {
            "type": "string",
            "required": True,
            "description": "Task description"
        },
        "target_agent": {
            "type": "string",
            "required": True,
            "description": "Agent to delegate to"
        },
        "source_agent": {
            "type": "string",
            "required": True,
            "description": "Agent delegating the task"
        },
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "default": "medium",
            "description": "Task priority"
        },
        "context": {
            "type": "object",
            "description": "Additional context data"
        }
    },
    tags=["task", "delegation", "agent", "workflow"],
    capabilities=["task_delegation", "agent_coordination"]
)
def delegate_task(
    description: str,
    target_agent: str,
    source_agent: str,
    priority: str = "medium",
    context: Optional[Dict] = None
) -> str:
    """Delegate a task to another agent.

    Args:
        description: Task description (non-empty)
        target_agent: Agent to assign task to
        source_agent: Agent creating the task
        priority: Task priority level (low/medium/high)
        context: Additional context data

    Returns:
        Success message with task ID, or error message

    VALIDATION:
    - No self-delegation (source != target)
    - Description must be non-empty
    - Priority must be valid enum value

    TOKEN COST: ~50 tokens (metadata only, no LLM call)
    """
    # Validate no self-delegation
    if target_agent == source_agent:
        return (
            f"Error: Cannot delegate task to self "
            f"(source and target are both '{source_agent}')"
        )

    # Validate description
    if not description.strip():
        return "Error: Task description cannot be empty"

    # Get session manager and create task
    try:
        sm = get_session_manager()
        task = sm.create_task(
            description=description,
            source_agent=source_agent,
            target_agent=target_agent,
            priority=TaskPriority(priority),
            context=context or {}
        )

        # Return success message with task preview
        task_id_preview = task.task_id[:8] if len(task.task_id) > 8 else task.task_id
        return (
            f"Task delegated: {task_id_preview}... to {target_agent} "
            f"(priority: {priority})"
        )

    except ValueError as e:
        # Invalid priority enum value
        return f"Error: Invalid priority '{priority}'. Must be low/medium/high."
    except Exception as e:
        # Unexpected error
        return f"Error delegating task: {str(e)}"


@registry.register(
    category=ToolCategory.AGENT,
    description="""Get pending tasks assigned to an agent.

USE WHEN:
- Agent needs to check its task queue
- Orchestrator wants to monitor delegated tasks
- Need to verify task assignment

RETURNS:
    List of pending tasks with ID, priority, and description preview

EXAMPLE OUTPUT:
    Pending tasks for 'code_reviewer' (2):
      - [high] a1b2c3d4...: Review authentication module changes...
      - [medium] e5f6g7h8...: Check test coverage for new features...
""",
    parameters={
        "agent_name": {
            "type": "string",
            "required": True,
            "description": "Agent to get tasks for"
        }
    },
    tags=["task", "query", "agent"],
    capabilities=["task_query", "agent_coordination"]
)
def get_pending_tasks(agent_name: str) -> str:
    """Get pending tasks assigned to an agent.

    Args:
        agent_name: Agent to query tasks for

    Returns:
        Formatted list of pending tasks, or "No pending tasks" message

    TOKEN COST: ~50-200 tokens (depends on task count)
    """
    try:
        sm = get_session_manager()
        tasks = sm.list_tasks(target_agent=agent_name, status=TaskStatus.PENDING)

        if not tasks:
            return f"No pending tasks for agent '{agent_name}'"

        # Format task list with priority and preview
        result = f"Pending tasks for '{agent_name}' ({len(tasks)}):\n"
        for task in tasks:
            task_id_preview = task.task_id[:8] if len(task.task_id) > 8 else task.task_id
            desc_preview = (
                task.description[:50] + "..."
                if len(task.description) > 50
                else task.description
            )
            result += f"  - [{task.priority.value}] {task_id_preview}...: {desc_preview}\n"

        return result.rstrip()

    except Exception as e:
        return f"Error retrieving tasks: {str(e)}"


@registry.register(
    category=ToolCategory.AGENT,
    description="""Update the status of a task.

USE WHEN:
- Agent starts working on a task (status: in_progress)
- Agent completes a task (status: completed, with result)
- Task execution fails (status: failed, with error_message)

VALIDATION:
- task_id must exist
- status must be valid enum (in_progress/completed/failed)
- completed status should include result
- failed status should include error_message

EXAMPLE:
    update_task_status(
        task_id="a1b2c3d4-5678-90ab-cdef-1234567890ab",
        status="completed",
        result={"files_reviewed": 5, "issues_found": 2}
    )
""",
    parameters={
        "task_id": {
            "type": "string",
            "required": True,
            "description": "Task ID to update"
        },
        "status": {
            "type": "string",
            "required": True,
            "enum": ["in_progress", "completed", "failed"],
            "description": "New status"
        },
        "result": {
            "type": "object",
            "description": "Task result (for completed status)"
        },
        "error_message": {
            "type": "string",
            "description": "Error message (for failed status)"
        }
    },
    tags=["task", "update", "status"],
    capabilities=["task_management", "agent_coordination"]
)
def update_task_status(
    task_id: str,
    status: str,
    result: Optional[Dict] = None,
    error_message: Optional[str] = None
) -> str:
    """Update the status of a task.

    Args:
        task_id: Task ID to update
        status: New status (in_progress/completed/failed)
        result: Task result data (optional, for completed tasks)
        error_message: Error description (optional, for failed tasks)

    Returns:
        Success message with status update, or error message

    VALIDATION:
    - Task must exist
    - Status must be valid enum
    - Completed tasks should have result
    - Failed tasks should have error_message

    TOKEN COST: ~50 tokens (metadata only, no LLM call)
    """
    try:
        sm = get_session_manager()
        task = sm.get_task(task_id)

        if not task:
            return f"Error: Task '{task_id}' not found"

        # Validate status value
        try:
            task_status = TaskStatus(status)
        except ValueError:
            return (
                f"Error: Invalid status '{status}'. "
                f"Must be in_progress/completed/failed."
            )

        # Update task status with appropriate metadata
        sm.update_task_status(
            task_id,
            task_status,
            result=result,
            error_message=error_message
        )

        # Return success message
        task_id_preview = task_id[:8] if len(task_id) > 8 else task_id
        return f"Task {task_id_preview}... status updated to '{status}'"

    except Exception as e:
        return f"Error updating task status: {str(e)}"


def request_help(
    session_manager: Any,
    help_request: str,
    required_capabilities: Optional[List[str]] = None,
    priority: str = "medium",
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Request help from capable agents when stuck on a task.

    Implements US6 Help Request Protocol:
    - Find agents with matching capabilities
    - Create high-priority help task delegated to capable agent
    - Falls back to broadcast if no matching capability found

    Args:
        session_manager: Session manager with multi-agent state
        help_request: Description of what help is needed
        required_capabilities: List of capabilities needed to help (e.g., ["code_analysis", "debugging"])
        priority: Task priority (low/medium/high) - defaults to high for help requests
        context: Additional context for the help request

    Returns:
        Dict with task_id, assigned_agent (or "broadcast"), and status
    """
    # Determine requesting agent
    requesting_agent = "default"
    if hasattr(session_manager, 'session') and session_manager.session:
        if hasattr(session_manager.session, 'current_agent'):
            requesting_agent = session_manager.session.current_agent or "default"

    # Find capable agents if capabilities specified
    assigned_agent = None
    if required_capabilities:
        capable_tools = registry.discover_capabilities(
            capability_filter=required_capabilities
        )
        # Get unique agents that own these tools
        capable_agents = set()
        for tool in capable_tools:
            if tool.allowed_agents:
                capable_agents.update(tool.allowed_agents)

        # Pick first capable agent that isn't the requester
        for agent in capable_agents:
            if agent != requesting_agent:
                assigned_agent = agent
                break

    # Create help task
    task_id = str(uuid.uuid4())
    task_priority = TaskPriority.HIGH  # Help requests default to high priority
    if priority == "low":
        task_priority = TaskPriority.LOW
    elif priority == "medium":
        task_priority = TaskPriority.MEDIUM

    task = Task.create(
        description=f"[HELP REQUEST] {help_request}",
        source_agent=requesting_agent,
        target_agent=assigned_agent or "broadcast",
        priority=task_priority,
        context={
            "type": "help_request",
            "required_capabilities": required_capabilities or [],
            "original_context": context or {}
        }
    )

    # Store in session if available
    if hasattr(session_manager, 'add_task'):
        session_manager.add_task(task)

    return {
        "task_id": task.task_id,
        "assigned_agent": assigned_agent or "broadcast",
        "is_broadcast": assigned_agent is None,
        "status": "pending",
        "message": f"Help request created and {'broadcast to all agents' if not assigned_agent else f'assigned to {assigned_agent}'}"
    }


@registry.register(
    category=ToolCategory.COLLABORATION,
    description="""Request help from capable agents when stuck on a task.

USE WHEN:
- Current agent lacks capability to complete a task
- Need specialized expertise from another agent
- Stuck on a problem and need assistance
- Want to find agent with specific capabilities

CAPABILITY MATCHING:
- If required_capabilities specified, finds agents with those capabilities
- If no match found, broadcasts help request to all agents
- Automatically excludes requesting agent from assignment

PRIORITY:
- Help requests default to "high" priority
- Can be overridden with priority parameter

EXAMPLE:
    request_help(
        help_request="Need help debugging authentication flow",
        required_capabilities=["debugging", "code_analysis"],
        priority="high",
        context={"file": "auth.py", "error": "Token validation failed"}
    )

RETURNS:
    JSON with task_id, assigned_agent, and broadcast status
""",
    parameters={
        "help_request": {
            "type": "string",
            "description": "Description of what help is needed",
            "required": True
        },
        "required_capabilities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of capabilities needed (e.g., ['debugging', 'code_analysis'])",
            "required": False
        },
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "default": "high",
            "description": "Task priority"
        },
        "context": {
            "type": "object",
            "description": "Additional context for the help request",
            "required": False
        }
    },
    tags=["help", "collaboration", "multi-agent", "delegation"],
    capabilities=["help_request", "agent_collaboration"]
)
def request_help_tool(
    help_request: str,
    required_capabilities: Optional[List[str]] = None,
    priority: str = "high",
    context: Optional[Dict[str, Any]] = None
) -> str:
    """Request help from capable agents - wrapper for session injection.

    Args:
        help_request: Description of what help is needed
        required_capabilities: List of capabilities needed
        priority: Task priority (low/medium/high)
        context: Additional context

    Returns:
        JSON string with task details

    TOKEN COST: ~100-200 tokens (capability discovery + task creation)
    """
    try:
        sm = get_session_manager()
        result = request_help(sm, help_request, required_capabilities, priority, context)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Error requesting help: {str(e)}",
            "status": "failed"
        }, indent=2)


# Public API
__all__ = [
    "delegate_task",
    "get_pending_tasks",
    "update_task_status",
    "request_help",
    "request_help_tool",
    "set_session_manager",
    "get_session_manager"
]
