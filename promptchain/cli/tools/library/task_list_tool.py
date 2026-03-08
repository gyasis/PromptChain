"""Task list tool for agent task tracking.

This tool allows agents to create and manage task lists for complex queries,
similar to Claude Code's TodoWrite functionality.
"""

import json
from typing import Any, Dict, List, Optional

from promptchain.observability import track_task

from ...models.task_list import TaskList, TaskListManager

# Global task list manager (shared across session)
_task_list_manager: Optional[TaskListManager] = None


def get_task_list_manager() -> TaskListManager:
    """Get or create the global task list manager."""
    global _task_list_manager
    if _task_list_manager is None:
        _task_list_manager = TaskListManager()
    return _task_list_manager


def set_task_list_manager(manager: TaskListManager) -> None:
    """Set the global task list manager (for session restoration)."""
    global _task_list_manager
    _task_list_manager = manager


@track_task(operation_type="UPDATE")
def task_list_write(todos_json: str) -> str:
    """Create or update the task list for tracking progress on complex queries.

    Use this tool when working on tasks that require multiple steps.
    Each task should have:
    - content: What needs to be done (imperative form, e.g., "Fix the bug")
    - status: "pending", "in_progress", or "completed"
    - active_form: Present continuous form (e.g., "Fixing the bug")

    IMPORTANT:
    - Only ONE task should be "in_progress" at a time
    - Mark tasks "completed" IMMEDIATELY after finishing them
    - Always provide both "content" and "activeForm" for each task

    Args:
        todos_json: JSON string containing array of task objects with keys:
            - content (str): Task description in imperative form
            - status (str): "pending", "in_progress", or "completed"
            - activeForm (str): Task in present continuous form

    Returns:
        Confirmation message with current task list display

    Example:
        task_list_write('[
            {"content": "Search for documentation", "status": "completed", "activeForm": "Searching for documentation"},
            {"content": "Analyze search results", "status": "in_progress", "activeForm": "Analyzing search results"},
            {"content": "Summarize findings", "status": "pending", "activeForm": "Summarizing findings"}
        ]')
    """
    manager = get_task_list_manager()

    try:
        tasks_data = json.loads(todos_json)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON - {e}"

    if not isinstance(tasks_data, list):
        return "Error: Expected JSON array of tasks"

    # Convert to internal format
    formatted_tasks = []
    for task in tasks_data:
        if not isinstance(task, dict):
            continue

        formatted_tasks.append(
            {
                "content": task.get("content", ""),
                "active_form": task.get(
                    "activeForm", task.get("active_form", task.get("content", ""))
                ),
                "status": task.get("status", "pending"),
                "result": task.get("result"),
            }
        )

    # Update the task list
    manager.update_list(formatted_tasks)

    # Return display
    display = manager.get_display()
    return f"Task list updated.\n\n{display}"


def task_list_get() -> str:
    """Get the current task list display.

    Returns:
        Formatted task list display
    """
    manager = get_task_list_manager()
    if not manager.current_list:
        return "No active task list."
    return manager.get_display()


def task_list_clear() -> str:
    """Clear the current task list.

    Returns:
        Confirmation message
    """
    manager = get_task_list_manager()
    manager.clear()
    return "Task list cleared."


# Tool schema for registration
TASK_LIST_WRITE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "task_list_write",
        "description": """Create or update a task list for tracking progress on complex multi-step tasks.

Use this tool when:
1. A task requires 3 or more distinct steps
2. You need to track progress on a complex query
3. The user provides multiple tasks to complete
4. You want to show the user your progress

IMPORTANT RULES:
- Only ONE task should be "in_progress" at a time
- Mark tasks as "completed" IMMEDIATELY after finishing
- Always provide both "content" and "activeForm" for each task
- Keep the list updated as you work

Task status values:
- "pending": Task not yet started
- "in_progress": Currently working on this task (max ONE at a time)
- "completed": Task finished successfully

Example usage:
[
    {"content": "Search codebase for function", "status": "completed", "activeForm": "Searching codebase"},
    {"content": "Analyze search results", "status": "in_progress", "activeForm": "Analyzing results"},
    {"content": "Suggest improvements", "status": "pending", "activeForm": "Suggesting improvements"}
]""",
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The updated todo list",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Task description in imperative form (e.g., 'Fix the bug')",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Current status of the task",
                            },
                            "activeForm": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Task in present continuous form (e.g., 'Fixing the bug')",
                            },
                        },
                        "required": ["content", "status", "activeForm"],
                    },
                }
            },
            "required": ["todos"],
        },
    },
}


def register_task_list_tools() -> List[Dict[str, Any]]:
    """Get task list tool schemas for registration."""
    return [TASK_LIST_WRITE_SCHEMA]


def get_task_list_functions() -> Dict[str, Any]:
    """Get task list functions for tool execution."""
    return {
        "task_list_write": lambda todos: task_list_write(json.dumps(todos)),
    }
