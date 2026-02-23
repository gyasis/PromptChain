"""Task list model for internal agent task tracking.

This module provides a task list that agents can use to:
1. Break down complex queries into discrete tasks
2. Track each task's status (pending, in_progress, completed)
3. Follow through until ALL tasks are complete
4. Report progress to the user

Similar to Claude Code's TodoWrite functionality.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json

from promptchain.observability import track_task


class TaskItemStatus(Enum):
    """Status of a task item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class TaskItem:
    """A single task item in the task list.

    Attributes:
        content: The task description (imperative form, e.g., "Fix the bug")
        active_form: Present continuous form (e.g., "Fixing the bug")
        status: Current status of the task
        created_at: When the task was created
        updated_at: When the task was last updated
        result: Optional result or notes from completing the task
    """
    content: str
    active_form: str
    status: TaskItemStatus = TaskItemStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    result: Optional[str] = None

    def mark_in_progress(self) -> None:
        """Mark task as in progress."""
        self.status = TaskItemStatus.IN_PROGRESS
        self.updated_at = datetime.now().isoformat()

    def mark_completed(self, result: Optional[str] = None) -> None:
        """Mark task as completed with optional result."""
        self.status = TaskItemStatus.COMPLETED
        self.result = result
        self.updated_at = datetime.now().isoformat()

    def mark_skipped(self, reason: Optional[str] = None) -> None:
        """Mark task as skipped with optional reason."""
        self.status = TaskItemStatus.SKIPPED
        self.result = reason
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "active_form": self.active_form,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskItem":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            active_form=data["active_form"],
            status=TaskItemStatus(data.get("status", "pending")),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            result=data.get("result"),
        )


@dataclass
class TaskList:
    """A task list for tracking agent progress.

    Attributes:
        tasks: List of task items
        objective: The overall objective/query being addressed
        created_at: When the task list was created
    """
    tasks: List[TaskItem] = field(default_factory=list)
    objective: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def current_task(self) -> Optional[TaskItem]:
        """Get the current in-progress task."""
        for task in self.tasks:
            if task.status == TaskItemStatus.IN_PROGRESS:
                return task
        return None

    @property
    def next_pending(self) -> Optional[TaskItem]:
        """Get the next pending task."""
        for task in self.tasks:
            if task.status == TaskItemStatus.PENDING:
                return task
        return None

    @property
    def completed_count(self) -> int:
        """Count of completed tasks."""
        return sum(1 for t in self.tasks if t.status == TaskItemStatus.COMPLETED)

    @property
    def total_count(self) -> int:
        """Total number of tasks."""
        return len(self.tasks)

    @property
    def progress_percentage(self) -> float:
        """Progress as percentage (0-100)."""
        if not self.tasks:
            return 0.0
        return (self.completed_count / self.total_count) * 100

    @property
    def is_complete(self) -> bool:
        """Check if all tasks are completed or skipped."""
        return all(
            t.status in (TaskItemStatus.COMPLETED, TaskItemStatus.SKIPPED)
            for t in self.tasks
        )

    @track_task(operation_type="CREATE")
    def add_task(self, content: str, active_form: str) -> TaskItem:
        """Add a new task to the list."""
        task = TaskItem(content=content, active_form=active_form)
        self.tasks.append(task)
        return task

    def update_tasks(self, tasks_data: List[Dict[str, Any]]) -> None:
        """Replace entire task list (used by agent tool)."""
        self.tasks = [TaskItem.from_dict(t) for t in tasks_data]

    def format_display(self) -> str:
        """Format task list for CLI display (grayscale)."""
        if not self.tasks:
            return "[dim]No tasks[/dim]"

        lines = []
        lines.append(f"[bold]Tasks ({self.completed_count}/{self.total_count})[/bold]")

        for i, task in enumerate(self.tasks, 1):
            if task.status == TaskItemStatus.COMPLETED:
                prefix = "[bold]+[/bold]"
                text = f"[dim]{task.content}[/dim]"
            elif task.status == TaskItemStatus.IN_PROGRESS:
                prefix = "[bold]>[/bold]"
                text = f"[bold]{task.active_form}[/bold]"
            elif task.status == TaskItemStatus.SKIPPED:
                prefix = "[dim]-[/dim]"
                text = f"[dim]{task.content} (skipped)[/dim]"
            else:  # PENDING
                prefix = "[dim]o[/dim]"
                text = f"{task.content}"

            lines.append(f"  {prefix} {text}")

        # Progress bar (text-based, grayscale)
        progress = int(self.progress_percentage / 10)
        bar = "[" + "=" * progress + "." * (10 - progress) + "]"
        lines.append(f"\n  {bar} {self.progress_percentage:.0f}%")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tasks": [t.to_dict() for t in self.tasks],
            "objective": self.objective,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskList":
        """Create from dictionary."""
        task_list = cls(
            objective=data.get("objective", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )
        task_list.tasks = [TaskItem.from_dict(t) for t in data.get("tasks", [])]
        return task_list

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TaskList":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class TaskListManager:
    """Manages task lists for agent sessions.

    This manager provides the interface for agents to:
    1. Create and update task lists
    2. Track progress across multiple queries
    3. Persist task state in the session
    """

    def __init__(self):
        """Initialize task list manager."""
        self.current_list: Optional[TaskList] = None
        self._history: List[TaskList] = []

    @track_task(operation_type="CREATE")
    def create_list(self, objective: str, tasks: List[Dict[str, str]]) -> TaskList:
        """Create a new task list.

        Args:
            objective: The overall objective being addressed
            tasks: List of task definitions with 'content' and 'active_form' keys

        Returns:
            Created TaskList
        """
        # Archive current list if exists
        if self.current_list:
            self._history.append(self.current_list)

        self.current_list = TaskList(objective=objective)
        for task_def in tasks:
            self.current_list.add_task(
                content=task_def.get("content", ""),
                active_form=task_def.get("active_form", task_def.get("content", ""))
            )
        return self.current_list

    @track_task(operation_type="UPDATE")
    def update_list(self, tasks: List[Dict[str, Any]]) -> TaskList:
        """Update current task list with new task states.

        Args:
            tasks: List of task data with status updates

        Returns:
            Updated TaskList
        """
        if not self.current_list:
            self.current_list = TaskList()

        self.current_list.update_tasks(tasks)
        return self.current_list

    @track_task(operation_type="STATE_CHANGE")
    def mark_task_in_progress(self, index: int) -> bool:
        """Mark a task as in progress by index.

        Args:
            index: Zero-based index of the task

        Returns:
            True if successful, False otherwise
        """
        if not self.current_list or index >= len(self.current_list.tasks):
            return False
        self.current_list.tasks[index].mark_in_progress()
        return True

    @track_task(operation_type="STATE_CHANGE")
    def mark_task_completed(self, index: int, result: Optional[str] = None) -> bool:
        """Mark a task as completed by index.

        Args:
            index: Zero-based index of the task
            result: Optional result or notes

        Returns:
            True if successful, False otherwise
        """
        if not self.current_list or index >= len(self.current_list.tasks):
            return False
        self.current_list.tasks[index].mark_completed(result)
        return True

    def get_display(self) -> str:
        """Get formatted display of current task list."""
        if not self.current_list:
            return ""
        return self.current_list.format_display()

    def is_complete(self) -> bool:
        """Check if current task list is complete."""
        if not self.current_list:
            return True
        return self.current_list.is_complete

    def clear(self) -> None:
        """Clear current task list."""
        if self.current_list:
            self._history.append(self.current_list)
        self.current_list = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize current state."""
        return {
            "current_list": self.current_list.to_dict() if self.current_list else None,
            "history_count": len(self._history),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskListManager":
        """Create from dictionary."""
        manager = cls()
        if data.get("current_list"):
            manager.current_list = TaskList.from_dict(data["current_list"])
        return manager
