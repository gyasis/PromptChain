"""
Task Model for Agent Delegation.

Represents a delegated unit of work between agents.

FR-006 to FR-010: Task Delegation Protocol
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import uuid
import json


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """
    Represents a delegated task between agents.

    Attributes:
        task_id: Unique identifier
        description: Task description
        source_agent: Agent that delegated the task
        target_agent: Agent assigned to execute the task
        priority: Task priority level
        status: Current execution status
        context: Additional context data
        created_at: Task creation timestamp
        started_at: When task execution started
        completed_at: When task completed or failed
        result: Task execution result
        error_message: Error message if failed
    """

    task_id: str
    description: str
    source_agent: str
    target_agent: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    @classmethod
    def create(
        cls,
        description: str,
        source_agent: str,
        target_agent: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ) -> "Task":
        """Create a new task with auto-generated ID."""
        return cls(
            task_id=str(uuid.uuid4()),
            description=description,
            source_agent=source_agent,
            target_agent=target_agent,
            priority=priority,
            context=context or {}
        )

    def start(self) -> None:
        """Mark task as in progress."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now().timestamp()

    def complete(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now().timestamp()
        self.result = result

    def fail(self, error_message: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now().timestamp()
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "priority": self.priority.value,
            "status": self.status.value,
            "context": self.context,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error_message": self.error_message
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            description=data["description"],
            source_agent=data["source_agent"],
            target_agent=data["target_agent"],
            priority=TaskPriority(data.get("priority", "medium")),
            status=TaskStatus(data.get("status", "pending")),
            context=data.get("context", {}),
            created_at=data.get("created_at", datetime.now().timestamp()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result=data.get("result"),
            error_message=data.get("error_message")
        )

    @classmethod
    def from_db_row(cls, row: tuple) -> "Task":
        """Create from database row (matches task_queue table columns)."""
        return cls(
            task_id=row[0],
            description=row[2],  # Skip session_id
            source_agent=row[3],
            target_agent=row[4],
            priority=TaskPriority(row[5]),
            status=TaskStatus(row[6]),
            context=json.loads(row[7]) if row[7] else {},
            created_at=row[8],
            started_at=row[9],
            completed_at=row[10],
            result=json.loads(row[11]) if row[11] else None,
            error_message=row[12]
        )

    def __repr__(self) -> str:
        return (
            f"Task(id={self.task_id[:8]}..., "
            f"desc='{self.description[:30]}...', "
            f"status={self.status.value}, "
            f"priority={self.priority.value})"
        )
