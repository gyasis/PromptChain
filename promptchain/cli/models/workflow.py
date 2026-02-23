"""Workflow state models for multi-session objective tracking.

This module defines workflow state tracking for complex multi-session objectives
that span across multiple conversation turns and agent interactions.

FR-021 to FR-025: Workflow State Management
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
import uuid
import json


class WorkflowStage(str, Enum):
    """Workflow execution stages (FR-021)."""
    PLANNING = "planning"
    EXECUTION = "execution"
    REVIEW = "review"
    COMPLETE = "complete"


@dataclass
class WorkflowStep:
    """Represents a single step in a multi-session workflow.

    Workflow steps track the execution state of individual objectives within
    a larger workflow, including completion status, results, and error handling.

    Attributes:
        description: Human-readable step description
        status: Current execution status (pending/in_progress/completed/failed)
        agent_name: Name of agent that executed or will execute this step
        started_at: Timestamp when step execution began (Unix seconds)
        completed_at: Timestamp when step finished (Unix seconds)
        result: Step execution result or output
        error_message: Error details if step failed
        retry_count: Number of retry attempts for failed steps
    """

    description: str
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    agent_name: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    def mark_in_progress(self, agent_name: str):
        """Mark step as in-progress with assigned agent.

        Args:
            agent_name: Name of agent executing this step
        """
        self.status = "in_progress"
        self.agent_name = agent_name
        self.started_at = datetime.now().timestamp()

    def mark_completed(self, result: str):
        """Mark step as successfully completed.

        Args:
            result: Step execution result or output
        """
        self.status = "completed"
        self.result = result
        self.completed_at = datetime.now().timestamp()
        self.error_message = None

    def mark_failed(self, error: str):
        """Mark step as failed with error message.

        Args:
            error: Error message describing failure
        """
        self.status = "failed"
        self.error_message = error
        self.completed_at = datetime.now().timestamp()
        self.retry_count += 1

    def reset(self):
        """Reset step to pending state for retry."""
        self.status = "pending"
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error_message = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "description": self.description,
            "status": self.status,
            "agent_name": self.agent_name,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        """Create from dictionary.

        Args:
            data: Step data dictionary

        Returns:
            WorkflowStep: Reconstructed step object
        """
        return cls(
            description=data["description"],
            status=data.get("status", "pending"),
            agent_name=data.get("agent_name"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result=data.get("result"),
            error_message=data.get("error_message"),
            retry_count=data.get("retry_count", 0),
        )


@dataclass
class WorkflowState:
    """Tracks multi-session workflow execution state.

    WorkflowState manages complex objectives that span multiple conversation turns,
    tracking progress, step execution, and resumption state for interrupted workflows.

    Attributes:
        objective: High-level workflow objective description
        steps: List of workflow steps in execution order
        current_step_index: Index of currently executing step
        created_at: Workflow creation timestamp (Unix seconds)
        updated_at: Last update timestamp (Unix seconds)
        completed_at: Workflow completion timestamp (Unix seconds)
        metadata: Extensible storage for workflow-specific data
    """

    objective: str
    steps: List[WorkflowStep] = field(default_factory=list)
    current_step_index: int = 0
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate workflow state after initialization."""
        if not self.objective or len(self.objective) > 512:
            raise ValueError(f"Objective must be 1-512 characters, got: {len(self.objective)}")

        if self.current_step_index < 0 or self.current_step_index >= len(self.steps):
            if len(self.steps) > 0:  # Allow index 0 for empty steps list
                raise ValueError(
                    f"current_step_index {self.current_step_index} out of range for {len(self.steps)} steps"
                )

    @property
    def is_completed(self) -> bool:
        """Check if workflow is fully completed.

        Returns:
            bool: True if all steps are completed
        """
        return all(step.status == "completed" for step in self.steps) and len(self.steps) > 0

    @property
    def is_failed(self) -> bool:
        """Check if workflow has failed (any step in failed state).

        Returns:
            bool: True if any step has failed status
        """
        return any(step.status == "failed" for step in self.steps)

    @property
    def progress_percentage(self) -> float:
        """Calculate workflow completion percentage.

        Returns:
            float: Percentage of completed steps (0.0-100.0)
        """
        if len(self.steps) == 0:
            return 0.0
        completed = sum(1 for step in self.steps if step.status == "completed")
        return (completed / len(self.steps)) * 100.0

    @property
    def current_step(self) -> Optional[WorkflowStep]:
        """Get currently executing step.

        Returns:
            Optional[WorkflowStep]: Current step or None if workflow complete
        """
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def add_step(self, description: str, agent_name: Optional[str] = None) -> WorkflowStep:
        """Add new step to workflow.

        Args:
            description: Step description
            agent_name: Optional agent assignment

        Returns:
            WorkflowStep: Created step object
        """
        step = WorkflowStep(description=description, agent_name=agent_name)
        self.steps.append(step)
        self.updated_at = datetime.now().timestamp()
        return step

    def advance_step(self):
        """Move to next step in workflow."""
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            self.updated_at = datetime.now().timestamp()

    def mark_completed(self):
        """Mark entire workflow as completed."""
        self.completed_at = datetime.now().timestamp()
        self.updated_at = self.completed_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage.

        Returns:
            Dict[str, Any]: Workflow state as dictionary
        """
        return {
            "objective": self.objective,
            "steps": [step.to_dict() for step in self.steps],
            "current_step_index": self.current_step_index,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowState":
        """Create from dictionary.

        Args:
            data: Workflow state dictionary

        Returns:
            WorkflowState: Reconstructed workflow state
        """
        steps = [WorkflowStep.from_dict(step_data) for step_data in data.get("steps", [])]

        return cls(
            objective=data["objective"],
            steps=steps,
            current_step_index=data.get("current_step_index", 0),
            created_at=data.get("created_at", datetime.now().timestamp()),
            updated_at=data.get("updated_at", datetime.now().timestamp()),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """Human-readable workflow representation.

        Returns:
            str: Formatted workflow string
        """
        status = "COMPLETED" if self.is_completed else "FAILED" if self.is_failed else "IN PROGRESS"
        progress = f"{self.progress_percentage:.1f}%"
        current_info = f"Step {self.current_step_index + 1}/{len(self.steps)}" if self.steps else "No steps"

        return f"[{status}] {self.objective} - {progress} complete ({current_info})"


@dataclass
class MultiAgentWorkflow:
    """
    Multi-agent workflow state for FR-021 to FR-025.

    Extends basic WorkflowState with multi-agent specific features:
    - Workflow stages (planning, execution, review, complete)
    - Agent involvement tracking
    - Task completion tracking

    Attributes:
        workflow_id: Unique workflow identifier
        stage: Current workflow stage
        agents_involved: List of agents participating in workflow
        completed_tasks: List of completed task IDs
        current_task: Currently executing task description
        context: Additional workflow context
        started_at: Workflow start timestamp
        updated_at: Last update timestamp
    """

    workflow_id: str
    stage: WorkflowStage = WorkflowStage.PLANNING
    agents_involved: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())

    @classmethod
    def create(
        cls,
        agents: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> "MultiAgentWorkflow":
        """Create a new multi-agent workflow."""
        return cls(
            workflow_id=str(uuid.uuid4()),
            agents_involved=agents,
            context=context or {}
        )

    def add_agent(self, agent_name: str) -> None:
        """Add an agent to the workflow."""
        if agent_name not in self.agents_involved:
            self.agents_involved.append(agent_name)
            self.updated_at = datetime.now().timestamp()

    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed."""
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
            self.updated_at = datetime.now().timestamp()

    def set_current_task(self, task_description: Optional[str]) -> None:
        """Set the current task being executed."""
        self.current_task = task_description
        self.updated_at = datetime.now().timestamp()

    def advance_stage(self) -> None:
        """Advance to the next workflow stage."""
        stage_order = [
            WorkflowStage.PLANNING,
            WorkflowStage.EXECUTION,
            WorkflowStage.REVIEW,
            WorkflowStage.COMPLETE
        ]
        current_idx = stage_order.index(self.stage)
        if current_idx < len(stage_order) - 1:
            self.stage = stage_order[current_idx + 1]
            self.updated_at = datetime.now().timestamp()

    def set_stage(self, stage: WorkflowStage) -> None:
        """Set workflow stage directly."""
        self.stage = stage
        self.updated_at = datetime.now().timestamp()

    @property
    def is_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.stage == WorkflowStage.COMPLETE

    @property
    def progress(self) -> Dict[str, Any]:
        """Get workflow progress summary."""
        return {
            "stage": self.stage.value,
            "agents_count": len(self.agents_involved),
            "completed_tasks_count": len(self.completed_tasks),
            "current_task": self.current_task
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "stage": self.stage.value,
            "agents_involved": self.agents_involved,
            "completed_tasks": self.completed_tasks,
            "current_task": self.current_task,
            "context": self.context,
            "started_at": self.started_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiAgentWorkflow":
        """Create from dictionary."""
        return cls(
            workflow_id=data["workflow_id"],
            stage=WorkflowStage(data.get("stage", "planning")),
            agents_involved=data.get("agents_involved", []),
            completed_tasks=data.get("completed_tasks", []),
            current_task=data.get("current_task"),
            context=data.get("context", {}),
            started_at=data.get("started_at", datetime.now().timestamp()),
            updated_at=data.get("updated_at", datetime.now().timestamp())
        )

    @classmethod
    def from_db_row(cls, row: tuple) -> "MultiAgentWorkflow":
        """Create from database row (matches workflow_state table columns)."""
        # Columns: workflow_id, session_id, stage, agents_involved_json,
        #          completed_tasks_json, current_task, context_json, started_at, updated_at
        return cls(
            workflow_id=row[0],
            stage=WorkflowStage(row[2]),  # Skip session_id
            agents_involved=json.loads(row[3]) if row[3] else [],
            completed_tasks=json.loads(row[4]) if row[4] else [],
            current_task=row[5],
            context=json.loads(row[6]) if row[6] else {},
            started_at=row[7],
            updated_at=row[8]
        )

    def __repr__(self) -> str:
        return (
            f"MultiAgentWorkflow(id={self.workflow_id[:8]}..., "
            f"stage={self.stage.value}, "
            f"agents={len(self.agents_involved)}, "
            f"completed={len(self.completed_tasks)})"
        )
