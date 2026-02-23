"""Agentic workflow control module for PromptChain CLI.

This module provides task completion detection, handoff logic,
and workflow control for agentic agents.
"""

from .task_controller import (
    TaskController,
    TaskControllerConfig,
    TaskState,
    TaskStatus,
    detect_task_status,
)

__all__ = [
    "TaskController",
    "TaskControllerConfig",
    "TaskState",
    "TaskStatus",
    "detect_task_status",
]
