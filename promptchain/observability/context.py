"""
Async-safe context management for nested MLflow runs.

This module provides ContextVar-based run tracking for async environments,
enabling proper nested run management in the TUI without thread-local issues.

Usage:
    from promptchain.observability.context import run_context

    # Start session-level run
    with run_context("chat_session"):
        # This automatically nests under session
        with run_context("llm_call"):
            # Further nesting for tool calls
            with run_context("tool_execution"):
                pass
"""

from contextvars import ContextVar
from contextlib import contextmanager
from typing import Optional, Generator
import logging

from .mlflow_adapter import (
    start_run,
    end_run,
    log_param,
    is_available
)

logger = logging.getLogger(__name__)

# Async-safe storage for current run ID (works across async tasks)
_current_run: ContextVar[Optional[str]] = ContextVar('mlflow_run', default=None)


def get_current_run() -> Optional[str]:
    """
    Get the currently active MLflow run ID.

    Returns:
        Run ID string if a run is active, None otherwise
    """
    return _current_run.get()


def set_current_run(run_id: Optional[str]) -> None:
    """
    Set the currently active MLflow run ID.

    Args:
        run_id: MLflow run ID to set as active, or None to clear
    """
    _current_run.set(run_id)


@contextmanager
def run_context(run_name: str, nested: bool = False) -> Generator:
    """
    Context manager for MLflow runs with proper async-safe nesting.

    Automatically handles:
    - Parent run preservation and restoration
    - Nested run creation when parent exists
    - Exception logging and failed run marking
    - Run cleanup on exit

    Args:
        run_name: Name for the MLflow run
        nested: Force nested run even without parent (default: auto-detect)

    Yields:
        ActiveRun object or None if MLflow unavailable

    Example:
        with run_context("session"):
            # Session-level tracking
            with run_context("llm_call"):
                # Automatically nested under session
                model.generate()
    """
    if not is_available():
        logger.debug(f"MLflow unavailable, skipping run: {run_name}")
        yield None
        return

    # Save parent run context
    parent_run = _current_run.get()

    # Determine if this should be a nested run
    should_nest = nested or (parent_run is not None)

    # Start new run
    run = start_run(run_name, nested=should_nest)

    if run:
        _current_run.set(run.info.run_id)
        logger.debug(f"Started run: {run_name} (nested={should_nest})")

    try:
        yield run
    except Exception as e:
        # Log exception details to MLflow
        if run:
            log_param("error_type", type(e).__name__)
            log_param("error_message", str(e))
            logger.error(f"Run {run_name} failed: {e}")

        # End run with failed status
        if run:
            end_run(status="FAILED")

        # Re-raise to preserve exception handling
        raise
    finally:
        # Normal completion
        if run:
            end_run(status="FINISHED")
            logger.debug(f"Ended run: {run_name}")

        # Restore parent run context (critical for nesting)
        _current_run.set(parent_run)
