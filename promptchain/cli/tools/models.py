"""Data models for tool execution results and metadata.

Provides structured result types for consistent tool execution reporting
with timing information, error details, and metadata tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """
    Result from a tool execution.

    Captures all information about a tool execution including success status,
    result data, error information, and performance metrics.

    Attributes:
        success: Whether execution succeeded
        tool_name: Name of the tool that was executed
        result: Result data (if successful)
        error: Error message (if failed)
        error_type: Exception type name (if failed)
        execution_time_ms: Execution time in milliseconds
        timestamp: When execution occurred
        metadata: Additional metadata about execution
    """

    success: bool
    tool_name: str
    result: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.

        Returns:
            Dictionary representation of result
        """
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "result": self.result,
            "error": self.error,
            "error_type": self.error_type,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """String representation of result."""
        if self.success:
            return f"ToolResult(success=True, tool={self.tool_name}, time={self.execution_time_ms:.2f}ms)"
        else:
            return f"ToolResult(success=False, tool={self.tool_name}, error={self.error_type}: {self.error})"


@dataclass
class ExecutionMetrics:
    """
    Performance metrics for tool execution.

    Tracks execution timing, parameter validation overhead, and other
    performance characteristics.

    Attributes:
        validation_time_ms: Time spent validating parameters
        execution_time_ms: Actual tool execution time
        total_time_ms: Total time including overhead
        parameter_count: Number of parameters provided
        was_cached: Whether result was retrieved from cache
    """

    validation_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    total_time_ms: float = 0.0
    parameter_count: int = 0
    was_cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "validation_time_ms": self.validation_time_ms,
            "execution_time_ms": self.execution_time_ms,
            "total_time_ms": self.total_time_ms,
            "parameter_count": self.parameter_count,
            "was_cached": self.was_cached,
        }


@dataclass
class ToolExecutionContext:
    """
    Context information for tool execution.

    Provides contextual information about the execution environment,
    user, session, and other metadata that may be needed for logging,
    security checks, or auditing.

    Attributes:
        session_id: ID of current session
        user_id: ID of user requesting execution
        agent_name: Name of agent executing tool
        project_root: Project root directory
        working_directory: Current working directory
        additional_context: Any additional context data
    """

    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_name: Optional[str] = None
    project_root: Optional[str] = None
    working_directory: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_name": self.agent_name,
            "project_root": self.project_root,
            "working_directory": self.working_directory,
            "additional_context": self.additional_context,
        }
