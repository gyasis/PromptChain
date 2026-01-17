# agentic_step_result.py
"""Dataclasses for comprehensive execution metadata from AgenticStepProcessor."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class StepExecutionMetadata:
    """Metadata for a single agentic step execution.

    This dataclass captures detailed information about each internal iteration
    within an AgenticStepProcessor's execution loop.

    Attributes:
        step_number: The iteration number (1-indexed)
        tool_calls: List of tool calls made during this step, each containing:
                   - name: Tool function name
                   - args: Tool arguments
                   - result: Tool execution result
                   - time_ms: Tool execution time in milliseconds
        tokens_used: Estimated tokens consumed in this step
        execution_time_ms: Total time for this step in milliseconds
        clarification_attempts: Number of clarification attempts in this step
        error: Optional error message if step failed
    """
    step_number: int
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    execution_time_ms: float = 0.0
    clarification_attempts: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and serialization.

        Returns:
            Dictionary representation with all metadata
        """
        return {
            "step_number": self.step_number,
            "tool_calls_count": len(self.tool_calls),
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "execution_time_ms": self.execution_time_ms,
            "clarification_attempts": self.clarification_attempts,
            "error": self.error
        }


@dataclass
class AgenticStepResult:
    """Complete execution result from AgenticStepProcessor.

    This dataclass captures comprehensive execution information when
    AgenticStepProcessor.run_async() is called with return_metadata=True.

    Attributes:
        final_answer: The final output string from the agentic step
        total_steps: Total number of internal iterations executed
        max_steps_reached: Whether max_internal_steps limit was hit
        objective_achieved: Whether the objective was successfully completed
        steps: List of StepExecutionMetadata for each iteration
        total_tools_called: Total number of tool calls across all steps
        total_tokens_used: Total estimated tokens consumed
        total_execution_time_ms: Total execution time in milliseconds
        history_mode: History accumulation mode used
        max_internal_steps: Maximum steps configured
        model_name: Model name used for execution
        errors: List of error messages encountered
        warnings: List of warning messages encountered
    """

    # Final output
    final_answer: str

    # Execution summary
    total_steps: int
    max_steps_reached: bool
    objective_achieved: bool

    # Step-by-step details
    steps: List[StepExecutionMetadata] = field(default_factory=list)

    # Overall statistics
    total_tools_called: int = 0
    total_tokens_used: int = 0
    total_execution_time_ms: float = 0.0

    # Configuration used
    history_mode: str = "minimal"
    max_internal_steps: int = 5
    model_name: Optional[str] = None

    # Errors/warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and serialization.

        Returns:
            Dictionary representation with all metadata
        """
        return {
            "final_answer": self.final_answer,
            "total_steps": self.total_steps,
            "max_steps_reached": self.max_steps_reached,
            "objective_achieved": self.objective_achieved,
            "steps": [step.to_dict() for step in self.steps],
            "total_tools_called": self.total_tools_called,
            "total_tokens_used": self.total_tokens_used,
            "total_execution_time_ms": self.total_execution_time_ms,
            "history_mode": self.history_mode,
            "max_internal_steps": self.max_internal_steps,
            "model_name": self.model_name,
            "errors_count": len(self.errors),
            "errors": self.errors,
            "warnings_count": len(self.warnings),
            "warnings": self.warnings
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to a condensed summary with key metrics only.

        Returns:
            Condensed dictionary with essential execution metrics
        """
        return {
            "total_steps": self.total_steps,
            "tools_called": self.total_tools_called,
            "execution_time_ms": self.total_execution_time_ms,
            "objective_achieved": self.objective_achieved,
            "max_steps_reached": self.max_steps_reached,
            "tokens_used": self.total_tokens_used,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings)
        }
