# execution_events.py
"""Event system for PromptChain execution lifecycle monitoring.

This module defines events that are emitted at key points during PromptChain
execution, enabling comprehensive observability and monitoring.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime


class ExecutionEventType(Enum):
    """Event types emitted during PromptChain execution.

    These events track the complete lifecycle of chain execution, from
    initialization through completion, including all intermediate steps.
    """

    # Chain lifecycle events
    CHAIN_START = auto()
    CHAIN_END = auto()
    CHAIN_ERROR = auto()

    # Step execution events
    STEP_START = auto()
    STEP_END = auto()
    STEP_ERROR = auto()
    STEP_SKIPPED = auto()

    # Model execution events
    MODEL_CALL_START = auto()
    MODEL_CALL_END = auto()
    MODEL_CALL_ERROR = auto()

    # Tool execution events
    TOOL_CALL_START = auto()
    TOOL_CALL_END = auto()
    TOOL_CALL_ERROR = auto()

    # Function execution events
    FUNCTION_CALL_START = auto()
    FUNCTION_CALL_END = auto()
    FUNCTION_CALL_ERROR = auto()

    # Agentic step events
    AGENTIC_STEP_START = auto()
    AGENTIC_STEP_END = auto()
    AGENTIC_STEP_ERROR = auto()
    AGENTIC_INTERNAL_STEP = auto()

    # Chain control events
    CHAIN_BREAK = auto()

    # History management events
    HISTORY_TRUNCATED = auto()

    # MCP events
    MCP_CONNECT = auto()
    MCP_DISCONNECT = auto()
    MCP_TOOL_DISCOVERED = auto()

    # Model management events
    MODEL_LOAD = auto()
    MODEL_UNLOAD = auto()


@dataclass
class ExecutionEvent:
    """Container for execution event data.

    This dataclass encapsulates all relevant information about an event
    that occurred during chain execution.

    Attributes:
        event_type: Type of event that occurred
        timestamp: When the event occurred
        step_number: Optional step number in the chain (0-indexed)
        step_instruction: Optional instruction being executed
        model_name: Optional model name being used
        metadata: Additional event-specific data
    """

    event_type: ExecutionEventType
    timestamp: datetime = field(default_factory=datetime.now)
    step_number: Optional[int] = None
    step_instruction: Optional[str] = None
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for logging and serialization.

        Returns:
            Dictionary representation with ISO formatted timestamp
        """
        return {
            "event_type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
            "step_number": self.step_number,
            "step_instruction": self.step_instruction,
            "model_name": self.model_name,
            "metadata": self.metadata
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert event to condensed summary.

        Returns:
            Condensed dictionary with essential event information
        """
        summary = {
            "event": self.event_type.name,
            "timestamp": self.timestamp.isoformat()
        }

        if self.step_number is not None:
            summary["step"] = self.step_number

        if self.model_name:
            summary["model"] = self.model_name

        # Include key metadata fields
        if "error" in self.metadata:
            summary["error"] = self.metadata["error"]
        if "execution_time_ms" in self.metadata:
            summary["time_ms"] = self.metadata["execution_time_ms"]
        if "tokens_used" in self.metadata:
            summary["tokens"] = self.metadata["tokens_used"]

        return summary
