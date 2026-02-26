"""Event emission system for LightRAG patterns.

Provides standardized event types, payloads, and utilities for all
pattern implementations. Ensures consistency in event naming and
structure across the pattern ecosystem.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EventSeverity(Enum):
    """Severity levels for pattern events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class EventLifecycle(Enum):
    """Standard lifecycle phases for pattern events."""

    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class PatternEvent:
    """Standardized event structure for pattern communication.

    All pattern events should be created using this structure to ensure
    consistency across the ecosystem.
    """

    event_type: str
    pattern_id: str
    pattern_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    severity: EventSeverity = EventSeverity.INFO
    correlation_id: Optional[str] = None  # For tracking related events

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
            "data": self.data,
            "severity": self.severity.value,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternEvent":
        """Create event from dictionary."""
        return cls(
            event_type=data["event_type"],
            pattern_id=data["pattern_id"],
            pattern_name=data["pattern_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_id=data.get("event_id", str(uuid.uuid4())),
            data=data.get("data", {}),
            severity=EventSeverity(data.get("severity", "info")),
            correlation_id=data.get("correlation_id"),
        )


# Standard event types for each pattern
PATTERN_EVENTS = {
    "branching": {
        "lifecycle": [
            "pattern.branching.started",
            "pattern.branching.hypothesis_generated",
            "pattern.branching.judging",
            "pattern.branching.scored",
            "pattern.branching.selected",
            "pattern.branching.completed",
            "pattern.branching.failed",
        ],
        "description": "Branching Thoughts pattern for hypothesis generation and selection",
    },
    "query_expansion": {
        "lifecycle": [
            "pattern.query_expansion.started",
            "pattern.query_expansion.analyzing",
            "pattern.query_expansion.expanded",
            "pattern.query_expansion.searching",
            "pattern.query_expansion.merging",
            "pattern.query_expansion.completed",
            "pattern.query_expansion.failed",
        ],
        "description": "Query Expansion pattern for search diversification",
    },
    "sharded": {
        "lifecycle": [
            "pattern.sharded.started",
            "pattern.sharded.shard_querying",
            "pattern.sharded.shard_completed",
            "pattern.sharded.aggregating",
            "pattern.sharded.completed",
            "pattern.sharded.failed",
        ],
        "description": "Sharded Retrieval pattern for multi-source queries",
    },
    "multi_hop": {
        "lifecycle": [
            "pattern.multi_hop.started",
            "pattern.multi_hop.decomposed",
            "pattern.multi_hop.hop_started",
            "pattern.multi_hop.hop_completed",
            "pattern.multi_hop.synthesizing",
            "pattern.multi_hop.completed",
            "pattern.multi_hop.failed",
        ],
        "description": "Multi-Hop Retrieval pattern for question decomposition",
    },
    "hybrid_search": {
        "lifecycle": [
            "pattern.hybrid_search.started",
            "pattern.hybrid_search.local_querying",
            "pattern.hybrid_search.global_querying",
            "pattern.hybrid_search.fusing",
            "pattern.hybrid_search.completed",
            "pattern.hybrid_search.failed",
        ],
        "description": "Hybrid Search Fusion pattern for multi-technique combination",
    },
    "speculative": {
        "lifecycle": [
            "pattern.speculative.started",
            "pattern.speculative.predicting",
            "pattern.speculative.executing",
            "pattern.speculative.cache_hit",
            "pattern.speculative.cache_miss",
            "pattern.speculative.completed",
            "pattern.speculative.failed",
        ],
        "description": "Speculative Execution pattern for predictive tool calling",
    },
}


# Event factory functions


def create_started_event(
    pattern_name: str,
    pattern_id: str,
    input_data: Dict[str, Any],
    correlation_id: Optional[str] = None,
) -> PatternEvent:
    """Create a pattern started event."""
    return PatternEvent(
        event_type=f"pattern.{pattern_name}.started",
        pattern_id=pattern_id,
        pattern_name=pattern_name,
        data={"input": input_data},
        correlation_id=correlation_id,
    )


def create_progress_event(
    pattern_name: str,
    pattern_id: str,
    step: str,
    progress: float,
    details: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> PatternEvent:
    """Create a pattern progress event."""
    return PatternEvent(
        event_type=f"pattern.{pattern_name}.progress",
        pattern_id=pattern_id,
        pattern_name=pattern_name,
        data={"step": step, "progress": progress, "details": details or {}},
        correlation_id=correlation_id,
    )


def create_completed_event(
    pattern_name: str,
    pattern_id: str,
    result: Any,
    execution_time_ms: float,
    metadata: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> PatternEvent:
    """Create a pattern completed event."""
    return PatternEvent(
        event_type=f"pattern.{pattern_name}.completed",
        pattern_id=pattern_id,
        pattern_name=pattern_name,
        data={
            "success": True,
            "result_summary": str(result)[:500] if result else None,
            "execution_time_ms": execution_time_ms,
            "metadata": metadata or {},
        },
        correlation_id=correlation_id,
    )


def create_failed_event(
    pattern_name: str,
    pattern_id: str,
    error: str,
    execution_time_ms: float,
    stack_trace: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> PatternEvent:
    """Create a pattern failed event."""
    return PatternEvent(
        event_type=f"pattern.{pattern_name}.failed",
        pattern_id=pattern_id,
        pattern_name=pattern_name,
        severity=EventSeverity.ERROR,
        data={
            "success": False,
            "error": error,
            "execution_time_ms": execution_time_ms,
            "stack_trace": stack_trace,
        },
        correlation_id=correlation_id,
    )


def create_timeout_event(
    pattern_name: str,
    pattern_id: str,
    timeout_seconds: float,
    elapsed_ms: float,
    correlation_id: Optional[str] = None,
) -> PatternEvent:
    """Create a pattern timeout event."""
    return PatternEvent(
        event_type=f"pattern.{pattern_name}.timeout",
        pattern_id=pattern_id,
        pattern_name=pattern_name,
        severity=EventSeverity.WARNING,
        data={
            "timeout_seconds": timeout_seconds,
            "elapsed_ms": elapsed_ms,
        },
        correlation_id=correlation_id,
    )


# Event validation


def validate_event_type(event_type: str) -> bool:
    """Check if an event type is a valid registered pattern event."""
    if not event_type.startswith("pattern."):
        return False

    parts = event_type.split(".")
    if len(parts) < 3:
        return False

    pattern_name = parts[1]
    if pattern_name not in PATTERN_EVENTS:
        return False

    return event_type in PATTERN_EVENTS[pattern_name]["lifecycle"]


def get_pattern_events(pattern_name: str) -> List[str]:
    """Get all registered event types for a pattern."""
    if pattern_name in PATTERN_EVENTS:
        return list(PATTERN_EVENTS[pattern_name]["lifecycle"])
    return []


def get_all_event_types() -> List[str]:
    """Get all registered event types across all patterns."""
    all_events: List[str] = []
    for pattern_info in PATTERN_EVENTS.values():
        all_events.extend(pattern_info["lifecycle"])
    return all_events


# Event subscription helpers


def subscribe_to_pattern(pattern_name: str) -> str:
    """Generate wildcard subscription pattern for a pattern's events.

    Example: subscribe_to_pattern("branching") -> "pattern.branching.*"
    """
    return f"pattern.{pattern_name}.*"


def subscribe_to_lifecycle(lifecycle: EventLifecycle) -> str:
    """Generate wildcard subscription for a lifecycle phase.

    Example: subscribe_to_lifecycle(EventLifecycle.COMPLETED) -> "pattern.*.completed"
    """
    return f"pattern.*.{lifecycle.value}"


def subscribe_to_all_patterns() -> str:
    """Generate wildcard subscription for all pattern events."""
    return "pattern.*"
