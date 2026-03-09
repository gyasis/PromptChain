"""Base classes for agentic patterns.

Provides abstract base class and common data structures for all pattern
implementations. Patterns integrate with the 003-multi-agent-communication
infrastructure (MessageBus, Blackboard).
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from promptchain.cli.models import (  # type: ignore[attr-defined]
        Blackboard, MessageBus)

logger = logging.getLogger(__name__)


@dataclass
class PatternConfig:
    """Configuration for pattern execution.

    Attributes:
        pattern_id: Unique identifier for this pattern instance.
        enabled: Whether the pattern is active.
        timeout_seconds: Maximum execution time before timeout.
        emit_events: Whether to emit events to MessageBus.
        use_blackboard: Whether to use Blackboard for state sharing.
    """

    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    enabled: bool = True
    timeout_seconds: float = 30.0
    emit_events: bool = True
    use_blackboard: bool = False


@dataclass
class PatternResult:
    """Result from pattern execution.

    Attributes:
        pattern_id: ID of the pattern that generated this result.
        success: Whether execution completed successfully.
        result: Pattern-specific result data.
        execution_time_ms: Time taken to execute in milliseconds.
        metadata: Additional execution metadata.
        errors: List of error messages if any.
        timestamp: When the result was generated.
    """

    pattern_id: str
    success: bool
    result: Any
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "pattern_id": self.pattern_id,
            "success": self.success,
            "result": self.result,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
        }


class BasePattern(ABC):
    """Abstract base class for all agentic patterns.

    Provides common functionality including:
    - Event emission to MessageBus (003 infrastructure)
    - State sharing via Blackboard (003 infrastructure)
    - Timeout handling
    - Error handling and logging

    Subclasses must implement the execute() method.

    Example:
        >>> class MyPattern(BasePattern):
        ...     async def execute(self, **kwargs) -> PatternResult:
        ...         self.emit_event("pattern.my.started", {"input": kwargs})
        ...         result = await self._do_work(**kwargs)
        ...         self.emit_event("pattern.my.completed", {"result": result})
        ...         return PatternResult(
        ...             pattern_id=self.config.pattern_id,
        ...             success=True,
        ...             result=result,
        ...             execution_time_ms=100.0
        ...         )
    """

    def __init__(self, config: Optional[PatternConfig] = None):
        """Initialize the pattern.

        Args:
            config: Pattern configuration. Uses defaults if not provided.
        """
        self.config = config or PatternConfig()
        self._event_handlers: List[Callable[[str, Dict[str, Any]], None]] = []
        self._bus: Optional["MessageBus"] = None
        self._blackboard: Optional["Blackboard"] = None
        self._execution_count = 0
        self._total_execution_time_ms = 0.0

    @abstractmethod
    async def execute(self, **kwargs) -> PatternResult:
        """Execute the pattern.

        This method must be implemented by subclasses to define the
        pattern-specific logic.

        Args:
            **kwargs: Pattern-specific arguments.

        Returns:
            PatternResult with execution outcome.

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout_seconds.
            PatternExecutionError: If execution fails.
        """
        pass

    async def execute_with_timeout(self, **kwargs) -> PatternResult:
        """Execute the pattern with timeout handling.

        Wraps the execute() method with timeout and error handling.

        Args:
            **kwargs: Arguments passed to execute().

        Returns:
            PatternResult with execution outcome.
        """
        if not self.config.enabled:
            return PatternResult(
                pattern_id=self.config.pattern_id,
                success=False,
                result=None,
                execution_time_ms=0.0,
                errors=["Pattern is disabled"],
            )

        start_time = time.perf_counter()
        self.emit_event(f"pattern.{self._pattern_name}.started", {"input": kwargs})

        try:
            result = await asyncio.wait_for(
                self.execute(**kwargs), timeout=self.config.timeout_seconds
            )
            self._execution_count += 1
            self._total_execution_time_ms += result.execution_time_ms

            self.emit_event(
                f"pattern.{self._pattern_name}.completed",
                {
                    "success": result.success,
                    "execution_time_ms": result.execution_time_ms,
                },
            )

            return result

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"Pattern {self.config.pattern_id} timed out after {elapsed_ms:.2f}ms"
            )
            self.emit_event(
                f"pattern.{self._pattern_name}.timeout",
                {
                    "elapsed_ms": elapsed_ms,
                },
            )
            return PatternResult(
                pattern_id=self.config.pattern_id,
                success=False,
                result=None,
                execution_time_ms=elapsed_ms,
                errors=[f"Execution timed out after {self.config.timeout_seconds}s"],
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Pattern {self.config.pattern_id} failed: {e}")
            self.emit_event(
                f"pattern.{self._pattern_name}.error",
                {
                    "error": str(e),
                    "elapsed_ms": elapsed_ms,
                },
            )
            return PatternResult(
                pattern_id=self.config.pattern_id,
                success=False,
                result=None,
                execution_time_ms=elapsed_ms,
                errors=[str(e)],
            )

    @property
    def _pattern_name(self) -> str:
        """Get the pattern name for event emission."""
        return (
            self.__class__.__name__.lower()
            .replace("lightrag", "")
            .replace("pattern", "")
        )

    # MessageBus integration (003 infrastructure)

    def connect_messagebus(self, bus: "MessageBus") -> None:
        """Connect to a MessageBus for event emission.

        Args:
            bus: MessageBus instance from 003 infrastructure.
        """
        self._bus = bus
        logger.debug(f"Pattern {self.config.pattern_id} connected to MessageBus")

    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to connected handlers and MessageBus.

        Args:
            event_type: Type of event (e.g., "pattern.branching.started").
            data: Event payload data.
        """
        if not self.config.emit_events:
            return

        event_data = {
            "pattern_id": self.config.pattern_id,
            "timestamp": datetime.utcnow().isoformat(),
            **data,
        }

        # Emit to local handlers
        for handler in self._event_handlers:
            try:
                handler(event_type, event_data)
            except Exception as e:
                logger.warning(f"Event handler error: {e}")

        # Emit to MessageBus if connected
        if self._bus is not None:
            try:
                self._bus.publish(event_type, event_data)
            except Exception as e:
                logger.warning(f"MessageBus publish error: {e}")

    def add_event_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add a local event handler.

        Args:
            handler: Callable that receives (event_type, event_data).
        """
        self._event_handlers.append(handler)

    def subscribe_to(self, pattern: str, handler: Callable) -> None:
        """Subscribe to events matching a pattern.

        Only works when connected to a MessageBus.

        Args:
            pattern: Event pattern to subscribe to (e.g., "pattern.branching.*").
            handler: Handler function for matching events.
        """
        if self._bus is not None:
            self._bus.subscribe(pattern, handler)
        else:
            logger.warning("Cannot subscribe: MessageBus not connected")

    # Blackboard integration (003 infrastructure)

    def connect_blackboard(self, blackboard: "Blackboard") -> None:
        """Connect to a Blackboard for state sharing.

        Args:
            blackboard: Blackboard instance from 003 infrastructure.
        """
        self._blackboard = blackboard
        logger.debug(f"Pattern {self.config.pattern_id} connected to Blackboard")

    def set_blackboard(self, blackboard: "Blackboard") -> None:
        """Alias for connect_blackboard for compatibility."""
        self.connect_blackboard(blackboard)

    def share_result(self, key: str, value: Any) -> None:
        """Share a result via Blackboard.

        Args:
            key: Key to store the value under.
            value: Value to share.
        """
        if not self.config.use_blackboard:
            return

        if self._blackboard is not None:
            try:
                self._blackboard.write(key, value, source=self.config.pattern_id)
            except Exception as e:
                logger.warning(f"Blackboard write error: {e}")
        else:
            logger.warning("Cannot share result: Blackboard not connected")

    def read_shared(self, key: str) -> Optional[Any]:
        """Read a shared value from Blackboard.

        Args:
            key: Key to read.

        Returns:
            Value if found, None otherwise.
        """
        if self._blackboard is not None:
            try:
                return self._blackboard.read(key)
            except Exception as e:
                logger.warning(f"Blackboard read error: {e}")
        return None

    # Statistics

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with execution count and average time.
        """
        avg_time = (
            self._total_execution_time_ms / self._execution_count
            if self._execution_count > 0
            else 0.0
        )
        return {
            "pattern_id": self.config.pattern_id,
            "execution_count": self._execution_count,
            "total_execution_time_ms": self._total_execution_time_ms,
            "average_execution_time_ms": avg_time,
        }


class PatternExecutionError(Exception):
    """Exception raised when pattern execution fails."""

    def __init__(
        self, message: str, pattern_id: str, details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.pattern_id = pattern_id
        self.details = details or {}
