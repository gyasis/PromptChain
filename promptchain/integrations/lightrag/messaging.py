"""MessageBus integration utilities for LightRAG patterns.

Provides additional messaging patterns beyond BasePattern's basic support,
including:
- PatternMessageBusMixin for legacy/alternative integration
- PatternEventBroadcaster for multi-pattern coordination
- Event topic conventions and helpers
"""

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
import logging
import fnmatch

if TYPE_CHECKING:
    from promptchain.cli.models import MessageBus

logger = logging.getLogger(__name__)


class PatternMessageBusMixin:
    """Mixin to add enhanced MessageBus integration to patterns.

    While BasePattern provides basic integration, this mixin adds:
    - Topic-based routing with wildcard support
    - Event filtering and transformation
    - Batch event publishing
    - Event history tracking
    """

    _bus: Optional["MessageBus"] = None
    _subscriptions: List[Dict[str, Any]] = []
    _event_history: List[Dict[str, Any]] = []
    _max_history: int = 100

    def connect_messagebus(self, bus: "MessageBus") -> None:
        """Connect to a MessageBus for event emission."""
        self._bus = bus
        self._subscriptions = []
        self._event_history = []
        logger.debug("Pattern connected to MessageBus")

    def disconnect_messagebus(self) -> None:
        """Disconnect from the MessageBus."""
        self._bus = None
        self._subscriptions.clear()
        logger.debug("Pattern disconnected from MessageBus")

    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a single event to the MessageBus."""
        if not self._bus:
            return

        event_data = {
            "pattern_id": getattr(self, 'config', {}).pattern_id if hasattr(self, 'config') else 'unknown',
            **data
        }

        try:
            self._bus.publish(event_type, event_data)
            self._record_event(event_type, event_data)
        except Exception as e:
            logger.warning(f"Failed to emit event {event_type}: {e}")

    def emit_batch(self, events: List[Dict[str, Any]]) -> None:
        """Emit multiple events in batch."""
        for event in events:
            self.emit_event(event.get("type", "unknown"), event.get("data", {}))

    def subscribe_to(self, pattern: str, handler: Callable, filter_fn: Optional[Callable] = None) -> str:
        """Subscribe to events matching a pattern with optional filtering.

        Args:
            pattern: Event pattern with wildcards (e.g., "pattern.branching.*")
            handler: Callback function for matching events
            filter_fn: Optional filter function to further filter events

        Returns:
            Subscription ID for later unsubscription
        """
        if not self._bus:
            logger.warning("Cannot subscribe: MessageBus not connected")
            return ""

        subscription_id = f"sub_{len(self._subscriptions)}"
        self._subscriptions.append({
            "id": subscription_id,
            "pattern": pattern,
            "handler": handler,
            "filter_fn": filter_fn
        })

        # Wrap handler with filter
        def filtered_handler(event_type: str, event_data: Dict[str, Any]):
            if filter_fn and not filter_fn(event_type, event_data):
                return
            handler(event_type, event_data)

        self._bus.subscribe(pattern, filtered_handler)
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events by subscription ID."""
        self._subscriptions = [s for s in self._subscriptions if s["id"] != subscription_id]
        return True

    def _record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record event in history for debugging."""
        from datetime import datetime
        self._event_history.append({
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        })
        # Trim history if needed
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

    def get_event_history(self, event_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recorded event history with optional filtering."""
        if event_type_filter:
            return [e for e in self._event_history if fnmatch.fnmatch(e["type"], event_type_filter)]
        return list(self._event_history)


class PatternEventBroadcaster:
    """Utility for broadcasting events to multiple patterns.

    Useful for orchestrating multi-pattern workflows where one event
    should trigger actions in multiple patterns.
    """

    def __init__(self, bus: "MessageBus"):
        self._bus = bus
        self._registered_patterns: Dict[str, "PatternMessageBusMixin"] = {}

    def register_pattern(self, name: str, pattern: "PatternMessageBusMixin") -> None:
        """Register a pattern for broadcasting."""
        self._registered_patterns[name] = pattern
        pattern.connect_messagebus(self._bus)

    def unregister_pattern(self, name: str) -> None:
        """Unregister a pattern."""
        if name in self._registered_patterns:
            self._registered_patterns[name].disconnect_messagebus()
            del self._registered_patterns[name]

    def broadcast(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast an event to the shared MessageBus."""
        self._bus.publish(event_type, data)

    def coordinate_workflow(self, workflow_id: str, steps: List[Dict[str, Any]]) -> None:
        """Emit workflow coordination events.

        Args:
            workflow_id: Unique workflow identifier
            steps: List of workflow steps with pattern and action
        """
        self.broadcast("workflow.started", {"workflow_id": workflow_id, "steps": len(steps)})

        for i, step in enumerate(steps):
            self.broadcast(f"workflow.step.{i}", {
                "workflow_id": workflow_id,
                "step_index": i,
                "pattern": step.get("pattern"),
                "action": step.get("action")
            })


# Topic convention helpers
def pattern_topic(pattern_name: str, event: str) -> str:
    """Generate a standard pattern topic string.

    Example: pattern_topic("branching", "started") -> "pattern.branching.started"
    """
    return f"pattern.{pattern_name}.{event}"


def is_pattern_event(event_type: str) -> bool:
    """Check if an event type is a pattern event."""
    return event_type.startswith("pattern.")


def extract_pattern_name(event_type: str) -> Optional[str]:
    """Extract pattern name from event type.

    Example: "pattern.branching.started" -> "branching"
    """
    if not is_pattern_event(event_type):
        return None
    parts = event_type.split(".")
    return parts[1] if len(parts) >= 2 else None
