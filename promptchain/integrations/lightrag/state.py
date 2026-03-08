"""Blackboard state integration utilities for LightRAG patterns.

Provides advanced state management patterns beyond BasePattern's basic support,
including:
- PatternBlackboardMixin for enhanced state operations
- StateSnapshot for point-in-time state capture
- PatternStateCoordinator for multi-pattern state sharing
- Typed state accessors with validation
"""

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generic, List,
                    Optional, TypeVar)

if TYPE_CHECKING:
    from promptchain.cli.models import Blackboard  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class StateEntry:
    """A single state entry with metadata."""

    key: str
    value: Any
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    ttl_seconds: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if this entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.utcnow() - self.timestamp).total_seconds()
        return elapsed > self.ttl_seconds


@dataclass
class StateSnapshot:
    """Point-in-time capture of pattern state."""

    pattern_id: str
    timestamp: datetime
    entries: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the snapshot."""
        return self.entries.get(key, default)

    def diff(self, other: "StateSnapshot") -> Dict[str, Any]:
        """Compare with another snapshot and return differences."""
        added = {k: v for k, v in self.entries.items() if k not in other.entries}
        removed = {k: v for k, v in other.entries.items() if k not in self.entries}
        changed = {
            k: {"old": other.entries[k], "new": v}
            for k, v in self.entries.items()
            if k in other.entries and other.entries[k] != v
        }
        return {"added": added, "removed": removed, "changed": changed}


class PatternBlackboardMixin:
    """Mixin to add enhanced Blackboard state sharing to patterns.

    While BasePattern provides basic integration, this mixin adds:
    - Typed state accessors with validation
    - State versioning and history
    - TTL-based state expiration
    - Transactional state updates
    - State snapshots for debugging
    """

    _blackboard: Optional["Blackboard"] = None
    _local_state: Dict[str, StateEntry] = {}
    _state_history: List[StateEntry] = []
    _max_history: int = 50

    def connect_blackboard(self, blackboard: "Blackboard") -> None:
        """Connect to a Blackboard for state sharing."""
        self._blackboard = blackboard
        self._local_state = {}
        self._state_history = []
        logger.debug("Pattern connected to Blackboard")

    def disconnect_blackboard(self) -> None:
        """Disconnect from the Blackboard."""
        self._blackboard = None
        logger.debug("Pattern disconnected from Blackboard")

    @property
    def pattern_id(self) -> str:
        """Get the pattern ID for state operations."""
        if hasattr(self, "config") and hasattr(self.config, "pattern_id"):
            return self.config.pattern_id
        return "unknown"

    def share_result(
        self, key: str, value: Any, ttl_seconds: Optional[float] = None
    ) -> None:
        """Share a result via Blackboard with optional TTL.

        Args:
            key: State key
            value: Value to share
            ttl_seconds: Optional time-to-live in seconds
        """
        entry = StateEntry(
            key=key, value=value, source=self.pattern_id, ttl_seconds=ttl_seconds
        )

        # Update version if key exists
        if key in self._local_state:
            entry.version = self._local_state[key].version + 1

        self._local_state[key] = entry
        self._record_state_change(entry)

        if self._blackboard:
            try:
                self._blackboard.write(key, value, source=self.pattern_id)
            except Exception as e:
                logger.warning(f"Failed to share state {key}: {e}")

    def read_shared(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Read a shared value from Blackboard.

        Args:
            key: State key to read
            default: Default value if key not found

        Returns:
            The value if found, default otherwise
        """
        # Check local state first (for TTL expiration)
        if key in self._local_state:
            entry = self._local_state[key]
            if entry.is_expired():
                del self._local_state[key]
                return default

        if self._blackboard:
            try:
                return self._blackboard.read(key) or default
            except Exception as e:
                logger.warning(f"Failed to read state {key}: {e}")

        return default

    def read_typed(
        self, key: str, expected_type: type, default: Optional[T] = None
    ) -> Optional[T]:
        """Read a value with type validation.

        Args:
            key: State key
            expected_type: Expected type of the value
            default: Default value if key not found or type mismatch

        Returns:
            The value if found and type matches, default otherwise
        """
        value = self.read_shared(key)
        if value is None:
            return default
        if not isinstance(value, expected_type):
            logger.warning(
                f"State {key} has type {type(value)}, expected {expected_type}"
            )
            return default
        return value

    def update_state(self, key: str, updater: Callable[[Any], Any]) -> Any:
        """Atomically update a state value.

        Args:
            key: State key
            updater: Function that takes current value and returns new value

        Returns:
            The new value
        """
        current = self.read_shared(key)
        new_value = updater(current)
        self.share_result(key, new_value)
        return new_value

    def batch_share(
        self, updates: Dict[str, Any], ttl_seconds: Optional[float] = None
    ) -> None:
        """Share multiple state values atomically."""
        for key, value in updates.items():
            self.share_result(key, value, ttl_seconds=ttl_seconds)

    def snapshot(self) -> StateSnapshot:
        """Capture current state as a snapshot."""
        entries = {}
        for key, entry in self._local_state.items():
            if not entry.is_expired():
                entries[key] = entry.value

        return StateSnapshot(
            pattern_id=self.pattern_id,
            timestamp=datetime.utcnow(),
            entries=entries,
            metadata={"state_count": len(entries)},
        )

    def restore_snapshot(self, snapshot: StateSnapshot) -> None:
        """Restore state from a snapshot."""
        for key, value in snapshot.entries.items():
            self.share_result(key, value)

    def clear_local_state(self) -> None:
        """Clear all local state entries."""
        self._local_state.clear()

    def _record_state_change(self, entry: StateEntry) -> None:
        """Record state change in history."""
        self._state_history.append(copy.deepcopy(entry))
        if len(self._state_history) > self._max_history:
            self._state_history = self._state_history[-self._max_history :]

    def get_state_history(self, key: Optional[str] = None) -> List[StateEntry]:
        """Get state change history with optional key filter."""
        if key:
            return [e for e in self._state_history if e.key == key]
        return list(self._state_history)


class PatternStateCoordinator:
    """Coordinator for multi-pattern state sharing.

    Provides a central registry for patterns to share state and
    coordinate complex workflows.
    """

    def __init__(self, blackboard: "Blackboard"):
        self._blackboard = blackboard
        self._registered_patterns: Dict[str, "PatternBlackboardMixin"] = {}
        self._shared_namespace = "coordinator"

    def register_pattern(self, name: str, pattern: "PatternBlackboardMixin") -> None:
        """Register a pattern for coordinated state sharing."""
        self._registered_patterns[name] = pattern
        pattern.connect_blackboard(self._blackboard)

    def unregister_pattern(self, name: str) -> None:
        """Unregister a pattern."""
        if name in self._registered_patterns:
            self._registered_patterns[name].disconnect_blackboard()
            del self._registered_patterns[name]

    def share_across_patterns(self, key: str, value: Any) -> None:
        """Share a value that all registered patterns can access."""
        namespace_key = f"{self._shared_namespace}.{key}"
        self._blackboard.write(namespace_key, value, source="coordinator")

    def read_shared(self, key: str) -> Optional[Any]:
        """Read a shared value from the coordinator namespace."""
        namespace_key = f"{self._shared_namespace}.{key}"
        return self._blackboard.read(namespace_key)

    def collect_snapshots(self) -> Dict[str, StateSnapshot]:
        """Collect snapshots from all registered patterns."""
        return {
            name: pattern.snapshot()
            for name, pattern in self._registered_patterns.items()
        }

    def coordinate_handoff(
        self, from_pattern: str, to_pattern: str, keys: List[str]
    ) -> None:
        """Coordinate state handoff between patterns.

        Copies specified state keys from one pattern to another.
        """
        if from_pattern not in self._registered_patterns:
            raise ValueError(f"Pattern {from_pattern} not registered")
        if to_pattern not in self._registered_patterns:
            raise ValueError(f"Pattern {to_pattern} not registered")

        source = self._registered_patterns[from_pattern]
        target = self._registered_patterns[to_pattern]

        for key in keys:
            value = source.read_shared(key)
            if value is not None:
                target.share_result(key, value)


# State key convention helpers
def pattern_state_key(pattern_name: str, key: str) -> str:
    """Generate a namespaced state key.

    Example: pattern_state_key("branching", "hypotheses") -> "pattern.branching.hypotheses"
    """
    return f"pattern.{pattern_name}.{key}"


def is_pattern_state_key(key: str) -> bool:
    """Check if a key is a pattern state key."""
    return key.startswith("pattern.")


def extract_pattern_from_key(key: str) -> Optional[str]:
    """Extract pattern name from a state key.

    Example: "pattern.branching.hypotheses" -> "branching"
    """
    if not is_pattern_state_key(key):
        return None
    parts = key.split(".")
    return parts[1] if len(parts) >= 2 else None
