"""
Communication Handler Decorator and Registry.

Provides @cli_communication_handler decorator for registering message handlers
that can filter by sender, receiver, and message type.

FR-016: System MUST support @cli_communication_handler decorator for message handlers
FR-017: Handlers MUST support filtering by sender, receiver, and message type
FR-020: Communication MUST be backward compatible - existing code works without handlers
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Message types for agent communication (FR-018)."""

    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    DELEGATION = "delegation"
    STATUS = "status"


@dataclass
class CommunicationHandler:
    """Registered communication handler with filter criteria."""

    func: Callable
    name: str
    message_types: Set[MessageType] = field(default_factory=set)
    senders: Set[str] = field(default_factory=set)
    receivers: Set[str] = field(default_factory=set)
    priority: int = 0

    def matches(self, message_type: MessageType, sender: str, receiver: str) -> bool:
        """Check if handler matches the given message criteria."""
        # Empty set means "match all"
        type_match = not self.message_types or message_type in self.message_types
        sender_match = not self.senders or sender in self.senders
        receiver_match = not self.receivers or receiver in self.receivers
        return type_match and sender_match and receiver_match


class HandlerRegistry:
    """Global registry for communication handlers."""

    _instance: Optional["HandlerRegistry"] = None
    _lock = threading.Lock()  # BUG-021 fix: Thread-safe singleton

    def __new__(cls) -> "HandlerRegistry":
        # BUG-021 fix: Use double-checked locking for thread safety
        if cls._instance is None:
            with cls._lock:
                # Double-check inside lock to prevent race condition
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Only initialize once — __init__ is called every time __new__ returns,
        # but _handlers already exists on subsequent calls so we guard here.
        if not hasattr(self, "_handlers"):
            self._handlers: List[CommunicationHandler] = []

    @classmethod
    def reset(cls) -> None:
        """Reset registry (for testing)."""
        if cls._instance:
            cls._instance._handlers = []

    def register(self, handler: CommunicationHandler) -> None:
        """Register a handler."""
        self._handlers.append(handler)
        self._handlers.sort(key=lambda h: -h.priority)
        logger.debug(f"Registered handler: {handler.name}")

    def unregister(self, name: str) -> bool:
        """Unregister a handler by name."""
        before = len(self._handlers)
        self._handlers = [h for h in self._handlers if h.name != name]
        return len(self._handlers) < before

    def get_matching_handlers(
        self, message_type: MessageType, sender: str, receiver: str
    ) -> List[CommunicationHandler]:
        """Get all handlers matching the criteria."""
        return [h for h in self._handlers if h.matches(message_type, sender, receiver)]

    async def dispatch(
        self,
        message_type: MessageType,
        sender: str,
        receiver: str,
        payload: Dict[str, Any],
        stop_on_error: bool = False,
    ) -> List[Any]:
        """Dispatch message to all matching handlers.

        Args:
            message_type: Type of message to dispatch
            sender: Sending agent/component name
            receiver: Receiving agent/component name
            payload: Message payload
            stop_on_error: If True, stop dispatching after first handler error.
                          If False (default), continue with remaining handlers (FR-020).

        Returns:
            List of handler results (may include error dicts if handlers failed)
        """
        handlers = self.get_matching_handlers(message_type, sender, receiver)
        results = []

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler.func):
                    result = await handler.func(payload, sender, receiver)
                else:
                    result = handler.func(payload, sender, receiver)
                results.append(result)
            except Exception as e:
                logger.error(f"Handler {handler.name} failed: {e}")
                error_result = {"error": str(e), "handler": handler.name}
                results.append(error_result)

                # BUG-014 fix: Optionally stop on first error for critical handler chains
                if stop_on_error:
                    logger.warning(
                        f"Stopping handler dispatch after {handler.name} failure "
                        f"(stop_on_error=True)"
                    )
                    break
                # FR-020: System continues on handler exception (default behavior)

        return results

    @property
    def handlers(self) -> List[CommunicationHandler]:
        """Get all registered handlers."""
        return self._handlers.copy()


def cli_communication_handler(
    type: Optional[MessageType] = None,
    types: Optional[List[MessageType]] = None,
    sender: Optional[str] = None,
    senders: Optional[List[str]] = None,
    receiver: Optional[str] = None,
    receivers: Optional[List[str]] = None,
    priority: int = 0,
    name: Optional[str] = None,
) -> Callable:
    """
    Decorator to register a function as a communication handler.

    Args:
        type: Single message type to filter (convenience for types=[type])
        types: List of message types to handle (empty = all)
        sender: Single sender to filter (convenience for senders=[sender])
        senders: List of senders to handle (empty = all)
        receiver: Single receiver to filter (convenience for receivers=[receiver])
        receivers: List of receivers to handle (empty = all)
        priority: Handler priority (higher = called first)
        name: Handler name (defaults to function name)

    Example:
        @cli_communication_handler(type=MessageType.REQUEST, sender="AgentA")
        async def handle_requests(payload, sender, receiver):
            return {"status": "processed"}
    """

    def decorator(func: Callable) -> Callable:
        handler_types: Set[MessageType] = set()
        if type:
            handler_types.add(type)
        if types:
            handler_types.update(types)

        handler_senders: Set[str] = set()
        if sender:
            handler_senders.add(sender)
        if senders:
            handler_senders.update(senders)

        handler_receivers: Set[str] = set()
        if receiver:
            handler_receivers.add(receiver)
        if receivers:
            handler_receivers.update(receivers)

        handler = CommunicationHandler(
            func=func,
            name=name or func.__name__,
            message_types=handler_types,
            senders=handler_senders,
            receivers=handler_receivers,
            priority=priority,
        )

        HandlerRegistry().register(handler)

        # Return original function to allow normal use
        return func

    return decorator


# Convenience singleton access
def get_handler_registry() -> HandlerRegistry:
    """Get the global handler registry."""
    return HandlerRegistry()
