"""
Message Bus for Agent-to-Agent Communication.

Provides message routing, dispatch, and logging for multi-agent workflows.

FR-018: System MUST support message types: request, response, broadcast, delegation, status
FR-019: Activity logger MUST capture all communication for debugging
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import uuid
import asyncio
import logging
import json

from .handlers import HandlerRegistry, MessageType, get_handler_registry

logger = logging.getLogger(__name__)


# Re-export MessageType for convenience
__all__ = ["MessageBus", "MessageType", "Message", "PubSubBus"]


@dataclass
class Message:
    """Agent communication message."""

    message_id: str
    sender: str
    receiver: str
    type: MessageType
    payload: Dict[str, Any]
    timestamp: float
    delivered: bool = False

    @classmethod
    def create(
        cls,
        sender: str,
        receiver: str,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> "Message":
        """Create a new message with auto-generated ID and timestamp."""
        return cls(
            message_id=str(uuid.uuid4()),
            sender=sender,
            receiver=receiver,
            type=message_type,
            payload=payload,
            timestamp=datetime.now().timestamp()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "delivered": self.delivered
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            sender=data["sender"],
            receiver=data["receiver"],
            type=MessageType(data["type"]),
            payload=data.get("payload", {}),
            timestamp=data["timestamp"],
            delivered=data.get("delivered", False)
        )


class MessageBus:
    """
    Central message bus for agent communication.

    Handles message routing, handler dispatch, and activity logging.
    """

    def __init__(
        self,
        session_id: str,
        activity_logger: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize message bus.

        Args:
            session_id: Session ID for message context
            activity_logger: Optional callback for activity logging (FR-019)
        """
        self.session_id = session_id
        self._activity_logger = activity_logger
        self._registry = get_handler_registry()
        self._message_queue: List[Message] = []
        self._message_history: List[Message] = []

    def _log_activity(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log activity for debugging (FR-019)."""
        log_entry = {
            "event_type": event_type,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            **data
        }

        logger.debug(f"Activity: {event_type} - {json.dumps(data, default=str)}")

        if self._activity_logger:
            try:
                self._activity_logger(log_entry)
            except Exception as e:
                logger.error(f"Activity logger failed: {e}")

    async def send(
        self,
        sender: str,
        receiver: str,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> Message:
        """
        Send a message to a specific receiver.

        Args:
            sender: Sending agent name
            receiver: Receiving agent name
            message_type: Type of message
            payload: Message payload

        Returns:
            The created Message object
        """
        message = Message.create(sender, receiver, message_type, payload)

        self._log_activity("message_sent", {
            "message_id": message.message_id,
            "sender": sender,
            "receiver": receiver,
            "type": message_type.value,
            "payload_keys": list(payload.keys())
        })

        # Dispatch to handlers
        results = await self._registry.dispatch(
            message_type, sender, receiver, payload
        )

        message.delivered = len(results) > 0
        self._message_history.append(message)

        self._log_activity("message_delivered", {
            "message_id": message.message_id,
            "handlers_invoked": len(results),
            "delivered": message.delivered
        })

        return message

    async def broadcast(
        self,
        sender: str,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.BROADCAST
    ) -> Message:
        """
        Broadcast a message to all agents.

        Args:
            sender: Sending agent name
            payload: Message payload
            message_type: Type of message (default: BROADCAST)

        Returns:
            The created Message object
        """
        return await self.send(
            sender=sender,
            receiver="*",  # Broadcast indicator
            message_type=message_type,
            payload=payload
        )

    async def request(
        self,
        sender: str,
        receiver: str,
        payload: Dict[str, Any]
    ) -> Message:
        """Send a request message."""
        return await self.send(sender, receiver, MessageType.REQUEST, payload)

    async def respond(
        self,
        sender: str,
        receiver: str,
        payload: Dict[str, Any]
    ) -> Message:
        """Send a response message."""
        return await self.send(sender, receiver, MessageType.RESPONSE, payload)

    async def delegate(
        self,
        sender: str,
        receiver: str,
        payload: Dict[str, Any]
    ) -> Message:
        """Send a delegation message."""
        return await self.send(sender, receiver, MessageType.DELEGATION, payload)

    async def status_update(
        self,
        sender: str,
        receiver: str,
        payload: Dict[str, Any]
    ) -> Message:
        """Send a status update message."""
        return await self.send(sender, receiver, MessageType.STATUS, payload)

    def get_history(
        self,
        sender: Optional[str] = None,
        receiver: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        limit: int = 100
    ) -> List[Message]:
        """
        Get message history with optional filters.

        Args:
            sender: Filter by sender
            receiver: Filter by receiver
            message_type: Filter by message type
            limit: Maximum messages to return

        Returns:
            List of matching messages (newest first)
        """
        messages = self._message_history.copy()

        if sender:
            messages = [m for m in messages if m.sender == sender]
        if receiver:
            messages = [m for m in messages if m.receiver == receiver]
        if message_type:
            messages = [m for m in messages if m.type == message_type]

        # BUG-011 fix: Ensure newest-first ordering
        return list(reversed(messages[-limit:]))

    def clear_history(self) -> int:
        """Clear message history. Returns count of cleared messages."""
        count = len(self._message_history)
        self._message_history.clear()
        return count

    async def subscribe(self, topic: str, callback: "Callable") -> None:
        """Subscribe a callback to a pub/sub topic (delegates to internal PubSubBus)."""
        if not hasattr(self, "_pubsub"):
            self._pubsub = PubSubBus()
        await self._pubsub.subscribe(topic, callback)

    async def unsubscribe(self, topic: str, callback: "Callable") -> None:
        """Unsubscribe a callback from a pub/sub topic."""
        if not hasattr(self, "_pubsub"):
            self._pubsub = PubSubBus()
        await self._pubsub.unsubscribe(topic, callback)

    async def publish_topic(self, topic: str, payload: "Any") -> None:
        """Publish a payload to a pub/sub topic."""
        if not hasattr(self, "_pubsub"):
            self._pubsub = PubSubBus()
        await self._pubsub.publish(topic, payload)

    async def send_global_override(self, new_prompt: str, sender_id: str = "") -> None:
        """
        Publish a global override to replace the active prompt (FR-014).

        Args:
            new_prompt: The new prompt/objective to broadcast.
            sender_id: Identifier of the sender (e.g. TUI session ID).
        """
        if not hasattr(self, "_pubsub"):
            self._pubsub = PubSubBus()
        await self._pubsub.publish(
            "agent.global_override",
            {"prompt": new_prompt, "sender_id": sender_id}
        )


# ---------------------------------------------------------------------------
# PubSubBus — async publish-subscribe bus for agent coordination (FR-017)
# ---------------------------------------------------------------------------

class PubSubBus:
    """
    Async publish-subscribe bus for agent coordination.

    FR-017: subscribe/unsubscribe/publish with per-subscriber error isolation.

    Attributes:
        _subscribers: Mapping from topic name to list of async callbacks.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable]] = {}

    async def subscribe(self, topic: str, callback: Callable) -> None:
        """Register an async callback for *topic*. Idempotent.

        Args:
            topic: Topic string to subscribe to.
            callback: Async callable that receives the published payload.
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        if callback not in self._subscribers[topic]:
            self._subscribers[topic].append(callback)

    async def unsubscribe(self, topic: str, callback: Callable) -> None:
        """Remove a callback from *topic*. No-op if not registered.

        Args:
            topic: Topic string to unsubscribe from.
            callback: The callback to remove.
        """
        if topic in self._subscribers:
            try:
                self._subscribers[topic].remove(callback)
            except ValueError:
                pass  # No-op if not registered

    async def publish(self, topic: str, payload: Any) -> None:
        """Deliver *payload* to all subscribers of *topic* concurrently.

        Per-subscriber exceptions are caught and logged; they do NOT
        propagate to the caller (FR-017 error isolation guarantee).

        Args:
            topic: Topic string to publish to.
            payload: Payload delivered to each subscriber callback.
        """
        callbacks = list(self._subscribers.get(topic, []))

        async def safe_call(cb: Callable) -> None:
            try:
                await cb(payload)
            except Exception as exc:
                logger.warning(
                    "PubSubBus subscriber error on topic %r: %s",
                    topic,
                    exc,
                )

        await asyncio.gather(
            *(safe_call(cb) for cb in callbacks),
            return_exceptions=False,
        )

    def publish_sync(self, topic: str, payload: Any) -> None:
        """Synchronous wrapper around :meth:`publish`.

        Safe to call from a non-async context only.  Uses
        ``asyncio.run()`` internally.

        Args:
            topic: Topic string to publish to.
            payload: Payload delivered to each subscriber callback.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.publish(topic, payload))
            else:
                loop.run_until_complete(self.publish(topic, payload))
        except RuntimeError:
            asyncio.run(self.publish(topic, payload))

    def send_global_override(self, new_prompt: str) -> None:
        """Publish a global override to replace the active prompt (FR-014).

        Convenience sync wrapper that publishes to the
        ``"agent.global_override"`` topic.

        Args:
            new_prompt: The new prompt/objective string to broadcast.
        """
        self.publish_sync("agent.global_override", {"prompt": new_prompt})
