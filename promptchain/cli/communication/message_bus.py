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
__all__ = ["MessageBus", "MessageType", "Message"]


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

        return messages[-limit:][::-1]

    def clear_history(self) -> int:
        """Clear message history. Returns count of cleared messages."""
        count = len(self._message_history)
        self._message_history.clear()
        return count
