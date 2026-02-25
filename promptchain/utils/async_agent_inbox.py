"""
AsyncAgentInbox: Priority queue for inter-agent messaging.

FR-016: asyncio.PriorityQueue-based inbox for non-blocking agent communication.

Priority levels:
    PRIORITY_INTERRUPT   = 0   # interrupt / abort signals — highest priority
    PRIORITY_NORMAL      = 1   # regular agent messages
    PRIORITY_BACKGROUND  = 2   # background info / notifications

Branch: 006-promptchain-improvements (US4, Wave 5)
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Priority level constants (FR-016)
# ---------------------------------------------------------------------------

PRIORITY_INTERRUPT = 0  # Highest urgency — interrupt / abort signals
PRIORITY_NORMAL = 1  # Regular agent messages
PRIORITY_BACKGROUND = 2  # Background info / low-urgency notifications


# ---------------------------------------------------------------------------
# InboxMessage dataclass
# ---------------------------------------------------------------------------


@dataclass
class InboxMessage:
    """
    A single message in an agent's inbox.

    Attributes:
        priority:   Urgency level; lower values are dequeued first.
                    Use PRIORITY_INTERRUPT (0), PRIORITY_NORMAL (1),
                    or PRIORITY_BACKGROUND (2).
        topic:      Short string categorising the message (e.g. "task", "abort").
        payload:    Arbitrary data associated with the message.
        sender_id:  Optional identifier of the originating agent.
        timestamp:  Unix timestamp set automatically at construction time.

    The ``__lt__`` and ``__le__`` dunder methods are required because
    ``asyncio.PriorityQueue`` uses heap ordering; it compares the full tuple
    ``(priority, message)`` when priorities are equal and falls back to the
    message comparison.
    """

    priority: int
    topic: str
    payload: Any
    sender_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def __lt__(self, other: "InboxMessage") -> bool:
        return self.priority < other.priority

    def __le__(self, other: "InboxMessage") -> bool:
        return self.priority <= other.priority


# ---------------------------------------------------------------------------
# AsyncAgentInbox
# ---------------------------------------------------------------------------


class AsyncAgentInbox:
    """
    Priority-based async inbox for a single agent.

    Wraps ``asyncio.PriorityQueue`` and provides a clean, typed interface
    for enqueuing and dequeuing ``InboxMessage`` objects.  Lower numerical
    priority values are dequeued first (0 = highest urgency).

    Args:
        agent_id: Unique identifier for the owning agent.
        maxsize:  Maximum number of messages that can be queued before
                  ``send`` blocks.  0 means unlimited.  Default: 100.

    Example::

        inbox = AsyncAgentInbox(agent_id="planner")
        msg = InboxMessage(priority=PRIORITY_INTERRUPT, topic="abort", payload=None)
        await inbox.send(msg)
        received = await inbox.receive()
        assert received.topic == "abort"
    """

    def __init__(self, agent_id: str, maxsize: int = 100) -> None:
        self.agent_id: str = agent_id
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=maxsize)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def send(self, message: InboxMessage) -> None:
        """
        Enqueue *message* into the inbox.

        Blocks if the queue is full (i.e. ``maxsize`` has been reached)
        until space becomes available.

        Args:
            message: The ``InboxMessage`` to enqueue.
        """
        # The queue stores (priority, message) tuples so the heap can order
        # entries by priority without comparing arbitrary payload objects.
        await self._queue.put((message.priority, message))

    async def receive(self) -> InboxMessage:
        """
        Dequeue and return the highest-priority message.

        Blocks until a message is available.  Lower numerical priority
        values are returned first (0 = interrupt = highest urgency).

        Returns:
            The next ``InboxMessage`` in priority order.
        """
        _, message = await self._queue.get()
        return message

    async def try_receive(self) -> Optional[InboxMessage]:
        """
        Non-blocking dequeue.

        Returns the highest-priority pending message, or ``None`` if the
        queue is currently empty.  Never blocks the event loop.

        Returns:
            ``InboxMessage`` if one is available, otherwise ``None``.
        """
        try:
            _, message = self._queue.get_nowait()
            return message
        except asyncio.QueueEmpty:
            return None

    def qsize(self) -> int:
        """Return the current number of messages in the queue."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Return ``True`` if the queue contains no messages."""
        return self._queue.empty()
