"""
Unit tests for AsyncAgentInbox class.

T041: test_send_receive_normal_priority
T042: test_priority_ordering
T043: test_try_receive_returns_none_when_empty

Discovery results (2026-02-25):
  - promptchain/utils/async_agent_inbox.py : NOT FOUND — AsyncAgentInbox does not exist yet
  - InboxMessage dataclass                 : NOT FOUND — defined in the same missing module

These tests are RED (expected to fail) because promptchain/utils/async_agent_inbox.py
has not been implemented yet.  They will turn GREEN once T041–T043 implement:

    @dataclass
    class InboxMessage:
        priority: int          # 0=interrupt, 1=normal, 2=background
        topic: str
        payload: Any
        sender_id: Optional[str] = None
        timestamp: float = field(default_factory=time.time)

        def __lt__(self, other):   # Required for PriorityQueue ordering
            return self.priority < other.priority

    class AsyncAgentInbox:
        def __init__(self, agent_id: str, maxsize: int = 100) -> None: ...
        async def send(self, message: InboxMessage) -> None: ...
        async def receive(self) -> InboxMessage: ...       # blocks until message available
        async def try_receive(self) -> Optional[InboxMessage]: ...  # non-blocking
        def qsize(self) -> int: ...
        def empty(self) -> bool: ...

Priority levels (FR-016):
    PRIORITY_INTERRUPT   = 0   # interrupt / abort signals — highest priority
    PRIORITY_NORMAL      = 1   # regular agent messages
    PRIORITY_BACKGROUND  = 2   # background info / notifications

Spec reference: specs/006-promptchain-improvements/contracts/async-execution.md (FR-016)
Branch: 006-promptchain-improvements (US4, Wave 5)
"""

import asyncio
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Import guard — tests produce clean FAIL (not ERROR) when module is missing
# ---------------------------------------------------------------------------

def _import_async_agent_inbox():
    """
    Attempt to import AsyncAgentInbox and InboxMessage.
    Raises ImportError with a descriptive message if the module is not yet
    implemented, which pytest will surface as a test failure (not a collection
    error).
    """
    from promptchain.utils.async_agent_inbox import AsyncAgentInbox, InboxMessage
    return AsyncAgentInbox, InboxMessage


# ---------------------------------------------------------------------------
# T041–T043: AsyncAgentInbox tests
# ---------------------------------------------------------------------------

class TestAsyncAgentInbox:
    """
    Tests for AsyncAgentInbox priority queue functionality.

    All three tests share the same Arrange-Act-Assert structure:
      Arrange — create an inbox (and optionally pre-populate it)
      Act     — call send() / receive() / try_receive()
      Assert  — verify the returned message or None value
    """

    @pytest.mark.asyncio
    async def test_send_receive_normal_priority(self):
        """
        T041 — RED until async_agent_inbox.py is implemented.

        Arrange:
          - A fresh AsyncAgentInbox (agent_id="agent-a")
          - An InboxMessage with priority=PRIORITY_NORMAL (1), topic="task",
            payload="hello"

        Act:
          - await inbox.send(msg)
          - received = await inbox.receive()

        Assert:
          - received.payload == "hello"
          - received.topic == "task"
          - received.priority == 1
        """
        AsyncAgentInbox, InboxMessage = _import_async_agent_inbox()

        inbox = AsyncAgentInbox(agent_id="agent-a")
        msg = InboxMessage(priority=1, topic="task", payload="hello")

        await inbox.send(msg)
        received = await inbox.receive()

        assert received.payload == "hello", (
            f"Expected payload 'hello', got {received.payload!r}"
        )
        assert received.topic == "task", (
            f"Expected topic 'task', got {received.topic!r}"
        )
        assert received.priority == 1, (
            f"Expected priority 1, got {received.priority}"
        )

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """
        T042 — RED until async_agent_inbox.py is implemented.

        Verifies that lower numerical priority values are dequeued first
        (PRIORITY_INTERRUPT=0 before PRIORITY_NORMAL=1 before
        PRIORITY_BACKGROUND=2), regardless of insertion order.

        Arrange:
          - A fresh AsyncAgentInbox (agent_id="agent-b")
          - Send a background message (priority=2) first
          - Then send an interrupt message (priority=0)

        Act:
          - first  = await inbox.receive()
          - second = await inbox.receive()

        Assert:
          - first.payload  == "high"  (priority 0 — interrupt, highest urgency)
          - second.payload == "low"   (priority 2 — background, lowest urgency)
        """
        AsyncAgentInbox, InboxMessage = _import_async_agent_inbox()

        inbox = AsyncAgentInbox(agent_id="agent-b")

        # Insert lower-urgency message first to confirm ordering is by priority
        # not by insertion order.
        await inbox.send(InboxMessage(priority=2, topic="background", payload="low"))
        await inbox.send(InboxMessage(priority=0, topic="interrupt", payload="high"))

        first = await inbox.receive()
        second = await inbox.receive()

        assert first.payload == "high", (
            f"Expected first dequeued message to be the interrupt (priority 0) "
            f"with payload 'high', but got payload={first.payload!r} "
            f"priority={first.priority}"
        )
        assert second.payload == "low", (
            f"Expected second dequeued message to be the background (priority 2) "
            f"with payload 'low', but got payload={second.payload!r} "
            f"priority={second.priority}"
        )

    @pytest.mark.asyncio
    async def test_try_receive_returns_none_when_empty(self):
        """
        T043 — RED until async_agent_inbox.py is implemented.

        Verifies that try_receive() returns None immediately when no messages
        are pending, instead of blocking.  This is the non-blocking contract
        from FR-016.

        Arrange:
          - A fresh AsyncAgentInbox (agent_id="agent-c") with no messages

        Act:
          - result = await inbox.try_receive()

        Assert:
          - result is None
          - The call completes without blocking (implicitly verified by the test
            finishing within the default asyncio timeout)

        Note on API: try_receive is async per the FR-016 spec contract so that
        it can be safely awaited inside async contexts without blocking the event
        loop.  The return value is Optional[InboxMessage].
        """
        AsyncAgentInbox, InboxMessage = _import_async_agent_inbox()

        inbox = AsyncAgentInbox(agent_id="agent-c")

        result = await inbox.try_receive()

        assert result is None, (
            f"Expected try_receive() to return None on an empty inbox, "
            f"but got {result!r}"
        )

    @pytest.mark.asyncio
    async def test_try_receive_returns_message_when_available(self):
        """
        T043-ext — Companion to T043: confirms try_receive() returns the message
        (not None) when one IS present in the queue.

        Arrange:
          - inbox with one pre-sent message

        Act:
          - result = await inbox.try_receive()

        Assert:
          - result is not None
          - result.payload matches the sent payload
        """
        AsyncAgentInbox, InboxMessage = _import_async_agent_inbox()

        inbox = AsyncAgentInbox(agent_id="agent-d")
        msg = InboxMessage(priority=1, topic="ping", payload="pong")
        await inbox.send(msg)

        result = await inbox.try_receive()

        assert result is not None, (
            "try_receive() returned None even though a message was available"
        )
        assert result.payload == "pong", (
            f"Expected payload 'pong', got {result.payload!r}"
        )

    @pytest.mark.asyncio
    async def test_qsize_reflects_queue_depth(self):
        """
        Structural test: qsize() tracks the current queue depth correctly
        across send and receive operations.
        """
        AsyncAgentInbox, InboxMessage = _import_async_agent_inbox()

        inbox = AsyncAgentInbox(agent_id="agent-e")

        assert inbox.qsize() == 0, "Fresh inbox should have qsize() == 0"
        assert inbox.empty() is True, "Fresh inbox should report empty() == True"

        await inbox.send(InboxMessage(priority=1, topic="t", payload="a"))
        await inbox.send(InboxMessage(priority=1, topic="t", payload="b"))

        assert inbox.qsize() == 2, (
            f"After 2 sends, expected qsize() == 2, got {inbox.qsize()}"
        )
        assert inbox.empty() is False

        await inbox.receive()
        assert inbox.qsize() == 1, (
            f"After 1 receive, expected qsize() == 1, got {inbox.qsize()}"
        )
