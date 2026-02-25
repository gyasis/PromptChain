"""
Unit tests for PubSubBus class.

T044: test_publish_triggers_all_subscribers_concurrently
T045: test_subscriber_exception_does_not_propagate
T046: test_unsubscribe_removes_callback
T047: test_publish_sync_works_from_non_async_context

Discovery results (2026-02-25):
  - promptchain/cli/communication/message_bus.py : EXISTS (MessageBus class only)
  - PubSubBus class                              : NOT FOUND in message_bus.py

PubSubBus is a separate class that must be added to (or imported from)
promptchain/cli/communication/message_bus.py per FR-017.

These tests are RED until PubSubBus is implemented with the following contract
(from specs/006-promptchain-improvements/contracts/async-execution.md FR-017):

    class PubSubBus:
        _subscribers: Dict[str, List[Callable]]   # topic -> list of async callbacks
        _lock: asyncio.Lock

        async def subscribe(
            self,
            topic: str,
            callback: Callable[[str, Any], Awaitable[None]]
        ) -> None:
            \"\"\"Register async callback for topic. Idempotent.\"\"\"

        async def unsubscribe(self, topic: str, callback: Callable) -> None:
            \"\"\"Remove callback for topic. No-op if not registered.\"\"\"

        async def publish(self, topic: str, payload: Any) -> None:
            \"\"\"
            Deliver payload to all subscribers concurrently via asyncio.gather().
            Per-subscriber exceptions are caught, logged, NOT propagated.
            Returns when all subscribers have been triggered (not completed).
            \"\"\"

        def publish_sync(self, topic: str, payload: Any) -> None:
            \"\"\"Synchronous wrapper. Safe to call from non-async context only.
            Uses asyncio.run() internally.\"\"\"

Note on subscribe/unsubscribe call style:
  The FR-017 spec defines subscribe() and unsubscribe() as *async* methods.
  Tests T044-T046 therefore await them.  If the final implementation makes
  these sync (acceptable simplification), the tests can drop the await —
  but the async variant is the authoritative contract.

Fan-out guarantee: asyncio.gather(*[cb(payload) for cb in subscribers])
  with return_exceptions=True so individual failures do not abort the fan-out.

Spec reference: specs/006-promptchain-improvements/contracts/async-execution.md (FR-017)
Branch: 006-promptchain-improvements (US4, Wave 5)
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _import_pubsub_bus():
    """
    Attempt to import PubSubBus from its canonical location.
    Raises ImportError with a clear message if not yet implemented.
    """
    from promptchain.cli.communication.message_bus import PubSubBus
    return PubSubBus


# ---------------------------------------------------------------------------
# T044–T047: PubSubBus tests
# ---------------------------------------------------------------------------

class TestPubSubBus:
    """
    Tests for PubSubBus publish/subscribe fan-out functionality.

    All tests follow Arrange-Act-Assert and use pure async callbacks so
    they are independent of any external LLM or network calls.
    """

    @pytest.mark.asyncio
    async def test_publish_triggers_all_subscribers_concurrently(self):
        """
        T044 — RED until PubSubBus is implemented.

        Verifies that publish() delivers the payload to every registered
        subscriber and that all three callbacks are invoked (concurrently via
        asyncio.gather).

        Arrange:
          - A fresh PubSubBus
          - Three asyncio.Event objects, one per subscriber
          - Three async callbacks, each setting its respective Event when called

        Act:
          - subscribe all three callbacks to "topic"
          - await bus.publish("topic", "data")

        Assert:
          - All three Events are set (each callback was invoked exactly once)
        """
        PubSubBus = _import_pubsub_bus()

        bus = PubSubBus()
        events = [asyncio.Event() for _ in range(3)]

        async def make_cb(evt: asyncio.Event):
            async def cb(payload):
                evt.set()
            return cb

        for evt in events:
            cb = await make_cb(evt)
            await bus.subscribe("topic", cb)

        await bus.publish("topic", "data")

        assert all(e.is_set() for e in events), (
            f"Not all subscribers were triggered. "
            f"Triggered: {[e.is_set() for e in events]}"
        )

    @pytest.mark.asyncio
    async def test_subscriber_exception_does_not_propagate(self):
        """
        T045 — RED until PubSubBus is implemented.

        Verifies that a RuntimeError / ValueError raised inside a subscriber
        callback is swallowed by publish() and does NOT propagate to the caller.

        This is the error-isolation guarantee from FR-017: individual subscriber
        failures must never abort the fan-out or crash the publisher.

        Arrange:
          - A fresh PubSubBus
          - One async callback that always raises ValueError("boom")

        Act:
          - subscribe bad_cb to "topic"
          - await bus.publish("topic", "data")  ← must NOT raise

        Assert:
          - No exception is raised by publish()
        """
        PubSubBus = _import_pubsub_bus()

        bus = PubSubBus()

        async def bad_cb(payload):
            raise ValueError("boom")

        await bus.subscribe("topic", bad_cb)

        # This must complete without raising any exception.
        await bus.publish("topic", "data")

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_callback(self):
        """
        T046 — RED until PubSubBus is implemented.

        Verifies that after unsubscribe() the removed callback is no longer
        invoked on subsequent publish() calls.

        Arrange:
          - A fresh PubSubBus
          - One async callback that appends its payload to a list
          - Subscribe and then immediately unsubscribe the callback

        Act:
          - await bus.publish("topic", "x")

        Assert:
          - called list is empty (callback was never invoked)
        """
        PubSubBus = _import_pubsub_bus()

        bus = PubSubBus()
        called = []

        async def cb(payload):
            called.append(payload)

        await bus.subscribe("topic", cb)
        await bus.unsubscribe("topic", cb)

        await bus.publish("topic", "x")

        assert called == [], (
            f"Expected no calls after unsubscribe, but callback received: {called}"
        )

    def test_publish_sync_works_from_non_async_context(self):
        """
        T047 — RED until PubSubBus is implemented.

        Verifies that publish_sync() can be called from a regular (non-async)
        function, runs the event loop internally (via asyncio.run()), and
        triggers all subscribers before returning.

        Arrange:
          - A fresh PubSubBus
          - One async callback that appends to a list
          - Subscribe the callback from outside any event loop

        Act:
          - bus.publish_sync("topic", "sync_data")  ← called from sync context

        Assert:
          - "sync_data" is in received (callback was invoked synchronously)

        Note: This test must subscribe synchronously.  If subscribe() is async,
        we run it via asyncio.run() here — matching publish_sync's own pattern.
        """
        PubSubBus = _import_pubsub_bus()

        bus = PubSubBus()
        received = []

        async def cb(payload):
            received.append(payload)

        # subscribe() may be async; run it via asyncio.run() if needed.
        # We detect this by checking the return type.
        import inspect
        sub_result = bus.subscribe("topic", cb)
        if inspect.isawaitable(sub_result):
            asyncio.run(sub_result)

        bus.publish_sync("topic", "sync_data")

        assert "sync_data" in received, (
            f"Expected 'sync_data' to be delivered synchronously, "
            f"but received: {received}"
        )

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_callback_is_noop(self):
        """
        Edge case: unsubscribe() on a callback that was never registered
        must be a no-op (must not raise KeyError or similar).

        Arrange:
          - A fresh PubSubBus
          - An async callback that was NEVER subscribed

        Act:
          - await bus.unsubscribe("topic", cb)

        Assert:
          - No exception raised
        """
        PubSubBus = _import_pubsub_bus()

        bus = PubSubBus()

        async def cb(payload):
            pass

        # Must not raise
        await bus.unsubscribe("topic", cb)

    @pytest.mark.asyncio
    async def test_publish_to_topic_with_no_subscribers_is_noop(self):
        """
        Edge case: publish() on a topic with zero subscribers must complete
        silently without error.

        Arrange:
          - A fresh PubSubBus with no subscribers

        Act:
          - await bus.publish("empty_topic", "payload")

        Assert:
          - No exception raised
        """
        PubSubBus = _import_pubsub_bus()

        bus = PubSubBus()

        # Must not raise
        await bus.publish("empty_topic", "payload")

    @pytest.mark.asyncio
    async def test_subscribe_is_idempotent(self):
        """
        Edge case: Subscribing the same callback twice must not cause it to
        be invoked twice per publish() — subscribe() is idempotent per spec.

        Arrange:
          - A fresh PubSubBus
          - One async callback registered twice to the same topic

        Act:
          - await bus.publish("topic", "msg")

        Assert:
          - Callback invocation count is exactly 1 (idempotent registration)
        """
        PubSubBus = _import_pubsub_bus()

        bus = PubSubBus()
        call_count = []

        async def cb(payload):
            call_count.append(1)

        await bus.subscribe("topic", cb)
        await bus.subscribe("topic", cb)  # idempotent — should not double-register

        await bus.publish("topic", "msg")

        assert len(call_count) == 1, (
            f"Expected callback to be invoked exactly once (idempotent subscribe), "
            f"but was invoked {len(call_count)} time(s)"
        )
