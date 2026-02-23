"""Unit tests for ExecutionEvent and ExecutionCallback."""

import pytest
import asyncio
from datetime import datetime
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType
from promptchain.utils.execution_callback import (
    CallbackManager,
    FilteredCallback,
    SyncCallback,
    AsyncCallback,
)


class TestExecutionEvent:
    """Test ExecutionEvent dataclass."""

    def test_event_creation(self):
        """Test creating an execution event."""
        event = ExecutionEvent(
            event_type=ExecutionEventType.CHAIN_START,
            step_number=1,
            model_name="gpt-4.1-mini-2025-04-14",
            metadata={"test": "value"}
        )

        assert event.event_type == ExecutionEventType.CHAIN_START
        assert event.step_number == 1
        assert event.model_name == "gpt-4.1-mini-2025-04-14"
        assert event.metadata["test"] == "value"
        assert isinstance(event.timestamp, datetime)

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = ExecutionEvent(
            event_type=ExecutionEventType.STEP_END,
            step_number=2,
            metadata={"execution_time_ms": 150}
        )

        event_dict = event.to_dict()

        assert event_dict["event_type"] == "STEP_END"
        assert event_dict["step_number"] == 2
        assert event_dict["metadata"]["execution_time_ms"] == 150
        assert "timestamp" in event_dict

    def test_event_to_summary_dict(self):
        """Test converting event to summary dictionary."""
        event = ExecutionEvent(
            event_type=ExecutionEventType.CHAIN_ERROR,
            step_number=3,
            model_name="claude-3",
            metadata={
                "error": "Test error",
                "execution_time_ms": 200,
                "tokens_used": 500
            }
        )

        summary = event.to_summary_dict()

        assert summary["event"] == "CHAIN_ERROR"
        assert summary["step"] == 3
        assert summary["model"] == "claude-3"
        assert summary["error"] == "Test error"
        assert summary["time_ms"] == 200
        assert summary["tokens"] == 500


class TestCallbackManager:
    """Test CallbackManager functionality."""

    def test_register_sync_callback(self):
        """Test registering a synchronous callback."""
        manager = CallbackManager()
        events_received = []

        def sync_callback(event: ExecutionEvent):
            events_received.append(event)

        manager.register(sync_callback)
        assert manager.has_callbacks()

    def test_register_async_callback(self):
        """Test registering an asynchronous callback."""
        manager = CallbackManager()

        async def async_callback(event: ExecutionEvent):
            pass

        manager.register(async_callback)
        assert manager.has_callbacks()

    def test_register_with_filter(self):
        """Test registering callback with event filter."""
        manager = CallbackManager()

        def filtered_callback(event: ExecutionEvent):
            pass

        event_filter = {ExecutionEventType.CHAIN_START, ExecutionEventType.CHAIN_END}
        manager.register(filtered_callback, event_filter)

        assert manager.has_callbacks()
        assert len(manager.callbacks) == 1
        assert manager.callbacks[0].event_filter == event_filter

    def test_unregister_callback(self):
        """Test unregistering a callback."""
        manager = CallbackManager()

        def callback(event: ExecutionEvent):
            pass

        manager.register(callback)
        assert manager.has_callbacks()

        result = manager.unregister(callback)
        assert result is True
        assert not manager.has_callbacks()

    def test_unregister_nonexistent_callback(self):
        """Test unregistering a callback that doesn't exist."""
        manager = CallbackManager()

        def callback(event: ExecutionEvent):
            pass

        result = manager.unregister(callback)
        assert result is False

    def test_clear_callbacks(self):
        """Test clearing all callbacks."""
        manager = CallbackManager()

        def callback1(event: ExecutionEvent):
            pass

        def callback2(event: ExecutionEvent):
            pass

        manager.register(callback1)
        manager.register(callback2)
        assert len(manager.callbacks) == 2

        manager.clear()
        assert len(manager.callbacks) == 0
        assert not manager.has_callbacks()

    @pytest.mark.asyncio
    async def test_emit_to_sync_callback(self):
        """Test emitting event to synchronous callback."""
        manager = CallbackManager()
        events_received = []

        def sync_callback(event: ExecutionEvent):
            events_received.append(event)

        manager.register(sync_callback)

        event = ExecutionEvent(
            event_type=ExecutionEventType.CHAIN_START,
            metadata={"test": "value"}
        )

        await manager.emit(event)

        assert len(events_received) == 1
        assert events_received[0].event_type == ExecutionEventType.CHAIN_START

    @pytest.mark.asyncio
    async def test_emit_to_async_callback(self):
        """Test emitting event to asynchronous callback."""
        manager = CallbackManager()
        events_received = []

        async def async_callback(event: ExecutionEvent):
            events_received.append(event)

        manager.register(async_callback)

        event = ExecutionEvent(
            event_type=ExecutionEventType.STEP_END,
            step_number=1
        )

        await manager.emit(event)

        assert len(events_received) == 1
        assert events_received[0].step_number == 1

    @pytest.mark.asyncio
    async def test_emit_with_filter(self):
        """Test that filtered callbacks only receive matching events."""
        manager = CallbackManager()
        received_events = []

        def filtered_callback(event: ExecutionEvent):
            received_events.append(event)

        # Only receive CHAIN_START and CHAIN_END
        event_filter = {ExecutionEventType.CHAIN_START, ExecutionEventType.CHAIN_END}
        manager.register(filtered_callback, event_filter)

        # Emit various events
        await manager.emit(ExecutionEvent(event_type=ExecutionEventType.CHAIN_START))
        await manager.emit(ExecutionEvent(event_type=ExecutionEventType.STEP_START))
        await manager.emit(ExecutionEvent(event_type=ExecutionEventType.STEP_END))
        await manager.emit(ExecutionEvent(event_type=ExecutionEventType.CHAIN_END))

        # Should only have received CHAIN_START and CHAIN_END
        assert len(received_events) == 2
        assert received_events[0].event_type == ExecutionEventType.CHAIN_START
        assert received_events[1].event_type == ExecutionEventType.CHAIN_END

    @pytest.mark.asyncio
    async def test_emit_to_multiple_callbacks(self):
        """Test emitting to multiple callbacks."""
        manager = CallbackManager()
        callback1_events = []
        callback2_events = []

        def callback1(event: ExecutionEvent):
            callback1_events.append(event)

        async def callback2(event: ExecutionEvent):
            callback2_events.append(event)

        manager.register(callback1)
        manager.register(callback2)

        event = ExecutionEvent(event_type=ExecutionEventType.CHAIN_START)
        await manager.emit(event)

        assert len(callback1_events) == 1
        assert len(callback2_events) == 1

    def test_emit_sync(self):
        """Test synchronous emit wrapper."""
        manager = CallbackManager()
        events_received = []

        def sync_callback(event: ExecutionEvent):
            events_received.append(event)

        manager.register(sync_callback)

        event = ExecutionEvent(event_type=ExecutionEventType.CHAIN_START)
        manager.emit_sync(event)

        assert len(events_received) == 1


class TestFilteredCallback:
    """Test FilteredCallback wrapper."""

    def test_should_handle_with_no_filter(self):
        """Test that callback with no filter handles all events."""
        def callback(event: ExecutionEvent):
            pass

        filtered = FilteredCallback(callback, None)

        assert filtered.should_handle(ExecutionEventType.CHAIN_START)
        assert filtered.should_handle(ExecutionEventType.STEP_END)
        assert filtered.should_handle(ExecutionEventType.CHAIN_ERROR)

    def test_should_handle_with_filter(self):
        """Test that callback with filter only handles matching events."""
        def callback(event: ExecutionEvent):
            pass

        event_filter = {ExecutionEventType.CHAIN_START, ExecutionEventType.CHAIN_END}
        filtered = FilteredCallback(callback, event_filter)

        assert filtered.should_handle(ExecutionEventType.CHAIN_START)
        assert filtered.should_handle(ExecutionEventType.CHAIN_END)
        assert not filtered.should_handle(ExecutionEventType.STEP_START)
        assert not filtered.should_handle(ExecutionEventType.TOOL_CALL_START)

    @pytest.mark.asyncio
    async def test_call_sync_callback(self):
        """Test calling a synchronous callback."""
        events_received = []

        def sync_callback(event: ExecutionEvent):
            events_received.append(event)

        filtered = FilteredCallback(sync_callback)
        event = ExecutionEvent(event_type=ExecutionEventType.CHAIN_START)

        await filtered(event)

        assert len(events_received) == 1

    @pytest.mark.asyncio
    async def test_call_async_callback(self):
        """Test calling an asynchronous callback."""
        events_received = []

        async def async_callback(event: ExecutionEvent):
            events_received.append(event)

        filtered = FilteredCallback(async_callback)
        event = ExecutionEvent(event_type=ExecutionEventType.CHAIN_START)

        await filtered(event)

        assert len(events_received) == 1

    @pytest.mark.asyncio
    async def test_call_respects_filter(self):
        """Test that __call__ respects event filter."""
        events_received = []

        def callback(event: ExecutionEvent):
            events_received.append(event)

        event_filter = {ExecutionEventType.CHAIN_START}
        filtered = FilteredCallback(callback, event_filter)

        # Should be called
        await filtered(ExecutionEvent(event_type=ExecutionEventType.CHAIN_START))

        # Should not be called
        await filtered(ExecutionEvent(event_type=ExecutionEventType.STEP_END))

        assert len(events_received) == 1
        assert events_received[0].event_type == ExecutionEventType.CHAIN_START
