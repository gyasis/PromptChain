"""Integration tests for PromptChain callback system."""

import pytest
import asyncio
from typing import List
from promptchain import PromptChain
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType
from promptchain.utils.execution_callback import CallbackFunction


@pytest.fixture
def event_collector():
    """Fixture that provides a list to collect events."""
    events = []
    return events


@pytest.fixture
def sync_callback(event_collector):
    """Fixture that provides a synchronous callback."""
    def callback(event: ExecutionEvent):
        event_collector.append(event)
    return callback


@pytest.fixture
def async_callback(event_collector):
    """Fixture that provides an asynchronous callback."""
    async def callback(event: ExecutionEvent):
        event_collector.append(event)
    return callback


class TestPromptChainCallbackRegistration:
    """Test callback registration on PromptChain."""

    def test_register_callback(self, sync_callback):
        """Test registering a callback."""
        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Test instruction"],
            verbose=False
        )

        chain.register_callback(sync_callback)
        assert chain.callback_manager.has_callbacks()

    def test_register_callback_with_filter(self, sync_callback):
        """Test registering a callback with event filter."""
        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Test instruction"],
            verbose=False
        )

        event_filter = {ExecutionEventType.CHAIN_START, ExecutionEventType.CHAIN_END}
        chain.register_callback(sync_callback, event_filter)

        assert chain.callback_manager.has_callbacks()
        assert len(chain.callback_manager.callbacks) == 1

    def test_unregister_callback(self, sync_callback):
        """Test unregistering a callback."""
        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Test instruction"],
            verbose=False
        )

        chain.register_callback(sync_callback)
        assert chain.callback_manager.has_callbacks()

        result = chain.unregister_callback(sync_callback)
        assert result is True
        assert not chain.callback_manager.has_callbacks()

    def test_clear_callbacks(self, sync_callback):
        """Test clearing all callbacks."""
        chain = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Test instruction"],
            verbose=False
        )

        chain.register_callback(sync_callback)
        chain.clear_callbacks()
        assert not chain.callback_manager.has_callbacks()


class TestPromptChainCallbackExecution:
    """Test callback execution during chain processing."""

    @pytest.mark.asyncio
    async def test_chain_start_end_events(self, event_collector, async_callback):
        """Test that CHAIN_START and CHAIN_END events are emitted."""
        # Simple function-based chain to avoid LLM calls
        def simple_func(input_str: str) -> str:
            return f"Processed: {input_str}"

        chain = PromptChain(
            models=[],
            instructions=[simple_func],
            verbose=False
        )

        chain.register_callback(async_callback)

        result = await chain.process_prompt_async("test input")

        # Check that we received events
        assert len(event_collector) >= 2

        # Find CHAIN_START and CHAIN_END events
        chain_start = next((e for e in event_collector if e.event_type == ExecutionEventType.CHAIN_START), None)
        chain_end = next((e for e in event_collector if e.event_type == ExecutionEventType.CHAIN_END), None)

        assert chain_start is not None
        assert chain_end is not None
        assert chain_start.metadata["initial_input"] == "test input"
        assert "execution_time_ms" in chain_end.metadata

    @pytest.mark.asyncio
    async def test_step_start_end_events(self, event_collector, async_callback):
        """Test that STEP_START and STEP_END events are emitted."""
        def func1(input_str: str) -> str:
            return f"Step1: {input_str}"

        def func2(input_str: str) -> str:
            return f"Step2: {input_str}"

        chain = PromptChain(
            models=[],
            instructions=[func1, func2],
            verbose=False
        )

        chain.register_callback(async_callback)

        result = await chain.process_prompt_async("test")

        # Should have STEP_START and STEP_END for each step
        step_starts = [e for e in event_collector if e.event_type == ExecutionEventType.STEP_START]
        step_ends = [e for e in event_collector if e.event_type == ExecutionEventType.STEP_END]

        assert len(step_starts) == 2
        assert len(step_ends) == 2

        # Check step numbers
        assert step_starts[0].step_number == 1
        assert step_starts[1].step_number == 2
        assert step_ends[0].step_number == 1
        assert step_ends[1].step_number == 2

    @pytest.mark.asyncio
    async def test_filtered_callback(self, event_collector, async_callback):
        """Test that filtered callbacks only receive specified events."""
        def simple_func(input_str: str) -> str:
            return f"Processed: {input_str}"

        chain = PromptChain(
            models=[],
            instructions=[simple_func],
            verbose=False
        )

        # Register callback that only receives CHAIN events
        event_filter = {ExecutionEventType.CHAIN_START, ExecutionEventType.CHAIN_END}
        chain.register_callback(async_callback, event_filter)

        result = await chain.process_prompt_async("test")

        # Should only have CHAIN_START and CHAIN_END
        assert len(event_collector) == 2
        assert all(e.event_type in event_filter for e in event_collector)

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self):
        """Test that multiple callbacks all receive events."""
        events1 = []
        events2 = []

        def callback1(event: ExecutionEvent):
            events1.append(event)

        async def callback2(event: ExecutionEvent):
            events2.append(event)

        def simple_func(input_str: str) -> str:
            return f"Processed: {input_str}"

        chain = PromptChain(
            models=[],
            instructions=[simple_func],
            verbose=False
        )

        chain.register_callback(callback1)
        chain.register_callback(callback2)

        result = await chain.process_prompt_async("test")

        # Both callbacks should have received events
        assert len(events1) > 0
        assert len(events2) > 0
        assert len(events1) == len(events2)

    @pytest.mark.asyncio
    async def test_chain_error_event(self, event_collector, async_callback):
        """Test that CHAIN_ERROR event is emitted on error."""
        def failing_func(input_str: str) -> str:
            raise ValueError("Intentional test error")

        chain = PromptChain(
            models=[],
            instructions=[failing_func],
            verbose=False
        )

        chain.register_callback(async_callback)

        with pytest.raises(ValueError):
            await chain.process_prompt_async("test")

        # Should have received CHAIN_START and CHAIN_ERROR
        chain_start = next((e for e in event_collector if e.event_type == ExecutionEventType.CHAIN_START), None)
        chain_error = next((e for e in event_collector if e.event_type == ExecutionEventType.CHAIN_ERROR), None)

        assert chain_start is not None
        assert chain_error is not None
        assert "error" in chain_error.metadata
        assert chain_error.metadata["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_step_metadata(self, event_collector, async_callback):
        """Test that step events contain correct metadata."""
        def simple_func(input_str: str) -> str:
            return f"Processed: {input_str}"

        chain = PromptChain(
            models=[],
            instructions=[simple_func],
            verbose=False
        )

        chain.register_callback(async_callback)

        result = await chain.process_prompt_async("test input")

        # Find STEP_START and STEP_END
        step_start = next((e for e in event_collector if e.event_type == ExecutionEventType.STEP_START), None)
        step_end = next((e for e in event_collector if e.event_type == ExecutionEventType.STEP_END), None)

        assert step_start is not None
        assert step_end is not None

        # Check metadata
        assert "input_length" in step_start.metadata
        assert "execution_time_ms" in step_end.metadata
        assert "step_type" in step_end.metadata
        assert step_end.metadata["step_type"] == "function"

    @pytest.mark.asyncio
    async def test_event_ordering(self, event_collector, async_callback):
        """Test that events are emitted in correct order."""
        def func1(input_str: str) -> str:
            return "output1"

        chain = PromptChain(
            models=[],
            instructions=[func1],
            verbose=False
        )

        chain.register_callback(async_callback)

        result = await chain.process_prompt_async("test")

        # Extract event types in order
        event_types = [e.event_type for e in event_collector]

        # Expected order: CHAIN_START, STEP_START, STEP_END, CHAIN_END
        expected_order = [
            ExecutionEventType.CHAIN_START,
            ExecutionEventType.STEP_START,
            ExecutionEventType.STEP_END,
            ExecutionEventType.CHAIN_END
        ]

        assert event_types == expected_order

    def test_sync_process_prompt(self, event_collector, sync_callback):
        """Test that callbacks work with synchronous process_prompt."""
        def simple_func(input_str: str) -> str:
            return f"Processed: {input_str}"

        chain = PromptChain(
            models=[],
            instructions=[simple_func],
            verbose=False
        )

        chain.register_callback(sync_callback)

        result = chain.process_prompt("test")

        # Should have received events
        assert len(event_collector) > 0
        assert any(e.event_type == ExecutionEventType.CHAIN_START for e in event_collector)
        assert any(e.event_type == ExecutionEventType.CHAIN_END for e in event_collector)
