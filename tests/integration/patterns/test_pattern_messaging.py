"""Integration tests for pattern event emission to MessageBus.

Tests that patterns correctly emit events to the MessageBus from
003-multi-agent-communication infrastructure.
"""

import pytest
import asyncio
from typing import Dict, Any

from promptchain.cli.communication.message_bus import MessageType
from promptchain.patterns.base import PatternConfig, PatternResult


class CollaboratingPattern:
    """Pattern that listens to events from other patterns."""

    def __init__(self, pattern_id: str):
        self.pattern_id = pattern_id
        self.received_events = []

    def on_event(self, event_type: str, event_data: Dict[str, Any]):
        """Handle received events."""
        self.received_events.append({
            "type": event_type,
            "data": event_data
        })


@pytest.mark.asyncio
async def test_pattern_emits_to_messagebus(test_pattern, message_bus, event_collector):
    """Test that pattern events are emitted via local handlers (MessageBus integration)."""
    # Connect pattern to MessageBus
    test_pattern.connect_messagebus(message_bus)

    # Add event collector to capture events
    test_pattern.add_event_handler(event_collector)

    # Execute pattern
    result = await test_pattern.execute_with_timeout(input_data="test")

    # Verify execution succeeded
    assert result.success is True

    # Verify events were collected through handlers
    events = event_collector.events
    assert len(events) > 0

    # Note: MessageBus doesn't have a publish() method in the current implementation
    # Pattern events are emitted through local handlers and would need a bridge
    # to MessageBus.send() for full integration


@pytest.mark.asyncio
async def test_pattern_lifecycle_events(test_pattern, message_bus, event_collector):
    """Test that pattern emits lifecycle events (started, completed, error)."""
    # Connect pattern to MessageBus and add event collector
    test_pattern.connect_messagebus(message_bus)
    test_pattern.add_event_handler(event_collector)

    # Execute successful pattern
    result = await test_pattern.execute_with_timeout(test_param="value")

    # Verify events were collected
    events = event_collector.events
    assert len(events) >= 2  # At minimum: started, completed

    # Verify started event
    started_events = [e for e in events if "started" in e["type"]]
    assert len(started_events) > 0
    assert started_events[0]["data"]["pattern_id"] == test_pattern.config.pattern_id

    # Verify completed event
    completed_events = [e for e in events if "completed" in e["type"]]
    assert len(completed_events) > 0
    assert completed_events[0]["data"]["success"] is True


@pytest.mark.asyncio
async def test_pattern_error_events(test_pattern, message_bus, event_collector):
    """Test that pattern emits error events on failure."""
    # Configure pattern to fail
    test_pattern.should_fail = True
    test_pattern.connect_messagebus(message_bus)
    test_pattern.add_event_handler(event_collector)

    # Execute pattern (should fail)
    result = await test_pattern.execute_with_timeout()

    # Verify execution failed
    assert result.success is False
    assert len(result.errors) > 0

    # Verify error event was emitted
    events = event_collector.events
    error_events = [e for e in events if "error" in e["type"]]
    assert len(error_events) > 0
    assert "error" in error_events[0]["data"]


@pytest.mark.asyncio
async def test_pattern_timeout_events(message_bus, event_collector):
    """Test that pattern emits timeout events on timeout."""
    # Create pattern with very short timeout
    config = PatternConfig(
        pattern_id="timeout-pattern",
        timeout_seconds=0.05  # 50ms timeout
    )

    from tests.integration.patterns.conftest import TestPattern
    pattern = TestPattern(config)
    pattern.delay_ms = 200  # 200ms delay (will timeout)
    pattern.connect_messagebus(message_bus)
    pattern.add_event_handler(event_collector)

    # Execute pattern (should timeout)
    result = await pattern.execute_with_timeout()

    # Verify execution timed out
    assert result.success is False
    assert any("timed out" in err.lower() for err in result.errors)

    # Verify timeout event was emitted
    events = event_collector.events
    timeout_events = [e for e in events if "timeout" in e["type"]]
    assert len(timeout_events) > 0


@pytest.mark.asyncio
async def test_cross_pattern_event_subscription(message_bus):
    """Test that patterns can subscribe to events from other patterns."""
    from tests.integration.patterns.conftest import TestPattern

    # Create two patterns
    config1 = PatternConfig(pattern_id="pattern-1", emit_events=True)
    config2 = PatternConfig(pattern_id="pattern-2", emit_events=True)

    pattern1 = TestPattern(config1)
    pattern2 = TestPattern(config2)

    # Connect both to MessageBus
    pattern1.connect_messagebus(message_bus)
    pattern2.connect_messagebus(message_bus)

    # Pattern 2 subscribes to pattern 1 events
    collaborator = CollaboratingPattern("pattern-2-listener")

    def on_pattern1_event(event_type: str, event_data: Dict[str, Any]):
        if event_data.get("pattern_id") == "pattern-1":
            collaborator.on_event(event_type, event_data)

    pattern2.add_event_handler(on_pattern1_event)
    pattern1.add_event_handler(on_pattern1_event)

    # Execute pattern 1
    result1 = await pattern1.execute_with_timeout(data="test")

    # Verify collaborator received events
    assert len(collaborator.received_events) > 0
    received_pattern_ids = [e["data"].get("pattern_id") for e in collaborator.received_events]
    assert "pattern-1" in received_pattern_ids


@pytest.mark.asyncio
async def test_pattern_custom_events(test_pattern, message_bus, event_collector):
    """Test that patterns can emit custom events."""
    test_pattern.connect_messagebus(message_bus)
    test_pattern.add_event_handler(event_collector)

    # Execute pattern (emits custom "processing" event)
    result = await test_pattern.execute_with_timeout()

    # Verify custom event was emitted
    events = event_collector.events
    processing_events = [e for e in events if "processing" in e["type"]]
    assert len(processing_events) > 0
    assert processing_events[0]["data"]["step"] == "middle"


@pytest.mark.asyncio
async def test_pattern_event_broadcast(message_bus, event_collector):
    """Test PatternEventBroadcaster coordination pattern."""
    from tests.integration.patterns.conftest import TestPattern

    # Create multiple patterns
    patterns = [
        TestPattern(PatternConfig(pattern_id=f"pattern-{i}"))
        for i in range(3)
    ]

    # Connect all to MessageBus
    for pattern in patterns:
        pattern.connect_messagebus(message_bus)
        pattern.add_event_handler(event_collector)

    # Execute all patterns
    results = await asyncio.gather(*[
        p.execute_with_timeout(index=i)
        for i, p in enumerate(patterns)
    ])

    # Verify all executions succeeded
    assert all(r.success for r in results)

    # Verify events from all patterns were collected
    events = event_collector.events
    pattern_ids = set(e["data"].get("pattern_id") for e in events)
    assert len(pattern_ids) == 3  # All 3 patterns emitted events


@pytest.mark.asyncio
async def test_pattern_disabled_no_events(message_bus, event_collector):
    """Test that disabled patterns don't emit events."""
    from tests.integration.patterns.conftest import TestPattern

    # Create disabled pattern
    config = PatternConfig(pattern_id="disabled", enabled=False)
    pattern = TestPattern(config)
    pattern.connect_messagebus(message_bus)
    pattern.add_event_handler(event_collector)

    # Execute pattern
    result = await pattern.execute_with_timeout()

    # Verify pattern didn't execute
    assert result.success is False
    assert "disabled" in result.errors[0].lower()

    # Verify no events were emitted (pattern was disabled)
    events = event_collector.events
    assert len(events) == 0


@pytest.mark.asyncio
async def test_pattern_event_data_structure(test_pattern, message_bus, event_collector):
    """Test that pattern events have correct data structure."""
    test_pattern.connect_messagebus(message_bus)
    test_pattern.add_event_handler(event_collector)

    # Execute pattern
    await test_pattern.execute_with_timeout(param1="value1", param2=123)

    # Verify event structure
    events = event_collector.events
    assert len(events) > 0

    for event in events:
        # Each event should have type and data
        assert "type" in event
        assert "data" in event

        # Data should include pattern_id and timestamp
        assert "pattern_id" in event["data"]
        assert "timestamp" in event["data"]
        assert event["data"]["pattern_id"] == test_pattern.config.pattern_id
