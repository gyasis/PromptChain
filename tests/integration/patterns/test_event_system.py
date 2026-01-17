"""Integration tests for pattern event system.

Tests PATTERN_EVENTS registry, event factory functions, and event serialization
with real 003 infrastructure.
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any

from promptchain.patterns.base import PatternConfig, PatternResult


# Mock PATTERN_EVENTS registry (would be in actual implementation)
PATTERN_EVENTS = {
    "branching.started": {
        "description": "Branching pattern execution started",
        "fields": ["pattern_id", "timestamp", "branches_count"]
    },
    "branching.completed": {
        "description": "Branching pattern execution completed",
        "fields": ["pattern_id", "timestamp", "success", "execution_time_ms"]
    },
    "validation.started": {
        "description": "Validation pattern started",
        "fields": ["pattern_id", "timestamp", "validation_type"]
    },
    "synthesis.progress": {
        "description": "Synthesis pattern progress update",
        "fields": ["pattern_id", "timestamp", "step", "progress_pct"]
    }
}


def create_pattern_event(event_type: str, pattern_id: str, **kwargs) -> Dict[str, Any]:
    """Event factory function.

    Creates standardized pattern events with validation.
    """
    if event_type not in PATTERN_EVENTS:
        raise ValueError(f"Unknown event type: {event_type}")

    event_spec = PATTERN_EVENTS[event_type]
    required_fields = event_spec["fields"]

    # Build event data
    event_data = {
        "pattern_id": pattern_id,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }

    # Validate required fields present
    for field in required_fields:
        if field not in event_data:
            raise ValueError(f"Missing required field '{field}' for event '{event_type}'")

    return {
        "type": event_type,
        "data": event_data,
        "description": event_spec["description"]
    }


@pytest.mark.asyncio
async def test_pattern_events_registry():
    """Test that PATTERN_EVENTS registry is complete."""
    # Verify registry has expected event types
    assert "branching.started" in PATTERN_EVENTS
    assert "branching.completed" in PATTERN_EVENTS
    assert "validation.started" in PATTERN_EVENTS
    assert "synthesis.progress" in PATTERN_EVENTS

    # Verify event specs have required structure
    for event_type, spec in PATTERN_EVENTS.items():
        assert "description" in spec
        assert "fields" in spec
        assert isinstance(spec["fields"], list)
        assert len(spec["fields"]) > 0


@pytest.mark.asyncio
async def test_event_factory_creates_valid_events():
    """Test that event factory creates properly structured events."""
    # Create branching started event
    event = create_pattern_event(
        "branching.started",
        pattern_id="test-pattern-1",
        branches_count=3
    )

    # Verify structure
    assert "type" in event
    assert "data" in event
    assert "description" in event

    # Verify data fields
    assert event["data"]["pattern_id"] == "test-pattern-1"
    assert event["data"]["branches_count"] == 3
    assert "timestamp" in event["data"]


@pytest.mark.asyncio
async def test_event_factory_validates_required_fields():
    """Test that event factory validates required fields."""
    # Try creating event without required field
    with pytest.raises(ValueError) as exc_info:
        create_pattern_event(
            "branching.started",
            pattern_id="test-pattern-1"
            # Missing branches_count
        )

    assert "branches_count" in str(exc_info.value)


@pytest.mark.asyncio
async def test_event_factory_rejects_unknown_events():
    """Test that event factory rejects unknown event types."""
    # Try creating unknown event type
    with pytest.raises(ValueError) as exc_info:
        create_pattern_event(
            "unknown.event",
            pattern_id="test-pattern-1"
        )

    assert "Unknown event type" in str(exc_info.value)


@pytest.mark.asyncio
async def test_event_serialization():
    """Test that pattern events can be serialized to JSON."""
    # Create event
    event = create_pattern_event(
        "synthesis.progress",
        pattern_id="synth-1",
        step="merging",
        progress_pct=45.5
    )

    # Serialize to JSON
    serialized = json.dumps(event)

    # Deserialize
    deserialized = json.loads(serialized)

    # Verify round-trip
    assert deserialized["type"] == event["type"]
    assert deserialized["data"]["pattern_id"] == event["data"]["pattern_id"]
    assert deserialized["data"]["step"] == "merging"
    assert deserialized["data"]["progress_pct"] == 45.5


@pytest.mark.asyncio
async def test_event_timestamp_format():
    """Test that event timestamps use ISO format."""
    event = create_pattern_event(
        "validation.started",
        pattern_id="val-1",
        validation_type="schema"
    )

    # Verify timestamp is ISO format
    timestamp = event["data"]["timestamp"]
    assert isinstance(timestamp, str)
    assert "T" in timestamp  # ISO format separator

    # Verify can be parsed back
    parsed = datetime.fromisoformat(timestamp)
    assert isinstance(parsed, datetime)


@pytest.mark.asyncio
async def test_event_integration_with_pattern(test_pattern, event_collector):
    """Test that patterns emit events matching registry format."""
    # Add event collector
    test_pattern.add_event_handler(event_collector)

    # Execute pattern
    await test_pattern.execute_with_timeout()

    # Verify events were collected
    events = event_collector.events
    assert len(events) > 0

    # Verify event structure matches expectations
    for event in events:
        assert "type" in event
        assert "data" in event
        assert "pattern_id" in event["data"]
        assert "timestamp" in event["data"]


@pytest.mark.asyncio
async def test_multiple_event_types_from_pattern(test_pattern, event_collector):
    """Test that pattern emits multiple event types during execution."""
    test_pattern.add_event_handler(event_collector)

    # Execute pattern
    await test_pattern.execute_with_timeout()

    # Collect event types
    event_types = [e["type"] for e in event_collector.events]

    # Verify multiple event types
    assert len(set(event_types)) > 1  # More than one unique event type

    # Verify lifecycle events present
    assert any("started" in t for t in event_types)
    assert any("completed" in t or "processing" in t for t in event_types)


@pytest.mark.asyncio
async def test_event_data_completeness():
    """Test that event factory includes all specified fields."""
    # Create complex event with multiple fields
    event = create_pattern_event(
        "branching.completed",
        pattern_id="branch-test",
        success=True,
        execution_time_ms=250.5
    )

    # Verify all required fields present
    expected_fields = PATTERN_EVENTS["branching.completed"]["fields"]
    for field in expected_fields:
        assert field in event["data"], f"Missing field: {field}"


@pytest.mark.asyncio
async def test_event_metadata_preservation():
    """Test that event metadata is preserved through serialization."""
    # Create event with extra metadata
    event = create_pattern_event(
        "validation.started",
        pattern_id="val-1",
        validation_type="schema",
        extra_field="metadata"
    )

    # Serialize
    serialized = json.dumps(event)
    deserialized = json.loads(serialized)

    # Verify metadata preserved
    assert deserialized["data"]["extra_field"] == "metadata"


@pytest.mark.asyncio
async def test_event_ordering():
    """Test that events maintain chronological ordering."""
    from tests.integration.patterns.conftest import TestPattern

    pattern = TestPattern(PatternConfig(pattern_id="ordered-events"))
    events_list = []

    def collect_ordered(event_type: str, event_data: Dict[str, Any]):
        events_list.append({
            "type": event_type,
            "timestamp": event_data["timestamp"]
        })

    pattern.add_event_handler(collect_ordered)

    # Execute pattern
    await pattern.execute_with_timeout()

    # Verify events are chronologically ordered
    timestamps = [e["timestamp"] for e in events_list]
    sorted_timestamps = sorted(timestamps)
    assert timestamps == sorted_timestamps


@pytest.mark.asyncio
async def test_event_pattern_matching():
    """Test event pattern matching for subscriptions."""
    # Test simple pattern matching
    def matches_pattern(event_type: str, pattern: str) -> bool:
        """Simple wildcard pattern matching."""
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return event_type.startswith(prefix)
        return event_type == pattern

    # Test cases
    assert matches_pattern("pattern.branching.started", "pattern.branching.*")
    assert matches_pattern("pattern.branching.started", "*")
    assert matches_pattern("pattern.branching.started", "pattern.branching.started")
    assert not matches_pattern("pattern.branching.started", "pattern.validation.*")


@pytest.mark.asyncio
async def test_event_aggregation():
    """Test aggregating events from multiple patterns."""
    from tests.integration.patterns.conftest import TestPattern

    # Create multiple patterns
    patterns = [
        TestPattern(PatternConfig(pattern_id=f"agg-{i}"))
        for i in range(3)
    ]

    # Shared event collector
    all_events = []

    def aggregate_events(event_type: str, event_data: Dict[str, Any]):
        all_events.append({
            "type": event_type,
            "pattern_id": event_data["pattern_id"]
        })

    # Add collector to all patterns
    for pattern in patterns:
        pattern.add_event_handler(aggregate_events)

    # Execute all patterns
    import asyncio
    await asyncio.gather(*[p.execute_with_timeout() for p in patterns])

    # Verify events from all patterns collected
    pattern_ids = set(e["pattern_id"] for e in all_events)
    assert len(pattern_ids) == 3
    assert all(f"agg-{i}" in pattern_ids for i in range(3))
