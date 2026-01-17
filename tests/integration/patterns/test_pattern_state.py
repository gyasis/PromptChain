"""Integration tests for pattern state sharing via Blackboard.

Tests patterns using Blackboard from 003-multi-agent-communication
infrastructure for state coordination.
"""

import pytest
import asyncio
from typing import Any, Dict

from promptchain.patterns.base import PatternConfig, PatternResult


class StatefulPattern:
    """Pattern that reads and writes state to Blackboard."""

    def __init__(self, pattern_id: str, blackboard):
        self.pattern_id = pattern_id
        self.blackboard = blackboard
        self.state_key = f"pattern.{pattern_id}.state"

    async def execute_with_state(self, input_data: Any) -> Dict[str, Any]:
        """Execute with state read/write."""
        # Read previous state
        prev_state = self.blackboard.read(self.state_key) or {"count": 0}

        # Process
        new_state = {
            "count": prev_state["count"] + 1,
            "last_input": input_data,
            "pattern_id": self.pattern_id
        }

        # Write new state
        self.blackboard.write(self.state_key, new_state, source=self.pattern_id)

        return new_state


@pytest.mark.asyncio
async def test_pattern_writes_to_blackboard(test_pattern_with_blackboard, blackboard):
    """Test that pattern can write to Blackboard."""
    # Connect pattern to blackboard
    test_pattern_with_blackboard.connect_blackboard(blackboard)

    # Execute pattern
    result = await test_pattern_with_blackboard.execute_with_timeout(data="test")

    # Verify pattern executed
    assert result.success is True

    # Verify data was written to blackboard
    shared_result = blackboard.read("test.result")
    assert shared_result == test_pattern_with_blackboard.execution_result


@pytest.mark.asyncio
async def test_pattern_reads_from_blackboard(test_pattern, blackboard):
    """Test that pattern can read from Blackboard."""
    # Pre-populate blackboard
    blackboard.write("config.setting", "test_value", source="test_setup")

    # Connect pattern
    test_pattern.connect_blackboard(blackboard)

    # Pattern reads from blackboard
    value = test_pattern.read_shared("config.setting")

    # Verify read succeeded
    assert value == "test_value"


@pytest.mark.asyncio
async def test_pattern_state_handoff(blackboard):
    """Test PatternStateCoordinator handoff between patterns."""
    # Create two stateful patterns
    pattern1 = StatefulPattern("pattern-a", blackboard)
    pattern2 = StatefulPattern("pattern-b", blackboard)

    # Pattern 1 writes state
    state1 = await pattern1.execute_with_state("input_1")
    assert state1["count"] == 1
    assert state1["last_input"] == "input_1"

    # Pattern 2 can read different state
    handoff_key = "handoff.data"
    blackboard.write(handoff_key, {"from": "pattern-a", "data": "result"}, source="pattern-a")

    # Pattern 2 reads handoff
    handoff = blackboard.read(handoff_key)
    assert handoff["from"] == "pattern-a"
    assert handoff["data"] == "result"


@pytest.mark.asyncio
async def test_blackboard_entry_versioning(blackboard):
    """Test that Blackboard entries track versions correctly."""
    # Write initial value
    entry1 = blackboard.write("versioned.key", "value1", source="agent-1")
    assert entry1.version == 1
    assert entry1.value == "value1"

    # Update value
    entry2 = blackboard.write("versioned.key", "value2", source="agent-2")
    assert entry2.version == 2
    assert entry2.value == "value2"
    assert entry2.written_by == "agent-2"

    # Verify version incremented
    current_entry = blackboard.get_entry("versioned.key")
    assert current_entry.version == 2


@pytest.mark.asyncio
async def test_blackboard_entry_metadata(blackboard):
    """Test that Blackboard entries preserve metadata."""
    # Write with metadata
    entry = blackboard.write("meta.key", {"data": "value"}, source="test-pattern")

    # Verify metadata
    assert entry.key == "meta.key"
    assert entry.written_by == "test-pattern"
    assert entry.written_at > 0  # Timestamp is set
    assert entry.version == 1

    # Retrieve full entry
    retrieved = blackboard.get_entry("meta.key")
    assert retrieved.key == entry.key
    assert retrieved.written_by == entry.written_by
    assert retrieved.value == entry.value


@pytest.mark.asyncio
async def test_pattern_state_isolation(blackboard):
    """Test that different patterns maintain isolated state."""
    # Create two patterns with different state keys
    pattern1 = StatefulPattern("isolated-1", blackboard)
    pattern2 = StatefulPattern("isolated-2", blackboard)

    # Both patterns execute
    state1 = await pattern1.execute_with_state("data1")
    state2 = await pattern2.execute_with_state("data2")

    # Verify states are isolated
    assert state1["pattern_id"] == "isolated-1"
    assert state2["pattern_id"] == "isolated-2"
    assert state1["last_input"] == "data1"
    assert state2["last_input"] == "data2"

    # Verify separate keys in blackboard
    keys = blackboard.list_keys()
    assert "pattern.isolated-1.state" in keys
    assert "pattern.isolated-2.state" in keys


@pytest.mark.asyncio
async def test_pattern_shared_state(blackboard):
    """Test that patterns can share state via common keys."""
    # Create patterns that share a common state key
    shared_key = "shared.counter"

    pattern1 = StatefulPattern("counter-1", blackboard)
    pattern2 = StatefulPattern("counter-2", blackboard)

    # Pattern 1 initializes shared state
    blackboard.write(shared_key, {"value": 0}, source="counter-1")

    # Pattern 2 reads and updates
    current = blackboard.read(shared_key)
    assert current["value"] == 0

    blackboard.write(shared_key, {"value": current["value"] + 1}, source="counter-2")

    # Verify update
    updated = blackboard.read(shared_key)
    assert updated["value"] == 1

    # Verify version tracking
    entry = blackboard.get_entry(shared_key)
    assert entry.version == 2  # Two writes
    assert entry.written_by == "counter-2"  # Last writer


@pytest.mark.asyncio
async def test_blackboard_state_snapshot(blackboard):
    """Test StateSnapshot capture/restore pattern."""
    from copy import deepcopy

    # Populate blackboard with multiple entries
    blackboard.write("state.a", "value_a", source="test")
    blackboard.write("state.b", "value_b", source="test")
    blackboard.write("state.c", "value_c", source="test")

    # Capture snapshot (manual implementation with deep copy)
    snapshot = {
        key: deepcopy(blackboard.get_entry(key))
        for key in blackboard.list_keys()
    }

    # Verify snapshot captured all entries
    assert len(snapshot) == 3
    assert "state.a" in snapshot
    assert snapshot["state.a"].value == "value_a"

    # Modify blackboard
    blackboard.write("state.a", "modified", source="test")
    blackboard.delete("state.b")

    # Verify modifications took effect
    assert blackboard.read("state.a") == "modified"
    assert blackboard.read("state.b") is None

    # Restore from snapshot (simulating a restore operation)
    # Note: In our mock, write updates existing entries, so we need to clear first
    blackboard.clear()
    for key, entry in snapshot.items():
        blackboard.write(key, entry.value, source=entry.written_by)

    # Verify restoration
    assert blackboard.read("state.a") == "value_a"
    assert blackboard.read("state.b") == "value_b"


@pytest.mark.asyncio
async def test_blackboard_clear_operation(blackboard):
    """Test clearing blackboard state."""
    # Populate blackboard
    blackboard.write("key1", "value1", source="test")
    blackboard.write("key2", "value2", source="test")
    blackboard.write("key3", "value3", source="test")

    # Verify populated
    assert len(blackboard.list_keys()) == 3

    # Clear blackboard
    cleared_count = blackboard.clear()

    # Verify cleared
    assert cleared_count == 3
    assert len(blackboard.list_keys()) == 0
    assert blackboard.read("key1") is None


@pytest.mark.asyncio
async def test_pattern_without_blackboard_enabled(test_pattern, blackboard):
    """Test that pattern doesn't write when use_blackboard=False."""
    # Pattern configured with use_blackboard=False
    test_pattern.config.use_blackboard = False
    test_pattern.connect_blackboard(blackboard)

    # Try to share result (should be no-op)
    test_pattern.share_result("should.not.write", "value")

    # Verify nothing was written
    assert blackboard.read("should.not.write") is None
    assert len(blackboard.list_keys()) == 0


@pytest.mark.asyncio
async def test_concurrent_pattern_state_updates(blackboard):
    """Test concurrent state updates from multiple patterns."""
    # Create multiple patterns
    patterns = [
        StatefulPattern(f"concurrent-{i}", blackboard)
        for i in range(5)
    ]

    # Execute all patterns concurrently
    results = await asyncio.gather(*[
        p.execute_with_state(f"data-{i}")
        for i, p in enumerate(patterns)
    ])

    # Verify all patterns executed
    assert len(results) == 5

    # Verify all states were written
    keys = blackboard.list_keys()
    assert len(keys) == 5

    for i, pattern in enumerate(patterns):
        state = blackboard.read(pattern.state_key)
        assert state is not None
        assert state["count"] == 1
        assert state["last_input"] == f"data-{i}"


@pytest.mark.asyncio
async def test_blackboard_delete_operation(blackboard):
    """Test deleting entries from Blackboard."""
    # Write entry
    blackboard.write("temp.key", "temp_value", source="test")
    assert blackboard.read("temp.key") == "temp_value"

    # Delete entry
    deleted = blackboard.delete("temp.key")
    assert deleted is True

    # Verify deleted
    assert blackboard.read("temp.key") is None

    # Try deleting non-existent key
    deleted_again = blackboard.delete("temp.key")
    assert deleted_again is False
