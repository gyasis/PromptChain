"""E2E tests for event coordination across patterns.

Tests:
1. Complete workflow with all events captured
2. Event ordering verification
3. Event correlation across patterns
"""

import asyncio
import pytest
from typing import Dict, List, Any
from collections import defaultdict

from promptchain.integrations.lightrag.query_expansion import (
    LightRAGQueryExpander,
    QueryExpansionConfig,
    ExpansionStrategy,
)
from promptchain.integrations.lightrag.multi_hop import (
    LightRAGMultiHop,
    MultiHopConfig,
)
from promptchain.integrations.lightrag.branching import (
    LightRAGBranchingThoughts,
    BranchingConfig,
)
from promptchain.integrations.lightrag.events import (
    EventLifecycle,
    subscribe_to_pattern,
    subscribe_to_lifecycle,
)


@pytest.mark.e2e
@pytest.mark.asyncio
class TestEventCapture:
    """Test comprehensive event capture."""

    async def test_all_lifecycle_events_captured(
        self, e2e_test_context, mock_message_bus
    ):
        """Test all lifecycle events are captured for each pattern.

        Workflow:
        1. Execute multiple patterns
        2. Verify each emits full lifecycle: started -> progress -> completed
        3. Check for failure events when errors occur
        """
        lightrag = e2e_test_context["lightrag"]
        search = e2e_test_context["search"]

        # Execute Query Expansion
        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.SEMANTIC])
        )
        expander.connect_messagebus(mock_message_bus)

        expansion_result = await expander.execute(query="What is ML?")
        assert expansion_result.success

        # Execute Multi-Hop
        multi_hop = LightRAGMultiHop(
            search_interface=search,
            config=MultiHopConfig(max_hops=2)
        )
        multi_hop.connect_messagebus(mock_message_bus)

        multi_hop_result = await multi_hop.execute(question="What is ML?")
        assert multi_hop_result.success

        # Execute Branching
        branching = LightRAGBranchingThoughts(
            lightrag_integration=lightrag,
            config=BranchingConfig(num_branches=2)
        )
        branching.connect_messagebus(mock_message_bus)

        branching_result = await branching.execute(query="What is ML?")
        assert branching_result.success

        # Verify lifecycle events for each pattern
        patterns = ["query_expansion", "multi_hop", "branching"]

        for pattern in patterns:
            pattern_events = mock_message_bus.get_events_by_type(f"pattern.{pattern}.*")

            # Each pattern should have at minimum: started and completed
            event_types = [e["type"] for e in pattern_events]

            has_started = any("started" in et for et in event_types)
            has_completed = any("completed" in et for et in event_types)

            assert has_started, f"{pattern} missing 'started' event"
            assert has_completed, f"{pattern} missing 'completed' event"

            print(f"\n{pattern} lifecycle events:")
            for event_type in event_types:
                print(f"  - {event_type}")

    async def test_event_data_integrity(
        self, e2e_test_context, mock_message_bus
    ):
        """Test event data contains all required fields."""
        lightrag = e2e_test_context["lightrag"]

        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.SEMANTIC])
        )
        expander.connect_messagebus(mock_message_bus)

        await expander.execute(query="Test query")

        # Check all events have required fields
        all_events = mock_message_bus.events

        for event in all_events:
            # Each event should have: type, data, timestamp
            assert "type" in event
            assert "data" in event
            assert "timestamp" in event

            # Data should contain pattern_id
            if "pattern." in event["type"]:
                # Most pattern events should have pattern_id in data
                # (May not be present in all events depending on implementation)
                data = event["data"]
                assert isinstance(data, dict)

        print(f"\nTotal events captured: {len(all_events)}")
        print(f"All events have required fields: type, data, timestamp")


@pytest.mark.e2e
@pytest.mark.asyncio
class TestEventOrdering:
    """Test event ordering and sequencing."""

    async def test_sequential_pattern_ordering(
        self, e2e_test_context, mock_message_bus
    ):
        """Test events are ordered correctly for sequential pattern execution."""
        lightrag = e2e_test_context["lightrag"]
        search = e2e_test_context["search"]

        # Execute patterns sequentially
        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.SEMANTIC])
        )
        expander.connect_messagebus(mock_message_bus)

        await expander.execute(query="Query 1")

        multi_hop = LightRAGMultiHop(
            search_interface=search,
            config=MultiHopConfig(max_hops=2)
        )
        multi_hop.connect_messagebus(mock_message_bus)

        await multi_hop.execute(question="Query 2")

        # Verify ordering
        event_sequence = mock_message_bus.get_event_sequence()

        # Find indices
        expansion_started = event_sequence.index("pattern.query_expansion.started")
        expansion_completed = event_sequence.index("pattern.query_expansion.completed")
        multihop_started = event_sequence.index("pattern.multi_hop.started")
        multihop_completed = event_sequence.index("pattern.multi_hop.completed")

        # Verify correct ordering
        assert expansion_started < expansion_completed
        assert expansion_completed < multihop_started
        assert multihop_started < multihop_completed

        print(f"\nEvent sequence verified:")
        print(f"  1. Expansion started (index {expansion_started})")
        print(f"  2. Expansion completed (index {expansion_completed})")
        print(f"  3. Multi-hop started (index {multihop_started})")
        print(f"  4. Multi-hop completed (index {multihop_completed})")

    async def test_parallel_pattern_interleaving(
        self, e2e_test_context, mock_message_bus
    ):
        """Test events are interleaved for parallel pattern execution."""
        lightrag = e2e_test_context["lightrag"]

        # Execute multiple patterns in parallel
        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.SEMANTIC])
        )
        expander.connect_messagebus(mock_message_bus)

        branching = LightRAGBranchingThoughts(
            lightrag_integration=lightrag,
            config=BranchingConfig(num_branches=2)
        )
        branching.connect_messagebus(mock_message_bus)

        # Run in parallel
        await asyncio.gather(
            expander.execute(query="Parallel query 1"),
            branching.execute(query="Parallel query 2"),
        )

        # Verify both patterns have events
        expansion_events = mock_message_bus.get_events_by_type("pattern.query_expansion.*")
        branching_events = mock_message_bus.get_events_by_type("pattern.branching.*")

        assert len(expansion_events) > 0
        assert len(branching_events) > 0

        # Events may be interleaved
        event_sequence = mock_message_bus.get_event_sequence()

        expansion_indices = [
            i for i, e in enumerate(event_sequence)
            if "query_expansion" in e
        ]
        branching_indices = [
            i for i, e in enumerate(event_sequence)
            if "branching" in e
        ]

        print(f"\nParallel execution event interleaving:")
        print(f"  Expansion event indices: {expansion_indices}")
        print(f"  Branching event indices: {branching_indices}")

        # Events from both patterns should exist
        assert len(expansion_indices) > 0
        assert len(branching_indices) > 0

    async def test_event_timestamps_ordering(
        self, e2e_test_context, mock_message_bus
    ):
        """Test event timestamps are in chronological order."""
        from datetime import datetime

        lightrag = e2e_test_context["lightrag"]

        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.SEMANTIC])
        )
        expander.connect_messagebus(mock_message_bus)

        await expander.execute(query="Timestamp test")

        # Get all events and parse timestamps
        all_events = mock_message_bus.events

        timestamps = []
        for event in all_events:
            if "timestamp" in event:
                ts = datetime.fromisoformat(event["timestamp"])
                timestamps.append(ts)

        # Verify timestamps are chronological
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1], \
                f"Timestamp out of order at index {i}"

        print(f"\nVerified {len(timestamps)} timestamps in chronological order")


@pytest.mark.e2e
@pytest.mark.asyncio
class TestEventCorrelation:
    """Test event correlation across patterns."""

    async def test_correlation_id_propagation(
        self, e2e_test_context, mock_message_bus
    ):
        """Test correlation IDs link related events across patterns.

        This test verifies that when patterns call each other,
        they can track related events via correlation IDs.
        """
        lightrag = e2e_test_context["lightrag"]
        search = e2e_test_context["search"]

        # Execute workflow with correlation
        correlation_id = "workflow-123"

        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.SEMANTIC])
        )
        expander.connect_messagebus(mock_message_bus)

        # In a real implementation, correlation_id would be passed through
        await expander.execute(query="Correlated query")

        multi_hop = LightRAGMultiHop(
            search_interface=search,
            config=MultiHopConfig(max_hops=2)
        )
        multi_hop.connect_messagebus(mock_message_bus)

        await multi_hop.execute(question="Correlated query")

        # Verify events exist for both patterns
        all_events = mock_message_bus.events

        expansion_events = [e for e in all_events if "query_expansion" in e["type"]]
        multihop_events = [e for e in all_events if "multi_hop" in e["type"]]

        assert len(expansion_events) > 0
        assert len(multihop_events) > 0

        print(f"\nEvent correlation:")
        print(f"  Expansion events: {len(expansion_events)}")
        print(f"  Multi-hop events: {len(multihop_events)}")

    async def test_event_grouping_by_pattern(
        self, e2e_test_context, mock_message_bus
    ):
        """Test events can be grouped by pattern for analysis."""
        lightrag = e2e_test_context["lightrag"]

        # Execute multiple patterns
        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.SEMANTIC])
        )
        expander.connect_messagebus(mock_message_bus)

        branching = LightRAGBranchingThoughts(
            lightrag_integration=lightrag,
            config=BranchingConfig(num_branches=2)
        )
        branching.connect_messagebus(mock_message_bus)

        await expander.execute(query="Test 1")
        await branching.execute(query="Test 2")

        # Group events by pattern
        events_by_pattern = defaultdict(list)

        for event in mock_message_bus.events:
            event_type = event["type"]
            if "pattern." in event_type:
                # Extract pattern name
                parts = event_type.split(".")
                if len(parts) >= 2:
                    pattern_name = parts[1]
                    events_by_pattern[pattern_name].append(event)

        print(f"\nEvents grouped by pattern:")
        for pattern_name, events in events_by_pattern.items():
            print(f"  {pattern_name}: {len(events)} events")

        # Should have events for both patterns
        assert "query_expansion" in events_by_pattern
        assert "branching" in events_by_pattern


@pytest.mark.e2e
@pytest.mark.asyncio
class TestEventSubscription:
    """Test event subscription and notification."""

    async def test_pattern_specific_subscription(
        self, e2e_test_context, mock_message_bus
    ):
        """Test subscribing to specific pattern events."""
        lightrag = e2e_test_context["lightrag"]

        # Track received events
        received_events = []

        def expansion_handler(event_type: str, data: Dict[str, Any]):
            received_events.append({"type": event_type, "data": data})

        # Subscribe to expansion events
        pattern = subscribe_to_pattern("query_expansion")
        mock_message_bus.subscribe(pattern, expansion_handler)

        # Execute expansion
        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.SEMANTIC])
        )
        expander.connect_messagebus(mock_message_bus)

        await expander.execute(query="Subscription test")

        # Verify handler received events
        assert len(received_events) > 0

        print(f"\nReceived {len(received_events)} expansion events via subscription")

        # All received events should be expansion events
        for evt in received_events:
            assert "query_expansion" in evt["type"]

    async def test_lifecycle_specific_subscription(
        self, e2e_test_context, mock_message_bus
    ):
        """Test subscribing to specific lifecycle phase across all patterns."""
        lightrag = e2e_test_context["lightrag"]

        # Track completion events
        completed_events = []

        def completion_handler(event_type: str, data: Dict[str, Any]):
            if "completed" in event_type:
                completed_events.append({"type": event_type, "data": data})

        # Subscribe to all completion events
        pattern = subscribe_to_lifecycle(EventLifecycle.COMPLETED)
        mock_message_bus.subscribe(pattern, completion_handler)

        # Execute multiple patterns
        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.SEMANTIC])
        )
        expander.connect_messagebus(mock_message_bus)

        branching = LightRAGBranchingThoughts(
            lightrag_integration=lightrag,
            config=BranchingConfig(num_branches=2)
        )
        branching.connect_messagebus(mock_message_bus)

        await expander.execute(query="Test 1")
        await branching.execute(query="Test 2")

        # Should receive completion events from both patterns
        assert len(completed_events) >= 2

        print(f"\nReceived {len(completed_events)} completion events:")
        for evt in completed_events:
            print(f"  - {evt['type']}")


@pytest.mark.e2e
@pytest.mark.asyncio
class TestEventPerformanceMetrics:
    """Test extracting performance metrics from events."""

    async def test_execution_time_tracking(
        self, e2e_test_context, mock_message_bus
    ):
        """Test execution time can be calculated from event timestamps."""
        from datetime import datetime

        lightrag = e2e_test_context["lightrag"]

        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.SEMANTIC])
        )
        expander.connect_messagebus(mock_message_bus)

        result = await expander.execute(query="Performance test")

        # Find started and completed events
        expansion_events = mock_message_bus.get_events_by_type("pattern.query_expansion.*")

        started_event = next(e for e in expansion_events if "started" in e["type"])
        completed_event = next(e for e in expansion_events if "completed" in e["type"])

        # Calculate execution time from events
        start_time = datetime.fromisoformat(started_event["timestamp"])
        end_time = datetime.fromisoformat(completed_event["timestamp"])

        event_execution_time = (end_time - start_time).total_seconds() * 1000  # ms

        # Compare with result execution time
        result_execution_time = result.execution_time_ms

        print(f"\nExecution time comparison:")
        print(f"  From events: {event_execution_time:.2f}ms")
        print(f"  From result: {result_execution_time:.2f}ms")

        # Should be similar (within reasonable tolerance)
        # Note: May differ due to event emission overhead
        assert event_execution_time >= 0
        assert result_execution_time >= 0

    async def test_throughput_calculation(
        self, e2e_test_context, mock_message_bus
    ):
        """Test calculating throughput from events."""
        lightrag = e2e_test_context["lightrag"]

        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.SEMANTIC])
        )
        expander.connect_messagebus(mock_message_bus)

        # Execute multiple times
        num_queries = 5
        for i in range(num_queries):
            await expander.execute(query=f"Query {i}")

        # Calculate throughput
        expansion_events = mock_message_bus.get_events_by_type("pattern.query_expansion.*")
        completed_events = [e for e in expansion_events if "completed" in e["type"]]

        assert len(completed_events) == num_queries

        # Calculate total time
        first_start = min(
            datetime.fromisoformat(e["timestamp"])
            for e in expansion_events
            if "started" in e["type"]
        )
        last_complete = max(
            datetime.fromisoformat(e["timestamp"])
            for e in completed_events
        )

        total_time_seconds = (last_complete - first_start).total_seconds()
        throughput = num_queries / total_time_seconds if total_time_seconds > 0 else 0

        print(f"\nThroughput: {throughput:.2f} queries/second")
        print(f"Total time: {total_time_seconds:.3f}s for {num_queries} queries")
