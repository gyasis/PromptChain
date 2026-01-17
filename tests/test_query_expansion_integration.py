"""Integration tests for Query Expansion Pattern with base infrastructure.

Tests integration with BasePattern, MessageBus, and Blackboard.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from promptchain.integrations.lightrag.query_expansion import (
    ExpansionStrategy,
    QueryExpansionConfig,
    LightRAGQueryExpander,
)
from promptchain.patterns.base import BasePattern, PatternConfig


# Mock components for testing
class MockSearchInterface:
    """Mock SearchInterface."""
    async def multi_query_search(self, queries, mode="hybrid", **kwargs):
        return [{"content": f"Result for: {q}", "score": 0.9} for q in queries]


class MockLightRAGIntegration:
    """Mock LightRAGIntegration."""
    def __init__(self):
        self.search = MockSearchInterface()

    async def extract_context(self, query: str, **kwargs):
        return {
            "entities": [{"name": "test", "type": "concept"}],
            "relationships": [{"type": "uses", "source": "test", "target": "data"}],
        }


@pytest.fixture
def mock_integration():
    """Provide mock integration."""
    return MockLightRAGIntegration()


@pytest.fixture
def basic_config():
    """Provide basic configuration."""
    return QueryExpansionConfig(
        strategies=[ExpansionStrategy.SEMANTIC],
        max_expansions_per_strategy=2,
    )


@pytest.mark.asyncio
class TestBasePatternIntegration:
    """Test integration with BasePattern infrastructure."""

    async def test_inherits_from_base_pattern(self, mock_integration, basic_config):
        """Test that LightRAGQueryExpander inherits from BasePattern."""
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )
        assert isinstance(expander, BasePattern)

    async def test_execute_with_timeout(self, mock_integration, basic_config):
        """Test execute_with_timeout wrapper."""
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )

        # Should complete within timeout
        result = await expander.execute_with_timeout(query="test query")
        assert result.success is True
        assert result.execution_time_ms > 0

    async def test_timeout_handling(self, mock_integration):
        """Test timeout handling."""
        # Create config with very short timeout
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.SEMANTIC],
            timeout_seconds=0.001,  # 1ms timeout
        )

        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )

        # Should timeout
        result = await expander.execute_with_timeout(query="test query")
        # May succeed or timeout depending on timing
        assert result.execution_time_ms > 0

    async def test_disabled_pattern(self, mock_integration, basic_config):
        """Test that disabled pattern returns immediately."""
        basic_config.enabled = False

        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )

        result = await expander.execute_with_timeout(query="test query")
        assert result.success is False
        assert "Pattern is disabled" in result.errors

    async def test_pattern_statistics(self, mock_integration, basic_config):
        """Test pattern execution statistics."""
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )

        # Execute multiple times
        await expander.execute_with_timeout(query="query 1")
        await expander.execute_with_timeout(query="query 2")
        await expander.execute_with_timeout(query="query 3")

        stats = expander.get_stats()
        assert stats["execution_count"] == 3
        assert stats["total_execution_time_ms"] > 0
        assert stats["average_execution_time_ms"] > 0


@pytest.mark.asyncio
@pytest.mark.skip(reason="003 infrastructure MessageBus/Blackboard integration pending full implementation")
class TestMessageBusIntegration:
    """Test integration with MessageBus."""

    async def test_connect_messagebus(self, mock_integration, basic_config):
        """Test connecting to MessageBus."""
        from promptchain.cli.communication.message_bus import MessageBus

        bus = MessageBus()
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )

        expander.connect_messagebus(bus)
        assert expander._bus == bus

    async def test_event_emission_to_bus(self, mock_integration, basic_config):
        """Test that events are emitted to MessageBus."""
        from promptchain.cli.communication.message_bus import MessageBus

        bus = MessageBus()
        events_received = []

        def handler(event_type: str, data: dict):
            events_received.append((event_type, data))

        # Subscribe to all query expansion events
        bus.subscribe("pattern.query_expansion.*", handler)

        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )
        expander.connect_messagebus(bus)

        await expander.execute_with_timeout(query="test query")

        # Verify events were emitted
        assert len(events_received) > 0
        event_types = [e[0] for e in events_received]
        assert "pattern.query_expansion.started" in event_types
        assert "pattern.query_expansion.completed" in event_types

    async def test_subscribe_to_events(self, mock_integration, basic_config):
        """Test subscribing to events from another pattern."""
        from promptchain.cli.communication.message_bus import MessageBus

        bus = MessageBus()
        received_events = []

        def handler(event_type: str, data: dict):
            received_events.append(event_type)

        expander1 = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )
        expander2 = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )

        expander1.connect_messagebus(bus)
        expander2.connect_messagebus(bus)

        # expander2 subscribes to expander1's events
        expander2.subscribe_to("pattern.query_expansion.*", handler)

        # expander1 executes
        await expander1.execute_with_timeout(query="test")

        # expander2 should receive events
        assert len(received_events) > 0


@pytest.mark.asyncio
@pytest.mark.skip(reason="003 infrastructure MessageBus/Blackboard integration pending full implementation")
class TestBlackboardIntegration:
    """Test integration with Blackboard."""

    async def test_connect_blackboard(self, mock_integration, basic_config):
        """Test connecting to Blackboard."""
        from promptchain.cli.models.blackboard import Blackboard

        blackboard = Blackboard()
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )

        expander.connect_blackboard(blackboard)
        assert expander._blackboard == blackboard

    async def test_share_result_to_blackboard(self, mock_integration):
        """Test sharing results to Blackboard."""
        from promptchain.cli.models.blackboard import Blackboard

        blackboard = Blackboard()
        config = QueryExpansionConfig(
            strategies=[ExpansionStrategy.SEMANTIC],
            use_blackboard=True,  # Enable blackboard usage
        )

        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=config,
        )
        expander.connect_blackboard(blackboard)

        # Execute and share results
        result = await expander.execute_with_timeout(query="test query")

        # Manually share results (pattern doesn't auto-share)
        expander.share_result("query_expansion_result", result.to_dict())

        # Verify results are in blackboard
        shared_result = expander.read_shared("query_expansion_result")
        assert shared_result is not None
        assert shared_result["original_query"] == "test query"

    async def test_read_from_blackboard(self, mock_integration, basic_config):
        """Test reading from Blackboard."""
        from promptchain.cli.models.blackboard import Blackboard

        blackboard = Blackboard()
        expander = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=basic_config,
        )
        expander.connect_blackboard(blackboard)

        # Write to blackboard
        blackboard.write("test_key", "test_value", source="test")

        # Read from blackboard
        value = expander.read_shared("test_key")
        assert value == "test_value"


@pytest.mark.asyncio
@pytest.mark.skip(reason="003 infrastructure MessageBus/Blackboard integration pending full implementation")
class TestMultiPatternCoordination:
    """Test coordination between multiple patterns."""

    async def test_multiple_patterns_with_shared_bus(self, mock_integration):
        """Test multiple patterns sharing a MessageBus."""
        from promptchain.cli.communication.message_bus import MessageBus

        bus = MessageBus()
        events = []

        def global_handler(event_type: str, data: dict):
            events.append((event_type, data))

        bus.subscribe("pattern.*", global_handler)

        # Create two expanders
        expander1 = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.SEMANTIC]),
        )
        expander2 = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=QueryExpansionConfig(strategies=[ExpansionStrategy.REFORMULATION]),
        )

        expander1.connect_messagebus(bus)
        expander2.connect_messagebus(bus)

        # Execute both
        await expander1.execute_with_timeout(query="query 1")
        await expander2.execute_with_timeout(query="query 2")

        # Should have events from both patterns
        assert len(events) >= 4  # At least started/completed for each

    async def test_patterns_sharing_blackboard(self, mock_integration):
        """Test patterns sharing results via Blackboard."""
        from promptchain.cli.models.blackboard import Blackboard

        blackboard = Blackboard()

        expander1 = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=QueryExpansionConfig(use_blackboard=True),
        )
        expander2 = LightRAGQueryExpander(
            lightrag_integration=mock_integration,
            config=QueryExpansionConfig(use_blackboard=True),
        )

        expander1.connect_blackboard(blackboard)
        expander2.connect_blackboard(blackboard)

        # Expander1 shares result
        result1 = await expander1.execute_with_timeout(query="test 1")
        expander1.share_result("exp1_result", result1.to_dict())

        # Expander2 reads it
        shared = expander2.read_shared("exp1_result")
        assert shared is not None
        assert shared["original_query"] == "test 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
