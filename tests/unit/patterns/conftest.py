"""Shared fixtures for pattern adapter tests.

Patterns report events through ``add_event_handler``; ``event_collector.collect``
has the handler signature ``(event_type, data)``.
"""

import pytest
from unittest.mock import AsyncMock
from typing import List, Dict, Any


@pytest.fixture
def mock_lightrag_core():
    """Mock HybridLightRAGCore with all search methods."""
    mock = AsyncMock()

    # Mock search methods
    mock.local_query = AsyncMock(return_value={
        "results": ["local result 1", "local result 2"],
        "metadata": {"search_type": "local"}
    })

    mock.global_query = AsyncMock(return_value={
        "results": ["global result 1", "global result 2"],
        "metadata": {"search_type": "global"}
    })

    mock.hybrid_query = AsyncMock(return_value={
        "results": ["hybrid result 1", "hybrid result 2"],
        "metadata": {"search_type": "hybrid"}
    })

    mock.naive_query = AsyncMock(return_value={
        "results": ["naive result 1"],
        "metadata": {"search_type": "naive"}
    })

    mock.agentic_search = AsyncMock(return_value={
        "results": ["agentic result 1"],
        "metadata": {"search_type": "agentic", "hops": 2}
    })

    return mock


@pytest.fixture
def event_collector():
    """Collect events emitted by patterns for testing."""
    events = []

    class EventCollector:
        def __init__(self):
            self.events = events

        def collect(self, event_type: str, data: Dict[str, Any]):
            """Collect an event."""
            self.events.append({
                "type": event_type,
                "data": data
            })

        def get_events(self, event_type: str = None) -> List[Dict]:
            """Get collected events, optionally filtered by type."""
            if event_type:
                return [e for e in self.events if e["type"] == event_type]
            return self.events.copy()

        def clear(self):
            """Clear collected events."""
            self.events.clear()

    collector = EventCollector()
    yield collector
    collector.clear()
