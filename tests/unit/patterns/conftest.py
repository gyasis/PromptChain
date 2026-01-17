"""Shared fixtures for pattern adapter tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock
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
def mock_search_interface():
    """Mock SearchInterface for query expansion tests."""
    mock = AsyncMock()

    mock.multi_query_search = AsyncMock(return_value={
        "results": [
            {"text": "Result 1", "score": 0.9, "id": "1"},
            {"text": "Result 2", "score": 0.8, "id": "2"},
            {"text": "Result 3", "score": 0.7, "id": "3"}
        ],
        "metadata": {"queries_used": 3}
    })

    mock.search = AsyncMock(return_value={
        "results": [{"text": "Single result", "score": 0.85, "id": "s1"}]
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


@pytest.fixture
def mock_llm():
    """Mock LLM for scoring and generation tasks."""
    mock = AsyncMock()

    # Mock for hypothesis scoring
    mock.generate = AsyncMock(return_value='{"scores": [0.9, 0.7, 0.8]}')

    # Mock for query expansion
    mock.expand_query = AsyncMock(return_value=[
        "expanded query 1",
        "expanded query 2",
        "expanded query 3"
    ])

    return mock


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            "text": "The quick brown fox jumps over the lazy dog",
            "score": 0.95,
            "id": "doc1",
            "metadata": {"source": "test"}
        },
        {
            "text": "A fast auburn canine leaps above a sleepy hound",
            "score": 0.85,
            "id": "doc2",
            "metadata": {"source": "test"}
        },
        {
            "text": "Programming is the art of solving problems",
            "score": 0.75,
            "id": "doc3",
            "metadata": {"source": "test"}
        }
    ]


@pytest.fixture
def mock_cache():
    """Mock cache for speculative execution tests."""
    cache_storage = {}

    class MockCache:
        def __init__(self):
            self.storage = cache_storage

        async def get(self, key: str):
            return self.storage.get(key)

        async def set(self, key: str, value: Any, ttl: int = None):
            self.storage[key] = {
                "value": value,
                "ttl": ttl
            }

        async def delete(self, key: str):
            self.storage.pop(key, None)

        def clear(self):
            self.storage.clear()

    cache = MockCache()
    yield cache
    cache.clear()
