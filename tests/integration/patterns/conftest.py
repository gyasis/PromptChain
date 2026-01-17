"""Pytest fixtures for pattern integration tests.

Provides real MessageBus and Blackboard from 003 infrastructure
with mocked LightRAG dependencies.
"""

import pytest
import asyncio
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Any, Dict, List

from promptchain.cli.communication.message_bus import MessageBus, MessageType
from promptchain.cli.models.blackboard import BlackboardEntry
from promptchain.patterns.base import BasePattern, PatternConfig, PatternResult


# MessageBus fixtures


@pytest.fixture
def session_id() -> str:
    """Generate a unique session ID for testing."""
    return "test-session-123"


@pytest.fixture
def activity_log() -> List[Dict[str, Any]]:
    """Create a list to capture activity logs."""
    return []


@pytest.fixture
def activity_logger(activity_log):
    """Create an activity logger that captures to a list."""
    def log_activity(entry: Dict[str, Any]):
        activity_log.append(entry)
    return log_activity


@pytest.fixture
def message_bus(session_id, activity_logger):
    """Create a real MessageBus instance with activity logging."""
    return MessageBus(
        session_id=session_id,
        activity_logger=activity_logger
    )


# Blackboard fixtures


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary SQLite database file."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_blackboard.db"
    return db_path


@pytest.fixture
def blackboard_db(temp_db, session_id):
    """Create a real Blackboard SQLite database."""
    conn = sqlite3.connect(str(temp_db))
    cursor = conn.cursor()

    # Create blackboard table matching the schema from 003
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS blackboard (
            session_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT,
            written_by TEXT NOT NULL,
            written_at REAL NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (session_id, key)
        )
    """)

    conn.commit()
    return conn


class MockBlackboard:
    """Mock Blackboard with real BlackboardEntry support.

    Mimics the Blackboard interface from 003 infrastructure
    with in-memory storage for testing.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._storage: Dict[str, BlackboardEntry] = {}

    def write(self, key: str, value: Any, source: str) -> BlackboardEntry:
        """Write a value to the blackboard."""
        if key in self._storage:
            entry = self._storage[key]
            entry.update(value, source)
        else:
            entry = BlackboardEntry.create(key, value, source)
            self._storage[key] = entry
        return entry

    def read(self, key: str) -> Any:
        """Read a value from the blackboard."""
        if key in self._storage:
            return self._storage[key].value
        return None

    def get_entry(self, key: str) -> BlackboardEntry:
        """Get the full entry including metadata."""
        return self._storage.get(key)

    def list_keys(self) -> List[str]:
        """List all keys in the blackboard."""
        return list(self._storage.keys())

    def delete(self, key: str) -> bool:
        """Delete an entry from the blackboard."""
        if key in self._storage:
            del self._storage[key]
            return True
        return False

    def clear(self) -> int:
        """Clear all entries and return count."""
        count = len(self._storage)
        self._storage.clear()
        return count


@pytest.fixture
def blackboard(session_id) -> MockBlackboard:
    """Create a mock Blackboard instance."""
    return MockBlackboard(session_id)


# Pattern fixtures


class TestPattern(BasePattern):
    """Concrete test pattern for integration testing."""

    def __init__(self, config: PatternConfig = None):
        super().__init__(config)
        self.executed = False
        self.execution_args = {}
        self.execution_result = "test_result"
        self.should_fail = False
        self.delay_ms = 0

    async def execute(self, **kwargs) -> PatternResult:
        """Execute test pattern logic."""
        self.executed = True
        self.execution_args = kwargs

        # Simulate delay if configured
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000.0)

        # Emit test events
        self.emit_event("test.pattern.processing", {"step": "middle"})

        # Share to blackboard if configured
        if self.config.use_blackboard:
            self.share_result("test.result", self.execution_result)

        # Fail if configured
        if self.should_fail:
            raise ValueError("Test pattern configured to fail")

        return PatternResult(
            pattern_id=self.config.pattern_id,
            success=True,
            result=self.execution_result,
            execution_time_ms=float(self.delay_ms),
            metadata={"test": True}
        )


@pytest.fixture
def test_pattern():
    """Create a test pattern instance."""
    config = PatternConfig(
        pattern_id="test-pattern-1",
        enabled=True,
        timeout_seconds=5.0,
        emit_events=True,
        use_blackboard=False
    )
    return TestPattern(config)


@pytest.fixture
def test_pattern_with_blackboard():
    """Create a test pattern configured to use blackboard."""
    config = PatternConfig(
        pattern_id="test-pattern-blackboard",
        enabled=True,
        timeout_seconds=5.0,
        emit_events=True,
        use_blackboard=True
    )
    return TestPattern(config)


# LightRAG mock fixtures


@pytest.fixture
def mock_lightrag_client():
    """Mock LightRAG client for pattern testing."""
    mock = MagicMock()
    mock.query = AsyncMock(return_value="Mocked LightRAG response")
    mock.insert = AsyncMock(return_value={"status": "success"})
    mock.search = AsyncMock(return_value=["result1", "result2"])
    return mock


@pytest.fixture
def mock_lightrag_context():
    """Mock LightRAG context manager."""
    with patch('promptchain.integrations.lightrag.client.LightRAG') as mock:
        mock_instance = MagicMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_instance.query = AsyncMock(return_value="Mocked response")
        mock.return_value = mock_instance
        yield mock


# Event collection fixtures


@pytest.fixture
def event_collector():
    """Create a collector for pattern events."""
    events = []

    def collect(event_type: str, event_data: Dict[str, Any]):
        events.append({"type": event_type, "data": event_data})

    collect.events = events
    return collect


# Integration setup fixtures


@pytest.fixture
def integrated_pattern(test_pattern, message_bus, blackboard):
    """Create a pattern with both MessageBus and Blackboard connected."""
    test_pattern.connect_messagebus(message_bus)
    test_pattern.connect_blackboard(blackboard)
    test_pattern.config.use_blackboard = True
    return test_pattern


@pytest.fixture
def async_event_loop():
    """Create a new event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
