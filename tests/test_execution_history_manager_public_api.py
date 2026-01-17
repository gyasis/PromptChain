"""
Unit tests for ExecutionHistoryManager public API (v0.4.1a).

Tests the new public properties and get_statistics() method to ensure
they provide stable, reliable access to history data without requiring
private attribute access.
"""
import pytest
from promptchain.utils.execution_history_manager import ExecutionHistoryManager


class TestExecutionHistoryManagerPublicAPI:
    """Test suite for ExecutionHistoryManager public API additions."""

    def test_current_token_count_property(self):
        """Test that current_token_count property returns correct value."""
        manager = ExecutionHistoryManager(max_tokens=1000)

        # Initially should be 0
        assert manager.current_token_count == 0

        # Add an entry and verify count increases
        manager.add_entry("user_input", "Hello, this is a test message", source="user")
        assert manager.current_token_count > 0

        # Token count should be an integer
        assert isinstance(manager.current_token_count, int)

    def test_history_property_returns_copy(self):
        """Test that history property returns a copy of internal history."""
        manager = ExecutionHistoryManager()

        # Add some entries
        manager.add_entry("user_input", "First message", source="user")
        manager.add_entry("agent_output", "First response", source="agent")

        # Get history
        history = manager.history

        # Should be a list
        assert isinstance(history, list)
        assert len(history) == 2

        # Modifying returned history should not affect manager's internal state
        original_len = len(manager.history)
        history.append({"type": "test", "content": "should not be added"})
        assert len(manager.history) == original_len

    def test_history_size_property(self):
        """Test that history_size property returns correct entry count."""
        manager = ExecutionHistoryManager()

        # Initially empty
        assert manager.history_size == 0

        # Add entries
        for i in range(5):
            manager.add_entry("user_input", f"Message {i}", source="user")

        assert manager.history_size == 5

        # history_size should match len(manager.history)
        assert manager.history_size == len(manager.history)

    def test_get_statistics_basic(self):
        """Test that get_statistics returns all expected fields."""
        manager = ExecutionHistoryManager(max_tokens=1000, max_entries=10)

        # Add some varied entries
        manager.add_entry("user_input", "User message", source="user")
        manager.add_entry("agent_output", "Agent response", source="agent")
        manager.add_entry("tool_call", "Tool called", source="tool")

        stats = manager.get_statistics()

        # Verify all required fields are present
        assert "total_tokens" in stats
        assert "total_entries" in stats
        assert "max_tokens" in stats
        assert "max_entries" in stats
        assert "utilization_pct" in stats
        assert "entry_types" in stats
        assert "truncation_strategy" in stats

        # Verify data types
        assert isinstance(stats["total_tokens"], int)
        assert isinstance(stats["total_entries"], int)
        assert isinstance(stats["utilization_pct"], float)
        assert isinstance(stats["entry_types"], dict)
        assert isinstance(stats["truncation_strategy"], str)

    def test_get_statistics_utilization_percentage(self):
        """Test that utilization percentage is calculated correctly."""
        manager = ExecutionHistoryManager(max_tokens=1000)

        # Add entries
        for i in range(3):
            manager.add_entry("user_input", f"Message {i}", source="user")

        stats = manager.get_statistics()

        # Utilization should be between 0 and 100
        assert 0.0 <= stats["utilization_pct"] <= 100.0

        # If we're within limit, utilization should be less than 100%
        if stats["total_tokens"] < 1000:
            assert stats["utilization_pct"] < 100.0

    def test_get_statistics_entry_types_distribution(self):
        """Test that entry_types distribution is correct."""
        manager = ExecutionHistoryManager()

        # Add varied entry types
        manager.add_entry("user_input", "User 1", source="user")
        manager.add_entry("user_input", "User 2", source="user")
        manager.add_entry("agent_output", "Agent 1", source="agent")
        manager.add_entry("tool_call", "Tool 1", source="tool")
        manager.add_entry("tool_call", "Tool 2", source="tool")
        manager.add_entry("tool_call", "Tool 3", source="tool")

        stats = manager.get_statistics()
        entry_types = stats["entry_types"]

        # Verify counts
        assert entry_types["user_input"] == 2
        assert entry_types["agent_output"] == 1
        assert entry_types["tool_call"] == 3

        # Total should match total entries
        assert sum(entry_types.values()) == stats["total_entries"]

    def test_get_statistics_unlimited_limits(self):
        """Test statistics when no limits are set."""
        manager = ExecutionHistoryManager()  # No limits

        manager.add_entry("user_input", "Test message", source="user")

        stats = manager.get_statistics()

        # Limits should be None
        assert stats["max_tokens"] is None
        assert stats["max_entries"] is None

        # Utilization should be 0.0 when unlimited
        assert stats["utilization_pct"] == 0.0

    def test_public_api_consistency(self):
        """Test that all public API methods return consistent data."""
        manager = ExecutionHistoryManager(max_tokens=500, max_entries=10)

        # Add entries
        for i in range(5):
            manager.add_entry("user_input", f"Message {i}", source="user")

        # Get data via different methods
        token_count = manager.current_token_count
        history_size = manager.history_size
        stats = manager.get_statistics()

        # Verify consistency
        assert token_count == stats["total_tokens"]
        assert history_size == stats["total_entries"]
        assert history_size == len(manager.history)

    def test_backward_compatibility_with_len(self):
        """Test that __len__ still works and matches history_size."""
        manager = ExecutionHistoryManager()

        # Add entries
        for i in range(7):
            manager.add_entry("user_input", f"Message {i}", source="user")

        # Both should work and return same value
        assert len(manager) == manager.history_size
        assert len(manager) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
