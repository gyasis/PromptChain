"""
Integration test to verify ExecutionHistoryManager public API works
in real usage scenarios (like agentic_team_chat.py).

Tests that the new public properties can be used instead of private attributes.
"""
import pytest
from promptchain.utils.execution_history_manager import ExecutionHistoryManager


def test_public_api_usage_pattern():
    """
    Test the exact usage pattern from agentic_team_chat.py.

    This mimics how the agentic chat script uses the history manager
    to verify our migration works correctly.
    """
    # Initialize like in agentic_team_chat.py
    history_manager = ExecutionHistoryManager(
        max_tokens=8000,
        max_entries=100,
        truncation_strategy="oldest_first"
    )

    # Add some entries like a chat session would
    history_manager.add_entry("user_input", "First user message", source="user")
    history_manager.add_entry("agent_output", "Agent response 1", source="agent")
    history_manager.add_entry("user_input", "Second user message", source="user")
    history_manager.add_entry("agent_output", "Agent response 2", source="agent")

    # Test Pattern 1: Check current history size (line 1162 pattern)
    current_size = history_manager.current_token_count  # NEW PUBLIC API
    max_size = history_manager.max_tokens

    assert isinstance(current_size, int)
    assert current_size > 0
    assert max_size == 8000

    # Test Pattern 2: Check if approaching limit (line 1165 pattern)
    if current_size > max_size * 0.9:
        # Would trigger truncation warning
        pass  # This shouldn't happen in this test

    # Test Pattern 3: Get total entries and tokens (lines 1192-1193 pattern)
    total_entries = history_manager.history_size  # NEW PUBLIC API
    total_tokens = history_manager.current_token_count  # NEW PUBLIC API

    assert total_entries == 4  # We added 4 entries
    assert total_tokens == current_size

    # Test Pattern 4: Iterate through history (line 1206 pattern)
    entry_types = {}
    for entry in history_manager.history:  # NEW PUBLIC API
        entry_type = entry.get('type', 'unknown')
        entry_types[entry_type] = entry_types.get(entry_type, 0) + 1

    assert entry_types["user_input"] == 2
    assert entry_types["agent_output"] == 2

    # Test Pattern 5: Display stats (lines 1272, 1310 patterns)
    stats_display = {
        "Total Entries": history_manager.history_size,  # NEW PUBLIC API
        "Current Tokens": history_manager.current_token_count,  # NEW PUBLIC API
        "Max Tokens": history_manager.max_tokens,
        "Usage": f"{(history_manager.current_token_count/history_manager.max_tokens*100):.1f}%"
    }

    assert stats_display["Total Entries"] == 4
    assert "%" in stats_display["Usage"]

    # Test Pattern 6: Log event metadata (lines 1310, 1333, 1363 patterns)
    log_data = {
        "history_entries": history_manager.history_size,  # NEW PUBLIC API
        "history_size_after": history_manager.current_token_count,  # NEW PUBLIC API
        "final_history_size": history_manager.current_token_count,  # NEW PUBLIC API
    }

    assert all(isinstance(v, int) for v in log_data.values())


def test_get_statistics_integration():
    """Test that get_statistics() provides all needed data for monitoring."""
    history_manager = ExecutionHistoryManager(max_tokens=5000, max_entries=50)

    # Add varied entries
    for i in range(10):
        history_manager.add_entry("user_input", f"User message {i}", source="user")
        history_manager.add_entry("agent_output", f"Agent response {i}", source="agent")
        if i % 3 == 0:
            history_manager.add_entry("tool_call", f"Tool call {i}", source="tool")

    # Get comprehensive stats
    stats = history_manager.get_statistics()

    # Verify we can use this for monitoring dashboards
    assert stats["total_entries"] > 0
    assert stats["total_tokens"] > 0
    assert stats["max_tokens"] == 5000
    assert stats["max_entries"] == 50
    assert 0.0 <= stats["utilization_pct"] <= 100.0

    # Verify entry type breakdown
    assert "user_input" in stats["entry_types"]
    assert "agent_output" in stats["entry_types"]
    assert "tool_call" in stats["entry_types"]

    # Should be able to display this in a dashboard
    dashboard_summary = f"Using {stats['utilization_pct']:.1f}% of token limit"
    assert "%" in dashboard_summary


def test_history_truncation_scenario():
    """Test automatic truncation like in agentic_team_chat.py manage_history_automatically()."""
    # Small limit to trigger truncation
    history_manager = ExecutionHistoryManager(max_tokens=500, max_entries=10)

    # Add entries until we approach the limit
    for i in range(20):
        history_manager.add_entry("user_input", f"Message {i} with some content to use tokens", source="user")

    # Check size like agentic_team_chat.py does (line 1162)
    current_size = history_manager.current_token_count
    max_size = history_manager.max_tokens

    # Should have been truncated
    assert current_size <= max_size

    # Should have fewer entries than we added
    assert history_manager.history_size < 20

    # New size should be reported correctly (line 1173 pattern)
    new_size = history_manager.current_token_count
    assert new_size == current_size  # Should match since we just checked


def test_no_private_attribute_access_needed():
    """
    Verify that all common operations can be done WITHOUT private attributes.

    This ensures our public API is complete and sufficient.
    """
    history_manager = ExecutionHistoryManager(max_tokens=1000)

    # Add some data
    history_manager.add_entry("user_input", "Test", source="user")

    # All these should work using ONLY public API
    # (No _current_token_count, no _history, no other private attrs)

    token_count = history_manager.current_token_count  # ✅ Public
    entry_count = history_manager.history_size  # ✅ Public
    history_copy = history_manager.history  # ✅ Public
    stats = history_manager.get_statistics()  # ✅ Public

    # Verify we got real data
    assert token_count > 0
    assert entry_count == 1
    assert len(history_copy) == 1
    assert stats["total_entries"] == 1

    # This proves the public API is complete!


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
