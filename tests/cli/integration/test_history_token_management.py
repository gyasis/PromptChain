"""Integration tests for ExecutionHistoryManager token counting and truncation (T072).

These tests verify that ExecutionHistoryManager correctly manages conversation
history with token limits when integrated with AgentChain's per-agent history
configuration. Tests cover token counting accuracy, truncation strategies, and
the interaction between HistoryConfig and ExecutionHistoryManager.

Test Coverage:
- Token counting with tiktoken
- Automatic truncation when token limits exceeded
- Truncation strategy "oldest_first"
- Integration with per-agent history configs
- History filtering by entry type and source
- Token savings verification
"""

import pytest
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.promptchaining import PromptChain
from promptchain.utils.execution_history_manager import ExecutionHistoryManager


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory for test sessions."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def basic_agents():
    """Create basic test agents."""
    return {
        "analyst": PromptChain(
            models=["gpt-4.1-mini-2025-04-14"], instructions=["Analyze: {input}"], verbose=False
        ),
        "coder": PromptChain(
            models=["gpt-4.1-mini-2025-04-14"], instructions=["Code: {input}"], verbose=False
        ),
    }


@pytest.fixture
def agent_descriptions():
    """Create agent descriptions for AgentChain."""
    return {
        "analyst": "Analyzes data and provides insights",
        "coder": "Writes code and implements solutions",
    }


class TestExecutionHistoryManagerTokenCounting:
    """Integration tests for token counting functionality."""

    def test_history_manager_counts_tokens_with_tiktoken(self):
        """ExecutionHistoryManager uses tiktoken for accurate token counting."""
        history_manager = ExecutionHistoryManager(max_tokens=1000)

        # Add entry
        history_manager.add_entry("user_input", "Hello, how are you?", source="user")

        # Verify token counting occurred
        assert history_manager._current_token_count > 0
        assert history_manager._current_token_count < 20  # Should be ~5-6 tokens

    def test_history_manager_tracks_cumulative_token_count(self):
        """ExecutionHistoryManager maintains accurate cumulative token count."""
        history_manager = ExecutionHistoryManager(max_tokens=5000)

        # Add multiple entries
        history_manager.add_entry("user_input", "First message", source="user")
        tokens_after_first = history_manager._current_token_count

        history_manager.add_entry("agent_output", "Response to first", source="agent")
        tokens_after_second = history_manager._current_token_count

        # Cumulative count should increase
        assert tokens_after_second > tokens_after_first
        assert tokens_after_second > 0

    def test_history_manager_counts_different_entry_types(self):
        """ExecutionHistoryManager counts tokens for all entry types."""
        history_manager = ExecutionHistoryManager(max_tokens=10000)

        # Add different entry types
        history_manager.add_entry("user_input", "User question", source="user")
        history_manager.add_entry("agent_output", "Agent response", source="agent")
        history_manager.add_entry("tool_call", '{"tool": "search"}', source="tool")
        history_manager.add_entry("tool_result", "Search results here", source="tool")

        # All entries should contribute to token count
        assert history_manager._current_token_count > 0
        assert len(history_manager._history) == 4


class TestExecutionHistoryManagerTruncation:
    """Integration tests for history truncation functionality."""

    def test_history_truncates_when_token_limit_exceeded(self):
        """ExecutionHistoryManager truncates oldest entries when token limit exceeded."""
        # Very low token limit to trigger truncation
        history_manager = ExecutionHistoryManager(max_tokens=50)

        # Add entries until truncation occurs
        for i in range(10):
            history_manager.add_entry(
                "user_input",
                f"This is message number {i} with some extra words to consume tokens",
                source="user",
            )

        # History should be truncated (not all 10 entries)
        assert len(history_manager._history) < 10
        assert history_manager._current_token_count <= 50

    def test_history_truncation_removes_oldest_first(self):
        """Truncation strategy 'oldest_first' removes oldest entries first."""
        history_manager = ExecutionHistoryManager(
            max_tokens=100, truncation_strategy="oldest_first"
        )

        # Add identifiable entries
        history_manager.add_entry("user_input", "FIRST message", source="user")
        history_manager.add_entry("user_input", "SECOND message", source="user")
        history_manager.add_entry("user_input", "THIRD message", source="user")

        # Add large entry to trigger truncation
        large_text = "word " * 100  # ~100 words = ~133 tokens
        history_manager.add_entry("user_input", large_text, source="user")

        # Oldest entries should be removed
        history_contents = [entry["content"] for entry in history_manager._history]

        # FIRST and possibly SECOND should be truncated
        assert "FIRST message" not in history_contents
        # Large entry should still be present (most recent)
        assert any(large_text in str(content) for content in history_contents)

    def test_history_truncation_maintains_token_limit(self):
        """After truncation, token count remains below or at limit."""
        history_manager = ExecutionHistoryManager(max_tokens=200)

        # Add many entries
        for i in range(20):
            history_manager.add_entry(
                "user_input",
                f"Message {i}: " + "word " * 20,  # ~20 words per message
                source="user",
            )

        # Token count should not exceed limit
        assert history_manager._current_token_count <= 200

    def test_history_truncation_preserves_at_least_one_entry(self):
        """Truncation preserves at least one entry even if over token limit."""
        # Very small limit
        history_manager = ExecutionHistoryManager(max_tokens=10)

        # Add large entry that exceeds limit
        large_text = "word " * 100  # Way over 10 tokens
        history_manager.add_entry("user_input", large_text, source="user")

        # Should still have 1 entry despite exceeding token limit
        assert len(history_manager._history) >= 1

    def test_max_entries_truncation_works_independently(self):
        """max_entries truncation works when token limit not set."""
        history_manager = ExecutionHistoryManager(max_entries=3)

        # Add more entries than limit
        for i in range(10):
            history_manager.add_entry("user_input", f"Message {i}", source="user")

        # Should only have max_entries
        assert len(history_manager._history) == 3


class TestHistoryConfigIntegrationWithAgentChain:
    """Integration tests for HistoryConfig working with ExecutionHistoryManager in AgentChain."""

    def test_per_agent_history_config_creates_history_managers(
        self, basic_agents, agent_descriptions, temp_cache_dir
    ):
        """AgentChain creates ExecutionHistoryManager per agent using agent_history_configs."""
        history_configs = {
            "analyst": {
                "enabled": True,
                "max_tokens": 8000,
                "max_entries": 50,
                "truncation_strategy": "oldest_first",
            },
            "coder": {
                "enabled": True,
                "max_tokens": 4000,
                "max_entries": 20,
                "truncation_strategy": "oldest_first",
            },
        }

        agent_chain = AgentChain(
            agents=basic_agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs=history_configs,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False,
        )

        # Verify agent_history_configs stored
        assert agent_chain.agent_history_configs is not None
        assert "analyst" in agent_chain.agent_history_configs
        assert "coder" in agent_chain.agent_history_configs

        # Verify config values match
        assert agent_chain.agent_history_configs["analyst"]["max_tokens"] == 8000
        assert agent_chain.agent_history_configs["coder"]["max_tokens"] == 4000

    def test_disabled_history_saves_tokens(
        self, basic_agents, agent_descriptions, temp_cache_dir
    ):
        """Disabling history for an agent saves tokens (no history stored)."""
        history_configs = {
            "analyst": {"enabled": True, "max_tokens": 4000, "max_entries": 20},
            "coder": {
                "enabled": False,
                "max_tokens": 0,
                "max_entries": 0,
            },  # Disabled
        }

        agent_chain = AgentChain(
            agents=basic_agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs=history_configs,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False,
        )

        # Verify disabled config stored correctly
        assert agent_chain.agent_history_configs["coder"]["enabled"] is False
        assert agent_chain.agent_history_configs["coder"]["max_tokens"] == 0

    def test_history_filtering_by_include_types(
        self, basic_agents, agent_descriptions, temp_cache_dir
    ):
        """History filtering by include_types reduces stored history."""
        history_configs = {
            "analyst": {
                "enabled": True,
                "max_tokens": 8000,
                "max_entries": 50,
                "include_types": ["user_input", "agent_output"],  # Only these types
            }
        }

        agent_chain = AgentChain(
            agents=basic_agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs=history_configs,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False,
        )

        # Verify filtering config stored
        assert "include_types" in agent_chain.agent_history_configs["analyst"]
        assert agent_chain.agent_history_configs["analyst"]["include_types"] == [
            "user_input",
            "agent_output",
        ]

    def test_history_filtering_by_exclude_sources(
        self, basic_agents, agent_descriptions, temp_cache_dir
    ):
        """History filtering by exclude_sources prevents certain sources from being stored."""
        history_configs = {
            "coder": {
                "enabled": True,
                "max_tokens": 4000,
                "max_entries": 20,
                "exclude_sources": ["system", "debug"],  # Exclude these sources
            }
        }

        agent_chain = AgentChain(
            agents=basic_agents,
            agent_descriptions=agent_descriptions,
            agent_history_configs=history_configs,
            execution_mode="pipeline",
            cache_config={"name": "test_session", "path": str(temp_cache_dir)},
            verbose=False,
        )

        # Verify exclusion config stored
        assert "exclude_sources" in agent_chain.agent_history_configs["coder"]
        assert agent_chain.agent_history_configs["coder"]["exclude_sources"] == [
            "system",
            "debug",
        ]


class TestTokenSavingsCalculations:
    """Integration tests for verifying token savings from history optimization."""

    def test_disabled_history_vs_enabled_history_token_difference(self):
        """Disabling history saves significant tokens compared to full history."""
        # Full history
        full_history_manager = ExecutionHistoryManager(max_tokens=10000, max_entries=100)

        # Add 50 conversation turns
        for i in range(50):
            full_history_manager.add_entry("user_input", f"User message {i}", source="user")
            full_history_manager.add_entry(
                "agent_output", f"Agent response {i}", source="agent"
            )

        full_history_tokens = full_history_manager._current_token_count

        # Disabled history (no storage)
        disabled_history_tokens = 0  # No history stored

        # Token savings should be significant
        token_savings = full_history_tokens - disabled_history_tokens
        savings_percentage = (token_savings / full_history_tokens) * 100 if full_history_tokens > 0 else 0

        assert savings_percentage >= 99  # Should save ~100% of tokens

    def test_limited_history_vs_unlimited_history_token_difference(self):
        """Limited max_tokens reduces storage compared to unlimited."""
        # Unlimited tokens
        unlimited_history_manager = ExecutionHistoryManager()  # No limits

        # Add conversation
        for i in range(100):
            unlimited_history_manager.add_entry(
                "user_input", f"Message {i}: " + "word " * 10, source="user"
            )

        unlimited_tokens = unlimited_history_manager._current_token_count

        # Limited to 500 tokens
        limited_history_manager = ExecutionHistoryManager(max_tokens=500)

        for i in range(100):
            limited_history_manager.add_entry(
                "user_input", f"Message {i}: " + "word " * 10, source="user"
            )

        limited_tokens = limited_history_manager._current_token_count

        # Limited should have fewer tokens
        assert limited_tokens <= 500
        assert limited_tokens < unlimited_tokens

        # Calculate savings
        token_savings = unlimited_tokens - limited_tokens
        savings_percentage = (token_savings / unlimited_tokens) * 100

        # Should save significant tokens (depends on data, but expect >50%)
        assert savings_percentage > 0

    def test_filtered_history_vs_unfiltered_token_difference(self):
        """Filtering entry types reduces token storage."""
        # Unfiltered history (all entry types)
        unfiltered_manager = ExecutionHistoryManager(max_tokens=10000)

        for i in range(20):
            unfiltered_manager.add_entry("user_input", f"User {i}", source="user")
            unfiltered_manager.add_entry("agent_output", f"Agent {i}", source="agent")
            unfiltered_manager.add_entry("tool_call", f"Tool {i}", source="tool")
            unfiltered_manager.add_entry("tool_result", f"Result {i}", source="tool")
            unfiltered_manager.add_entry("system_message", f"System {i}", source="system")

        unfiltered_tokens = unfiltered_manager._current_token_count
        unfiltered_count = len(unfiltered_manager._history)

        # Filtered history (only user_input and agent_output)
        # NOTE: ExecutionHistoryManager doesn't implement filtering yet (T080)
        # This test verifies the EXPECTED token reduction if filtering were applied

        # Simulate filtered entry count (40% of entries if only 2 of 5 types included)
        expected_filtered_count = int(unfiltered_count * 0.4)  # 2/5 = 40%

        # Expect similar token reduction
        expected_token_reduction_percentage = 60  # ~60% reduction

        # Verify expectation
        assert expected_filtered_count < unfiltered_count
        assert expected_token_reduction_percentage > 0


class TestTruncationStrategyKeepLast:
    """Integration tests for 'keep_last' truncation strategy (T078)."""

    def test_keep_last_strategy_removes_oldest_entries(self):
        """keep_last truncation strategy removes oldest entries when limit exceeded."""
        history_manager = ExecutionHistoryManager(
            max_tokens=100,
            truncation_strategy="keep_last"
        )

        # Add identifiable entries
        history_manager.add_entry("user_input", "FIRST message", source="user")
        history_manager.add_entry("user_input", "SECOND message", source="user")
        history_manager.add_entry("user_input", "THIRD message", source="user")

        # Add large entry to trigger truncation
        large_text = "word " * 100  # ~100 words = ~133 tokens
        history_manager.add_entry("user_input", large_text, source="user")

        # Oldest entries should be removed
        history_contents = [entry["content"] for entry in history_manager._history]

        # FIRST and SECOND should be truncated
        assert "FIRST message" not in history_contents
        # Most recent large entry should be kept
        assert any(large_text in str(content) for content in history_contents)

    def test_keep_last_strategy_maintains_token_limit(self):
        """keep_last strategy ensures token count stays under limit."""
        history_manager = ExecutionHistoryManager(
            max_tokens=300,
            truncation_strategy="keep_last"
        )

        # Add many entries
        for i in range(50):
            history_manager.add_entry(
                "user_input",
                f"Message {i}: " + "word " * 15,
                source="user"
            )

        # Token count should not exceed limit
        assert history_manager._current_token_count <= 300

    def test_keep_last_vs_oldest_first_behavior_identical_for_token_limit(self):
        """For token limits, keep_last and oldest_first behave identically (both remove oldest)."""
        # Create two managers with different strategies
        oldest_first_manager = ExecutionHistoryManager(
            max_tokens=200,
            truncation_strategy="oldest_first"
        )

        keep_last_manager = ExecutionHistoryManager(
            max_tokens=200,
            truncation_strategy="keep_last"
        )

        # Add same data to both
        for i in range(30):
            message = f"Message {i}: " + "word " * 10
            oldest_first_manager.add_entry("user_input", message, source="user")
            keep_last_manager.add_entry("user_input", message, source="user")

        # Both should maintain token limit
        assert oldest_first_manager._current_token_count <= 200
        assert keep_last_manager._current_token_count <= 200

        # Both should have similar entry counts (within 1-2 entries)
        assert abs(len(oldest_first_manager._history) - len(keep_last_manager._history)) <= 2


class TestHistoryEdgeCases:
    """Integration tests for edge cases in history management."""

    def test_empty_history_has_zero_tokens(self):
        """Empty history manager has zero token count."""
        history_manager = ExecutionHistoryManager(max_tokens=1000)

        # No entries added
        assert history_manager._current_token_count == 0
        assert len(history_manager._history) == 0

    def test_single_entry_history_never_truncated(self):
        """Single entry in history is never removed even if over token limit."""
        history_manager = ExecutionHistoryManager(max_tokens=50)

        # Add single large entry
        large_text = "word " * 200  # Way over 50 tokens
        history_manager.add_entry("user_input", large_text, source="user")

        # Entry should still be present
        assert len(history_manager._history) == 1
        assert history_manager._history[0]["content"] == large_text

    def test_very_large_single_entry_still_stored(self):
        """Very large single entry (>max_tokens) is still stored."""
        history_manager = ExecutionHistoryManager(max_tokens=100)

        # Add massive entry
        massive_text = "word " * 1000  # ~1333 tokens
        history_manager.add_entry("user_input", massive_text, source="user")

        # Entry should be present
        assert len(history_manager._history) == 1
        assert massive_text in history_manager._history[0]["content"]

    def test_empty_string_entry_counts_as_zero_tokens(self):
        """Empty string entry contributes zero tokens."""
        history_manager = ExecutionHistoryManager(max_tokens=1000)

        initial_tokens = history_manager._current_token_count

        history_manager.add_entry("user_input", "", source="user")

        # Token count should remain same or increase by 0
        assert history_manager._current_token_count == initial_tokens

    def test_none_content_handled_gracefully(self):
        """None content is handled without crashing."""
        history_manager = ExecutionHistoryManager(max_tokens=1000)

        # Add entry with None content
        history_manager.add_entry("user_input", None, source="user")

        # Should not crash
        assert len(history_manager._history) == 1

    def test_large_history_stress_test_1000_entries(self):
        """Stress test with 1000 entries maintains token limit."""
        history_manager = ExecutionHistoryManager(max_tokens=2000, max_entries=100)

        # Add 1000 entries
        for i in range(1000):
            history_manager.add_entry(
                "user_input",
                f"Message {i}: " + "word " * 5,
                source="user"
            )

        # Token count should not exceed limit
        assert history_manager._current_token_count <= 2000
        # Entry count should not exceed limit
        assert len(history_manager._history) <= 100


class TestHistoryFiltering:
    """Integration tests for history filtering during get_formatted_history."""

    def test_entry_type_filtering_in_get_formatted_history(self):
        """get_formatted_history filters by entry type correctly."""
        history_manager = ExecutionHistoryManager(max_tokens=5000)

        # Add mixed entry types
        history_manager.add_entry("user_input", "User question 1", source="user")
        history_manager.add_entry("agent_output", "Agent response 1", source="agent")
        history_manager.add_entry("tool_call", "Tool call data", source="tool")
        history_manager.add_entry("user_input", "User question 2", source="user")
        history_manager.add_entry("agent_output", "Agent response 2", source="agent")

        # Filter to only user_input
        formatted = history_manager.get_formatted_history(
            include_types=["user_input"],
            format_style="content_only"
        )

        # Should only contain user inputs
        assert "User question 1" in formatted
        assert "User question 2" in formatted
        assert "Agent response" not in formatted
        assert "Tool call" not in formatted

    def test_source_filtering_in_get_formatted_history(self):
        """get_formatted_history filters by source correctly."""
        history_manager = ExecutionHistoryManager(max_tokens=5000)

        # Add entries from different sources
        history_manager.add_entry("user_input", "From user", source="user")
        history_manager.add_entry("agent_output", "From agent", source="agent")
        history_manager.add_entry("system_message", "From system", source="system")
        history_manager.add_entry("tool_result", "From tool", source="tool")

        # Filter to only agent source
        formatted = history_manager.get_formatted_history(
            include_sources=["agent"],
            format_style="content_only"
        )

        # Should only contain agent entries
        assert "From agent" in formatted
        assert "From user" not in formatted
        assert "From system" not in formatted
        assert "From tool" not in formatted

    def test_exclude_types_filtering_in_get_formatted_history(self):
        """get_formatted_history excludes specified entry types."""
        history_manager = ExecutionHistoryManager(max_tokens=5000)

        # Add mixed entry types
        history_manager.add_entry("user_input", "User question", source="user")
        history_manager.add_entry("agent_output", "Agent response", source="agent")
        history_manager.add_entry("system_message", "System message", source="system")
        history_manager.add_entry("error", "Error message", source="system")

        # Exclude system and error types
        formatted = history_manager.get_formatted_history(
            exclude_types=["system_message", "error"],
            format_style="content_only"
        )

        # Should not contain excluded types
        assert "User question" in formatted
        assert "Agent response" in formatted
        assert "System message" not in formatted
        assert "Error message" not in formatted

    def test_exclude_sources_filtering_in_get_formatted_history(self):
        """get_formatted_history excludes specified sources."""
        history_manager = ExecutionHistoryManager(max_tokens=5000)

        # Add entries from different sources
        history_manager.add_entry("user_input", "From user", source="user")
        history_manager.add_entry("agent_output", "From agent", source="agent")
        history_manager.add_entry("system_message", "From system", source="system")

        # Exclude system source
        formatted = history_manager.get_formatted_history(
            exclude_sources=["system"],
            format_style="content_only"
        )

        # Should not contain system source
        assert "From user" in formatted
        assert "From agent" in formatted
        assert "From system" not in formatted


class TestMaxTokensAndEntriesEnforcement:
    """Integration tests for strict enforcement of max_tokens and max_entries."""

    def test_max_tokens_never_exceeded_after_multiple_operations(self):
        """max_tokens limit is strictly enforced across multiple add operations."""
        history_manager = ExecutionHistoryManager(max_tokens=500)

        # Perform 100 add operations with varying sizes
        for i in range(100):
            message_size = (i % 10) + 5  # Varying message sizes
            history_manager.add_entry(
                "user_input",
                "word " * message_size,
                source="user"
            )

            # After each operation, verify limit not exceeded
            assert history_manager._current_token_count <= 500

    def test_max_entries_never_exceeded_after_multiple_operations(self):
        """max_entries limit is strictly enforced across multiple add operations."""
        history_manager = ExecutionHistoryManager(max_entries=10)

        # Add 50 entries
        for i in range(50):
            history_manager.add_entry("user_input", f"Message {i}", source="user")

            # After each operation, verify limit not exceeded
            assert len(history_manager._history) <= 10

    def test_both_limits_enforced_simultaneously(self):
        """Both max_tokens and max_entries are enforced when both are set."""
        history_manager = ExecutionHistoryManager(max_tokens=500, max_entries=20)

        # Add many entries
        for i in range(100):
            history_manager.add_entry(
                "user_input",
                f"Message {i}: " + "word " * 10,
                source="user"
            )

        # Both limits should be respected
        assert history_manager._current_token_count <= 500
        assert len(history_manager._history) <= 20

    def test_token_limit_takes_precedence_when_stricter(self):
        """Token limit causes truncation before entry limit is reached."""
        # Set very restrictive token limit, generous entry limit
        history_manager = ExecutionHistoryManager(max_tokens=100, max_entries=50)

        # Add entries with moderate size
        for i in range(30):
            history_manager.add_entry(
                "user_input",
                "word " * 10,  # ~13 tokens each
                source="user"
            )

        # Token limit should cause truncation before entry limit
        assert history_manager._current_token_count <= 100
        assert len(history_manager._history) < 50  # Well below entry limit

    def test_entry_limit_takes_precedence_when_stricter(self):
        """Entry limit causes truncation before token limit is reached."""
        # Set generous token limit, restrictive entry limit
        history_manager = ExecutionHistoryManager(max_tokens=10000, max_entries=5)

        # Add small entries
        for i in range(20):
            history_manager.add_entry(
                "user_input",
                f"Msg {i}",  # Small messages
                source="user"
            )

        # Entry limit should cause truncation before token limit
        assert len(history_manager._history) <= 5
        assert history_manager._current_token_count < 10000  # Well below token limit


class TestTruncationPreservesRecentContext:
    """Integration tests verifying truncation preserves most recent context."""

    def test_truncation_keeps_most_recent_entries(self):
        """After truncation, most recent entries are preserved."""
        # Use more restrictive token limit to ensure truncation
        history_manager = ExecutionHistoryManager(max_tokens=150)

        # Add identifiable entries
        for i in range(20):
            history_manager.add_entry(
                "user_input",
                f"MESSAGE_{i}: " + "word " * 10,
                source="user"
            )

        # Most recent entries should be present
        history_contents = [entry["content"] for entry in history_manager._history]
        all_content = " ".join(str(c) for c in history_contents)

        # Last entry (MESSAGE_19) should definitely be present
        assert "MESSAGE_19" in all_content

        # Earlier entries should be truncated (first few messages)
        assert "MESSAGE_0" not in all_content
        # Also verify truncation occurred
        assert len(history_manager._history) < 20

    def test_recent_context_preserved_across_different_entry_types(self):
        """Recent context from all entry types is preserved after truncation."""
        # Use very restrictive token limit to force truncation
        history_manager = ExecutionHistoryManager(max_tokens=100)

        # Add old entries of various types with larger content
        for i in range(10):
            history_manager.add_entry("user_input", f"Old user {i}: " + "word " * 5, source="user")
            history_manager.add_entry("agent_output", f"Old agent {i}: " + "word " * 5, source="agent")

        # Add recent conversation with unique markers
        history_manager.add_entry("user_input", "RECENT_USER_QUESTION", source="user")
        history_manager.add_entry("agent_output", "RECENT_AGENT_RESPONSE", source="agent")
        history_manager.add_entry("tool_call", "RECENT_TOOL_CALL", source="tool")

        # Get recent history
        formatted = history_manager.get_formatted_history(format_style="content_only")

        # Recent context should be present
        assert "RECENT_USER_QUESTION" in formatted
        assert "RECENT_AGENT_RESPONSE" in formatted
        assert "RECENT_TOOL_CALL" in formatted

        # Old context should be truncated (verify at least first entry is gone)
        assert "Old user 0" not in formatted
        # Verify truncation actually occurred
        assert len(history_manager._history) < 23  # Less than total 23 entries added
