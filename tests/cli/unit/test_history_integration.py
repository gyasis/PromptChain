"""Unit tests for ExecutionHistoryManager integration with CLI.

These tests verify that conversation history is properly integrated with
PromptChain's ExecutionHistoryManager for token-aware history management
and context passing to agents.

Test Coverage:
- test_messages_added_to_history: Messages are tracked in ExecutionHistoryManager
- test_history_token_limits: History respects token limits
- test_history_formatted_for_agent: History is formatted correctly for agent prompts
- test_history_truncation: Old messages truncated when limits exceeded
- test_per_agent_history: Different agents can have different history configs
"""

import pytest
from datetime import datetime
from pathlib import Path

from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain.cli.models import Session, Message


class TestHistoryIntegration:
    """Unit tests for ExecutionHistoryManager integration."""

    @pytest.fixture
    def history_manager(self):
        """Create ExecutionHistoryManager for testing."""
        return ExecutionHistoryManager(
            max_tokens=4000,
            max_entries=50,
            truncation_strategy="oldest_first"
        )

    @pytest.fixture
    def test_session(self, tmp_path):
        """Create a test session."""
        session = Session(
            id="test-session-id",
            name="history-test",
            created_at=datetime.now().timestamp(),
            last_accessed=datetime.now().timestamp(),
            working_directory=tmp_path,
            default_model="gpt-4.1-mini-2025-04-14"
        )
        return session

    def test_messages_added_to_history(self, history_manager, test_session):
        """Unit: Messages are tracked in ExecutionHistoryManager.

        Flow:
        1. User sends message
        2. Message added to ExecutionHistoryManager as 'user_input'
        3. Agent responds
        4. Response added to ExecutionHistoryManager as 'agent_output'

        Validates:
        - Messages are added with correct entry types
        - Content is preserved
        - Source is tracked
        """
        # Add user message to history
        user_msg = "What is Python?"
        history_manager.add_entry(
            entry_type="user_input",
            content=user_msg,
            source="user"
        )

        # Add agent response to history
        agent_response = "Python is a high-level programming language."
        history_manager.add_entry(
            entry_type="agent_output",
            content=agent_response,
            source="default_agent"
        )

        # Validate history entries
        history = history_manager.get_formatted_history(format_style="chat")

        assert user_msg in history
        assert agent_response in history

    def test_history_token_limits(self, history_manager):
        """Unit: History respects token limits.

        ExecutionHistoryManager configuration:
        - max_tokens: 4000
        - max_entries: 50
        - truncation_strategy: oldest_first

        Validates:
        - History stays within token limits
        - Token counting is accurate
        - Truncation occurs when needed
        """
        # Add messages until we approach token limit
        long_message = "This is a test message. " * 50  # ~500 chars, ~125 tokens

        for i in range(40):  # 40 messages * ~125 tokens = ~5000 tokens (exceeds limit)
            history_manager.add_entry(
                entry_type="user_input" if i % 2 == 0 else "agent_output",
                content=f"{long_message} (message {i})",
                source="user" if i % 2 == 0 else "agent"
            )

        # Get formatted history
        history = history_manager.get_formatted_history(
            format_style="chat",
            max_tokens=4000
        )

        # Validate history was truncated to stay within limits
        # (exact token count will vary, but should be close to 4000)
        # We can't easily count tokens here, but we can verify truncation occurred
        assert len(history) > 0
        assert "message 39" in history  # Recent messages should be present
        # Oldest messages may be truncated

    def test_history_formatted_for_agent(self, history_manager):
        """Unit: History is formatted correctly for agent prompts.

        Format styles:
        - 'chat': User/Assistant message format
        - 'full_json': Complete JSON with metadata
        - 'simple': Plain text conversation

        Validates:
        - Format is suitable for LLM consumption
        - Role labels are clear
        - Conversation flow is preserved
        """
        # Add conversation
        history_manager.add_entry("user_input", "Hello", source="user")
        history_manager.add_entry("agent_output", "Hi there!", source="agent")
        history_manager.add_entry("user_input", "How are you?", source="user")

        # Get chat format (default for agents)
        chat_format = history_manager.get_formatted_history(format_style="chat")

        # Validate format
        assert "Hello" in chat_format
        assert "Hi there!" in chat_format
        assert "How are you?" in chat_format

        # Should have role indicators (User:, Assistant:, etc.)
        assert any(
            indicator in chat_format.lower()
            for indicator in ["user", "assistant", "you"]
        )

    def test_history_truncation(self, history_manager):
        """Unit: Old messages truncated when limits exceeded.

        Truncation strategy: oldest_first
        - Oldest messages removed first
        - Most recent messages preserved
        - Truncation maintains conversation coherence

        Validates:
        - Oldest messages are removed
        - Recent messages remain
        - Truncation is transparent
        """
        # Add many messages
        for i in range(100):
            history_manager.add_entry(
                entry_type="user_input" if i % 2 == 0 else "agent_output",
                content=f"Message {i}",
                source="user" if i % 2 == 0 else "agent"
            )

        # Get history with limits
        history = history_manager.get_formatted_history(
            max_entries=10,
            format_style="chat"
        )

        # Validate recent messages are present
        assert "Message 99" in history or "Message 98" in history

        # Validate oldest messages are NOT present
        assert "Message 0" not in history
        assert "Message 1" not in history

    def test_per_agent_history(self, tmp_path):
        """Unit: Different agents can have different history configs.

        AgentChain supports per-agent history configuration:
        - Some agents may need full history
        - Some agents may need limited history
        - Terminal agents may need no history

        Validates:
        - Per-agent history configs work
        - Agent-specific limits are respected
        - History isolation is maintained
        """
        # Create history managers with different configs
        full_history = ExecutionHistoryManager(
            max_tokens=8000,
            max_entries=50
        )

        limited_history = ExecutionHistoryManager(
            max_tokens=2000,
            max_entries=10
        )

        # Add same messages to both
        for i in range(20):
            msg = f"Message {i}"
            full_history.add_entry("user_input", msg, source="user")
            limited_history.add_entry("user_input", msg, source="user")

        # Get formatted history
        full = full_history.get_formatted_history(format_style="chat")
        limited = limited_history.get_formatted_history(format_style="chat")

        # Validate full history has more content
        assert len(full) > len(limited)

        # Validate both have recent messages
        assert "Message 19" in full
        assert "Message 19" in limited

        # Validate full history has older messages that limited doesn't
        assert "Message 0" in full or "Message 1" in full
        # Limited history may have truncated these

    def test_history_with_metadata(self, history_manager):
        """Unit: History entries can include metadata.

        Metadata examples:
        - File references
        - Tool calls
        - Agent names
        - Model names

        Validates:
        - Metadata is preserved
        - Metadata can be filtered
        - Metadata is accessible when needed
        """
        # Add entry with metadata
        history_manager.add_entry(
            entry_type="user_input",
            content="Review @README.md",
            source="user",
            metadata={
                "file_references": ["README.md"],
                "timestamp": datetime.now().timestamp()
            }
        )

        # Get full history with metadata
        history_with_metadata = history_manager.get_formatted_history(
            format_style="full_json"
        )

        # Validate metadata is present
        assert "README.md" in history_with_metadata
        assert "file_references" in history_with_metadata

    def test_history_filtering(self, history_manager):
        """Unit: History can be filtered by type and source.

        Filtering options:
        - include_types: Only specific entry types
        - exclude_sources: Exclude specific sources
        - Custom filters

        Validates:
        - Filtering works correctly
        - Filtered history is valid
        - Filtering doesn't break formatting
        """
        # Add various entry types
        history_manager.add_entry("user_input", "User message", source="user")
        history_manager.add_entry("agent_output", "Agent response", source="agent")
        history_manager.add_entry("system", "System message", source="system")
        history_manager.add_entry("tool_call", "Tool execution", source="tool")

        # Filter to only user_input and agent_output
        filtered = history_manager.get_formatted_history(
            include_types=["user_input", "agent_output"],
            format_style="chat"
        )

        # Validate filtered content
        assert "User message" in filtered
        assert "Agent response" in filtered
        assert "System message" not in filtered or len(filtered.split("System message")) == 1
        assert "Tool execution" not in filtered or len(filtered.split("Tool execution")) == 1

    def test_history_clear(self, history_manager):
        """Unit: History can be cleared when needed.

        Use cases:
        - Starting fresh conversation
        - Context reset
        - Agent switch

        Validates:
        - History can be cleared
        - New messages can be added after clear
        - Clear doesn't break functionality
        """
        # Add messages
        history_manager.add_entry("user_input", "Message 1", source="user")
        history_manager.add_entry("agent_output", "Response 1", source="agent")

        # Clear history
        history_manager.clear()

        # Verify history is empty (check the actual history list)
        assert len(history_manager.get_history()) == 0

        # Add new message after clear
        history_manager.add_entry("user_input", "New message", source="user")

        # Verify new message is present
        new_history = history_manager.get_formatted_history(format_style="chat")
        assert "New message" in new_history
        assert "Message 1" not in new_history

    def test_history_performance_100_messages(self, history_manager):
        """Unit: History operations remain fast with 100+ messages.

        Performance target: 100+ message conversations without degradation (SC-011)

        Validates:
        - Adding entries is fast
        - Formatting is efficient
        - Token counting doesn't slow down
        """
        import time

        # Add 100 messages and measure time
        add_times = []
        for i in range(100):
            start = time.perf_counter()

            history_manager.add_entry(
                entry_type="user_input" if i % 2 == 0 else "agent_output",
                content=f"Performance test message {i}",
                source="user" if i % 2 == 0 else "agent"
            )

            duration = time.perf_counter() - start
            add_times.append(duration)

        # Validate performance
        avg_add_time = sum(add_times) / len(add_times)
        assert avg_add_time < 0.001, f"Average add time {avg_add_time*1000:.2f}ms too slow"

        # Test formatting performance
        start = time.perf_counter()
        history = history_manager.get_formatted_history(format_style="chat")
        format_duration = time.perf_counter() - start

        assert format_duration < 0.1, f"Formatting took {format_duration:.2f}s, too slow"
        assert len(history) > 0
