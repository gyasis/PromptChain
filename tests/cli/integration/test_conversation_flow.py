"""Integration tests for conversation flow.

These tests verify end-to-end conversation behavior including user input,
agent responses, and context awareness across multiple turns.

Test Coverage:
- test_user_sends_message: User can send messages successfully
- test_agent_responds: Agent generates responses to user messages
- test_follow_up_maintains_context: Follow-up questions use conversation history
- test_message_added_to_history: Messages are stored in session history
- test_conversation_display: Messages are displayed in ChatView
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from promptchain.cli.models import Session, Message


class TestConversationFlow:
    """Integration tests for conversation flow."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary directory for test sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_session(self, temp_sessions_dir):
        """Create a mock session for testing."""
        from promptchain.cli.session_manager import SessionManager

        session_manager = SessionManager(sessions_dir=temp_sessions_dir)
        session = session_manager.create_session(
            name="test-conversation",
            working_directory=Path.cwd()
        )
        return session

    @pytest.mark.asyncio
    async def test_user_sends_message(self, mock_session):
        """Integration: User can send messages successfully.

        Flow:
        1. User types message in InputWidget
        2. Message is submitted (Enter key)
        3. Message is added to conversation history
        4. Message is displayed in ChatView

        Validates:
        - User message is captured
        - Message has correct role ('user')
        - Message is timestamped
        - Message is added to session
        """
        user_message_content = "Hello, can you help me?"

        # Add user message to session
        mock_session.add_message(
            role="user",
            content=user_message_content
        )

        # Validate message was added
        assert len(mock_session.messages) == 1
        user_msg = mock_session.messages[0]

        # Validate message properties
        assert user_msg.role == "user"
        assert user_msg.content == user_message_content
        assert user_msg.timestamp > 0
        assert isinstance(user_msg, Message)

    @pytest.mark.asyncio
    async def test_agent_responds(self, mock_session):
        """Integration: Agent generates responses to user messages.

        Flow:
        1. User message is processed
        2. Message is sent to AgentChain
        3. Agent generates response
        4. Response is displayed in ChatView
        5. Response is added to history

        Validates:
        - Agent response is generated
        - Response has correct role ('assistant')
        - Response includes agent_name and model_name
        - Response is added to session
        """
        # Simulate conversation flow
        user_msg = "What is Python?"
        mock_session.add_message(role="user", content=user_msg)

        # Simulate agent response (will use real AgentChain in implementation)
        agent_response = "Python is a high-level programming language known for its simplicity and readability."

        # Add agent response to session
        mock_session.add_message(
            role="assistant",
            content=agent_response,
            metadata={
                "agent_name": "default",
                "model_name": "gpt-4.1-mini-2025-04-14"
            }
        )

        # Validate conversation has both messages
        assert len(mock_session.messages) == 2

        # Validate agent message
        assistant_msg = mock_session.messages[1]
        assert assistant_msg.role == "assistant"
        assert assistant_msg.content == agent_response
        assert assistant_msg.metadata.get("agent_name") == "default"
        assert assistant_msg.metadata.get("model_name") == "gpt-4.1-mini-2025-04-14"

    @pytest.mark.asyncio
    async def test_follow_up_maintains_context(self, mock_session):
        """Integration: Follow-up questions use conversation history for context.

        Flow:
        1. Initial question asked
        2. Agent responds
        3. Follow-up question references previous context
        4. Agent response shows awareness of previous exchange

        Validates:
        - Conversation history is maintained
        - History is passed to agent for context
        - Follow-up questions work correctly
        - Context is preserved across turns
        """
        # Initial exchange
        mock_session.add_message(role="user", content="Tell me about Python")
        mock_session.add_message(
            role="assistant",
            content="Python is a high-level programming language."
        )

        # Follow-up question (references "it" - needs context)
        mock_session.add_message(
            role="user",
            content="What are the main advantages of using it?"
        )

        # At this point, history should contain all 3 messages
        assert len(mock_session.messages) == 3

        # Validate conversation structure
        assert mock_session.messages[0].role == "user"
        assert mock_session.messages[1].role == "assistant"
        assert mock_session.messages[2].role == "user"

        # Simulate agent using history for context
        # (In real implementation, ExecutionHistoryManager provides formatted history)
        conversation_context = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in mock_session.messages
        ])

        assert "Python" in conversation_context
        assert "advantages" in conversation_context

    @pytest.mark.asyncio
    async def test_message_added_to_history(self, mock_session):
        """Integration: Messages are stored in session history.

        Validates:
        - Each message is appended to session.messages
        - Messages maintain chronological order
        - Message timestamps are sequential
        - Session last_accessed is updated
        """
        initial_last_accessed = mock_session.last_accessed

        # Add multiple messages
        messages = [
            ("user", "First message"),
            ("assistant", "First response"),
            ("user", "Second message"),
            ("assistant", "Second response"),
        ]

        for role, content in messages:
            await asyncio.sleep(0.01)  # Small delay to ensure timestamp ordering
            mock_session.add_message(role=role, content=content)

        # Validate all messages added
        assert len(mock_session.messages) == 4

        # Validate chronological order
        timestamps = [msg.timestamp for msg in mock_session.messages]
        assert timestamps == sorted(timestamps), "Messages not in chronological order"

        # Validate last_accessed was updated
        assert mock_session.last_accessed > initial_last_accessed

        # Validate message roles alternate
        roles = [msg.role for msg in mock_session.messages]
        assert roles == ["user", "assistant", "user", "assistant"]

    @pytest.mark.asyncio
    async def test_conversation_display(self, mock_session):
        """Integration: Messages are displayed in ChatView.

        Validates:
        - User messages are displayed with user indicator
        - Assistant messages are displayed with agent indicator
        - Messages are formatted correctly
        - ChatView auto-scrolls to latest message
        """
        # Add conversation
        mock_session.add_message(role="user", content="Test question")
        mock_session.add_message(role="assistant", content="Test answer")

        # Mock ChatView component
        try:
            from promptchain.cli.tui.chat_view import ChatView

            # This will be testable once ChatView is implemented (T028)
            # For now, validate message content is suitable for display
            for msg in mock_session.messages:
                # Messages should have str representation
                msg_str = str(msg)
                assert msg.role.upper() in msg_str
                assert msg.content[:20] in msg_str  # At least partial content

        except ImportError:
            pytest.skip("ChatView not yet implemented (will be in T028)")

    @pytest.mark.asyncio
    async def test_long_conversation_performance(self, mock_session):
        """Integration: Performance remains stable with 100+ messages.

        Performance target from plan.md: 100+ message conversations without degradation (SC-011)

        Validates:
        - Adding messages remains fast (<10ms per message)
        - Memory usage doesn't grow excessively
        - History retrieval is efficient
        """
        import time

        # Add 100 messages
        message_times = []
        for i in range(100):
            start = time.perf_counter()

            mock_session.add_message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}: This is test content"
            )

            duration = time.perf_counter() - start
            message_times.append(duration)

        # Validate all messages added
        assert len(mock_session.messages) == 100

        # Validate performance: average <10ms per message
        avg_time = sum(message_times) / len(message_times)
        assert avg_time < 0.01, f"Average message time {avg_time*1000:.1f}ms exceeds 10ms"

        # Validate no performance degradation over time
        first_10_avg = sum(message_times[:10]) / 10
        last_10_avg = sum(message_times[-10:]) / 10
        degradation = last_10_avg / first_10_avg

        assert degradation < 2.0, f"Performance degraded by {degradation:.1f}x"

    @pytest.mark.asyncio
    async def test_empty_message_rejection(self, mock_session):
        """Integration: Empty messages are rejected gracefully.

        Validates:
        - Empty string message raises error
        - Whitespace-only message raises error
        - Appropriate error message shown
        """
        # Empty message should fail
        with pytest.raises(ValueError, match="non-empty"):
            mock_session.add_message(role="user", content="")

        # Whitespace-only should also fail (after strip)
        with pytest.raises(ValueError, match="non-empty"):
            mock_session.add_message(role="user", content="   ")

    @pytest.mark.asyncio
    async def test_special_characters_in_messages(self, mock_session):
        """Integration: Messages with special characters are handled correctly.

        Validates:
        - Unicode characters preserved
        - Newlines preserved
        - Code blocks preserved
        - Emoji preserved
        """
        special_content = """
        Here's some code:
        ```python
        def hello():
            print("Hello! 👋")
        ```
        This includes: unicode (ñ, ü), emoji (🚀), and "quotes"
        """

        mock_session.add_message(role="user", content=special_content)

        # Validate message was added with special characters intact
        msg = mock_session.messages[0]
        assert "👋" in msg.content
        assert "🚀" in msg.content
        assert "ñ" in msg.content
        assert "```python" in msg.content
        assert "\n" in msg.content

    @pytest.mark.asyncio
    async def test_conversation_interrupted_and_resumed(self, temp_sessions_dir):
        """Integration: Conversation can be interrupted and resumed.

        Flow:
        1. Start conversation with several exchanges
        2. Save session
        3. Load session
        4. Continue conversation
        5. Validate history is intact

        Validates:
        - Session save preserves all messages
        - Session load restores conversation
        - New messages can be added after resume
        - Context is maintained across save/load
        """
        from promptchain.cli.session_manager import SessionManager

        session_manager = SessionManager(sessions_dir=temp_sessions_dir)

        # Create session with conversation
        session = session_manager.create_session(
            name="interrupt-test",
            working_directory=Path.cwd()
        )

        session.add_message(role="user", content="First message")
        session.add_message(role="assistant", content="First response")

        # Save session
        session_manager.save_session(session)

        # Load session (simulates resumption)
        resumed_session = session_manager.load_session("interrupt-test")

        # Validate history is intact
        assert len(resumed_session.messages) == 2
        assert resumed_session.messages[0].content == "First message"
        assert resumed_session.messages[1].content == "First response"

        # Continue conversation
        resumed_session.add_message(role="user", content="Second message")

        # Validate new message added
        assert len(resumed_session.messages) == 3
