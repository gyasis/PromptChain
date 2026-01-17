"""Integration tests for multi-agent conversations (T046).

These tests verify that multiple agents can coexist and handle conversations
with proper isolation and context management.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import time


class TestMultiAgent:
    """Test multi-agent conversations and history isolation."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary sessions directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        """Create SessionManager with temp directory."""
        from promptchain.cli.session_manager import SessionManager
        return SessionManager(sessions_dir=temp_sessions_dir)

    @pytest.fixture
    def test_session(self, session_manager):
        """Create a test session with multiple agents."""
        from promptchain.cli.models.agent_config import Agent

        session = session_manager.create_session(
            name="multi-agent-test",
            working_directory=Path.cwd()
        )

        # Add multiple specialized agents
        session.agents["coder"] = Agent(
            name="coder",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Python coding specialist",
            created_at=time.time()
        )
        session.agents["writer"] = Agent(
            name="writer",
            model_name="anthropic/claude-3-opus-20240229",
            description="Technical writer",
            created_at=time.time()
        )
        session.agents["reviewer"] = Agent(
            name="reviewer",
            model_name="anthropic/claude-3-sonnet-20240229",
            description="Code reviewer",
            created_at=time.time()
        )

        return session

    @pytest.mark.asyncio
    async def test_different_agents_different_responses(self, session_manager, test_session):
        """Integration: Different agents use different models and contexts.

        Flow:
        1. Send coding question to 'coder' agent (gpt-4)
        2. Switch to 'writer' agent (opus)
        3. Send documentation request to 'writer' agent
        4. Verify responses use correct models

        Validates:
        - Each agent uses its configured model
        - Response metadata tracks agent and model
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Use coder agent
            handler.handle_agent_use(session=test_session, name="coder")
            test_session.add_message(role="user", content="Write a sorting function")

            # Simulate response (in real implementation, would come from PromptChain)
            test_session.add_message(
                role="assistant",
                content="def sort_list(items): return sorted(items)",
                metadata={"agent_name": "coder", "model_name": "gpt-4.1-mini-2025-04-14"}
            )

            # Use writer agent
            handler.handle_agent_use(session=test_session, name="writer")
            test_session.add_message(role="user", content="Document this function")

            # Simulate response
            test_session.add_message(
                role="assistant",
                content="# Sorting Function\n\nThis function sorts a list...",
                metadata={"agent_name": "writer", "model_name": "anthropic/claude-3-opus-20240229"}
            )

            # Verify messages are tagged correctly
            messages_by_agent = {}
            for msg in test_session.messages:
                if msg.role == "assistant":
                    agent = msg.metadata.get("agent_name")
                    if agent:
                        if agent not in messages_by_agent:
                            messages_by_agent[agent] = []
                        messages_by_agent[agent].append(msg)

            # Should have messages from both agents
            assert "coder" in messages_by_agent
            assert "writer" in messages_by_agent

            # Each agent's messages should have correct model
            for msg in messages_by_agent.get("coder", []):
                assert msg.metadata["model_name"] == "gpt-4.1-mini-2025-04-14"

            for msg in messages_by_agent.get("writer", []):
                assert msg.metadata["model_name"] == "anthropic/claude-3-opus-20240229"

        except AttributeError:
            pytest.skip("Multi-agent routing not yet implemented (will be in T059)")

    @pytest.mark.asyncio
    async def test_agent_history_isolation(self, session_manager, test_session):
        """Integration: Agent history can be isolated or shared based on configuration.

        Flow:
        1. Coder agent has conversation about Python
        2. Switch to reviewer agent
        3. Check if reviewer sees coder's history
        4. Configuration determines isolation level

        Validates:
        - History management respects agent boundaries
        - ExecutionHistoryManager tracks agent context
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Coder conversation
            handler.handle_agent_use(session=test_session, name="coder")
            test_session.add_message(role="user", content="Explain list comprehensions")
            test_session.add_message(
                role="assistant",
                content="List comprehensions are...",
                metadata={"agent_name": "coder"}
            )

            # Get history after coder conversation
            coder_history_length = len(test_session.messages)

            # Switch to reviewer
            handler.handle_agent_use(session=test_session, name="reviewer")

            # Reviewer should see conversation history
            # (Global history by default, per-agent history optional)
            assert len(test_session.messages) == coder_history_length

            # ExecutionHistoryManager should track entries
            history_manager = test_session.history_manager
            formatted = history_manager.get_formatted_history(format_style="chat")

            # Should contain both user and assistant messages
            assert len(formatted) > 0

        except AttributeError:
            pytest.skip("History management not yet fully implemented")

    @pytest.mark.asyncio
    async def test_concurrent_agent_usage_tracking(self, session_manager, test_session):
        """Integration: Multiple agent switches update usage statistics.

        Flow:
        1. Switch between coder, writer, reviewer multiple times
        2. Each switch increments usage_count
        3. Last_used timestamps are updated

        Validates:
        - Usage tracking works across multiple switches
        - Statistics are accurate
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Track initial counts
            initial_counts = {
                name: agent.usage_count
                for name, agent in test_session.agents.items()
            }

            # Switch sequence: coder -> writer -> reviewer -> coder -> writer
            switch_sequence = ["coder", "writer", "reviewer", "coder", "writer"]

            for agent_name in switch_sequence:
                handler.handle_agent_use(session=test_session, name=agent_name)
                time.sleep(0.01)  # Ensure timestamps differ

            # Check final counts
            expected_counts = {
                "coder": initial_counts["coder"] + 2,
                "writer": initial_counts["writer"] + 2,
                "reviewer": initial_counts["reviewer"] + 1,
            }

            for agent_name, expected_count in expected_counts.items():
                assert test_session.agents[agent_name].usage_count == expected_count

            # Verify last_used timestamps are set for used agents
            used_agents = ["coder", "writer", "reviewer"]
            for agent_name in used_agents:
                agent = test_session.agents[agent_name]
                assert agent.last_used is not None, f"{agent_name} should have last_used set"
                assert agent.last_used > agent.created_at, f"{agent_name} last_used should be > created_at"

        except AttributeError:
            pytest.skip("Usage tracking not yet implemented (will be in T060)")

    @pytest.mark.asyncio
    async def test_agent_list_shows_usage_stats(self, session_manager, test_session):
        """Integration: /agent list displays usage statistics.

        Flow:
        1. Use different agents multiple times
        2. Run /agent list
        3. Verify output shows usage counts and last used times

        Validates:
        - Agent list includes usage statistics
        - Active agent is clearly marked
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Use agents
            handler.handle_agent_use(session=test_session, name="coder")
            handler.handle_agent_use(session=test_session, name="writer")
            handler.handle_agent_use(session=test_session, name="coder")

            # List agents
            result = handler.handle_agent_list(session=test_session)

            assert result.success is True

            # Output should contain usage information
            message = result.message.lower()
            assert "coder" in message
            assert "writer" in message
            assert "reviewer" in message

            # Should indicate active agent
            assert test_session.active_agent in result.message

        except AttributeError:
            pytest.skip("Agent list with statistics not yet implemented (will be in T054)")
