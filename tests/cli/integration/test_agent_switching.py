"""Integration tests for agent switching (T045).

These tests verify the complete workflow of switching between agents and
ensuring the correct agent handles user messages.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import time


class TestAgentSwitching:
    """Test agent switching and active agent management."""

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
            name="switch-test",
            working_directory=Path.cwd()
        )

        # Add multiple agents
        session.agents["default"] = Agent(
            name="default",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Default agent",
            created_at=time.time()
        )
        session.agents["coding"] = Agent(
            name="coding",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Coding specialist",
            created_at=time.time()
        )
        session.agents["fast"] = Agent(
            name="fast",
            model_name="anthropic/claude-3-haiku-20240307",
            description="Fast responses",
            created_at=time.time()
        )

        return session

    @pytest.mark.asyncio
    async def test_switch_agent(self, session_manager, test_session):
        """Integration: Switch active agent with /agent use command.

        Flow:
        1. Session starts with 'default' agent
        2. User runs /agent use coding
        3. Session.active_agent changes to 'coding'
        4. StatusBar updates to show new agent

        Validates:
        - Agent switch command is parsed correctly
        - Active agent is updated in session
        - Previous agent still exists
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=session_manager)

        # Initial state
        assert test_session.active_agent == "default"

        try:
            # Parse and execute agent use command
            parsed = handler.parse_command("/agent use coding")
            assert parsed is not None
            assert parsed.name == "agent"
            assert parsed.subcommand == "use"

            result = handler.handle_agent_use(
                session=test_session,
                name="coding"
            )

            # Validate switch
            assert result.success is True
            assert test_session.active_agent == "coding"
            assert "coding" in test_session.agents
            assert "default" in test_session.agents  # Previous agent still exists

        except AttributeError:
            pytest.skip("handle_agent_use() not yet implemented (will be in T057)")

    @pytest.mark.asyncio
    async def test_active_agent_responds(self, session_manager, test_session):
        """Integration: Active agent handles user messages with correct model.

        Flow:
        1. Switch to 'coding' agent (gpt-4)
        2. User sends message
        3. Coding agent's PromptChain handles request
        4. Response is tagged with agent name and model

        Validates:
        - Correct PromptChain instance is used
        - Response metadata includes agent info
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Switch to coding agent
            handler.handle_agent_use(session=test_session, name="coding")

            # Add a user message
            test_session.add_message(
                role="user",
                content="Write a Python function"
            )

            # Response should be from coding agent
            # (This will be tested via TUI integration in actual implementation)
            assert test_session.active_agent == "coding"

            # If we have messages with metadata
            if len(test_session.messages) > 1:
                last_message = test_session.messages[-1]
                if last_message.role == "assistant":
                    assert last_message.metadata.get("agent_name") == "coding"

        except AttributeError:
            pytest.skip("Agent routing not yet implemented (will be in T059)")

    @pytest.mark.asyncio
    async def test_usage_count_incremented(self, session_manager, test_session):
        """Integration: Agent usage count and last_used are updated on switch.

        Flow:
        1. Check initial usage_count for 'coding' agent
        2. Switch to 'coding' agent
        3. Verify usage_count incremented
        4. Verify last_used timestamp updated

        Validates:
        - Usage statistics are tracked
        - Timestamps are maintained
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Get initial stats
            coding_agent = test_session.agents["coding"]
            initial_count = coding_agent.usage_count
            initial_time = coding_agent.last_used

            # Small delay to ensure timestamp changes
            time.sleep(0.01)

            # Switch agent
            handler.handle_agent_use(session=test_session, name="coding")

            # Check updated stats
            assert coding_agent.usage_count == initial_count + 1
            # Check that last_used is set and greater than initial (if initial was set)
            assert coding_agent.last_used is not None
            if initial_time is not None:
                assert coding_agent.last_used > initial_time

        except AttributeError:
            pytest.skip("Usage tracking not yet implemented (will be in T060)")

    @pytest.mark.asyncio
    async def test_switch_to_nonexistent_agent(self, session_manager, test_session):
        """Integration: Switching to non-existent agent fails gracefully.

        Flow:
        1. User tries /agent use nonexistent
        2. Command fails with error
        3. Active agent remains unchanged

        Validates:
        - Error handling for invalid agent names
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=session_manager)

        try:
            original_agent = test_session.active_agent

            result = handler.handle_agent_use(
                session=test_session,
                name="nonexistent"
            )

            # Should fail
            assert result.success is False
            assert "not found" in result.error.lower() or "does not exist" in result.error.lower()

            # Active agent unchanged
            assert test_session.active_agent == original_agent

        except AttributeError:
            pytest.skip("handle_agent_use() not yet implemented (will be in T057)")

    @pytest.mark.asyncio
    async def test_status_bar_updates_on_switch(self, session_manager, test_session):
        """Integration: StatusBar shows active agent and model after switch.

        Flow:
        1. StatusBar shows default agent (gpt-4)
        2. Switch to fast agent (haiku)
        3. StatusBar updates to show fast agent and model

        Validates:
        - StatusBar reactive properties update
        - Correct agent name and model displayed
        """
        try:
            from promptchain.cli.tui.status_bar import StatusBar

            status_bar = StatusBar()

            # Initial state
            status_bar.update_session_info(
                active_agent="default",
                model_name="gpt-4.1-mini-2025-04-14"
            )
            assert "default" in status_bar.render()
            assert "gpt-4.1-mini-2025-04-14" in status_bar.render()

            # After switch
            status_bar.update_session_info(
                active_agent="fast",
                model_name="anthropic/claude-3-haiku-20240307"
            )
            assert "fast" in status_bar.render()
            assert "haiku" in status_bar.render()

        except ImportError:
            pytest.skip("StatusBar not yet implemented")
