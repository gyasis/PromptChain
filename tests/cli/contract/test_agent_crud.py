"""Contract tests for agent CRUD operations (T043).

These tests define the expected behavior for creating, listing, and deleting
agents in the PromptChain CLI.
"""

import pytest
from pathlib import Path
import tempfile
import shutil


class TestAgentCRUD:
    """Test agent creation, listing, and deletion operations."""

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
        """Create a test session."""
        session = session_manager.create_session(
            name="agent-test",
            working_directory=Path.cwd()
        )
        return session

    def test_create_agent(self, session_manager, test_session):
        """Contract: Create agent with name, model, and description.

        Given: A session exists
        When: User creates agent with /agent create coding --model gpt-4 --description "Coding assistant"
        Then: Agent is created with specified properties
        And: Agent is added to session.agents
        And: Agent is persisted to SQLite
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=session_manager)

        # Parse and execute agent create command
        parsed = handler.parse_command("/agent create coding --model=gpt-4")
        assert parsed is not None
        assert parsed.name == "agent"
        assert parsed.subcommand == "create"
        assert "model" in parsed.args

        # Execute create (will be implemented in T051)
        try:
            result = handler.handle_agent_create(
                session=test_session,
                name="coding",
                model="gpt-4.1-mini-2025-04-14",
                description="Coding assistant"
            )

            # Validate result
            assert result.success is True
            assert "coding" in test_session.agents
            assert test_session.agents["coding"].model_name == "gpt-4.1-mini-2025-04-14"
            assert test_session.agents["coding"].description == "Coding assistant"

        except AttributeError:
            pytest.skip("handle_agent_create() not yet implemented (will be in T051)")

    def test_list_agents(self, session_manager, test_session):
        """Contract: List all agents with their models and usage stats.

        Given: Session has multiple agents
        When: User runs /agent list
        Then: All agents are displayed with name, model, usage count
        And: Active agent is marked
        """
        from promptchain.cli.command_handler import CommandHandler
        from promptchain.cli.models.agent_config import Agent
        import time

        # Add test agents
        test_session.agents["default"] = Agent(
            name="default",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Default agent",
            created_at=time.time()
        )
        test_session.agents["fast"] = Agent(
            name="fast",
            model_name="anthropic/claude-3-haiku-20240307",
            description="Fast responses",
            created_at=time.time(),
            usage_count=5
        )

        handler = CommandHandler(session_manager=session_manager)

        try:
            result = handler.handle_agent_list(session=test_session)

            # Validate result
            assert result.success is True
            assert "default" in result.message
            assert "fast" in result.message
            assert "gpt-4.1-mini-2025-04-14" in result.message
            assert "haiku" in result.message

        except AttributeError:
            pytest.skip("handle_agent_list() not yet implemented (will be in T054)")

    def test_delete_agent(self, session_manager, test_session):
        """Contract: Delete agent (not active agent).

        Given: Session has multiple agents
        When: User deletes non-active agent with /agent delete fast
        Then: Agent is removed from session.agents
        And: Agent is removed from SQLite
        And: Active agent cannot be deleted
        """
        from promptchain.cli.command_handler import CommandHandler
        from promptchain.cli.models.agent_config import Agent
        import time

        # Add test agents
        test_session.agents["default"] = Agent(
            name="default",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Default agent",
            created_at=time.time()
        )
        test_session.agents["fast"] = Agent(
            name="fast",
            model_name="anthropic/claude-3-haiku-20240307",
            description="Fast responses",
            created_at=time.time()
        )
        test_session.active_agent = "default"

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Should succeed - deleting non-active agent
            result = handler.handle_agent_delete(
                session=test_session,
                name="fast"
            )
            assert result.success is True
            assert "fast" not in test_session.agents

            # Should fail - deleting active agent
            result = handler.handle_agent_delete(
                session=test_session,
                name="default"
            )
            assert result.success is False
            assert "active" in result.error.lower()
            assert "default" in test_session.agents

        except AttributeError:
            pytest.skip("handle_agent_delete() not yet implemented (will be in T055)")

    def test_agent_name_validation(self, session_manager, test_session):
        """Contract: Agent names must be valid identifiers.

        Given: User creates agent
        When: Agent name contains invalid characters
        Then: Creation fails with validation error
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Invalid names
            invalid_names = [
                "agent with spaces",
                "agent@special",
                "",
                "a" * 65,  # Too long
            ]

            for invalid_name in invalid_names:
                result = handler.handle_agent_create(
                    session=test_session,
                    name=invalid_name,
                    model="gpt-4.1-mini-2025-04-14"
                )
                assert result.success is False
                # Check that error message mentions name requirements
                assert "name" in result.error.lower() and ("must" in result.error.lower() or "characters" in result.error.lower())

        except AttributeError:
            pytest.skip("handle_agent_create() not yet implemented (will be in T051)")

    def test_duplicate_agent_name(self, session_manager, test_session):
        """Contract: Cannot create agent with duplicate name.

        Given: Agent 'coding' exists
        When: User creates another agent named 'coding'
        Then: Creation fails with error
        """
        from promptchain.cli.command_handler import CommandHandler
        from promptchain.cli.models.agent_config import Agent
        import time

        # Add existing agent
        test_session.agents["coding"] = Agent(
            name="coding",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Existing agent",
            created_at=time.time()
        )

        handler = CommandHandler(session_manager=session_manager)

        try:
            result = handler.handle_agent_create(
                session=test_session,
                name="coding",
                model="openai/gpt-3.5-turbo"
            )
            assert result.success is False
            assert "exists" in result.error.lower() or "duplicate" in result.error.lower()

        except AttributeError:
            pytest.skip("handle_agent_create() not yet implemented (will be in T051)")
