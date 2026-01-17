"""Integration tests for graceful degradation when MCP servers fail (T068a).

These tests verify that the CLI continues to function properly even when
MCP servers fail to connect or encounter errors during operation.

Test Coverage:
- test_session_creation_succeeds_with_failed_server: Session created even if MCP fails
- test_failed_server_does_not_prevent_other_servers: One failure doesn't block others
- test_cli_commands_work_with_failed_mcp_server: Commands execute despite failures
- test_agent_can_work_without_mcp_tools: Agents function without MCP server tools
- test_mcp_reconnection_after_failure: Can retry connection after failure
- test_auto_connect_failure_does_not_block_startup: Auto-connect failures are graceful
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from promptchain.cli.models.mcp_config import MCPServerConfig
from promptchain.cli.models.agent_config import Agent
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.utils.mcp_manager import MCPManager


class TestMCPGracefulDegradation:
    """Integration tests for graceful degradation when MCP servers fail."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary sessions directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        """Create SessionManager for testing."""
        return SessionManager(sessions_dir=temp_sessions_dir)

    @pytest.fixture
    def failing_server(self):
        """Create MCP server config that will fail."""
        return MCPServerConfig(
            id="failing_server",
            type="stdio",
            command="nonexistent-mcp-server",
            auto_connect=False
        )

    @pytest.fixture
    def working_server(self):
        """Create MCP server config for successful connection."""
        return MCPServerConfig(
            id="filesystem",
            type="stdio",
            command="mcp-server-filesystem",
            args=["--root", "/tmp"],
            auto_connect=False
        )

    @pytest.fixture
    def test_agent(self):
        """Create test agent."""
        return Agent(
            name="test_agent",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Test agent for graceful degradation"
        )

    @pytest.mark.asyncio
    async def test_session_creation_succeeds_with_failed_server(
        self, session_manager, failing_server, test_agent
    ):
        """Graceful: Session creation succeeds even if MCP server fails.

        Validates:
        - Session created successfully
        - Failed MCP server doesn't prevent session creation
        - Session has failed server in config (marked as error)
        - Agent can be added to session
        - Session is fully functional
        """
        # Create session with failing server (should not raise)
        session = session_manager.create_session(
            name="degradation-test",
            working_directory=Path("/tmp"),
            mcp_servers=[failing_server]
        )

        # Session should be created successfully
        assert session is not None
        assert session.name == "degradation-test"

        # Failed server should be in session config
        assert len(session.mcp_servers) == 1
        assert session.mcp_servers[0].id == "failing_server"

        # Can add agent to session
        session.agents["test_agent"] = test_agent
        assert "test_agent" in session.agents

        # Session is fully functional
        assert session.working_directory == Path("/tmp")
        assert session.active_agent == "default"  # Default agent is active

    @pytest.mark.asyncio
    async def test_failed_server_does_not_prevent_other_servers(
        self, session_manager, failing_server, working_server
    ):
        """Graceful: One failed server doesn't prevent others from connecting.

        Validates:
        - Multiple servers can be configured
        - Failed server marked as error
        - Working server connects successfully
        - Failed server doesn't block other connections
        - Each server state is independent
        """
        # Create session with both failing and working servers
        session = session_manager.create_session(
            name="multi-server-test",
            working_directory=Path("/tmp"),
            mcp_servers=[failing_server, working_server]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            # First call for failing server
            failing_helper = Mock()
            failing_helper.connect_mcp_async = AsyncMock(
                side_effect=Exception("Connection failed")
            )

            # Second call for working server
            working_helper = Mock()
            working_helper.connect_mcp_async = AsyncMock()
            working_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}}
            ]

            MockMCPHelper.side_effect = [failing_helper, working_helper]

            # Attempt to connect both servers
            await mcp_manager.connect_server("failing_server")
            await mcp_manager.connect_server("filesystem")

            # Verify failing server marked as error
            assert failing_server.state == "error"
            assert failing_server.error_message is not None
            assert len(failing_server.discovered_tools) == 0

            # Verify working server connected successfully
            assert working_server.state == "connected"
            assert working_server.error_message is None
            assert len(working_server.discovered_tools) == 1
            assert "mcp_filesystem_read_file" in working_server.discovered_tools

    @pytest.mark.asyncio
    async def test_cli_commands_work_with_failed_mcp_server(
        self, session_manager, failing_server, test_agent
    ):
        """Graceful: CLI commands execute despite MCP server failures.

        Validates:
        - Session can be saved with failed server
        - Session can be loaded with failed server
        - Agent commands work with failed server
        - Failed server state persists correctly
        """
        # Create session with failing server
        session = session_manager.create_session(
            name="command-test",
            working_directory=Path("/tmp"),
            mcp_servers=[failing_server]
        )
        session.agents["test_agent"] = test_agent

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper = Mock()
            mock_helper.connect_mcp_async = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            MockMCPHelper.return_value = mock_helper

            # Attempt connection (will fail)
            await mcp_manager.connect_server("failing_server")

        # Save session with failed server (should not raise)
        session_manager.save_session(session)

        # Load session with failed server (should not raise)
        loaded_session = session_manager.load_session("command-test")

        # Verify session loaded correctly
        assert loaded_session.name == "command-test"
        assert "test_agent" in loaded_session.agents

        # Verify failed server state persisted
        assert len(loaded_session.mcp_servers) == 1
        assert loaded_session.mcp_servers[0].state == "error"

    @pytest.mark.asyncio
    async def test_agent_can_work_without_mcp_tools(
        self, session_manager, failing_server, test_agent
    ):
        """Graceful: Agents function without MCP server tools.

        Validates:
        - Agent created without MCP tools
        - Agent has no tools from failed server
        - Agent is still functional (can be selected)
        - Failed server doesn't corrupt agent config
        """
        session = session_manager.create_session(
            name="agent-no-tools-test",
            working_directory=Path("/tmp"),
            mcp_servers=[failing_server]
        )
        session.agents["test_agent"] = test_agent

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper = Mock()
            mock_helper.connect_mcp_async = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            MockMCPHelper.return_value = mock_helper

            # Attempt connection (will fail)
            await mcp_manager.connect_server("failing_server")

            # Try to register tools (should fail gracefully)
            with pytest.raises(ValueError, match="Server 'failing_server' is not connected"):
                await mcp_manager.register_tools_with_agent(
                    server_id="failing_server",
                    agent_name="test_agent"
                )

        # Agent should still be functional with no tools
        agent = session.agents["test_agent"]
        assert agent.name == "test_agent"
        assert agent.model_name == "gpt-4.1-mini-2025-04-14"
        assert len(agent.tools) == 0  # No tools from failed server

        # Agent can still be selected
        session.active_agent = "test_agent"
        assert session.active_agent == "test_agent"

    @pytest.mark.asyncio
    async def test_mcp_reconnection_after_failure(
        self, session_manager, failing_server
    ):
        """Graceful: Can retry connection after initial failure.

        Validates:
        - Initial connection fails
        - Server marked as error
        - Retry connection succeeds
        - Server state transitions from error to connected
        - Error message cleared on successful connection
        """
        session = session_manager.create_session(
            name="reconnection-test",
            working_directory=Path("/tmp"),
            mcp_servers=[failing_server]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            # First attempt fails
            failing_helper = Mock()
            failing_helper.connect_mcp_async = AsyncMock(
                side_effect=Exception("Initial connection failed")
            )

            # Second attempt succeeds
            working_helper = Mock()
            working_helper.connect_mcp_async = AsyncMock()
            working_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_failing_server_tool1"}}
            ]

            MockMCPHelper.side_effect = [failing_helper, working_helper]

            # First attempt (fails)
            await mcp_manager.connect_server("failing_server")
            assert failing_server.state == "error"
            assert failing_server.error_message is not None

            # Retry connection (succeeds)
            await mcp_manager.connect_server("failing_server")
            assert failing_server.state == "connected"
            assert failing_server.error_message is None  # Error cleared
            assert len(failing_server.discovered_tools) == 1

    @pytest.mark.asyncio
    async def test_auto_connect_failure_does_not_block_startup(
        self, session_manager, test_agent
    ):
        """Graceful: Auto-connect failures don't prevent CLI startup.

        Validates:
        - Server with auto_connect=True that fails
        - Startup continues despite failure
        - Failed server marked as error
        - Other session initialization completes
        - Session is fully functional
        """
        # Create server with auto_connect enabled
        auto_connect_server = MCPServerConfig(
            id="auto_server",
            type="stdio",
            command="nonexistent-auto-server",
            auto_connect=True  # This will be attempted at startup
        )

        session = session_manager.create_session(
            name="auto-connect-test",
            working_directory=Path("/tmp"),
            mcp_servers=[auto_connect_server]
        )
        session.agents["test_agent"] = test_agent

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper = Mock()
            mock_helper.connect_mcp_async = AsyncMock(
                side_effect=Exception("Auto-connect failed")
            )
            MockMCPHelper.return_value = mock_helper

            # Simulate startup auto-connect (should not raise)
            connected_ids = await mcp_manager.connect_all_auto_servers()

            # No servers connected (failed gracefully)
            assert len(connected_ids) == 0

            # Server marked as error
            assert auto_connect_server.state == "error"
            assert auto_connect_server.error_message is not None

        # Session is still fully functional
        assert session.active_agent == "default"  # Default agent is active
        assert "test_agent" in session.agents
        assert session.working_directory == Path("/tmp")

    @pytest.mark.asyncio
    async def test_disconnect_from_error_state_server(
        self, session_manager, failing_server
    ):
        """Graceful: Can disconnect from server in error state.

        Validates:
        - Server in error state
        - Disconnect succeeds
        - Server transitions to disconnected state
        - No exceptions raised
        """
        session = session_manager.create_session(
            name="disconnect-error-test",
            working_directory=Path("/tmp"),
            mcp_servers=[failing_server]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper = Mock()
            mock_helper.connect_mcp_async = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            MockMCPHelper.return_value = mock_helper

            # Connect (fails)
            await mcp_manager.connect_server("failing_server")
            assert failing_server.state == "error"

        # Disconnect from error state (should succeed)
        await mcp_manager.disconnect_server("failing_server")

        # Server should be disconnected
        assert failing_server.state == "disconnected"

    @pytest.mark.asyncio
    async def test_get_all_discovered_tools_excludes_failed_servers(
        self, session_manager, failing_server, working_server
    ):
        """Graceful: get_all_discovered_tools excludes tools from failed servers.

        Validates:
        - Failed servers have no tools
        - get_all_discovered_tools only returns tools from connected servers
        - Failed servers don't pollute tool list
        """
        session = session_manager.create_session(
            name="tools-list-test",
            working_directory=Path("/tmp"),
            mcp_servers=[failing_server, working_server]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            # Failing server helper
            failing_helper = Mock()
            failing_helper.connect_mcp_async = AsyncMock(
                side_effect=Exception("Connection failed")
            )

            # Working server helper
            working_helper = Mock()
            working_helper.connect_mcp_async = AsyncMock()
            working_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}},
                {"type": "function", "function": {"name": "mcp_filesystem_write_file"}}
            ]

            MockMCPHelper.side_effect = [failing_helper, working_helper]

            # Connect both
            await mcp_manager.connect_server("failing_server")
            await mcp_manager.connect_server("filesystem")

            # Get all discovered tools
            all_tools = mcp_manager.get_all_discovered_tools()

            # Should only include tools from working server
            assert len(all_tools) == 2
            assert "mcp_filesystem_read_file" in all_tools
            assert "mcp_filesystem_write_file" in all_tools

            # No tools from failing server
            assert len(failing_server.discovered_tools) == 0
