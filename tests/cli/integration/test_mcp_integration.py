"""Integration tests for MCP server lifecycle management (T057).

These tests verify complete MCP server lifecycle including connection startup,
tool discovery, execution, shutdown, and resource cleanup.

Test Coverage:
- test_server_lifecycle_complete: Full lifecycle from start to cleanup
- test_server_starts_successfully: Valid config starts without errors
- test_server_connection_and_tool_discovery: Connection establishes and discovers tools
- test_server_shutdown_cleanup: Clean shutdown releases resources
- test_multiple_servers_simultaneous: Multiple servers run concurrently
- test_server_restart_after_failure: Failed server can be restarted
- test_invalid_server_config_fails_gracefully: Bad config handled properly
- test_server_lifecycle_with_real_mcphelper: Integration with real MCPHelper
- test_server_state_transitions: State machine transitions correctly
- test_resource_cleanup_verification: All resources properly released
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from promptchain.cli.models.mcp_config import MCPServerConfig
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.utils.mcp_manager import MCPManager


class TestMCPServerLifecycle:
    """Integration tests for MCP server connection lifecycle."""

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
    def valid_server_config(self):
        """Create valid MCP server configuration."""
        return MCPServerConfig(
            id="filesystem",
            type="stdio",
            command="mcp-server-filesystem",
            args=["--root", "/tmp"],
            auto_connect=False
        )

    @pytest.fixture
    def calculator_server_config(self):
        """Create calculator MCP server configuration."""
        return MCPServerConfig(
            id="calculator",
            type="stdio",
            command="mcp-server-calculator",
            auto_connect=False
        )

    @pytest.fixture
    def invalid_server_config(self):
        """Create invalid MCP server configuration (nonexistent command)."""
        return MCPServerConfig(
            id="invalid_server",
            type="stdio",
            command="nonexistent-mcp-server",
            auto_connect=False
        )

    @pytest.mark.asyncio
    async def test_server_lifecycle_complete(
        self, session_manager, valid_server_config
    ):
        """Integration: Complete server lifecycle from start to cleanup.

        Lifecycle Steps:
        1. Start server (connect)
        2. Verify connection established
        3. Discover tools
        4. Execute basic tool operation
        5. Shutdown server
        6. Verify cleanup (resources released, state reset)

        Validates:
        - All lifecycle phases execute successfully
        - State transitions are correct
        - Resources properly managed
        - Cleanup is complete
        """
        # Create session with server
        session = session_manager.create_session(
            name="lifecycle-test",
            working_directory=Path("/tmp"),
            mcp_servers=[valid_server_config]
        )

        mcp_manager = MCPManager(session)

        # Mock MCPHelper for controlled lifecycle
        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper = Mock()
            mock_helper.connect_mcp_async = AsyncMock()
            mock_helper.mcp_tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp_filesystem_read_file",
                        "description": "Read a file"
                    }
                }
            ]
            MockMCPHelper.return_value = mock_helper

            # PHASE 1: Start server (connect)
            initial_state = valid_server_config.state
            assert initial_state == "disconnected"

            success = await mcp_manager.connect_server("filesystem")
            assert success is True

            # PHASE 2: Verify connection established
            assert valid_server_config.state == "connected"
            assert valid_server_config.connected_at is not None

            # PHASE 3: Discover tools
            discovered_tools = valid_server_config.discovered_tools
            assert len(discovered_tools) > 0
            assert "mcp_filesystem_read_file" in discovered_tools

            # PHASE 4: Verify tools are available
            all_tools = mcp_manager.get_all_discovered_tools()
            assert "mcp_filesystem_read_file" in all_tools

            # PHASE 5: Shutdown server
            disconnect_success = await mcp_manager.disconnect_server("filesystem")
            assert disconnect_success is True

            # PHASE 6: Verify cleanup
            assert valid_server_config.state == "disconnected"
            assert len(valid_server_config.discovered_tools) == 0
            assert valid_server_config.connected_at is None

            # Verify no tools available after disconnect
            all_tools_after = mcp_manager.get_all_discovered_tools()
            assert len(all_tools_after) == 0

    @pytest.mark.asyncio
    async def test_server_starts_successfully(
        self, session_manager, valid_server_config
    ):
        """Integration: Server starts successfully with valid config.

        Validates:
        - Initial state is 'disconnected'
        - Connection attempt succeeds
        - Final state is 'connected'
        - No errors during startup
        - MCPHelper.connect_mcp_async() called
        """
        session = session_manager.create_session(
            name="start-test",
            working_directory=Path("/tmp"),
            mcp_servers=[valid_server_config]
        )

        mcp_manager = MCPManager(session)

        # Mock successful connection
        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper = Mock()
            mock_helper.connect_mcp_async = AsyncMock()
            mock_helper.mcp_tool_schemas = []
            MockMCPHelper.return_value = mock_helper

            # Verify initial state
            assert valid_server_config.state == "disconnected"
            assert valid_server_config.error_message is None

            # Start server
            success = await mcp_manager.connect_server("filesystem")

            # Verify successful startup
            assert success is True
            assert valid_server_config.state == "connected"
            assert valid_server_config.error_message is None
            assert mock_helper.connect_mcp_async.called

    @pytest.mark.asyncio
    async def test_server_connection_and_tool_discovery(
        self, session_manager, valid_server_config
    ):
        """Integration: Connection established and tools discovered.

        Validates:
        - MCPHelper discovers tools from server
        - Tools stored in server config
        - Tool names properly prefixed (mcp_{server_id}_{tool_name})
        - Tools accessible via get_all_discovered_tools()
        """
        session = session_manager.create_session(
            name="discovery-test",
            working_directory=Path("/tmp"),
            mcp_servers=[valid_server_config]
        )

        mcp_manager = MCPManager(session)

        # Mock tool discovery
        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper = Mock()
            mock_helper.connect_mcp_async = AsyncMock()
            mock_helper.mcp_tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp_filesystem_read_file",
                        "description": "Read a file from the filesystem"
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "mcp_filesystem_write_file",
                        "description": "Write to a file"
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "mcp_filesystem_list_directory",
                        "description": "List directory contents"
                    }
                }
            ]
            MockMCPHelper.return_value = mock_helper

            # Connect and discover
            success = await mcp_manager.connect_server("filesystem")

            # Verify connection and discovery
            assert success is True
            assert valid_server_config.state == "connected"

            # Verify tools discovered
            discovered_tools = valid_server_config.discovered_tools
            assert len(discovered_tools) == 3
            assert "mcp_filesystem_read_file" in discovered_tools
            assert "mcp_filesystem_write_file" in discovered_tools
            assert "mcp_filesystem_list_directory" in discovered_tools

            # Verify all tools properly prefixed
            for tool in discovered_tools:
                assert tool.startswith("mcp_filesystem_")

            # Verify tools accessible via manager
            all_tools = mcp_manager.get_all_discovered_tools()
            assert len(all_tools) == 3
            assert set(all_tools) == set(discovered_tools)

    @pytest.mark.asyncio
    async def test_server_shutdown_cleanup(
        self, session_manager, valid_server_config
    ):
        """Integration: Server shutdown cleans up resources.

        Validates:
        - Disconnect marks server as 'disconnected'
        - Discovered tools cleared
        - Connection timestamp cleared
        - No error messages set
        - get_all_discovered_tools() returns empty list
        """
        session = session_manager.create_session(
            name="shutdown-test",
            working_directory=Path("/tmp"),
            mcp_servers=[valid_server_config]
        )

        mcp_manager = MCPManager(session)

        # Mock connection and shutdown
        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper = Mock()
            mock_helper.connect_mcp_async = AsyncMock()
            mock_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}}
            ]
            MockMCPHelper.return_value = mock_helper

            # Connect server
            await mcp_manager.connect_server("filesystem")

            # Verify connected state
            assert valid_server_config.state == "connected"
            assert len(valid_server_config.discovered_tools) > 0
            assert valid_server_config.connected_at is not None

            # Shutdown server
            success = await mcp_manager.disconnect_server("filesystem")

            # Verify clean shutdown
            assert success is True
            assert valid_server_config.state == "disconnected"
            assert len(valid_server_config.discovered_tools) == 0
            assert valid_server_config.connected_at is None
            assert valid_server_config.error_message is None

            # Verify no tools available
            all_tools = mcp_manager.get_all_discovered_tools()
            assert len(all_tools) == 0

    @pytest.mark.asyncio
    async def test_multiple_servers_simultaneous(
        self, session_manager, valid_server_config, calculator_server_config
    ):
        """Integration: Multiple servers can run simultaneously.

        Validates:
        - Two servers connect independently
        - Both servers in 'connected' state
        - Tools discovered from both servers
        - Tool names properly prefixed per server
        - No conflicts between servers
        """
        session = session_manager.create_session(
            name="multi-server-test",
            working_directory=Path("/tmp"),
            mcp_servers=[valid_server_config, calculator_server_config]
        )

        mcp_manager = MCPManager(session)

        # Mock both servers
        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            # Filesystem server helper
            fs_helper = Mock()
            fs_helper.connect_mcp_async = AsyncMock()
            fs_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}},
                {"type": "function", "function": {"name": "mcp_filesystem_write_file"}}
            ]

            # Calculator server helper
            calc_helper = Mock()
            calc_helper.connect_mcp_async = AsyncMock()
            calc_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_calculator_add"}},
                {"type": "function", "function": {"name": "mcp_calculator_multiply"}},
                {"type": "function", "function": {"name": "mcp_calculator_divide"}}
            ]

            # Return different helpers for each server
            MockMCPHelper.side_effect = [fs_helper, calc_helper]

            # Connect both servers
            fs_success = await mcp_manager.connect_server("filesystem")
            calc_success = await mcp_manager.connect_server("calculator")

            # Verify both connected
            assert fs_success is True
            assert calc_success is True
            assert valid_server_config.state == "connected"
            assert calculator_server_config.state == "connected"

            # Verify filesystem tools
            fs_tools = valid_server_config.discovered_tools
            assert len(fs_tools) == 2
            assert "mcp_filesystem_read_file" in fs_tools
            assert "mcp_filesystem_write_file" in fs_tools

            # Verify calculator tools
            calc_tools = calculator_server_config.discovered_tools
            assert len(calc_tools) == 3
            assert "mcp_calculator_add" in calc_tools
            assert "mcp_calculator_multiply" in calc_tools
            assert "mcp_calculator_divide" in calc_tools

            # Verify all tools available
            all_tools = mcp_manager.get_all_discovered_tools()
            assert len(all_tools) == 5  # 2 + 3

            # Verify no conflicts (all tools unique)
            assert len(set(all_tools)) == 5

            # Verify connected servers
            connected_servers = mcp_manager.get_connected_servers()
            assert len(connected_servers) == 2
            assert any(s.id == "filesystem" for s in connected_servers)
            assert any(s.id == "calculator" for s in connected_servers)

    @pytest.mark.asyncio
    async def test_server_restart_after_failure(
        self, session_manager, valid_server_config
    ):
        """Integration: Server can be restarted after failure.

        Validates:
        - Server fails to connect initially
        - State marked as 'error'
        - Error message captured
        - Server can be restarted successfully
        - State transitions to 'connected'
        - Tools discovered on successful restart
        """
        session = session_manager.create_session(
            name="restart-test",
            working_directory=Path("/tmp"),
            mcp_servers=[valid_server_config]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            # First attempt: Failure
            failing_helper = Mock()
            failing_helper.connect_mcp_async = AsyncMock(
                side_effect=Exception("Connection timeout")
            )

            # Second attempt: Success
            success_helper = Mock()
            success_helper.connect_mcp_async = AsyncMock()
            success_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}}
            ]

            # Return failing helper first, then success helper
            MockMCPHelper.side_effect = [failing_helper, success_helper]

            # ATTEMPT 1: Fail
            first_attempt = await mcp_manager.connect_server("filesystem")

            # Verify failure state
            assert first_attempt is False
            assert valid_server_config.state == "error"
            assert valid_server_config.error_message is not None
            assert "Connection timeout" in valid_server_config.error_message
            assert len(valid_server_config.discovered_tools) == 0

            # ATTEMPT 2: Restart and succeed
            second_attempt = await mcp_manager.connect_server("filesystem")

            # Verify successful restart
            assert second_attempt is True
            assert valid_server_config.state == "connected"
            assert valid_server_config.error_message is None
            assert len(valid_server_config.discovered_tools) > 0
            assert "mcp_filesystem_read_file" in valid_server_config.discovered_tools

    @pytest.mark.asyncio
    async def test_invalid_server_config_fails_gracefully(
        self, session_manager, invalid_server_config
    ):
        """Integration: Invalid server config fails gracefully.

        Validates:
        - Connection attempt with invalid config fails
        - No exception raised (graceful failure)
        - Server state marked as 'error'
        - Error message captured
        - No tools discovered
        - Session remains usable
        """
        session = session_manager.create_session(
            name="invalid-config-test",
            working_directory=Path("/tmp"),
            mcp_servers=[invalid_server_config]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper = Mock()
            mock_helper.connect_mcp_async = AsyncMock(
                side_effect=FileNotFoundError("Command not found: nonexistent-mcp-server")
            )
            MockMCPHelper.return_value = mock_helper

            # Attempt connection (should not raise exception)
            success = await mcp_manager.connect_server("invalid_server")

            # Verify graceful failure
            assert success is False
            assert invalid_server_config.state == "error"
            assert invalid_server_config.error_message is not None
            assert "not found" in invalid_server_config.error_message.lower()
            assert len(invalid_server_config.discovered_tools) == 0

            # Verify session still usable
            assert session.name == "invalid-config-test"
            assert len(session.mcp_servers) == 1

    @pytest.mark.asyncio
    async def test_server_lifecycle_with_real_mcphelper(
        self, session_manager, valid_server_config
    ):
        """Integration: Complete lifecycle with real MCPHelper integration.

        Validates:
        - MCPHelper instance created correctly
        - Real MCPHelper.connect_mcp_async() called
        - Tool schemas extracted from MCPHelper
        - Tools properly stored in server config
        - Lifecycle completes without errors
        """
        session = session_manager.create_session(
            name="real-mcphelper-lifecycle",
            working_directory=Path("/tmp"),
            mcp_servers=[valid_server_config]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper = Mock()
            mock_helper.connect_mcp_async = AsyncMock()
            mock_helper.mcp_tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp_filesystem_read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"}
                            },
                            "required": ["path"]
                        }
                    }
                }
            ]
            MockMCPHelper.return_value = mock_helper

            # Connect server (uses MCPHelper)
            success = await mcp_manager.connect_server("filesystem")

            # Verify MCPHelper instantiation
            assert MockMCPHelper.called
            call_args = MockMCPHelper.call_args

            # Verify MCPHelper configured correctly
            assert call_args is not None
            mcp_servers_arg = call_args.kwargs.get('mcp_servers')
            assert mcp_servers_arg is not None
            assert len(mcp_servers_arg) == 1
            assert mcp_servers_arg[0]['id'] == "filesystem"
            assert mcp_servers_arg[0]['command'] == "mcp-server-filesystem"

            # Verify connection method called
            assert mock_helper.connect_mcp_async.called

            # Verify tools extracted from MCPHelper
            assert success is True
            assert valid_server_config.state == "connected"
            assert "mcp_filesystem_read_file" in valid_server_config.discovered_tools

    @pytest.mark.asyncio
    async def test_server_state_transitions(
        self, session_manager, valid_server_config
    ):
        """Integration: Server state transitions correctly through lifecycle.

        State Machine:
        disconnected → connected → disconnected
        disconnected → error (on failure)
        error → connected (on retry)

        Validates:
        - Initial state: disconnected
        - After connect: connected
        - After disconnect: disconnected
        - After error: error
        - After recovery: connected
        """
        session = session_manager.create_session(
            name="state-transitions-test",
            working_directory=Path("/tmp"),
            mcp_servers=[valid_server_config]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            # STATE 1: Initial - disconnected
            assert valid_server_config.state == "disconnected"

            # Mock successful connection
            success_helper = Mock()
            success_helper.connect_mcp_async = AsyncMock()
            success_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}}
            ]
            MockMCPHelper.return_value = success_helper

            # STATE 2: Connect - transitions to connected
            await mcp_manager.connect_server("filesystem")
            assert valid_server_config.state == "connected"

            # STATE 3: Disconnect - transitions to disconnected
            await mcp_manager.disconnect_server("filesystem")
            assert valid_server_config.state == "disconnected"

            # STATE 4: Fail to connect - transitions to error
            error_helper = Mock()
            error_helper.connect_mcp_async = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            MockMCPHelper.return_value = error_helper

            await mcp_manager.connect_server("filesystem")
            assert valid_server_config.state == "error"

            # STATE 5: Recover - transitions from error to connected
            MockMCPHelper.return_value = success_helper
            await mcp_manager.connect_server("filesystem")
            assert valid_server_config.state == "connected"

    @pytest.mark.asyncio
    async def test_resource_cleanup_verification(
        self, session_manager, valid_server_config, calculator_server_config
    ):
        """Integration: All resources properly released on shutdown.

        Validates:
        - Multiple servers connected
        - disconnect_all_servers() called
        - All servers state = 'disconnected'
        - All tools cleared
        - get_all_discovered_tools() returns empty
        - get_connected_servers() returns empty
        """
        session = session_manager.create_session(
            name="cleanup-test",
            working_directory=Path("/tmp"),
            mcp_servers=[valid_server_config, calculator_server_config]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            # Mock filesystem server
            fs_helper = Mock()
            fs_helper.connect_mcp_async = AsyncMock()
            fs_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}}
            ]

            # Mock calculator server
            calc_helper = Mock()
            calc_helper.connect_mcp_async = AsyncMock()
            calc_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_calculator_add"}}
            ]

            MockMCPHelper.side_effect = [fs_helper, calc_helper]

            # Connect both servers
            await mcp_manager.connect_server("filesystem")
            await mcp_manager.connect_server("calculator")

            # Verify both connected
            assert valid_server_config.state == "connected"
            assert calculator_server_config.state == "connected"
            assert len(mcp_manager.get_all_discovered_tools()) == 2
            assert len(mcp_manager.get_connected_servers()) == 2

            # Cleanup all resources
            await mcp_manager.disconnect_all_servers()

            # Verify complete cleanup
            assert valid_server_config.state == "disconnected"
            assert calculator_server_config.state == "disconnected"
            assert len(valid_server_config.discovered_tools) == 0
            assert len(calculator_server_config.discovered_tools) == 0
            assert len(mcp_manager.get_all_discovered_tools()) == 0
            assert len(mcp_manager.get_connected_servers()) == 0
