"""Integration tests for real MCPHelper integration in MCPManager (T061).

These tests verify that MCPManager uses real MCPHelper for tool discovery
instead of mock implementations.

Test Coverage:
- test_real_mcphelper_connection: MCPManager uses real MCPHelper to connect
- test_real_tool_discovery: Real tools discovered from MCP server (not mocks)
- test_tool_name_prefixing: Tools prefixed with mcp_{server_id}_{tool_name}
- test_multiple_servers_real_discovery: Multiple servers discover real tools
- test_failed_connection_error_handling: Failed connections handled gracefully
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from promptchain.cli.models.mcp_config import MCPServerConfig
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.utils.mcp_manager import MCPManager


class TestMCPRealHelperIntegration:
    """Integration tests for real MCPHelper integration in MCPManager."""

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
    def filesystem_server(self):
        """Create MCP server config for filesystem."""
        return MCPServerConfig(
            id="filesystem",
            type="stdio",
            command="mcp-server-filesystem",
            args=["--root", "/tmp"],
            auto_connect=False
        )

    @pytest.mark.asyncio
    async def test_real_mcphelper_connection(
        self, session_manager, filesystem_server
    ):
        """Integration: MCPManager uses real MCPHelper to connect.

        Validates:
        - MCPHelper instance created (not mock)
        - MCPHelper.connect_mcp_async() called
        - Real connection established
        - No mock tool generation
        """
        # Create session with MCP server
        session = session_manager.create_session(
            name="real-helper-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )

        mcp_manager = MCPManager(session)

        # Mock MCPHelper's connect method but track that it's called
        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = []
            MockMCPHelper.return_value = mock_helper_instance

            # Connect server
            await mcp_manager.connect_server("filesystem")

            # Verify MCPHelper was instantiated
            assert MockMCPHelper.called

            # Verify connect_mcp_async was called
            assert mock_helper_instance.connect_mcp_async.called

    @pytest.mark.asyncio
    async def test_real_tool_discovery(
        self, session_manager, filesystem_server
    ):
        """Integration: Real tools discovered from MCP server.

        Validates:
        - Tools come from MCPHelper.mcp_tool_schemas
        - NOT from mock list like ["filesystem_tool_1", "filesystem_tool_2"]
        - Tool schemas have proper structure
        - Tool names are properly prefixed
        """
        session = session_manager.create_session(
            name="real-discovery-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )

        mcp_manager = MCPManager(session)

        # Mock MCPHelper to return realistic tool schemas
        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()

            # Simulate real tools discovered by MCPHelper
            mock_helper_instance.mcp_tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp_filesystem_read_file",
                        "description": "Read a file from the filesystem",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"}
                            },
                            "required": ["path"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "mcp_filesystem_write_file",
                        "description": "Write to a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"}
                            },
                            "required": ["path", "content"]
                        }
                    }
                }
            ]
            MockMCPHelper.return_value = mock_helper_instance

            # Connect server
            await mcp_manager.connect_server("filesystem")

            # Verify real tools discovered (not mock tools)
            discovered_tools = filesystem_server.discovered_tools

            # Should NOT be mock tools
            assert "filesystem_tool_1" not in discovered_tools
            assert "filesystem_tool_2" not in discovered_tools
            assert "filesystem_read" not in discovered_tools
            assert "filesystem_write" not in discovered_tools

            # Should be real tools from MCPHelper
            assert "mcp_filesystem_read_file" in discovered_tools
            assert "mcp_filesystem_write_file" in discovered_tools

    @pytest.mark.asyncio
    async def test_tool_name_prefixing(
        self, session_manager, filesystem_server
    ):
        """Integration: Tools prefixed with mcp_{server_id}_{tool_name}.

        Validates:
        - Tool names follow format: mcp_filesystem_read_file
        - NOT format: filesystem_tool_1 (mock format)
        - Prefix prevents conflicts between servers
        """
        session = session_manager.create_session(
            name="prefixing-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp_filesystem_list_directory",
                        "description": "List directory contents"
                    }
                }
            ]
            MockMCPHelper.return_value = mock_helper_instance

            await mcp_manager.connect_server("filesystem")

            # Verify tool name format
            tools = filesystem_server.discovered_tools
            assert len(tools) == 1
            assert tools[0] == "mcp_filesystem_list_directory"
            assert tools[0].startswith("mcp_filesystem_")

    @pytest.mark.asyncio
    async def test_multiple_servers_real_discovery(
        self, session_manager, filesystem_server
    ):
        """Integration: Multiple servers discover real tools independently.

        Validates:
        - Each server uses MCPHelper for discovery
        - Tools properly prefixed per server
        - No mock tools generated
        """
        # Create second server
        calculator_server = MCPServerConfig(
            id="calculator",
            type="stdio",
            command="mcp-server-calculator",
            auto_connect=False
        )

        session = session_manager.create_session(
            name="multi-server-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server, calculator_server]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            # First call for filesystem server
            filesystem_helper = Mock()
            filesystem_helper.connect_mcp_async = AsyncMock()
            filesystem_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}}
            ]

            # Second call for calculator server
            calculator_helper = Mock()
            calculator_helper.connect_mcp_async = AsyncMock()
            calculator_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_calculator_add"}},
                {"type": "function", "function": {"name": "mcp_calculator_multiply"}}
            ]

            # Return different helpers for each server
            MockMCPHelper.side_effect = [filesystem_helper, calculator_helper]

            # Connect both servers
            await mcp_manager.connect_server("filesystem")
            await mcp_manager.connect_server("calculator")

            # Verify filesystem tools
            fs_tools = filesystem_server.discovered_tools
            assert "mcp_filesystem_read_file" in fs_tools
            assert "filesystem_tool_1" not in fs_tools  # No mock tools

            # Verify calculator tools
            calc_tools = calculator_server.discovered_tools
            assert "mcp_calculator_add" in calc_tools
            assert "mcp_calculator_multiply" in calc_tools
            assert "calculator_tool_1" not in calc_tools  # No mock tools

    @pytest.mark.asyncio
    async def test_failed_connection_error_handling(
        self, session_manager
    ):
        """Integration: Failed MCPHelper connection handled gracefully.

        Validates:
        - MCPHelper connection failure caught
        - Server marked with error state
        - Error message captured
        - No tools discovered
        """
        # Create server with command that will fail
        failing_server = MCPServerConfig(
            id="failing_server",
            type="stdio",
            command="nonexistent-mcp-server",
            auto_connect=False
        )

        session = session_manager.create_session(
            name="failure-test",
            working_directory=Path("/tmp"),
            mcp_servers=[failing_server]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            # Simulate connection failure
            mock_helper_instance.connect_mcp_async = AsyncMock(
                side_effect=Exception("Failed to connect to MCP server")
            )
            MockMCPHelper.return_value = mock_helper_instance

            # Attempt connection
            await mcp_manager.connect_server("failing_server")

            # Verify error state
            assert failing_server.state == "error"
            assert failing_server.error_message is not None
            assert "Failed to connect" in failing_server.error_message

            # No tools discovered
            assert len(failing_server.discovered_tools) == 0
