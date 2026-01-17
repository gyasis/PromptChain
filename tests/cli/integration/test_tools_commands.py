"""Integration tests for /tools commands (T065-T067).

These tests verify that users can manage MCP tools through slash commands,
viewing available tools and registering/unregistering them with agents.

Test Coverage:
- test_tools_list_no_servers: List tools when no MCP servers connected
- test_tools_list_single_server: List tools from one connected server
- test_tools_list_multiple_servers: List tools from multiple servers
- test_tools_list_shows_registration_status: Show which agents have which tools
- test_tools_add_registers_with_agent: Add tools to specific agent
- test_tools_remove_unregisters_from_agent: Remove tools from specific agent
- test_tools_commands_persist: Tool registrations persist across save/load
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from promptchain.cli.models.mcp_config import MCPServerConfig
from promptchain.cli.models.agent_config import Agent
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.command_handler import CommandHandler
from promptchain.cli.utils.mcp_manager import MCPManager


class TestToolsCommands:
    """Integration tests for /tools commands."""

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
    def command_handler(self, session_manager):
        """Create CommandHandler for testing."""
        return CommandHandler(session_manager)

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

    @pytest.fixture
    def calculator_server(self):
        """Create MCP server config for calculator."""
        return MCPServerConfig(
            id="calculator",
            type="stdio",
            command="mcp-server-calculator",
            auto_connect=False
        )

    @pytest.fixture
    def test_agent(self):
        """Create test agent."""
        return Agent(
            name="test_agent",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Test agent for tools"
        )

    @pytest.mark.asyncio
    async def test_tools_list_no_servers(
        self, session_manager, command_handler
    ):
        """Integration: /tools list when no MCP servers connected.

        Validates:
        - Returns success with empty list
        - User-friendly message explaining no servers
        - Suggests connecting servers
        """
        # Create session with no servers
        session = session_manager.create_session(
            name="no-servers-test",
            working_directory=Path("/tmp"),
            mcp_servers=[]
        )

        # Run /tools list command
        result = await command_handler.handle_tools_list(session)

        # Verify empty result
        assert result.success
        assert "no mcp servers connected" in result.message.lower()
        assert result.data["tools"] == []
        assert result.data["server_count"] == 0

    @pytest.mark.asyncio
    async def test_tools_list_single_server(
        self, session_manager, command_handler, filesystem_server
    ):
        """Integration: /tools list with one connected server.

        Validates:
        - Lists all tools from connected server
        - Shows server name for each tool
        - Tool count is accurate
        - Tools formatted with mcp_{server_id}_{tool_name}
        """
        # Create session with server
        session = session_manager.create_session(
            name="single-server-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )

        # Connect server and discover tools
        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}},
                {"type": "function", "function": {"name": "mcp_filesystem_write_file"}},
                {"type": "function", "function": {"name": "mcp_filesystem_list_directory"}}
            ]
            MockMCPHelper.return_value = mock_helper_instance

            await mcp_manager.connect_server("filesystem")

            # Run /tools list command
            result = await command_handler.handle_tools_list(session)

            # Verify tools listed
            assert result.success
            assert result.data["server_count"] == 1
            assert len(result.data["tools"]) == 3

            # Verify tool details
            tool_names = [t["name"] for t in result.data["tools"]]
            assert "mcp_filesystem_read_file" in tool_names
            assert "mcp_filesystem_write_file" in tool_names
            assert "mcp_filesystem_list_directory" in tool_names

            # Verify server association
            for tool in result.data["tools"]:
                assert tool["server_id"] == "filesystem"

    @pytest.mark.asyncio
    async def test_tools_list_multiple_servers(
        self, session_manager, command_handler, filesystem_server, calculator_server
    ):
        """Integration: /tools list with multiple connected servers.

        Validates:
        - Lists tools from all connected servers
        - Tools grouped or marked by server
        - Total count accurate
        - Different server IDs distinguished
        """
        # Create session with multiple servers
        session = session_manager.create_session(
            name="multi-server-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server, calculator_server]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            # First call for filesystem
            filesystem_helper = Mock()
            filesystem_helper.connect_mcp_async = AsyncMock()
            filesystem_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}},
                {"type": "function", "function": {"name": "mcp_filesystem_write_file"}}
            ]

            # Second call for calculator
            calculator_helper = Mock()
            calculator_helper.connect_mcp_async = AsyncMock()
            calculator_helper.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_calculator_add"}},
                {"type": "function", "function": {"name": "mcp_calculator_multiply"}},
                {"type": "function", "function": {"name": "mcp_calculator_subtract"}}
            ]

            MockMCPHelper.side_effect = [filesystem_helper, calculator_helper]

            # Connect both servers
            await mcp_manager.connect_server("filesystem")
            await mcp_manager.connect_server("calculator")

            # Run /tools list command
            result = await command_handler.handle_tools_list(session)

            # Verify tools from both servers
            assert result.success
            assert result.data["server_count"] == 2
            assert len(result.data["tools"]) == 5

            # Verify server grouping
            filesystem_tools = [t for t in result.data["tools"] if t["server_id"] == "filesystem"]
            calculator_tools = [t for t in result.data["tools"] if t["server_id"] == "calculator"]

            assert len(filesystem_tools) == 2
            assert len(calculator_tools) == 3

    @pytest.mark.asyncio
    async def test_tools_list_shows_registration_status(
        self, session_manager, command_handler, filesystem_server, test_agent
    ):
        """Integration: /tools list shows which agents have which tools.

        Validates:
        - Each tool shows list of agents using it
        - Unregistered tools show empty agent list
        - Registered tools show agent names
        - Status updates when tools are registered/unregistered
        """
        # Create session with server and agent
        session = session_manager.create_session(
            name="registration-status-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        session.agents["test_agent"] = test_agent

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}},
                {"type": "function", "function": {"name": "mcp_filesystem_write_file"}}
            ]
            MockMCPHelper.return_value = mock_helper_instance

            await mcp_manager.connect_server("filesystem")

            # List tools before registration
            result_before = await command_handler.handle_tools_list(session)
            assert result_before.success

            # Verify no agents registered initially
            for tool in result_before.data["tools"]:
                assert tool["registered_agents"] == []

            # Register tools with agent
            await mcp_manager.register_tools_with_agent(
                server_id="filesystem",
                agent_name="test_agent"
            )

            # List tools after registration
            result_after = await command_handler.handle_tools_list(session)
            assert result_after.success

            # Verify agent now shows as registered
            for tool in result_after.data["tools"]:
                assert "test_agent" in tool["registered_agents"]

    @pytest.mark.asyncio
    async def test_tools_list_formatting(
        self, session_manager, command_handler, filesystem_server
    ):
        """Integration: /tools list formats output for user readability.

        Validates:
        - User-friendly message formatting
        - Tool names clearly displayed
        - Server names visible
        - Registration status clear
        - Table or list format readable
        """
        session = session_manager.create_session(
            name="formatting-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}},
            ]
            MockMCPHelper.return_value = mock_helper_instance

            await mcp_manager.connect_server("filesystem")

            # Run command
            result = await command_handler.handle_tools_list(session)

            # Verify formatting
            assert result.success
            assert "Available MCP Tools" in result.message or "tools" in result.message.lower()
            assert "filesystem" in result.message
            assert "mcp_filesystem_read_file" in result.message

    @pytest.mark.asyncio
    async def test_tools_list_disconnected_servers(
        self, session_manager, command_handler, filesystem_server
    ):
        """Integration: /tools list handles disconnected servers.

        Validates:
        - Disconnected servers don't show tools
        - Only connected servers listed
        - User-friendly indication of disconnected state
        """
        session = session_manager.create_session(
            name="disconnected-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )

        # Don't connect server

        # Run /tools list command
        result = await command_handler.handle_tools_list(session)

        # Verify no tools from disconnected server
        assert result.success
        assert result.data["server_count"] == 0
        assert len(result.data["tools"]) == 0
        assert "no mcp servers connected" in result.message.lower()

    @pytest.mark.asyncio
    async def test_tools_add_registers_with_current_agent(
        self, session_manager, command_handler, filesystem_server, test_agent
    ):
        """Integration: /tools add registers MCP tools with current agent (T066).

        Validates:
        - Tools from server registered with current agent
        - Agent's tools list updated with server tools
        - Success message confirms registration
        - Requires server to be connected
        """
        # Create session with server and agent
        session = session_manager.create_session(
            name="tools-add-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        session.agents["test_agent"] = test_agent
        session.current_agent = "test_agent"

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}},
                {"type": "function", "function": {"name": "mcp_filesystem_write_file"}}
            ]
            MockMCPHelper.return_value = mock_helper_instance

            # Connect server
            await mcp_manager.connect_server("filesystem")

            # Verify tools not registered initially
            assert len(test_agent.tools) == 0

            # Run /tools add command
            result = await command_handler.handle_tools_add(session, "filesystem")

            # Verify tools registered
            assert result.success
            assert "mcp_filesystem_read_file" in test_agent.tools
            assert "mcp_filesystem_write_file" in test_agent.tools
            assert len(test_agent.tools) == 2
            assert "registered" in result.message.lower()
            assert "filesystem" in result.message.lower()

    @pytest.mark.asyncio
    async def test_tools_add_requires_connected_server(
        self, session_manager, command_handler, filesystem_server, test_agent
    ):
        """Integration: /tools add fails if server not connected (T066).

        Validates:
        - Cannot register tools from disconnected server
        - Error message explains server not connected
        - Agent tools unchanged
        """
        session = session_manager.create_session(
            name="disconnected-add-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        session.agents["test_agent"] = test_agent
        session.current_agent = "test_agent"

        # Don't connect server

        # Try to add tools from disconnected server
        result = await command_handler.handle_tools_add(session, "filesystem")

        # Verify failure
        assert not result.success
        assert "not connected" in result.message.lower()
        assert len(test_agent.tools) == 0

    @pytest.mark.asyncio
    async def test_tools_add_requires_current_agent(
        self, session_manager, command_handler, filesystem_server
    ):
        """Integration: /tools add requires current agent to be set (T066).

        Validates:
        - Cannot add tools without current agent
        - Error message explains no current agent
        """
        session = session_manager.create_session(
            name="no-agent-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        # No current_agent set

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}}
            ]
            MockMCPHelper.return_value = mock_helper_instance

            await mcp_manager.connect_server("filesystem")

            # Try to add tools without current agent
            result = await command_handler.handle_tools_add(session, "filesystem")

            # Verify failure
            assert not result.success
            assert "no current agent" in result.message.lower() or "no agent" in result.message.lower()

    @pytest.mark.asyncio
    async def test_tools_add_nonexistent_server(
        self, session_manager, command_handler, test_agent
    ):
        """Integration: /tools add fails for nonexistent server (T066).

        Validates:
        - Error when server ID doesn't exist
        - Clear error message
        - Agent tools unchanged
        """
        session = session_manager.create_session(
            name="nonexistent-server-test",
            working_directory=Path("/tmp"),
            mcp_servers=[]
        )
        session.agents["test_agent"] = test_agent
        session.current_agent = "test_agent"

        # Try to add tools from nonexistent server
        result = await command_handler.handle_tools_add(session, "nonexistent")

        # Verify failure
        assert not result.success
        assert "not found" in result.message.lower()
        assert len(test_agent.tools) == 0

    @pytest.mark.asyncio
    async def test_tools_add_idempotent(
        self, session_manager, command_handler, filesystem_server, test_agent
    ):
        """Integration: /tools add is idempotent (T066).

        Validates:
        - Running /tools add twice doesn't duplicate tools
        - Second call still succeeds
        - Tool list remains consistent
        """
        session = session_manager.create_session(
            name="idempotent-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        session.agents["test_agent"] = test_agent
        session.current_agent = "test_agent"

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}}
            ]
            MockMCPHelper.return_value = mock_helper_instance

            await mcp_manager.connect_server("filesystem")

            # First add
            result1 = await command_handler.handle_tools_add(session, "filesystem")
            assert result1.success
            assert len(test_agent.tools) == 1

            # Second add (should be idempotent)
            result2 = await command_handler.handle_tools_add(session, "filesystem")
            assert result2.success
            assert len(test_agent.tools) == 1  # No duplicates
            assert "mcp_filesystem_read_file" in test_agent.tools

    @pytest.mark.asyncio
    async def test_tools_remove_unregisters_from_current_agent(
        self, session_manager, command_handler, filesystem_server, test_agent
    ):
        """Integration: /tools remove unregisters MCP tools from current agent (T067).

        Validates:
        - Tools from server unregistered from current agent
        - Agent's tools list updated (tools removed)
        - Success message confirms unregistration
        - Only affects specified server's tools
        """
        # Create session with server and agent
        session = session_manager.create_session(
            name="tools-remove-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        session.agents["test_agent"] = test_agent
        session.current_agent = "test_agent"

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}},
                {"type": "function", "function": {"name": "mcp_filesystem_write_file"}}
            ]
            MockMCPHelper.return_value = mock_helper_instance

            # Connect server and register tools
            await mcp_manager.connect_server("filesystem")
            await mcp_manager.register_tools_with_agent(
                server_id="filesystem",
                agent_name="test_agent"
            )

            # Verify tools registered initially
            assert len(test_agent.tools) == 2
            assert "mcp_filesystem_read_file" in test_agent.tools
            assert "mcp_filesystem_write_file" in test_agent.tools

            # Run /tools remove command
            result = await command_handler.handle_tools_remove(session, "filesystem")

            # Verify tools unregistered
            assert result.success
            assert "mcp_filesystem_read_file" not in test_agent.tools
            assert "mcp_filesystem_write_file" not in test_agent.tools
            assert len(test_agent.tools) == 0
            assert "unregistered" in result.message.lower() or "removed" in result.message.lower()
            assert "filesystem" in result.message.lower()

    @pytest.mark.asyncio
    async def test_tools_remove_requires_current_agent(
        self, session_manager, command_handler, filesystem_server
    ):
        """Integration: /tools remove requires current agent to be set (T067).

        Validates:
        - Cannot remove tools without current agent
        - Error message explains no current agent
        """
        session = session_manager.create_session(
            name="no-agent-remove-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        # No current_agent set

        # Try to remove tools without current agent
        result = await command_handler.handle_tools_remove(session, "filesystem")

        # Verify failure
        assert not result.success
        assert "no current agent" in result.message.lower() or "no agent" in result.message.lower()

    @pytest.mark.asyncio
    async def test_tools_remove_nonexistent_server(
        self, session_manager, command_handler, test_agent
    ):
        """Integration: /tools remove fails for nonexistent server (T067).

        Validates:
        - Error when server ID doesn't exist
        - Clear error message
        - Agent tools unchanged
        """
        session = session_manager.create_session(
            name="nonexistent-server-remove-test",
            working_directory=Path("/tmp"),
            mcp_servers=[]
        )
        session.agents["test_agent"] = test_agent
        session.current_agent = "test_agent"

        # Add some tools to agent first
        test_agent.tools.append("some_existing_tool")

        # Try to remove tools from nonexistent server
        result = await command_handler.handle_tools_remove(session, "nonexistent")

        # Verify failure
        assert not result.success
        assert "not found" in result.message.lower()
        # Agent tools unchanged
        assert "some_existing_tool" in test_agent.tools
        assert len(test_agent.tools) == 1

    @pytest.mark.asyncio
    async def test_tools_remove_idempotent(
        self, session_manager, command_handler, filesystem_server, test_agent
    ):
        """Integration: /tools remove is idempotent (T067).

        Validates:
        - Running /tools remove twice doesn't cause errors
        - Second call still succeeds (or returns informative message)
        - Tool list remains consistent
        """
        session = session_manager.create_session(
            name="idempotent-remove-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        session.agents["test_agent"] = test_agent
        session.current_agent = "test_agent"

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}}
            ]
            MockMCPHelper.return_value = mock_helper_instance

            # Connect and register tools
            await mcp_manager.connect_server("filesystem")
            await mcp_manager.register_tools_with_agent(
                server_id="filesystem",
                agent_name="test_agent"
            )

            # Verify tool registered
            assert len(test_agent.tools) == 1

            # First remove
            result1 = await command_handler.handle_tools_remove(session, "filesystem")
            assert result1.success
            assert len(test_agent.tools) == 0

            # Second remove (should be idempotent - no error even though tools already removed)
            result2 = await command_handler.handle_tools_remove(session, "filesystem")
            assert result2.success
            assert len(test_agent.tools) == 0  # Still 0

    @pytest.mark.asyncio
    async def test_tools_remove_only_affects_specified_server(
        self, session_manager, command_handler, filesystem_server, test_agent
    ):
        """Integration: /tools remove only removes tools from specified server (T067).

        Validates:
        - Removing filesystem tools doesn't affect calculator tools
        - Other agents' tools unchanged
        - Selective tool removal
        """
        # Create calculator server
        calculator_server = MCPServerConfig(
            id="calculator",
            type="stdio",
            command="mcp-server-calculator",
            auto_connect=False
        )

        session = session_manager.create_session(
            name="selective-remove-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server, calculator_server]
        )
        session.agents["test_agent"] = test_agent
        session.current_agent = "test_agent"

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

            MockMCPHelper.side_effect = [filesystem_helper, calculator_helper]

            # Connect both servers and register tools
            await mcp_manager.connect_server("filesystem")
            await mcp_manager.connect_server("calculator")
            await mcp_manager.register_tools_with_agent("filesystem", "test_agent")
            await mcp_manager.register_tools_with_agent("calculator", "test_agent")

            # Verify all tools registered
            assert len(test_agent.tools) == 3
            assert "mcp_filesystem_read_file" in test_agent.tools
            assert "mcp_calculator_add" in test_agent.tools
            assert "mcp_calculator_multiply" in test_agent.tools

            # Remove only filesystem tools
            result = await command_handler.handle_tools_remove(session, "filesystem")

            # Verify only filesystem tools removed
            assert result.success
            assert "mcp_filesystem_read_file" not in test_agent.tools
            assert "mcp_calculator_add" in test_agent.tools
            assert "mcp_calculator_multiply" in test_agent.tools
            assert len(test_agent.tools) == 2
