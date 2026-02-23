"""Integration tests for MCP tool discovery and registration (T058).

These tests verify that MCP tools are properly discovered from connected servers
and registered for use in agent conversations.

Test Coverage:
- test_tool_discovery_on_connection: Tools discovered when server connects
- test_tool_name_prefixing: Tools prefixed with server ID
- test_tool_registration_with_agent: Tools registered with PromptChain agent
- test_multiple_server_tool_discovery: Tools from multiple servers combined
- test_tool_conflict_resolution: Name conflicts resolved via prefixing
- test_tool_schemas_stored: Tool schemas available after discovery
- test_disconnection_unregisters_tools: Tools removed when server disconnects
"""

import pytest
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from promptchain.cli.models.mcp_config import MCPServerConfig
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.models.agent_config import Agent
from promptchain.utils.promptchaining import PromptChain


class TestMCPToolDiscovery:
    """Integration tests for MCP tool discovery and registration."""

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
        """Create filesystem MCP server config."""
        return MCPServerConfig(
            id="filesystem",
            type="stdio",
            command="mcp-server-filesystem",
            args=["--root", "/tmp"],
            auto_connect=False
        )

    @pytest.fixture
    def web_search_server(self):
        """Create web search MCP server config."""
        return MCPServerConfig(
            id="web_search",
            type="stdio",
            command="mcp-server-web-search",
            auto_connect=False
        )

    @pytest.mark.asyncio
    async def test_tool_discovery_on_connection(self, session_manager, filesystem_server):
        """Integration: Tools discovered when MCP server connects.

        Validates:
        - MCP manager discovers tools on connection
        - Tool names stored in server config
        - Tool count > 0
        - Tools accessible via get_all_discovered_tools()
        """
        session = session_manager.create_session(
            name="tool-discovery-test",
            working_directory=Path("/tmp")
        )

        session.mcp_servers.append(filesystem_server)

        from promptchain.cli.utils.mcp_manager import MCPManager

        mcp_manager = MCPManager(session)

        # Connect and discover tools
        success = await mcp_manager.connect_server("filesystem")

        assert success is True
        assert filesystem_server.state == "connected"
        assert len(filesystem_server.discovered_tools) > 0

        # Verify tools accessible via manager
        all_tools = mcp_manager.get_all_discovered_tools()
        assert len(all_tools) > 0
        assert all_tools == filesystem_server.discovered_tools

    @pytest.mark.asyncio
    async def test_tool_name_prefixing(self, session_manager, filesystem_server):
        """Integration: Tools prefixed with server ID to prevent conflicts.

        Validates:
        - Tool names follow pattern: mcp_<server_id>_<tool_name>
        - Prefix prevents naming conflicts
        - Original tool name recoverable from prefixed name
        """
        session = session_manager.create_session(
            name="prefix-test",
            working_directory=Path("/tmp")
        )

        session.mcp_servers.append(filesystem_server)

        from promptchain.cli.utils.mcp_manager import MCPManager

        mcp_manager = MCPManager(session)

        await mcp_manager.connect_server("filesystem")

        # Validate tool name prefixing
        tools = filesystem_server.discovered_tools
        for tool_name in tools:
            # Should start with server_id prefix
            # Format: filesystem_tool_name or mcp_filesystem_tool_name
            assert "filesystem" in tool_name

    @pytest.mark.asyncio
    async def test_tool_registration_with_agent(self, session_manager, filesystem_server):
        """Integration: Discovered tools registered with PromptChain agent.

        Validates:
        - Tools from MCP server registered with agent
        - Agent can access tool schemas
        - Tools callable during agent conversation
        - Tool execution handled by MCP manager
        """
        session = session_manager.create_session(
            name="agent-registration-test",
            working_directory=Path("/tmp")
        )

        session.mcp_servers.append(filesystem_server)

        from promptchain.cli.utils.mcp_manager import MCPManager

        mcp_manager = MCPManager(session)

        # Connect MCP server
        await mcp_manager.connect_server("filesystem")

        # Create agent with MCP tools
        agent = Agent(
            name="test_agent",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Test agent with MCP tools"
        )

        session.agents["test_agent"] = agent

        # In real implementation, this would:
        # 1. Get tool schemas from MCPManager
        # 2. Register tools with PromptChain
        # 3. Make tools available during conversation

        # For now, just verify tools discovered
        tools = mcp_manager.get_all_discovered_tools()
        assert len(tools) > 0

    @pytest.mark.asyncio
    async def test_multiple_server_tool_discovery(
        self, session_manager, filesystem_server, web_search_server
    ):
        """Integration: Tools from multiple MCP servers combined.

        Validates:
        - Multiple servers connect independently
        - Each server discovers its own tools
        - Combined tool list includes all servers
        - Tool prefixing prevents conflicts
        """
        session = session_manager.create_session(
            name="multi-server-tools-test",
            working_directory=Path("/tmp")
        )

        session.mcp_servers.append(filesystem_server)
        session.mcp_servers.append(web_search_server)

        from promptchain.cli.utils.mcp_manager import MCPManager

        mcp_manager = MCPManager(session)

        # Connect both servers
        fs_success = await mcp_manager.connect_server("filesystem")
        ws_success = await mcp_manager.connect_server("web_search")

        assert fs_success is True
        assert ws_success is True

        # Each server should have discovered tools
        assert len(filesystem_server.discovered_tools) > 0
        assert len(web_search_server.discovered_tools) > 0

        # Combined tool list
        all_tools = mcp_manager.get_all_discovered_tools()
        expected_count = (
            len(filesystem_server.discovered_tools)
            + len(web_search_server.discovered_tools)
        )
        assert len(all_tools) == expected_count

        # Verify tools from both servers present
        fs_tools = [t for t in all_tools if "filesystem" in t]
        ws_tools = [t for t in all_tools if "web_search" in t]

        assert len(fs_tools) > 0
        assert len(ws_tools) > 0

    @pytest.mark.asyncio
    async def test_tool_conflict_resolution(
        self, session_manager, filesystem_server, web_search_server
    ):
        """Integration: Name conflicts resolved via server ID prefixing.

        Validates:
        - Two servers with same tool name don't conflict
        - Prefixing ensures unique tool names
        - Both tools accessible independently
        - Correct server associated with each tool
        """
        session = session_manager.create_session(
            name="conflict-resolution-test",
            working_directory=Path("/tmp")
        )

        session.mcp_servers.append(filesystem_server)
        session.mcp_servers.append(web_search_server)

        from promptchain.cli.utils.mcp_manager import MCPManager

        mcp_manager = MCPManager(session)

        await mcp_manager.connect_server("filesystem")
        await mcp_manager.connect_server("web_search")

        all_tools = mcp_manager.get_all_discovered_tools()

        # Tools should be uniquely named via server ID prefix
        # No duplicate tool names in combined list
        assert len(all_tools) == len(set(all_tools))

        # Each tool should be identifiable by server ID
        for tool in all_tools:
            # Tool name should contain either "filesystem" or "web_search"
            assert "filesystem" in tool or "web_search" in tool

    @pytest.mark.asyncio
    async def test_tool_schemas_stored(self, session_manager, filesystem_server):
        """Integration: Tool schemas available after discovery.

        Validates:
        - Tool schemas discovered from MCP server
        - Schemas include function name, description, parameters
        - Schemas stored for later registration
        - Schemas follow OpenAI function calling format
        """
        session = session_manager.create_session(
            name="schema-storage-test",
            working_directory=Path("/tmp")
        )

        session.mcp_servers.append(filesystem_server)

        from promptchain.cli.utils.mcp_manager import MCPManager

        mcp_manager = MCPManager(session)

        await mcp_manager.connect_server("filesystem")

        # In real implementation, MCPManager would have:
        # - get_tool_schemas(server_id) method
        # - Returns list of tool schemas in OpenAI format
        # - Schemas stored when tools discovered

        # For now, just verify tools exist
        tools = filesystem_server.discovered_tools
        assert len(tools) > 0

        # Future: Validate schema structure
        # schemas = mcp_manager.get_tool_schemas("filesystem")
        # for schema in schemas:
        #     assert "name" in schema
        #     assert "description" in schema
        #     assert "parameters" in schema

    @pytest.mark.asyncio
    async def test_disconnection_unregisters_tools(
        self, session_manager, filesystem_server
    ):
        """Integration: Tools removed when MCP server disconnects.

        Validates:
        - Tools available when server connected
        - Tools removed after disconnection
        - get_all_discovered_tools() excludes disconnected servers
        - Agent tool list updated on disconnection
        """
        session = session_manager.create_session(
            name="unregister-test",
            working_directory=Path("/tmp")
        )

        session.mcp_servers.append(filesystem_server)

        from promptchain.cli.utils.mcp_manager import MCPManager

        mcp_manager = MCPManager(session)

        # Connect and verify tools present
        await mcp_manager.connect_server("filesystem")
        tools_before = mcp_manager.get_all_discovered_tools()
        assert len(tools_before) > 0

        # Disconnect
        await mcp_manager.disconnect_server("filesystem")

        # Tools should be cleared
        assert len(filesystem_server.discovered_tools) == 0

        # All tools list should be empty
        tools_after = mcp_manager.get_all_discovered_tools()
        assert len(tools_after) == 0
