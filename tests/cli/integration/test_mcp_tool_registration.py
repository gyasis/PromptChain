"""Integration tests for MCP tool registration with agents (T063).

These tests verify that MCP tools discovered from servers can be registered
with agents, making them available for use in conversations.

Test Coverage:
- test_register_mcp_tools_with_agent: Tools registered with specific agent after server connection
- test_tools_persist_across_session_save_load: Registered tools persist through session lifecycle
- test_multiple_agents_different_tools: Different agents can have different MCP tools
- test_tool_registration_preserves_agent_properties: Registration doesn't affect other agent fields
- test_auto_register_tools_on_server_connect: Optional auto-registration when server connects
- test_unregister_tools_on_server_disconnect: Tools removed when server disconnects
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from promptchain.cli.models.mcp_config import MCPServerConfig
from promptchain.cli.models.agent_config import Agent
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.utils.mcp_manager import MCPManager


class TestMCPToolRegistration:
    """Integration tests for MCP tool registration with agents."""

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

    @pytest.fixture
    def test_agent(self):
        """Create test agent for tool registration."""
        return Agent(
            name="test_agent",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Test agent for tool registration"
        )

    @pytest.mark.asyncio
    async def test_register_mcp_tools_with_agent(
        self, session_manager, filesystem_server, test_agent
    ):
        """Integration: Register MCP tools with specific agent after server connection.

        Validates:
        - Server connects and discovers tools
        - Tools can be registered with specific agent
        - Agent's tools list contains MCP tool names
        - Tool names follow mcp_{server_id}_{tool_name} format
        """
        # Create session with server and agent
        session = session_manager.create_session(
            name="tool-registration-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        session.agents["test_agent"] = test_agent

        mcp_manager = MCPManager(session)

        # Mock MCPHelper to simulate tool discovery
        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
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
                }
            ]
            MockMCPHelper.return_value = mock_helper_instance

            # Connect server (discovers tools)
            await mcp_manager.connect_server("filesystem")

            # Register discovered tools with agent
            await mcp_manager.register_tools_with_agent(
                server_id="filesystem",
                agent_name="test_agent"
            )

            # Verify agent has registered tools
            agent = session.agents["test_agent"]
            assert "mcp_filesystem_read_file" in agent.tools
            assert "mcp_filesystem_write_file" in agent.tools
            assert len(agent.tools) == 2

            # Verify tool name format
            for tool in agent.tools:
                assert tool.startswith("mcp_filesystem_")

    @pytest.mark.asyncio
    async def test_tools_persist_across_session_save_load(
        self, session_manager, filesystem_server, test_agent
    ):
        """Integration: Registered tools persist through session save/load.

        Validates:
        - Session with registered tools can be saved
        - Session can be loaded from storage
        - Agent's tools list is restored correctly
        - Tool registration survives session lifecycle
        """
        # Create session with server and agent
        session = session_manager.create_session(
            name="persistence-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        session.agents["test_agent"] = test_agent

        mcp_manager = MCPManager(session)

        # Mock MCPHelper for tool discovery
        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
                {
                    "type": "function",
                    "function": {"name": "mcp_filesystem_read_file"}
                }
            ]
            MockMCPHelper.return_value = mock_helper_instance

            # Connect and register tools
            await mcp_manager.connect_server("filesystem")
            await mcp_manager.register_tools_with_agent(
                server_id="filesystem",
                agent_name="test_agent"
            )

            # Save session
            session_manager.save_session(session)

        # Load session (should restore tools)
        loaded_session = session_manager.load_session("persistence-test")
        loaded_agent = loaded_session.agents["test_agent"]

        # Verify tools persisted
        assert "mcp_filesystem_read_file" in loaded_agent.tools
        assert len(loaded_agent.tools) == 1

    @pytest.mark.asyncio
    async def test_multiple_agents_different_tools(
        self, session_manager, filesystem_server
    ):
        """Integration: Different agents can have different MCP tools registered.

        Validates:
        - Multiple agents in same session
        - Tools registered selectively per agent
        - One agent's tools don't affect another
        - Each agent maintains independent tool list
        """
        # Create two agents
        agent1 = Agent(
            name="filesystem_agent",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Agent with filesystem tools"
        )
        agent2 = Agent(
            name="calculator_agent",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Agent with calculator tools"
        )

        # Create calculator server
        calculator_server = MCPServerConfig(
            id="calculator",
            type="stdio",
            command="mcp-server-calculator",
            auto_connect=False
        )

        # Create session with both servers and agents
        session = session_manager.create_session(
            name="multi-agent-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server, calculator_server]
        )
        session.agents["filesystem_agent"] = agent1
        session.agents["calculator_agent"] = agent2

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

            # Connect both servers
            await mcp_manager.connect_server("filesystem")
            await mcp_manager.connect_server("calculator")

            # Register filesystem tools with agent1
            await mcp_manager.register_tools_with_agent(
                server_id="filesystem",
                agent_name="filesystem_agent"
            )

            # Register calculator tools with agent2
            await mcp_manager.register_tools_with_agent(
                server_id="calculator",
                agent_name="calculator_agent"
            )

            # Verify agent1 has only filesystem tools
            assert "mcp_filesystem_read_file" in agent1.tools
            assert "mcp_calculator_add" not in agent1.tools
            assert len(agent1.tools) == 1

            # Verify agent2 has only calculator tools
            assert "mcp_calculator_add" in agent2.tools
            assert "mcp_calculator_multiply" in agent2.tools
            assert "mcp_filesystem_read_file" not in agent2.tools
            assert len(agent2.tools) == 2

    @pytest.mark.asyncio
    async def test_tool_registration_preserves_agent_properties(
        self, session_manager, filesystem_server, test_agent
    ):
        """Integration: Tool registration doesn't affect other agent properties.

        Validates:
        - Agent name preserved
        - Model name preserved
        - Description preserved
        - System prompt preserved
        - Only tools list modified
        """
        # Set additional agent properties
        test_agent.system_prompt = "You are a helpful assistant"
        test_agent.temperature = 0.7

        session = session_manager.create_session(
            name="properties-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        session.agents["test_agent"] = test_agent

        mcp_manager = MCPManager(session)

        # Store original properties
        original_name = test_agent.name
        original_model = test_agent.model_name
        original_description = test_agent.description
        original_prompt = test_agent.system_prompt
        original_temp = test_agent.temperature

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

            # Verify all properties preserved
            assert test_agent.name == original_name
            assert test_agent.model_name == original_model
            assert test_agent.description == original_description
            assert test_agent.system_prompt == original_prompt
            assert test_agent.temperature == original_temp

            # Verify only tools changed
            assert len(test_agent.tools) == 1
            assert "mcp_filesystem_read_file" in test_agent.tools

    @pytest.mark.asyncio
    async def test_auto_register_tools_on_server_connect(
        self, session_manager, filesystem_server, test_agent
    ):
        """Integration: Optional auto-registration when server connects.

        Validates:
        - Server connection with auto_register_with_agents flag
        - Tools automatically registered with specified agents
        - No manual registration call needed
        - Useful for default agent tool setup
        """
        session = session_manager.create_session(
            name="auto-register-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        session.agents["test_agent"] = test_agent

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}}
            ]
            MockMCPHelper.return_value = mock_helper_instance

            # Connect with auto-registration enabled
            await mcp_manager.connect_server(
                server_id="filesystem",
                auto_register_with_agents=["test_agent"]
            )

            # Tools should be automatically registered
            assert "mcp_filesystem_read_file" in test_agent.tools

    @pytest.mark.asyncio
    async def test_unregister_tools_on_server_disconnect(
        self, session_manager, filesystem_server, test_agent
    ):
        """Integration: Tools removed when server disconnects.

        Validates:
        - Tools registered with agent
        - Server disconnection
        - Tools automatically removed from agent
        - Clean cleanup on disconnect
        """
        session = session_manager.create_session(
            name="unregister-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        session.agents["test_agent"] = test_agent

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}}
            ]
            MockMCPHelper.return_value = mock_helper_instance

            # Connect and register
            await mcp_manager.connect_server("filesystem")
            await mcp_manager.register_tools_with_agent(
                server_id="filesystem",
                agent_name="test_agent"
            )

            # Verify tool registered
            assert "mcp_filesystem_read_file" in test_agent.tools

            # Disconnect server (should unregister tools)
            await mcp_manager.disconnect_server("filesystem")

            # Verify tool removed
            assert "mcp_filesystem_read_file" not in test_agent.tools
            assert len(test_agent.tools) == 0

    @pytest.mark.asyncio
    async def test_register_tools_with_nonexistent_agent(
        self, session_manager, filesystem_server
    ):
        """Integration: Error handling when registering with nonexistent agent.

        Validates:
        - Attempting to register tools with agent that doesn't exist
        - Raises ValueError with clear error message
        - Server state unaffected by failed registration
        """
        session = session_manager.create_session(
            name="nonexistent-agent-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )

        mcp_manager = MCPManager(session)

        with patch('promptchain.cli.utils.mcp_manager.MCPHelper') as MockMCPHelper:
            mock_helper_instance = Mock()
            mock_helper_instance.connect_mcp_async = AsyncMock()
            mock_helper_instance.mcp_tool_schemas = [
                {"type": "function", "function": {"name": "mcp_filesystem_read_file"}}
            ]
            MockMCPHelper.return_value = mock_helper_instance

            # Connect server
            await mcp_manager.connect_server("filesystem")

            # Try to register with nonexistent agent
            with pytest.raises(ValueError, match="Agent 'nonexistent' not found"):
                await mcp_manager.register_tools_with_agent(
                    server_id="filesystem",
                    agent_name="nonexistent"
                )

            # Verify server still connected
            assert filesystem_server.state == "connected"

    @pytest.mark.asyncio
    async def test_register_tools_from_disconnected_server(
        self, session_manager, filesystem_server, test_agent
    ):
        """Integration: Error handling when registering from disconnected server.

        Validates:
        - Server not connected
        - Attempting to register tools raises ValueError
        - Clear error message about server state
        """
        session = session_manager.create_session(
            name="disconnected-server-test",
            working_directory=Path("/tmp"),
            mcp_servers=[filesystem_server]
        )
        session.agents["test_agent"] = test_agent

        mcp_manager = MCPManager(session)

        # Don't connect server, try to register tools
        with pytest.raises(ValueError, match="Server 'filesystem' is not connected"):
            await mcp_manager.register_tools_with_agent(
                server_id="filesystem",
                agent_name="test_agent"
            )

        # Verify agent has no tools
        assert len(test_agent.tools) == 0
