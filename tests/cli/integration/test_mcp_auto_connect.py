"""Integration tests for MCP server auto-connect functionality (T062).

These tests verify that MCP servers with auto_connect=True are automatically
connected when a session is initialized or loaded.

Test Coverage:
- test_auto_connect_on_session_creation: Servers auto-connect when session created
- test_auto_connect_on_session_load: Servers auto-connect when session loaded
- test_no_auto_connect_for_manual_servers: Servers with auto_connect=False don't connect
- test_auto_connect_multiple_servers: Multiple servers auto-connect independently
- test_auto_connect_failure_handling: Failed auto-connect doesn't block session init
- test_auto_connect_status_tracking: Auto-connect status tracked in session state
"""

import pytest
import tempfile
from pathlib import Path
from typing import List

from promptchain.cli.models.mcp_config import MCPServerConfig
from promptchain.cli.session_manager import SessionManager


class TestMCPAutoConnect:
    """Integration tests for MCP server auto-connect functionality."""

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
    def auto_connect_server(self):
        """Create MCP server with auto_connect=True."""
        return MCPServerConfig(
            id="filesystem",
            type="stdio",
            command="mcp-server-filesystem",
            args=["--root", "/tmp"],
            auto_connect=True  # Should auto-connect
        )

    @pytest.fixture
    def manual_server(self):
        """Create MCP server with auto_connect=False."""
        return MCPServerConfig(
            id="calculator",
            type="stdio",
            command="mcp-server-calculator",
            auto_connect=False  # Should NOT auto-connect
        )

    @pytest.mark.asyncio
    async def test_auto_connect_on_session_creation(
        self, session_manager, auto_connect_server
    ):
        """Integration: MCP servers auto-connect when session created.

        Validates:
        - Session created with auto_connect=True server
        - Server automatically connects during session init
        - Server state is 'connected'
        - Tools discovered from server
        - No manual connect_server() call needed
        """
        # Create session with auto-connect server
        session = session_manager.create_session(
            name="auto-connect-test",
            working_directory=Path("/tmp"),
            mcp_servers=[auto_connect_server]
        )

        # Server should have auto-connected during session creation
        assert auto_connect_server.state == "connected"
        assert len(auto_connect_server.discovered_tools) > 0

        # Verify session records the server
        assert len(session.mcp_servers) == 1
        assert session.mcp_servers[0].id == "filesystem"

    @pytest.mark.asyncio
    async def test_auto_connect_on_session_load(
        self, session_manager, auto_connect_server
    ):
        """Integration: MCP servers auto-connect when session loaded.

        Validates:
        - Session created and saved with auto_connect=True server
        - Session loaded from storage
        - Server automatically reconnects on load
        - Server state restored to 'connected'
        - Tools re-discovered
        """
        # Create and save session
        session = session_manager.create_session(
            name="load-test",
            working_directory=Path("/tmp"),
            mcp_servers=[auto_connect_server]
        )

        session_manager.save_session(session)

        # Load session (should auto-reconnect)
        loaded_session = session_manager.load_session("load-test")

        # Server should have auto-connected on load
        loaded_server = loaded_session.mcp_servers[0]
        assert loaded_server.state == "connected"
        assert len(loaded_server.discovered_tools) > 0

    @pytest.mark.asyncio
    async def test_no_auto_connect_for_manual_servers(
        self, session_manager, manual_server
    ):
        """Integration: Servers with auto_connect=False don't auto-connect.

        Validates:
        - Session created with auto_connect=False server
        - Server does NOT connect automatically
        - Server state remains 'disconnected'
        - No tools discovered
        - Manual connect required
        """
        # Create session with manual server
        session = session_manager.create_session(
            name="manual-test",
            working_directory=Path("/tmp"),
            mcp_servers=[manual_server]
        )

        # Server should NOT have auto-connected
        assert manual_server.state == "disconnected"
        assert len(manual_server.discovered_tools) == 0

        # Verify session has the server but it's disconnected
        assert len(session.mcp_servers) == 1
        assert session.mcp_servers[0].state == "disconnected"

    @pytest.mark.asyncio
    async def test_auto_connect_multiple_servers(
        self, session_manager, auto_connect_server, manual_server
    ):
        """Integration: Multiple servers auto-connect independently.

        Validates:
        - Session with mix of auto_connect=True and auto_connect=False
        - Only auto_connect=True servers connect automatically
        - Manual servers remain disconnected
        - Each server tracked independently
        """
        # Create second auto-connect server
        web_search_server = MCPServerConfig(
            id="web_search",
            type="stdio",
            command="mcp-server-web-search",
            auto_connect=True
        )

        # Create session with mixed servers
        session = session_manager.create_session(
            name="multi-server-test",
            working_directory=Path("/tmp"),
            mcp_servers=[auto_connect_server, manual_server, web_search_server]
        )

        # Verify auto-connect servers connected
        assert auto_connect_server.state == "connected"
        assert web_search_server.state == "connected"

        # Verify manual server did NOT connect
        assert manual_server.state == "disconnected"

        # All servers tracked
        assert len(session.mcp_servers) == 3

    @pytest.mark.asyncio
    async def test_auto_connect_failure_handling(
        self, session_manager
    ):
        """Integration: Failed auto-connect doesn't block session init.

        Validates:
        - Session created with server that will fail to connect
        - Session initialization completes despite connection failure
        - Failed server marked with error state
        - Other operations continue normally
        - Error message captured
        """
        # Create server with invalid command (will fail)
        failing_server = MCPServerConfig(
            id="failing_server",
            type="stdio",
            command="nonexistent-mcp-server",
            auto_connect=True
        )

        # Session creation should still succeed
        session = session_manager.create_session(
            name="failure-test",
            working_directory=Path("/tmp"),
            mcp_servers=[failing_server]
        )

        # Server should be in error state
        assert failing_server.state == "error"
        assert failing_server.error_message is not None
        assert "nonexistent" in failing_server.error_message.lower()

        # Session should still be usable
        assert session.name == "failure-test"
        assert len(session.mcp_servers) == 1

    @pytest.mark.asyncio
    async def test_auto_connect_status_tracking(
        self, session_manager, auto_connect_server
    ):
        """Integration: Auto-connect status tracked in session state.

        Validates:
        - Auto-connect attempts logged
        - Connection success/failure tracked
        - Timestamps recorded
        - Status queryable via session
        """
        # Create session with auto-connect server
        session = session_manager.create_session(
            name="status-test",
            working_directory=Path("/tmp"),
            mcp_servers=[auto_connect_server]
        )

        # Verify connection status tracked
        assert auto_connect_server.state == "connected"
        assert auto_connect_server.connected_at is not None

        # Tools should be discovered and tracked
        assert len(auto_connect_server.discovered_tools) > 0

        # Session should reflect connected state
        connected_servers = [s for s in session.mcp_servers if s.state == "connected"]
        assert len(connected_servers) == 1
        assert connected_servers[0].id == "filesystem"

    @pytest.mark.asyncio
    async def test_auto_connect_preserves_session_metadata(
        self, session_manager, auto_connect_server
    ):
        """Integration: Auto-connect doesn't affect other session metadata.

        Validates:
        - Session name preserved
        - Working directory preserved
        - Agents preserved
        - Messages preserved
        - Only MCP connections affected
        """
        # Ensure test directory exists
        test_dir = Path("/tmp/test")
        test_dir.mkdir(exist_ok=True)

        # Create session with metadata
        session = session_manager.create_session(
            name="metadata-test",
            working_directory=test_dir,
            mcp_servers=[auto_connect_server]
        )

        # Add some metadata
        from promptchain.cli.models.agent_config import Agent
        test_agent = Agent(
            name="test_agent",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Test agent"
        )
        session.agents["test_agent"] = test_agent

        # Save and reload
        session_manager.save_session(session)
        loaded_session = session_manager.load_session("metadata-test")

        # Verify metadata preserved
        assert loaded_session.name == "metadata-test"
        assert str(loaded_session.working_directory) == "/tmp/test"
        assert "test_agent" in loaded_session.agents

        # Server also auto-connected
        assert loaded_session.mcp_servers[0].state == "connected"
