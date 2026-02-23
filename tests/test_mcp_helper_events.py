"""Tests for MCPHelper event firing integration.

This module tests that MCPHelper correctly fires events during MCP operations
including connection, disconnection, tool discovery, and tool execution.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List
from promptchain.utils.mcp_helpers import MCPHelper
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType
from promptchain.utils.execution_callback import CallbackManager


class EventCollector:
    """Helper class to collect events during tests."""

    def __init__(self):
        self.events: List[ExecutionEvent] = []

    async def collect_event(self, event: ExecutionEvent) -> None:
        """Collect an event for later inspection."""
        self.events.append(event)

    def get_events_by_type(self, event_type: ExecutionEventType) -> List[ExecutionEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def clear(self):
        """Clear collected events."""
        self.events.clear()


@pytest.fixture
def event_collector():
    """Fixture providing an event collector."""
    return EventCollector()


@pytest.fixture
def callback_manager(event_collector):
    """Fixture providing a callback manager with event collector."""
    manager = CallbackManager()
    manager.register(event_collector.collect_event)
    return manager


@pytest.fixture
def mcp_helper_no_callback():
    """Fixture providing MCPHelper without callback manager."""
    import logging
    return MCPHelper(
        mcp_servers=[],
        verbose=False,
        logger_instance=logging.getLogger(__name__),
        local_tool_schemas_ref=[],
        local_tool_functions_ref={},
        callback_manager=None
    )


@pytest.fixture
def mcp_helper_with_callback(callback_manager):
    """Fixture providing MCPHelper with callback manager."""
    import logging
    return MCPHelper(
        mcp_servers=[],
        verbose=False,
        logger_instance=logging.getLogger(__name__),
        local_tool_schemas_ref=[],
        local_tool_functions_ref={},
        callback_manager=callback_manager
    )


class TestMCPHelperEventFiring:
    """Test suite for MCPHelper event firing."""

    def test_helper_without_callback_manager(self, mcp_helper_no_callback):
        """Test that MCPHelper works without callback manager."""
        # Should not raise any errors
        assert mcp_helper_no_callback.callback_manager is None

    def test_helper_with_callback_manager(self, mcp_helper_with_callback, callback_manager):
        """Test that MCPHelper accepts callback manager."""
        assert mcp_helper_with_callback.callback_manager is callback_manager

    @pytest.mark.asyncio
    async def test_emit_event_without_callback_manager(self, mcp_helper_no_callback):
        """Test that _emit_event handles missing callback manager gracefully."""
        # Should not raise any errors
        await mcp_helper_no_callback._emit_event(
            ExecutionEventType.MCP_CONNECT_START,
            {"server_id": "test"}
        )

    @pytest.mark.asyncio
    async def test_emit_event_with_callback_manager(
        self,
        mcp_helper_with_callback,
        event_collector
    ):
        """Test that _emit_event fires events correctly."""
        await mcp_helper_with_callback._emit_event(
            ExecutionEventType.MCP_CONNECT_START,
            {"server_id": "test", "transport": "stdio"}
        )

        events = event_collector.get_events_by_type(ExecutionEventType.MCP_CONNECT_START)
        assert len(events) == 1
        assert events[0].metadata["server_id"] == "test"
        assert events[0].metadata["transport"] == "stdio"


class TestMCPConnectionEvents:
    """Test suite for MCP connection/disconnection events."""

    @pytest.mark.asyncio
    async def test_connection_start_event_fired(
        self,
        mcp_helper_with_callback,
        event_collector
    ):
        """Test that MCP_CONNECT_START event is fired."""
        # Mock MCP availability
        with patch('promptchain.utils.mcp_helpers.MCP_AVAILABLE', True), \
             patch('promptchain.utils.mcp_helpers.experimental_mcp_client') as mock_client:

            # Configure helper with test server
            mcp_helper_with_callback.mcp_servers = [{
                "id": "test_server",
                "type": "stdio",
                "command": "test-command",
                "args": []
            }]

            # Mock the connection to fail early (we just want to test event firing)
            mock_client.load_mcp_tools = AsyncMock(side_effect=Exception("Test error"))

            try:
                await mcp_helper_with_callback.connect_mcp_async()
            except:
                pass  # We expect errors, we just want to verify events

            # Verify MCP_CONNECT_START was fired
            start_events = event_collector.get_events_by_type(ExecutionEventType.MCP_CONNECT_START)
            assert len(start_events) == 1
            assert start_events[0].metadata["server_id"] == "test_server"
            assert start_events[0].metadata["server_type"] == "stdio"

    @pytest.mark.asyncio
    async def test_connection_error_event_fired(
        self,
        mcp_helper_with_callback,
        event_collector
    ):
        """Test that MCP_ERROR event is fired on connection failure."""
        with patch('promptchain.utils.mcp_helpers.MCP_AVAILABLE', True), \
             patch('promptchain.utils.mcp_helpers.stdio_client') as mock_stdio:

            # Configure helper with test server
            mcp_helper_with_callback.mcp_servers = [{
                "id": "test_server",
                "type": "stdio",
                "command": "test-command"
            }]

            # Mock connection failure
            mock_stdio.side_effect = Exception("Connection failed")

            try:
                await mcp_helper_with_callback.connect_mcp_async()
            except:
                pass

            # Verify MCP_ERROR was fired
            error_events = event_collector.get_events_by_type(ExecutionEventType.MCP_ERROR)
            assert len(error_events) >= 1
            # Find the connection error
            conn_errors = [e for e in error_events if e.metadata.get("phase") == "connection"]
            assert len(conn_errors) >= 1
            assert "Connection failed" in conn_errors[0].metadata["error"]

    @pytest.mark.asyncio
    async def test_disconnect_events_fired(
        self,
        mcp_helper_with_callback,
        event_collector
    ):
        """Test that disconnect events are fired."""
        # Simulate having a connected session
        mcp_helper_with_callback.mcp_sessions = {"test_server": Mock()}
        mcp_helper_with_callback.exit_stack = AsyncMock()
        mcp_helper_with_callback.exit_stack.aclose = AsyncMock()

        await mcp_helper_with_callback.close_mcp_async()

        # Verify disconnect events
        start_events = event_collector.get_events_by_type(ExecutionEventType.MCP_DISCONNECT_START)
        end_events = event_collector.get_events_by_type(ExecutionEventType.MCP_DISCONNECT_END)

        assert len(start_events) == 1
        assert len(end_events) == 1
        assert start_events[0].metadata["server_id"] == "test_server"
        assert end_events[0].metadata["server_id"] == "test_server"
        assert end_events[0].metadata["status"] == "disconnected"


class TestMCPToolEvents:
    """Test suite for MCP tool-related events."""

    @pytest.mark.asyncio
    async def test_tool_not_found_error_event(
        self,
        mcp_helper_with_callback,
        event_collector
    ):
        """Test that MCP_ERROR is fired when tool not found."""
        # Mock tool call
        tool_call = {
            "id": "test_call_1",
            "function": {
                "name": "nonexistent_tool",
                "arguments": "{}"
            }
        }

        result = await mcp_helper_with_callback.execute_mcp_tool(tool_call)

        # Verify error event
        error_events = event_collector.get_events_by_type(ExecutionEventType.MCP_ERROR)
        assert len(error_events) >= 1
        tool_errors = [e for e in error_events if e.metadata.get("phase") == "tool_execution"]
        assert len(tool_errors) >= 1
        assert "not found in map" in tool_errors[0].metadata["error"]
        assert tool_errors[0].metadata["tool_name"] == "nonexistent_tool"

    @pytest.mark.asyncio
    async def test_mcp_library_unavailable_error(
        self,
        mcp_helper_with_callback,
        event_collector
    ):
        """Test error event when MCP library is unavailable."""
        # Add tool to map but mock MCP as unavailable
        mcp_helper_with_callback.mcp_tools_map = {
            "test_tool": {
                "server_id": "test_server",
                "original_schema": {"function": {"name": "test_tool"}}
            }
        }

        with patch('promptchain.utils.mcp_helpers.MCP_AVAILABLE', False):
            tool_call = {
                "id": "test_call_2",
                "function": {"name": "test_tool", "arguments": "{}"}
            }

            result = await mcp_helper_with_callback.execute_mcp_tool(tool_call)

            # Verify error event
            error_events = event_collector.get_events_by_type(ExecutionEventType.MCP_ERROR)
            tool_errors = [e for e in error_events if "library not available" in e.metadata.get("error", "")]
            assert len(tool_errors) >= 1

    @pytest.mark.asyncio
    async def test_session_not_found_error(
        self,
        mcp_helper_with_callback,
        event_collector
    ):
        """Test error event when MCP session is not found."""
        # Add tool but no session
        mcp_helper_with_callback.mcp_tools_map = {
            "test_tool": {
                "server_id": "test_server",
                "original_schema": {"function": {"name": "test_tool"}}
            }
        }
        mcp_helper_with_callback.mcp_sessions = {}  # No sessions

        with patch('promptchain.utils.mcp_helpers.MCP_AVAILABLE', True), \
             patch('promptchain.utils.mcp_helpers.experimental_mcp_client'):

            tool_call = {
                "id": "test_call_3",
                "function": {"name": "test_tool", "arguments": "{}"}
            }

            result = await mcp_helper_with_callback.execute_mcp_tool(tool_call)

            # Verify error event
            error_events = event_collector.get_events_by_type(ExecutionEventType.MCP_ERROR)
            session_errors = [e for e in error_events if "Session not found" in e.metadata.get("reason", "")]
            assert len(session_errors) >= 1
            assert session_errors[0].metadata["server_id"] == "test_server"


class TestEventMetadata:
    """Test suite for event metadata completeness."""

    @pytest.mark.asyncio
    async def test_connect_start_metadata(
        self,
        mcp_helper_with_callback,
        event_collector
    ):
        """Test MCP_CONNECT_START event metadata."""
        await mcp_helper_with_callback._emit_event(
            ExecutionEventType.MCP_CONNECT_START,
            {
                "server_id": "test",
                "server_type": "stdio",
                "command": "test-cmd",
                "transport": "stdio"
            }
        )

        events = event_collector.get_events_by_type(ExecutionEventType.MCP_CONNECT_START)
        assert len(events) == 1
        metadata = events[0].metadata
        assert "server_id" in metadata
        assert "server_type" in metadata
        assert "command" in metadata
        assert "transport" in metadata

    @pytest.mark.asyncio
    async def test_error_event_metadata(
        self,
        mcp_helper_with_callback,
        event_collector
    ):
        """Test MCP_ERROR event metadata completeness."""
        await mcp_helper_with_callback._emit_event(
            ExecutionEventType.MCP_ERROR,
            {
                "error": "Test error",
                "error_type": "TestException",
                "server_id": "test",
                "phase": "connection"
            }
        )

        events = event_collector.get_events_by_type(ExecutionEventType.MCP_ERROR)
        assert len(events) == 1
        metadata = events[0].metadata
        assert "error" in metadata
        assert "error_type" in metadata
        assert "server_id" in metadata
        assert "phase" in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
