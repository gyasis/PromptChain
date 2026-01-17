"""Fixtures and mocks for CLI integration tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any


@pytest.fixture
def mock_mcp_helper():
    """Create a mock MCPHelper that simulates tool discovery.

    This mock replaces the real MCPHelper to avoid requiring actual
    MCP server executables during tests.
    """
    mock = AsyncMock()

    # Mock tool schemas for filesystem server
    mock.filesystem_tools = [
        {
            "type": "function",
            "function": {
                "name": "mcp_filesystem_read_file",
                "description": "Read contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_filesystem_write_file",
                "description": "Write contents to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "File content"}
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_filesystem_list_directory",
                "description": "List files in a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path"}
                    },
                    "required": ["path"]
                }
            }
        }
    ]

    # Mock tool schemas for web search server
    mock.web_search_tools = [
        {
            "type": "function",
            "function": {
                "name": "mcp_web_search_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_web_search_get_url",
                "description": "Fetch content from URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"}
                    },
                    "required": ["url"]
                }
            }
        }
    ]

    return mock


@pytest.fixture
def mock_mcp_helper_class(mock_mcp_helper):
    """Patch MCPHelper class to return mock instance.

    This fixture patches the MCPHelper class constructor so that
    when MCPManager creates an MCPHelper instance, it gets our mock.
    """
    def create_mock_helper(mcp_servers=None, **kwargs):
        """Factory function that returns configured mock based on server config."""
        # Determine which server is being connected based on mcp_servers config
        if mcp_servers and len(mcp_servers) > 0:
            server_id = mcp_servers[0].get("id", "")

            # Set tool schemas based on server type
            if "filesystem" in server_id:
                mock_mcp_helper.mcp_tool_schemas = mock_mcp_helper.filesystem_tools
            elif "web_search" in server_id:
                mock_mcp_helper.mcp_tool_schemas = mock_mcp_helper.web_search_tools
            else:
                mock_mcp_helper.mcp_tool_schemas = []
        else:
            mock_mcp_helper.mcp_tool_schemas = []

        # Mock the connect method to simulate successful connection
        async def mock_connect():
            """Simulate successful MCP connection and tool discovery."""
            return True

        mock_mcp_helper.connect_mcp_async = AsyncMock(side_effect=mock_connect)

        return mock_mcp_helper

    with patch("promptchain.cli.utils.mcp_manager.MCPHelper", side_effect=create_mock_helper):
        yield mock_mcp_helper
