"""Unit tests for MCP tool name prefixing validation (T060).

These tests verify that MCP tool names are properly prefixed with server IDs
to prevent naming conflicts when multiple servers provide tools with the same name.

Test Coverage:
- test_tool_name_prefix_format: Validates prefix format (server_id_tool_name)
- test_prefix_prevents_duplicate_names: Two servers with same tool name don't conflict
- test_prefix_extraction: Can extract server ID and original tool name from prefixed name
- test_invalid_prefix_detection: Detects incorrectly formatted tool names
- test_multiple_tools_from_same_server: All tools from server get same prefix
"""

import pytest
from typing import List

from promptchain.cli.utils.mcp_tool_prefixing import (
    prefix_tool_name,
    extract_server_id,
    extract_original_tool_name,
    validate_tool_name_format,
    is_prefixed_tool_name
)


class TestMCPToolPrefixing:
    """Unit tests for MCP tool name prefixing validation."""

    def test_tool_name_prefix_format(self):
        """Unit: Tool names follow format server_id__tool_name.

        Validates:
        - Tool name prefixed with server ID
        - Double underscore separator between server ID and tool name
        - Prefix maintains readability
        """
        server_id = "filesystem"
        tool_name = "read_file"

        prefixed_name = prefix_tool_name(server_id, tool_name)

        # Should follow format: server_id__tool_name (double underscore)
        assert prefixed_name == f"{server_id}__{tool_name}"
        assert prefixed_name == "filesystem__read_file"

        # Verify prefix is at the start
        assert prefixed_name.startswith(f"{server_id}__")

    def test_prefix_prevents_duplicate_names(self):
        """Unit: Prefixing prevents conflicts from duplicate tool names.

        Validates:
        - Two servers with same tool name produce different prefixed names
        - Prefix makes tool names unique
        - No collision in tool registry
        """
        # Two different servers with same tool name
        server1_id = "calculator1"
        server2_id = "calculator2"
        tool_name = "add"

        prefixed1 = prefix_tool_name(server1_id, tool_name)
        prefixed2 = prefix_tool_name(server2_id, tool_name)

        # Should be different
        assert prefixed1 != prefixed2
        assert prefixed1 == "calculator1__add"
        assert prefixed2 == "calculator2__add"

        # Both should be valid prefixed names
        assert is_prefixed_tool_name(prefixed1)
        assert is_prefixed_tool_name(prefixed2)

    def test_prefix_extraction(self):
        """Unit: Can extract server ID and original tool name from prefixed name.

        Validates:
        - extract_server_id() returns correct server ID
        - extract_original_tool_name() returns original tool name
        - Round-trip: prefix → extract → matches original
        """
        server_id = "web_search"
        tool_name = "search_google"

        prefixed_name = prefix_tool_name(server_id, tool_name)

        # Extract components
        extracted_server_id = extract_server_id(prefixed_name)
        extracted_tool_name = extract_original_tool_name(prefixed_name)

        # Should match originals
        assert extracted_server_id == server_id
        assert extracted_tool_name == tool_name

        # Round-trip test
        re_prefixed = prefix_tool_name(extracted_server_id, extracted_tool_name)
        assert re_prefixed == prefixed_name

    def test_invalid_prefix_detection(self):
        """Unit: Detects incorrectly formatted tool names.

        Validates:
        - validate_tool_name_format() rejects malformed names
        - is_prefixed_tool_name() returns False for invalid names
        - Clear error messages for invalid formats
        """
        # Valid formats (double underscore)
        assert validate_tool_name_format("filesystem__read_file") is True
        assert is_prefixed_tool_name("calculator__add") is True

        # Invalid formats (no double underscore prefix)
        assert is_prefixed_tool_name("read_file") is False
        assert is_prefixed_tool_name("add") is False

        # Invalid formats (malformed)
        assert is_prefixed_tool_name("") is False
        assert is_prefixed_tool_name("__tool") is False  # Missing server ID
        assert is_prefixed_tool_name("server__") is False  # Missing tool name

    def test_multiple_tools_from_same_server(self):
        """Unit: All tools from same server get same prefix.

        Validates:
        - Multiple tools from one server all prefixed with same server ID
        - Prefixes consistent across tool list
        - Can identify all tools belonging to a server
        """
        server_id = "filesystem"
        tools = ["read_file", "write_file", "list_directory", "delete_file"]

        # Prefix all tools
        prefixed_tools = [prefix_tool_name(server_id, tool) for tool in tools]

        # All should have same server ID prefix
        for prefixed_tool in prefixed_tools:
            assert extract_server_id(prefixed_tool) == server_id
            assert prefixed_tool.startswith(f"{server_id}_")

        # All should be different
        assert len(prefixed_tools) == len(set(prefixed_tools))

        # Verify original tool names preserved
        extracted_tools = [extract_original_tool_name(pt) for pt in prefixed_tools]
        assert extracted_tools == tools

    def test_prefix_with_special_characters(self):
        """Unit: Handles tool names with special characters.

        Validates:
        - Tool names with hyphens, dots, numbers
        - Server IDs with special characters
        - Prefix format maintained for edge cases
        """
        # Tool with hyphen
        assert prefix_tool_name("server", "tool-name") == "server__tool-name"

        # Tool with dot
        assert prefix_tool_name("server", "tool.execute") == "server__tool.execute"

        # Tool with number
        assert prefix_tool_name("server", "tool_v2") == "server__tool_v2"

        # Server ID with hyphen
        assert prefix_tool_name("my-server", "tool") == "my-server__tool"

    def test_empty_and_none_handling(self):
        """Unit: Graceful handling of empty/None inputs.

        Validates:
        - Empty server ID raises ValueError
        - Empty tool name raises ValueError
        - None inputs raise TypeError
        """
        # Empty strings should raise ValueError
        with pytest.raises(ValueError, match="server_id cannot be empty"):
            prefix_tool_name("", "tool_name")

        with pytest.raises(ValueError, match="tool_name cannot be empty"):
            prefix_tool_name("server_id", "")

        # None should raise TypeError
        with pytest.raises(TypeError):
            prefix_tool_name(None, "tool_name")

        with pytest.raises(TypeError):
            prefix_tool_name("server_id", None)
