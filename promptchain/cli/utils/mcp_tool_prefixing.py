"""MCP tool name prefixing utilities (T064).

This module provides functions for prefixing MCP tool names with server IDs
to prevent naming conflicts when multiple MCP servers provide tools with the
same name.

Tool Name Format: {server_id}__{original_tool_name}

Example:
    >>> prefix_tool_name("filesystem", "read_file")
    'filesystem__read_file'

    >>> extract_server_id("filesystem__read_file")
    'filesystem'

    >>> extract_original_tool_name("filesystem__read_file")
    'read_file'
"""

from typing import Optional


# Use double underscore to prevent conflicts with underscores in server IDs or tool names
TOOL_NAME_SEPARATOR = "__"


def prefix_tool_name(server_id: str, tool_name: str) -> str:
    """Prefix tool name with server ID to prevent conflicts.

    Args:
        server_id: MCP server identifier (e.g., "filesystem", "calculator")
        tool_name: Original tool name from MCP server (e.g., "read_file")

    Returns:
        str: Prefixed tool name in format: server_id_tool_name

    Raises:
        ValueError: If server_id or tool_name is empty
        TypeError: If server_id or tool_name is None

    Examples:
        >>> prefix_tool_name("filesystem", "read_file")
        'filesystem_read_file'

        >>> prefix_tool_name("web_search", "search_google")
        'web_search_search_google'
    """
    # Validate inputs
    if server_id is None or tool_name is None:
        raise TypeError("server_id and tool_name cannot be None")

    if not server_id:
        raise ValueError("server_id cannot be empty")

    if not tool_name:
        raise ValueError("tool_name cannot be empty")

    # Combine with separator
    return f"{server_id}{TOOL_NAME_SEPARATOR}{tool_name}"


def extract_server_id(prefixed_tool_name: str) -> str:
    """Extract server ID from prefixed tool name.

    Args:
        prefixed_tool_name: Tool name with server prefix (e.g., "filesystem_read_file")

    Returns:
        str: Server ID extracted from prefix

    Raises:
        ValueError: If tool name is not properly prefixed

    Examples:
        >>> extract_server_id("filesystem_read_file")
        'filesystem'

        >>> extract_server_id("web_search_search_google")
        'web_search'
    """
    if not is_prefixed_tool_name(prefixed_tool_name):
        raise ValueError(f"Tool name '{prefixed_tool_name}' is not properly prefixed")

    # Split on first separator and return server ID
    parts = prefixed_tool_name.split(TOOL_NAME_SEPARATOR, 1)
    return parts[0]


def extract_original_tool_name(prefixed_tool_name: str) -> str:
    """Extract original tool name from prefixed tool name.

    Args:
        prefixed_tool_name: Tool name with server prefix (e.g., "filesystem_read_file")

    Returns:
        str: Original tool name without prefix

    Raises:
        ValueError: If tool name is not properly prefixed

    Examples:
        >>> extract_original_tool_name("filesystem_read_file")
        'read_file'

        >>> extract_original_tool_name("web_search_search_google")
        'search_google'
    """
    if not is_prefixed_tool_name(prefixed_tool_name):
        raise ValueError(f"Tool name '{prefixed_tool_name}' is not properly prefixed")

    # Split on first separator and return tool name
    parts = prefixed_tool_name.split(TOOL_NAME_SEPARATOR, 1)
    return parts[1]


def is_prefixed_tool_name(tool_name: str) -> bool:
    """Check if tool name follows proper prefix format.

    Args:
        tool_name: Tool name to validate

    Returns:
        bool: True if properly prefixed, False otherwise

    Examples:
        >>> is_prefixed_tool_name("filesystem_read_file")
        True

        >>> is_prefixed_tool_name("read_file")
        False

        >>> is_prefixed_tool_name("")
        False
    """
    if not tool_name:
        return False

    # Must contain at least one separator
    if TOOL_NAME_SEPARATOR not in tool_name:
        return False

    # Split on first separator
    parts = tool_name.split(TOOL_NAME_SEPARATOR, 1)

    # Must have exactly 2 parts (server_id and tool_name)
    if len(parts) != 2:
        return False

    # Both parts must be non-empty
    server_id, original_name = parts
    if not server_id or not original_name:
        return False

    return True


def validate_tool_name_format(tool_name: str) -> bool:
    """Validate tool name follows proper prefix format.

    Alias for is_prefixed_tool_name() for explicit validation usage.

    Args:
        tool_name: Tool name to validate

    Returns:
        bool: True if properly formatted, False otherwise

    Examples:
        >>> validate_tool_name_format("filesystem_read_file")
        True

        >>> validate_tool_name_format("invalid_format_")
        False
    """
    return is_prefixed_tool_name(tool_name)
