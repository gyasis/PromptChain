"""MCP (Model Context Protocol) server configuration model.

This module defines the configuration and state tracking for external MCP servers
that provide tools for agent use (filesystem, code execution, web search, etc.).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional


@dataclass
class MCPServerConfig:
    """Configuration for MCP (Model Context Protocol) server connection.

    MCP servers provide external tool capabilities that agents can use during
    conversations (file operations, code execution, web search, database queries).

    Attributes:
        id: Unique server identifier (used in tool name prefixing)
        type: Connection type (stdio or http)
        command: Command to start server (for stdio type)
        args: Command-line arguments for stdio servers
        url: HTTP endpoint URL (for http type)
        auto_connect: Whether to connect automatically at session start
        state: Current connection state
        discovered_tools: List of tool names discovered from server
        error_message: Last error message (if connection failed)
        connected_at: Timestamp of successful connection
    """

    id: str
    type: Literal["stdio", "http"]
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    url: Optional[str] = None
    auto_connect: bool = False  # Default to False for explicit control
    state: Literal["disconnected", "connected", "error"] = "disconnected"
    discovered_tools: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    connected_at: Optional[float] = None

    def __post_init__(self):
        """Validate MCP server configuration."""
        if not self.id or len(self.id) > 50:
            raise ValueError(f"Server id must be 1-50 characters, got: {len(self.id)}")

        if self.type == "stdio":
            if not self.command:
                raise ValueError("stdio type requires 'command' parameter")
        elif self.type == "http":
            if not self.url:
                raise ValueError("http type requires 'url' parameter")
        else:
            raise ValueError(f"Invalid type: {self.type}")

    def mark_connected(self, tools: List[str]):
        """Mark server as successfully connected with discovered tools.

        Args:
            tools: List of tool names discovered from server
        """
        self.state = "connected"
        self.discovered_tools = tools
        self.connected_at = datetime.now().timestamp()
        self.error_message = None

    def mark_error(self, error: str):
        """Mark server connection as failed with error message.

        Args:
            error: Error message describing connection failure
        """
        self.state = "error"
        self.error_message = error
        self.discovered_tools = []

    def mark_disconnected(self):
        """Mark server as disconnected (intentional disconnect)."""
        self.state = "disconnected"
        self.discovered_tools = []
        self.connected_at = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "type": self.type,
            "command": self.command,
            "args": self.args,
            "url": self.url,
            "auto_connect": self.auto_connect,
            "state": self.state,
            "discovered_tools": self.discovered_tools,
            "error_message": self.error_message,
            "connected_at": self.connected_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServerConfig":
        """Create from dictionary.

        Args:
            data: Server configuration dictionary

        Returns:
            MCPServerConfig: Reconstructed configuration
        """
        return cls(
            id=data["id"],
            type=data["type"],
            command=data.get("command"),
            args=data.get("args", []),
            url=data.get("url"),
            auto_connect=data.get("auto_connect", False),  # Default to False
            state=data.get("state", "disconnected"),
            discovered_tools=data.get("discovered_tools", []),
            error_message=data.get("error_message"),
            connected_at=data.get("connected_at"),
        )

    def __str__(self) -> str:
        """Human-readable server representation."""
        status_emoji = {"connected": "✓", "disconnected": "○", "error": "✗"}
        emoji = status_emoji.get(self.state, "?")

        tool_count = len(self.discovered_tools)
        tool_info = f"{tool_count} tools" if tool_count > 0 else "no tools"

        connection_info = f"{self.command}" if self.type == "stdio" else f"{self.url}"

        if self.state == "error":
            return f"{emoji} {self.id} ({connection_info}) - ERROR: {self.error_message}"
        else:
            return f"{emoji} {self.id} ({connection_info}) - {self.state}, {tool_info}"
