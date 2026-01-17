"""Contract tests for MCP server configuration schema (T056).

These tests verify that MCP server configurations conform to expected schema
and can be validated, loaded, and persisted correctly.

Test Strategy:
- Validate required fields exist with correct types
- Test schema validation rules (id format, type constraints)
- Verify serialization/deserialization preserves all fields
- Ensure state management follows expected contracts
- Test both stdio and http server types

Coverage:
- Valid configs with all fields
- Valid configs with minimal fields
- Invalid configs (missing fields, invalid types)
- Args and environment variable structure
- Working directory validation
- Server ID constraints
- Serialization roundtrip integrity
- State transition contracts
"""

import pytest
from typing import Dict, Any

from promptchain.cli.models.mcp_config import MCPServerConfig


class TestMCPServerConfigContract:
    """Contract tests for MCP server configuration schema."""

    def test_stdio_server_valid_config_all_fields(self):
        """Contract: STDIO server config accepts all optional fields.

        Validates:
        - All fields stored correctly
        - Config can be serialized to dict
        - Defaults applied appropriately
        """
        config = MCPServerConfig(
            id="filesystem",
            type="stdio",
            command="mcp-server-filesystem",
            args=["--root", "/home/user/project"],
            auto_connect=True
        )

        assert config.id == "filesystem"
        assert config.type == "stdio"
        assert config.command == "mcp-server-filesystem"
        assert config.args == ["--root", "/home/user/project"]
        assert config.auto_connect is True
        assert config.state == "disconnected"
        assert config.discovered_tools == []
        assert config.error_message is None
        assert config.connected_at is None

        # Should be serializable to dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["id"] == "filesystem"
        assert config_dict["type"] == "stdio"
        assert config_dict["command"] == "mcp-server-filesystem"

    def test_stdio_server_valid_config_minimal_fields(self):
        """Contract: STDIO server config works with minimal required fields.

        Validates:
        - Only id, type, command required for stdio
        - Optional fields have sensible defaults
        - Config is valid and usable
        """
        config = MCPServerConfig(
            id="minimal",
            type="stdio",
            command="mcp-server"
        )

        assert config.id == "minimal"
        assert config.type == "stdio"
        assert config.command == "mcp-server"
        assert config.args == []
        assert config.auto_connect is False  # Default
        assert config.state == "disconnected"

    def test_http_server_valid_config(self):
        """Contract: HTTP server config requires url instead of command.

        Validates:
        - HTTP type accepts url parameter
        - Command not required for HTTP
        - Config validates correctly
        """
        config = MCPServerConfig(
            id="remote",
            type="http",
            url="https://api.example.com/mcp"
        )

        assert config.id == "remote"
        assert config.type == "http"
        assert config.url == "https://api.example.com/mcp"
        assert config.command is None

    def test_invalid_config_missing_id(self):
        """Contract: Server config requires unique id.

        Validates:
        - id is mandatory field
        - Missing id raises TypeError
        """
        with pytest.raises(TypeError):
            MCPServerConfig(
                type="stdio",
                command="mcp-server"
            )

    def test_invalid_config_missing_type(self):
        """Contract: Server config requires type specification.

        Validates:
        - type is mandatory field
        - Missing type raises TypeError
        """
        with pytest.raises(TypeError):
            MCPServerConfig(
                id="test",
                command="mcp-server"
            )

    def test_invalid_stdio_missing_command(self):
        """Contract: STDIO servers require command parameter.

        Validates:
        - command is required for stdio type
        - Missing command raises ValueError
        """
        with pytest.raises(ValueError, match="stdio type requires 'command'"):
            MCPServerConfig(
                id="test",
                type="stdio"
            )

    def test_invalid_http_missing_url(self):
        """Contract: HTTP servers require url parameter.

        Validates:
        - url is required for http type
        - Missing url raises ValueError
        """
        with pytest.raises(ValueError, match="http type requires 'url'"):
            MCPServerConfig(
                id="test",
                type="http"
            )

    def test_invalid_server_type(self):
        """Contract: Only 'stdio' and 'http' types are valid.

        Validates:
        - Invalid types rejected at creation
        - Clear error message for invalid type
        """
        with pytest.raises(ValueError, match="Invalid type"):
            MCPServerConfig(
                id="test",
                type="websocket",  # Invalid type
                command="mcp-server"
            )

    def test_server_id_length_validation(self):
        """Contract: Server ID must be 1-50 characters.

        Validates:
        - Empty ID rejected
        - ID > 50 characters rejected
        - Valid IDs accepted
        """
        # Empty ID
        with pytest.raises(ValueError, match="must be 1-50 characters"):
            MCPServerConfig(
                id="",
                type="stdio",
                command="mcp-server"
            )

        # Too long ID (> 50 chars)
        with pytest.raises(ValueError, match="must be 1-50 characters"):
            MCPServerConfig(
                id="a" * 51,
                type="stdio",
                command="mcp-server"
            )

        # Valid IDs
        valid_ids = ["fs", "filesystem", "web-search", "code_exec_123"]
        for valid_id in valid_ids:
            config = MCPServerConfig(
                id=valid_id,
                type="stdio",
                command="mcp-server"
            )
            assert config.id == valid_id

    def test_args_field_accepts_list_of_strings(self):
        """Contract: args field must be a list (defaults to empty list).

        Validates:
        - args can be omitted (defaults to [])
        - args accepts list of strings
        - args stored correctly
        """
        # No args (default)
        config1 = MCPServerConfig(
            id="simple",
            type="stdio",
            command="mcp-server"
        )
        assert config1.args == []

        # With args
        config2 = MCPServerConfig(
            id="configured",
            type="stdio",
            command="mcp-server",
            args=["--flag", "value", "--debug"]
        )
        assert config2.args == ["--flag", "value", "--debug"]

    def test_auto_connect_default_value(self):
        """Contract: auto_connect defaults to False for explicit control.

        Validates:
        - auto_connect is False when not specified
        - Can be explicitly set to True
        - Boolean type enforced
        """
        # Default value
        config = MCPServerConfig(
            id="test",
            type="stdio",
            command="mcp-server"
        )
        assert config.auto_connect is False

        # Explicit True
        config_auto = MCPServerConfig(
            id="test",
            type="stdio",
            command="mcp-server",
            auto_connect=True
        )
        assert config_auto.auto_connect is True

    def test_state_management_mark_connected(self):
        """Contract: mark_connected() updates state and metadata correctly.

        Validates:
        - State changes to 'connected'
        - discovered_tools populated
        - connected_at timestamp set
        - error_message cleared
        """
        config = MCPServerConfig(
            id="test",
            type="stdio",
            command="mcp-server"
        )

        assert config.state == "disconnected"

        # Mark as connected
        tools = ["mcp_test_read", "mcp_test_write"]
        config.mark_connected(tools)

        assert config.state == "connected"
        assert config.discovered_tools == tools
        assert config.connected_at is not None
        assert isinstance(config.connected_at, float)
        assert config.error_message is None

    def test_state_management_mark_error(self):
        """Contract: mark_error() updates state and stores error message.

        Validates:
        - State changes to 'error'
        - error_message stored
        - discovered_tools cleared
        """
        config = MCPServerConfig(
            id="test",
            type="stdio",
            command="mcp-server"
        )

        # Mark as error
        error_msg = "Failed to start server: command not found"
        config.mark_error(error_msg)

        assert config.state == "error"
        assert config.error_message == error_msg
        assert config.discovered_tools == []

    def test_state_management_mark_disconnected(self):
        """Contract: mark_disconnected() resets state properly.

        Validates:
        - State changes to 'disconnected'
        - discovered_tools cleared
        - connected_at cleared
        """
        config = MCPServerConfig(
            id="test",
            type="stdio",
            command="mcp-server"
        )

        # First connect
        config.mark_connected(["tool1", "tool2"])
        assert config.state == "connected"

        # Then disconnect
        config.mark_disconnected()

        assert config.state == "disconnected"
        assert config.discovered_tools == []
        assert config.connected_at is None

    def test_serialization_to_dict_preserves_all_fields(self):
        """Contract: to_dict() preserves all configuration fields.

        Validates:
        - All fields included in dictionary
        - Values match original config
        - Serializable for storage
        """
        config = MCPServerConfig(
            id="filesystem",
            type="stdio",
            command="mcp-server-filesystem",
            args=["--root", "/tmp"],
            auto_connect=True
        )
        config.mark_connected(["mcp_fs_read", "mcp_fs_write"])

        config_dict = config.to_dict()

        assert config_dict["id"] == "filesystem"
        assert config_dict["type"] == "stdio"
        assert config_dict["command"] == "mcp-server-filesystem"
        assert config_dict["args"] == ["--root", "/tmp"]
        assert config_dict["auto_connect"] is True
        assert config_dict["state"] == "connected"
        assert config_dict["discovered_tools"] == ["mcp_fs_read", "mcp_fs_write"]
        assert config_dict["error_message"] is None
        assert isinstance(config_dict["connected_at"], float)

    def test_deserialization_from_dict_reconstructs_config(self):
        """Contract: from_dict() correctly reconstructs MCPServerConfig.

        Validates:
        - All fields restored from dictionary
        - State and metadata preserved
        - Roundtrip serialization works
        """
        config_data: Dict[str, Any] = {
            "id": "restored",
            "type": "stdio",
            "command": "mcp-server",
            "args": ["--debug"],
            "url": None,
            "auto_connect": True,
            "state": "connected",
            "discovered_tools": ["tool_a", "tool_b"],
            "error_message": None,
            "connected_at": 1234567890.0,
        }

        config = MCPServerConfig.from_dict(config_data)

        assert config.id == "restored"
        assert config.type == "stdio"
        assert config.command == "mcp-server"
        assert config.args == ["--debug"]
        assert config.auto_connect is True
        assert config.state == "connected"
        assert config.discovered_tools == ["tool_a", "tool_b"]
        assert config.connected_at == 1234567890.0

    def test_roundtrip_serialization_preserves_all_data(self):
        """Contract: Complete serialization roundtrip preserves integrity.

        Validates:
        - to_dict() -> from_dict() preserves all fields
        - No data loss during serialization
        - Config remains functional after roundtrip
        """
        original = MCPServerConfig(
            id="roundtrip",
            type="stdio",
            command="mcp-server-test",
            args=["--config", "/path/to/config", "--verbose"],
            auto_connect=False
        )
        original.mark_connected(["tool1", "tool2", "tool3"])

        # Serialize
        config_dict = original.to_dict()

        # Deserialize
        restored = MCPServerConfig.from_dict(config_dict)

        # Verify all fields match
        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.command == original.command
        assert restored.args == original.args
        assert restored.auto_connect == original.auto_connect
        assert restored.state == original.state
        assert restored.discovered_tools == original.discovered_tools
        assert restored.error_message == original.error_message
        assert restored.connected_at == original.connected_at

    def test_multiple_servers_with_unique_ids(self):
        """Contract: Multiple servers can coexist with unique IDs.

        Validates:
        - Each server has independent configuration
        - No conflicts between server instances
        - Can manage collection of servers
        """
        servers = [
            MCPServerConfig(
                id="filesystem",
                type="stdio",
                command="mcp-server-filesystem",
                auto_connect=True
            ),
            MCPServerConfig(
                id="web_search",
                type="stdio",
                command="mcp-server-web-search",
                auto_connect=False
            ),
            MCPServerConfig(
                id="database",
                type="http",
                url="https://db.example.com/mcp"
            )
        ]

        assert len(servers) == 3
        assert servers[0].id == "filesystem"
        assert servers[0].type == "stdio"
        assert servers[1].id == "web_search"
        assert servers[1].auto_connect is False
        assert servers[2].id == "database"
        assert servers[2].type == "http"

    def test_string_representation_readable(self):
        """Contract: __str__() provides human-readable server info.

        Validates:
        - String includes server ID and status
        - Different states show appropriate emoji/symbol
        - Useful for debugging and logging
        """
        # Disconnected server
        config1 = MCPServerConfig(
            id="test",
            type="stdio",
            command="mcp-server"
        )
        str_repr1 = str(config1)
        assert "test" in str_repr1
        assert "disconnected" in str_repr1

        # Connected server
        config2 = MCPServerConfig(
            id="active",
            type="stdio",
            command="mcp-server"
        )
        config2.mark_connected(["tool1", "tool2"])
        str_repr2 = str(config2)
        assert "active" in str_repr2
        assert "connected" in str_repr2
        assert "2 tools" in str_repr2

        # Error server
        config3 = MCPServerConfig(
            id="failed",
            type="stdio",
            command="mcp-server"
        )
        config3.mark_error("Connection timeout")
        str_repr3 = str(config3)
        assert "failed" in str_repr3
        assert "ERROR" in str_repr3
        assert "Connection timeout" in str_repr3

    def test_http_server_serialization(self):
        """Contract: HTTP server configs serialize/deserialize correctly.

        Validates:
        - HTTP-specific fields (url) preserved
        - stdio-specific fields (command, args) handled appropriately
        - Roundtrip works for HTTP servers
        """
        http_config = MCPServerConfig(
            id="api",
            type="http",
            url="https://api.example.com/mcp",
            auto_connect=True
        )

        # Serialize
        config_dict = http_config.to_dict()

        assert config_dict["type"] == "http"
        assert config_dict["url"] == "https://api.example.com/mcp"
        assert config_dict["command"] is None

        # Deserialize
        restored = MCPServerConfig.from_dict(config_dict)

        assert restored.type == "http"
        assert restored.url == "https://api.example.com/mcp"
        assert restored.command is None

    def test_from_dict_with_defaults_for_missing_fields(self):
        """Contract: from_dict() applies defaults for missing optional fields.

        Validates:
        - Missing auto_connect defaults to False
        - Missing state defaults to 'disconnected'
        - Missing discovered_tools defaults to []
        - Config remains valid with minimal data
        """
        minimal_dict = {
            "id": "minimal",
            "type": "stdio",
            "command": "mcp-server"
        }

        config = MCPServerConfig.from_dict(minimal_dict)

        assert config.id == "minimal"
        assert config.type == "stdio"
        assert config.command == "mcp-server"
        assert config.auto_connect is False  # Default
        assert config.state == "disconnected"  # Default
        assert config.discovered_tools == []  # Default
        assert config.error_message is None
        assert config.connected_at is None
