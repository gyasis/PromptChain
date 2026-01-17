"""Integration tests for configuration display commands (T110).

These tests verify the /config show command functionality, including:
- Session information display
- MCP server status and tools
- Agent template listing
- History configuration display
- Agent summary
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import time


class TestConfigShowCommand:
    """Test /config show command functionality."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary sessions directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        """Create SessionManager with temp directory."""
        from promptchain.cli.session_manager import SessionManager
        return SessionManager(sessions_dir=temp_sessions_dir)

    @pytest.fixture
    def command_handler(self, session_manager):
        """Create CommandHandler instance."""
        from promptchain.cli.command_handler import CommandHandler
        return CommandHandler(session_manager=session_manager)

    @pytest.fixture
    def basic_session(self, session_manager):
        """Create a basic test session."""
        from promptchain.cli.models.agent_config import Agent

        session = session_manager.create_session(
            name="config-test",
            working_directory=Path.cwd()
        )

        # Add one agent
        session.agents["default"] = Agent(
            name="default",
            model_name="openai/gpt-4",
            description="Default test agent",
            created_at=time.time()
        )
        session.active_agent = "default"

        return session

    @pytest.fixture
    def session_with_mcp(self, session_manager):
        """Create session with MCP servers configured."""
        from promptchain.cli.models.agent_config import Agent
        from promptchain.cli.models.mcp_config import MCPServerConfig

        session = session_manager.create_session(
            name="mcp-test",
            working_directory=Path.cwd()
        )

        # Add MCP servers
        server1 = MCPServerConfig(
            id="filesystem",
            type="stdio",
            command="mcp-server-filesystem",
            args=["--root", "/tmp"],
            auto_connect=False
        )
        server1.mark_connected(["read_file", "write_file", "list_directory"])

        server2 = MCPServerConfig(
            id="browser",
            type="stdio",
            command="mcp-server-browser",
            auto_connect=False
        )
        server2.mark_disconnected()

        session.mcp_servers = [server1, server2]

        # Add agent
        session.agents["default"] = Agent(
            name="default",
            model_name="openai/gpt-4",
            description="Test agent",
            created_at=time.time()
        )

        return session

    @pytest.fixture
    def session_with_multiple_agents(self, session_manager):
        """Create session with multiple agents and various configurations."""
        from promptchain.cli.models.agent_config import Agent, HistoryConfig

        session = session_manager.create_session(
            name="multi-agent-test",
            working_directory=Path.cwd()
        )

        # Agent with history enabled
        session.agents["researcher"] = Agent(
            name="researcher",
            model_name="openai/gpt-4",
            description="Research specialist",
            tools=["web_search", "mcp_filesystem_read"],
            history_config=HistoryConfig(
                enabled=True,
                max_tokens=8000,
                max_entries=50,
                truncation_strategy="oldest_first"
            ),
            created_at=time.time()
        )

        # Agent with history disabled
        session.agents["terminal"] = Agent(
            name="terminal",
            model_name="openai/gpt-3.5-turbo",
            description="Fast terminal agent",
            tools=["execute_shell"],
            history_config=HistoryConfig(enabled=False, max_tokens=0, max_entries=0),
            created_at=time.time()
        )

        # Agent with default history
        session.agents["coder"] = Agent(
            name="coder",
            model_name="openai/gpt-4",
            description="Code generation",
            tools=["mcp_filesystem_read", "mcp_filesystem_write", "execute_code"],
            created_at=time.time()
        )

        session.active_agent = "researcher"

        return session

    def test_config_show_basic(self, command_handler, basic_session):
        """Test /config show with basic session configuration.

        Validates:
        - Command executes successfully
        - Session info is displayed
        - Agent count is correct
        - Working directory is shown
        """
        result = command_handler.handle_config_show(session=basic_session)

        # Validate success
        assert result.success is True
        assert result.error is None

        # Validate message contains key sections
        message = result.message
        assert "=== Session Configuration ===" in message
        assert f"Session Name: {basic_session.name}" in message
        assert f"Active Agent: {basic_session.active_agent}" in message
        assert "Total Agents: 1" in message
        assert f"Working Directory: {basic_session.working_directory}" in message

        # Validate data structure
        assert result.data is not None
        assert "session" in result.data
        assert result.data["session"]["name"] == basic_session.name
        assert result.data["session"]["agent_count"] == 1

    def test_config_show_no_mcp_servers(self, command_handler, basic_session):
        """Test config display when no MCP servers configured.

        Validates:
        - MCP section shows "No MCP servers configured"
        - No server details are displayed
        """
        result = command_handler.handle_config_show(session=basic_session)

        assert result.success is True
        message = result.message

        # Check MCP section
        assert "=== MCP Servers ===" in message
        assert "No MCP servers configured" in message

        # Validate data
        assert result.data["mcp_servers"]["total"] == 0
        assert result.data["mcp_servers"]["connected"] == 0

    def test_config_show_with_mcp_servers(self, command_handler, session_with_mcp):
        """Test config display with MCP servers configured.

        Validates:
        - Connected and disconnected servers are shown
        - Server status icons are correct
        - Tool counts are displayed
        - Server commands/URLs are shown
        """
        result = command_handler.handle_config_show(session=session_with_mcp)

        assert result.success is True
        message = result.message

        # Check MCP section
        assert "=== MCP Servers ===" in message
        assert "Total Servers: 2" in message
        assert "Connected: 1/2" in message

        # Check filesystem server (connected)
        assert "✓ filesystem" in message
        assert "Status: connected" in message
        assert "Command: mcp-server-filesystem" in message
        assert "Tools: 3 discovered" in message
        assert "read_file" in message

        # Check browser server (disconnected)
        assert "○ browser" in message
        assert "Status: disconnected" in message

        # Validate data
        assert result.data["mcp_servers"]["total"] == 2
        assert result.data["mcp_servers"]["connected"] == 1

    def test_config_show_agent_templates(self, command_handler, basic_session):
        """Test that all agent templates are displayed.

        Validates:
        - All 4 templates are shown (researcher, coder, analyst, terminal)
        - Template metadata is displayed (model, category, complexity)
        - History configuration is shown for each template
        """
        result = command_handler.handle_config_show(session=basic_session)

        assert result.success is True
        message = result.message

        # Check templates section
        assert "=== Available Agent Templates ===" in message
        assert "Total Templates: 4" in message

        # Check each template
        assert "Researcher (researcher)" in message
        assert "Coder (coder)" in message
        assert "Analyst (analyst)" in message
        assert "Terminal (terminal)" in message

        # Check template details
        assert "Model: openai/gpt-4" in message
        assert "Category:" in message
        assert "Complexity:" in message

        # Validate data
        assert result.data["templates"]["count"] == 4
        assert "researcher" in result.data["templates"]["available"]
        assert "coder" in result.data["templates"]["available"]

    def test_config_show_history_configuration(self, command_handler, session_with_multiple_agents):
        """Test history configuration display for multiple agents.

        Validates:
        - Global history manager status
        - Per-agent history overrides
        - Token savings information for disabled history
        """
        result = command_handler.handle_config_show(session=session_with_multiple_agents)

        assert result.success is True
        message = result.message

        # Check history section
        assert "=== History Configuration ===" in message
        assert "Per-Agent History Overrides:" in message

        # Check researcher (history enabled)
        assert "researcher:" in message
        assert "Enabled: True" in message
        assert "Max Tokens: 8000" in message

        # Check terminal (history disabled)
        assert "terminal:" in message
        assert "Enabled: False" in message
        assert "Token Savings: ~60%" in message

    def test_config_show_agent_summary(self, command_handler, session_with_multiple_agents):
        """Test configured agents summary display.

        Validates:
        - All agents are listed
        - Active agent is marked
        - Agent tools are shown
        - Usage count is displayed
        """
        result = command_handler.handle_config_show(session=session_with_multiple_agents)

        assert result.success is True
        message = result.message

        # Check agent section
        assert "=== Configured Agents ===" in message
        # Session manager creates a default agent, so we have 4 total (default + 3 custom)
        assert "Total Agents: 4" in message

        # Check researcher (active)
        assert "researcher (active)" in message
        assert "Model: openai/gpt-4" in message
        assert "Tools: 2 registered" in message
        assert "web_search" in message

        # Check terminal
        assert "terminal\n" in message or "terminal (active)" in message
        assert "Model: openai/gpt-3.5-turbo" in message
        assert "Tools: 1 registered" in message

        # Validate data (includes default agent)
        assert result.data["agents"]["count"] == 4
        assert "default" in result.data["agents"]["names"]
        assert "researcher" in result.data["agents"]["names"]
        assert "terminal" in result.data["agents"]["names"]
        assert "coder" in result.data["agents"]["names"]

    def test_config_show_auto_save_settings(self, command_handler, basic_session):
        """Test auto-save configuration display.

        Validates:
        - Auto-save status is shown
        - Message and time intervals are displayed when enabled
        """
        # Test with auto-save enabled
        basic_session.auto_save_enabled = True
        basic_session.autosave_message_interval = 5
        basic_session.autosave_time_interval = 120

        result = command_handler.handle_config_show(session=basic_session)

        assert result.success is True
        message = result.message

        assert "Auto-Save: enabled" in message
        assert "Message Interval: 5 messages" in message
        assert "Time Interval: 120 seconds" in message

        # Test with auto-save disabled
        basic_session.auto_save_enabled = False
        result = command_handler.handle_config_show(session=basic_session)

        assert result.success is True
        assert "Auto-Save: disabled" in result.message

    def test_config_show_no_agents(self, session_manager, command_handler):
        """Test config display with minimal agents (just default).

        Validates:
        - Default agent is shown
        - No errors when only default agent exists
        """
        session = session_manager.create_session(
            name="empty-test",
            working_directory=Path.cwd()
        )

        result = command_handler.handle_config_show(session=session)

        assert result.success is True
        message = result.message

        # Session manager creates a default agent automatically
        assert "Total Agents: 1" in message
        assert "default" in message
        assert result.data["agents"]["count"] == 1

    def test_config_show_orchestration_config(self, command_handler, basic_session):
        """Test orchestration configuration display (if present).

        Validates:
        - Orchestration section is shown when config exists
        - Execution mode is displayed
        - Auto-include history setting is shown
        """
        # Create orchestration config
        from promptchain.cli.models.orchestration_config import OrchestrationConfig, RouterConfig

        basic_session.orchestration_config = OrchestrationConfig(
            execution_mode="router",
            auto_include_history=True,
            router_config=RouterConfig(
                model="openai/gpt-4",
                decision_prompt_template="""User: {user_input}\nAgents: {agent_details}\nHistory: {history}\nChoose agent.""",
                timeout_seconds=10
            )
        )

        result = command_handler.handle_config_show(session=basic_session)

        assert result.success is True
        message = result.message

        assert "=== Orchestration Configuration ===" in message
        assert "Execution Mode: router" in message
        assert "Auto-Include History: True" in message
        assert "Router Model: openai/gpt-4" in message

    def test_config_show_error_recovery_server_with_error(self, session_manager, command_handler):
        """Test config display when MCP server has error status.

        Validates:
        - Error icon is shown
        - Error message is displayed
        - No tools are listed
        """
        from promptchain.cli.models.agent_config import Agent
        from promptchain.cli.models.mcp_config import MCPServerConfig

        session = session_manager.create_session(
            name="error-test",
            working_directory=Path.cwd()
        )

        # Add server with error
        server = MCPServerConfig(
            id="broken",
            type="stdio",
            command="non-existent-command"
        )
        server.mark_error("Command not found: non-existent-command")

        session.mcp_servers = [server]
        session.agents["default"] = Agent(
            name="default",
            model_name="openai/gpt-4",
            description="Test",
            created_at=time.time()
        )

        result = command_handler.handle_config_show(session=session)

        assert result.success is True
        message = result.message

        assert "✗ broken" in message
        assert "Status: error" in message
        assert "Error: Command not found" in message

    def test_config_show_truncated_tool_list(self, session_manager, command_handler):
        """Test that tool lists are truncated when server has many tools.

        Validates:
        - First 5 tools are shown
        - "... and X more" is displayed for remaining tools
        """
        from promptchain.cli.models.agent_config import Agent
        from promptchain.cli.models.mcp_config import MCPServerConfig

        session = session_manager.create_session(
            name="many-tools-test",
            working_directory=Path.cwd()
        )

        # Create server with 10 tools
        server = MCPServerConfig(
            id="many-tools",
            type="stdio",
            command="test-command"
        )
        tools = [f"tool_{i}" for i in range(10)]
        server.mark_connected(tools)

        session.mcp_servers = [server]
        session.agents["default"] = Agent(
            name="default",
            model_name="openai/gpt-4",
            description="Test",
            created_at=time.time()
        )

        result = command_handler.handle_config_show(session=session)

        assert result.success is True
        message = result.message

        assert "Tools: 10 discovered" in message
        assert "tool_0" in message
        assert "... and 5 more" in message

    def test_config_show_data_structure(self, command_handler, session_with_mcp):
        """Test that returned data structure is complete and correct.

        Validates:
        - All expected keys are present
        - Data types are correct
        - Values match session state
        """
        result = command_handler.handle_config_show(session=session_with_mcp)

        assert result.success is True
        data = result.data

        # Session data
        assert "session" in data
        assert data["session"]["name"] == session_with_mcp.name
        assert data["session"]["id"] == session_with_mcp.id
        assert isinstance(data["session"]["agent_count"], int)

        # MCP servers data
        assert "mcp_servers" in data
        assert data["mcp_servers"]["total"] == 2
        assert data["mcp_servers"]["connected"] == 1
        assert isinstance(data["mcp_servers"]["servers"], list)

        # Templates data
        assert "templates" in data
        assert data["templates"]["count"] == 4
        assert isinstance(data["templates"]["available"], list)

        # Agents data
        assert "agents" in data
        assert isinstance(data["agents"]["count"], int)
        assert isinstance(data["agents"]["names"], list)

    def test_config_show_integration_with_parse_command(self, command_handler, basic_session):
        """Test integration of /config show with command parser.

        Validates:
        - Command is parsed correctly
        - Handler method is callable with session
        """
        # Parse command
        parsed = command_handler.parse_command("/config show")

        assert parsed is not None
        assert parsed.name == "config"
        assert parsed.subcommand == "show"

        # Execute handler directly (in real TUI, this would be routed)
        result = command_handler.handle_config_show(session=basic_session)

        assert result.success is True
        assert len(result.message) > 100  # Substantial output


class TestConfigExportCommand:
    """Test /config export command functionality (T111)."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary sessions directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        """Create SessionManager with temp directory."""
        from promptchain.cli.session_manager import SessionManager
        return SessionManager(sessions_dir=temp_sessions_dir)

    @pytest.fixture
    def command_handler(self, session_manager):
        """Create CommandHandler instance."""
        from promptchain.cli.command_handler import CommandHandler
        return CommandHandler(session_manager=session_manager)

    @pytest.fixture
    def full_session(self, session_manager):
        """Create session with comprehensive configuration for export testing."""
        from promptchain.cli.models.agent_config import Agent, HistoryConfig
        from promptchain.cli.models.mcp_config import MCPServerConfig
        from promptchain.cli.models.orchestration_config import OrchestrationConfig, RouterConfig

        session = session_manager.create_session(
            name="export-test",
            working_directory=Path.cwd()
        )

        # Add agents with various configurations
        session.agents["researcher"] = Agent(
            name="researcher",
            model_name="openai/gpt-4",
            description="Research specialist",
            tools=["web_search", "mcp_filesystem_read"],
            history_config=HistoryConfig(
                enabled=True,
                max_tokens=8000,
                max_entries=50,
                truncation_strategy="oldest_first"
            ),
            created_at=time.time()
        )

        session.agents["terminal"] = Agent(
            name="terminal",
            model_name="openai/gpt-3.5-turbo",
            description="Fast terminal",
            tools=["execute_shell"],
            history_config=HistoryConfig(enabled=False, max_tokens=0, max_entries=0),
            created_at=time.time()
        )

        # Add MCP servers
        server = MCPServerConfig(
            id="filesystem",
            type="stdio",
            command="mcp-server-filesystem",
            args=["--root", "/tmp"],
            auto_connect=True
        )
        session.mcp_servers = [server]

        # Add orchestration config
        session.orchestration_config = OrchestrationConfig(
            execution_mode="router",
            auto_include_history=True,
            router_config=RouterConfig(
                model="openai/gpt-4o-mini",
                decision_prompt_template="User: {user_input}\nAgents: {agent_details}\nChoose agent",
                timeout_seconds=10
            )
        )

        session.active_agent = "researcher"

        return session

    def test_export_yaml_basic(self, command_handler, full_session, temp_sessions_dir):
        """Test basic YAML export functionality.

        Validates:
        - Export succeeds
        - File is created
        - YAML format is valid
        - Contains expected sections
        """
        import yaml

        output_file = temp_sessions_dir / "test_export.yml"
        result = command_handler.handle_config_export(
            session=full_session,
            filename=str(output_file)
        )

        # Validate success
        assert result.success is True
        assert result.error is None
        assert "Configuration exported" in result.message
        assert "YAML" in result.message

        # Validate file exists
        assert output_file.exists()

        # Validate YAML content
        with open(output_file, "r") as f:
            config = yaml.safe_load(f)

        assert config["version"] == "1.0"
        assert "session" in config
        assert "agents" in config
        assert "mcp_servers" in config
        assert "history_configuration" in config
        assert "orchestration" in config

        # Validate data structure
        assert result.data["format"] == "yaml"
        assert result.data["agent_count"] == 3  # default + researcher + terminal
        assert result.data["mcp_server_count"] == 1

    def test_export_json_basic(self, command_handler, full_session, temp_sessions_dir):
        """Test basic JSON export functionality.

        Validates:
        - Export succeeds with .json extension
        - File is valid JSON
        - Contains all expected fields
        """
        import json

        output_file = temp_sessions_dir / "test_export.json"
        result = command_handler.handle_config_export(
            session=full_session,
            filename=str(output_file)
        )

        # Validate success
        assert result.success is True
        assert "JSON" in result.message

        # Validate file exists and is valid JSON
        assert output_file.exists()

        with open(output_file, "r") as f:
            config = json.load(f)

        assert config["version"] == "1.0"
        assert isinstance(config["agents"], dict)
        assert isinstance(config["mcp_servers"], list)

        # Validate data
        assert result.data["format"] == "json"

    def test_export_session_metadata(self, command_handler, full_session, temp_sessions_dir):
        """Test that session metadata is correctly exported.

        Validates:
        - Session name
        - Working directory
        - Active agent
        - Default model
        - Auto-save settings
        """
        import yaml

        output_file = temp_sessions_dir / "session_test.yml"
        result = command_handler.handle_config_export(
            session=full_session,
            filename=str(output_file)
        )

        assert result.success is True

        with open(output_file, "r") as f:
            config = yaml.safe_load(f)

        session_data = config["session"]
        assert session_data["name"] == full_session.name
        assert session_data["active_agent"] == "researcher"
        assert session_data["default_model"] == full_session.default_model
        assert "working_directory" in session_data
        assert "auto_save_enabled" in session_data

    def test_export_agent_configurations(self, command_handler, full_session, temp_sessions_dir):
        """Test that all agent configurations are exported.

        Validates:
        - All agents are present
        - Agent models, descriptions, tools
        - History configurations
        """
        import yaml

        output_file = temp_sessions_dir / "agents_test.yml"
        result = command_handler.handle_config_export(
            session=full_session,
            filename=str(output_file)
        )

        assert result.success is True

        with open(output_file, "r") as f:
            config = yaml.safe_load(f)

        agents = config["agents"]

        # Check researcher agent
        assert "researcher" in agents
        researcher = agents["researcher"]
        assert researcher["model"] == "openai/gpt-4"
        assert researcher["description"] == "Research specialist"
        assert "web_search" in researcher["tools"]
        assert researcher["history_config"]["enabled"] is True
        assert researcher["history_config"]["max_tokens"] == 8000

        # Check terminal agent
        assert "terminal" in agents
        terminal = agents["terminal"]
        assert terminal["model"] == "openai/gpt-3.5-turbo"
        assert terminal["history_config"]["enabled"] is False

    def test_export_mcp_servers(self, command_handler, full_session, temp_sessions_dir):
        """Test that MCP server configurations are exported.

        Validates:
        - Server ID, type, command
        - Args are preserved
        - Auto-connect setting
        """
        import yaml

        output_file = temp_sessions_dir / "mcp_test.yml"
        result = command_handler.handle_config_export(
            session=full_session,
            filename=str(output_file)
        )

        assert result.success is True

        with open(output_file, "r") as f:
            config = yaml.safe_load(f)

        servers = config["mcp_servers"]
        assert len(servers) == 1

        server = servers[0]
        assert server["id"] == "filesystem"
        assert server["type"] == "stdio"
        assert server["command"] == "mcp-server-filesystem"
        assert "--root" in server["args"]
        assert server["auto_connect"] is True

    def test_export_orchestration_config(self, command_handler, full_session, temp_sessions_dir):
        """Test that orchestration configuration is exported.

        Validates:
        - Execution mode
        - Auto-include history
        - Router configuration
        """
        import yaml

        output_file = temp_sessions_dir / "orchestration_test.yml"
        result = command_handler.handle_config_export(
            session=full_session,
            filename=str(output_file)
        )

        assert result.success is True

        with open(output_file, "r") as f:
            config = yaml.safe_load(f)

        orc = config["orchestration"]
        assert orc["execution_mode"] == "router"
        assert orc["auto_include_history"] is True
        assert "router_config" in orc
        assert orc["router_config"]["model"] == "openai/gpt-4o-mini"

    def test_export_history_configuration(self, command_handler, full_session, temp_sessions_dir):
        """Test that history configuration is exported.

        Validates:
        - Per-agent history overrides
        - Enabled/disabled agents
        - Token and entry limits
        """
        import yaml

        output_file = temp_sessions_dir / "history_test.yml"
        result = command_handler.handle_config_export(
            session=full_session,
            filename=str(output_file)
        )

        assert result.success is True

        with open(output_file, "r") as f:
            config = yaml.safe_load(f)

        history_config = config["history_configuration"]
        assert "per_agent" in history_config

        per_agent = history_config["per_agent"]
        assert "researcher" in per_agent
        assert per_agent["researcher"]["enabled"] is True
        assert per_agent["researcher"]["max_tokens"] == 8000

        assert "terminal" in per_agent
        assert per_agent["terminal"]["enabled"] is False

    def test_export_invalid_extension(self, command_handler, full_session):
        """Test that invalid file extensions are rejected.

        Validates:
        - .txt extension fails
        - .py extension fails
        - Error message is clear
        """
        result = command_handler.handle_config_export(
            session=full_session,
            filename="config.txt"
        )

        assert result.success is False
        assert "Invalid file extension" in result.message
        assert result.error == "Invalid file extension"

    def test_export_path_traversal_prevention(self, command_handler, full_session):
        """Test that path traversal attempts are blocked.

        Validates:
        - ../ patterns are detected
        - Relative paths with .. are blocked
        - Error message indicates security issue
        """
        result = command_handler.handle_config_export(
            session=full_session,
            filename="../etc/config.yml"
        )

        assert result.success is False
        assert "path traversal" in result.message.lower()

    def test_export_creates_parent_directories(self, command_handler, full_session, temp_sessions_dir):
        """Test that parent directories are created if needed.

        Validates:
        - Nested directory paths work
        - Parent directories are created
        - Export succeeds
        """
        output_file = temp_sessions_dir / "subdir" / "nested" / "config.yml"
        result = command_handler.handle_config_export(
            session=full_session,
            filename=str(output_file)
        )

        assert result.success is True
        assert output_file.exists()
        assert output_file.parent.exists()


class TestConfigImportCommand:
    """Test /config import command functionality (T111)."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary sessions directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        """Create SessionManager with temp directory."""
        from promptchain.cli.session_manager import SessionManager
        return SessionManager(sessions_dir=temp_sessions_dir)

    @pytest.fixture
    def command_handler(self, session_manager):
        """Create CommandHandler instance."""
        from promptchain.cli.command_handler import CommandHandler
        return CommandHandler(session_manager=session_manager)

    @pytest.fixture
    def empty_session(self, session_manager):
        """Create empty session for import testing."""
        session = session_manager.create_session(
            name="import-test",
            working_directory=Path.cwd()
        )
        return session

    @pytest.fixture
    def sample_yaml_config(self, temp_sessions_dir):
        """Create sample YAML config file."""
        import yaml

        config = {
            "version": "1.0",
            "session": {
                "name": "imported-session",
                "working_directory": str(Path.cwd()),
                "active_agent": "coder",
            },
            "agents": {
                "coder": {
                    "model": "openai/gpt-4",
                    "description": "Code generation agent",
                    "tools": ["mcp_filesystem_read", "execute_code"],
                    "history_config": {
                        "enabled": True,
                        "max_tokens": 6000,
                        "max_entries": 30,
                        "truncation_strategy": "oldest_first",
                    }
                },
                "analyst": {
                    "model": "anthropic/claude-3-sonnet-20240229",
                    "description": "Data analysis",
                    "tools": ["web_search"],
                    "history_config": {
                        "enabled": False,
                        "max_tokens": 0,
                        "max_entries": 0,
                    }
                }
            },
            "mcp_servers": [
                {
                    "id": "test-server",
                    "type": "stdio",
                    "command": "test-command",
                    "args": ["--arg1", "--arg2"],
                    "auto_connect": False,
                }
            ],
            "orchestration": {
                "execution_mode": "pipeline",
                "auto_include_history": False,
            }
        }

        config_file = temp_sessions_dir / "sample_config.yml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)

        return config_file

    def test_import_yaml_basic(self, command_handler, empty_session, sample_yaml_config):
        """Test basic YAML import functionality.

        Validates:
        - Import succeeds
        - Agents are created
        - MCP servers are added
        - Changes are persisted
        """
        result = command_handler.handle_config_import(
            session=empty_session,
            filename=str(sample_yaml_config)
        )

        # Validate success
        assert result.success is True
        assert result.error is None
        assert "Configuration imported" in result.message

        # Validate data
        assert result.data["format"] == "yaml"
        assert len(result.data["changes"]) > 0
        assert result.data["change_count"] > 0

        # Validate agents were created
        assert "coder" in empty_session.agents
        assert "analyst" in empty_session.agents

        # Validate MCP server was added
        assert len(empty_session.mcp_servers) > 0
        server_ids = [s.id for s in empty_session.mcp_servers]
        assert "test-server" in server_ids

    def test_import_json_basic(self, command_handler, empty_session, temp_sessions_dir):
        """Test basic JSON import functionality.

        Validates:
        - JSON import works
        - Configuration is applied correctly
        """
        import json

        config = {
            "version": "1.0",
            "agents": {
                "test-agent": {
                    "model": "openai/gpt-4",
                    "description": "Test agent",
                    "tools": [],
                }
            },
            "mcp_servers": [],
        }

        config_file = temp_sessions_dir / "test.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        result = command_handler.handle_config_import(
            session=empty_session,
            filename=str(config_file)
        )

        assert result.success is True
        assert result.data["format"] == "json"
        assert "test-agent" in empty_session.agents

    def test_import_creates_agents(self, command_handler, empty_session, sample_yaml_config):
        """Test that agents are created with correct configurations.

        Validates:
        - Agent models
        - Descriptions
        - Tools
        - History configurations
        """
        result = command_handler.handle_config_import(
            session=empty_session,
            filename=str(sample_yaml_config)
        )

        assert result.success is True

        # Check coder agent
        coder = empty_session.agents["coder"]
        assert coder.model_name == "openai/gpt-4"
        assert coder.description == "Code generation agent"
        assert "mcp_filesystem_read" in coder.tools
        assert coder.history_config.enabled is True
        assert coder.history_config.max_tokens == 6000

        # Check analyst agent
        analyst = empty_session.agents["analyst"]
        assert analyst.model_name == "anthropic/claude-3-sonnet-20240229"
        assert analyst.history_config.enabled is False

    def test_import_creates_mcp_servers(self, command_handler, empty_session, sample_yaml_config):
        """Test that MCP servers are created correctly.

        Validates:
        - Server ID, type, command
        - Args are preserved
        - Auto-connect setting
        """
        result = command_handler.handle_config_import(
            session=empty_session,
            filename=str(sample_yaml_config)
        )

        assert result.success is True

        # Find test server
        test_server = None
        for server in empty_session.mcp_servers:
            if server.id == "test-server":
                test_server = server
                break

        assert test_server is not None
        assert test_server.type == "stdio"
        assert test_server.command == "test-command"
        assert "--arg1" in test_server.args
        assert test_server.auto_connect is False

    def test_import_updates_orchestration(self, command_handler, empty_session, sample_yaml_config):
        """Test that orchestration settings are applied.

        Validates:
        - Execution mode is set
        - Auto-include history is set
        """
        result = command_handler.handle_config_import(
            session=empty_session,
            filename=str(sample_yaml_config)
        )

        assert result.success is True

        assert empty_session.orchestration_config.execution_mode == "pipeline"
        assert empty_session.orchestration_config.auto_include_history is False

    def test_import_skips_existing_agents(self, command_handler, empty_session, sample_yaml_config):
        """Test that existing agents are not overwritten.

        Validates:
        - Existing agents are skipped
        - Changes list shows skipped items
        - No error occurs
        """
        from promptchain.cli.models.agent_config import Agent

        # Create agent that exists in config
        empty_session.agents["coder"] = Agent(
            name="coder",
            model_name="openai/gpt-3.5-turbo",  # Different model
            description="Existing coder",
            created_at=time.time()
        )

        result = command_handler.handle_config_import(
            session=empty_session,
            filename=str(sample_yaml_config)
        )

        assert result.success is True

        # Check that existing agent was not overwritten
        assert empty_session.agents["coder"].model_name == "openai/gpt-3.5-turbo"

        # Check changes list
        changes = result.data["changes"]
        skipped_found = any("Skipped existing agent: coder" in change for change in changes)
        assert skipped_found is True

    def test_import_file_not_found(self, command_handler, empty_session):
        """Test error handling for non-existent file.

        Validates:
        - Error is returned
        - Message is clear
        """
        result = command_handler.handle_config_import(
            session=empty_session,
            filename="nonexistent.yml"
        )

        assert result.success is False
        assert "not found" in result.message
        assert result.error == "File not found"

    def test_import_invalid_extension(self, command_handler, empty_session):
        """Test error handling for invalid file extension.

        Validates:
        - .txt files are rejected
        - Error message is clear
        """
        result = command_handler.handle_config_import(
            session=empty_session,
            filename="config.txt"
        )

        assert result.success is False
        assert "Invalid file extension" in result.message

    def test_import_invalid_yaml(self, command_handler, empty_session, temp_sessions_dir):
        """Test error handling for invalid YAML syntax.

        Validates:
        - YAML parse errors are caught
        - Error message mentions YAML
        """
        bad_yaml = temp_sessions_dir / "bad.yml"
        with open(bad_yaml, "w") as f:
            f.write("invalid: yaml:\n  - no: close\n  bracket")

        result = command_handler.handle_config_import(
            session=empty_session,
            filename=str(bad_yaml)
        )

        assert result.success is False
        assert "YAML" in result.message or "parse" in result.message.lower()

    def test_import_missing_version(self, command_handler, empty_session, temp_sessions_dir):
        """Test error handling for config missing version field.

        Validates:
        - Version validation works
        - Error message is clear
        """
        import yaml

        config = {
            "agents": {},
            # Missing "version" field
        }

        config_file = temp_sessions_dir / "no_version.yml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)

        result = command_handler.handle_config_import(
            session=empty_session,
            filename=str(config_file)
        )

        assert result.success is False
        assert "version" in result.message.lower()

    def test_import_export_roundtrip(self, command_handler, session_manager, temp_sessions_dir):
        """Test export -> import roundtrip preserves configuration.

        Validates:
        - Export and then import produces equivalent config
        - All agents, servers, and settings preserved
        """
        from promptchain.cli.models.agent_config import Agent, HistoryConfig
        from promptchain.cli.models.mcp_config import MCPServerConfig

        # Create source session
        source = session_manager.create_session(
            name="source",
            working_directory=Path.cwd()
        )

        source.agents["test"] = Agent(
            name="test",
            model_name="openai/gpt-4",
            description="Test",
            tools=["tool1", "tool2"],
            history_config=HistoryConfig(enabled=True, max_tokens=5000),
            created_at=time.time()
        )

        server = MCPServerConfig(
            id="test-server",
            type="stdio",
            command="test"
        )
        source.mcp_servers = [server]

        # Export
        export_file = temp_sessions_dir / "roundtrip.yml"
        export_result = command_handler.handle_config_export(
            session=source,
            filename=str(export_file)
        )
        assert export_result.success is True

        # Create target session
        target = session_manager.create_session(
            name="target",
            working_directory=Path.cwd()
        )

        # Import
        import_result = command_handler.handle_config_import(
            session=target,
            filename=str(export_file)
        )
        assert import_result.success is True

        # Validate roundtrip
        assert "test" in target.agents
        assert target.agents["test"].model_name == "openai/gpt-4"
        assert target.agents["test"].history_config.max_tokens == 5000
        assert len(target.mcp_servers) > 0
