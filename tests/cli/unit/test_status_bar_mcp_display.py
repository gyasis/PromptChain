"""Unit tests for MCP server status display in TUI StatusBar (T069)."""

import pytest
from promptchain.cli.tui.status_bar import StatusBar


class TestStatusBarMCPDisplay:
    """Unit tests for MCP server status display in StatusBar widget."""

    def test_status_bar_displays_no_mcp_servers_by_default(self):
        """StatusBar displays no MCP status when no servers configured."""
        status_bar = StatusBar()

        # No MCP servers
        assert status_bar.mcp_servers == []

        # Render should not include MCP status
        rendered = status_bar.render()
        assert "MCP:" not in rendered

    def test_status_bar_displays_connected_mcp_server(self):
        """StatusBar displays connected MCP server with green checkmark."""
        status_bar = StatusBar()

        # Update with connected server
        status_bar.update_session_info(
            mcp_servers=[{"id": "filesystem", "state": "connected"}]
        )

        # Render should include MCP status with green checkmark
        rendered = status_bar.render()
        assert "MCP:" in rendered
        assert "filesystem" in rendered
        assert "[green]✓[/green]" in rendered

    def test_status_bar_displays_disconnected_mcp_server(self):
        """StatusBar displays disconnected MCP server with dim circle."""
        status_bar = StatusBar()

        # Update with disconnected server
        status_bar.update_session_info(
            mcp_servers=[{"id": "calculator", "state": "disconnected"}]
        )

        # Render should include MCP status with dim circle
        rendered = status_bar.render()
        assert "MCP:" in rendered
        assert "calculator" in rendered
        assert "[dim]○[/dim]" in rendered

    def test_status_bar_displays_error_mcp_server(self):
        """StatusBar displays error MCP server with red X."""
        status_bar = StatusBar()

        # Update with error server
        status_bar.update_session_info(
            mcp_servers=[{"id": "failing_server", "state": "error"}]
        )

        # Render should include MCP status with red X
        rendered = status_bar.render()
        assert "MCP:" in rendered
        assert "failing_server" in rendered
        assert "[red]✗[/red]" in rendered

    def test_status_bar_displays_multiple_mcp_servers(self):
        """StatusBar displays multiple MCP servers with different states."""
        status_bar = StatusBar()

        # Update with multiple servers
        status_bar.update_session_info(
            mcp_servers=[
                {"id": "filesystem", "state": "connected"},
                {"id": "calculator", "state": "disconnected"},
                {"id": "failing_server", "state": "error"},
            ]
        )

        # Render should include all servers
        rendered = status_bar.render()
        assert "MCP:" in rendered
        assert "filesystem" in rendered
        assert "calculator" in rendered
        assert "failing_server" in rendered

        # Verify emoji indicators
        assert "[green]✓[/green]" in rendered
        assert "[dim]○[/dim]" in rendered
        assert "[red]✗[/red]" in rendered

    def test_status_bar_updates_mcp_server_status(self):
        """StatusBar can update MCP server status dynamically."""
        status_bar = StatusBar()

        # Initially connected
        status_bar.update_session_info(
            mcp_servers=[{"id": "filesystem", "state": "connected"}]
        )
        rendered = status_bar.render()
        assert "[green]✓[/green]" in rendered

        # Update to error state
        status_bar.update_session_info(
            mcp_servers=[{"id": "filesystem", "state": "error"}]
        )
        rendered = status_bar.render()
        assert "[red]✗[/red]" in rendered
        assert "[green]✓[/green]" not in rendered

    def test_status_bar_preserves_other_fields_when_updating_mcp(self):
        """Updating MCP servers preserves other status bar fields."""
        status_bar = StatusBar()

        # Set initial state
        status_bar.update_session_info(
            session_name="test-session",
            active_agent="test_agent",
            model_name="gpt-4.1-mini-2025-04-14",
            message_count=5,
        )

        # Update MCP servers
        status_bar.update_session_info(
            mcp_servers=[{"id": "filesystem", "state": "connected"}]
        )

        # All fields should still be present
        rendered = status_bar.render()
        assert "test-session" in rendered
        assert "test_agent" in rendered
        assert "gpt-4.1-mini-2025-04-14" in rendered
        assert "Messages: 5" in rendered
        assert "MCP:" in rendered
        assert "filesystem" in rendered

    def test_status_bar_mcp_display_format(self):
        """MCP server status display format matches specification."""
        status_bar = StatusBar()

        # Update with multiple servers
        status_bar.update_session_info(
            mcp_servers=[
                {"id": "fs", "state": "connected"},
                {"id": "calc", "state": "error"},
            ]
        )

        # Verify format: "MCP: ✓fs ✗calc"
        rendered = status_bar.render()
        assert "MCP:" in rendered

        # Extract MCP portion
        mcp_part = rendered.split("MCP: ")[1].split(" | ")[0] if "MCP: " in rendered else ""
        assert "fs" in mcp_part
        assert "calc" in mcp_part
