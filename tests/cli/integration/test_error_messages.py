"""Tests for comprehensive error messages (T105).

Verifies that all error scenarios produce user-friendly, actionable error messages
with proper context and suggestions.
"""

import pytest
from promptchain.cli.utils.error_messages import (
    # MCP Server Errors
    mcp_server_not_found,
    mcp_connection_timeout,
    mcp_auth_failed,
    mcp_tool_discovery_failed,
    mcp_unsupported_type,
    mcp_library_not_available,
    mcp_session_init_failed,
    # Router Mode Errors
    router_decision_timeout,
    router_invalid_agent,
    router_llm_failure,
    router_invalid_response,
    router_no_config,
    # Token Limit Errors
    token_limit_warning,
    token_limit_exceeded,
    token_truncation_warning,
    # Agent Creation Errors
    agent_template_not_found,
    agent_missing_tools,
    agent_invalid_model,
    agent_duplicate_name,
    agent_validation_failed,
    # Base classes
    CLIError,
    ErrorCode
)


class TestMCPServerErrors:
    """Test MCP server error messages."""

    def test_mcp_server_not_found(self):
        """Test server not found error message."""
        error = mcp_server_not_found("missing-server", ["server1", "server2"])

        assert error.error_code == ErrorCode.MCP_SERVER_NOT_FOUND
        assert "missing-server" in error.message
        assert "server1" in error.context["available_servers"]
        assert len(error.suggestions) > 0
        assert "/mcp list" in error.suggestions[1]

        # Verify formatted output includes all elements
        formatted = error.format_for_display()
        assert "Error 1000" in formatted
        assert "Context:" in formatted
        assert "Suggested fixes:" in formatted

    def test_mcp_connection_timeout(self):
        """Test connection timeout error message."""
        error = mcp_connection_timeout("slow-server", 30)

        assert error.error_code == ErrorCode.MCP_CONNECTION_TIMEOUT
        assert "30s" in error.message
        assert "timeout" in error.context
        assert any("Check if the server process is running" in s for s in error.suggestions)

    def test_mcp_auth_failed(self):
        """Test authentication failed error message."""
        error = mcp_auth_failed("secure-server", "oauth")

        assert error.error_code == ErrorCode.MCP_AUTH_FAILED
        assert "oauth" in error.context["auth_method"]
        assert any("credentials" in s.lower() for s in error.suggestions)

    def test_mcp_tool_discovery_failed(self):
        """Test tool discovery failed error message."""
        error = mcp_tool_discovery_failed("test-server", "Connection refused")

        assert error.error_code == ErrorCode.MCP_TOOL_DISCOVERY_FAILED
        assert "Connection refused" in error.context["error"]
        assert any("reconnecting" in s.lower() for s in error.suggestions)

    def test_mcp_unsupported_type(self):
        """Test unsupported server type error message."""
        error = mcp_unsupported_type("bad-server", "websocket")

        assert error.error_code == ErrorCode.MCP_UNSUPPORTED_TYPE
        assert "websocket" in error.context["server_type"]
        assert "stdio, http" in error.context["supported_types"]

    def test_mcp_library_not_available(self):
        """Test MCP library not available error message."""
        error = mcp_library_not_available()

        assert error.error_code == ErrorCode.MCP_LIBRARY_NOT_AVAILABLE
        assert "pip install" in error.suggestions[0]

    def test_mcp_session_init_failed(self):
        """Test session initialization failed error message."""
        error = mcp_session_init_failed("test-server", "Invalid arguments")

        assert error.error_code == ErrorCode.MCP_SESSION_INIT_FAILED
        assert "Invalid arguments" in error.context["error"]


class TestRouterModeErrors:
    """Test router mode error messages."""

    def test_router_decision_timeout(self):
        """Test router decision timeout error message."""
        error = router_decision_timeout(60, "Long complex input...")

        assert error.error_code == ErrorCode.ROUTER_DECISION_TIMEOUT
        assert "60s" in error.message
        assert "Long complex input" in error.context["user_input_preview"]
        assert any("faster" in s.lower() or "reduce" in s.lower() for s in error.suggestions)

    def test_router_invalid_agent(self):
        """Test router invalid agent selection error message."""
        error = router_invalid_agent("nonexistent", ["agent1", "agent2"])

        assert error.error_code == ErrorCode.ROUTER_INVALID_AGENT
        assert "nonexistent" in error.context["chosen_agent"]
        assert "agent1, agent2" in error.context["available_agents"]

    def test_router_llm_failure(self):
        """Test router LLM failure error message."""
        error = router_llm_failure("gpt-4", "Rate limit exceeded")

        assert error.error_code == ErrorCode.ROUTER_LLM_FAILURE
        assert "gpt-4" in error.context["model"]
        assert "Rate limit exceeded" in error.context["error"]
        assert any("api key" in s.lower() for s in error.suggestions)

    def test_router_invalid_response(self):
        """Test router invalid response error message."""
        error = router_invalid_response("invalid json response")

        assert error.error_code == ErrorCode.ROUTER_INVALID_RESPONSE
        assert "invalid json" in error.context["response_preview"]
        assert any("JSON" in s for s in error.suggestions)

    def test_router_no_config(self):
        """Test router missing configuration error message."""
        error = router_no_config()

        assert error.error_code == ErrorCode.ROUTER_NO_CONFIG
        assert any("supervisor" in s.lower() for s in error.suggestions)


class TestTokenLimitErrors:
    """Test token limit error messages."""

    def test_token_limit_warning(self):
        """Test token limit warning message."""
        error = token_limit_warning(3200, 4000, 80.0, "researcher")

        assert error.error_code == ErrorCode.TOKEN_LIMIT_WARNING
        assert "80.0%" in error.message
        assert error.context["current_tokens"] == 3200
        assert error.context["max_tokens"] == 4000
        assert error.context["agent"] == "researcher"
        assert any("truncated" in s.lower() for s in error.suggestions)

    def test_token_limit_exceeded(self):
        """Test token limit exceeded error message."""
        error = token_limit_exceeded(4500, 4000, 5, "coder")

        assert error.error_code == ErrorCode.TOKEN_LIMIT_EXCEEDED
        assert "truncated" in error.message
        assert error.context["entries_removed"] == 5
        assert error.context["agent"] == "coder"

    def test_token_truncation_warning(self):
        """Test token truncation warning message."""
        error = token_truncation_warning("oldest_first", 3, "max_entries limit")

        assert error.error_code == ErrorCode.TOKEN_TRUNCATION_WARNING
        assert "3 entries removed" in error.message
        assert error.context["strategy"] == "oldest_first"


class TestAgentCreationErrors:
    """Test agent creation error messages."""

    def test_agent_template_not_found(self):
        """Test agent template not found error message."""
        error = agent_template_not_found("invalid", ["researcher", "coder", "analyst"])

        assert error.error_code == ErrorCode.AGENT_TEMPLATE_NOT_FOUND
        assert "invalid" in error.context["template"]
        assert "researcher, coder, analyst" in error.context["available"]
        assert any("/agent list-templates" in s for s in error.suggestions)

    def test_agent_missing_tools(self):
        """Test agent missing tools error message."""
        error = agent_missing_tools("my-agent", ["tool1", "tool2"])

        assert error.error_code == ErrorCode.AGENT_MISSING_TOOLS
        assert "my-agent" in error.context["agent"]
        assert "tool1, tool2" in error.context["missing_tools"]

    def test_agent_invalid_model(self):
        """Test agent invalid model error message."""
        error = agent_invalid_model("my-agent", "invalid-model", "Invalid format")

        assert error.error_code == ErrorCode.AGENT_INVALID_MODEL
        assert "invalid-model" in error.context["model_spec"]
        assert "Invalid format" in error.context["error"]
        assert any("provider/model-name" in s for s in error.suggestions)

    def test_agent_duplicate_name(self):
        """Test agent duplicate name error message."""
        error = agent_duplicate_name("existing-agent")

        assert error.error_code == ErrorCode.AGENT_DUPLICATE_NAME
        assert "existing-agent" in error.context["agent"]
        assert any("/agent update" in s for s in error.suggestions)

    def test_agent_validation_failed(self):
        """Test agent validation failed error message."""
        error = agent_validation_failed("bad-agent", ["Name too short", "Invalid characters"])

        assert error.error_code == ErrorCode.AGENT_VALIDATION_FAILED
        assert "bad-agent" in error.context["agent"]
        assert "Name too short; Invalid characters" in error.context["validation_errors"]


class TestCLIErrorBase:
    """Test CLIError base class functionality."""

    def test_cli_error_initialization(self):
        """Test CLIError initialization."""
        error = CLIError(
            message="Test error",
            error_code=ErrorCode.MCP_SERVER_NOT_FOUND,
            context={"key": "value"},
            suggestions=["Fix 1", "Fix 2"]
        )

        assert error.message == "Test error"
        assert error.error_code == ErrorCode.MCP_SERVER_NOT_FOUND
        assert error.context == {"key": "value"}
        assert error.suggestions == ["Fix 1", "Fix 2"]

    def test_cli_error_format_for_display(self):
        """Test CLIError formatting."""
        error = CLIError(
            message="Test error",
            error_code=ErrorCode.MCP_SERVER_NOT_FOUND,
            context={"server": "test"},
            suggestions=["Fix it"]
        )

        formatted = error.format_for_display()

        # Verify all components present
        assert "Error 1000" in formatted
        assert "Test error" in formatted
        assert "Context:" in formatted
        assert "server: test" in formatted
        assert "Suggested fixes:" in formatted
        assert "• Fix it" in formatted

    def test_cli_error_no_context_or_suggestions(self):
        """Test CLIError with minimal information."""
        error = CLIError(
            message="Simple error",
            error_code=ErrorCode.MCP_SERVER_NOT_FOUND
        )

        formatted = error.format_for_display()

        # Should still display without context/suggestions
        assert "Error 1000" in formatted
        assert "Simple error" in formatted
        # Context and suggestions sections should be absent or empty
        assert formatted.count("Context:") == 0 or "Context:\n\n" in formatted


class TestErrorCodes:
    """Test error code enumeration."""

    def test_error_codes_unique(self):
        """Test that all error codes are unique."""
        codes = [e.value for e in ErrorCode]
        assert len(codes) == len(set(codes)), "Error codes must be unique"

    def test_error_code_ranges(self):
        """Test error code ranges."""
        # MCP errors: 1000-1099
        mcp_codes = [e.value for e in ErrorCode if 1000 <= e.value < 1100]
        assert len(mcp_codes) > 0

        # Router errors: 2000-2099
        router_codes = [e.value for e in ErrorCode if 2000 <= e.value < 2100]
        assert len(router_codes) > 0

        # Token errors: 3000-3099
        token_codes = [e.value for e in ErrorCode if 3000 <= e.value < 3100]
        assert len(token_codes) > 0

        # Agent errors: 4000-4099
        agent_codes = [e.value for e in ErrorCode if 4000 <= e.value < 4100]
        assert len(agent_codes) > 0


class TestErrorIntegration:
    """Test error message integration scenarios."""

    def test_error_message_completeness(self):
        """Test that all error messages have required components."""
        # Sample a few error messages from each category
        errors = [
            mcp_server_not_found("test", ["a", "b"]),
            router_decision_timeout(30, "input"),
            token_limit_warning(100, 1000, 10.0),
            agent_template_not_found("test", ["a", "b"])
        ]

        for error in errors:
            # Every error must have:
            assert error.message  # Non-empty message
            assert error.error_code  # Valid error code
            assert isinstance(error.context, dict)  # Context dictionary
            assert isinstance(error.suggestions, list)  # Suggestions list
            assert len(error.suggestions) > 0  # At least one suggestion

            # Formatted output must be informative
            formatted = error.format_for_display()
            assert len(formatted) > 50  # Reasonably detailed
            assert str(error.error_code.value) in formatted  # Error code visible
