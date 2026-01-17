"""Comprehensive error message catalog for CLI operations.

This module provides user-friendly, actionable error messages for common
failure scenarios with consistent formatting and error codes.
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(Enum):
    """Error codes for programmatic handling."""

    # MCP Server Errors (1000-1099)
    MCP_SERVER_NOT_FOUND = 1000
    MCP_CONNECTION_TIMEOUT = 1001
    MCP_AUTH_FAILED = 1002
    MCP_TOOL_DISCOVERY_FAILED = 1003
    MCP_UNSUPPORTED_TYPE = 1004
    MCP_SERVER_CRASHED = 1005
    MCP_LIBRARY_NOT_AVAILABLE = 1006
    MCP_SESSION_INIT_FAILED = 1007

    # Router Mode Errors (2000-2099)
    ROUTER_DECISION_TIMEOUT = 2000
    ROUTER_INVALID_AGENT = 2001
    ROUTER_LLM_FAILURE = 2002
    ROUTER_INVALID_RESPONSE = 2003
    ROUTER_STRATEGY_MISMATCH = 2004
    ROUTER_NO_CONFIG = 2005

    # Token Limit Errors (3000-3099)
    TOKEN_LIMIT_WARNING = 3000
    TOKEN_LIMIT_EXCEEDED = 3001
    TOKEN_TRUNCATION_WARNING = 3002
    TOKEN_UTILIZATION_HIGH = 3003

    # Agent Creation Errors (4000-4099)
    AGENT_TEMPLATE_NOT_FOUND = 4000
    AGENT_MISSING_TOOLS = 4001
    AGENT_INVALID_MODEL = 4002
    AGENT_DUPLICATE_NAME = 4003
    AGENT_VALIDATION_FAILED = 4004


class CLIError(Exception):
    """Base exception for CLI errors with user-friendly messages."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[list[str]] = None
    ):
        """Initialize CLI error.

        Args:
            message: User-friendly error message
            error_code: Error code for programmatic handling
            context: Additional context about the error
            suggestions: List of suggested fixes
        """
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.suggestions = suggestions or []
        super().__init__(message)

    def format_for_display(self) -> str:
        """Format error for CLI display with suggestions."""
        lines = [
            f"Error {self.error_code.value}: {self.message}",
            ""
        ]

        # Add context if available
        if self.context:
            lines.append("Context:")
            for key, value in self.context.items():
                lines.append(f"  - {key}: {value}")
            lines.append("")

        # Add suggestions
        if self.suggestions:
            lines.append("Suggested fixes:")
            for suggestion in self.suggestions:
                lines.append(f"  • {suggestion}")

        return "\n".join(lines)


# === MCP Server Error Messages ===

def mcp_server_not_found(server_id: str, available_servers: list[str]) -> CLIError:
    """MCP server not found in session."""
    return CLIError(
        message=f"MCP server '{server_id}' not found in session",
        error_code=ErrorCode.MCP_SERVER_NOT_FOUND,
        context={
            "server_id": server_id,
            "available_servers": ", ".join(available_servers) if available_servers else "none"
        },
        suggestions=[
            "Check the server ID spelling",
            f"Use /mcp list to see available servers",
            "Add the server to your session configuration"
        ]
    )


def mcp_connection_timeout(server_id: str, timeout_seconds: int) -> CLIError:
    """MCP server connection timed out."""
    return CLIError(
        message=f"Connection to MCP server '{server_id}' timed out after {timeout_seconds}s",
        error_code=ErrorCode.MCP_CONNECTION_TIMEOUT,
        context={
            "server_id": server_id,
            "timeout": f"{timeout_seconds}s"
        },
        suggestions=[
            "Check if the server process is running",
            "Verify network connectivity if using remote server",
            "Increase timeout in MCP server configuration",
            f"Try connecting again with /mcp connect {server_id}"
        ]
    )


def mcp_auth_failed(server_id: str, auth_method: str) -> CLIError:
    """MCP server authentication failed."""
    return CLIError(
        message=f"Authentication failed for MCP server '{server_id}' (method: {auth_method})",
        error_code=ErrorCode.MCP_AUTH_FAILED,
        context={
            "server_id": server_id,
            "auth_method": auth_method
        },
        suggestions=[
            "Verify authentication credentials in server configuration",
            "Check environment variables for API keys",
            "Ensure server accepts the authentication method",
            "Review server logs for authentication errors"
        ]
    )


def mcp_tool_discovery_failed(server_id: str, error_details: str) -> CLIError:
    """MCP tool discovery failed."""
    return CLIError(
        message=f"Failed to discover tools on MCP server '{server_id}'",
        error_code=ErrorCode.MCP_TOOL_DISCOVERY_FAILED,
        context={
            "server_id": server_id,
            "error": error_details
        },
        suggestions=[
            "Verify server is running and responding",
            "Check server logs for tool registration errors",
            "Ensure server implements tool listing endpoint",
            "Try reconnecting to the server"
        ]
    )


def mcp_unsupported_type(server_id: str, server_type: str) -> CLIError:
    """MCP server type not supported."""
    return CLIError(
        message=f"MCP server '{server_id}' has unsupported type '{server_type}'",
        error_code=ErrorCode.MCP_UNSUPPORTED_TYPE,
        context={
            "server_id": server_id,
            "server_type": server_type,
            "supported_types": "stdio, http"
        },
        suggestions=[
            "Use 'stdio' type for local command-line servers",
            "Use 'http' type for remote HTTP servers",
            "Check server configuration syntax",
            "Review MCP server documentation"
        ]
    )


def mcp_library_not_available() -> CLIError:
    """MCP library not installed."""
    return CLIError(
        message="MCP library not available - MCP features are disabled",
        error_code=ErrorCode.MCP_LIBRARY_NOT_AVAILABLE,
        context={
            "required_package": "mcp"
        },
        suggestions=[
            "Install MCP support: pip install mcp",
            "Or install full dependencies: pip install promptchain[mcp]",
            "Restart CLI after installation"
        ]
    )


def mcp_session_init_failed(server_id: str, error_details: str) -> CLIError:
    """MCP session initialization failed."""
    return CLIError(
        message=f"Failed to initialize MCP session for server '{server_id}'",
        error_code=ErrorCode.MCP_SESSION_INIT_FAILED,
        context={
            "server_id": server_id,
            "error": error_details
        },
        suggestions=[
            "Check if server command is valid and executable",
            "Verify server arguments are correct",
            "Review server logs for initialization errors",
            "Try a different MCP server to isolate the issue"
        ]
    )


# === Router Mode Error Messages ===

def router_decision_timeout(timeout_seconds: int, user_input: str) -> CLIError:
    """Router decision making timed out."""
    return CLIError(
        message=f"Router decision timed out after {timeout_seconds}s",
        error_code=ErrorCode.ROUTER_DECISION_TIMEOUT,
        context={
            "timeout": f"{timeout_seconds}s",
            "user_input_preview": user_input[:100] + "..." if len(user_input) > 100 else user_input
        },
        suggestions=[
            "Simplify the input or break into smaller requests",
            "Reduce conversation history size",
            "Use a faster LLM model for routing decisions",
            "Increase router timeout in configuration"
        ]
    )


def router_invalid_agent(chosen_agent: str, available_agents: list[str]) -> CLIError:
    """Router selected invalid agent."""
    return CLIError(
        message=f"Router selected invalid agent '{chosen_agent}'",
        error_code=ErrorCode.ROUTER_INVALID_AGENT,
        context={
            "chosen_agent": chosen_agent,
            "available_agents": ", ".join(available_agents)
        },
        suggestions=[
            "Check router decision logic for agent name validation",
            "Verify agent descriptions are clear and distinct",
            "Review router prompt template for clarity",
            f"Valid agents: {', '.join(available_agents)}"
        ]
    )


def router_llm_failure(model_name: str, error_details: str) -> CLIError:
    """Router LLM call failed."""
    return CLIError(
        message=f"Router LLM call failed (model: {model_name})",
        error_code=ErrorCode.ROUTER_LLM_FAILURE,
        context={
            "model": model_name,
            "error": error_details
        },
        suggestions=[
            "Verify LLM API key is configured correctly",
            "Check network connectivity to LLM provider",
            "Try a different model for routing",
            "Review rate limits and quota usage",
            "Check if model name is correct"
        ]
    )


def router_invalid_response(response_text: str) -> CLIError:
    """Router returned invalid JSON response."""
    return CLIError(
        message="Router returned invalid response format",
        error_code=ErrorCode.ROUTER_INVALID_RESPONSE,
        context={
            "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text
        },
        suggestions=[
            "Router must return JSON: {\"chosen_agent\": \"agent_name\"}",
            "Review router prompt template for clarity",
            "Use a more capable LLM model for routing",
            "Check for JSON formatting errors in response"
        ]
    )


def router_no_config() -> CLIError:
    """Router mode requires configuration."""
    return CLIError(
        message="Router mode requires router configuration",
        error_code=ErrorCode.ROUTER_NO_CONFIG,
        context={},
        suggestions=[
            "Provide 'router' parameter (Dict or Callable)",
            "Or set use_supervisor=True to use OrchestratorSupervisor",
            "Review AgentChain documentation for router configuration",
            "Use a different execution mode (pipeline, round_robin, broadcast)"
        ]
    )


# === Token Limit Error Messages ===

def token_limit_warning(
    current_tokens: int,
    max_tokens: int,
    utilization_pct: float,
    agent_name: Optional[str] = None
) -> CLIError:
    """Warning: approaching token limit."""
    agent_context = f" for agent '{agent_name}'" if agent_name else ""
    return CLIError(
        message=f"History approaching token limit{agent_context}: {utilization_pct:.1f}% used",
        error_code=ErrorCode.TOKEN_LIMIT_WARNING,
        context={
            "current_tokens": current_tokens,
            "max_tokens": max_tokens,
            "utilization_pct": f"{utilization_pct:.1f}%",
            "agent": agent_name or "global"
        },
        suggestions=[
            "History will be automatically truncated when limit is reached",
            f"Increase max_tokens in agent configuration (current: {max_tokens})",
            "Use /history stats to review token usage",
            "Clear old messages with /history clear if needed"
        ]
    )


def token_limit_exceeded(
    current_tokens: int,
    max_tokens: int,
    entries_removed: int,
    agent_name: Optional[str] = None
) -> CLIError:
    """Token limit exceeded - history truncated."""
    agent_context = f" for agent '{agent_name}'" if agent_name else ""
    return CLIError(
        message=f"Token limit exceeded{agent_context} - history truncated",
        error_code=ErrorCode.TOKEN_LIMIT_EXCEEDED,
        context={
            "tokens_before": current_tokens,
            "max_tokens": max_tokens,
            "entries_removed": entries_removed,
            "agent": agent_name or "global"
        },
        suggestions=[
            f"{entries_removed} oldest entries removed to stay within limit",
            f"Increase max_tokens in configuration (current: {max_tokens})",
            "Use agent_history_configs to customize per-agent limits",
            "Consider disabling history for agents that don't need it"
        ]
    )


def token_truncation_warning(
    truncation_strategy: str,
    entries_removed: int,
    reason: str
) -> CLIError:
    """History truncation occurred."""
    return CLIError(
        message=f"History truncated: {entries_removed} entries removed ({reason})",
        error_code=ErrorCode.TOKEN_TRUNCATION_WARNING,
        context={
            "strategy": truncation_strategy,
            "entries_removed": entries_removed,
            "reason": reason
        },
        suggestions=[
            f"Truncation strategy: {truncation_strategy}",
            "Use 'keep_last' strategy to preserve recent messages",
            "Use 'oldest_first' strategy for chronological truncation",
            "Adjust max_tokens or max_entries to reduce truncation frequency"
        ]
    )


# === Agent Creation Error Messages ===

def agent_template_not_found(template_name: str, available_templates: list[str]) -> CLIError:
    """Agent template not found."""
    return CLIError(
        message=f"Agent template '{template_name}' not found",
        error_code=ErrorCode.AGENT_TEMPLATE_NOT_FOUND,
        context={
            "template": template_name,
            "available": ", ".join(available_templates)
        },
        suggestions=[
            "Use /agent list-templates to see available templates",
            f"Available templates: {', '.join(available_templates)}",
            "Check template name spelling",
            "Create custom agent with /agent create instead"
        ]
    )


def agent_missing_tools(agent_name: str, missing_tools: list[str]) -> CLIError:
    """Agent missing required tools."""
    return CLIError(
        message=f"Agent '{agent_name}' is missing required tools",
        error_code=ErrorCode.AGENT_MISSING_TOOLS,
        context={
            "agent": agent_name,
            "missing_tools": ", ".join(missing_tools)
        },
        suggestions=[
            f"Register missing tools: {', '.join(missing_tools)}",
            "Use /tools list to see available tools",
            "Connect MCP servers to provide missing tools",
            "Update agent configuration to remove tool dependencies"
        ]
    )


def agent_invalid_model(agent_name: str, model_spec: str, error_details: str) -> CLIError:
    """Agent has invalid model specification."""
    return CLIError(
        message=f"Invalid model specification for agent '{agent_name}': {model_spec}",
        error_code=ErrorCode.AGENT_INVALID_MODEL,
        context={
            "agent": agent_name,
            "model_spec": model_spec,
            "error": error_details
        },
        suggestions=[
            "Use format: provider/model-name (e.g., openai/gpt-4)",
            "Supported providers: openai, anthropic, google, ollama",
            "Verify model name spelling",
            "Check if API key for provider is configured"
        ]
    )


def agent_duplicate_name(agent_name: str) -> CLIError:
    """Agent name already exists."""
    return CLIError(
        message=f"Agent '{agent_name}' already exists in this session",
        error_code=ErrorCode.AGENT_DUPLICATE_NAME,
        context={
            "agent": agent_name
        },
        suggestions=[
            "Choose a different agent name",
            f"Use /agent update {agent_name} to modify existing agent",
            f"Delete existing agent with /agent delete {agent_name}",
            "Use /agent list to see all agents"
        ]
    )


def agent_validation_failed(agent_name: str, validation_errors: list[str]) -> CLIError:
    """Agent validation failed."""
    return CLIError(
        message=f"Agent '{agent_name}' failed validation",
        error_code=ErrorCode.AGENT_VALIDATION_FAILED,
        context={
            "agent": agent_name,
            "validation_errors": "; ".join(validation_errors)
        },
        suggestions=[
            "Review validation errors above",
            "Check agent configuration syntax",
            "Verify all required fields are provided",
            "See agent creation documentation for requirements"
        ]
    )
