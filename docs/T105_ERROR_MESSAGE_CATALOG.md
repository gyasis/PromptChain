# T105: Comprehensive Error Message Catalog

## Overview

This document catalogs all user-friendly error messages implemented in the CLI with their error codes, contexts, and suggested fixes.

## Error Message Design Principles

1. **Clear and Actionable**: Tell the user exactly what went wrong and what they can do about it
2. **Contextual**: Provide relevant information about the operation that failed
3. **Suggestive**: Always include specific suggestions for resolution
4. **Consistent Format**: All errors follow the same structure for predictability

## Error Message Format

```
Error {code}: {message}

Context:
  - {key}: {value}
  - ...

Suggested fixes:
  • {suggestion 1}
  • {suggestion 2}
  • ...
```

## Error Code Ranges

- **1000-1099**: MCP Server Errors
- **2000-2099**: Router Mode Errors
- **3000-3099**: Token Limit Errors
- **4000-4099**: Agent Creation Errors

---

## MCP Server Errors (1000-1099)

### 1000: MCP_SERVER_NOT_FOUND

**Trigger**: Attempting to connect to an MCP server that doesn't exist in the session

**Example**:
```
Error 1000: MCP server 'missing-server' not found in session

Context:
  - server_id: missing-server
  - available_servers: server1, server2

Suggested fixes:
  • Check the server ID spelling
  • Use /mcp list to see available servers
  • Add the server to your session configuration
```

**Usage**:
```python
from promptchain.cli.utils.error_messages import mcp_server_not_found

raise mcp_server_not_found("missing-server", ["server1", "server2"])
```

---

### 1001: MCP_CONNECTION_TIMEOUT

**Trigger**: MCP server connection exceeds timeout limit

**Example**:
```
Error 1001: Connection to MCP server 'slow-server' timed out after 30s

Context:
  - server_id: slow-server
  - timeout: 30s

Suggested fixes:
  • Check if the server process is running
  • Verify network connectivity if using remote server
  • Increase timeout in MCP server configuration
  • Try connecting again with /mcp connect slow-server
```

---

### 1002: MCP_AUTH_FAILED

**Trigger**: MCP server authentication fails

**Example**:
```
Error 1002: Authentication failed for MCP server 'secure-server' (method: oauth)

Context:
  - server_id: secure-server
  - auth_method: oauth

Suggested fixes:
  • Verify authentication credentials in server configuration
  • Check environment variables for API keys
  • Ensure server accepts the authentication method
  • Review server logs for authentication errors
```

---

### 1003: MCP_TOOL_DISCOVERY_FAILED

**Trigger**: Failed to discover tools after connecting to MCP server

**Example**:
```
Error 1003: Failed to discover tools on MCP server 'test-server'

Context:
  - server_id: test-server
  - error: Connection refused

Suggested fixes:
  • Verify server is running and responding
  • Check server logs for tool registration errors
  • Ensure server implements tool listing endpoint
  • Try reconnecting to the server
```

---

### 1004: MCP_UNSUPPORTED_TYPE

**Trigger**: MCP server configuration uses unsupported server type

**Example**:
```
Error 1004: MCP server 'bad-server' has unsupported type 'websocket'

Context:
  - server_id: bad-server
  - server_type: websocket
  - supported_types: stdio, http

Suggested fixes:
  • Use 'stdio' type for local command-line servers
  • Use 'http' type for remote HTTP servers
  • Check server configuration syntax
  • Review MCP server documentation
```

---

### 1006: MCP_LIBRARY_NOT_AVAILABLE

**Trigger**: MCP library not installed but MCP features requested

**Example**:
```
Error 1006: MCP library not available - MCP features are disabled

Context:
  - required_package: mcp

Suggested fixes:
  • Install MCP support: pip install mcp
  • Or install full dependencies: pip install promptchain[mcp]
  • Restart CLI after installation
```

---

### 1007: MCP_SESSION_INIT_FAILED

**Trigger**: Failed to initialize MCP session

**Example**:
```
Error 1007: Failed to initialize MCP session for server 'test-server'

Context:
  - server_id: test-server
  - error: ValueError: Invalid arguments

Suggested fixes:
  • Check if server command is valid and executable
  • Verify server arguments are correct
  • Review server logs for initialization errors
  • Try a different MCP server to isolate the issue
```

---

## Router Mode Errors (2000-2099)

### 2000: ROUTER_DECISION_TIMEOUT

**Trigger**: Router LLM decision-making exceeds timeout

**Example**:
```
Error 2000: Router decision timed out after 60s

Context:
  - timeout: 60s
  - user_input_preview: Long complex input requiring multi-step analysis...

Suggested fixes:
  • Simplify the input or break into smaller requests
  • Reduce conversation history size
  • Use a faster LLM model for routing decisions
  • Increase router timeout in configuration
```

---

### 2001: ROUTER_INVALID_AGENT

**Trigger**: Router selects an agent name that doesn't exist

**Example**:
```
Error 2001: Router selected invalid agent 'nonexistent'

Context:
  - chosen_agent: nonexistent
  - available_agents: agent1, agent2, agent3

Suggested fixes:
  • Check router decision logic for agent name validation
  • Verify agent descriptions are clear and distinct
  • Review router prompt template for clarity
  • Valid agents: agent1, agent2, agent3
```

---

### 2002: ROUTER_LLM_FAILURE

**Trigger**: Router LLM call fails (API error, rate limit, etc.)

**Example**:
```
Error 2002: Router LLM call failed (model: gpt-4)

Context:
  - model: gpt-4
  - error: Rate limit exceeded

Suggested fixes:
  • Verify LLM API key is configured correctly
  • Check network connectivity to LLM provider
  • Try a different model for routing
  • Review rate limits and quota usage
  • Check if model name is correct
```

---

### 2003: ROUTER_INVALID_RESPONSE

**Trigger**: Router returns non-JSON or malformed response

**Example**:
```
Error 2003: Router returned invalid response format

Context:
  - response_preview: I think agent1 would be best for this task...

Suggested fixes:
  • Router must return JSON: {"chosen_agent": "agent_name"}
  • Review router prompt template for clarity
  • Use a more capable LLM model for routing
  • Check for JSON formatting errors in response
```

---

### 2005: ROUTER_NO_CONFIG

**Trigger**: Router mode selected but no router configuration provided

**Example**:
```
Error 2005: Router mode requires router configuration

Suggested fixes:
  • Provide 'router' parameter (Dict or Callable)
  • Or set use_supervisor=True to use OrchestratorSupervisor
  • Review AgentChain documentation for router configuration
  • Use a different execution mode (pipeline, round_robin, broadcast)
```

---

## Token Limit Errors (3000-3099)

### 3000: TOKEN_LIMIT_WARNING

**Trigger**: History token count reaches 80% of maximum (warning threshold)

**Example**:
```
Error 3000: History approaching token limit for agent 'researcher': 80.0% used

Context:
  - current_tokens: 3200
  - max_tokens: 4000
  - utilization_pct: 80.0%
  - agent: researcher

Suggested fixes:
  • History will be automatically truncated when limit is reached
  • Increase max_tokens in agent configuration (current: 4000)
  • Use /history stats to review token usage
  • Clear old messages with /history clear if needed
```

---

### 3001: TOKEN_LIMIT_EXCEEDED

**Trigger**: History token count exceeds maximum and truncation occurs

**Example**:
```
Error 3001: Token limit exceeded for agent 'coder' - history truncated

Context:
  - tokens_before: 4500
  - max_tokens: 4000
  - entries_removed: 5
  - agent: coder

Suggested fixes:
  • 5 oldest entries removed to stay within limit
  • Increase max_tokens in configuration (current: 4000)
  • Use agent_history_configs to customize per-agent limits
  • Consider disabling history for agents that don't need it
```

---

### 3002: TOKEN_TRUNCATION_WARNING

**Trigger**: History truncated due to max_entries or other limit

**Example**:
```
Error 3002: History truncated: 3 entries removed (max_entries limit)

Context:
  - strategy: oldest_first
  - entries_removed: 3
  - reason: max_entries limit

Suggested fixes:
  • Truncation strategy: oldest_first
  • Use 'keep_last' strategy to preserve recent messages
  • Use 'oldest_first' strategy for chronological truncation
  • Adjust max_tokens or max_entries to reduce truncation frequency
```

---

## Agent Creation Errors (4000-4099)

### 4000: AGENT_TEMPLATE_NOT_FOUND

**Trigger**: Attempting to create agent from non-existent template

**Example**:
```
Error 4000: Agent template 'invalid' not found

Context:
  - template: invalid
  - available: researcher, coder, analyst, writer

Suggested fixes:
  • Use /agent list-templates to see available templates
  • Available templates: researcher, coder, analyst, writer
  • Check template name spelling
  • Create custom agent with /agent create instead
```

---

### 4001: AGENT_MISSING_TOOLS

**Trigger**: Agent requires tools that are not available

**Example**:
```
Error 4001: Agent 'my-agent' is missing required tools

Context:
  - agent: my-agent
  - missing_tools: search_tool, database_tool

Suggested fixes:
  • Register missing tools: search_tool, database_tool
  • Use /tools list to see available tools
  • Connect MCP servers to provide missing tools
  • Update agent configuration to remove tool dependencies
```

---

### 4002: AGENT_INVALID_MODEL

**Trigger**: Agent model specification is invalid or unsupported

**Example**:
```
Error 4002: Invalid model specification for agent 'my-agent': invalid-model

Context:
  - agent: my-agent
  - model_spec: invalid-model
  - error: Invalid format

Suggested fixes:
  • Use format: provider/model-name (e.g., openai/gpt-4)
  • Supported providers: openai, anthropic, google, ollama
  • Verify model name spelling
  • Check if API key for provider is configured
```

---

### 4003: AGENT_DUPLICATE_NAME

**Trigger**: Attempting to create agent with name that already exists

**Example**:
```
Error 4003: Agent 'existing-agent' already exists in this session

Context:
  - agent: existing-agent

Suggested fixes:
  • Choose a different agent name
  • Use /agent update existing-agent to modify existing agent
  • Delete existing agent with /agent delete existing-agent
  • Use /agent list to see all agents
```

---

### 4004: AGENT_VALIDATION_FAILED

**Trigger**: Agent configuration fails validation

**Example**:
```
Error 4004: Agent 'bad-agent' failed validation

Context:
  - agent: bad-agent
  - validation_errors: Name too short; Invalid characters

Suggested fixes:
  • Review validation errors above
  • Check agent configuration syntax
  • Verify all required fields are provided
  • See agent creation documentation for requirements
```

---

## Usage Examples

### In Application Code

```python
from promptchain.cli.utils.error_messages import (
    mcp_server_not_found,
    router_invalid_agent,
    token_limit_warning,
    CLIError
)

# Example 1: MCP server not found
try:
    server_config = find_server("missing-server")
except:
    raise mcp_server_not_found("missing-server", available_servers)

# Example 2: Router invalid agent
if chosen_agent not in available_agents:
    raise router_invalid_agent(chosen_agent, list(available_agents.keys()))

# Example 3: Token limit warning (logged, not raised)
if utilization >= 80:
    warning = token_limit_warning(current, max, utilization, agent_name)
    logger.warning(warning.format_for_display())

# Example 4: Custom CLI error
raise CLIError(
    message="Custom error occurred",
    error_code=ErrorCode.CUSTOM_ERROR,
    context={"detail": "value"},
    suggestions=["Try this", "Or that"]
)
```

### In Tests

```python
from promptchain.cli.utils.error_messages import (
    mcp_server_not_found,
    ErrorCode
)

def test_mcp_server_not_found_error():
    error = mcp_server_not_found("test", ["a", "b"])

    # Verify error code
    assert error.error_code == ErrorCode.MCP_SERVER_NOT_FOUND

    # Verify context
    assert "test" in error.context["server_id"]

    # Verify suggestions exist
    assert len(error.suggestions) > 0

    # Verify formatted output
    formatted = error.format_for_display()
    assert "Error 1000" in formatted
```

---

## Implementation Locations

### Error Message Catalog
- **File**: `/home/gyasis/Documents/code/PromptChain/promptchain/cli/utils/error_messages.py`
- **Classes**: `CLIError`, `ErrorCode`
- **Functions**: All error message factory functions

### Integration Points

1. **MCP Manager** (`promptchain/cli/utils/mcp_manager.py`):
   - Server connection errors (1000-1007)
   - Tool discovery failures
   - Timeout handling

2. **Command Handler** (`promptchain/cli/command_handler.py`):
   - Agent creation errors (4000-4004)
   - Template validation
   - Duplicate name detection

3. **Execution History Manager** (`promptchain/utils/execution_history_manager.py`):
   - Token limit warnings (3000-3002)
   - Truncation notifications
   - Memory management alerts

4. **Agent Chain** (`promptchain/utils/agent_chain.py`):
   - Router mode errors (2000-2005)
   - Agent selection failures
   - LLM communication errors

---

## Testing

### Test Coverage
- **File**: `/home/gyasis/Documents/code/PromptChain/tests/cli/integration/test_error_messages.py`
- **Test Classes**:
  - `TestMCPServerErrors`: 7 tests
  - `TestRouterModeErrors`: 5 tests
  - `TestTokenLimitErrors`: 3 tests
  - `TestAgentCreationErrors`: 5 tests
  - `TestCLIErrorBase`: 3 tests
  - `TestErrorCodes`: 2 tests
  - `TestErrorIntegration`: 1 test

### Run Tests
```bash
python -m pytest tests/cli/integration/test_error_messages.py -v
```

**Expected Output**: 26 passed

---

## Future Enhancements

### Potential Additions

1. **Error Code 1005**: `MCP_SERVER_CRASHED` - Server process terminated unexpectedly
2. **Error Code 2004**: `ROUTER_STRATEGY_MISMATCH` - Router output doesn't match strategy
3. **Error Code 3003**: `TOKEN_UTILIZATION_HIGH` - High token usage alert
4. **Error Code 5000-5099**: Session management errors
5. **Error Code 6000-6099**: File I/O errors

### Localization
- Add support for multiple languages
- Provide translation files for error messages
- Locale-aware error formatting

### Error Analytics
- Track error frequency and patterns
- Generate error reports
- Suggest common fixes based on error history

---

## Related Documentation

- **Phase 9 Tasks**: See `PHASE9_TASKS.md` for T105 requirements
- **CLI Architecture**: See `CLI_ARCHITECTURE.md` for overall CLI design
- **Agent Templates**: See `AGENT_TEMPLATES.md` for agent creation patterns
- **MCP Integration**: See `MCP_INTEGRATION.md` for MCP server details

---

**Last Updated**: 2025-01-23
**Version**: 1.0.0
**Status**: Complete
