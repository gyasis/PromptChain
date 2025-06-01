import asyncio
import json
import logging
from typing import List, Dict, Any, Optional

# Try importing MCP components, handle gracefully if not installed
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from litellm import experimental_mcp_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None # Define dummy types for annotations if MCP not available
    StdioServerParameters = None
    stdio_client = None
    experimental_mcp_client = None

logger = logging.getLogger(__name__)

async def execute_mcp_tool(
    tool_call: Any,
    mcp_sessions: Dict[str, ClientSession],
    mcp_tools_map: Dict[str, Dict[str, Any]],
    verbose: bool = False
) -> str:
    """
    Executes an MCP tool based on the tool_call object using provided context.

    Args:
        tool_call: The tool call object (LiteLLM format).
        mcp_sessions: Dictionary mapping server_id to active MCP sessions.
        mcp_tools_map: Dictionary mapping prefixed tool names to original schema and server_id.
        verbose: Verbosity flag for logging.

    Returns:
        A string representing the tool's output (usually JSON stringified).
    """
    function_name = getattr(getattr(tool_call, 'function', None), 'name', None) # Prefixed name
    function_args_str = getattr(getattr(tool_call, 'function', None), 'arguments', "{}")
    tool_call_id = getattr(tool_call, 'id', 'N/A')
    tool_output = f'Error: MCP tool {function_name} execution failed.'

    if not function_name or function_name not in mcp_tools_map:
        logger.warning(f"[MCP Manager] MCP tool function '{function_name}' not found in map.")
        return json.dumps({"error": f"MCP tool function '{function_name}' not found in map."})

    if not MCP_AVAILABLE or not experimental_mcp_client:
        logger.error(f"[MCP Manager] MCP tool '{function_name}' called, but MCP library/client not available.")
        return json.dumps({"error": f"MCP library not available to call tool '{function_name}'."})

    original_tool_name = "unknown" # Initialize for logging
    server_id = "unknown" # Initialize for logging

    try:
        mcp_info = mcp_tools_map[function_name]
        server_id = mcp_info['server_id']
        session = mcp_sessions.get(server_id)

        if not session:
            logger.error(f"[MCP Manager] MCP Session '{server_id}' not found for tool '{function_name}'.")
            return json.dumps({"error": f"MCP session '{server_id}' unavailable."})

        original_schema = mcp_info['original_schema']
        original_tool_name = original_schema['function']['name']

        openai_tool_for_mcp = {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": original_tool_name,
                "arguments": function_args_str
            }
        }
        log_msg = f"  [MCP Manager] Calling MCP Tool: {original_tool_name} on server {server_id} via {function_name}"
        logger.debug(log_msg)
        if verbose: print(log_msg) # Print if verbose

        call_result = await experimental_mcp_client.call_openai_tool(
            session=session,
            openai_tool=openai_tool_for_mcp
        )

        # Process result
        if call_result and hasattr(call_result, 'content') and call_result.content and hasattr(call_result.content[0], 'text'):
            tool_output = str(call_result.content[0].text)
        elif call_result: # Handle cases where result might not have the expected structure
            tool_output = str(call_result)
        else:
            tool_output = json.dumps({"warning": f"MCP tool {original_tool_name} executed but returned no structured content."})

        log_msg = f"  [MCP Manager] MCP Result (ID: {tool_call_id}): {tool_output[:150]}..."
        logger.debug(log_msg)
        if verbose: print(log_msg) # Print if verbose

    # This is the correct placement for the except block for the MCP tool call
    except Exception as e:
        err_msg = f"Error executing MCP tool {function_name} (Original: {original_tool_name}, ID: {tool_call_id}) on {server_id}: {str(e)}"
        logger.error(f"[MCP Manager] {err_msg}", exc_info=verbose) # Log traceback only if verbose
        tool_output = json.dumps({"error": err_msg})

    return tool_output 