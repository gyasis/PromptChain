import asyncio
import logging
import os
import json
from contextlib import AsyncExitStack
from typing import Dict, List, Any, Optional, Callable

# Attempt to import MCP components, define dummies if not available
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
    # Define dummy AsyncExitStack if needed for type hints when MCP_AVAILABLE is False
    # though it's better to check MCP_AVAILABLE before using it.
    class AsyncExitStack:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def enter_async_context(self, cm): return await cm.__aenter__()
        async def aclose(self): pass

logger = logging.getLogger(__name__)

# Helper async context manager for subprocesses
class AsyncProcessContextManager:
    """Manages the lifecycle of an asyncio subprocess, ensuring termination."""
    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process

    async def __aenter__(self):
        return self.process

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.process.returncode is None:
            try:
                # Try graceful termination first
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
                logger.debug(f"MCP process {self.process.pid} terminated gracefully.")
            except asyncio.TimeoutError:
                logger.warning(f"MCP process {self.process.pid} did not terminate gracefully, killing.")
                self.process.kill()
                await self.process.wait() # Ensure kill completes
            except ProcessLookupError:
                 logger.debug(f"MCP process {self.process.pid} already exited.")
            except Exception as e:
                 logger.error(f"Error terminating/killing MCP process {self.process.pid}: {e}")
        else:
            logger.debug(f"MCP process {self.process.pid} already exited with code {self.process.returncode}.")

# New MCP Helper Class
class MCPHelper:
    """Manages MCP connections, tool discovery, and execution."""
    def __init__(self, mcp_servers: Optional[List[Dict[str, Any]]], verbose: bool, logger_instance: logging.Logger,
                 local_tool_schemas_ref: List[Dict], local_tool_functions_ref: Dict[str, Callable]):
        self.mcp_servers = mcp_servers or []
        self.verbose = verbose
        self.logger = logger_instance
        self.local_tool_schemas_ref = local_tool_schemas_ref # Reference for conflict check
        self.local_tool_functions_ref = local_tool_functions_ref # Reference for conflict check

        self.mcp_sessions: Dict[str, ClientSession] = {} # Maps server_id to MCP session
        self.mcp_tools_map: Dict[str, Dict[str, Any]] = {} # Maps prefixed_tool_name -> {'original_schema': ..., 'server_id': ...}
        self.mcp_tool_schemas: List[Dict] = [] # Holds the prefixed tool schemas discovered from MCP

        # Initialize exit_stack conditionally only if MCP is available and servers are configured
        self.exit_stack = AsyncExitStack() if MCP_AVAILABLE and self.mcp_servers else None

        if self.mcp_servers and not MCP_AVAILABLE:
            self.logger.warning("MCPHelper initialized with servers, but 'mcp' library not installed. MCP features disabled.")

    def connect_mcp(self):
        """ Synchronous version of connect_mcp_async. """
        if not MCP_AVAILABLE: return # Do nothing if MCP not installed
        try:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running(): raise RuntimeError("connect_mcp (sync) called from within a running event loop.")
                else: return loop.run_until_complete(self.connect_mcp_async())
            except RuntimeError: # No running loop
                 return asyncio.run(self.connect_mcp_async())
        except Exception as e:
            self.logger.error(f"Error in MCPHelper.connect_mcp sync wrapper: {e}", exc_info=True)
            # Consider re-raising or handling more gracefully depending on need

    async def connect_mcp_async(self):
        """ Asynchronous version of connect_mcp. Connects to configured servers and discovers tools."""
        if self.mcp_sessions:
             self.logger.info("MCP sessions already exist. Skipping connection.")
             return

        if not MCP_AVAILABLE or not experimental_mcp_client:
            if self.mcp_servers: self.logger.warning("MCP library/client not available, skipping MCP connections.")
            return

        if not self.mcp_servers:
            if self.verbose: self.logger.debug("🔌 No MCP servers configured.")
            return

        if self.verbose: self.logger.debug(f"🔌 MCPHelper connecting to {len(self.mcp_servers)} MCP server(s)...")

        if self.exit_stack is None: self.exit_stack = AsyncExitStack() # Initialize if not already

        # Clear previous MCP tools before discovery
        self.mcp_tools_map = {}
        self.mcp_tool_schemas = []

        # Get current local tool names for conflict checking
        local_tool_names = set(self.local_tool_functions_ref.keys()) | \
                           {t['function']['name'] for t in self.local_tool_schemas_ref if t.get("function", {}).get("name")}

        connect_success_count = 0
        for server_config in self.mcp_servers:
            server_id = server_config.get("id")
            server_type = server_config.get("type")
            command = server_config.get("command")
            args = server_config.get("args")
            env = server_config.get("env")

            if not server_id: self.logger.warning("Skipping MCP config with missing 'id'."); continue
            if server_id in self.mcp_sessions: self.logger.warning(f"Server ID '{server_id}' already connected, skipping."); connect_success_count +=1; continue

            try:
                if server_type == "stdio":
                    if not command: self.logger.warning(f"Skipping stdio server '{server_id}' missing 'command'."); continue
                    args = args if isinstance(args, list) else []
                    server_params = StdioServerParameters(command=command, args=args, env=env or None)

                    self.logger.debug(f"Attempting stdio connection for server '{server_id}'...")
                    stdio_conn_manager = stdio_client(server_params)
                    # Ensure self.exit_stack is not None before using it
                    if self.exit_stack is None:
                         self.logger.error("Exit stack is None, cannot manage MCP connection context.")
                         continue # Skip this server
                    reader, writer = await self.exit_stack.enter_async_context(stdio_conn_manager)

                    self.logger.debug(f"Stdio client connected for '{server_id}'. Creating session...")
                    session = ClientSession(reader, writer)
                    await self.exit_stack.enter_async_context(session)
                    self.logger.debug(f"Session context entered for '{server_id}'. Initializing...")
                    await session.initialize()
                    self.mcp_sessions[server_id] = session
                    connect_success_count += 1
                    if self.verbose: self.logger.debug(f"  ✅ Connected MCP server '{server_id}' (Cmd: {command} {' '.join(args)})")

                    try: # Discover tools
                        if self.verbose: self.logger.debug(f"  🔍 Discovering tools on '{server_id}'...")
                        mcp_tools_discovered = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
                        if self.verbose: self.logger.debug(f"  🔬 Found {len(mcp_tools_discovered)} tools on '{server_id}'.")

                        current_all_tool_names = local_tool_names | set(self.mcp_tools_map.keys()) # Check against already mapped MCP tools too
                        for tool_schema in mcp_tools_discovered:
                            tool_schema_dict = tool_schema.model_dump(exclude_unset=True) if hasattr(tool_schema, 'model_dump') else tool_schema if isinstance(tool_schema, dict) else None
                            if not tool_schema_dict: self.logger.warning(f"Skipping MCP tool from '{server_id}' type: {type(tool_schema)}"); continue

                            original_tool_name = tool_schema_dict.get("function", {}).get("name")
                            if not original_tool_name: self.logger.warning(f"Skipping MCP tool from '{server_id}' missing name."); continue

                            prefixed_tool_name = f"mcp_{server_id}_{original_tool_name}"
                            if prefixed_tool_name in current_all_tool_names: self.logger.warning(f"Prefixed MCP tool '{prefixed_tool_name}' conflicts. Skipping."); continue
                            if original_tool_name in local_tool_names: self.logger.warning(f"MCP tool '{original_tool_name}' conflicts with local tool. Prefixed as '{prefixed_tool_name}'.")

                            # Create a deep copy before modifying
                            prefixed_schema = json.loads(json.dumps(tool_schema_dict))
                            prefixed_schema["function"]["name"] = prefixed_tool_name
                            self.mcp_tool_schemas.append(prefixed_schema) # Store the prefixed schema
                            self.mcp_tools_map[prefixed_tool_name] = {'original_schema': tool_schema_dict, 'server_id': server_id}
                            current_all_tool_names.add(prefixed_tool_name) # Update the set for next check
                            if self.verbose: self.logger.debug(f"    -> Registered MCP tool: {prefixed_tool_name} (original: {original_tool_name})")
                    except Exception as tool_err: self.logger.error(f"Error discovering tools on MCP server '{server_id}': {tool_err}", exc_info=self.verbose)
                else:
                    self.logger.warning(f"Unsupported MCP server type '{server_type}' for '{server_id}'.")
            except Exception as conn_err:
                self.logger.error(f"Error connecting/initializing MCP server '{server_id}': {conn_err}", exc_info=self.verbose)

        if connect_success_count > 0: self.logger.info(f"MCP Connection: {connect_success_count}/{len(self.mcp_servers)} servers connected.")
        elif self.mcp_servers: self.logger.error("MCP Connection failed for all configured servers.")

    def close_mcp(self):
        """ Synchronous version of close_mcp_async. """
        if not MCP_AVAILABLE: return
        try:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running(): raise RuntimeError("close_mcp (sync) called from within a running event loop.")
                else: return loop.run_until_complete(self.close_mcp_async())
            except RuntimeError: # No running loop
                 return asyncio.run(self.close_mcp_async())
        except Exception as e:
             self.logger.error(f"Error in MCPHelper.close_mcp sync wrapper: {e}", exc_info=True)

    async def close_mcp_async(self):
        """ Asynchronous version of close_mcp. Closes connections and cleans up state."""
        if not self.exit_stack:
            if self.verbose: self.logger.debug("🔌 No MCP connections managed by exit_stack.")
            self._reset_mcp_state() # Ensure clean state
            return

        if self.verbose: self.logger.debug("🔌 Closing MCP connections via AsyncExitStack...")
        try:
            await self.exit_stack.aclose()
            # Re-initialize exit_stack for potential reuse within the same helper instance
            self.exit_stack = AsyncExitStack() if MCP_AVAILABLE and self.mcp_servers else None
            if self.verbose:
                self.logger.debug("  ✅ MCP connections closed and exit stack reset.")
        except Exception as e:
             self.logger.error(f"Error during MCP connection closing: {e}", exc_info=True)
             # Attempt to reset stack even if closing failed partially
             self.exit_stack = AsyncExitStack() if MCP_AVAILABLE and self.mcp_servers else None

        # Clear state regardless of close success
        self._reset_mcp_state()

    def _reset_mcp_state(self):
        """ Resets MCP session and tool state."""
        self.mcp_sessions = {}
        self.mcp_tools_map = {}
        self.mcp_tool_schemas = []
        if self.verbose: self.logger.debug("MCP Helper state reset.")


    async def execute_mcp_tool(self, tool_call: Any) -> str:
        """
        Executes an MCP tool based on the tool_call object using the LiteLLM MCP client.
        Returns the result as a string (plain or JSON).
        Renamed from _execute_mcp_tool_internal for external use by PromptChain.
        """
        # Import the helper function for robust function name extraction
        from promptchain.utils.agentic_step_processor import get_function_name_from_tool_call
        
        # Extract function name using the robust helper function
        function_name = get_function_name_from_tool_call(tool_call)
        
        # Extract arguments string - handle different object types
        if isinstance(tool_call, dict):
            function_obj = tool_call.get('function', {})
            function_args_str = function_obj.get('arguments', '{}') if isinstance(function_obj, dict) else '{}'
            tool_call_id = tool_call.get('id', 'N/A')
        else:
            # Get function arguments from object
            function_obj = getattr(tool_call, 'function', None)
            if function_obj:
                if hasattr(function_obj, 'arguments'):
                    function_args_str = function_obj.arguments
                elif isinstance(function_obj, dict) and 'arguments' in function_obj:
                    function_args_str = function_obj['arguments']
                else:
                    function_args_str = '{}'
            else:
                function_args_str = '{}'
            
            # Get tool call ID
            tool_call_id = getattr(tool_call, 'id', 'N/A')

        # Log debug information
        self.logger.debug(f"[MCP Helper] Processing MCP tool call: {function_name} (ID: {tool_call_id})")
        self.logger.debug(f"[MCP Helper] Tool call argument string: {function_args_str}")
        
        tool_output_str = json.dumps({"error": f"MCP tool '{function_name}' execution failed."})

        if not function_name or function_name not in self.mcp_tools_map:
            self.logger.error(f"[MCP Helper] Tool function '{function_name}' (ID: {tool_call_id}) not found in map.")
            self.logger.debug(f"[MCP Helper] Available MCP tools: {list(self.mcp_tools_map.keys())}")
            return json.dumps({"error": f"MCP tool function '{function_name}' not found in map."})

        if not MCP_AVAILABLE or not experimental_mcp_client:
            self.logger.error(f"[MCP Helper] Tool '{function_name}' (ID: {tool_call_id}) called, but MCP library/client not available.")
            return json.dumps({"error": f"MCP library not available to call tool '{function_name}'."})

        try:
            mcp_info = self.mcp_tools_map[function_name]
            server_id = mcp_info['server_id']
            session = self.mcp_sessions.get(server_id)

            if not session:
                self.logger.error(f"[MCP Helper] Session '{server_id}' not found for tool '{function_name}' (ID: {tool_call_id}).")
                return json.dumps({"error": f"MCP session '{server_id}' unavailable for tool '{function_name}'."})

            original_schema = mcp_info['original_schema']
            original_tool_name = original_schema['function']['name']

            # Construct the tool call payload in the format expected by experimental_mcp_client
            openai_tool_for_mcp = {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": original_tool_name,
                    "arguments": function_args_str
                }
            }

            if self.verbose: 
                self.logger.debug(f"  [MCP Helper] Calling Tool: {original_tool_name} on {server_id} (Prefixed: {function_name}, ID: {tool_call_id})")
                self.logger.debug(f"  [MCP Helper] Arguments: {function_args_str}")

            call_result = await experimental_mcp_client.call_openai_tool(
                session=session,
                openai_tool=openai_tool_for_mcp
            )

            # Process the result (assuming structure from litellm)
            # TODO: Verify the exact structure returned by call_openai_tool
            if call_result and hasattr(call_result, 'content') and call_result.content and hasattr(call_result.content[0], 'text'):
                 tool_output_str = str(call_result.content[0].text) # Ensure string
            elif call_result:
                 self.logger.debug(f"[MCP Helper] Tool {original_tool_name} returned unexpected structure. Converting result to string. Result: {call_result}")
                 try: tool_output_str = json.dumps(call_result) if not isinstance(call_result, str) else call_result
                 except TypeError: tool_output_str = str(call_result) # Fallback to plain string
            else:
                 self.logger.warning(f"[MCP Helper] Tool {original_tool_name} (ID: {tool_call_id}) executed but returned no structured content.")
                 tool_output_str = json.dumps({"warning": f"Tool {original_tool_name} executed but returned no structured content."}) # Return warning as JSON

            if self.verbose: self.logger.debug(f"  [MCP Helper] Result (ID: {tool_call_id}): {tool_output_str[:150]}...")

        except Exception as e:
            # Extract original name if possible for better error message
            original_tool_name_err = mcp_info.get('original_schema',{}).get('function',{}).get('name','?') if 'mcp_info' in locals() else '?'
            server_id_err = server_id if 'server_id' in locals() else '?'
            self.logger.error(f"[MCP Helper] Error executing tool {function_name} (Original: {original_tool_name_err}, ID: {tool_call_id}) on {server_id_err}: {e}", exc_info=self.verbose)
            tool_output_str = json.dumps({"error": f"Error executing MCP tool {function_name}: {str(e)}"}) # Return error as JSON

        return tool_output_str

    # --- Methods to access state ---
    def get_mcp_tool_schemas(self) -> List[Dict]:
        """Returns the list of discovered (prefixed) MCP tool schemas."""
        return self.mcp_tool_schemas

    def get_mcp_tools_map(self) -> Dict[str, Dict[str, Any]]:
        """Returns the map of prefixed MCP tool names to their info."""
        return self.mcp_tools_map 