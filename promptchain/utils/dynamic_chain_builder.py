import os
import json
import asyncio
import logging
import inspect
import re
from contextlib import AsyncExitStack
from typing import Union, Callable, List, Dict, Optional, Any, Tuple

# Relative imports from within the utils package
from .models import ChainStep, ChainTechnique, ChainInstruction # Assuming models.py is in the same directory
from .mcp_helpers import AsyncProcessContextManager # Assuming mcp_helpers.py is in the same directory

# Imports potentially needed by methods (ensure these are covered)
from litellm import acompletion

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

class DynamicChainBuilder:
    """
    Manages the dynamic creation, execution, and state of multiple PromptChains or instruction sequences.
    Allows for complex workflows involving branching, parallel execution, and state sharing.
    Integrates with MCP servers for extended tool capabilities.
    """
    def __init__(
        self,
        base_model: Union[str, Dict[str, Any]],
        base_instruction: Union[str, Callable[[str], str]],
        available_functions: Optional[List[Callable]] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None, # <-- Add MCP servers config
        chain_registry: Optional[Dict[str, Any]] = None,
        group_registry: Optional[Dict[str, Any]] = None,
        shared_context: Optional[Dict[str, Any]] = None,
        default_chain_config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        """
        Initializes the DynamicChainBuilder.

        Args:
            base_model: The default model configuration for new chains.
            base_instruction: The default initial instruction for new chains.
            available_functions: List of Python functions available as tools.
            mcp_servers: Configuration for MCP servers providing additional tools. # <-- Add docstring
            chain_registry: Optional initial registry of chains.
            group_registry: Optional initial registry of chain groups.
            shared_context: Optional shared context dictionary accessible by chains.
            default_chain_config: Default configuration for new PromptChain instances created implicitly.
            verbose: Enable verbose logging for the builder and created chains.
        """
        self.base_model = base_model
        self.base_instruction = base_instruction
        self.available_functions = available_functions or []
        self.chain_registry = chain_registry or {}
        self.group_registry = group_registry or {}
        self.shared_context = shared_context or {"global": {}} # Ensure global context exists
        self.default_chain_config = default_chain_config or {}
        self.verbose = verbose

        # --- MCP Initialization ---
        self.mcp_servers_config = mcp_servers or []
        self.mcp_sessions: Dict[str, ClientSession] = {} # Store ClientSession objects
        self.mcp_tools_map: Dict[str, Dict[str, Any]] = {} # Store discovered tools {tool_name: tool_schema}
        self.mcp_exit_stack = AsyncExitStack() # Manage context of MCP connections
        self.mcp_connected = False # Flag to track connection status
        self._combined_tools: List[Dict[str, Any]] = [] # Add attribute to store final tool list

        # Prepare initial tool list (Python functions)
        self.py_tools = self._prepare_py_tools(self.available_functions)

        if self.verbose:
            logger.info("DynamicChainBuilder initialized.")
            if self.mcp_servers_config:
                logger.info(f"MCP servers configured: {[s.get('id', 'unknown') for s in self.mcp_servers_config]}")

    def _prepare_py_tools(self, functions: Optional[List[Callable]] = None) -> List[Dict[str, Any]]:
        """Prepares Python functions in the format expected by the model."""
        tools = []
        if functions:
            for func in functions:
                # Basic structure, assuming simple function signature
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": func.__name__,
                        "description": inspect.getdoc(func) or "No description provided.",
                        # Basic parameter handling - assumes single string input 'input'
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {"type": "string", "description": "Input string"}
                            },
                            "required": ["input"],
                        },
                    },
                }
                tools.append(tool_schema)
        return tools

    def _get_all_tools(self) -> List[Dict[str, Any]]:
        """Combines Python tools and discovered MCP tools."""
        # Return the combined list prepared during connection
        return self._combined_tools or [] # Return empty list if not connected/no tools

    # --- MCP Connection Management (Using LiteLLM Client) ---
    async def connect_mcp_async(self):
        """Connects to all configured MCP servers using LiteLLM's MCP client."""
        if not MCP_AVAILABLE or not experimental_mcp_client or not stdio_client:
            if self.mcp_servers_config:
                logger.warning("MCP library/client components not available. Skipping MCP connections.")
            return

        if self.mcp_connected:
            logger.info("MCP connections already established.")
            return

        if not self.mcp_servers_config:
            logger.info("No MCP servers configured.")
            return

        logger.info("Connecting to MCP servers via LiteLLM client...")
        self.mcp_sessions = {} # Reset sessions
        self.mcp_tools_map = {} # Reset tool map
        if not hasattr(self, 'mcp_exit_stack') or self.mcp_exit_stack is None:
            self.mcp_exit_stack = AsyncExitStack()

        self.py_tools = self._prepare_py_tools(self.available_functions)
        all_discovered_mcp_schemas = {} 

        for server_config in self.mcp_servers_config:
            server_id = server_config.get("id")
            server_type = server_config.get("type")
            command = server_config.get("command")
            args = server_config.get("args")
            env_custom = server_config.get("env")

            if not server_id or server_id in self.mcp_sessions:
                if not server_id:
                    logger.error("MCP server config missing 'id'. Skipping.")
                else:
                     logger.warning(f"Duplicate MCP server ID '{server_id}'. Skipping connection attempt.")
                continue

            logger.info(f"Initiating connection for MCP server: {server_id}")
            try:
                if server_type == "stdio":
                    if not command or not args:
                        logger.error(f"Stdio MCP server '{server_id}' missing 'command' or 'args'. Skipping.")
                        continue

                    process_env = os.environ.copy()
                    if env_custom:
                        process_env.update(env_custom)
                    server_params = StdioServerParameters(command=command, args=args, env=process_env)

                    reader, writer = await self.mcp_exit_stack.enter_async_context(stdio_client(server_params))
                    session = ClientSession(reader, writer)
                    await self.mcp_exit_stack.enter_async_context(session)
                    await session.initialize()
                    self.mcp_sessions[server_id] = session
                    logger.info(f"MCP Session established for '{server_id}'. Discovering tools...")

                    mcp_tools_openai_format = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
                    all_discovered_mcp_schemas[server_id] = mcp_tools_openai_format
                    logger.info(f"Discovered {len(mcp_tools_openai_format)} tools from '{server_id}'.")
                else:
                    logger.warning(f"Unsupported MCP server type '{server_type}' for server '{server_id}'.")
            except Exception as e:
                logger.error(f"Failed to connect or discover tools for MCP server '{server_id}': {e}", exc_info=True)

        # Process discovered tools
        final_mcp_tool_schemas_for_llm = []
        existing_tool_names = set(t['function']['name'] for t in self.py_tools) | set(f.__name__ for f in self.available_functions)

        for server_id, tool_schemas in all_discovered_mcp_schemas.items():
            for tool_schema in tool_schemas:
                 original_tool_name = tool_schema.get("function", {}).get("name")
                 if not original_tool_name:
                     logger.warning(f"Skipping MCP tool from '{server_id}' with missing function name.")
                     continue
                 prefixed_tool_name = f"mcp_{server_id}_{original_tool_name}"
                 if prefixed_tool_name in existing_tool_names:
                     logger.warning(f"Prefixed MCP tool name '{prefixed_tool_name}' conflicts with an existing tool. Skipping registration.")
                     continue

                 self.mcp_tools_map[prefixed_tool_name] = {
                    'original_schema': tool_schema, 
                    'server_id': server_id
                 }
                 prefixed_schema_copy = json.loads(json.dumps(tool_schema))
                 prefixed_schema_copy['function']['name'] = prefixed_tool_name
                 final_mcp_tool_schemas_for_llm.append(prefixed_schema_copy)
                 existing_tool_names.add(prefixed_tool_name)

        self._combined_tools = self.py_tools + final_mcp_tool_schemas_for_llm 

        if self.mcp_sessions:
            self.mcp_connected = True
            log_tool_names = [t.get('function', {}).get('name', '?') for t in self._combined_tools]
            logger.info(f"Connected to MCP servers: {list(self.mcp_sessions.keys())}")
            logger.info(f"Registered Tools (Local + MCP Prefixed): {log_tool_names}") 
        else:
            self.mcp_connected = False
            logger.warning("Failed to establish any MCP connections.")
            await self.mcp_exit_stack.aclose()




    async def close_mcp_async(self):
        """Closes connections to all MCP servers via AsyncExitStack."""
        if not self.mcp_connected:
            return
        logger.info("Closing MCP server connections...")
        if hasattr(self, 'mcp_exit_stack') and self.mcp_exit_stack:
            await self.mcp_exit_stack.aclose()
            self.mcp_exit_stack = AsyncExitStack() # Recreate
        self.mcp_sessions = {}
        self.mcp_tools_map = {}
        self.mcp_connected = False
        logger.info("MCP connections closed.")

    # --- Chain Management Methods ---
    def create_chain(
        self,
        chain_id: str,
        instructions: Optional[List[Union[str, Callable[[str], str]]]] = None,
        model: Optional[Union[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        group_id: Optional[str] = None,
    ):
        """Creates and registers a new chain configuration."""
        if chain_id in self.chain_registry:
            logger.warning(f"Chain ID '{chain_id}' already exists. Overwriting.")
        chain_config = self.default_chain_config.copy()
        if config:
            chain_config.update(config)
        chain_config['verbose'] = self.verbose or chain_config.get('verbose', False)
        self.chain_registry[chain_id] = {
            "instructions": instructions or [self.base_instruction],
            "model": model or self.base_model,
            "config": chain_config,
            "context": initial_context or {},
            "group_id": group_id,
            "history": [],
            "step_results": [],
            "status": "created",
        }
        if self.verbose:
            logger.info(f"Chain '{chain_id}' created and registered.")
        if group_id:
            self.add_chain_to_group(group_id, chain_id)

    def add_chain_to_group(self, group_id: str, chain_id: str):
        """Adds a chain to a specified group."""
        if chain_id not in self.chain_registry:
            raise ValueError(f"Chain ID '{chain_id}' not found in registry.")
        if group_id not in self.group_registry:
            self.group_registry[group_id] = {"chain_ids": set(), "status": "created"}
        self.group_registry[group_id]["chain_ids"].add(chain_id)
        self.chain_registry[chain_id]["group_id"] = group_id
        if self.verbose:
            logger.info(f"Chain '{chain_id}' added to group '{group_id}'.")

    def get_chain_state(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the current state (config, context, history) of a chain."""
        return self.chain_registry.get(chain_id)

    def update_chain_context(self, chain_id: str, updates: Dict[str, Any]):
        """Updates the context dictionary for a specific chain."""
        if chain_id not in self.chain_registry:
            raise ValueError(f"Chain ID '{chain_id}' not found.")
        self.chain_registry[chain_id].setdefault("context", {}).update(updates)
        if self.verbose:
             logger.debug(f"Context updated for chain '{chain_id}'.")

    # --- Execution Methods ---
    async def execute_chain(
        self,
        chain_id: str,
        initial_input: str,
        context_updates: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Executes a registered chain sequentially."""
        if self.mcp_servers_config and not self.mcp_connected:
             logger.info("MCP servers configured but not connected. Attempting connection...")
             await self.connect_mcp_async()
             if self.mcp_servers_config and not self.mcp_connected: 
                 logger.warning("Proceeding without MCP tool functionality as connection failed.")

        if chain_id not in self.chain_registry:
            raise ValueError(f"Chain ID '{chain_id}' not found.")

        chain_data = self.chain_registry[chain_id]
        chain_data["status"] = "running"
        chain_data["history"] = []  
        chain_data["step_results"] = [] 

        if context_updates:
            self.update_chain_context(chain_id, context_updates)

        current_input = initial_input
        combined_tools = self._get_all_tools()

        if self.verbose:
            logger.info(f"Executing chain '{chain_id}' with input: '{initial_input[:100]}...'")
            if combined_tools:
                 tool_names = [t.get("function", {}).get("name", "?") for t in combined_tools]
                 logger.debug(f"Available tools for chain '{chain_id}': {tool_names}")

        model_config = chain_data["model"]
        client = self._get_openai_client()

        full_history = [] 
        final_output = ""
        try:
            for i, instruction in enumerate(chain_data["instructions"]):
                step_input = current_input
                step_number = i + 1
                logger.info(f"Chain '{chain_id}' - Step {step_number}: Processing instruction...")

                if callable(instruction):
                    func_name = instruction.__name__
                    logger.info(f"Chain '{chain_id}' - Step {step_number}: Executing Python function '{func_name}'")
                    try:
                        step_output = instruction(step_input) 
                        if not isinstance(step_output, str):
                             logger.warning(f"Function '{func_name}' did not return a string. Converting.")
                             step_output = str(step_output)
                        logger.info(f"Chain '{chain_id}' - Step {step_number}: Function '{func_name}' output: '{step_output[:100]}...'")
                        full_history.append({"role": "assistant", "content": f"Executing Python function: {func_name}"}) 
                        full_history.append({"role": "function", "name": func_name, "content": step_output})
                    except Exception as e:
                        logger.error(f"Error executing function '{func_name}' in chain '{chain_id}': {e}", exc_info=True)
                        chain_data["status"] = "failed"
                        raise
                else:
                    prompt_context = {**self.shared_context.get("global", {}), **chain_data.get("context", {})}
                    prompt = instruction.format(input=step_input, **prompt_context)
                    logger.info(f"Chain '{chain_id}' - Step {step_number}: Calling LLM with prompt: '{prompt[:200]}...'")
                message = {"role": "user", "content": prompt}
                current_llm_history = full_history + [message] 
                full_history.append(message) 

                model_response = await self._run_model_step_async(\
                    client=client,\
                    model_config=model_config,\
                    history=current_llm_history, \
                    tools=combined_tools if combined_tools else None,\
                    chain_id=chain_id, \
                    step_number=step_number \
                )
                step_output = await self._process_model_response_async(\
                    response=model_response,\
                    history=full_history, \
                    tools=combined_tools, \
                    model_config=model_config, \
                    chain_id=chain_id,\
                    step_number=step_number \
                )
                logger.info(f"Chain '{chain_id}' - Step {step_number}: LLM Response/Final Output: '{step_output[:100]}...'")
                current_input = step_output
                chain_data["step_results"].append({"step": step_number, "input": step_input, "output": step_output})

            final_output = current_input 
        except Exception as e:
            logger.error(f"Chain '{chain_id}' failed during execution: {e}", exc_info=True)
            chain_data["status"] = "failed"
            final_output = f"ERROR in chain '{chain_id}': {e}" 
            # raise 

        chain_data["history"] = full_history 
        if chain_data["status"] != "failed":
            chain_data["status"] = "completed"
            logger.info(f"Chain '{chain_id}' executed successfully.")
        return final_output

    async def _run_model_step_async(
        self, client, model_config, history, tools, chain_id, step_number
    ) -> Any:
         """Runs a single step involving an LLM call."""
         model_name = model_config if isinstance(model_config, str) else model_config.get("model", "gpt-4")
         params = {}
         if isinstance(model_config, dict):
             params = {k: v for k, v in model_config.items() if k != "model"}
         request_params = {
             "model": model_name,
             "messages": history, 
             **params, 
         }
         if tools:
             request_params["tools"] = tools
             request_params["tool_choice"] = "auto" 
         if self.verbose:
              log_params = request_params.copy()
              log_params['messages'] = f"{len(history)} messages (last: {history[-1]['role']})" if history else "No messages" 
              log_params['tools'] = f"{len(tools)} tools" if tools else "None" 
              logger.debug(f"Chain '{chain_id}' - Step {step_number}: Sending request to model '{model_name}' with params: {log_params}")
         try:
             response = await acompletion(**request_params) 
             if self.verbose:
                 if response and response.choices:
                     logger.debug(f"Chain '{chain_id}' - Step {step_number}: Received choice from model: {response.choices[0].message}")
                 else:
                     logger.debug(f"Chain '{chain_id}' - Step {step_number}: Received potentially empty/unexpected response from model: {response}")
             return response 
         except Exception as e:
             logger.error(f"Error calling model '{model_name}' for chain '{chain_id}' step {step_number}: {e}", exc_info=True)
             raise

    async def _process_model_response_async(
        self, response, history: List[Dict], tools: List[Dict], model_config: Union[str, Dict], chain_id: str, step_number: int
    ) -> str:
        """Processes the model's response, handling potential tool calls using LiteLLM MCP client.
           Updates the passed-in 'history' list directly.
        """
        if not response or not response.choices:
            logger.warning(f"Chain '{chain_id}' - Step {step_number}: Received empty or invalid response from model: {response}")
            history.append({"role": "assistant", "content": "Error: Received invalid response from model."})
            return "Error: Received invalid response from model." 
        response_message = response.choices[0].message
        if hasattr(response_message, 'model_dump'):
            history.append(response_message.model_dump(exclude_unset=True)) 
        else: 
            history.append(response_message) 
        tool_calls = response_message.tool_calls
        if tool_calls:
            logger.info(f"Chain '{chain_id}' - Step {step_number}: Model requested {len(tool_calls)} tool call(s).")
            tool_results_messages = []
            for tool_call in tool_calls:
                 tool_call_id = tool_call.id
                 function_name = tool_call.function.name 
                 function_args_str = tool_call.function.arguments 
                 tool_output_content = json.dumps({"error": "Tool execution failed."})
                 logger.info(f"Chain '{chain_id}' - Step {step_number}: Processing tool call ID {tool_call_id}: {function_name}")
                 if function_name in self.mcp_tools_map:
                     if not MCP_AVAILABLE or not experimental_mcp_client:
                         logger.error(f"MCP tool '{function_name}' called, but MCP client unavailable.")
                         tool_output_content = json.dumps({"error": "MCP client library not available."})
                     else:
                         mcp_info = self.mcp_tools_map[function_name]
                         server_id = mcp_info['server_id']
                         session = self.mcp_sessions.get(server_id)
                         original_schema = mcp_info['original_schema']
                         original_tool_name = original_schema['function']['name']
                         if not session:
                             logger.error(f"MCP Session '{server_id}' not found for tool '{function_name}'.")
                             tool_output_content = json.dumps({"error": f"MCP session '{server_id}' unavailable."})
                         else:
                             openai_tool_for_mcp = {
                                 "id": tool_call_id,
                                 "type": "function",
                                 "function": {
                                     "name": original_tool_name, 
                                     "arguments": function_args_str
                                 }
                             }
                             if self.verbose:
                                 logger.debug(f"  - Calling MCP Tool: {original_tool_name} on server {server_id} via LiteLLM client (ID: {tool_call_id})")
                                 logger.debug(f"    Args: {function_args_str}")
                             try:
                                 call_result = await experimental_mcp_client.call_openai_tool(
                                     session=session,
                                     openai_tool=openai_tool_for_mcp 
                                 )
                                 # --- Robust MCP Result Processing --- 
                                 tool_output_content = None
                                 extracted_json_str = None

                                 if isinstance(call_result, str):
                                     # Scenario 1: Result is already a string (Likely the complex meta=None... format)
                                     logger.debug(f"MCP call_result is a string: {call_result[:100]}...")
                                     match = re.search(r"text='({.*?})'", call_result, re.DOTALL)
                                     if match:
                                         extracted_json_str = match.group(1) # Raw JSON string from text field
                                         logger.debug(f"Extracted JSON string via regex: {extracted_json_str[:100]}...")
                                     else:
                                         logger.warning(f"MCP result was string, but regex failed to extract JSON from text='...'. Storing raw string.")
                                         # Fallback: treat the whole string as the content (wrapped in JSON quotes)
                                         tool_output_content = json.dumps(call_result) 
                                 elif call_result and hasattr(call_result, 'content') and call_result.content and hasattr(call_result.content[0], 'text') and call_result.content[0].text is not None:
                                     # Scenario 2: Result is object-like with TextContent
                                     extracted_json_str = call_result.content[0].text
                                     logger.debug(f"Extracted text from call_result.content[0].text: {extracted_json_str[:100]}...")
                                 elif call_result and hasattr(call_result, 'text') and call_result.text is not None:
                                     # Scenario 3: Result is object-like with direct .text attribute
                                     extracted_json_str = call_result.text
                                     logger.debug(f"Extracted text from call_result.text: {extracted_json_str[:100]}...")
                                 else:
                                     # Scenario 4: Unexpected format or empty result
                                     if call_result:
                                         logger.warning(f"MCP tool '{original_tool_name}' result format unexpected or missing text content: {call_result}. Storing string representation.")
                                         tool_output_content = json.dumps(str(call_result)) 
                                     else:
                                         logger.warning(f"MCP tool '{original_tool_name}' executed but returned no content (result was None or empty).")
                                         tool_output_content = json.dumps({"warning": f"MCP tool '{original_tool_name}' returned no content."})
                                 
                                 # If we extracted a potential JSON string, try to parse and store it
                                 if extracted_json_str and tool_output_content is None:
                                     try:
                                         # Important: Unescape potential double-escaped chars from regex/string representation
                                         processed_str = extracted_json_str.encode('utf-8').decode('unicode_escape')
                                         parsed_dict = json.loads(processed_str)
                                         # Store the valid dictionary, re-serialized as a JSON string
                                         tool_output_content = json.dumps(parsed_dict)
                                         logger.debug("Successfully parsed extracted MCP JSON string.")
                                     except json.JSONDecodeError as json_err:
                                         logger.warning(f"Extracted text from MCP tool '{original_tool_name}' is not valid JSON: {extracted_json_str[:100]}... Error: {json_err}")
                                         # Store the raw extracted text as a JSON string literal if parsing fails
                                         tool_output_content = json.dumps(extracted_json_str) 
                                     except Exception as parse_err:
                                        logger.error(f"Unexpected error parsing extracted MCP text '{extracted_json_str[:100]}...': {parse_err}")
                                        tool_output_content = json.dumps({"error": f"Parsing error: {parse_err}", "raw_extracted": extracted_json_str})
                                 
                                 # Ensure tool_output_content is set (should be by now)
                                 if tool_output_content is None:
                                     logger.error(f"Logic error: tool_output_content was not set for MCP tool '{original_tool_name}'. Defaulting to error.")
                                     tool_output_content = json.dumps({"error": "Internal processing error retrieving MCP result."})       
                                 # --- End Robust MCP Result Processing --- 
                                 
                                 if self.verbose:
                                     logger.debug(f"  - MCP Result Content (Stored): {tool_output_content[:150]}...")
                             except Exception as e:
                                 logger.error(f"Error executing MCP tool '{function_name}' via LiteLLM client: {e}", exc_info=True)
                                 tool_output_content = json.dumps({"error": f"Failed executing MCP tool '{function_name}': {e}"})
                 elif function_name in self.tool_functions:
                     func_to_call = self.tool_functions[function_name]
                     try:
                         arguments = json.loads(function_args_str)
                         logger.debug(f"  - Calling Local Python Tool: {function_name}(**{arguments})")
                         result = func_to_call(**arguments)
                         if isinstance(result, (dict, list)): 
                             tool_output_content = json.dumps(result)
                         elif isinstance(result, str):
                             try: 
                                 json.loads(result) 
                                 tool_output_content = result
                             except json.JSONDecodeError:
                                 logger.debug("Local tool result is non-JSON string, using directly.")
                                 tool_output_content = result 
                         else:
                             tool_output_content = json.dumps(str(result))
                         logger.info(f"Local function '{function_name}' executed successfully.")
                     except json.JSONDecodeError as e:
                         logger.error(f"Invalid JSON arguments for local function {function_name}: {function_args_str}. Error: {e}")
                         tool_output_content = json.dumps({"error": f"Invalid JSON arguments for {function_name}"})
                     except Exception as e:
                         logger.error(f"Error executing Python function {function_name}: {e}", exc_info=True)
                         tool_output_content = json.dumps({"error": f"Failed to execute Python function '{function_name}': {e}"})
                 else:
                     logger.warning(f"Function '{function_name}' requested by model not found locally or via MCP.")
                     tool_output_content = json.dumps({"error": f"Function '{function_name}' not found."})
                 tool_results_messages.append({
                     "tool_call_id": tool_call_id,
                     "role": "tool",
                     "name": function_name, 
                     "content": tool_output_content if isinstance(tool_output_content, str) else json.dumps(tool_output_content), 
                 })
            history.extend(tool_results_messages)
            logger.info(f"Chain '{chain_id}' - Step {step_number}: Sending tool results back to model.")
            # Get the client instance for the follow-up call
            client = self._get_openai_client()
            # Pass the updated history and the original model_config
            follow_up_response = await self._run_model_step_async(
                 client=client, 
                 model_config=model_config, # Use model_config passed into this method
                 history=history,
                 tools=tools, # Pass tools again 
                 chain_id=chain_id,
                 step_number=step_number 
            )
            if not follow_up_response or not follow_up_response.choices:
                logger.error(f"Chain '{chain_id}' - Step {step_number}: Received empty response after submitting tool results.")
                history.append({"role": "assistant", "content": "Error: No response from model after tool execution."})
                return "Error: No response from model after tool execution."
            final_message = follow_up_response.choices[0].message
            if hasattr(final_message, 'model_dump'):
                history.append(final_message.model_dump(exclude_unset=True))
            else:
                history.append(final_message)
            return final_message.content or "" 
        else:
             return response_message.content or ""

    # --- Helper Methods ---
    def _get_openai_client(self):
        """Helper to get initialized OpenAI client. Needs error handling/config."""
        try:
             from openai import AsyncOpenAI
        except ImportError:
             logger.error("OpenAI library not installed. Please install with `pip install openai`")
             raise RuntimeError("OpenAI library not installed.")
        try:
             api_key = os.getenv("OPENAI_API_KEY")
             if not api_key:
                  logger.warning("OPENAI_API_KEY environment variable not set. OpenAI calls may fail.")
             return AsyncOpenAI(api_key=api_key) 
        except Exception as e:
             logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
             raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    # --- Context Management ---
    def set_global_context(self, key: str, value: Any):
         """Sets a value in the shared global context."""
         self.shared_context.setdefault("global", {})[key] = value
         if self.verbose:
             logger.debug(f"Global context updated: {key} = {value}")

    def get_global_context(self, key: str, default: Any = None) -> Any:
         """Retrieves a value from the shared global context."""
         return self.shared_context.get("global", {}).get(key, default)

    async def __aenter__(self):
        """Enter context manager, connect MCP."""
        if self.mcp_servers_config:
            await self.connect_mcp_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, close MCP."""
        await self.close_mcp_async()

# Example usage
# (Keep example usage if needed, or remove if this file is purely library code)
# if __name__ == "__main__": ... 