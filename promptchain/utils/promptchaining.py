# REMOVED: from future import annotations (Not needed in modern Python 3.7+)
from litellm import completion, acompletion
import os
from typing import Union, Callable, List, Dict, Optional, Any, Tuple
from dotenv import load_dotenv
import inspect
import json
import asyncio
from contextlib import AsyncExitStack
import logging
import time
import functools # Added import

# CORRECTED: Standard Python import syntax
# Assuming preprompt.py and other .py files are in the same directory or package
try:
    from .preprompt import PrePrompt
except ImportError:
    print("Warning: Could not import PrePrompt relative to current file. Ensure preprompt.py exists.")
    # Define a dummy class if needed for type hints or basic functionality
    class PrePrompt:
        def __init__(self, *args, **kwargs): pass
        def load(self, instruction_id: str) -> str: return instruction_id # Basic fallback

try:
    # Keep if used (adjust path if necessary)
    # from .models import ChainStep, ChainTechnique, ChainInstruction
    # from .mcp_helpers import AsyncProcessContextManager
    from .agentic_step_processor import AgenticStepProcessor
    from .mcp_helpers import MCPHelper # Import the new helper class
except ImportError as e:
    print(f"Warning: Could not import dependencies relative to current file: {e}")
    # Define dummy classes/types if needed
    class AgenticStepProcessor:
        objective = "Dummy Objective" # Add objective attribute for process_prompt_async
        def __init__(self, *args, **kwargs): pass
        async def run_async(self, *args, **kwargs): return "AgenticStepProcessor not available"
    class MCPHelper: # Define dummy if import fails
        def __init__(self, *args, **kwargs): self.verbose = False
        async def connect_mcp_async(self): pass
        def connect_mcp(self): pass
        async def close_mcp_async(self): pass
        def close_mcp(self): pass
        async def execute_mcp_tool(self, *args, **kwargs): return json.dumps({"error": "MCP Helper dummy"})
        def get_mcp_tool_schemas(self): return []
        def get_mcp_tools_map(self): return {}

    ChainStep = Dict
    ChainTechnique = str
    ChainInstruction = Union[str, Callable, AgenticStepProcessor]
    # AsyncProcessContextManager = Any <-- Removed

# Try importing MCP components, handle gracefully if not installed
# This check is now also done within MCPHelper, but keep here for general awareness
try:
    from mcp import ClientSession, StdioServerParameters # Keep potentially needed types
    from mcp.client.stdio import stdio_client
    from litellm import experimental_mcp_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    experimental_mcp_client = None

# Load environment variables from .env file
# Ensure the path is correct relative to where the script is run
# Using a relative path like "../.env" might be more robust if structure is fixed
load_dotenv(".env") # Adjusted path assumption

# Configure environment variables for API keys
# These will be loaded from the .env file if present, otherwise use placeholders
os.environ.setdefault("OPENAI_API_KEY", "your-openai-api-key-from-env-or-here")
os.environ.setdefault("ANTHROPIC_API_KEY", "your-anthropic-api-key-from-env-or-here")

# Setup logger
# CORRECTED: Use __name__ for the logger name
logging.basicConfig(level=logging.INFO) # Configure basic logging
logger = logging.getLogger(__name__)

# Python Class for Prompt Chaining
class PromptChain:
    # CORRECTED: __init__ instead of init
    def __init__(self, models: List[Union[str, dict]],
                 instructions: List[Union[str, Callable, AgenticStepProcessor]], # Use ChainInstruction alias?
                 full_history: bool = False,
                 store_steps: bool = False,
                 verbose: bool = False,
                 chainbreakers: List[Callable] = None,
                 mcp_servers: Optional[List[Dict[str, Any]]] = None,
                 additional_prompt_dirs: Optional[List[str]] = None):
        """
        Initialize the PromptChain with optional step storage and verbose output.

        :param models: List of model names or dicts with model config.
                      If single model provided, it will be used for all instructions.
        :param instructions: List of instruction templates, prompt IDs, ID:strategy strings, or callable functions or AgenticStepProcessors
        :param full_history: Whether to pass full chain history
        :param store_steps: If True, stores step outputs in self.step_outputs without returning full history
        :param verbose: If True, prints detailed output for each step with formatting
        :param chainbreakers: List of functions that can break the chain if conditions are met
                             Each function should take (step_number, current_output) and return
                             (should_break: bool, break_reason: str, final_output: Any)
        :param mcp_servers: Optional list of MCP server configurations.
                            Each dict should have 'id', 'type' ('stdio' supported), and type-specific args (e.g., 'command', 'args').
        :param additional_prompt_dirs: Optional list of paths to directories containing custom prompts for PrePrompt.
        """
        self.verbose = verbose
        # Setup logger level based on verbosity
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Extract model names and parameters
        self.models = []
        self.model_params = []

        # Count non-function/non-agentic instructions to match with models
        model_instruction_count = sum(1 for instr in instructions
                                      if not callable(instr) and not isinstance(instr, AgenticStepProcessor))

        # Handle single model case
        if models and len(models) == 1 and model_instruction_count > 0:
            num_models_needed = model_instruction_count
            models = models * num_models_needed # Create a list with the model repeated

        # Process models (now using the potentially expanded list)
        for model_config in models:
            if isinstance(model_config, dict):
                self.models.append(model_config["name"])
                self.model_params.append(model_config.get("params", {}))
            else:
                self.models.append(model_config)
                self.model_params.append({})

        # Validate model count against instructions needing models
        if len(self.models) != model_instruction_count:
            raise ValueError(
                f"Number of models ({len(self.models)}) must match number of non-function/non-agentic instructions ({model_instruction_count})"
                f"\nModels provided: {models}"
                f"\nInstructions: {instructions}"
                "\nOr provide a single model name/dict in the 'models' list to use for all applicable instructions."
            )

        self.instructions = instructions
        self.full_history = full_history
        self.store_steps = store_steps
        self.preprompter = PrePrompt(additional_prompt_dirs=additional_prompt_dirs)
        self.step_outputs = {}
        self.chainbreakers = chainbreakers or []

        # Renamed attributes for local tools
        self.local_tools: List[Dict] = [] # Schemas for LOCAL functions
        self.local_tool_functions: Dict[str, Callable] = {} # Implementations for LOCAL functions

        # Instantiate MCPHelper if MCP is available and configured
        self.mcp_helper: Optional[MCPHelper] = None
        if MCP_AVAILABLE and mcp_servers:
            self.mcp_helper = MCPHelper(
                mcp_servers=mcp_servers,
                verbose=self.verbose,
                logger_instance=logger,
                local_tool_schemas_ref=self.local_tools, # Pass reference
                local_tool_functions_ref=self.local_tool_functions # Pass reference
            )
        elif mcp_servers and not MCP_AVAILABLE:
             logger.warning("mcp_servers configured but 'mcp' library not installed. MCP tools will not be available.")

        self.reset_model_index()

        # ADDED: Initialize memory bank and shared context (Keep as these are not MCP related)
        self.memory_bank: Dict[str, Dict[str, Any]] = {}
        self.shared_context: Dict[str, Dict[str, Any]] = {"global": {}} # Example structure

    def reset_model_index(self):
        """Reset the model index counter."""
        self.model_index = 0

    @property
    def tools(self) -> List[Dict]:
        """Returns the combined list of local and MCP tool schemas."""
        mcp_schemas = self.mcp_helper.get_mcp_tool_schemas() if self.mcp_helper else []
        return self.local_tools + mcp_schemas

    def add_tools(self, tools: List[Dict]):
        """
        Add LOCAL tool definitions (schemas) that the LLM can call via registered Python functions.
        These should be in the format expected by litellm (e.g., OpenAI tool format).
        MCP tools are added separately via discover_mcp_tools after connection (handled by MCPHelper).

        :param tools: A list of tool schema dictionaries for LOCAL functions.
        """
        # Get existing local names and potential MCP names for conflict check
        existing_local_names = set(self.local_tool_functions.keys()) | \
                               {t['function']['name'] for t in self.local_tools if t.get("function", {}).get("name")}
        mcp_tool_map = self.mcp_helper.get_mcp_tools_map() if self.mcp_helper else {}

        added_count = 0
        for tool in tools:
            tool_name = tool.get("function", {}).get("name")
            if not tool_name:
                logger.warning("Skipping tool schema with missing function name.")
                continue

            # Check conflict with existing local and MCP tools
            if tool_name in existing_local_names:
                 logger.warning(f"Local tool name '{tool_name}' conflicts with an existing local tool. Overwriting schema if already present.")
            if tool_name in mcp_tool_map:
                 logger.warning(f"Local tool name '{tool_name}' conflicts with a prefixed MCP tool name ('{tool_name}'). This might cause issues.")
            if tool_name.startswith("mcp_"):
                logger.warning(f"Local tool name '{tool_name}' starts with reserved prefix 'mcp_'. Skipping.")
                continue

            # Append to local_tools (overwrite if name exists? Current logic appends duplicates)
            # Let's prevent duplicates by name
            if not any(t.get("function", {}).get("name") == tool_name for t in self.local_tools):
                self.local_tools.append(tool)
                existing_local_names.add(tool_name) # Add to set after successful add
                added_count += 1
            else:
                 logger.info(f"Local tool schema '{tool_name}' already exists. Skipping duplicate add.")


        if self.verbose:
            logger.debug(f"🛠️ Added {added_count} local tool schemas.")

    def register_tool_function(self, func: Callable):
        """
        Register the Python function that implements a LOCAL tool.
        The function's __name__ must match the 'name' defined in the tool schema added via add_tools.

        :param func: The callable function implementing the tool.
        """
        tool_name = func.__name__
        # Ensure a schema exists for this function in local_tools
        if not any(t.get("function", {}).get("name") == tool_name for t in self.local_tools):
             logger.warning(f"Registering function '{tool_name}' but no corresponding schema found in self.local_tools. Add schema via add_tools().")

        if tool_name in self.local_tool_functions:
             logger.warning(f"Overwriting registered local tool function '{tool_name}'")

        # Check for conflicts with MCP tool names (prefixed)
        mcp_tool_map = self.mcp_helper.get_mcp_tools_map() if self.mcp_helper else {}
        if tool_name in mcp_tool_map:
             logger.warning(f"Local function name '{tool_name}' conflicts with a prefixed MCP tool name.")

        self.local_tool_functions[tool_name] = func
        if self.verbose:
            logger.debug(f"🔧 Registered local function for tool: {tool_name}")

    def process_prompt(self, initial_input: Optional[str] = None):
        """
        Synchronous version of process_prompt_async.
        Wraps the async version using asyncio.run() for backward compatibility.
        :param initial_input: Optional initial string input for the chain.
        """
        # Ensure MCP connections are handled if needed via helper
        if self.mcp_helper and not self.mcp_helper.mcp_sessions: # Check helper's sessions
             logger.debug("Sync call: Connecting MCP via helper before running chain...")
             self.mcp_helper.connect_mcp() # Run sync connection wrapper on helper

        try:
            # Pass the initial_input (which could be None) to the async version
            # Handle running loop detection
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # If in a running loop (e.g., Jupyter), create a task
                    logger.debug("Detected running event loop. Creating task for process_prompt_async.")
                    # Running sync wrapper from async context is tricky.
                    # Simplest approach for compatibility is often a new loop,
                    # but this can have side effects. Raising error is safer.
                    # Forcing via run_until_complete in new loop as per original logic.
                    new_loop = asyncio.new_event_loop()
                    try:
                        return new_loop.run_until_complete(self.process_prompt_async(initial_input))
                    finally:
                        new_loop.close()
                else:
                    # Loop exists but isn't running, can use run_until_complete
                    return loop.run_until_complete(self.process_prompt_async(initial_input))
            except RuntimeError: # No running loop
                # Need to create a new event loop via asyncio.run()
                return asyncio.run(self.process_prompt_async(initial_input))

        except Exception as e:
            logger.error(f"Error in process_prompt sync wrapper: {e}", exc_info=True)
            raise # Re-raise the exception

    async def process_prompt_async(self, initial_input: Optional[str] = None):
        """
        Asynchronous version of process_prompt.
        This is the main implementation that handles MCP and async operations.
        :param initial_input: Optional initial string input for the chain.
        """
        # Reset model index at the start of each chain
        self.reset_model_index()

        # Ensure MCP is connected if configured and not already connected, via helper
        if self.mcp_helper and not self.mcp_helper.mcp_sessions:
            logger.info("MCP helper configured but not connected. Connecting...")
            await self.mcp_helper.connect_mcp_async()
            # Check if connection succeeded for verbose logging
            if self.mcp_helper and not self.mcp_helper.mcp_sessions:
                logger.warning("MCP connection attempt via helper failed. Proceeding without MCP tools.")

        # Handle optional initial input
        actual_initial_input = initial_input if initial_input is not None else ""

        # Decide if the input is just placeholder/empty
        using_prompt_only = not actual_initial_input or actual_initial_input.strip() == "" or actual_initial_input == "Please follow these instructions"

        result = actual_initial_input # Initialize result with the actual input or empty string

        if self.verbose:
            logger.debug("\n" + "="*50)
            logger.debug("🔄 Starting Prompt Chain")
            logger.debug("="*50)
            logger.debug(f"\n📝 Initial Input:\n{actual_initial_input}\n")

        chain_history = [{
            "step": 0,
            "input": actual_initial_input,
            "output": actual_initial_input,
            "type": "initial"
        }]

        # Store initial step if requested
        if self.store_steps:
            self.step_outputs["step_0"] = {
                "type": "initial",
                "output": actual_initial_input
            }

        # Keep track of messages *within* the current chain execution
        # Initialize with user role, content might be empty if using_prompt_only
        current_messages = [{"role": "user", "content": "" if using_prompt_only else actual_initial_input}]
        # Filter out the initial empty message if it exists and there are more steps
        if current_messages and not current_messages[0]["content"] and len(self.instructions) > 0:
             current_messages = []


        # Initialize step_num here for potential use in error reporting
        step_num = 0

        # == Define helper callbacks for AgenticStepProcessor ==
        # These callbacks need access to 'self'

        async def llm_runner_callback(messages: List[Dict], tools: List[Dict], tool_choice: str = None, agentic_instruction=None) -> Any:
            # Determine model for agentic step: use model_name if set, else default to first model
            model_name = self.models[0] if self.models else "openai/gpt-4o" # Default
            params = self.model_params[0] if self.model_params else {}
            # If called from an AgenticStepProcessor, allow explicit model override
            if agentic_instruction is not None and hasattr(agentic_instruction, "model_name") and agentic_instruction.model_name:
                model_name = agentic_instruction.model_name
                # Also get model_params from the AgenticStepProcessor if available
                if hasattr(agentic_instruction, "model_params"):
                    params = agentic_instruction.model_params
            if not self.models:
                 logger.warning("[Agentic Callback] No models defined in PromptChain, using fallback 'openai/gpt-4o'.")

            logger.debug(f"[Agentic Callback] Running LLM: {model_name} with {len(messages)} messages. Tools: {len(tools) if tools else 0}")
            # Use the existing static method, ensures consistent calling
            return await PromptChain.run_model_async(
                model_name=model_name,
                messages=messages,
                params=params,
                tools=tools,
                tool_choice=tool_choice
            )

        # --- REVISED tool_executor_callback using MCPHelper ---
        async def tool_executor_callback(tool_call: Any) -> str:
            """
            Callback passed to AgenticStepProcessor.
            Determines tool type and calls the appropriate internal helper.
            Returns JSON string representation of the tool result or error.
            """
            # Import the helper function from agentic_step_processor.py
            from promptchain.utils.agentic_step_processor import get_function_name_from_tool_call
            
            function_name = get_function_name_from_tool_call(tool_call)
            logger.debug(f"[Agentic Callback] Received request to execute tool: {function_name}")

            if not function_name:
                logger.error("[Agentic Callback] Tool call missing function name.")
                return json.dumps({"error": "Tool call missing function name."})

            # Dispatch to the appropriate internal helper method
            if function_name in self.local_tool_functions:
                logger.debug(f"[Agentic Callback] Dispatching to local tool helper for: {function_name}")
                result_str = await self._execute_local_tool_internal(tool_call)
                return result_str
            elif self.mcp_helper and function_name in self.mcp_helper.get_mcp_tools_map(): # Check helper's map
                logger.debug(f"[Agentic Callback] Dispatching to MCP tool helper for: {function_name}")
                # Call the helper's execution method
                result_str = await self.mcp_helper.execute_mcp_tool(tool_call)
                return result_str
            else:
                logger.warning(f"[Agentic Callback] Tool function '{function_name}' not found locally or via MCP.")
                return json.dumps({"error": f"Tool function '{function_name}' is not available."})
        # --- END of REVISED tool_executor_callback ---

        # == End of Helper Callbacks ==

        # Get combined list of tools available for this run using the property
        available_tools = self.tools

        try:
            for step, instruction in enumerate(self.instructions):
                step_num = step + 1
                if self.verbose:
                    logger.debug("\n" + "-"*50)
                    logger.debug(f"Step {step_num}:")
                    logger.debug("-"*50)

                # Determine input for this step based on history mode
                step_input_content = result

                # Prepare messages for model call if it's a model step
                step_input_messages = [] # Start fresh for logic below
                if self.full_history:
                    step_input_messages = list(current_messages) # Use a copy
                # If not full history, messages determined later based on step type

                step_type = "unknown"
                step_model_params = None
                step_output = None # Output of the current step

                # Check if this step is a function, an agentic processor, or an instruction
                if callable(instruction) and not isinstance(instruction, AgenticStepProcessor): # Explicitly exclude AgenticStepProcessor here
                    step_type = "function"
                    func_name = getattr(instruction, '__name__', 'partial_function')
                    if isinstance(instruction, functools.partial):
                         func_name = getattr(instruction.func, '__name__', 'partial_function')

                    if self.verbose:
                        logger.debug(f"\n🔧 Executing Local Function: {func_name}")

                    start_time = time.time()
                    try:
                        # --- Execute local function --- 
                        sig = inspect.signature(instruction)
                        takes_args = len(sig.parameters) > 0

                        # Determine how to call the function based on signature and type
                        if asyncio.iscoroutinefunction(instruction):
                            if takes_args:
                                step_output = await instruction(step_input_content)
                            else:
                                step_output = await instruction()
                        else:
                            # Run synchronous function in a thread pool executor
                            loop = asyncio.get_running_loop()
                            if takes_args:
                                partial_func = functools.partial(instruction, step_input_content)
                                step_output = await loop.run_in_executor(None, partial_func)
                            else:
                                step_output = await loop.run_in_executor(None, instruction)

                        if not isinstance(step_output, str):
                            logger.warning(f"Function {func_name} did not return a string. Attempting to convert. Output type: {type(step_output)}")
                            step_output = str(step_output)

                    except Exception as e:
                        logger.error(f"Error executing function {func_name}: {e}", exc_info=self.verbose)
                        logger.error(f"\n❌ Error in chain processing at step {step_num}: Function execution failed.")
                        raise # Re-raise the exception to stop processing

                    end_time = time.time()
                    step_time = end_time - start_time

                    if self.verbose:
                        logger.debug(f"Function executed in {step_time:.2f} seconds.")
                        logger.debug(f"Output:\n{step_output}")

                    # Add function output as assistant message if using full history
                    if self.full_history:
                        current_messages.append({"role": "assistant", "content": step_output})
                    # If not full history, 'result' (set later) holds the output for the next step


                elif isinstance(instruction, AgenticStepProcessor):
                    step_type = "agentic_step"
                    if self.verbose:
                        logger.debug(f"\n🤖 Executing Agentic Step: Objective = {instruction.objective[:100]}...")

                    start_time = time.time()
                    try:
                        # Agentic step receives the current input content
                        step_output = await instruction.run_async(
                            initial_input=step_input_content, # Pass current content
                            available_tools=available_tools, # Pass all available tools (combined)
                            llm_runner=lambda messages, tools, tool_choice=None: llm_runner_callback(messages, tools, tool_choice, agentic_instruction=instruction), # Pass the LLM runner helper with agentic_instruction
                            tool_executor=tool_executor_callback # Pass the revised tool executor helper
                        )
                        # Ensure output is string
                        if not isinstance(step_output, str):
                            logger.warning(f"Agentic step did not return a string. Converting. Output type: {type(step_output)}")
                            step_output = str(step_output)

                    except Exception as e:
                        logger.error(f"Error executing agentic step: {e}", exc_info=self.verbose)
                        logger.error(f"\n❌ Error in chain processing at step {step_num} (Agentic Step): {e}")
                        raise # Re-raise the exception

                    end_time = time.time()
                    step_time = end_time - start_time

                    if self.verbose:
                        logger.debug(f"Agentic step executed in {step_time:.2f} seconds.")
                        logger.debug(f"Output:\n{step_output}")

                    # Treat agentic output like model/function output. Add to history if using full history.
                    if self.full_history:
                        current_messages.append({"role": "assistant", "content": step_output})


                else: # It's a model instruction (string)
                    step_type = "model"
                    instruction_str = str(instruction) # Work with string version

                    # === Logic for handling {input} append vs replace === 
                    should_append_input = False
                    load_instruction_id_or_path = instruction_str # Default
                    if instruction_str.endswith("{input}"):
                        should_append_input = True
                        load_instruction_id_or_path = instruction_str[:-len("{input}")].strip()
                        if not load_instruction_id_or_path:
                            load_instruction_id_or_path = "" # Handle instruction being just "{input}"
                    # === End {input} handling === 

                    resolved_instruction_content = "" # Initialize resolved content

                    # --- Try resolving using PrePrompt or loading file path --- 
                    if load_instruction_id_or_path: # Only load if we have an ID/path
                        try:
                            resolved_instruction_content = self.preprompter.load(load_instruction_id_or_path)
                            if self.verbose:
                                logger.debug(f"Resolved instruction '{load_instruction_id_or_path}' using PrePrompt.")
                        except FileNotFoundError:
                            if os.path.isfile(load_instruction_id_or_path):
                                try:
                                    with open(load_instruction_id_or_path, 'r', encoding='utf-8') as file:
                                        resolved_instruction_content = file.read()
                                    if self.verbose:
                                        logger.debug(f"Loaded instruction from file path: {load_instruction_id_or_path}")
                                except Exception as e:
                                    logger.warning(f"Could not read instruction file '{load_instruction_id_or_path}': {e}. Treating as literal.")
                                    resolved_instruction_content = load_instruction_id_or_path # Fallback
                            else:
                                if self.verbose:
                                    logger.debug(f"Treating instruction '{load_instruction_id_or_path}' as literal template (not found via PrePrompt or as file path).")
                                resolved_instruction_content = load_instruction_id_or_path # Use literal
                        except (ValueError, IOError) as e:
                            logger.warning(f"Error loading instruction '{load_instruction_id_or_path}' via PrePrompt: {e}. Treating as literal.")
                            resolved_instruction_content = load_instruction_id_or_path # Fallback
                        except Exception as e:
                             logger.error(f"Unexpected error loading instruction '{load_instruction_id_or_path}' via PrePrompt: {e}. Treating as literal.")
                             resolved_instruction_content = load_instruction_id_or_path # Fallback
                    else: # If load_instruction_id_or_path was empty (e.g., instruction was just "{input}")
                         resolved_instruction_content = "" # Start with empty content

                    # --- Prepare final prompt for model --- 
                    if should_append_input:
                        separator = "\n\n" if resolved_instruction_content and step_input_content else ""
                        prompt_for_model = resolved_instruction_content + separator + step_input_content
                    else:
                        # Standard replace {input} if it exists in the resolved content
                        prompt_for_model = resolved_instruction_content.replace("{input}", step_input_content)

                    if self.model_index >= len(self.models):
                        logger.error(f"Model index {self.model_index} out of range for available models ({len(self.models)}).")
                        raise IndexError(f"Not enough models configured for instruction at step {step_num}")

                    model = self.models[self.model_index]
                    step_model_params = self.model_params[self.model_index]
                    self.model_index += 1 # Increment for the next model step

                    if self.verbose:
                        logger.debug(f"\n🤖 Using Model: {model}")
                        if step_model_params:
                            logger.debug(f"Parameters: {step_model_params}")
                        logger.debug(f"\nInstruction Prompt (for model):\n{prompt_for_model}")

                    # Add the user's prompt message for this step
                    user_message_this_step = {"role": "user", "content": prompt_for_model}

                    # Determine messages to send to the model based on history mode
                    if self.full_history:
                        current_messages.append(user_message_this_step) # Add to running history
                        messages_for_llm = current_messages
                    else:
                        # If not full history, only send this step's prompt
                        messages_for_llm = [user_message_this_step]


                    # ===== Tool Calling Logic integrated with Model Call =====
                    response_message_dict = await self.run_model_async(
                        model_name=model,
                        messages=messages_for_llm, # Pass appropriate history
                        params=step_model_params,
                        tools=available_tools if available_tools else None, # Pass combined tools
                        tool_choice="auto" if available_tools else None
                    )

                    # Add model's response to history if using full history
                    if self.full_history:
                        current_messages.append(response_message_dict)

                    # Process potential tool calls from the response dictionary
                    tool_calls = response_message_dict.get('tool_calls') # Access as dict

                    if tool_calls:
                        if self.verbose:
                            logger.debug(f"\n🛠️ Model requested tool calls: {[tc['function']['name'] for tc in tool_calls]}")

                        # Prepare list to hold tool result messages
                        tool_results_messages = []

                        # Process each tool call
                        for tool_call in tool_calls:
                            # Ensure tool_call is a dictionary before accessing keys
                            if not isinstance(tool_call, dict):
                                logger.warning(f"Skipping invalid tool_call item: {tool_call}")
                                continue

                            function_call_info = tool_call.get('function', {})
                            function_name = function_call_info.get('name')
                            function_args_str = function_call_info.get('arguments', '{}')
                            tool_call_id = tool_call.get('id')

                            if not function_name or not tool_call_id:
                                logger.warning(f"Skipping tool call with missing name or id: {tool_call}")
                                continue

                            tool_output_str = None # Initialize tool output string

                            # Create a temporary object mimicking LiteLLM's tool_call structure if needed by helpers
                            # Or adapt helpers to take dicts
                            # Let's adapt helpers to work with dicts if possible, but create object if needed
                            class TempToolCall:
                                def __init__(self, tc_dict):
                                    self.id = tc_dict.get('id')
                                    func_dict = tc_dict.get('function', {})
                                    class TempFunc:
                                        name = func_dict.get('name')
                                        arguments = func_dict.get('arguments')
                                    self.function = TempFunc()

                            tool_call_obj = TempToolCall(tool_call)

                            # --- Routing: Local vs MCP --- 
                            if function_name in self.local_tool_functions:
                                logger.debug(f"  - Executing local tool via internal helper: {function_name}")
                                tool_output_str = await self._execute_local_tool_internal(tool_call_obj) # Pass object
                            elif self.mcp_helper and function_name in self.mcp_helper.get_mcp_tools_map():
                                logger.debug(f"  - Executing MCP tool via helper: {function_name}")
                                tool_output_str = await self.mcp_helper.execute_mcp_tool(tool_call_obj) # Pass object to helper
                            else: # Tool name not found
                                logger.error(f"Tool function '{function_name}' requested by model not registered.")
                                tool_output_str = json.dumps({"error": f"Tool function '{function_name}' is not available."})

                            # Append result message for this tool call
                            tool_results_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": function_name, # Use the name LLM called
                                "content": tool_output_str, # Content must be string
                            })

                        # Prepare messages for the second model call
                        # CORRECTED: Ensure the assistant message with tool_calls is included
                        messages_for_follow_up = list(messages_for_llm) # Start with the messages sent initially
                        messages_for_follow_up.append(response_message_dict) # Add the assistant message containing the tool_calls
                        messages_for_follow_up.extend(tool_results_messages) # Add the tool results

                        # Update the main history if needed (before potentially overriding messages_for_follow_up)
                        if self.full_history:
                             # The assistant message was already added earlier if full_history is True
                             # Now add the tool results to the persistent history
                             current_messages.extend(tool_results_messages)
                             # Use the full updated history for the next call if full_history is enabled
                             messages_for_follow_up = list(current_messages)


                        if self.verbose:
                                logger.debug("\n🔄 Sending tool results back to model...")
                                # Add detailed logging of messages being sent
                                logger.debug(f"Messages for follow-up call ({len(messages_for_follow_up)} messages):")
                                for msg_idx, msg in enumerate(messages_for_follow_up):
                                    logger.debug(f"  [{msg_idx}] Role: {msg.get('role')}, Content: {str(msg.get('content', ''))[:100]}..., ToolCalls: {bool(msg.get('tool_calls'))}")

                        # Second model call with tool results
                        final_response_message_dict = await self.run_model_async(
                            model_name=model, # Use the same model
                            messages=messages_for_follow_up, # Pass history including tool results
                            params=step_model_params,
                            tools=available_tools, # Pass tools again
                            tool_choice="auto" # Or maybe "none"? Stick with auto.
                        )

                        # Add the final response to history if using full history
                        if self.full_history:
                             current_messages.append(final_response_message_dict)

                        # The final text output for this step is the content of this message
                        step_output = final_response_message_dict.get('content', '')

                    else: # No tool calls in the initial response
                        # The output is simply the content of the first response
                        step_output = response_message_dict.get('content', '')
                        # response_message_dict was already added to current_messages if full_history

                    # ===== Tool Calling Logic End =====

                    if self.verbose:
                        logger.debug(f"\nFinal Output for Step: {step_output if step_output is not None else 'None'}")

                # --- After Function, Agentic, or Model step --- 

                # Update 'result' for the next step, ensuring it's a string
                result = str(step_output) if step_output is not None else ""

                # --- Chain History Update --- 
                step_input_actual_for_history = step_input_content
                if step_type == "model":
                   step_input_actual_for_history = str(instruction) # Record original instruction
                elif step_type == "function":
                    func_name_hist = getattr(instruction, '__name__', 'partial_function')
                    if isinstance(instruction, functools.partial): func_name_hist = getattr(instruction.func, '__name__', 'partial_function')
                    step_input_actual_for_history = f"Function: {func_name_hist}"
                elif step_type == "agentic_step":
                    step_input_actual_for_history = f"Agentic Step: {instruction.objective[:50]}..."


                chain_history.append({
                    "step": step_num,
                    "input": step_input_actual_for_history, # Input *to* the step
                    "output": result, # Final output *of* the step
                    "type": step_type,
                    "model_params": step_model_params if step_type == "model" else None
                })

                # Store step output if requested
                if self.store_steps:
                    self.step_outputs[f"step_{step_num}"] = {
                        "type": step_type,
                        "output": result,
                        "model_params": step_model_params if step_type == "model" else None
                    }

                # Check chainbreakers
                break_chain = False
                for breaker in self.chainbreakers:
                    try:
                        # CORRECTED: Indentation for lines within the try block
                        sig = inspect.signature(breaker)
                        step_info = chain_history[-1] # Pass the latest history entry
                        num_params = len(sig.parameters)

                        if num_params >= 3:
                            should_break, break_reason, break_output = breaker(step_num, result, step_info)
                        elif num_params >= 2:
                            # Provide default step_info if breaker only takes 2 args
                            should_break, break_reason, break_output = breaker(step_num, result)
                        else:
                             logger.warning(f"Chainbreaker function {breaker.__name__} takes {num_params} arguments, expected 2 or 3. Skipping.")
                             continue # Skip this breaker

                        # CORRECTED: Indentation for if should_break and its contents
                        if should_break:
                            if self.verbose:
                                logger.debug("\n" + "="*50)
                                logger.debug(f"⛔ Chain Broken at Step {step_num}: {break_reason}")
                                logger.debug("="*50)
                                # CORRECTED: Indentation for nested if
                                if break_output is not None and break_output != result:
                                    logger.debug("\n📊 Modified Output:")
                                    logger.debug(f"{break_output}\n")

                            # CORRECTED: Alignment for lines after if break_output is not None
                            if break_output is not None:
                                result = str(break_output) # Update the final result
                                # Update history/stored output if modified
                                chain_history[-1]["output"] = result
                                if self.store_steps:
                                    self.step_outputs[f"step_{step_num}"]["output"] = result

                            # CORRECTED: Alignment
                            break_chain = True # Set flag to exit outer loop
                            break # Exit inner chainbreaker loop

                    # CORRECTED: Alignment for except
                    except Exception as breaker_ex:
                        logger.error(f"Error executing chainbreaker {breaker.__name__}: {breaker_ex}", exc_info=self.verbose)

                if break_chain:
                    break # Exit the main instruction loop

        except Exception as e:
            logger.error(f"\n❌ Error in chain processing at step {step_num}: {e}", exc_info=True) # Use full traceback
            raise # Re-raise the exception after logging

        if self.verbose:
            logger.debug("\n" + "="*50)
            logger.debug("✅ Chain Completed" if not break_chain else "🔗 Chain Broken")
            logger.debug("="*50)
            logger.debug("\n📊 Final Output:")
            logger.debug(f"{result}\n")

        # Return the final result or the full history based on the flag
        return chain_history if self.full_history else result


    @staticmethod
    def run_model(model_name: str, messages: List[Dict], params: dict = None,
                  tools: List[Dict] = None, tool_choice: str = None) -> Dict:
        """
        Synchronous version of run_model_async.
        Wraps the async version using asyncio.run() for backward compatibility.
        Returns the message dictionary from the response.
        """
        try:
            # Handle running loop detection
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    raise RuntimeError("run_model (sync) called from within a running event loop. Use run_model_async instead.")
                else:
                    return loop.run_until_complete(
                        PromptChain.run_model_async(
                            model_name, messages, params, tools, tool_choice
                        )
                    )
            except RuntimeError: # No running loop
                 return asyncio.run(PromptChain.run_model_async(
                     model_name, messages, params, tools, tool_choice
                 ))
        except Exception as e:
             logger.error(f"Error in run_model sync wrapper for {model_name}: {e}", exc_info=True)
             return {"role": "assistant", "content": f"Error running model: {e}", "error": True}


    @staticmethod
    async def run_model_async(model_name: str, messages: List[Dict], params: dict = None,
                           tools: List[Dict] = None, tool_choice: str = None) -> Dict:
        """
        Asynchronous version of run_model. Calls LiteLLM's acompletion.
        Returns the message dictionary (e.g., response['choices'][0]['message'])
        which may contain 'content' or 'tool_calls'.
        """
        if not messages:
            logger.warning(f"run_model_async called for {model_name} with empty messages list.")
            return {"role": "assistant", "content": "", "warning": "Empty messages list"}

        try:
            model_params = {
                "model": model_name,
                "messages": messages
            }

            # Add any custom parameters from the chain definition
            if params:
                model_params.update(params)

            # Add tools if provided
            if tools:
                model_params["tools"] = tools
                
                # Handle tool_choice with priority:
                # 1. Explicitly passed tool_choice parameter
                # 2. tool_choice in params dictionary
                # 3. Default "auto"
                if tool_choice:
                    model_params["tool_choice"] = tool_choice
                elif params and "tool_choice" in params:
                    # Don't override if explicitly passed
                    if "tool_choice" not in model_params:
                        model_params["tool_choice"] = params["tool_choice"]
                else:
                    model_params["tool_choice"] = "auto"

            if logger.level == logging.DEBUG: # Only log detailed params if verbose/debug
                log_params_safe = model_params.copy()
                log_params_safe['messages'] = f"<{len(messages)} messages>"
                if 'tools' in log_params_safe:
                    log_params_safe['tools'] = f"<{len(log_params_safe['tools'])} tools>"
                logger.debug(f"Calling acompletion with params: {log_params_safe}")


            response = await acompletion(**model_params)

            # Check response structure (LiteLLM usually returns this structure)
            if not response or 'choices' not in response or not response['choices']:
                if isinstance(response, (asyncio.StreamReader, object)): # Basic check for stream object
                    logger.warning("run_model_async received unexpected response type (possibly stream). Returning empty.")
                    return {"role": "assistant", "content": "", "warning": "Received unexpected stream response"}
                else:
                    logger.error(f"Invalid response structure received from LiteLLM for model {model_name}: {response}")
                    raise ValueError(f"Invalid response structure received from LiteLLM for model {model_name}.")

            # Return the full message object from the first choice as a dictionary
            message = response['choices'][0]['message']
            
            # Add debug for message format and content
            if logger.level == logging.DEBUG:
                if hasattr(message, 'model_dump'):
                    logger.debug(f"Message from LiteLLM (model_dump available): {message}")
                    try:
                        dump = message.model_dump()
                        logger.debug(f"Message dump: {dump}")
                        logger.debug(f"Message has tool_calls: {bool(dump.get('tool_calls'))}")
                    except Exception as e:
                        logger.debug(f"Error dumping message: {e}")
                elif isinstance(message, dict):
                    logger.debug(f"Message from LiteLLM (dict): {message}")
                    logger.debug(f"Message has tool_calls: {bool(message.get('tool_calls'))}")
                else:
                    logger.debug(f"Message from LiteLLM (type: {type(message)}): {message}")
                    logger.debug(f"Message has tool_calls attribute: {hasattr(message, 'tool_calls')}")
            
            if hasattr(message, 'model_dump'):
                return message.model_dump(exclude_unset=True) # Use Pydantic's method if available
            elif isinstance(message, dict):
                 return message # Already a dict
            else:
                 logger.warning(f"Unexpected message type in response: {type(message)}. Attempting basic conversion.")
                 return {"role": getattr(message, 'role', 'assistant'), "content": getattr(message, 'content', str(message)), "tool_calls": getattr(message, 'tool_calls', None)}


        except Exception as e:
            error_context = f"Error running model {model_name} asynchronously with {len(messages)} messages"
            if tools: error_context += f" and {len(tools)} tools"
            error_context += f": {str(e)}"
            logger.error(error_context, exc_info=True) # Log full traceback
            raise Exception(error_context) from e

    def get_step_output(self, step_number: int) -> Optional[dict]:
        """Retrieve output dictionary for a specific step, if stored."""
        if not self.store_steps:
            logger.warning("Step storage is not enabled. Cannot get step output. Initialize with store_steps=True")
            return None

        step_key = f"step_{step_number}"
        output = self.step_outputs.get(step_key)
        if output is None:
             logger.warning(f"Step {step_number} not found in stored outputs.")
        return output

    def add_techniques(self, techniques: List[str]) -> None:
        """
        Injects additional prompt engineering techniques into all string-based instructions.
        Each technique can include an optional parameter using the format "technique:parameter"
        """
        REQUIRED_PARAMS = { "role_playing": "profession/role", "style_mimicking": "author/style", "persona_emulation": "expert name", "forbidden_words": "comma-separated words" }
        OPTIONAL_PARAMS = { "few_shot": "number of examples", "reverse_prompting": "number of questions", "context_expansion": "context type", "comparative_answering": "aspects to compare", "tree_of_thought": "number of paths" }
        NO_PARAMS = { "step_by_step", "chain_of_thought", "iterative_refinement", "contrarian_perspective", "react" }
        prompt_techniques = {
            "role_playing": lambda p: f"You are an experienced {p} explaining in a clear and simple way. Use relatable examples.",
            "step_by_step": lambda _: "Explain your reasoning step-by-step before providing the final answer.",
            "few_shot": lambda p: f"Include {p or 'a few'} examples to demonstrate the pattern before generating your answer.",
            "chain_of_thought": lambda _: "Outline your reasoning in multiple steps before delivering the final result.",
            "persona_emulation": lambda p: f"Adopt the persona of {p} in this field.",
            "context_expansion": lambda p: f"Consider {p or 'additional background'} context and relevant details in your explanation.",
            "reverse_prompting": lambda p: f"First, generate {p or 'key'} questions about this topic before answering.",
            "style_mimicking": lambda p: f"Emulate the writing style of {p} in your response.",
            "iterative_refinement": lambda _: "Iteratively refine your response to improve clarity and detail.",
            "forbidden_words": lambda p: f"Avoid using these words in your response: {p}. Use more precise alternatives.",
            "comparative_answering": lambda p: f"Compare and contrast {p or 'relevant'} aspects thoroughly in your answer.",
            "contrarian_perspective": lambda _: "Argue a contrarian viewpoint that challenges common beliefs.",
            "tree_of_thought": lambda p: f"Explore {p or 'multiple'} solution paths and evaluate each before concluding.",
            "react": lambda _: "Follow this process: \n1. Reason about the problem\n2. Act based on your reasoning\n3. Observe the results"
        }

        applied_techniques_text = ""
        # Process each technique and build the combined text
        for tech in techniques:
            parts = tech.split(":", 1)
            tech_name = parts[0]
            tech_param = parts[1] if len(parts) > 1 else None

            if tech_name not in prompt_techniques: raise ValueError(f"Technique '{tech_name}' not recognized.")
            if tech_name in REQUIRED_PARAMS and not tech_param: raise ValueError(f"Technique '{tech_name}' requires a {REQUIRED_PARAMS[tech_name]} parameter.")
            if tech_name in NO_PARAMS and tech_param: logger.warning(f"Technique '{tech_name}' doesn't use parameters, ignoring '{tech_param}'")

            technique_text = prompt_techniques[tech_name](tech_param)
            applied_techniques_text += technique_text + "\n" # Append with newline

        # CORRECTED: Indentation - Apply techniques after processing all of them
        # Apply the combined techniques text to all string instructions
        if applied_techniques_text: # Only apply if any techniques were added
            applied_techniques_text = applied_techniques_text.strip() # Remove trailing newline
            for i, instruction in enumerate(self.instructions):
                if isinstance(instruction, str):
                    # Append the techniques block after the original instruction
                    self.instructions[i] = instruction.strip() + "\n\n" + applied_techniques_text

            if self.verbose: # Log after applying
                logger.debug(f"Applied techniques to string instructions: {techniques}")

    # --- MCP Methods Removed (Moved to MCPHelper) ---
    # connect_mcp, connect_mcp_async, close_mcp, close_mcp_async

    # --- Async Context Manager (Calls MCPHelper methods) ---
    async def __aenter__(self):
        """Enter context manager, connect MCP if configured via helper."""
        if self.mcp_helper:
            await self.mcp_helper.connect_mcp_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, close MCP connections via helper."""
        if self.mcp_helper:
            await self.mcp_helper.close_mcp_async()

    # --- Internal Tool Execution Helpers (Local tool part remains) ---

    async def _execute_local_tool_internal(self, tool_call: Any) -> str:
        """
        Executes a registered local Python tool based on the tool_call object.
        Returns the result as a string (plain or JSON).
        """
        # Use the helper function for safe function name extraction
        from promptchain.utils.agentic_step_processor import get_function_name_from_tool_call
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
        
        tool_output_str = json.dumps({"error": f"Local tool '{function_name}' execution failed."})

        if not function_name or function_name not in self.local_tool_functions: # Check local_tool_functions
            logger.error(f"[Tool Helper] Local tool function '{function_name}' (ID: {tool_call_id}) not registered.")
            return json.dumps({"error": f"Local tool function '{function_name}' not registered."})

        try:
            # CORRECTED: Indentation for lines within the try block
            function_to_call = self.local_tool_functions[function_name] # Use local_tool_functions
            # Ensure args are valid JSON
            try:
                function_args = json.loads(function_args_str)
                if not isinstance(function_args, dict):
                     raise json.JSONDecodeError("Arguments must be a JSON object (dict)", function_args_str, 0)
            except json.JSONDecodeError as json_err:
                logger.error(f"[Tool Helper] Error decoding JSON arguments for local tool {function_name} (ID: {tool_call_id}): {function_args_str}. Error: {json_err}")
                return json.dumps({"error": f"Invalid JSON arguments for local tool {function_name}: {str(json_err)}"})

            if self.verbose:
                logger.debug(f"  [Tool Helper] Calling Local Tool: {function_name}(**{function_args}) (ID: {tool_call_id})")

            # Handle sync/async local functions
            if asyncio.iscoroutinefunction(function_to_call):
                 # CORRECTED: Indentation
                 result = await function_to_call(**function_args)
            else:
                 # CORRECTED: Indentation
                 loop = asyncio.get_running_loop()
                 partial_func = functools.partial(function_to_call, **function_args)
                 result = await loop.run_in_executor(None, partial_func)

            # CORRECTED: Alignment - These lines execute after result is obtained
            # Convert result to string (prefer JSON for structured data)
            if isinstance(result, (dict, list)):
                tool_output_str = json.dumps(result)
            else:
                tool_output_str = str(result) # Ensure string output for anything else

            if self.verbose:
                logger.debug(f"  [Tool Helper] Local Result (ID: {tool_call_id}): {tool_output_str[:150]}...")

        # CORRECTED: Alignment for except
        except Exception as e:
            logger.error(f"[Tool Helper] Error executing local tool {function_name} (ID: {tool_call_id}): {e}", exc_info=self.verbose)
            tool_output_str = json.dumps({"error": f"Error executing local tool {function_name}: {str(e)}"}) # Return error as JSON

        # CORRECTED: Alignment for return
        return tool_output_str

    # --- MCP Tool Execution Helper Removed (_execute_mcp_tool_internal) ---
    # --- Calls are now directed to self.mcp_helper.execute_mcp_tool ---