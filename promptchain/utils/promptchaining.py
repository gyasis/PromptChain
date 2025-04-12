from litellm import completion, acompletion
import os
from typing import Union, Callable, List, Literal, Dict, Optional, Any
from dotenv import load_dotenv
import inspect
from pydantic import BaseModel, Field, validator
from enum import Enum
import json
import asyncio
from contextlib import AsyncExitStack
import logging # Added for logging
import time
import functools # Added import

# Import PrePrompt
from .preprompt import PrePrompt # Assuming preprompt.py is in the same directory

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

# Load environment variables from .env file
load_dotenv("../../.env")

# Configure environment variables for API keys
# These will be loaded from the .env file
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Replace with your actual OpenAI API key
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"  # Replace with your actual Anthropic API key

# Setup logger
logger = logging.getLogger(__name__) # Added logger

### Python Class for Prompt Chaining

class ChainStep(BaseModel):
    step: int
    input: str
    output: str
    type: Literal["initial", "model", "function", "mcp_tool"]

class PromptChain:
    def __init__(self, models: List[Union[str, dict]], 
                 instructions: List[Union[str, Callable]], 
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
        :param instructions: List of instruction templates, prompt IDs, ID:strategy strings, or callable functions
        :param full_history: Whether to pass full chain history
        :param store_steps: If True, stores step outputs in self.step_outputs without returning full history
        :param verbose: If True, prints detailed output for each step with formatting
        :param chainbreakers: List of functions that can break the chain if conditions are met
                             Each function should take (step_number, current_output) and return
                             (should_break: bool, break_reason: str, final_output: Any)
        :param mcp_servers: Optional list of MCP server configurations.
                            Each dict should have 'id', 'type' ('stdio' supported), and type-specific args (e.g., 'script_path').
        :param additional_prompt_dirs: Optional list of paths to directories containing custom prompts for PrePrompt.
        """
        self.verbose = verbose
        # Extract model names and parameters
        self.models = []
        self.model_params = []
        
        # Count non-function instructions to match with models
        model_instruction_count = sum(1 for instr in instructions if not callable(instr))
        
        # Handle single model case
        if len(models) == 1:
            # Replicate the single model for all instructions
            models = models * model_instruction_count
        
        # Process models
        for model in models:
            if isinstance(model, dict):
                self.models.append(model["name"])
                self.model_params.append(model.get("params", {}))
            else:
                self.models.append(model)
                self.model_params.append({})

        # Validate model count
        if len(self.models) != model_instruction_count:
            raise ValueError(
                f"Number of models ({len(self.models)}) must match number of non-function instructions ({model_instruction_count})"
                "\nOr provide a single model to use for all instructions."
            )
            
        self.instructions = instructions
        self.full_history = full_history
        self.store_steps = store_steps
        # Initialize PrePrompt instance
        self.preprompter = PrePrompt(additional_prompt_dirs=additional_prompt_dirs)
        self.step_outputs = {}
        self.chainbreakers = chainbreakers or []
        self.tools = []  # Combined list of local and MCP tool schemas (prefixed for MCP)
        self.tool_functions = {} # Local Python functions map
        self.mcp_servers = mcp_servers or []
        self.mcp_sessions: Dict[str, ClientSession] = {} # Maps server_id to MCP session
        # Maps prefixed_tool_name -> {'original_schema': ..., 'server_id': ...}
        self.mcp_tools_map: Dict[str, Dict[str, Any]] = {}
        # Initialize exit_stack conditionally
        self.exit_stack = AsyncExitStack() if MCP_AVAILABLE and self.mcp_servers else None
        self.reset_model_index()

        if self.mcp_servers and not MCP_AVAILABLE:
            print("Warning: mcp_servers configured but 'mcp' library not installed. MCP tools will not be available.")

    def reset_model_index(self):
        """Reset the model index counter."""
        self.model_index = 0

    def add_tools(self, tools: List[Dict]):
        """
        Add LOCAL tool definitions (schemas) that the LLM can call via registered Python functions.
        These should be in the format expected by litellm (e.g., OpenAI tool format).
        MCP tools are added separately via discover_mcp_tools after connection.

        :param tools: A list of tool schema dictionaries for LOCAL functions.
        """
        # Check for name conflicts with existing local or MCP tools
        existing_tool_names = set(t['function']['name'] for t in self.tools) | set(self.tool_functions.keys())
        for tool in tools:
            tool_name = tool.get("function", {}).get("name")
            if not tool_name:
                print("Warning: Skipping tool schema with missing function name.")
                continue
            if tool_name in existing_tool_names:
                 print(f"Warning: Tool name '{tool_name}' conflicts with an existing local or MCP tool. Overwriting/Ignoring depends on registration order.")
            self.tools.append(tool)
            existing_tool_names.add(tool_name)

        if self.verbose:
            print(f"🛠️ Added {len(tools)} local tool schemas.")

    def register_tool_function(self, func: Callable):
        """
        Register the Python function that implements a LOCAL tool.
        The function's __name__ must match the 'name' defined in the tool schema added via add_tools.

        :param func: The callable function implementing the tool.
        """
        tool_name = func.__name__
        # Ensure a schema exists for this function
        if not any(t.get("function", {}).get("name") == tool_name for t in self.tools):
             print(f"Warning: Registering function '{tool_name}' but no corresponding schema found in self.tools. Add schema via add_tools().")

        if tool_name in self.tool_functions:
             print(f"Warning: Overwriting registered local tool function '{tool_name}'")
        # Check for conflicts with MCP tool names (prefixed) - unlikely but possible
        if tool_name in self.mcp_tools_map:
             print(f"Warning: Local function name '{tool_name}' conflicts with a prefixed MCP tool name.")

        self.tool_functions[tool_name] = func
        if self.verbose:
            print(f"🔧 Registered local function for tool: {tool_name}")

    def is_function(self, instruction: Union[str, Callable]) -> bool:
        """Check if an instruction is actually a function"""
        return callable(instruction)

    def process_prompt(self, initial_input: Optional[str] = None):
        """
        Synchronous version of process_prompt.
        Wraps the async version using asyncio.run() for backward compatibility.
        :param initial_input: Optional initial string input for the chain.
        """
        try:
            # Pass the initial_input (which could be None) to the async version
            return asyncio.run(self.process_prompt_async(initial_input))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                # We're already in an event loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.process_prompt_async(initial_input))
                finally:
                    loop.close()
            raise

    async def process_prompt_async(self, initial_input: Optional[str] = None):
        """
        Asynchronous version of process_prompt.
        This is the main implementation that handles MCP and async operations.
        :param initial_input: Optional initial string input for the chain.
        """
        # Reset model index at the start of each chain
        self.reset_model_index()

        # Handle optional initial input
        actual_initial_input = initial_input if initial_input is not None else ""
        
        # If the initial input is essentially empty, it's likely we're just using the prompt itself
        # or starting with a function that generates its own input.
        using_prompt_only = not actual_initial_input or actual_initial_input.strip() == "" or actual_initial_input == "Please follow these instructions"
        
        result = actual_initial_input # Initialize result with the actual input or empty string
        
        if self.verbose:
            print("\n" + "="*50)
            print("🔄 Starting Prompt Chain")
            print("="*50)
            print("\n📝 Initial Input:")
            print(f"{actual_initial_input}\n")
        
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
        # Initialize with empty content if no meaningful input provided
        current_messages = [{"role": "user", "content": actual_initial_input if not using_prompt_only else ""}]
        
        # Initialize step_num here for potential use in error reporting
        step_num = 0

        try:
            for step, instruction in enumerate(self.instructions):
                step_num = step + 1
                if self.verbose:
                    print("\n" + "-"*50)
                    print(f"Step {step_num}:")
                    print("-"*50)
                
                # Determine input for this step
                if self.full_history:
                    history_text = "\n".join(
                        f"Step {entry['step']}: {entry['output']}" 
                        for entry in chain_history if entry['type'] != 'initial'
                    )
                    content_to_process = f"Previous steps summary:\n{history_text}\n\nCurrent input: {result}"
                    step_input_messages = [{"role": "user", "content": content_to_process}]
                else:
                    content_to_process = result
                    if using_prompt_only and step == 0:
                        content_to_process = ""
                    step_input_messages = [{"role": "user", "content": content_to_process}]

                step_type = "unknown"
                step_model_params = None
                step_output = None

                # Check if this step is a function or an instruction
                if self.is_function(instruction):
                    step_type = "function"
                    if self.verbose:
                        # FIX: Handle functools.partial objects correctly for logging
                        func_name = instruction.func.__name__ if isinstance(instruction, functools.partial) else instruction.__name__
                        print(f"\n🔧 Executing Local Function: {func_name}")
                    
                    start_time = time.time()
                    try:
                        # --- Execute local function --- 
                        sig = inspect.signature(instruction)
                        takes_args = len(sig.parameters) > 0
                        
                        # Determine how to call the function based on signature and type
                        if asyncio.iscoroutinefunction(instruction):
                            if takes_args:
                                step_output = await instruction(content_to_process)
                            else:
                                step_output = await instruction()
                        else:
                            # Run synchronous function in a thread pool executor
                            loop = asyncio.get_running_loop()
                            if takes_args:
                                step_output = await loop.run_in_executor(None, instruction, content_to_process)
                            else:
                                step_output = await loop.run_in_executor(None, instruction)
                        
                        if not isinstance(step_output, str):
                            logger.warning(f"Function {func_name} did not return a string. Attempting to convert. Output type: {type(step_output)}")
                            step_output = str(step_output)

                    except Exception as e:
                        logger.error(f"Error executing function {func_name}: {e}", exc_info=self.verbose)
                        print(f"\n❌ Error in chain processing at step {step_num}: {e}")
                        raise # Re-raise the exception to stop processing
                    
                    end_time = time.time()
                    step_time = end_time - start_time
                    
                    if self.verbose:
                        print(f"Function executed in {step_time:.2f} seconds.")
                        print(f"Output:\n{step_output}")
                    
                    # Add function output as an assistant message for the next step
                    current_messages.append({"role": "assistant", "content": str(step_output)})

                else: # It's a model instruction (string)
                    step_type = "model"
                    instruction_str = str(instruction) # Work with string version

                    # === NEW LOGIC: Check for appended {input} ===
                    should_append_input = False
                    load_instruction_id_or_path = instruction_str # Default to the original string
                    if instruction_str.endswith("{input}"):
                        should_append_input = True
                        # Treat the part before {input} as the ID/path
                        load_instruction_id_or_path = instruction_str[:-len("{input}")].strip()
                        if not load_instruction_id_or_path: 
                            # Handle case where instruction was just "{input}" -> treat as empty base
                            load_instruction_id_or_path = ""
                    # === End NEW LOGIC ===

                    resolved_content = "" # Initialize resolved content

                    # --- Try resolving using PrePrompt or loading file path --- 
                    # Use the potentially modified load_instruction_id_or_path
                    if load_instruction_id_or_path: # Only try loading if we have an ID/path
                        try:
                            # Attempt to load as promptID or promptID:strategy
                            resolved_content = self.preprompter.load(load_instruction_id_or_path)
                            if self.verbose:
                                logger.info(f"Resolved instruction '{load_instruction_id_or_path}' using PrePrompt.")
                        except FileNotFoundError:
                            # Not found by PrePrompt, try as a direct file path
                            if os.path.isfile(load_instruction_id_or_path):
                                try:
                                    with open(load_instruction_id_or_path, 'r', encoding='utf-8') as file:
                                        resolved_content = file.read()
                                    if self.verbose:
                                        logger.info(f"Loaded instruction from file path: {load_instruction_id_or_path}")
                                except Exception as e:
                                    logger.warning(f"Could not read instruction file '{load_instruction_id_or_path}': {e}. Treating as literal.")
                                    resolved_content = load_instruction_id_or_path # Fallback to literal on read error
                            else:
                                # Not a file path either, treat as literal string template
                                if self.verbose:
                                    logger.info(f"Treating instruction '{load_instruction_id_or_path}' as literal template (not found via PrePrompt or as file path).")
                                resolved_content = load_instruction_id_or_path # Use the potentially stripped string
                        except (ValueError, IOError) as e:
                            # Error during PrePrompt loading (e.g., bad strategy JSON)
                            logger.warning(f"Error loading instruction '{load_instruction_id_or_path}' via PrePrompt: {e}. Treating as literal.")
                            resolved_content = load_instruction_id_or_path # Fallback to literal
                        except Exception as e:
                             # Catch any other unexpected error during PrePrompt load
                             logger.error(f"Unexpected error loading instruction '{load_instruction_id_or_path}' via PrePrompt: {e}. Treating as literal.")
                             resolved_content = load_instruction_id_or_path # Fallback to literal
                    else: # If load_instruction_id_or_path was empty (i.e., instruction was just "{input}")
                         resolved_content = "" # Start with empty content

                    # --- Prepare prompt for model --- 
                    # === MODIFIED PROMPT PREPARATION ===
                    if should_append_input:
                        # Append the runtime input *after* the loaded content
                        # Ensure separation if both parts exist
                        separator = "\n\n" if resolved_content and content_to_process else ""
                        prompt = resolved_content + separator + content_to_process 
                    else:
                        # Standard behavior: replace {input} *within* the loaded content
                        prompt = resolved_content.replace("{input}", content_to_process)
                    # === MODIFIED PROMPT PREPARATION END ===

                    if self.model_index >= len(self.models):
                        raise IndexError(f"Not enough models provided for instruction at step {step_num}")

                    model = self.models[self.model_index]
                    step_model_params = self.model_params[self.model_index]
                    self.model_index += 1
                    
                    if self.verbose:
                        print(f"\n🤖 Using Model: {model}")
                        if step_model_params:
                            print(f"Parameters: {step_model_params}")
                        print(f"\nInstruction Prompt:\n{prompt}")
                        # print(f"\nFull Prompt (Messages):\n{step_input_messages}") # Can be verbose

                    # Add the user's prompt message for this step
                    current_messages.append({"role": "user", "content": prompt})

                    # ===== Tool Calling Logic Start =====
                    response_message = await self.run_model_async(
                        model_name=model, 
                        messages=current_messages, # Pass full conversation history
                        params=step_model_params,
                        tools=self.tools if self.tools else None,
                        tool_choice="auto" if self.tools else None
                    )

                    current_messages.append(response_message) 

                    tool_calls = getattr(response_message, 'tool_calls', None)

                    if tool_calls:
                        if self.verbose:
                            print(f"\n🛠️ Model requested tool calls: {[tc.function.name for tc in tool_calls]}")
                        
                        tool_results_messages = []
                        for tool_call in tool_calls:
                            function_name = tool_call.function.name # This is the name LLM used (potentially prefixed)
                            function_args_str = tool_call.function.arguments 
                            tool_call_id = tool_call.id
                            tool_output = None # Initialize tool output

                            # --- Routing: Local vs MCP ---
                            if function_name in self.tool_functions:
                                # --- Execute Local Tool ---
                                try:
                                    function_to_call = self.tool_functions[function_name]
                                    try:
                                        function_args = json.loads(function_args_str)
                                    except json.JSONDecodeError:
                                         print(f"Error: Could not decode arguments for local tool {function_name}: {function_args_str}")
                                         tool_output = f"Error: Invalid arguments format for local tool {function_name}."
                                         function_args = None

                                    if function_args is not None:
                                        if self.verbose:
                                            print(f"  - Calling Local Tool: {function_name}({function_args})")
                                        tool_output = function_to_call(**function_args)
                                        if self.verbose:
                                            print(f"  - Local Result: {tool_output}")
                                    
                                except Exception as e:
                                    print(f"Error executing local tool {function_name}: {e}")
                                    tool_output = f"Error executing local tool {function_name}: {str(e)}"

                            elif function_name in self.mcp_tools_map:
                                # --- Execute MCP Tool ---
                                if not MCP_AVAILABLE or not experimental_mcp_client:
                                     print(f"Error: MCP tool '{function_name}' called, but MCP library/client is not available.")
                                     tool_output = f"Error: MCP library not available to call tool '{function_name}'."
                                else:
                                    try:
                                        mcp_info = self.mcp_tools_map[function_name]
                                        server_id = mcp_info['server_id']
                                        session = self.mcp_sessions.get(server_id)
                                        original_schema = mcp_info['original_schema']
                                        original_tool_name = original_schema['function']['name']

                                        if not session:
                                            print(f"Error: MCP Session '{server_id}' not found for tool '{function_name}'.")
                                            tool_output = f"Error: MCP session '{server_id}' unavailable."
                                        else:
                                            # Construct the object expected by call_openai_tool
                                            # It needs the original name, not the prefixed one.
                                            openai_tool_for_mcp = {
                                                "id": tool_call_id,
                                                "type": "function",
                                                "function": {
                                                    "name": original_tool_name, 
                                                    "arguments": function_args_str
                                                }
                                            }
                                            if self.verbose:
                                                print(f"  - Calling MCP Tool: {original_tool_name} on server {server_id} via {function_name}")
                                                print(f"    Args: {function_args_str}")
                                            
                                            # Call the MCP tool using experimental client
                                            call_result = await experimental_mcp_client.call_openai_tool(
                                                session=session,
                                                openai_tool=openai_tool_for_mcp 
                                            )

                                            # Extract result - Adjust based on actual MCP tool response structure
                                            # Example from docs: call_result.content[0].text
                                            if call_result and hasattr(call_result, 'content') and call_result.content and hasattr(call_result.content[0], 'text'):
                                                 tool_output = str(call_result.content[0].text) 
                                            elif call_result:
                                                 # Fallback if structure differs
                                                 tool_output = str(call_result) 
                                            else:
                                                 tool_output = f"MCP tool {original_tool_name} executed but returned no structured content."
                                            
                                            if self.verbose:
                                                 print(f"  - MCP Result: {tool_output}")

                                    except Exception as e:
                                        print(f"Error executing MCP tool {function_name} (original: {mcp_info.get('original_schema',{}).get('function',{}).get('name','?') if 'mcp_info' in locals() else '?'}) on {server_id if 'server_id' in locals() else '?'}: {e}")
                                        tool_output = f"Error executing MCP tool {function_name}: {str(e)}"
                            
                            else: # Tool name not found in local or MCP maps
                                print(f"Error: Tool function '{function_name}' not registered locally or via MCP.")
                                tool_output = f"Error: Tool function '{function_name}' is not available."

                            # Append result message (using the prefixed name LLM called)
                            tool_results_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": function_name, # Use the name LLM called
                                "content": str(tool_output), 
                            })
                        
                        current_messages.extend(tool_results_messages)

                        if self.verbose:
                                print("\n🔄 Sending tool results back to model...")

                        # Second model call with tool results
                        final_response_message = await self.run_model_async(
                            model_name=model,
                            messages=current_messages, 
                            params=step_model_params,
                            tools=self.tools, 
                            tool_choice="auto"
                        )
                        step_output = getattr(final_response_message, 'content', '')
                        current_messages.append(final_response_message)

                    else: # No tool calls in the initial response
                        step_output = getattr(response_message, 'content', '')
                        # response_message already added to current_messages

                    # ===== Tool Calling Logic End =====

                    if self.verbose:
                        print(f"\nFinal Output for Step: {step_output}")
                
                # Update result for the next step or final output
                result = step_output

                # --- Chain History Update ---
                # Determine input used for the step
                step_input_actual = content_to_process
                if step_type == "model":
                   # For model steps, record the *original* instruction string for clarity
                   step_input_actual = str(instruction)

                chain_history.append({
                    "step": step_num,
                    "input": step_input_actual, # Input *to* the step (original instruction for models)
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
                for breaker in self.chainbreakers:
                    sig = inspect.signature(breaker)
                    step_info = chain_history[-1] # Pass the latest history entry
                    if len(sig.parameters) >= 3:
                        should_break, break_reason, break_output = breaker(step_num, result, step_info)
                    else:
                        should_break, break_reason, break_output = breaker(step_num, result)
                    
                    if should_break:
                        if self.verbose:
                            print("\n" + "="*50)
                            print(f"⛔ Chain Broken at Step {step_num}: {break_reason}")
                            print("="*50)
                            if break_output != result:
                                print("\n📊 Modified Output:")
                                print(f"{break_output}\n")
                        
                        if break_output is not None:
                            result = break_output
                            chain_history[-1]["output"] = result
                            if self.store_steps:
                                self.step_outputs[f"step_{step_num}"]["output"] = result
                        
                        return result if not self.full_history else chain_history

        except Exception as e:
            if self.verbose:
                 # Use step_num which is initialized outside the loop
                print(f"\n❌ Error in chain processing at step {step_num}: {str(e)}")
            raise # Re-raise the exception after logging

        if self.verbose:
            print("\n" + "="*50)
            print("✅ Chain Completed")
            print("="*50)
            print("\n📊 Final Output:")
            print(f"{result}\n")
        
        return result if not self.full_history else chain_history

    @staticmethod
    def run_model(model_name: str, messages: List[Dict], params: dict = None,
                  tools: List[Dict] = None, tool_choice: str = None) -> Dict:
        """
        Synchronous version of run_model.
        Wraps the async version using asyncio.run() for backward compatibility.
        """
        try:
            return asyncio.run(PromptChain.run_model_async(
                model_name, messages, params, tools, tool_choice
            ))
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        PromptChain.run_model_async(
                            model_name, messages, params, tools, tool_choice
                        )
                    )
                finally:
                    loop.close()
            raise

    @staticmethod
    async def run_model_async(model_name: str, messages: List[Dict], params: dict = None,
                           tools: List[Dict] = None, tool_choice: str = None) -> Dict:
        """
        Asynchronous version of run_model.
        This is the main implementation that handles async LLM calls.
        """
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
                if tool_choice:
                     model_params["tool_choice"] = tool_choice
            
            response = await acompletion(**model_params)
            
            # Check if response structure is as expected
            if not response or 'choices' not in response or not response['choices']:
                # Handle case where response might be streaming chunks if stream=True was used
                if params and params.get("stream"):
                     print("Warning: run_model received streaming response chunks. Returning combined content.")
                     # Attempt to accumulate content from streaming response (basic example)
                     full_content = ""
                     async for chunk in response:
                         content = chunk.choices[0].delta.content
                         if content:
                             full_content += content
                     # Return a simulated message dictionary
                     return {"role": "assistant", "content": full_content}
                else:
                    raise ValueError("Invalid response structure received from LiteLLM.")
            
            # Return the full message object which might contain 'content' or 'tool_calls'
            message = response['choices'][0]['message']
            return message

        except Exception as e:
            # Provide more context in the error
            error_context = f"Error running model {model_name} asynchronously with {len(messages)} messages"
            if tools:
                error_context += f" and {len(tools)} tools"
            error_context += f": {str(e)}"
            # Consider logging the messages/params/tools here for debugging if needed
            # print(f"DEBUG: LiteLLM async call failed. Params: {model_params}") 
            raise Exception(error_context)

    def get_step_output(self, step_number: int) -> dict:
        """Retrieve output for a specific step."""
        if not self.store_steps:
            raise ValueError("Step storage is not enabled. Initialize with store_steps=True")
        
        step_key = f"step_{step_number}"
        if step_key not in self.step_outputs:
            raise ValueError(f"Step {step_number} not found")
        
        return self.step_outputs[step_key]

    def add_techniques(self, techniques: List[str]) -> None:
        """
        Injects additional prompt engineering techniques into all string-based instructions.
        Each technique can include an optional parameter using the format "technique:parameter"
        
        Some techniques REQUIRE parameters:
        - role_playing: requires profession/role (e.g., "role_playing:scientist")
        - style_mimicking: requires author/style (e.g., "style_mimicking:Richard Feynman")
        - persona_emulation: requires expert name (e.g., "persona_emulation:Warren Buffett")
        - forbidden_words: requires comma-separated words (e.g., "forbidden_words:maybe,probably,perhaps")
        
        :param techniques: List of technique strings (e.g., ["step_by_step", "role_playing:scientist"])
        """
        # Define which techniques require parameters
        REQUIRED_PARAMS = {
            "role_playing": "profession/role",
            "style_mimicking": "author/style",
            "persona_emulation": "expert name",
            "forbidden_words": "comma-separated words"
        }
        
        # Define which techniques accept optional parameters
        OPTIONAL_PARAMS = {
            "few_shot": "number of examples",
            "reverse_prompting": "number of questions",
            "context_expansion": "context type",
            "comparative_answering": "aspects to compare",
            "tree_of_thought": "number of paths"
        }
        
        # Techniques that don't use parameters
        NO_PARAMS = {
            "step_by_step",
            "chain_of_thought",
            "iterative_refinement",
            "contrarian_perspective",
            "react"
        }

        prompt_techniques = {
            "role_playing": lambda param: (
                f"You are an experienced {param} explaining in a clear and simple way. "
                "Use relatable examples."
            ),
            "step_by_step": lambda _: (
                "Explain your reasoning step-by-step before providing the final answer."
            ),
            "few_shot": lambda param: (
                f"Include {param or 'a few'} examples to demonstrate the pattern before generating your answer."
            ),
            "chain_of_thought": lambda _: (
                "Outline your reasoning in multiple steps before delivering the final result."
            ),
            "persona_emulation": lambda param: (
                f"Adopt the persona of {param} in this field."
            ),
            "context_expansion": lambda param: (
                f"Consider {param or 'additional background'} context and relevant details in your explanation."
            ),
            "reverse_prompting": lambda param: (
                f"First, generate {param or 'key'} questions about this topic before answering."
            ),
            "style_mimicking": lambda param: (
                f"Emulate the writing style of {param} in your response."
            ),
            "iterative_refinement": lambda _: (
                "Iteratively refine your response to improve clarity and detail."
            ),
            "forbidden_words": lambda param: (
                f"Avoid using these words in your response: {param}. "
                "Use more precise alternatives."
            ),
            "comparative_answering": lambda param: (
                f"Compare and contrast {param or 'relevant'} aspects thoroughly in your answer."
            ),
            "contrarian_perspective": lambda _: (
                "Argue a contrarian viewpoint that challenges common beliefs."
            ),
            "tree_of_thought": lambda param: (
                f"Explore {param or 'multiple'} solution paths and evaluate each before concluding."
            ),
            "react": lambda _: (
                "Follow this process: \n"
                "1. Reason about the problem\n"
                "2. Act based on your reasoning\n"
                "3. Observe the results"
            )
        }

        # Process each technique
        for tech in techniques:
            # Split technique and parameter if provided
            parts = tech.split(":", 1)
            tech_name = parts[0]
            tech_param = parts[1] if len(parts) > 1 else None

            # Validate technique exists
            if tech_name not in prompt_techniques:
                raise ValueError(
                    f"Technique '{tech_name}' not recognized.\n"
                    f"Available techniques:\n"
                    f"- Required parameters: {list(REQUIRED_PARAMS.keys())}\n"
                    f"- Optional parameters: {list(OPTIONAL_PARAMS.keys())}\n"
                    f"- No parameters: {list(NO_PARAMS)}"
                )

            # Validate required parameters
            if tech_name in REQUIRED_PARAMS and not tech_param:
                raise ValueError(
                    f"Technique '{tech_name}' requires a {REQUIRED_PARAMS[tech_name]} parameter.\n"
                    f"Use format: {tech_name}:parameter"
                )

            # Warning for unexpected parameters
            if tech_name in NO_PARAMS and tech_param:
                print(f"Warning: Technique '{tech_name}' doesn't use parameters, ignoring '{tech_param}'")

            # Generate technique text with parameter
            technique_text = prompt_techniques[tech_name](tech_param)

            # Apply to all string instructions
            for i, instruction in enumerate(self.instructions):
                if isinstance(instruction, str):
                    self.instructions[i] = instruction.strip() + "\n" + technique_text

    def connect_mcp(self):
        """
        Synchronous version of connect_mcp.
        Wraps the async version using asyncio.run() for backward compatibility.
        """
        try:
            return asyncio.run(self.connect_mcp_async())
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.connect_mcp_async())
                finally:
                    loop.close()
            raise

    async def connect_mcp_async(self):
        """
        Asynchronous version of connect_mcp.
        Connects to configured MCP servers and discovers their tools.
        """
        if not MCP_AVAILABLE or not experimental_mcp_client:
            if self.mcp_servers: # Only warn if servers were configured
                 print("Info: MCP library or experimental_mcp_client not installed/available, skipping MCP server connections and tool discovery.")
            return

        if not self.mcp_servers:
            if self.verbose:
                print("🔌 No MCP servers configured.")
            return

        if self.verbose:
            print(f"🔌 Connecting to {len(self.mcp_servers)} MCP server(s) and discovering tools...")

        # Clear previous MCP tool mappings if reconnecting
        self.mcp_tools_map = {}
        # We need to preserve local tools, so filter self.tools before potentially adding MCP tools again
        self.tools = [t for t in self.tools if not t.get("function",{}).get("name","").startswith("mcp_")]


        for server_config in self.mcp_servers:
            server_id = server_config.get("id")
            server_type = server_config.get("type")
            command = server_config.get("command")
            args = server_config.get("args")
            env = server_config.get("env") 

            if not server_id:
                 print("Warning: Skipping MCP server config with missing 'id'.")
                 continue
            if server_id in self.mcp_sessions:
                 print(f"Warning: Server ID '{server_id}' already connected, skipping duplicate connection attempt.")
                 continue # Skip if already connected

            try:
                if server_type == "stdio":
                    if not command:
                        print(f"Warning: Skipping stdio server '{server_id}' due to missing 'command'.")
                        continue
                    if not args or not isinstance(args, list):
                         print(f"Warning: Skipping stdio server '{server_id}' due to missing or invalid 'args' (must be a list).")
                         continue

                    server_params = StdioServerParameters(command=command, args=args, env=env)
                    
                    # --- Establish Session ---
                    # Correctly enter the async context manager from stdio_client
                    reader, writer = await self.exit_stack.enter_async_context(stdio_client(server_params))
                    # Create session with the obtained reader and writer
                    session = ClientSession(reader, writer)
                    await self.exit_stack.enter_async_context(session) # Manage session lifecycle
                    await session.initialize() # Initialize the MCP session
                    self.mcp_sessions[server_id] = session
                    if self.verbose:
                        print(f"  ✅ Connected to MCP server '{server_id}' via stdio (Cmd: {command} {' '.join(args)})")

                    # --- Discover and Register MCP Tools ---
                    try:
                        if self.verbose:
                            print(f"  🔍 Discovering tools on server '{server_id}'...")
                        # Get tools in OpenAI format
                        mcp_tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
                        
                        if self.verbose:
                            print(f"  🔬 Found {len(mcp_tools)} tools on '{server_id}'.")

                        existing_tool_names = set(t['function']['name'] for t in self.tools) | set(self.tool_functions.keys()) | set(self.mcp_tools_map.keys())

                        for tool_schema in mcp_tools:
                            original_tool_name = tool_schema.get("function", {}).get("name")
                            if not original_tool_name:
                                print(f"Warning: Skipping MCP tool from '{server_id}' with missing function name.")
                                continue
                            
                            # Create a prefixed name to avoid conflicts
                            prefixed_tool_name = f"mcp_{server_id}_{original_tool_name}"

                            if prefixed_tool_name in existing_tool_names:
                                print(f"Warning: Prefixed MCP tool name '{prefixed_tool_name}' from server '{server_id}' conflicts with an existing tool. Skipping.")
                                continue
                            
                            # Create a *copy* of the schema with the prefixed name
                            prefixed_schema = json.loads(json.dumps(tool_schema)) # Deep copy
                            prefixed_schema["function"]["name"] = prefixed_tool_name

                            # Add prefixed schema to the list LLM sees
                            self.tools.append(prefixed_schema)
                            
                            # Store mapping from prefixed name -> original schema + server_id
                            self.mcp_tools_map[prefixed_tool_name] = {
                                'original_schema': tool_schema, # Store original for execution
                                'server_id': server_id
                            }
                            existing_tool_names.add(prefixed_tool_name)
                            if self.verbose:
                                print(f"    -> Registered MCP tool: {prefixed_tool_name} (original: {original_tool_name})")

                    except Exception as tool_discovery_error:
                        print(f"Error discovering tools on MCP server '{server_id}': {tool_discovery_error}")
                        # Continue connecting to other servers even if one fails tool discovery

                else:
                    print(f"Warning: Unsupported MCP server type '{server_type}' for server '{server_id}'.")

            except Exception as connection_error:
                print(f"Error connecting to MCP server '{server_id}': {connection_error}")

    def close_mcp(self):
        """
        Synchronous version of close_mcp.
        Wraps the async version using asyncio.run() for backward compatibility.
        """
        try:
            return asyncio.run(self.close_mcp_async())
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.close_mcp_async())
                finally:
                    loop.close()
            raise

    async def close_mcp_async(self):
        """
        Asynchronous version of close_mcp.
        This is the main implementation that handles closing MCP connections.
        """
        # Check if exit_stack was initialized
        if not self.exit_stack:
            if self.verbose:
                 print("🔌 No MCP connections were established or exit_stack not initialized.")
            return

        if self.verbose:
            print("🔌 Closing MCP connections...")
        # The actual closing happens here
        await self.exit_stack.aclose()
        self.mcp_sessions = {}
        if self.verbose:
             print("  ✅ MCP connections closed.")

class ChainTechnique(str, Enum):
    NORMAL = "normal"
    HEAD_TAIL = "head-tail"

class ChainInstruction(BaseModel):
    instructions: Union[List[Union[str, Callable]], str]
    technique: ChainTechnique
    
    @validator('instructions')
    def validate_instructions(cls, v, values):
        if values.get('technique') == ChainTechnique.NORMAL:
            if not isinstance(v, list):
                raise ValueError("Normal technique requires a list of instructions")
            if len(v) == 0:
                raise ValueError("Instruction list cannot be empty")
            for instruction in v:
                if not isinstance(instruction, (str, Callable)):
                    raise ValueError("Instructions must be strings or callable functions")
        return v

class DynamicChainBuilder:
    def __init__(self, base_model: Union[str, dict], 
                 base_instruction: str,
                 technique: Literal["normal", "head-tail"] = "normal"):
        """
        Initialize a dynamic chain builder with a base model and instruction.
        
        Args:
            base_model: Base model to use for all dynamic chains
            base_instruction: Base instruction template to build upon
            technique: Chain building technique ("normal" or "head-tail")
        """
        self.base_model = base_model
        self.base_instruction = base_instruction
        self.technique = technique
        self.chain_outputs = {}
        self.chain_registry = {}
        self.execution_groups = {}
        self.memory_bank = {}
        
        # Validate base instruction template
        self._validate_template(base_instruction)
        
    def create_chain(self, chain_id: str, 
                    instructions: Union[List[Union[str, Callable]], str],
                    execution_mode: Literal["serial", "parallel", "independent"] = "serial",
                    group: str = "default",
                    dependencies: Optional[List[str]] = None) -> PromptChain:
        """
        Create a new chain with the given instructions.
        
        Args:
            chain_id: Unique identifier for this chain
            instructions: List of instructions for normal technique, or string prompt for head-tail
            execution_mode: How this chain should be executed
            group: Execution group for organizing related chains
            dependencies: List of chain_ids this chain depends on
        
        Returns:
            Configured PromptChain instance
        
        Raises:
            ValueError: If validation fails for instruction format
        """
        # Validate instructions using Pydantic model
        chain_instruction = ChainInstruction(
            instructions=instructions,
            technique=self.technique
        )
        
        # Process validated instructions
        if self.technique == "normal":
            validated_instructions = chain_instruction.instructions
        else:  # head-tail
            validated_instructions = [chain_instruction.instructions] if isinstance(chain_instruction.instructions, str) else chain_instruction.instructions
            
        # Create chain with validated instructions
        chain = PromptChain(
            models=[self.base_model] * len(validated_instructions),
            instructions=validated_instructions,
            store_steps=True
        )
        
        # Register chain with metadata
        self.chain_registry[chain_id] = {
            "chain": chain,
            "execution_mode": execution_mode,
            "group": group,
            "dependencies": dependencies or [],
            "status": "created"
        }
        
        # Add to execution group
        if group not in self.execution_groups:
            self.execution_groups[group] = []
        self.execution_groups[group].append(chain_id)
        
        return chain
    
    @staticmethod
    def _validate_template(template: str) -> None:
        """Validate that template has required placeholders."""
        required = ["{instruction}", "{input}"]
        for req in required:
            if req not in template:
                raise ValueError(f"Template missing required placeholder: {req}")

    def execute_chain(self, chain_id: str, input_data: str) -> str:
        """
        Execute a specific chain and store its output.
        
        :param chain_id: ID of chain to execute
        :param input_data: Input data for the chain
        :return: Chain output
        """
        if chain_id not in self.chain_registry:
            raise ValueError(f"Chain {chain_id} not found")
            
        chain_info = self.chain_registry[chain_id]
        
        # Check dependencies only for serial chains
        if chain_info["execution_mode"] == "serial":
            for dep_id in chain_info["dependencies"]:
                if dep_id not in self.chain_outputs:
                    raise ValueError(f"Dependency {dep_id} must be executed before {chain_id}")
        
        # Execute chain
        chain_info["status"] = "running"
        result = chain_info["chain"].process_prompt(input_data)
        chain_info["status"] = "completed"
        
        # Store output
        self.chain_outputs[chain_id] = result
        
        return result
        
    def execute_group(self, group: str, input_data: str, 
                     parallel_executor: Callable = None) -> Dict[str, str]:
        """
        Execute all chains in a group respecting their execution modes.
        
        :param group: Group identifier
        :param input_data: Input data for the chains
        :param parallel_executor: Optional function to handle parallel execution
                              Function should accept list of (chain_id, input_data) tuples
        :return: Dictionary of chain outputs
        """
        if group not in self.execution_groups:
            raise ValueError(f"Group {group} not found")
            
        # Organize chains by execution mode
        serial_chains = []
        parallel_chains = []
        independent_chains = []
        
        for chain_id in self.execution_groups[group]:
            mode = self.chain_registry[chain_id]["execution_mode"]
            if mode == "serial":
                serial_chains.append(chain_id)
            elif mode == "parallel":
                parallel_chains.append(chain_id)
            else:  # independent
                independent_chains.append(chain_id)
                
        # Execute independent chains (can be run anytime)
        for chain_id in independent_chains:
            self.execute_chain(chain_id, input_data)
            
        # Execute serial chains in dependency order
        executed = set()
        while serial_chains:
            for chain_id in serial_chains[:]:
                deps = self.chain_registry[chain_id]["dependencies"]
                if all(dep in executed for dep in deps):
                    self.execute_chain(chain_id, input_data)
                    executed.add(chain_id)
                    serial_chains.remove(chain_id)
                    
        # Execute parallel chains
        if parallel_chains:
            if parallel_executor:
                # Use provided parallel executor
                chain_inputs = [(cid, input_data) for cid in parallel_chains]
                results = parallel_executor(chain_inputs)
                for chain_id, result in results.items():
                    self.chain_outputs[chain_id] = result
                    self.chain_registry[chain_id]["status"] = "completed"
            else:
                # Sequential fallback for parallel chains
                for chain_id in parallel_chains:
                    self.execute_chain(chain_id, input_data)
                    
        return {chain_id: self.chain_outputs[chain_id] 
                for chain_id in self.execution_groups[group]
                if chain_id in self.chain_outputs}
                
    def get_chain_output(self, chain_id: str) -> Union[str, None]:
        """Get the output of a previously executed chain."""
        return self.chain_outputs.get(chain_id)
        
    def get_chain_status(self, chain_id: str) -> Union[str, None]:
        """Get the status of a chain (created, running, completed)."""
        return self.chain_registry.get(chain_id, {}).get("status")
        
    def get_group_status(self, group: str) -> Dict[str, str]:
        """Get the status of all chains in a group."""
        if group not in self.execution_groups:
            raise ValueError(f"Group {group} not found")
            
        return {chain_id: self.get_chain_status(chain_id)
                for chain_id in self.execution_groups[group]}
        
    def insert_chain(self, target_chain_id: str, new_instructions: List[str], 
                    position: int = -1) -> None:
        """
        Insert new instructions into an existing chain.
        
        :param target_chain_id: ID of chain to modify
        :param new_instructions: New instructions to insert
        :param position: Position to insert at (-1 for end)
        """
        if target_chain_id not in self.chain_registry:
            raise ValueError(f"Chain {target_chain_id} not found")
            
        chain_info = self.chain_registry[target_chain_id]
        chain = chain_info["chain"]
        
        # Calculate insert position
        if position < 0:
            position = len(chain.instructions)
        
        # Insert new instructions
        chain.instructions[position:position] = new_instructions
        
        # Add corresponding models
        new_models = [self.base_model] * len(new_instructions)
        chain.models[position:position] = new_models
        chain.model_params[position:position] = [{} for _ in range(len(new_instructions))]
        
    def merge_chains(self, chain_ids: List[str], new_chain_id: str,
                    execution_mode: Literal["serial", "parallel", "independent"] = "serial",
                    group: str = None) -> PromptChain:
        """
        Merge multiple chains into a new chain.
        
        :param chain_ids: List of chain IDs to merge
        :param new_chain_id: ID for the merged chain
        :param execution_mode: Execution mode for the new chain
        :param group: Optional group for the new chain
        :return: New merged PromptChain
        """
        if new_chain_id in self.chain_registry:
            raise ValueError(f"Chain {new_chain_id} already exists")
            
        # Collect all instructions and models
        all_instructions = []
        all_models = []
        
        for chain_id in chain_ids:
            if chain_id not in self.chain_registry:
                raise ValueError(f"Chain {chain_id} not found")
                
            chain = self.chain_registry[chain_id]["chain"]
            all_instructions.extend(chain.instructions)
            all_models.extend(chain.models)
        
        # Create new merged chain
        merged_chain = PromptChain(
            models=all_models,
            instructions=all_instructions,
            store_steps=True
        )
        
        # Register merged chain
        self.chain_registry[new_chain_id] = {
            "chain": merged_chain,
            "execution_mode": execution_mode,
            "group": group or "merged",
            "dependencies": chain_ids if execution_mode == "serial" else [],
            "status": "created"
        }
        
        # Add to group if specified
        if group:
            if group not in self.execution_groups:
                self.execution_groups[group] = []
            self.execution_groups[group].append(new_chain_id)
        
        return merged_chain

    def inject_chain(self, target_chain_id: str, source_chain_id: str, 
                    position: int = -1, adjust_dependencies: bool = True) -> None:
        """
        Inject an entire chain into another chain, adjusting steps and dependencies.
        
        :param target_chain_id: ID of chain to inject into
        :param source_chain_id: ID of chain to inject
        :param position: Position to inject at (-1 for end)
        :param adjust_dependencies: Whether to adjust dependencies of subsequent steps
        """
        if target_chain_id not in self.chain_registry:
            raise ValueError(f"Target chain {target_chain_id} not found")
        if source_chain_id not in self.chain_registry:
            raise ValueError(f"Source chain {source_chain_id} not found")
            
        target_chain = self.chain_registry[target_chain_id]["chain"]
        source_chain = self.chain_registry[source_chain_id]["chain"]
        
        # Calculate injection position
        if position < 0:
            position = len(target_chain.instructions)
        
        # Get number of steps being injected
        injection_size = len(source_chain.instructions)
        
        # Store original steps for dependency adjustment
        original_steps = {
            i: step for i, step in enumerate(target_chain.instructions)
        }
        
        # Insert instructions from source chain
        target_chain.instructions[position:position] = source_chain.instructions
        
        # Insert corresponding models
        target_chain.models[position:position] = source_chain.models
        target_chain.model_params[position:position] = source_chain.model_params
        
        # Adjust step storage if enabled
        if target_chain.store_steps:
            # Shift existing step outputs
            new_outputs = {}
            for step_num in sorted(target_chain.step_outputs.keys(), reverse=True):
                if step_num.startswith("step_"):
                    step_idx = int(step_num.split("_")[1])
                    if step_idx >= position:
                        new_step = f"step_{step_idx + injection_size}"
                        new_outputs[new_step] = target_chain.step_outputs[step_num]
                    else:
                        new_outputs[step_num] = target_chain.step_outputs[step_num]
            target_chain.step_outputs = new_outputs
        
        # Update chain registry metadata
        if adjust_dependencies:
            # Adjust dependencies for all chains that depend on steps in the target chain
            for chain_id, chain_info in self.chain_registry.items():
                if chain_id == target_chain_id:
                    continue
                    
                new_deps = []
                for dep in chain_info["dependencies"]:
                    if dep == target_chain_id:
                        # If depending on the whole chain, no adjustment needed
                        new_deps.append(dep)
                    else:
                        # If depending on specific steps, adjust step numbers
                        try:
                            step_num = int(dep.split("_")[1])
                            if step_num >= position:
                                new_step = f"step_{step_num + injection_size}"
                                new_deps.append(new_step)
                            else:
                                new_deps.append(dep)
                        except (IndexError, ValueError):
                            new_deps.append(dep)
                            
                chain_info["dependencies"] = new_deps
    
    def get_step_dependencies(self, chain_id: str) -> Dict[int, List[str]]:
        """
        Get dependencies for each step in a chain.
        
        :param chain_id: Chain ID to analyze
        :return: Dictionary mapping step numbers to their dependencies
        """
        if chain_id not in self.chain_registry:
            raise ValueError(f"Chain {chain_id} not found")
            
        chain_info = self.chain_registry[chain_id]
        chain = chain_info["chain"]
        
        step_deps = {}
        for i in range(len(chain.instructions)):
            deps = []
            # Check explicit dependencies
            for dep_chain_id, dep_info in self.chain_registry.items():
                if dep_chain_id == chain_id:
                    continue
                if f"step_{i}" in dep_info["dependencies"]:
                    deps.append(dep_chain_id)
            step_deps[i] = deps
            
        return step_deps
    
    def validate_injection(self, target_chain_id: str, source_chain_id: str, 
                         position: int = -1) -> bool:
        """
        Validate if a chain injection would create circular dependencies.
        
        :param target_chain_id: ID of chain to inject into
        :param source_chain_id: ID of chain to inject
        :param position: Position to inject at
        :return: True if injection is valid, False otherwise
        """
        if target_chain_id not in self.chain_registry:
            raise ValueError(f"Target chain {target_chain_id} not found")
        if source_chain_id not in self.chain_registry:
            raise ValueError(f"Source chain {source_chain_id} not found")
            
        # Check if source chain depends on target chain
        def has_dependency(chain_id, target_id, visited=None):
            if visited is None:
                visited = set()
            if chain_id in visited:
                return False
            visited.add(chain_id)
            
            chain_info = self.chain_registry[chain_id]
            if target_id in chain_info["dependencies"]:
                return True
            
            for dep in chain_info["dependencies"]:
                if dep in self.chain_registry and has_dependency(dep, target_id, visited):
                    return True
            return False
            
        return not has_dependency(source_chain_id, target_chain_id)

    def reorder_steps(self, chain_id: str) -> None:
        """
        Reorder steps in a chain based on dependencies.
        
        :param chain_id: Chain ID to reorder
        """
        if chain_id not in self.chain_registry:
            raise ValueError(f"Chain {chain_id} not found")
            
        chain_info = self.chain_registry[chain_id]
        chain = chain_info["chain"]
        
        # Get step dependencies
        step_deps = self.get_step_dependencies(chain_id)
        
        # Create dependency graph
        from collections import defaultdict
        graph = defaultdict(list)
        for step, deps in step_deps.items():
            for dep in deps:
                graph[dep].append(step)
                
        # Topologically sort steps
        visited = set()
        temp = set()
        order = []
        
        def visit(step):
            if step in temp:
                raise ValueError("Circular dependency detected")
            if step in visited:
                return
            temp.add(step)
            for neighbor in graph[step]:
                visit(neighbor)
            temp.remove(step)
            visited.add(step)
            order.append(step)
            
        for step in range(len(chain.instructions)):
            if step not in visited:
                visit(step)
                
        # Reorder steps based on topological sort
        chain.instructions = [chain.instructions[i] for i in order]
        chain.models = [chain.models[i] for i in order]
        chain.model_params = [chain.model_params[i] for i in order]
        
        # Update step storage if enabled
        if chain.store_steps:
            new_outputs = {}
            for i, step in enumerate(order):
                if f"step_{step}" in chain.step_outputs:
                    new_outputs[f"step_{i}"] = chain.step_outputs[f"step_{step}"]
            chain.step_outputs = new_outputs

    # Add memory bank methods
    def store_memory(self, key: str, value: any, namespace: str = "default") -> None:
        """
        Store a value in memory bank for later retrieval.
        
        Args:
            key: Unique identifier for this memory item
            value: Any value to store
            namespace: Optional grouping namespace (default: "default")
        """
        if namespace not in self.memory_bank:
            self.memory_bank[namespace] = {}
        self.memory_bank[namespace][key] = value
    
    def retrieve_memory(self, key: str, namespace: str = "default", default: any = None) -> any:
        """
        Retrieve a value from memory bank.
        
        Args:
            key: Identifier for the memory item
            namespace: Namespace to look in (default: "default")
            default: Value to return if key not found
            
        Returns:
            Stored value or default if not found
        """
        if namespace not in self.memory_bank:
            return default
        return self.memory_bank[namespace].get(key, default)
    
    def memory_exists(self, key: str, namespace: str = "default") -> bool:
        """
        Check if a memory item exists.
        
        Args:
            key: Memory item identifier
            namespace: Namespace to check in
            
        Returns:
            True if memory exists, False otherwise
        """
        return namespace in self.memory_bank and key in self.memory_bank[namespace]
    
    def list_memories(self, namespace: str = "default") -> List[str]:
        """
        List all memory keys in a namespace.
        
        Args:
            namespace: Namespace to list keys from
            
        Returns:
            List of memory keys
        """
        if namespace not in self.memory_bank:
            return []
        return list(self.memory_bank[namespace].keys())
    
    def clear_memories(self, namespace: str = None) -> None:
        """
        Clear memories in specified namespace or all if none specified.
        
        Args:
            namespace: Namespace to clear or None for all
        """
        if namespace is None:
            self.memory_bank = {}
        elif namespace in self.memory_bank:
            self.memory_bank[namespace] = {}
    
    def create_memory_function(self, namespace: str = "default") -> callable:
        """
        Creates a specialized memory access function for use in chain steps.
        
        Args:
            namespace: Namespace for this memory function
            
        Returns:
            Function that can be used in chain steps to access memory
        """
        def memory_function(input_text: str) -> str:
            """Parse input to store or retrieve from memory bank"""
            parts = input_text.strip().split("\n")
            
            # Command format: MEMORY [STORE|GET] key=value
            results = []
            
            for part in parts:
                if part.upper().startswith("MEMORY"):
                    try:
                        command_parts = part.split()
                        if len(command_parts) >= 3:
                            action = command_parts[1].upper()
                            key_value = " ".join(command_parts[2:])
                            
                            if action == "STORE" and "=" in key_value:
                                key, value = key_value.split("=", 1)
                                self.store_memory(key.strip(), value.strip(), namespace)
                                results.append(f"Stored '{key.strip()}' in memory")
                            elif action == "GET":
                                key = key_value.strip()
                                value = self.retrieve_memory(key, namespace, "Not found")
                                results.append(f"{key} = {value}")
                            elif action == "LIST":
                                keys = self.list_memories(namespace)
                                results.append(f"Memory keys: {', '.join(keys) if keys else 'none'}")
                    except Exception as e:
                        results.append(f"Memory error: {str(e)}")
                else:
                    results.append(part)
                    
            return "\n".join(results)
        
        return memory_function
    
    def create_memory_chain(self, chain_id: str, namespace: str = "default", 
                           instructions: List[str] = None) -> PromptChain:
        """
        Creates a specialized chain with memory access capabilities.
        
        Args:
            chain_id: Unique identifier for this chain
            namespace: Memory namespace for this chain
            instructions: Optional list of instructions (defaults to memory processing)
            
        Returns:
            Configured PromptChain with memory capabilities
        """
        memory_function = self.create_memory_function(namespace)
        
        default_instructions = [
            "Process the following input and update or retrieve from memory as needed: {input}",
            memory_function
        ]
        
        return self.create_chain(
            chain_id=chain_id,
            instructions=instructions or default_instructions,
            execution_mode="independent",
            group="memory_chains"
        )

# Example usage
if __name__ == "__main__":
    # Create builder with base configuration
    builder = DynamicChainBuilder(
        base_model={
            "name": "openai/gpt-4",
            "params": {"temperature": 0.7}
        },
        base_instruction="Base analysis: {input}"
    )
    
    # Create chains with different execution modes
    builder.create_chain(
        "initial",
        ["Extract key points: {input}"],
        execution_mode="serial",
        group="analysis"
    )
    
    builder.create_chain(
        "sentiment",
        ["Analyze sentiment: {input}"],
        execution_mode="parallel",
        group="analysis"
    )
    
    builder.create_chain(
        "keywords",
        ["Extract keywords: {input}"],
        execution_mode="parallel",
        group="analysis"
    )
    
    builder.create_chain(
        "final",
        ["Synthesize findings: {input}"],
        execution_mode="serial",
        dependencies=["initial"],
        group="analysis"
    )
    
    # Execute all chains in the group
    results = builder.execute_group(
        "analysis",
        "This is a test input",
        parallel_executor=None  # Add your parallel execution function here
    )
    
    print("Results:", results)