from typing import List, Dict, Any, Callable, Awaitable, Optional
import logging
import json

logger = logging.getLogger(__name__)

def get_function_name_from_tool_call(tool_call) -> Optional[str]:
    """
    Safely extract function name from different tool_call object formats.
    
    Args:
        tool_call: A tool call object from various LLM providers
        
    Returns:
        The function name or None if not found
    """
    try:
        # If it's a dictionary (like from OpenAI direct API)
        if isinstance(tool_call, dict):
            function_obj = tool_call.get('function', {})
            if isinstance(function_obj, dict):
                return function_obj.get('name')
        # If it's an object with attributes (like from LiteLLM)
        else:
            function_obj = getattr(tool_call, 'function', None)
            if function_obj:
                # Direct attribute access
                if hasattr(function_obj, 'name'):
                    return function_obj.name
                # Dictionary-like access for nested objects
                elif isinstance(function_obj, dict):
                    return function_obj.get('name')
                # Get function name through __dict__
                elif hasattr(function_obj, '__dict__'):
                    return function_obj.__dict__.get('name')
        # Last resort: try to convert to dictionary if possible
        if hasattr(tool_call, 'model_dump'):
            model_dump = tool_call.model_dump()
            if isinstance(model_dump, dict) and 'function' in model_dump:
                function_dict = model_dump['function']
                if isinstance(function_dict, dict):
                    return function_dict.get('name')
    except Exception as e:
        logger.warning(f"Error extracting function name: {e}")
    
    return None  # Return None if we couldn't extract the name

class AgenticStepProcessor:
    """
    Represents a single step within a PromptChain that executes an internal
    agentic loop to achieve its objective, potentially involving multiple
    LLM calls and tool executions within this single step.
    Optionally, a specific model can be set for this step; otherwise, the chain's default model is used.
    """
    def __init__(self, objective: str, max_internal_steps: int = 5, model_name: str = None, model_params: Dict[str, Any] = None):
        """
        Initializes the agentic step.

        Args:
            objective: The specific goal or instruction for this agentic step.
                       This will be used to guide the internal loop.
            max_internal_steps: Maximum number of internal iterations (LLM calls)
                                to prevent infinite loops.
            model_name: (Optional) The model to use for this agentic step. If None, defaults to the chain's first model.
            model_params: (Optional) Dictionary of parameters to pass to the model, such as tool_choice settings.
        """
        if not objective or not isinstance(objective, str):
            raise ValueError("Objective must be a non-empty string.")
        self.objective = objective
        self.max_internal_steps = max_internal_steps
        self.model_name = model_name
        self.model_params = model_params or {}
        logger.debug(f"AgenticStepProcessor initialized with objective: {self.objective[:100]}... Model: {self.model_name}, Params: {self.model_params}")

    async def run_async(
        self,
        initial_input: str,
        available_tools: List[Dict[str, Any]],
        llm_runner: Callable[..., Awaitable[Any]], # Should return LiteLLM-like response message
        tool_executor: Callable[[Any], Awaitable[str]] # Takes tool_call, returns result string
    ) -> str:
        """
        Executes the internal agentic loop for this step.

        Args:
            initial_input: The input string received from the previous PromptChain step.
            available_tools: List of tool schemas available for this step.
            llm_runner: An async function to call the LLM.
                        Expected signature: llm_runner(messages, tools, tool_choice) -> response_message
            tool_executor: An async function to execute a tool call.
                           Expected signature: tool_executor(tool_call) -> result_content_string

        Returns:
            The final string output of the agentic loop for this step.
        """
        # Store available tools as an attribute for inspection/debugging
        self.available_tools = available_tools

        # Log the tool names for proof
        tool_names = [t['function']['name'] for t in available_tools if 'function' in t]
        logger.info(f"[AgenticStepProcessor] Available tools at start: {tool_names}")

        logger.info(f"Starting agentic step. Objective: {self.objective[:100]}...")
        logger.debug(f"Initial input: {initial_input[:100]}...")

        # Full internal history for debugging/tracking (not sent to LLM)
        internal_history = []
        # Minimal valid LLM message history (sent to LLM)
        llm_history = []

        # Start with the objective and initial input
        system_prompt = f"""Your goal is to achieve the following objective: {self.objective}
You will receive input and can use tools to achieve this goal.
Reason step-by-step. When you believe the objective is fully met based on the history, respond with the final answer directly without calling any more tools. Otherwise, decide the next best tool call to make progress."""
        system_message = {"role": "system", "content": system_prompt}
        user_message = {"role": "user", "content": f"The initial input for this step is: {initial_input}"} if initial_input else None
        internal_history.append(system_message)
        if user_message:
            internal_history.append(user_message)
        # Initialize llm_history for first call
        llm_history = [system_message]
        if user_message:
            llm_history.append(user_message)

        final_answer = None
        last_tool_call_assistant_msg = None
        last_tool_msgs = []
        clarification_attempts = 0
        max_clarification_attempts = 3

        for step_num in range(self.max_internal_steps):
            logger.info(f"Agentic step internal iteration {step_num + 1}/{self.max_internal_steps}")
            while True:
                # Prepare minimal valid messages for LLM
                messages_for_llm = llm_history[:]
                # Log the full structure of messages and tools for debugging
                print("Messages sent to LLM:", json.dumps(messages_for_llm, indent=2))
                print("Tools sent to LLM:", json.dumps(available_tools, indent=2))
                try:
                    # Call the LLM to decide next action or give final answer
                    logger.debug("Calling LLM for next action/final answer...")
                    
                    # Get tool_choice from model_params if available, otherwise default to "auto"
                    tool_choice = self.model_params.get("tool_choice", "auto")
                    
                    response_message = await llm_runner(
                        messages=messages_for_llm,
                        tools=available_tools,
                        tool_choice=tool_choice  # Use model_params or default
                    )
                    
                    # Add more detailed debugging
                    if isinstance(response_message, dict):
                        logger.debug(f"Response message (dict): {json.dumps(response_message)}")
                    else:
                        logger.debug(f"Response message (object): type={type(response_message)}, dir={dir(response_message)}")
                        if hasattr(response_message, 'model_dump'):
                            logger.debug(f"Response message dump: {json.dumps(response_message.model_dump())}")

                    # Append assistant's response (thought process + potential tool call) to internal history
                    internal_history.append(response_message if isinstance(response_message, dict) else {"role": "assistant", "content": str(response_message)})

                    # Enhanced tool call detection - try multiple approaches
                    tool_calls = None
                    
                    # Method 1: Try standard attribute/key access
                    if isinstance(response_message, dict):
                        tool_calls = response_message.get('tool_calls')
                    else:
                        tool_calls = getattr(response_message, 'tool_calls', None)
                    
                    # Method 2: For LiteLLM objects, try nested access via model_dump
                    if tool_calls is None and hasattr(response_message, 'model_dump'):
                        try:
                            response_dict = response_message.model_dump()
                            if isinstance(response_dict, dict):
                                tool_calls = response_dict.get('tool_calls')
                        except Exception as dump_err:
                            logger.warning(f"Error accessing model_dump: {dump_err}")
                    
                    # Log the detected tool calls for debugging
                    if tool_calls:
                        logger.debug(f"Detected tool calls: {tool_calls}")

                    if tool_calls:
                        clarification_attempts = 0  # Reset on tool call
                        logger.info(f"LLM requested {len(tool_calls)} tool(s).")
                        # Prepare new minimal LLM history: system, user, assistant (with tool_calls), tool(s)
                        last_tool_call_assistant_msg = response_message if isinstance(response_message, dict) else {"role": "assistant", "content": str(response_message)}
                        last_tool_msgs = []
                        # Execute tools sequentially for simplicity
                        for tool_call in tool_calls:
                            # Extract tool call ID and function name using the helper function
                            if isinstance(tool_call, dict):
                                tool_call_id = tool_call.get('id')
                            else:
                                tool_call_id = getattr(tool_call, 'id', None)
                            
                            function_name = get_function_name_from_tool_call(tool_call)
                            
                            if not function_name:
                                logger.error(f"Could not extract function name from tool call: {tool_call}")
                                tool_result_content = json.dumps({"error": "Could not determine function name for tool execution"})
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": "unknown_function",
                                    "content": tool_result_content,
                                }
                                internal_history.append(tool_msg)
                                last_tool_msgs.append(tool_msg)
                                continue
                                
                            logger.info(f"Executing tool: {function_name} (ID: {tool_call_id})")
                            try:
                                # Use the provided executor callback
                                tool_result_content = await tool_executor(tool_call)
                                logger.info(f"Tool {function_name} executed successfully.")
                                logger.debug(f"Tool result content: {tool_result_content[:150]}...")
                                # Append tool result to internal history
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": function_name,
                                    "content": tool_result_content,
                                }
                                internal_history.append(tool_msg)
                                last_tool_msgs.append(tool_msg)
                            except Exception as tool_exec_error:
                                logger.error(f"Error executing tool {function_name}: {tool_exec_error}", exc_info=True)
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": function_name,
                                    "content": f"Error executing tool: {str(tool_exec_error)}",
                                }
                                internal_history.append(tool_msg)
                                last_tool_msgs.append(tool_msg)
                        # For next LLM call, only send: system, user, last assistant (with tool_calls), tool(s)
                        llm_history = [system_message]
                        if user_message:
                            llm_history.append(user_message)
                        llm_history.append(last_tool_call_assistant_msg)
                        llm_history.extend(last_tool_msgs)
                        
                        # After tool(s) executed, continue inner while loop (do not increment step_num)
                        continue
                    else:
                        # No tool calls, check for content or clarify
                        final_answer_content = None
                        
                        # Try multiple approaches to get content
                        if isinstance(response_message, dict):
                            final_answer_content = response_message.get('content')
                        else:
                            final_answer_content = getattr(response_message, 'content', None)
                        
                        if final_answer_content is not None:
                            clarification_attempts = 0  # Reset on final answer
                            logger.info("LLM provided final answer for the step.")
                            final_answer = str(final_answer_content)
                            break # Break inner while loop, increment step_num
                        else:
                            clarification_attempts += 1
                            logger.warning(f"LLM did not request tools and did not provide content. Clarification attempt {clarification_attempts}/{max_clarification_attempts}.")
                            # Add a message indicating confusion or request for clarification
                            clarification_msg = {"role": "user", "content": "Please either call a tool or provide the final answer."}
                            internal_history.append(clarification_msg)
                            # For next LLM call, send: system, user, clarification
                            llm_history = [system_message]
                            if user_message:
                                llm_history.append(user_message)
                            llm_history.append(clarification_msg)
                            if clarification_attempts >= max_clarification_attempts:
                                logger.error(f"Agentic step exceeded {max_clarification_attempts} clarification attempts without tool call or answer. Breaking loop.")
                                final_answer = "Error: LLM did not call a tool or provide an answer after multiple attempts."
                                break
                            continue
                except Exception as llm_error:
                    logger.error(f"Error during LLM call in agentic step: {llm_error}", exc_info=True)
                    final_answer = f"Error during agentic step processing: {llm_error}"
                    break # Break inner while loop, increment step_num
            # If we have a final answer or error, break outer for loop
            if final_answer is not None:
                break

        # After loop finishes (or breaks)
        if final_answer is None:
            logger.warning(f"Agentic step reached max iterations ({self.max_internal_steps}) without a final answer.")
            # Try to return the last assistant message with content, if any
            last_content = None
            for msg in reversed(internal_history):
                if msg.get("role") == "assistant" and msg.get("content"):
                    last_content = msg["content"]
                    break
            if last_content:
                final_answer = last_content
            else:
                final_answer = "No tool was called, and the LLM did not provide an answer. Please try rephrasing your request or check tool availability."

        logger.info(f"Agentic step finished. Final output: {final_answer[:150]}...")
        return final_answer 