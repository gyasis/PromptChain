from typing import List, Dict, Any, Callable, Awaitable, Optional, Union
import logging
import json
from enum import Enum
from datetime import datetime
from .agentic_step_result import StepExecutionMetadata, AgenticStepResult

logger = logging.getLogger(__name__)


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """
    Estimate token count for a list of messages.
    Uses simple approximation: ~4 characters per token.
    For more accuracy, integrate tiktoken library.

    Args:
        messages: List of message dictionaries

    Returns:
        Estimated token count
    """
    total_chars = 0
    for msg in messages:
        if isinstance(msg, dict):
            # Count content
            content = msg.get('content', '')
            if content:
                total_chars += len(str(content))
            # Count tool calls if present
            tool_calls = msg.get('tool_calls', [])
            if tool_calls:
                total_chars += len(str(tool_calls))
    return total_chars // 4  # Approximate: 4 chars per token


class HistoryMode(str, Enum):
    """History accumulation modes for AgenticStepProcessor."""
    MINIMAL = "minimal"         # Only keep last assistant + tool results (default, backward compatible)
    PROGRESSIVE = "progressive" # Accumulate assistant messages + tool results progressively
    KITCHEN_SINK = "kitchen_sink"  # Keep everything - all reasoning, tool calls, results

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
    def __init__(
        self,
        objective: str,
        max_internal_steps: int = 5,
        model_name: str = None,
        model_params: Dict[str, Any] = None,
        history_mode: str = "minimal",
        max_context_tokens: Optional[int] = None
    ):
        """
        Initializes the agentic step.

        Args:
            objective: The specific goal or instruction for this agentic step.
                       This will be used to guide the internal loop.
            max_internal_steps: Maximum number of internal iterations (LLM calls)
                                to prevent infinite loops.
            model_name: (Optional) The model to use for this agentic step. If None, defaults to the chain's first model.
            model_params: (Optional) Dictionary of parameters to pass to the model, such as tool_choice settings.
            history_mode: (Optional) History accumulation mode. Options:
                         - "minimal" (default): Only keep last assistant + tool results (backward compatible)
                           ⚠️  DEPRECATION NOTICE: This mode may be deprecated in future versions.
                           Consider using "progressive" for better multi-hop reasoning.
                         - "progressive": Accumulate assistant messages + tool results progressively (RECOMMENDED)
                         - "kitchen_sink": Keep everything - all reasoning, tool calls, results
            max_context_tokens: (Optional) Maximum context tokens before warning. Default is None (no limit).
        """
        if not objective or not isinstance(objective, str):
            raise ValueError("Objective must be a non-empty string.")

        # Validate history_mode
        if history_mode not in [mode.value for mode in HistoryMode]:
            raise ValueError(f"Invalid history_mode: {history_mode}. Must be one of {[mode.value for mode in HistoryMode]}")

        self.objective = objective
        self.max_internal_steps = max_internal_steps
        self.model_name = model_name
        self.model_params = model_params or {}
        self.history_mode = history_mode
        self.max_context_tokens = max_context_tokens
        self.conversation_history: List[Dict[str, Any]] = []  # Accumulated history for progressive/kitchen_sink modes

        # Deprecation warning for minimal mode
        if self.history_mode == HistoryMode.MINIMAL.value:
            logger.info(
                "⚠️  Using 'minimal' history mode (default). This mode may be deprecated in future versions. "
                "Consider using 'progressive' mode for better multi-hop reasoning capabilities."
            )

        logger.debug(
            f"AgenticStepProcessor initialized with objective: {self.objective[:100]}... "
            f"Model: {self.model_name}, Params: {self.model_params}, "
            f"History Mode: {self.history_mode}, Max Tokens: {self.max_context_tokens}"
        )

    async def run_async(
        self,
        initial_input: str,
        available_tools: List[Dict[str, Any]],
        llm_runner: Callable[..., Awaitable[Any]], # Should return LiteLLM-like response message
        tool_executor: Callable[[Any], Awaitable[str]], # Takes tool_call, returns result string
        return_metadata: bool = False  # NEW: Return metadata instead of just string
    ) -> Union[str, AgenticStepResult]:
        """
        Executes the internal agentic loop for this step with optional metadata return.

        Args:
            initial_input: The input string received from the previous PromptChain step.
            available_tools: List of tool schemas available for this step.
            llm_runner: An async function to call the LLM.
                        Expected signature: llm_runner(messages, tools, tool_choice) -> response_message
            tool_executor: An async function to execute a tool call.
                           Expected signature: tool_executor(tool_call) -> result_content_string
            return_metadata: If True, return AgenticStepResult with full execution metadata.
                           If False, return just the final answer string (backward compatible).

        Returns:
            - str: Just final answer (default, backward compatible)
            - AgenticStepResult: Full execution metadata when return_metadata=True
        """
        # Store available tools as an attribute for inspection/debugging
        self.available_tools = available_tools

        # Log the tool names for proof
        tool_names = [t['function']['name'] for t in available_tools if 'function' in t]
        logger.info(f"[AgenticStepProcessor] Available tools at start: {tool_names}")

        logger.info(f"Starting agentic step. Objective: {self.objective[:100]}...")
        logger.debug(f"Initial input: {initial_input[:100]}...")

        # Metadata tracking variables
        execution_start_time = datetime.now()
        steps_metadata: List[StepExecutionMetadata] = []
        total_tools_called = 0
        total_tokens_used = 0
        execution_errors: List[str] = []
        execution_warnings: List[str] = []

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

            # Track metadata for this step
            step_start_time = datetime.now()
            step_tool_calls: List[Dict[str, Any]] = []
            step_clarification_attempts = 0
            step_error: Optional[str] = None

            while True:
                # Prepare minimal valid messages for LLM
                messages_for_llm = llm_history[:]
                # Log the full structure of messages and tools for debugging
                logger.debug(f"Messages sent to LLM: {json.dumps(messages_for_llm, indent=2)}")
                logger.debug(f"Tools sent to LLM: {json.dumps(available_tools, indent=2)}")
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
                                tool_call_args = tool_call.get('function', {}).get('arguments', {})
                            else:
                                tool_call_id = getattr(tool_call, 'id', None)
                                func_obj = getattr(tool_call, 'function', None)
                                tool_call_args = getattr(func_obj, 'arguments', {}) if func_obj else {}

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
                                # Track failed tool call
                                step_tool_calls.append({
                                    "name": "unknown_function",
                                    "args": {},
                                    "result": tool_result_content,
                                    "time_ms": 0,
                                    "error": "Could not determine function name"
                                })
                                continue

                            logger.info(f"Executing tool: {function_name} (ID: {tool_call_id})")
                            tool_exec_start = datetime.now()
                            try:
                                # Use the provided executor callback
                                tool_result_content = await tool_executor(tool_call)
                                tool_exec_time = (datetime.now() - tool_exec_start).total_seconds() * 1000
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
                                # Track successful tool call
                                step_tool_calls.append({
                                    "name": function_name,
                                    "args": tool_call_args,
                                    "result": tool_result_content,
                                    "time_ms": tool_exec_time
                                })
                                total_tools_called += 1
                            except Exception as tool_exec_error:
                                tool_exec_time = (datetime.now() - tool_exec_start).total_seconds() * 1000
                                logger.error(f"Error executing tool {function_name}: {tool_exec_error}", exc_info=True)
                                error_msg = str(tool_exec_error)
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": function_name,
                                    "content": f"Error executing tool: {error_msg}",
                                }
                                internal_history.append(tool_msg)
                                last_tool_msgs.append(tool_msg)
                                # Track failed tool call
                                step_tool_calls.append({
                                    "name": function_name,
                                    "args": tool_call_args,
                                    "result": f"Error: {error_msg}",
                                    "time_ms": tool_exec_time,
                                    "error": error_msg
                                })
                                execution_errors.append(f"Tool {function_name}: {error_msg}")

                        # Build llm_history based on history_mode
                        if self.history_mode == HistoryMode.MINIMAL.value:
                            # MINIMAL: Only keep last assistant + tool results (original behavior)
                            llm_history = [system_message]
                            if user_message:
                                llm_history.append(user_message)
                            llm_history.append(last_tool_call_assistant_msg)
                            llm_history.extend(last_tool_msgs)

                        elif self.history_mode == HistoryMode.PROGRESSIVE.value:
                            # PROGRESSIVE: Accumulate assistant messages + tool results
                            # Add to conversation history
                            self.conversation_history.append(last_tool_call_assistant_msg)
                            self.conversation_history.extend(last_tool_msgs)

                            # Build history: system + user + accumulated conversation
                            llm_history = [system_message]
                            if user_message:
                                llm_history.append(user_message)
                            llm_history.extend(self.conversation_history)

                        elif self.history_mode == HistoryMode.KITCHEN_SINK.value:
                            # KITCHEN_SINK: Keep everything including user messages between iterations
                            self.conversation_history.append(last_tool_call_assistant_msg)
                            self.conversation_history.extend(last_tool_msgs)

                            # Build history: system + user + accumulated everything
                            llm_history = [system_message]
                            if user_message:
                                llm_history.append(user_message)
                            llm_history.extend(self.conversation_history)

                        # Token limit warning
                        if self.max_context_tokens:
                            estimated_tokens = estimate_tokens(llm_history)
                            if estimated_tokens > self.max_context_tokens:
                                logger.warning(
                                    f"Context size ({estimated_tokens} tokens) exceeds max_context_tokens "
                                    f"({self.max_context_tokens}). Consider using 'minimal' history_mode or "
                                    f"increasing max_context_tokens. History mode: {self.history_mode}"
                                )

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
                            step_clarification_attempts += 1
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
                                step_error = final_answer
                                execution_errors.append(final_answer)
                                break
                            continue
                except Exception as llm_error:
                    logger.error(f"Error during LLM call in agentic step: {llm_error}", exc_info=True)
                    error_msg = f"Error during agentic step processing: {llm_error}"
                    final_answer = error_msg
                    step_error = error_msg
                    execution_errors.append(error_msg)
                    break # Break inner while loop, increment step_num

            # Create step metadata after inner while loop completes
            step_execution_time = (datetime.now() - step_start_time).total_seconds() * 1000
            step_tokens = estimate_tokens(llm_history)
            total_tokens_used += step_tokens

            step_metadata = StepExecutionMetadata(
                step_number=step_num + 1,
                tool_calls=step_tool_calls,
                tokens_used=step_tokens,
                execution_time_ms=step_execution_time,
                clarification_attempts=step_clarification_attempts,
                error=step_error
            )
            steps_metadata.append(step_metadata)

            # If we have a final answer or error, break outer for loop
            if final_answer is not None:
                break

        # After loop finishes (or breaks)
        if final_answer is None:
            logger.warning(f"Agentic step reached max iterations ({self.max_internal_steps}) without a final answer.")
            execution_warnings.append(f"Reached max iterations ({self.max_internal_steps}) without final answer")
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

        # Calculate total execution time
        total_execution_time = (datetime.now() - execution_start_time).total_seconds() * 1000

        # Return metadata if requested, otherwise just the answer
        if return_metadata:
            return AgenticStepResult(
                final_answer=final_answer,
                total_steps=len(steps_metadata),
                max_steps_reached=(len(steps_metadata) >= self.max_internal_steps),
                objective_achieved=(final_answer is not None and not any(steps_metadata[-1].error for _ in [1] if steps_metadata)),
                steps=steps_metadata,
                total_tools_called=total_tools_called,
                total_tokens_used=total_tokens_used,
                total_execution_time_ms=total_execution_time,
                history_mode=self.history_mode,
                max_internal_steps=self.max_internal_steps,
                model_name=self.model_name,
                errors=execution_errors,
                warnings=execution_warnings
            )
        else:
            # Backward compatible: just return the string
            return final_answer 