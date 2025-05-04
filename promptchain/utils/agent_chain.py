# agent_chain.py
import asyncio
import json # Added for potential robust parsing
import re # Import regex module
from promptchain.utils.promptchaining import PromptChain
from promptchain.utils.logging_utils import RunLogger # Import logger
from typing import List, Dict, Any, Optional, Union, Tuple, Callable # Added Callable
import os # Added for example MCP check
import traceback # Added for error logging
import functools # Added for partial

class AgentChain:
    """
    Orchestrates multiple specialized PromptChain "agents" using a configurable
    routing mechanism (simple checks, default LLM chain, or custom function)
    to route tasks based on user input and history. Can also operate in a
    round-robin mode where each agent processes the input sequentially.

    Args:
        router (Union[Dict[str, Any], Callable]): The core routing logic (used in 'router' mode).
            - If Dict: Configuration for the default 2-step LLM decision chain.
              Requires 'models' (list of 1), 'instructions' ([None, template_str]),
              'decision_prompt_template' (str containing full logic for LLM).
            - If Callable: An async function `custom_router(user_input: str,
              history: List[Dict[str, str]], agent_descriptions: Dict[str, str]) -> str`.
              This function MUST return a JSON string like '{"chosen_agent": "agent_name"}'.
        agents (Dict[str, PromptChain]): Dictionary of pre-initialized PromptChain agents.
        agent_descriptions (Dict[str, str]): Descriptions for each agent.
        execution_mode (str): The mode of operation ('router' or 'pipeline').
                               Defaults to 'router'.
        max_history_tokens (int): Max tokens for history context formatting.
        log_dir (str): Directory for RunLogger.
        additional_prompt_dirs (Optional[List[str]]): Paths for loading prompts.
        verbose (bool): Enables detailed print logging.
    """
    def __init__(self,
                 router: Union[Dict[str, Any], Callable],
                 agents: Dict[str, PromptChain],
                 agent_descriptions: Dict[str, str],
                 execution_mode: str = "router",
                 max_history_tokens: int = 4000,
                 log_dir: str = "logs",
                 additional_prompt_dirs: Optional[List[str]] = None,
                 verbose: bool = False):

        # --- Validation ---
        if execution_mode not in ["router", "pipeline"]:
            raise ValueError("execution_mode must be either 'router' or 'pipeline'.")
        if not agents: raise ValueError("At least one agent must be provided.")
        if not agent_descriptions or agents.keys() != agent_descriptions.keys():
             raise ValueError("agent_descriptions must be provided for all agents.")

        # --- Initialize basic attributes --- 
        self.agents = agents
        self.agent_names = list(agents.keys())
        self.agent_descriptions = agent_descriptions
        self.verbose = verbose
        self.execution_mode = execution_mode
        self.max_history_tokens = max_history_tokens
        self.additional_prompt_dirs = additional_prompt_dirs
        self.logger = RunLogger(log_dir=log_dir)
        self.decision_maker_chain: Optional[PromptChain] = None
        self.custom_router_function: Optional[Callable] = None
        self._conversation_history: List[Dict[str, str]] = []
        self._tokenizer = None # Initialize tokenizer
        log_init_data = { "event": "AgentChain initialized", "execution_mode": self.execution_mode, "router_type": type(router).__name__, "agent_names": self.agent_names, "verbose": verbose }

        # --- Configure Router --- 
        if self.execution_mode == 'router':
            if isinstance(router, dict):
                # Validate and configure the default 2-step LLM router chain
                log_init_data["router_config"] = router.get('models', 'N/A')
                self._configure_default_llm_router(router) # Call helper method
                if self.verbose: print("AgentChain initialized with default 2-step LLM router.")
            elif callable(router):
                # ... (Store custom router function - unchanged) ...
                if not asyncio.iscoroutinefunction(router):
                     raise TypeError("Custom router function must be an async function (defined with 'async def').")
                self.custom_router_function = router
                if self.verbose: print("AgentChain initialized with custom router function.")
            else:
                raise TypeError("Invalid 'router' type. Must be Dict or async Callable.")

        self.logger.log_run(log_init_data)
        # ... (tokenizer setup - unchanged) ...
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            print("Warning: tiktoken not installed.")
            self.logger.log_run({"event": "warning", "message": "tiktoken not installed"})

    def _configure_default_llm_router(self, config: Dict[str, Any]):
        """Initializes the default 2-step LLM decision maker chain."""
        # Validation for 2-step structure
        if not all(k in config for k in ['models', 'instructions', 'decision_prompt_template']):
            raise ValueError("Router Dict config must contain 'models', 'instructions', and 'decision_prompt_template'.")
        if not isinstance(config['instructions'], list) or len(config['instructions']) != 2 \
           or config['instructions'][0] is not None \
           or not isinstance(config['instructions'][1], str): # Step 2 should be template like '{input}'
             raise ValueError("Router Dict 'instructions' must be a list like [None, template_str].")
        if not isinstance(config['models'], list) or len(config['models']) != 1: # Now needs 1 model
             raise ValueError("Router Dict 'models' must be a list of one model configuration (for step 2).")
        if not isinstance(config['decision_prompt_template'], str):
             raise ValueError("Router Dict 'decision_prompt_template' must be a string.")

        self._decision_prompt_template = config['decision_prompt_template'] # Store the main template
        decision_instructions = list(config['instructions'])
        # Inject the preparation function at index 0 (for step 1)
        decision_instructions[0] = functools.partial(self._prepare_full_decision_prompt, self)
        # Step 2 instruction should already be in the list (e.g., '{input}')

        try:
            self.decision_maker_chain = PromptChain(
                models=config['models'], # Expecting 1 model for step 2
                instructions=decision_instructions, # Now has 2 items [func, str]
                verbose=self.verbose,
                store_steps=True, # To see the prepared prompt if needed
                additional_prompt_dirs=self.additional_prompt_dirs,
                 **{k: v for k, v in config.items() if k not in ['models', 'instructions', 'decision_prompt_template']}
            )
            if self.verbose: print("Default 2-step decision maker chain configured.")
        except Exception as e:
            self.logger.log_run({"event": "error", "message": f"Failed to initialize default LLM router chain: {e}"})
            raise RuntimeError(f"Failed to initialize default LLM router chain: {e}") from e

    def _prepare_full_decision_prompt(self, context_self, user_input: str) -> str:
        """
        Prepares the single, comprehensive prompt for the LLM decision maker step.
        Incorporates history, agent descriptions, reasoning guidance, and JSON format instruction.
        This method is used as the callable instruction for Step 1.

        Args:
            context_self: The AgentChain instance (passed via functools.partial).
            user_input: The original user input starting the process.

        Returns:
            The formatted prompt string for the LLM decision step.
        """
        history_context = context_self._format_chat_history()
        agent_details = "\n".join([f" - {name}: {desc}" for name, desc in context_self.agent_descriptions.items()])

        # Use the stored template
        try:
            prompt = context_self._decision_prompt_template.format(
                user_input=user_input,
                history=history_context,
                agent_details=agent_details
            )
            if context_self.verbose:
                print("\n--- Preparing Full Decision Prompt (Step 1 Output / Step 2 Input) ---")
                print(f"User Input: {user_input}")
                print(f"Agent Details:\n{agent_details}")
                print(f"Full Prompt:\n{prompt}")
                print("-------------------------------------------------------------------\n")
            context_self.logger.log_run({"event": "prepare_full_decision_prompt", "input": user_input})
            return prompt
        except KeyError as e:
            error_msg = f"Missing placeholder in decision prompt template: {e}. Template requires {{user_input}}, {{history}}, {{agent_details}}."
            print(f"ERROR: {error_msg}")
            context_self.logger.log_run({"event": "error", "message": error_msg, "template": context_self._decision_prompt_template})
            return f"ERROR: Prompt template formatting failed: {e}"
        except Exception as e:
            error_msg = f"Unexpected error preparing full decision prompt: {e}"
            print(f"ERROR: {error_msg}")
            context_self.logger.log_run({"event": "error", "message": error_msg})
            return f"ERROR: Failed to prepare full decision prompt: {e}"

    def _count_tokens(self, text: str) -> int:
        """Counts tokens using tiktoken if available, otherwise uses character length estimate."""
        if not text: return 0
        if self._tokenizer:
             try:
                 return len(self._tokenizer.encode(text))
             except Exception as e:
                  if self.verbose: print(f"Tiktoken encoding error (falling back to char count): {e}")
                  self.logger.log_run({"event": "warning", "message": f"Tiktoken encoding error: {e}"})
                  return len(text) // 4 # Fallback on error
        else:
            return len(text) // 4 # Rough estimate

    def _format_chat_history(self, max_tokens: Optional[int] = None) -> str:
        """Formats chat history, truncating based on token count."""
        if not self._conversation_history:
            return "No previous conversation history."

        limit = max_tokens if max_tokens is not None else self.max_history_tokens
        formatted_history = []
        current_tokens = 0
        token_count_method = "tiktoken" if self._tokenizer else "character estimate"

        for message in reversed(self._conversation_history):
            role = message.get("role", "user").capitalize() # Default role if missing
            content = message.get("content", "")
            if not content: continue # Skip empty messages

            entry = f"{role}: {content}"
            entry_tokens = self._count_tokens(entry)

            # Check token limits *before* adding
            if current_tokens + entry_tokens <= limit:
                formatted_history.insert(0, entry)
                current_tokens += entry_tokens
            else:
                if self.verbose:
                    print(f"History truncation: Limit {limit} tokens ({token_count_method}) reached. Stopping history inclusion.")
                self.logger.log_run({"event": "history_truncated", "limit": limit, "current_tokens": current_tokens, "method": token_count_method})
                break # Stop adding messages

        final_history_str = "\n".join(formatted_history)
        if not final_history_str:
             return "No recent relevant history (token limit)."
        return final_history_str

    def _add_to_history(self, role: str, content: str):
        """Adds a message to the conversation history, ensuring role and content are present."""
        if not role or not content:
             self.logger.log_run({"event": "history_add_skipped", "reason": "Missing role or content"})
             return
             
        entry = {"role": role, "content": content}
        self._conversation_history.append(entry)
        if self.verbose:
            print(f"History updated - Role: {role}, Content: {content[:100]}...")
        # Log history addition minimally to avoid excessive logging
        self.logger.log_run({"event": "history_add", "role": role, "content_length": len(content)})

    def _parse_decision(self, decision_output: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses the decision maker's JSON output (expected from the final step).
        Requires valid JSON with a 'chosen_agent' key pointing to a known agent.
        """
        if not decision_output:
            self.logger.log_run({"event": "parse_decision_error", "reason": "Empty output from decision maker"})
            return None, None

        chosen_agent = None
        refined_query = None # Placeholder for potential future use

        try:
            # Attempt to parse the output as JSON, stripping potential ```json ``` markdown
            cleaned_output = decision_output.strip()
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[7:]
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[:-3]
            cleaned_output = cleaned_output.strip()

            if not cleaned_output: # Handle cases where cleaning results in empty string
                 raise json.JSONDecodeError("Cleaned output is empty", cleaned_output, 0)

            data = json.loads(cleaned_output)
            parsed_agent = data.get("chosen_agent")

            if parsed_agent and parsed_agent in self.agent_names:
                chosen_agent = parsed_agent
                refined_query = data.get("refined_query") # Extract if present
                # Logging and verbose printing happens within process_input after validation now
                # Return successful parse immediately
                return chosen_agent, refined_query
            else:
                # JSON parsed, but agent name is missing, invalid, or not in the list
                reason = f"JSON parsed, but 'chosen_agent' key missing, invalid ({parsed_agent}), or not in known agent list ({self.agent_names})."
                if self.verbose: print(f"Warning: {reason}")
                self.logger.log_run({"event": "parse_decision_warning", "method": "json", "raw_output": decision_output, "reason": reason})
                # Return None as agent is invalid
                return None, None

        except json.JSONDecodeError as e:
            # JSON parsing failed entirely
            reason = f"Decision Maker final output was not valid JSON: {e}. Raw output: '{decision_output}'"
            if self.verbose: print(f"Error: {reason}")
            self.logger.log_run({"event": "parse_decision_error", "method": "json", "raw_output": decision_output, "reason": f"JSONDecodeError: {e}"})
            # Return None as parsing failed
            return None, None

        except Exception as e:
            # Catch other potential errors during parsing
            error_msg = f"Unexpected error parsing decision: {e}"
            print(f"ERROR: {error_msg}")
            self.logger.log_run({"event": "parse_decision_error", "method": "unexpected", "raw_output": decision_output, "error": str(e)})
            # Return None on unexpected error
            return None, None

    def _simple_router(self, user_input: str) -> Optional[str]:
        """
        Performs simple pattern-based routing for common cases.
        Returns the name of the agent if a clear match is found, otherwise None.
        """
        # Simple Math Check: Look for numbers and common math operators
        # This is a basic example; more sophisticated regex/parsing could be used.
        math_pattern = r'[\d\.\s]+[\+\-\*\/^%]+[\d\.\s]+' # Looks for number-operator-number pattern
        # Keywords that often indicate math
        math_keywords = ['calculate', 'what is', 'solve for', 'compute', 'math problem']

        input_lower = user_input.lower()

        if re.search(math_pattern, user_input) or any(keyword in input_lower for keyword in math_keywords):
            # Check if the math agent exists
            if "math_agent" in self.agents:
                 if self.verbose: print("Simple Router: Detected potential math query.")
                 self.logger.log_run({"event": "simple_router_match", "input": user_input, "matched_agent": "math_agent"})
                 return "math_agent"
            else:
                 if self.verbose: print("Simple Router: Detected math query, but 'math_agent' not available.")
                 self.logger.log_run({"event": "simple_router_warning", "input": user_input, "reason": "Math detected, but no math_agent"})

        # Add more simple rules here if needed (e.g., for creative writing prompts)
        # creative_keywords = ['write a poem', 'tell a story', 'haiku']
        # if any(keyword in input_lower for keyword in creative_keywords):
        #      if "creative_agent" in self.agents:
        #          # ... log and return "creative_agent" ...
        #          pass

        # If no simple rules match, defer to LLM decision maker
        if self.verbose: print("Simple Router: No direct match, deferring to LLM Decision Maker.")
        self.logger.log_run({"event": "simple_router_defer", "input": user_input})
        return None

    async def process_input(self, user_input: str) -> str:
        """
        Processes user input based on the configured execution_mode.

        - 'router': Uses simple/complex routing to select and run one agent.
        - 'pipeline': Executes agents sequentially in definition order, passing output as input.
        """
        self.logger.log_run({"event": "process_input_start", "mode": self.execution_mode, "input": user_input})
        self._add_to_history("user", user_input)

        final_response = ""

        # --- Handle Pipeline Mode ---
        if self.execution_mode == 'pipeline':
            if self.verbose: print(f"\n--- Executing in Pipeline Mode ---")
            current_input = user_input
            agent_order = list(self.agents.keys()) # Get execution order

            for i, agent_name in enumerate(agent_order):
                agent_instance = self.agents[agent_name]
                step_num = i + 1
                if self.verbose: print(f"  Pipeline Step {step_num}/{len(agent_order)}: Running agent '{agent_name}'")
                self.logger.log_run({
                    "event": "pipeline_step_start",
                    "step": step_num,
                    "total_steps": len(agent_order),
                    "agent": agent_name,
                    "input_length": len(current_input)
                })
                try:
                    # Run the agent with the output from the previous step (or initial input)
                    agent_response = await agent_instance.process_prompt_async(current_input)
                    current_input = agent_response # Output becomes input for the next step
                    self.logger.log_run({
                        "event": "pipeline_step_success",
                        "step": step_num,
                        "agent": agent_name,
                        "response_length": len(current_input)
                    })
                    if self.verbose: print(f"    Output from {agent_name} (Input for next): {current_input[:150]}...") # Show truncated output
                except Exception as e:
                    error_msg = f"Error running agent {agent_name} in pipeline step {step_num}: {e}"
                    print(f"ERROR: {error_msg}")
                    if self.verbose: traceback.print_exc()
                    self.logger.log_run({
                        "event": "pipeline_step_error",
                        "step": step_num,
                        "agent": agent_name,
                        "error": str(e)
                    })
                    # Decide how to handle errors: stop pipeline or return error msg?
                    # Returning error message for now.
                    final_response = f"Pipeline failed at step {step_num} ({agent_name}): {e}"
                    # Add error to history and break pipeline
                    self._add_to_history("system_error", final_response)
                    self.logger.log_run({"event": "process_input_end", "mode": self.execution_mode, "status": "failed", "error_step": step_num})
                    return final_response # Exit pipeline on error

            # If pipeline completes, the final 'current_input' is the result
            final_response = current_input
            if not final_response:
                 final_response = "Pipeline completed but produced an empty final response."
                 self.logger.log_run({"event": "pipeline_empty_response"})


        # --- Handle Router Mode ---
        elif self.execution_mode == 'router':
            if self.verbose: print(f"\n--- Executing in Router Mode ---")
            chosen_agent_name: Optional[str] = None
            refined_query: Optional[str] = None
            decision_output: Optional[str] = None

            # 1. Simple Router Check
            chosen_agent_name = self._simple_router(user_input)
            if chosen_agent_name:
                 if self.verbose: print(f"Simple router selected: {chosen_agent_name}")
                 self.logger.log_run({"event": "router_decision", "source": "simple", "agent": chosen_agent_name})
            else:
                 # 2. Complex Router (LLM or Custom Function)
                 if self.verbose: print("Simple router deferred. Invoking complex router...")
                 try:
                     if self.custom_router_function:
                         if self.verbose: print("Using custom router function.")
                         decision_output = await self.custom_router_function(
                             user_input=user_input,
                             history=self._conversation_history,
                             agent_descriptions=self.agent_descriptions
                         )
                     elif self.decision_maker_chain:
                         if self.verbose: print("Using default LLM router chain.")
                         decision_output = await self.decision_maker_chain.process_prompt_async(user_input)
                     else:
                          raise RuntimeError("No valid router (LLM or custom) configured for router mode.")

                     if self.verbose: print(f"Complex router output: {decision_output}")
                     self.logger.log_run({"event": "complex_router_output", "output": decision_output})
                     chosen_agent_name, refined_query = self._parse_decision(decision_output)

                     if chosen_agent_name:
                          if self.verbose: print(f"Complex router selected: {chosen_agent_name}", f" (Refined query: {refined_query})" if refined_query else "")
                          self.logger.log_run({"event": "router_decision", "source": "complex", "agent": chosen_agent_name, "refined_query_present": bool(refined_query)})
                     else:
                          if self.verbose: print("Complex router failed to select a valid agent.")
                          self.logger.log_run({"event": "router_decision_failed", "source": "complex", "raw_output": decision_output})

                 except Exception as e:
                      error_msg = f"Error during complex routing: {e}"
                      print(f"ERROR: {error_msg}")
                      if self.verbose: traceback.print_exc()
                      self.logger.log_run({"event": "router_error", "error": str(e)})
                      final_response = f"Sorry, an error occurred during agent selection: {e}"

            # 3. Execute Chosen Agent (if selection successful)
            if not final_response and chosen_agent_name and chosen_agent_name in self.agents: # Check final_response flag
                 selected_agent = self.agents[chosen_agent_name]
                 query_for_agent = refined_query if refined_query else user_input
                 if self.verbose: print(f"Executing agent: {chosen_agent_name} with input: {query_for_agent[:100]}...")
                 self.logger.log_run({"event": "agent_execution_start", "agent": chosen_agent_name, "input_length": len(query_for_agent)})
                 try:
                     final_response = await selected_agent.process_prompt_async(query_for_agent)
                     self.logger.log_run({"event": "agent_execution_success", "agent": chosen_agent_name, "response_length": len(final_response)})
                 except Exception as e:
                      error_msg = f"Error executing agent {chosen_agent_name}: {e}"
                      print(f"ERROR: {error_msg}")
                      if self.verbose: traceback.print_exc()
                      self.logger.log_run({"event": "agent_execution_error", "agent": chosen_agent_name, "error": str(e)})
                      final_response = f"Sorry, agent {chosen_agent_name} encountered an error: {e}"
            elif not final_response: # Only set default error if no routing/execution error occurred
                 final_response = "Sorry, I could not determine the appropriate agent to handle your request."
                 self.logger.log_run({"event": "agent_selection_failed"})


        # --- Fallback for unknown mode ---
        else:
             final_response = f"Error: Unknown execution mode '{self.execution_mode}'."
             self.logger.log_run({"event": "error", "message": "Unknown execution mode", "mode": self.execution_mode})


        # --- Finalize and Log ---
        # Only add final response if pipeline didn't fail midway
        if self.execution_mode != 'pipeline' or "Pipeline failed at step" not in final_response:
             self._add_to_history("assistant", final_response)

        self.logger.log_run({"event": "process_input_end", "mode": self.execution_mode, "status": "success" if "Pipeline failed at step" not in final_response else "failed", "response_length": len(final_response)})
        return final_response

    async def run_agent_direct(self, agent_name: str, user_input: str) -> str:
        """
        Runs a specific agent directly, bypassing the routing logic.

        Args:
            agent_name: The name of the agent to run (must exist in self.agents).
            user_input: The input/query for the agent.

        Returns:
            The response string from the agent.

        Raises:
            ValueError: If the specified agent_name does not exist.
        """
        if not user_input: return "Input cannot be empty."
        if agent_name not in self.agents:
            error_msg = f"Error: Agent '{agent_name}' not found. Available agents: {list(self.agents.keys())}"
            print(error_msg)
            self.logger.log_run({"event": "run_direct_error", "reason": "Agent not found", "requested_agent": agent_name})
            # Optionally add to history? Decided against it for now as it wasn't a full processing attempt.
            # self._add_to_history("system_error", error_msg)
            raise ValueError(error_msg) # Raise error instead of returning message

        start_time = asyncio.get_event_loop().time()
        log_event_start = {"event": "run_direct_start", "agent_name": agent_name, "input": user_input}
        if self.verbose: print(f"\n=== Running Agent Directly: {agent_name} ===")
        if self.verbose: print(f"Input: '{user_input}'")
        self.logger.log_run(log_event_start)
        self._add_to_history("user", f"(Direct to {agent_name}) {user_input}") # Add user input, noting direct call

        selected_agent_chain = self.agents[agent_name]
        query_for_agent = user_input # Direct input for the agent

        try:
            if self.verbose:
                 print(f"--- Executing Agent: {agent_name} ---")
            self.logger.log_run({"event": "direct_agent_running", "agent_name": agent_name, "input": query_for_agent})

            # Execute the chosen agent
            agent_response = await selected_agent_chain.process_prompt_async(query_for_agent)

            if self.verbose:
                print(f"Agent Response ({agent_name}): {agent_response}")
                print(f"--- Finished Agent: {agent_name} ---")
            self.logger.log_run({"event": "direct_agent_finished", "agent_name": agent_name, "response_length": len(agent_response)})

            # Update History & Return
            self._add_to_history("assistant", agent_response) # Store the agent's final response

            end_time = asyncio.get_event_loop().time()
            duration = int((end_time - start_time) * 1000)
            self.logger.log_run({"event": "run_direct_end", "agent_name": agent_name, "response_length": len(agent_response), "duration_ms": duration})
            return agent_response

        except Exception as e:
             error_msg = f"An error occurred during direct execution of agent '{agent_name}': {e}"
             print(f"ERROR: {error_msg}")
             traceback.print_exc()
             self._add_to_history("system_error", error_msg)
             self.logger.log_run({"event": "run_direct_error", "agent_name": agent_name, "error": str(e), "traceback": traceback.format_exc()})
             # Return the error message or re-raise? Returning for now.
             return f"I encountered an error running the {agent_name}: {e}"

    async def run_chat(self):
        """
        Runs an interactive chat loop. Supports automatic routing or direct
        agent execution using '@agent_name: your message' syntax.
        """
        print("Starting Agentic Chat...")
        print("Type 'exit' to quit.")
        print("Use '@agent_name: your message' to run an agent directly (e.g., '@math_agent: 5*5').")
        self.logger.log_run({"event": "chat_started"})

        turn = 1
        while True:
            try:
                user_message_full = input(f"\n[{turn}] You: ")
            except EOFError:
                print("\nInput stream closed. Exiting.")
                self.logger.log_run({"event": "chat_ended", "reason": "EOF"})
                break
            except KeyboardInterrupt:
                 print("\nKeyboard interrupt detected. Exiting.")
                 self.logger.log_run({"event": "chat_ended", "reason": "KeyboardInterrupt"})
                 break

            if user_message_full.lower() == 'exit':
                print("🛑 Exit command received.")
                self.logger.log_run({"event": "chat_ended", "reason": "Exit command"})
                break

            if not user_message_full.strip():
                print("Please enter a message.")
                continue

            response = ""
            try:
                # Check for direct agent call syntax: @agent_name: message
                direct_match = re.match(r"@(\w+):\s*(.*)", user_message_full, re.DOTALL)

                if direct_match:
                    target_agent, user_actual_message = direct_match.groups()
                    if target_agent in self.agents:
                        if self.verbose: print(f"~~ Direct agent call detected for: {target_agent} ~~")
                        # Directly execute run_agent_direct and handle its potential errors
                        try:
                            response = await self.run_agent_direct(target_agent, user_actual_message.strip())
                        except ValueError as ve: # Catch agent not found from run_agent_direct itself
                            print(str(ve))
                            response = "Please try again with a valid agent name."
                        except Exception as direct_e: # Catch other errors during direct run
                            print(f"Error during direct agent execution: {direct_e}")
                            response = f"Sorry, an error occurred while running {target_agent}: {direct_e}"
                            self.logger.log_run({"event": "run_direct_unexpected_error", "agent_name": target_agent, "error": str(direct_e), "traceback": traceback.format_exc()})
                    else:
                        # Agent specified directly but not found - DO NOT fall through
                        print(f"Error: Agent '{target_agent}' not found. Available: {list(self.agents.keys())}")
                        response = f"Agent '{target_agent}' not recognized. Please choose from {list(self.agents.keys())} or use automatic routing."
                        # Log this specific error
                        self.logger.log_run({"event": "direct_call_agent_not_found", "requested_agent": target_agent})
                        # We have a response, skip the automatic routing below

                else:
                    # No direct call syntax detected, proceed with automatic routing
                    response = await self.process_input(user_message_full)

            # Removed outer ValueError catch as run_agent_direct now raises it, handled above
            except Exception as e: # Catch other unexpected errors during processing (e.g., in process_input)
                 print(f"An unexpected error occurred in chat loop: {e}")
                 response = f"Sorry, an error occurred: {e}"
                 # Log this unexpected error if not already logged by process_input/run_agent_direct
                 self.logger.log_run({"event": "chat_loop_error", "error": str(e), "traceback": traceback.format_exc()})


            print(f"\nAssistant: {response}")
            turn += 1

        print("\n--- Chat Finished ---")


# --- Example Usage ---
async def run_example_agent_chat():

    # --- Agent Setup (remains the same) ---
    # math_agent (with local tool)
    # ... (calculator function and schema) ...
    def simple_calculator(expression: str) -> str:
        try:
            allowed_names = {"abs": abs, "pow": pow}
            result = eval(expression, {"__builtins__": None}, allowed_names)
            return f"The result of {expression} is {result}"
        except Exception as e:
            return f"Could not calculate '{expression}': {e}"
    calculator_schema = { "type": "function", "function": { "name": "simple_calculator", "description": "Evaluates a simple mathematical expression (e.g., '2 + 2', '5 * 8').", "parameters": { "type": "object", "properties": { "expression": {"type": "string", "description": "The mathematical expression to evaluate."} }, "required": ["expression"] } } }
    math_agent = PromptChain( models=["gpt-4o-mini"], instructions=["Use the calculator tool if needed to answer the question: {input}"], verbose=False )
    math_agent.add_tools([calculator_schema])
    math_agent.register_tool_function(simple_calculator)

    # doc_agent (potential MCP)
    # ... (MCP config setup remains the same, using placeholders) ...
    python_executable = "/path/to/your/python/env/bin/python" # <<< CHANGE THIS
    mcp_server_script = "/path/to/your/context7_mcp_server.py" # <<< CHANGE THIS
    mcp_server_config = None
    if os.path.exists(python_executable) and os.path.exists(mcp_server_script):
         mcp_server_config = [{"id": "context7_server", "type": "stdio", "command": python_executable, "args": [mcp_server_script], "env": {}}]
    else:
        print("Warning: MCP server path not found. 'doc_agent' will run without MCP tools.")
    doc_agent = PromptChain( models=["gpt-4o-mini"], instructions=["Answer questions based on documentation. Use Context7 tools if available. Query: {input}"], mcp_servers=mcp_server_config, verbose=False )
    # MCP connection logic placeholder remains the same

    # creative_agent
    creative_agent = PromptChain( models=["gpt-4o-mini"], instructions=["Write a short, creative piece based on this topic: {input}"], verbose=False )

    # Agent Descriptions (NEW)
    agent_descriptions = {
        "math_agent": "Solves mathematical problems or calculations. Uses a calculator tool.",
        "doc_agent": "Answers questions based on technical documentation (may use Context7 tools if available and connected).",
        "creative_agent": "Writes poems, stories, or other creative text."
    }

    # --- Router Configuration --- 

    # Option 1: Use the default 2-step LLM Router
    # This template is now used by the prepare function (Step 1)
    full_decision_template = """Analyze the user input and conversation history to choose the best agent.
Consider the agent's purpose and the user's request category (e.g., math, creative, documentation).

Available Agents:
{agent_details}

Conversation History (recent first):
{history}

User Input:
{user_input}

Respond with ONLY a JSON object containing the key "chosen_agent" and the name of the best agent. Example: {{"chosen_agent": "creative_agent"}}"""

    default_llm_router_config = {
        "models": ["gpt-4o-mini"], # Only one model needed for the execution step
        "instructions": [None, "{input}"], # Step 1 func placeholder, Step 2 executes prepared prompt
        "decision_prompt_template": full_decision_template # Template for Step 1 function
    }

    # Option 2: Define a Custom Async Router Function
    async def my_custom_router(user_input: str, history: List[Dict[str, str]], agent_descriptions: Dict[str, str]) -> str:
        # ... (custom router implementation unchanged) ...
        input_lower = user_input.lower()
        print(f"--- Custom Router Running for: '{user_input}' ---")
        chosen = "creative_agent" # Default
        if "calculate" in input_lower or any(char.isdigit() for char in user_input): chosen = "math_agent"
        elif "doc" in input_lower or "context7" in input_lower: chosen = "doc_agent"
        decision = {"chosen_agent": chosen}
        print(f"--- Custom Router Chose: {chosen} ---")
        return json.dumps(decision)

    # --- Initialize AgentChain --- 
    # CHOOSE WHICH ROUTER TO USE HERE:
    router_to_use = default_llm_router_config
    # router_to_use = my_custom_router

    agent_orchestrator = AgentChain(
        router=router_to_use,
        agents={
            "math_agent": math_agent,
            "doc_agent": doc_agent,
            "creative_agent": creative_agent
        },
        agent_descriptions=agent_descriptions,
        verbose=True,
    )

    # --- Run Chat (remains the same) ---
    try:
         await agent_orchestrator.run_chat()
    finally:
        # --- Cleanup (remains the same placeholder) ---
        pass

if __name__ == "__main__":
    # ... (main execution block remains the same) ...
    try:
        asyncio.run(run_example_agent_chat())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
    except Exception as e:
        traceback.print_exc() 