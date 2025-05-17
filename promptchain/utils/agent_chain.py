# agent_chain.py
import asyncio
import json
import re
from promptchain.utils.promptchaining import PromptChain
from promptchain.utils.logging_utils import RunLogger
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, TYPE_CHECKING
import os
import traceback
import functools
import logging
import uuid
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.theme import Theme

# Strategy imports
from .strategies.single_dispatch_strategy import execute_single_dispatch_strategy_async
from .strategies.static_plan_strategy import execute_static_plan_strategy_async
from .strategies.dynamic_decomposition_strategy import execute_dynamic_decomposition_strategy_async

# Setup logger for this module
logger = logging.getLogger(__name__)

class AgentChain:
    """
    Orchestrates multiple specialized PromptChain "agents" using a configurable
    routing mechanism or predefined execution flow.

    Execution Modes:
    - `router`: Uses a router (simple checks, LLM, or custom func) to select
                one agent per turn. Can support internal agent-to-agent loops
                if agents output `[REROUTE] next_input`.
    - `pipeline`: Executes all agents sequentially in definition order,
                  passing output as input to the next.
    - `round_robin`: Cycles through agents, executing one per turn.
    - `broadcast`: Executes all agents in parallel and synthesizes results
                   (requires `synthesizer_config`).

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
        execution_mode (str): The mode of operation ('router', 'pipeline', 'round_robin', 'broadcast').
                               Defaults to 'router'.
        max_history_tokens (int): Max tokens for history context formatting.
        log_dir (Optional[str]): Directory for optional RunLogger JSONL file logging.
                                 If None (default), only console logging occurs via RunLogger.
        additional_prompt_dirs (Optional[List[str]]): Paths for loading prompts.
        verbose (bool): Enables detailed print logging by AgentChain itself (distinct from RunLogger console output).
        max_internal_steps (int): Max loops within router mode for agent-to-agent re-routes (default: 3).
        synthesizer_config (Optional[Dict[str, Any]]): Config for broadcast synthesizer.
                                                       Required for 'broadcast' mode.
                                                       Example: {"model": "gpt-4o-mini", "prompt": "Synthesize: {agent_responses}"}
        router_strategy (str): The strategy for selecting the decision prompt template.
        auto_include_history (bool): When True, automatically include conversation history in all agent calls.
    """
    def __init__(self,
                 # Required arguments first
                 agents: Dict[str, PromptChain],
                 agent_descriptions: Dict[str, str],
                 # Optional arguments follow
                 router: Optional[Union[Dict[str, Any], Callable]] = None,
                 execution_mode: str = "router",
                 max_history_tokens: int = 4000,
                 log_dir: Optional[str] = None,
                 additional_prompt_dirs: Optional[List[str]] = None,
                 verbose: bool = False,
                 **kwargs): # Use kwargs for new optional parameters

        # --- Validation ---
        valid_modes = ["router", "pipeline", "round_robin", "broadcast"]
        if execution_mode not in valid_modes:
            raise ValueError(f"execution_mode must be one of {valid_modes}.")
        if not agents:
            raise ValueError("At least one agent must be provided.")
        if not agent_descriptions or set(agents.keys()) != set(agent_descriptions.keys()):
            raise ValueError("agent_descriptions must be provided for all agents and match agent keys.")
        # --> Add validation: Router is required if mode is router <--
        if execution_mode == "router" and router is None:
            raise ValueError("A 'router' configuration (Dict or Callable) must be provided when execution_mode is 'router'.")

        # --- Initialize basic attributes ---
        self.agents = agents
        self.agent_names = list(agents.keys())
        self.agent_descriptions = agent_descriptions
        self.verbose = verbose
        self.execution_mode = execution_mode
        self.max_history_tokens = max_history_tokens
        self.additional_prompt_dirs = additional_prompt_dirs
        self.max_internal_steps = kwargs.get("max_internal_steps", 3) # For router internal loop
        self.synthesizer_config = kwargs.get("synthesizer_config", None) # For broadcast mode
        
        # <<< New: Router Strategy >>>
        self.router_strategy = kwargs.get("router_strategy", "single_agent_dispatch")
        # Enable auto-history inclusion when set
        self.auto_include_history = kwargs.get("auto_include_history", False)
        valid_router_strategies = ["single_agent_dispatch", "static_plan", "dynamic_decomposition"]
        if self.execution_mode == 'router' and self.router_strategy not in valid_router_strategies:
            raise ValueError(f"Invalid router_strategy '{self.router_strategy}'. Must be one of {valid_router_strategies}")
        log_init_data = {
            "event": "AgentChain initialized",
            "execution_mode": self.execution_mode,
            "router_type": type(router).__name__ if execution_mode == 'router' else 'N/A',
            "agent_names": self.agent_names,
            "verbose": verbose,
            "file_logging_enabled": bool(log_dir),
            "max_internal_steps": self.max_internal_steps if execution_mode == 'router' else 'N/A',
            "synthesizer_configured": bool(self.synthesizer_config) if execution_mode == 'broadcast' else 'N/A',
            "router_strategy": self.router_strategy if self.execution_mode == 'router' else 'N/A',
            "auto_include_history": self.auto_include_history
        }
        # <<< End New: Router Strategy >>>
        
        # --- Session log filename (set to None initially, will be set at chat start) ---
        self.session_log_filename = None
        self.logger = RunLogger(log_dir=log_dir)
        self.decision_maker_chain: Optional[PromptChain] = None
        self.custom_router_function: Optional[Callable] = None
        self._round_robin_index = 0 # For round_robin mode
        self._conversation_history: List[Dict[str, str]] = []
        self._tokenizer = None # Initialize tokenizer

        # --- Configure Router ---
        if self.execution_mode == 'router':
            if isinstance(router, dict):
                log_init_data["router_config_model"] = router.get('models', ['N/A'])[0]
                self._configure_default_llm_router(router) # Pass the whole config dict
                if self.verbose: print(f"AgentChain router mode initialized with default LLM router (Strategy: {self.router_strategy}).")
            elif callable(router):
                # Custom router functions aren't inherently strategy-aware unless designed to be.
                # We might need to disallow custom functions if strategy != single_agent_dispatch, or document the required output format.
                if self.router_strategy != "single_agent_dispatch":
                    logging.warning(f"Custom router function provided with strategy '{self.router_strategy}'. Ensure the function output matches the strategy's expectations.")
                if not asyncio.iscoroutinefunction(router):
                    raise TypeError("Custom router function must be an async function (defined with 'async def').")
                self.custom_router_function = router
                if self.verbose: print("AgentChain router mode initialized with custom router function.")
            else:
                raise TypeError("Invalid 'router' type for router mode. Must be Dict or async Callable.")

        self.logger.log_run(log_init_data)
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logging.warning("tiktoken not installed. Using character count for history truncation.")
            self.logger.log_run({"event": "warning", "message": "tiktoken not installed, using char count for history"})

    def _configure_default_llm_router(self, config: Dict[str, Any]):
        """Initializes the default 2-step LLM decision maker chain."""
        # --- Updated Validation --- 
        # We now expect 'decision_prompt_templates' (plural) which should be a Dict
        if not all(k in config for k in ['models', 'instructions', 'decision_prompt_templates']):
            raise ValueError("Router Dict config must contain 'models', 'instructions', and 'decision_prompt_templates'.")
        if not isinstance(config['decision_prompt_templates'], dict):
            raise ValueError("Router Dict 'decision_prompt_templates' must be a dictionary mapping strategy names to prompt strings.")
        if self.router_strategy not in config['decision_prompt_templates']:
            raise ValueError(f"Router Dict 'decision_prompt_templates' is missing a template for the selected strategy: '{self.router_strategy}'. Available: {list(config['decision_prompt_templates'].keys())}")
        if not isinstance(config['decision_prompt_templates'][self.router_strategy], str):
            raise ValueError(f"Value for strategy '{self.router_strategy}' in 'decision_prompt_templates' must be a string.")
             
        # Basic validation for instructions and models remains the same
        if not isinstance(config['instructions'], list) or len(config['instructions']) != 2 \
           or config['instructions'][0] is not None \
           or not isinstance(config['instructions'][1], str): # Step 2 should be template like '{input}'
            raise ValueError("Router Dict 'instructions' must be a list like [None, template_str].")
        if not isinstance(config['models'], list) or len(config['models']) != 1:
            raise ValueError("Router Dict 'models' must be a list of one model configuration (for step 2).")
        # --- End Updated Validation ---

        # Store the *dictionary* of templates
        self._decision_prompt_templates = config['decision_prompt_templates'] 
        decision_instructions = list(config['instructions'])
        # Inject the preparation function at index 0 (for step 1)
        # _prepare_full_decision_prompt will now use self.router_strategy to pick the template
        decision_instructions[0] = functools.partial(self._prepare_full_decision_prompt, self)
        # Step 2 instruction should already be in the list (e.g., '{input}')

        try:
            self.decision_maker_chain = PromptChain(
                models=config['models'], # Expecting 1 model for step 2
                instructions=decision_instructions, # Now has 2 items [func, str]
                verbose=self.verbose,
                store_steps=True, # To see the prepared prompt if needed
                additional_prompt_dirs=self.additional_prompt_dirs,
                # Pass other config items, excluding the ones we handled
                **{k: v for k, v in config.items() if k not in ['models', 'instructions', 'decision_prompt_templates']}
            )
            if self.verbose: print("Default 2-step decision maker chain configured.")
        except Exception as e:
            self.logger.log_run({"event": "error", "message": f"Failed to initialize default LLM router chain: {e}"})
            raise RuntimeError(f"Failed to initialize default LLM router chain: {e}") from e

    def _prepare_full_decision_prompt(self, context_self, user_input: str) -> str:
        """
        Prepares the single, comprehensive prompt for the LLM decision maker step,
        selecting the correct template based on the current router_strategy.

        Args:
            context_self: The AgentChain instance (passed via functools.partial).
            user_input: The original user input starting the process.

        Returns:
            The formatted prompt string for the LLM decision step.
        """
        history_context = context_self._format_chat_history()
        agent_details = "\n".join([f" - {name}: {desc}" for name, desc in context_self.agent_descriptions.items()])

        # --- Select the template based on strategy ---
        template = context_self._decision_prompt_templates.get(context_self.router_strategy)
        if not template:
            # This should have been caught in __init__, but double-check
            error_msg = f"Internal Error: No decision prompt template found for strategy '{context_self.router_strategy}'."
            logging.error(error_msg)
            context_self.logger.log_run({"event": "error", "message": error_msg})
            return f"ERROR: {error_msg}"
        # --- End Template Selection ---
        
        # Use the selected template
        try:
            prompt = template.format(
                user_input=user_input,
                history=history_context,
                agent_details=agent_details
            )
            if context_self.verbose:
                print("\n--- Preparing Full Decision Prompt (Step 1 Output / Step 2 Input) ---")
                print(f"Full Prompt for Decision LLM:\n{prompt}")
                print("-------------------------------------------------------------------\n")
            context_self.logger.log_run({"event": "prepare_full_decision_prompt", "input_length": len(user_input)})
            return prompt
        except KeyError as e:
            error_msg = f"Missing placeholder in decision prompt template: {e}. Template requires {{user_input}}, {{history}}, {{agent_details}}."
            logging.error(error_msg)
            context_self.logger.log_run({"event": "error", "message": error_msg, "template": template})
            # Return error message so chain processing stops cleanly
            return f"ERROR: Prompt template formatting failed: {e}"
        except Exception as e:
            error_msg = f"Unexpected error preparing full decision prompt: {e}"
            logging.error(error_msg, exc_info=True)
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
                # Log the error but fallback
                return len(text) // 4 # Fallback estimate
        else:
            # Tokenizer not available
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
                self.logger.log_run({"event": "history_truncated", "limit": limit, "final_token_count": current_tokens, "method": token_count_method})
                break # Stop adding messages

        final_history_str = "\n".join(formatted_history)
        if not final_history_str:
            # Check if history actually existed but was truncated completely
            if self._conversation_history:
                return "No recent relevant history (token limit reached)."
            else:
                return "No previous conversation history."
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

    def _clean_json_output(self, output: str) -> str:
        """Strips markdown code fences from potential JSON output."""
        cleaned = output.strip()
        if cleaned.startswith("```json"):
            match = re.search(r"```json\s*({.*?})\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
            else:
                cleaned = cleaned.split("```json", 1)[-1]
                cleaned = cleaned.rsplit("```", 1)[0]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def _parse_decision(self, decision_output: str) -> Dict[str, Any]:
        """Parses the decision maker's JSON output based on the router strategy."""
        if not decision_output:
            self.logger.log_run({"event": "parse_decision_error", "reason": "Empty output"})
            return {}

        # Clean the output
        cleaned_output = self._clean_json_output(decision_output)
        if not cleaned_output:
            self.logger.log_run({"event": "parse_decision_error", "reason": "Cleaned output is empty", "raw": decision_output})
            return {}
        
        data = None # Initialize data to avoid UnboundLocalError in exception handlers
        try:
            data = json.loads(cleaned_output)
            parsed_result = {}

            if self.router_strategy == "single_agent_dispatch":
                chosen_agent = data.get("chosen_agent")
                if chosen_agent and chosen_agent in self.agent_names:
                    parsed_result["chosen_agent"] = chosen_agent
                    parsed_result["refined_query"] = data.get("refined_query") # Optional
                    self.logger.log_run({
                        "event": "parse_decision_success", "strategy": "single_agent_dispatch",
                        "result": parsed_result
                    })
                else:
                    raise ValueError(f"Missing or invalid 'chosen_agent' for single_agent_dispatch strategy. Found: {chosen_agent}")

            elif self.router_strategy == "static_plan":
                plan = data.get("plan")
                initial_input = data.get("initial_input")
                # Validate plan
                if isinstance(plan, list) and all(agent_name in self.agent_names for agent_name in plan) and plan:
                    parsed_result["plan"] = plan
                    # Validate initial_input (must be present and a string)
                    if isinstance(initial_input, str):
                        parsed_result["initial_input"] = initial_input
                        self.logger.log_run({
                            "event": "parse_decision_success", "strategy": "static_plan",
                            "result": parsed_result
                        })
                    else:
                        raise ValueError(f"Missing or invalid 'initial_input' (must be string) for static_plan strategy. Found: {initial_input}")
                else:
                    raise ValueError(f"Missing or invalid 'plan' (must be non-empty list of valid agent names) for static_plan strategy. Found: {plan}")

            elif self.router_strategy == "dynamic_decomposition":
                next_action = data.get("next_action")
                is_complete = data.get("is_task_complete", False) # Default to false

                if next_action == "call_agent":
                    agent_name = data.get("agent_name")
                    agent_input = data.get("agent_input")
                    if agent_name and agent_name in self.agent_names and isinstance(agent_input, str):
                        parsed_result["next_action"] = "call_agent"
                        parsed_result["agent_name"] = agent_name
                        parsed_result["agent_input"] = agent_input
                        parsed_result["is_task_complete"] = False # Explicitly false
                        self.logger.log_run({
                            "event": "parse_decision_success", "strategy": "dynamic_decomposition",
                            "result": parsed_result
                        })
                    else:
                        raise ValueError(f"Missing or invalid 'agent_name' or 'agent_input' for dynamic_decomposition 'call_agent' action. Found: name={agent_name}, input_type={type(agent_input).__name__}")
                elif next_action == "final_answer":
                    final_answer = data.get("final_answer")
                    if isinstance(final_answer, str):
                        parsed_result["next_action"] = "final_answer"
                        parsed_result["final_answer"] = final_answer
                        parsed_result["is_task_complete"] = True # Explicitly true
                        self.logger.log_run({
                            "event": "parse_decision_success", "strategy": "dynamic_decomposition",
                            "result": parsed_result
                        })
                    else:
                        raise ValueError(f"Missing or invalid 'final_answer' (must be string) for dynamic_decomposition 'final_answer' action. Found type: {type(final_answer).__name__}")
                # Add elif for "clarify" action later if needed
                else:
                    raise ValueError(f"Invalid 'next_action' value for dynamic_decomposition strategy. Must be 'call_agent' or 'final_answer'. Found: {next_action}")

            else:
                raise ValueError(f"Parsing logic not implemented for router strategy: {self.router_strategy}")

            return parsed_result

        except json.JSONDecodeError as e:
            self.logger.log_run({
                "event": "parse_decision_error", "strategy": self.router_strategy,
                "reason": f"JSONDecodeError: {e}", "raw": decision_output, "cleaned": cleaned_output
            })
            return {}
        except ValueError as e: # Catch validation errors
            self.logger.log_run({
                "event": "parse_decision_error", "strategy": self.router_strategy,
                "reason": f"ValidationError: {e}", "raw": decision_output, "cleaned": cleaned_output, "parsed_data": data
            })
            return {}
        except Exception as e:
            self.logger.log_run({
                "event": "parse_decision_error", "strategy": self.router_strategy,
                "reason": f"Unexpected error: {e}", "raw": decision_output, "cleaned": cleaned_output
            }, exc_info=True)
            return {}

    def _simple_router(self, user_input: str) -> Optional[str]:
        """Performs simple pattern-based routing (example)."""
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
        #     if "creative_agent" in self.agents:
        #         # ... log and return "creative_agent" ...
        #         pass

        # If no simple rules match, defer to LLM decision maker
        if self.verbose: print("Simple Router: No direct match, deferring to Complex Router.")
        self.logger.log_run({"event": "simple_router_defer", "input": user_input})
        return None

    async def process_input(self, user_input: str, override_include_history: Optional[bool] = None) -> str:
        """Processes user input based on the configured execution_mode."""
        self.logger.log_run({"event": "process_input_start", "mode": self.execution_mode, "input": user_input})
        self._add_to_history("user", user_input)

        # Determine if we should include history for this call
        include_history = self.auto_include_history
        if override_include_history is not None:
            include_history = override_include_history
            if self.verbose and include_history != self.auto_include_history:
                print(f"History inclusion overridden for this call: {include_history}")
                self.logger.log_run({"event": "history_inclusion_override", "value": include_history})

        final_response = ""
        agent_order = list(self.agents.keys()) # Defined once for pipeline/round_robin

        if self.execution_mode == 'pipeline':
            if self.verbose: print(f"\n--- Executing in Pipeline Mode ---")
            current_input = user_input
            pipeline_failed = False
            for i, agent_name in enumerate(agent_order):
                agent_instance = self.agents[agent_name]
                step_num = i + 1
                pipeline_step_input = current_input
                
                # Include history if configured
                if include_history:
                    formatted_history = self._format_chat_history()
                    pipeline_step_input = f"Conversation History:\n---\n{formatted_history}\n---\n\nCurrent Input:\n{pipeline_step_input}"
                    if self.verbose:
                        print(f"  Including history for pipeline step {step_num} ({agent_name})")
                
                if self.verbose:
                    print(f"  Pipeline Step {step_num}/{len(agent_order)}: Running agent '{agent_name}'")
                    print(f"    Input Type: String")
                    print(f"    Input Content (Truncated): {str(pipeline_step_input)[:100]}...")
                self.logger.log_run({
                    "event": "pipeline_step_start",
                    "step": step_num,
                    "total_steps": len(agent_order),
                    "agent": agent_name,
                    "input_type": "string",
                    "input_length": len(str(pipeline_step_input)),
                    "history_included": include_history
                })
                try:
                    agent_response = await agent_instance.process_prompt_async(pipeline_step_input)
                    current_input = agent_response
                    self.logger.log_run({
                        "event": "pipeline_step_success",
                        "step": step_num,
                        "agent": agent_name,
                        "response_length": len(current_input)
                    })
                    if self.verbose: print(f"    Output from {agent_name} (Input for next): {current_input[:150]}...")
                except Exception as e:
                    error_msg = f"Error running agent {agent_name} in pipeline step {step_num}: {e}"
                    logging.error(error_msg, exc_info=True)
                    self.logger.log_run({
                        "event": "pipeline_step_error",
                        "step": step_num,
                        "agent": agent_name,
                        "error": str(e)
                    })
                    final_response = f"Pipeline failed at step {step_num} ({agent_name}): {e}"
                    pipeline_failed = True
                    break
            if not pipeline_failed:
                final_response = current_input
            if not final_response:
                final_response = "[Pipeline completed but produced an empty final response.]"
                self.logger.log_run({"event": "pipeline_empty_response"})

        elif self.execution_mode == 'round_robin':
            if self.verbose: print(f"\n--- Executing in Round Robin Mode ---")
            if not self.agents:
                final_response = "Error: No agents configured for round_robin."
                self.logger.log_run({"event": "round_robin_error", "reason": "No agents"})
            else:
                # Choose the next agent in the rotation
                selected_agent_name = agent_order[self._round_robin_index]
                # Update the index for next call
                self._round_robin_index = (self._round_robin_index + 1) % len(agent_order)
                
                agent_instance = self.agents[selected_agent_name]
                round_robin_input = user_input
                
                # Include history if configured
                if include_history:
                    formatted_history = self._format_chat_history()
                    round_robin_input = f"Conversation History:\n---\n{formatted_history}\n---\n\nCurrent Input:\n{round_robin_input}"
                    if self.verbose:
                        print(f"  Including history for round-robin agent {selected_agent_name}")
                
                if self.verbose:
                    print(f"  Round Robin: Selected agent '{selected_agent_name}' (Next index: {self._round_robin_index})")
                    print(f"    Input Content (Truncated): {round_robin_input[:100]}...")
                self.logger.log_run({
                    "event": "round_robin_agent_start",
                    "agent": selected_agent_name,
                    "next_index": self._round_robin_index,
                    "history_included": include_history
                })
                try:
                    final_response = await agent_instance.process_prompt_async(round_robin_input)
                    self.logger.log_run({
                        "event": "round_robin_agent_success",
                        "agent": selected_agent_name,
                        "response_length": len(final_response)
                    })
                    if self.verbose: print(f"    Output from {selected_agent_name}: {final_response[:150]}...")
                except Exception as e:
                    error_msg = f"Error running agent {selected_agent_name} in round_robin: {e}"
                    logging.error(error_msg, exc_info=True)
                    self.logger.log_run({
                        "event": "round_robin_agent_error",
                        "agent": selected_agent_name,
                        "error": str(e)
                    })
                    final_response = f"Round robin agent {selected_agent_name} failed: {e}"

        elif self.execution_mode == 'router':
            # For router mode we'll delegate to a strategy-specific function
            # based on the configured router_strategy
            if self.router_strategy == "single_agent_dispatch":
                # Execute with the original single-agent dispatch strategy
                if self.verbose: print(f"\n--- Executing in Router Mode (Strategy: single_agent_dispatch) ---")
                self.logger.log_run({"event": "router_strategy", "strategy": "single_agent_dispatch"})
                strategy_data = {
                    "user_input": user_input,
                    "include_history": include_history
                }
                final_response = await execute_single_dispatch_strategy_async(self, user_input, strategy_data)
            
            elif self.router_strategy == "static_plan":
                # Execute with the static plan strategy
                if self.verbose: print(f"\n--- Executing in Router Mode (Strategy: static_plan) ---")
                self.logger.log_run({"event": "router_strategy", "strategy": "static_plan"})
                final_response = await execute_static_plan_strategy_async(self, user_input)
            
            elif self.router_strategy == "dynamic_decomposition":
                # Execute with the dynamic decomposition strategy
                if self.verbose: print(f"\n--- Executing in Router Mode (Strategy: dynamic_decomposition) ---")
                self.logger.log_run({"event": "router_strategy", "strategy": "dynamic_decomposition"})
                final_response = await execute_dynamic_decomposition_strategy_async(self, user_input)
            
            else:
                # This shouldn't happen due to validation in __init__ but just in case
                raise ValueError(f"Unsupported router_strategy: {self.router_strategy}")

        elif self.execution_mode == 'broadcast':
            if self.verbose: print(f"\n--- Executing in Broadcast Mode ---")
            if not self.agents:
                final_response = "Error: No agents configured for broadcast."
                self.logger.log_run({"event": "broadcast_error", "reason": "No agents"})
            else:
                tasks = []
                agent_names_list = list(self.agents.keys())
                for agent_name in agent_names_list:
                    agent_instance = self.agents[agent_name]
                    
                    # Prepare input with history if configured
                    broadcast_input = user_input
                    if include_history:
                        formatted_history = self._format_chat_history()
                        broadcast_input = f"Conversation History:\n---\n{formatted_history}\n---\n\nCurrent Input:\n{broadcast_input}"
                        if self.verbose:
                            print(f"  Including history for broadcast agent {agent_name}")
                    
                    async def run_agent_task(name, agent, inp):
                        try:
                            return name, await agent.process_prompt_async(inp)
                        except Exception as e:
                            return name, e
                    tasks.append(run_agent_task(agent_name, agent_instance, broadcast_input))
                    self.logger.log_run({
                        "event": "broadcast_agent_start", 
                        "agent": agent_name,
                        "history_included": include_history
                    })
                results_with_names = await asyncio.gather(*tasks)
                processed_results = {}
                for agent_name, result in results_with_names:
                    if isinstance(result, Exception):
                        logging.error(f"Error executing agent {agent_name} during broadcast: {result}", exc_info=result)
                        self.logger.log_run({"event": "broadcast_agent_error", "agent": agent_name, "error": str(result)})
                        processed_results[agent_name] = f"Error: {result}"
                    else:
                        self.logger.log_run({"event": "broadcast_agent_success", "agent": agent_name, "response_length": len(result)})
                        processed_results[agent_name] = result
                
                if not self.synthesizer_config:
                    # Simple concatenation if no synthesizer is configured
                    sections = [f"### {name} Response:\n{response}" for name, response in processed_results.items()]
                    final_response = "\n\n".join(sections)
                    if self.verbose: print(f"  Broadcast completed without synthesizer.")
                    self.logger.log_run({"event": "broadcast_concatenation"})
                else:
                    # Use a configured synthesizer to combine results
                    try:
                        if self.verbose: print(f"  Running broadcast synthesizer...")
                        
                        # Convert to JSON for template insertion
                        results_json = json.dumps(processed_results)
                        
                        # Extract config
                        synth_model = self.synthesizer_config.get("model", "openai/gpt-4o-mini")
                        synth_prompt_template = self.synthesizer_config.get("prompt", "Synthesize these agent responses: {agent_responses}")
                        synth_params = self.synthesizer_config.get("params", {})
                        
                        # Format the prompt
                        synth_prompt = synth_prompt_template.format(agent_responses=results_json)
                        
                        # Create a temporary PromptChain for synthesis
                        synthesizer = PromptChain(
                            models=[synth_model], 
                            instructions=[synth_prompt],
                            verbose=self.verbose,
                            **synth_params
                        )
                        
                        final_response = await synthesizer.process_prompt_async("")
                        if self.verbose: print(f"  Synthesis complete.")
                        self.logger.log_run({"event": "broadcast_synthesis_success"})
                    except Exception as e:
                        error_msg = f"Error in broadcast synthesizer: {e}"
                        logging.error(error_msg, exc_info=True)
                        self.logger.log_run({"event": "broadcast_synthesis_error", "error": str(e)})
                        # Fall back to concatenation
                        sections = [f"### {name} Response:\n{response}" for name, response in processed_results.items()]
                        sections.append(f"### Synthesis Error:\n{error_msg}")
                        final_response = "\n\n".join(sections)
        else:
            final_response = f"Unsupported execution mode: {self.execution_mode}"
            self.logger.log_run({"event": "error", "reason": "Unsupported mode", "mode": self.execution_mode})

        self._add_to_history("assistant", final_response)
        self.logger.log_run({
            "event": "process_input_end",
            "mode": self.execution_mode,
            "response_length": len(final_response)
        })
        return final_response

    async def run_agent_direct(self, agent_name: str, user_input: str, send_history: bool = False) -> str:
        """Runs a specific agent directly, bypassing the routing logic.

        Args:
            agent_name (str): The name of the agent to run.
            user_input (str): The user's message for the agent.
            send_history (bool): If True, prepends the formatted conversation history to the input.
        """
        if not user_input: return "Input cannot be empty."
        if agent_name not in self.agents:
            error_msg = f"Error: Agent '{agent_name}' not found. Available agents: {list(self.agents.keys())}"
            logging.error(error_msg)
            self.logger.log_run({"event": "run_direct_error", "reason": "Agent not found", "requested_agent": agent_name})
            raise ValueError(error_msg)

        start_time = asyncio.get_event_loop().time()
        log_event_start = {"event": "run_direct_start", "agent_name": agent_name, "input": user_input}
        if self.verbose: print(f"\n=== Running Agent Directly: {agent_name} ===\nInput: '{user_input}'")
        self.logger.log_run(log_event_start)
        self._add_to_history("user", f"(Direct to {agent_name}{' with history' if send_history else ''}) {user_input}")

        selected_agent_chain = self.agents[agent_name]
        query_for_agent = user_input

        # --- Prepend history if requested ---
        if send_history:
            if self.verbose: print(f"  Prepending conversation history for {agent_name}...")
            formatted_history = self._format_chat_history() # Use internal method with its truncation
            history_tokens = self._count_tokens(formatted_history)
            # Add a warning threshold (e.g., 6000 tokens)
            HISTORY_WARNING_THRESHOLD = 6000
            if history_tokens > HISTORY_WARNING_THRESHOLD:
                warning_msg = f"Warning: Sending long history ({history_tokens} tokens) to agent '{agent_name}'. This might be slow or exceed limits."
                print(f"\033[93m{warning_msg}\033[0m") # Yellow warning
                self.logger.log_run({"event": "run_direct_history_warning", "agent": agent_name, "history_tokens": history_tokens, "threshold": HISTORY_WARNING_THRESHOLD})

            query_for_agent = f"Conversation History:\n---\n{formatted_history}\n---\n\nUser Request:\n{user_input}"
            if self.verbose:
                print(f"  Input for {agent_name} with history (truncated): {query_for_agent[:200]}...")
            self.logger.log_run({"event": "run_direct_history_prepended", "agent": agent_name, "history_tokens": history_tokens})
        # --- End history prepending ---

        try:
            if self.verbose:
                print(f"--- Executing Agent: {agent_name} ---")
            self.logger.log_run({"event": "direct_agent_running", "agent_name": agent_name, "input_length": len(query_for_agent)})

            agent_response = await selected_agent_chain.process_prompt_async(query_for_agent)

            if self.verbose:
                print(f"Agent Response ({agent_name}): {agent_response}")
                print(f"--- Finished Agent: {agent_name} ---")
            self.logger.log_run({"event": "direct_agent_finished", "agent_name": agent_name, "response_length": len(agent_response)})

            self._add_to_history("assistant", agent_response)

            end_time = asyncio.get_event_loop().time()
            duration = int((end_time - start_time) * 1000)
            self.logger.log_run({"event": "run_direct_end", "agent_name": agent_name, "response_length": len(agent_response), "duration_ms": duration})
            return agent_response

        except Exception as e:
            error_msg = f"An error occurred during direct execution of agent '{agent_name}': {e}"
            logging.error(error_msg, exc_info=True)
            self._add_to_history("system_error", error_msg)
            self.logger.log_run({"event": "run_direct_error", "agent_name": agent_name, "error": str(e), "traceback": traceback.format_exc()})
            return f"I encountered an error running the {agent_name}: {e}"

    async def run_chat(self):
        """Runs an interactive chat loop."""
        # --- Generate a session log filename and set it in the logger ---
        if self.logger.log_dir:
            session_id = uuid.uuid4().hex[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_log_filename = os.path.join(self.logger.log_dir, f"router_chat_{timestamp}_{session_id}.jsonl")
            self.logger.set_session_filename(self.session_log_filename)
            print(f"[Log] Session log file: {self.session_log_filename}")
        print(f"Starting Agentic Chat (Mode: {self.execution_mode})...")
        print("Type 'exit' to quit.")
        print("Use '@agent_name: your message' to run an agent directly (e.g., '@math_agent: 5*5').")
        print("Use '@history @agent_name: your message' to run an agent directly *with* full history.")
        self.logger.log_run({"event": "chat_started", "mode": self.execution_mode})

        turn = 1
        # Create console with custom theme for better markdown rendering
        theme = Theme({
            "markdown.code": "bold cyan",
            "markdown.h1": "bold red",
            "markdown.h2": "bold yellow",
            "markdown.h3": "bold green",
            "markdown.h4": "bold blue",
            "markdown.link": "underline cyan",
            "markdown.item": "green"
        })
        console = Console(theme=theme, force_terminal=True)
        
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
                print("\ud83d\udcdb Exit command received.")
                self.logger.log_run({"event": "chat_ended", "reason": "Exit command"})
                break

            if not user_message_full.strip():
                continue

            response = ""
            try:
                # --- Check for Direct Agent Calls (with or without history) ---
                send_history_flag = False
                user_actual_message = user_message_full # Default if no flags
                target_agent = None

                # Pattern 1: @history @agent_name: message
                history_direct_match = re.match(r"@history\s+@(\w+):\s*(.*)", user_message_full, re.DOTALL | re.IGNORECASE)
                if history_direct_match:
                    send_history_flag = True
                    target_agent, user_actual_message = history_direct_match.groups()
                    if self.verbose: print(f"~~ History + Direct agent call detected for: {target_agent} ~~")
                    self.logger.log_run({"event": "direct_call_detected", "agent": target_agent, "with_history": True})

                # Pattern 2: @agent_name: message (only if Pattern 1 didn't match)
                elif not target_agent:
                    direct_match = re.match(r"@(\w+):\s*(.*)", user_message_full, re.DOTALL)
                    if direct_match:
                        send_history_flag = False # Explicitly false for this pattern
                        target_agent, user_actual_message = direct_match.groups()
                        if self.verbose: 
                            print(f"~~ Direct agent call detected for: {target_agent} (no history flag) ~~")
                        self.logger.log_run({"event": "direct_call_detected", "agent": target_agent, "with_history": False})

                # --- Execute Direct Call or Normal Processing ---
                if target_agent:
                    if target_agent in self.agents:
                        try:
                            response = await self.run_agent_direct(
                                target_agent,
                                user_actual_message.strip(),
                                send_history=send_history_flag
                            )
                        except ValueError as ve:
                            print(str(ve))
                            response = "Please try again with a valid agent name."
                        except Exception as direct_e:
                            logging.error(f"Error during direct agent execution: {direct_e}", exc_info=True)
                            response = f"Sorry, an error occurred while running {target_agent}: {direct_e}"
                            self.logger.log_run({"event": "run_direct_unexpected_error", "agent_name": target_agent, "error": str(direct_e), "traceback": traceback.format_exc()})
                    else:
                        print(f"Error: Agent '{target_agent}' not found. Available: {list(self.agents.keys())}")
                        response = f"Agent '{target_agent}' not recognized. Please choose from {list(self.agents.keys())} or use automatic routing."
                        self.logger.log_run({"event": "direct_call_agent_not_found", "requested_agent": target_agent})
                else:
                    response = await self.process_input(user_message_full)

            except Exception as e:
                logging.error(f"An unexpected error occurred in chat loop: {e}", exc_info=True)
                response = f"Sorry, an error occurred: {e}"
                self.logger.log_run({"event": "chat_loop_error", "error": str(e), "traceback": traceback.format_exc()})

            # Print the assistant's response using rich.Markdown
            console.print("\n[bold cyan]Assistant:[/bold cyan]")
            try:
                # Create a custom markdown class that properly handles code blocks
                class CustomMarkdown(Markdown):
                    def __init__(self, markup: str):
                        super().__init__(markup)
                        
                    def render_code_block(self, code: str, info_string: str = "") -> Syntax:
                        """Override code block rendering to add line numbers and proper syntax highlighting"""
                        lexer = info_string or "python"  # Default to Python if no language specified
                        return Syntax(
                            code,
                            lexer,
                            theme="monokai",
                            line_numbers=True,
                            indent_guides=True,
                            word_wrap=True
                        )
                
                # Render the response as markdown
                md = CustomMarkdown(response)
                console.print(md)
            except Exception as md_error:
                # Fallback to plain text if markdown rendering fails
                logging.warning(f"Failed to render response as markdown: {md_error}")
                console.print(response)
            
            turn += 1

        print("\n--- Chat Finished ---")


# Example usage (remains the same, using default router mode)
# ... (The example `run_example_agent_chat` function would be here in the original file) ...