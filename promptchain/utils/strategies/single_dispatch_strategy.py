import asyncio
import logging
import traceback
from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from promptchain.utils.agent_chain import AgentChain

logger = logging.getLogger(__name__)

async def execute_single_dispatch_strategy_async(agent_chain_instance: 'AgentChain', user_input: str, strategy_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Execute the single agent dispatch router strategy.
    
    This strategy takes a user query and dispatches it to a single agent,
    potentially with a refined version of the query.
    
    1. Try simple router first
    2. If no match, use complex router (LLM or custom)
    3. Execute the chosen agent with the input (refined if available)
    
    Args:
        agent_chain_instance: The AgentChain instance
        user_input: The user's input message
        strategy_data: Additional data for the strategy including:
          - include_history: Whether to include conversation history

    Returns:
        The response from the chosen agent
    """
    if agent_chain_instance.verbose: print(f"\n--- Executing Router Strategy: single_agent_dispatch ---")
    current_turn_input = user_input
    final_response = ""
    loop_broken = False
    agent_for_this_turn: Optional[str] = None
    
    # Get include_history setting from strategy_data or default to auto_include_history
    include_history = False
    if strategy_data and "include_history" in strategy_data:
        include_history = strategy_data["include_history"]
    else:
        include_history = agent_chain_instance.auto_include_history
    
    if agent_chain_instance.verbose and include_history:
        print(f"  Including conversation history in agent calls (auto={agent_chain_instance.auto_include_history})")
    
    # This loop handles the agent's internal [REROUTE] requests
    for internal_step in range(agent_chain_instance.max_internal_steps):
        if agent_chain_instance.verbose: print(f"  Router Internal Step {internal_step + 1}/{agent_chain_instance.max_internal_steps} (Agent Reroute Check)")
        agent_chain_instance.logger.log_run({"event": "router_internal_step_start", "strategy": "single_agent_dispatch", "step": internal_step + 1, "input_length": len(current_turn_input)})

        chosen_agent_name: Optional[str] = None
        refined_query: Optional[str] = None
        decision_output: Optional[str] = None

        # --- Routing Decision --- 
        # Only make a routing decision on the first internal step for this strategy
        if internal_step == 0:
            # Simple Router Check
            # Access simple_router via agent_chain_instance
            chosen_agent_name = agent_chain_instance._simple_router(current_turn_input)
            if chosen_agent_name:
                if agent_chain_instance.verbose: print(f"  Simple router selected: {chosen_agent_name}")
                agent_chain_instance.logger.log_run({"event": "router_decision", "strategy": "single_agent_dispatch", "source": "simple", "agent": chosen_agent_name, "internal_step": internal_step + 1})
            else:
                # Complex Router (LLM or Custom Function)
                if agent_chain_instance.verbose: print(f"  Simple router deferred. Invoking complex router with input: {current_turn_input[:100]}...")
                try:
                    if agent_chain_instance.custom_router_function:
                        decision_output = await agent_chain_instance.custom_router_function(
                            user_input=current_turn_input,
                            history=agent_chain_instance._conversation_history, # Access via instance
                            agent_descriptions=agent_chain_instance.agent_descriptions # Access via instance
                        )
                    elif agent_chain_instance.decision_maker_chain:
                        # Pass the current input for this turn/reroute
                        decision_output = await agent_chain_instance.decision_maker_chain.process_prompt_async(current_turn_input)
                    else:
                        raise RuntimeError("No valid router (LLM or custom) configured for router mode.")

                    if agent_chain_instance.verbose: print(f"  Complex router output: {decision_output}")
                    agent_chain_instance.logger.log_run({"event": "complex_router_output", "strategy": "single_agent_dispatch", "output": decision_output, "internal_step": internal_step + 1})
                    
                    # Parse the decision 
                    # Note: _parse_decision now takes the decision_output string directly
                    parsed_decision_dict = agent_chain_instance._parse_decision(decision_output)
                    chosen_agent_name = parsed_decision_dict.get("chosen_agent")
                    refined_query = parsed_decision_dict.get("refined_query")


                    if chosen_agent_name:
                        if agent_chain_instance.verbose: print(f"  Complex router selected: {chosen_agent_name}", f" (Refined query: {refined_query})" if refined_query else "")
                        agent_chain_instance.logger.log_run({"event": "router_decision", "strategy": "single_agent_dispatch", "source": "complex", "agent": chosen_agent_name, "refined_query_present": bool(refined_query), "internal_step": internal_step + 1})
                    else:
                        if agent_chain_instance.verbose: print("  Complex router failed to select a valid agent.")
                        agent_chain_instance.logger.log_run({"event": "router_decision_failed", "strategy": "single_agent_dispatch", "source": "complex", "raw_output": decision_output, "internal_step": internal_step + 1})
                        final_response = "Error: Router failed to select an agent."
                        loop_broken = True 
                        break

                except Exception as e:
                    error_msg = f"Error during complex routing (Step {internal_step + 1}): {e}"
                    logging.error(error_msg, exc_info=True)
                    agent_chain_instance.logger.log_run({"event": "router_error", "strategy": "single_agent_dispatch", "error": str(e), "internal_step": internal_step + 1})
                    final_response = f"Sorry, an error occurred during agent selection: {e}"
                    loop_broken = True 
                    break
            
            if loop_broken:
                break
            
            if internal_step == 0:
                agent_for_this_turn = chosen_agent_name
            
            if not agent_for_this_turn:
                final_response = final_response or "Sorry, I could not determine the appropriate agent to handle your request."
                agent_chain_instance.logger.log_run({"event": "agent_selection_failed", "strategy": "single_agent_dispatch", "internal_step": internal_step + 1})
                loop_broken = True
                break

            selected_agent = agent_chain_instance.agents.get(agent_for_this_turn)
            if not selected_agent: # Should not happen if agent_for_this_turn is valid
                final_response = f"Error: Agent '{agent_for_this_turn}' configured but not found in agents list."
                agent_chain_instance.logger.log_run({"event": "agent_execution_error", "reason": "Agent instance not found", "agent": agent_for_this_turn})
                loop_broken = True
                break

            query_for_agent = refined_query if internal_step == 0 and refined_query else current_turn_input
            
            # Add history to query if configured
            if include_history:
                formatted_history = agent_chain_instance._format_chat_history()
                query_for_agent = f"Conversation History:\n---\n{formatted_history}\n---\n\nCurrent Input:\n{query_for_agent}"
                if agent_chain_instance.verbose:
                    print(f"  Including history for agent {agent_for_this_turn} ({len(formatted_history)} chars)")
                agent_chain_instance.logger.log_run({"event": "include_history", "agent": agent_for_this_turn, "history_length": len(formatted_history)})
            
            if agent_chain_instance.verbose: print(f"  Executing agent: {agent_for_this_turn} with input: {query_for_agent[:100]}...")
            agent_chain_instance.logger.log_run({
                "event": "agent_execution_start", 
                "strategy": "single_agent_dispatch", 
                "agent": agent_for_this_turn, 
                "input_length": len(query_for_agent), 
                "internal_step": internal_step + 1,
                "history_included": include_history
            })
            try:
                agent_response_raw = await selected_agent.process_prompt_async(query_for_agent)
                agent_chain_instance.logger.log_run({"event": "agent_execution_success", "strategy": "single_agent_dispatch", "agent": agent_for_this_turn, "response_length": len(agent_response_raw), "internal_step": internal_step + 1})

                reroute_prefix = "[REROUTE]"
                final_prefix = "[FINAL]"

                response_stripped = agent_response_raw.strip()
                if response_stripped.startswith(reroute_prefix):
                    next_input_content = response_stripped[len(reroute_prefix):].strip()
                    if agent_chain_instance.verbose: print(f"  Agent {agent_for_this_turn} requested reroute. Next input: {next_input_content[:100]}...")
                    agent_chain_instance.logger.log_run({"event": "agent_reroute_requested", "strategy": "single_agent_dispatch", "agent": agent_for_this_turn, "next_input_length": len(next_input_content)})
                    current_turn_input = next_input_content 
                elif response_stripped.startswith(final_prefix):
                    final_response = response_stripped[len(final_prefix):].strip()
                    if agent_chain_instance.verbose: print(f"  Agent {agent_for_this_turn} provided final answer (via [FINAL] prefix).")
                    agent_chain_instance.logger.log_run({"event": "agent_final_response_prefix", "strategy": "single_agent_dispatch", "agent": agent_for_this_turn})
                    loop_broken = True
                    break 
                else:
                    final_response = agent_response_raw
                    if agent_chain_instance.verbose: print(f"  Agent {agent_for_this_turn} provided final answer (no prefix).")
                    agent_chain_instance.logger.log_run({"event": "agent_final_response_no_prefix", "strategy": "single_agent_dispatch", "agent": agent_for_this_turn})
                    loop_broken = True
                    break

            except Exception as e:
                error_msg = f"Error executing agent {agent_for_this_turn} (Internal Step {internal_step + 1}): {e}"
                logging.error(error_msg, exc_info=True)
                agent_chain_instance.logger.log_run({"event": "agent_execution_error", "strategy": "single_agent_dispatch", "agent": agent_for_this_turn, "error": str(e), "internal_step": internal_step + 1})
                final_response = f"Sorry, agent {agent_for_this_turn} encountered an error: {e}"
                loop_broken = True
                break 
                
    if not loop_broken and internal_step == agent_chain_instance.max_internal_steps - 1:
        logging.warning(f"Router strategy 'single_agent_dispatch' reached max internal steps ({agent_chain_instance.max_internal_steps}) without a final answer (likely during agent reroute).")
        agent_chain_instance.logger.log_run({"event": "router_max_steps_reached", "strategy": "single_agent_dispatch"})
        final_response = f"[Info] Reached max internal steps ({agent_chain_instance.max_internal_steps}) during agent processing. Processing stopped. Last internal input was: {current_turn_input[:150]}..."
            
    return final_response 