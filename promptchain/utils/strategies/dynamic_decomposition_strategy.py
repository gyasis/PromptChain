import asyncio
import logging
from typing import TYPE_CHECKING, Dict, Any, Optional

if TYPE_CHECKING:
    from promptchain.utils.agent_chain import AgentChain

logger = logging.getLogger(__name__)

async def execute_dynamic_decomposition_strategy_async(agent_chain_instance: 'AgentChain', initial_user_input: str) -> str:
    """Executes a task by iteratively calling the router to determine the next step."""
    if agent_chain_instance.verbose: print(f"\n--- Executing Router Strategy: dynamic_decomposition ---")

    current_task_state = "[INITIAL_REQUEST]"
    overall_user_request = initial_user_input
    final_response = ""
    
    for dynamic_step in range(agent_chain_instance.max_internal_steps):
        if agent_chain_instance.verbose: print(f"  Dynamic Decomposition Step {dynamic_step + 1}/{agent_chain_instance.max_internal_steps}")
        agent_chain_instance.logger.log_run({"event": "dynamic_decomp_step_start", "step": dynamic_step + 1, "last_output_len": len(current_task_state)})

        history_context = agent_chain_instance._format_chat_history()
        agent_details = "\n".join([f" - {name}: {desc}" for name, desc in agent_chain_instance.agent_descriptions.items()])
        template = agent_chain_instance._decision_prompt_templates.get("dynamic_decomposition")
        if not template:
            final_response = "Error: Dynamic decomposition prompt template not found."
            agent_chain_instance.logger.log_run({"event": "dynamic_decomp_error", "reason": "Template missing"})
            break
        
        try:
            router_llm_input = template.format(
                agent_details=agent_details,
                history=history_context,
                initial_user_request=overall_user_request,
                last_step_output=current_task_state
            )
        except KeyError as e:
            final_response = f"Error: Missing placeholder in dynamic decomposition template: {e}"
            agent_chain_instance.logger.log_run({"event": "dynamic_decomp_error", "reason": f"Template format error: {e}"})
            break
        
        decision_output: Optional[str] = None
        parsed_decision: Dict[str, Any] = {}
        try:
            if agent_chain_instance.custom_router_function:
                logging.warning("Custom router function may not work correctly with dynamic_decomposition strategy without updates.")
                final_response = "Error: Custom router not yet supported for dynamic_decomposition."
                break
            elif agent_chain_instance.decision_maker_chain:
                if len(agent_chain_instance.decision_maker_chain.instructions) == 2 and \
                   callable(agent_chain_instance.decision_maker_chain.instructions[1]):
                    final_response = "Error: Decision maker chain step 2 is unexpectedly a function."
                    break
                decision_output = await agent_chain_instance.decision_maker_chain.run_model_async(router_llm_input, step_index=1)
            else:
                raise RuntimeError("No LLM router configured for dynamic_decomposition strategy.")

            if agent_chain_instance.verbose: print(f"  Dynamic Router Decision Output: {decision_output}")
            parsed_decision = agent_chain_instance._parse_decision(decision_output)

            if not parsed_decision or "next_action" not in parsed_decision:
                error_msg = "Router failed to generate a valid next action."
                if agent_chain_instance.verbose: print(f"  Error: {error_msg} Router Output: {decision_output}")
                agent_chain_instance.logger.log_run({"event": "dynamic_decomp_invalid_action", "raw_output": decision_output, "parsed": parsed_decision})
                final_response = f"Error: Could not determine next action. Router output was: {decision_output or '[Empty]'}"
                break

        except Exception as e:
            error_msg = f"Error getting next action from dynamic router: {e}"
            logging.error(error_msg, exc_info=True)
            agent_chain_instance.logger.log_run({"event": "dynamic_decomp_router_error", "error": str(e)})
            final_response = f"Sorry, an error occurred while determining the next step: {e}"
            break

        next_action = parsed_decision["next_action"]
        is_task_complete = parsed_decision.get("is_task_complete", False)

        if next_action == "call_agent":
            agent_name = parsed_decision["agent_name"]
            agent_input = parsed_decision["agent_input"]
            if agent_chain_instance.verbose: print(f"  Dynamic Action: Calling agent '{agent_name}' with input: {agent_input[:100]}...")
            agent_chain_instance.logger.log_run({"event": "dynamic_decomp_agent_call", "step": dynamic_step + 1, "agent": agent_name, "input_len": len(agent_input)})
            
            if agent_name not in agent_chain_instance.agents:
                final_response = f"Error in dynamic plan: Agent '{agent_name}' not found."
                agent_chain_instance.logger.log_run({"event": "dynamic_decomp_step_error", "step": dynamic_step + 1, "agent": agent_name, "reason": "Agent not found"})
                break
            
            selected_agent = agent_chain_instance.agents[agent_name]
            try:
                agent_response_raw: str = ""
                agent_final_output: Optional[str] = None
                agent_loop_broken = False
                agent_input_for_this_run = agent_input
                
                for agent_internal_step in range(agent_chain_instance.max_internal_steps):
                    agent_response_raw = await selected_agent.process_prompt_async(agent_input_for_this_run)
                    reroute_prefix = "[REROUTE]"
                    final_prefix = "[FINAL]"
                    response_stripped = agent_response_raw.strip()

                    if response_stripped.startswith(reroute_prefix):
                        next_input_content = response_stripped[len(reroute_prefix):].strip()
                        agent_input_for_this_run = next_input_content
                        continue
                    elif response_stripped.startswith(final_prefix):
                        agent_final_output = response_stripped[len(final_prefix):].strip()
                        if agent_chain_instance.verbose: print(f"    Agent '{agent_name}' provided [FINAL] during dynamic step. Overriding router.")
                        agent_chain_instance.logger.log_run({"event": "dynamic_decomp_agent_final_override", "step": dynamic_step + 1, "agent": agent_name})
                        is_task_complete = True
                        agent_loop_broken = True
                        break
                    else:
                        agent_final_output = agent_response_raw
                        agent_loop_broken = True
                        break
                        
                if not agent_loop_broken:
                    agent_final_output = agent_response_raw
                
                if agent_final_output is None:
                    raise RuntimeError(f"Agent '{agent_name}' failed to produce output.")
                
                current_task_state = agent_final_output 
                agent_chain_instance.logger.log_run({"event": "dynamic_decomp_agent_success", "step": dynamic_step + 1, "agent": agent_name, "output_len": len(current_task_state)})
                
                if is_task_complete:
                    final_response = current_task_state
                    break
                    
            except Exception as e:
                error_msg = f"Error executing agent '{agent_name}' in dynamic step {dynamic_step + 1}: {e}"
                logging.error(error_msg, exc_info=True)
                agent_chain_instance.logger.log_run({"event": "dynamic_decomp_step_error", "step": dynamic_step + 1, "agent": agent_name, "error": str(e)})
                final_response = f"Dynamic execution failed at step {dynamic_step + 1} ({agent_name}): {e}"
                break
        
        elif next_action == "final_answer":
            final_response = parsed_decision["final_answer"]
            if agent_chain_instance.verbose: print(f"  Dynamic Action: Router declared task complete. Final Answer: {final_response[:100]}...")
            agent_chain_instance.logger.log_run({"event": "dynamic_decomp_final_answer", "step": dynamic_step + 1, "answer_len": len(final_response)})
            break
        
        else:
            final_response = f"Error: Unknown next_action '{next_action}' received from router."
            agent_chain_instance.logger.log_run({"event": "dynamic_decomp_error", "reason": "Unknown next_action", "action": next_action})
            break
             
        if is_task_complete and not final_response:
            if agent_chain_instance.verbose: print(f"  Dynamic Action: Router declared task complete, using last agent output as final.")
            final_response = current_task_state
            agent_chain_instance.logger.log_run({"event": "dynamic_decomp_complete_flag", "step": dynamic_step + 1})
            break

    if not final_response and dynamic_step == agent_chain_instance.max_internal_steps - 1:
        logging.warning(f"Router strategy 'dynamic_decomposition' reached max steps ({agent_chain_instance.max_internal_steps}).")
        agent_chain_instance.logger.log_run({"event": "dynamic_decomp_max_steps"})
        final_response = f"[Info] Reached max steps ({agent_chain_instance.max_internal_steps}) for dynamic task decomposition. Last state: {current_task_state[:150]}..."

    if not final_response:
        final_response = "[Dynamic decomposition finished unexpectedly without a final response.]"
        agent_chain_instance.logger.log_run({"event": "dynamic_decomp_empty_response"})
        
    agent_chain_instance.logger.log_run({"event": "dynamic_decomp_end", "final_response_length": len(final_response)})
    return final_response 