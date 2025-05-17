---
noteId: "fb2c313028e311f099a79bc8f395ca4b"
tags: []

---

# AgentChain: Orchestrating Multiple PromptChains

## Introduction

The `AgentChain` class provides a powerful way to manage and orchestrate multiple, specialized `PromptChain` instances (referred to as "agents"). It acts as a central controller, receiving user input and directing it to the appropriate agent(s) based on a configured execution flow. This allows for building sophisticated conversational systems where different agents handle specific tasks like math calculations, document retrieval, code generation, or creative writing.

## Initialization (`__init__`)

You initialize `AgentChain` by providing the agents it will manage, descriptions for those agents (used by some routing mechanisms), the desired execution mode, and other optional configurations.

```python
from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain
import asyncio

# --- Example Agent Setup (using placeholder PromptChains) ---
math_agent = PromptChain(models=["openai/gpt-4o-mini"], instructions=["Solve: {input}"])
creative_agent = PromptChain(models=["openai/gpt-4o-mini"], instructions=["Write about: {input}"])
rag_agent = PromptChain(models=["openai/gpt-4o-mini"], instructions=["Retrieve info: {input}"])

agent_dict = {
    "math_agent": math_agent,
    "creative_agent": creative_agent,
    "rag_agent": rag_agent
}

agent_descriptions_dict = {
    "math_agent": "Handles mathematical calculations and problem-solving.",
    "creative_agent": "Generates creative text like poems, stories, or ideas.",
    "rag_agent": "Retrieves information from a knowledge base based on the query."
}

# --- Example Router Config (for 'router' mode) ---
default_llm_router_config = {
    "models": ["openai/gpt-4o-mini"],
    "instructions": [None, "{input}"],
    "decision_prompt_template": "Analyze history: {history} and input: {user_input} considering agent details: {agent_details}. Choose the best agent. Respond JSON: {"chosen_agent": "<agent_name>"}"
}

# --- Example Synthesizer Config (for 'broadcast' mode) ---
synthesizer_config_example = {
   "model": "openai/gpt-4o-mini",
   "prompt": "Synthesize the best answer based on the original query '{user_input}' and the following responses from different agents:\n\n{agent_responses}\n\nProvide a single, coherent final answer."
}


# --- AgentChain Initialization Example ('router' mode) ---
agent_orchestrator = AgentChain(
    router=default_llm_router_config, # Required for 'router' mode
    agents=agent_dict,
    agent_descriptions=agent_descriptions_dict,
    execution_mode="router", # Specify the mode
    log_dir="logs/agent_chain_logs", # Optional: Enable file logging
    verbose=True, # Enable AgentChain's own print statements
    max_internal_steps=5 # Optional: Max re-routes for 'router' mode
    # synthesizer_config=synthesizer_config_example # Add this if execution_mode="broadcast"
)

```

### Parameters:

*   `router` (Union[Dict, Callable]): **Required only for `execution_mode="router"`**. Defines the routing logic.
    *   **Dict:** Configures the default 2-step LLM router. Requires keys:
        *   `models`: List containing one model configuration for the LLM decision step.
        *   `instructions`: Must be `[None, "{input}"]`.
        *   `decision_prompt_template`: A string template instructing the LLM how to choose an agent based on `{user_input}`, `{history}`, and `{agent_details}`. Must guide the LLM to output JSON like `{"chosen_agent": "agent_name"}`.
        *   Can include other `PromptChain` parameters (e.g., `temperature`) for the router chain.
    *   **Callable:** An `async def` function you provide: `async def custom_router(user_input: str, history: List[Dict[str, str]], agent_descriptions: Dict[str, str]) -> str`. It must return the JSON string `{"chosen_agent": "agent_name"}`.
*   `agents` (Dict[str, PromptChain]): A dictionary mapping unique agent names (strings) to their initialized `PromptChain` instances. **Required**.
*   `agent_descriptions` (Dict[str, str]): A dictionary mapping the same agent names to human-readable descriptions of their capabilities. Used by routing logic. **Required**.
*   `execution_mode` (str): Determines the operational flow. Options: `"router"`, `"pipeline"`, `"round_robin"`, `"broadcast"`. Defaults to `"router"`. **Required**.
*   `max_history_tokens` (int): Maximum tokens to use when formatting conversation history for the router. Defaults to `4000`.
*   `log_dir` (Optional[str]): Path to a directory for saving detailed run logs as `.jsonl` files. If `None` (default), only console logging occurs via the internal `RunLogger`.
*   `additional_prompt_dirs` (Optional[List[str]]): List of additional directories to search for prompt files if agents use file-based prompts.
*   `verbose` (bool): If `True`, `AgentChain` prints detailed status updates to the console (distinct from the `RunLogger`'s console output). Defaults to `False`.
*   `**kwargs`: Accepts additional optional parameters:
    *   `max_internal_steps` (int): **Used only in `"router"` mode**. Maximum number of times the router can re-route based on an agent's `[REROUTE]` response within a single `process_input` call. Prevents infinite loops. Defaults to `3`.
    *   `synthesizer_config` (Optional[Dict[str, Any]]): **Required only for `"broadcast"` mode**. A dictionary specifying how to synthesize results from parallel agent executions. Must contain:
        *   `"model"` (str): The model name for the synthesizer LLM call.
        *   `"prompt"` (str): A prompt template string. Must include `{agent_responses}` and can optionally include `{user_input}`.

## Execution Modes

The `execution_mode` parameter dictates how `AgentChain` handles user input.

### 1. `router` Mode (Default)

*   **Goal:** Select the single best agent to handle the user's request for that turn. Supports agent-to-agent communication within a turn.
*   **Flow:**
    1.  **Simple Router:** Checks for basic keyword/pattern matches (`_simple_router`, currently example only). If matched, select agent.
    2.  **Complex Router:** If no simple match, invokes the configured `router` (LLM chain or custom function) using the current input and history.
    3.  **Parse Decision:** Parses the router's output (`_parse_decision`) to get the `chosen_agent` name.
    4.  **Execute Agent:** Runs the chosen agent's `process_prompt_async`.
    5.  **Check Output Prefix:** Examines the agent's raw response:
        *   `[REROUTE] next_input`: If found, strips the prefix, uses `next_input` as the input for the *next* routing decision, and loops back to Step 1 (up to `max_internal_steps`). This allows one agent to trigger another via the router.
        *   `[FINAL] final_answer`: If found, strips the prefix, uses `final_answer` as the final result for the user, and stops processing for this turn.
        *   No Prefix: Treats the entire response as the final answer for the user and stops processing for this turn.
    6.  **Max Steps:** If the internal loop reaches `max_internal_steps` without a `[FINAL]` or non-prefix response, it stops and returns an informational message.
*   **Use Case:** Flexible request handling, complex workflows involving conditional agent calls initiated by other agents.
*   **Requires:** `router` configuration (Dict or Callable), `agents`, `agent_descriptions`.
*   **Optional:** `max_internal_steps`.
*   **Prompt Engineering:** Agents intended to use the re-route feature *must* be explicitly prompted to output the `[REROUTE]` or `[FINAL]` prefixes.

### 2. `pipeline` Mode

*   **Goal:** Process the user input through all defined agents sequentially within a single turn.
*   **Flow:**
    1.  Executes the first agent in the `agents` dictionary order with the initial user input string.
    2.  The *string output* (`agent_response`) of the first agent is captured.
    3.  Executes the second agent, passing the first agent's output string as its input.
    4.  This continues for all agents. Each agent receives only the final string output of the agent immediately preceding it.
    5.  The final string output of the *last* agent is returned to the user.
    6.  If any agent errors, the pipeline stops, and an error message is returned.
*   **Use Case:** Fixed workflows where input needs sequential refinement or processing by different specialized agents (e.g., Retrieve -> Format -> Synthesize), assuming each step only needs the direct output of the previous one.
*   **Requires:** `agents`. Agents used in pipeline mode **must** be designed to accept a single string input (`{input}`) and produce a single string output suitable for the next agent.
*   **History Management:** The overall conversation history managed by `AgentChain` is **not** automatically passed to agents in pipeline mode. If an agent requires historical context, it must be explicitly included in the string passed from a previous step, or the agent needs a more complex design (e.g., using `AgenticStepProcessor` with access to external history if implemented).

### 3. `round_robin` Mode

*   **Goal:** Distribute user requests evenly across agents over multiple turns.
*   **Flow:**
    1.  Maintains an internal index (`_round_robin_index`).
    2.  For the current turn, selects the agent at `index % num_agents`.
    3.  Executes the selected agent with the user input.
    4.  Returns the agent's response.
    5.  Increments the index for the *next* turn.
*   **Use Case:** Load balancing simple requests across identical agents, or ensuring different perspectives are applied over time.
*   **Requires:** `agents`.

### 4. `broadcast` Mode

*   **Goal:** Get perspectives/results from all agents simultaneously and combine them.
*   **Flow:**
    1.  Sends the user input to *all* agents concurrently using `asyncio.gather`.
    2.  Collects all responses (or errors).
    3.  **Synthesizer Step:** If a valid `synthesizer_config` is provided:
        *   Formats the collected agent responses into the `synthesizer_config["prompt"]` template (filling `{agent_responses}` and optionally `{user_input}`).
        *   Makes a separate LLM call using the `synthesizer_config["model"]` and the formatted prompt.
        *   Returns the synthesized response.
    4.  **Fallback:** If `synthesizer_config` is missing or invalid, returns a JSON string containing the raw responses from all agents.
*   **Use Case:** Gathering diverse information, comparing outputs from different agents, generating a consolidated answer from multiple sources.
*   **Requires:** `agents`, `synthesizer_config` (for synthesized output).

## Key Methods

*   **`process_input(self, user_input: str) -> str`**: The main entry point for processing user input according to the configured `execution_mode`. Handles all the logic described above.
*   **`run_agent_direct(self, agent_name: str, user_input: str) -> str`**: Bypasses all routing/mode logic and executes a specific agent by name directly with the given input. Useful for testing or specific commands.
*   **`run_chat(self)`**: Starts an interactive command-line chat loop. It uses `process_input` for standard messages and `run_agent_direct` if the user uses the `@agent_name: message` syntax.

## Logging

*   **Console Summaries:** The internal `RunLogger` always logs concise summaries of key events (initialization, routing decisions, agent execution, errors) to the console using the standard Python `logging` system.
*   **File Logs (Optional):** If `log_dir` is provided during initialization, `RunLogger` also appends detailed JSON information for each event to timestamped `.jsonl` files within that directory.
*   **Verbose Output:** Setting `verbose=True` enables additional `print` statements directly from `AgentChain` for step-by-step execution flow details, separate from the `RunLogger` output.

## MCP Integration

*   `AgentChain` itself does *not* directly manage MCP connections.
*   If any of the `PromptChain` instances passed in the `agents` dictionary are configured with `mcp_servers`, you **must** manually connect them *before* calling `agent_orchestrator.process_input` or `agent_orchestrator.run_chat`.
*   Iterate through your agent instances and call `await agent_instance.mcp_helper.connect_mcp_async()` for each one that uses MCP.
*   Similarly, ensure you call `await agent_instance.mcp_helper.close_mcp_async()` in a `finally` block for proper cleanup.

```python
# --- In your main execution script (like base_chat.py) ---

# After creating agent_orchestrator...

# Connect MCPs
logging.info("Connecting MCP servers...")
for agent_name, agent_instance in agent_orchestrator.agents.items():
    if hasattr(agent_instance, 'mcp_helper') and agent_instance.mcp_helper:
        try:
            await agent_instance.mcp_helper.connect_mcp_async()
            # ... logging ...
        except Exception as mcp_err:
            # ... error handling ...

# Run the chat
try:
    await agent_orchestrator.run_chat()
finally:
    # Close MCPs
    logging.info("Closing MCP connections...")
    for agent_name, agent_instance in agent_orchestrator.agents.items():
         if hasattr(agent_instance, 'mcp_helper') and agent_instance.mcp_helper:
             try:
                 await agent_instance.mcp_helper.close_mcp_async()
                 # ... logging ...
             except Exception as close_err:
                 # ... error handling ...
```

## Important Considerations

*   **State Management:** The `router` mode's internal loop handles state *within* a single turn. `round_robin` handles state *across* turns. `pipeline` and `broadcast` are generally stateless regarding agent selection for a given turn.
*   **Prompt Engineering:** The enhanced `router` mode's re-routing (`[REROUTE]`) and explicit final answer (`[FINAL]`) features depend entirely on the underlying agents being prompted correctly to output these specific prefixes when appropriate.
*   **Error Handling:** Consider how errors in different modes (e.g., an agent failing mid-pipeline, routing failing mid-loop, synthesizer failing) should affect the final output returned to the user. The current implementation provides basic error messages.
*   **Complexity:** Adding multiple execution modes increases the complexity. Ensure thorough testing for your chosen mode and agent interactions. 