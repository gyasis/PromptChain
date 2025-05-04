# AgentChain Orchestrator

The `AgentChain` class, located in `promptchain/utils/agent_chain.py`, provides a framework for orchestrating multiple specialized `PromptChain` instances (referred to as "agents"). It uses a configurable routing mechanism to direct user input to the most appropriate agent based on simple rules or more complex logic (LLM or custom function).

## Core Concepts

-   **Orchestration:** Manages a collection of named `PromptChain` agents.
-   **Routing:** Determines which agent should handle a given user input.
    -   **Simple Router:** Performs initial checks using basic patterns (e.g., regex for math).
    -   **Complex Router:** If the simple router doesn't find a match, a configurable complex router is invoked. This can be:
        -   The default 2-step LLM decision chain (Prepare Prompt -> Execute LLM).
        -   A custom asynchronous Python function provided by the user.
    -   **Direct Execution:** Allows bypassing the router entirely and calling a specific agent by name.
-   **History Management:** Maintains a conversation history and uses it (truncated by token limit) as context for the complex router.
-   **Agent Encapsulation:** `AgentChain` focuses on routing. Individual agents (`PromptChain` instances) are responsible for their own internal logic, including model usage, prompt execution, and tool/MCP integration. Agents must be fully configured *before* being passed to `AgentChain`.
-   **Logging:** Uses `RunLogger` to create detailed JSON logs for each session in the specified `log_dir`.

## Initialization (`__init__`)

```python
from promptchain.utils import AgentChain, PromptChain
# Assuming agent instances (math_agent, etc.) and descriptions are defined

agent_orchestrator = AgentChain(
    router=router_config_or_function, # Required: Dict or async Callable
    agents=all_agents_dict,            # Required: Dict[str, PromptChain]
    agent_descriptions=descriptions_dict, # Required: Dict[str, str]
    max_history_tokens=4000,           # Optional
    log_dir="logs",                    # Optional
    additional_prompt_dirs=None,       # Optional (for default router)
    verbose=True                       # Optional
)
```

**Key Parameters:**

-   `router` **(Required)**: Defines the complex routing logic used when the simple router defers.
    -   **Option 1: Dictionary (Default LLM Router)**
        ```python
        default_llm_router_config = {
            "models": ["gpt-4o-mini"], # Single model for the execution step
            "instructions": [None, "{input}"], # [Func placeholder, Exec template]
            "decision_prompt_template": "..." # Full prompt template string (see example)
            # Optional: other PromptChain args like 'temperature'
        }
        router = default_llm_router_config
        ```
        -   `models`: A list containing **one** model configuration for the final LLM execution step.
        -   `instructions`: Must be `[None, "{input}"]`. `None` is replaced by the internal prompt preparation function, and `"{input}"` tells the final step to execute the prepared prompt.
        -   `decision_prompt_template`: A **required** string template used by the internal preparation function. It **must** accept `{user_input}`, `{history}`, and `{agent_details}` placeholders via `.format()`. This template should guide the LLM to analyze the context and output **only** a JSON object like `{"chosen_agent": "agent_name"}`.
    -   **Option 2: Async Callable (Custom Router)**
        ```python
        async def my_custom_router(user_input: str, history: List[Dict[str, str]], agent_descriptions: Dict[str, str]) -> str:
            # ... your logic ...
            chosen_agent = "some_agent_name"
            return json.dumps({"chosen_agent": chosen_agent})

        router = my_custom_router
        ```
        -   Must be an `async def` function.
        -   Must accept `user_input` (str), `history` (List[Dict]), and `agent_descriptions` (Dict).
        -   Must return a **string** containing valid JSON like `{"chosen_agent": "agent_name"}`. It can optionally include `"refined_query": "..."`.
-   `agents` **(Required)**: A dictionary mapping agent names (strings) to their pre-configured `PromptChain` instances.
-   `agent_descriptions` **(Required)**: A dictionary mapping agent names to descriptive strings. These descriptions are passed to the router (default or custom) to aid in selection.
-   `max_history_tokens` (Optional): Max tokens for history context passed to the router.
-   `log_dir` (Optional): Directory for saving JSON run logs.
-   `additional_prompt_dirs` (Optional): Used by the default LLM router if its templates reference prompts loadable by ID/filename.
-   `verbose` (Optional): Enables detailed console printing during execution.

## Core Methods

-   `async process_input(user_input: str) -> str`:
    -   The main method for handling user input via automatic routing.
    -   Executes the flow: Simple Router -> Complex Router (if needed) -> Parse Decision -> Execute Chosen Agent.
    -   Manages history updates and logging.
-   `async run_agent_direct(agent_name: str, user_input: str) -> str`:
    -   Bypasses all routing logic.
    -   Directly executes the specified `agent_name` with the given `user_input`.
    -   Raises `ValueError` if the `agent_name` is not found.
    -   Manages history updates and logging for the direct execution.
-   `async run_chat()`:
    -   Starts an interactive command-line chat loop.
    -   Uses `process_input` for standard messages.
    -   Parses input starting with `@agent_name:` to use `run_agent_direct`.

## Routing Logic (`process_input` Flow)

1.  **Simple Router (`_simple_router`)**: Checks input against basic patterns (currently configured for math). If a match is found and the corresponding agent exists, returns the agent name. Otherwise, returns `None`.
2.  **Complex Router Invocation**: If Simple Router returns `None`:
    -   If `self.custom_router_function` is set, it's `await`ed with context (`user_input`, `history`, `agent_descriptions`).
    -   If `self.decision_maker_chain` is set (default LLM router), it's `await`ed via `process_prompt_async(user_input)`. This executes the internal 2-step chain (prepare prompt -> execute LLM).
3.  **Parse Decision (`_parse_decision`)**: Takes the output string from the Complex Router.
    -   **Requires** valid JSON containing a `"chosen_agent"` key whose value is a known agent name.
    -   Strips common markdown formatting (` ```json ... ``` `).
    -   Returns `(chosen_agent_name, refined_query)` on success, `(None, None)` on failure. **No string fallback.**
4.  **Validation**: Checks if a valid `chosen_agent_name` was returned by the routing process. If not, returns an error message.
5.  **Agent Execution**: Retrieves the chosen agent's `PromptChain` instance. Executes it using `process_prompt_async`, passing either the `refined_query` (if provided by the router's JSON) or the original `user_input`.
6.  **History & Return**: Updates history with the agent's response and returns it.

## Agent Configuration Notes

-   `AgentChain` **does not** configure the individual `PromptChain` agents passed to it.
-   If an agent needs specific tools (local Python functions) or needs to connect to MCP servers, this setup (e.g., `add_tools`, `register_tool_function`, `mcp_servers` parameter, calling `connect_mcp_async`) must be done on the `PromptChain` instance *before* it is passed in the `agents` dictionary to `AgentChain`.
-   `AgentChain` simply selects which pre-configured agent to run based on the router's decision.

## Example Snippet (from `run_example_agent_chat`)

```python
# --- Define Agents (Pre-configured) ---
math_agent = PromptChain(...)
math_agent.add_tools(...)
math_agent.register_tool_function(...)
doc_agent = PromptChain(..., mcp_servers=...)
creative_agent = PromptChain(...)

agent_descriptions = {
    "math_agent": "Solves math.",
    "doc_agent": "Answers docs.",
    "creative_agent": "Writes creative."
}
all_agents = {
    "math_agent": math_agent,
    "doc_agent": doc_agent,
    "creative_agent": creative_agent
}

# --- Choose and Configure Router ---

# Option 1: Default LLM Router Config
full_decision_template = """... (prompt guiding LLM to output JSON) ..."""
default_llm_router_config = {
    "models": ["gpt-4o-mini"],
    "instructions": [None, "{input}"],
    "decision_prompt_template": full_decision_template
}
router_to_use = default_llm_router_config

# Option 2: Custom Router Function
# async def my_custom_router(...): -> str: ...
# router_to_use = my_custom_router

# --- Initialize AgentChain ---
agent_orchestrator = AgentChain(
    router=router_to_use,
    agents=all_agents,
    agent_descriptions=agent_descriptions,
    verbose=True
)

# --- Run ---
# await agent_orchestrator.run_chat()
# or
# result = await agent_orchestrator.process_input("some query")
# or
# result = await agent_orchestrator.run_agent_direct("math_agent", "5 * 10")
``` 