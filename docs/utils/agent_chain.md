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
    verbose=True,                      # Optional
    cache_config={                     # Optional: SQLite caching configuration
        "name": "session_name",        # Session ID and database filename
        "path": "./cache",             # Directory to store database
        "include_all_instances": False # Whether to include all program runs
    }
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
-   `cache_config` (Optional): Configures SQLite-based conversation caching (see Conversation Caching section for details).

## Conversation Caching

AgentChain includes a robust SQLite-based conversation caching system that automatically persists all conversation history to a database file. This provides:

1. **Persistence**: Conversation history is preserved across application restarts
2. **Organization**: Sessions and instances are tracked separately
3. **Reliability**: SQLite handles special characters and code snippets properly
4. **Agent Tracking**: The responding agent's name is stored with each assistant message

### Caching Configuration

The caching system can be configured through the `cache_config` parameter:

```python
agent_orchestrator = AgentChain(
    # ... other parameters ...
    cache_config={
        "name": "customer_support",    # Session name (used as DB filename)
        "path": "./cache",             # Directory to store DB files
        "include_all_instances": False # Only include current instance by default
    }
)
```

### Session and Instance Tracking

The caching system uses a two-level organization approach:

1. **Session**: Identified by the `name` parameter in cache_config
   - Each session gets its own SQLite database file (`name.db`)
   - Use different sessions for different conversation topics
   
2. **Instance**: A unique UUID automatically generated each time the application runs
   - Each program run creates a new instance within the session
   - Allows tracking which messages belong to which execution
   - Query either just the current instance or all instances

### Agent Role Tracking

The system automatically stores which agent generated each response:

1. **Database Storage**: For assistant messages, the role is stored as `assistant:agent_name` 
2. **History Display**: When viewing history, agent names are shown in parentheses: `assistant (agent_name)`
3. **Direct Access**: In the web interface, use `@history:get` or `@history:get all` to view history with agent names

This feature makes it easy to track which specialized agent handled each request, particularly useful in applications with multiple agent types.

### Accessing Cached History

To retrieve cached history from the database:

```python
# Get history from current instance only
history = agent_orchestrator.get_cached_history()

# Get history from all instances in this session
history = agent_orchestrator.get_cached_history(include_all_instances=True)

# Access individual entries
for entry in history["conversation"]:
    # Check if this is an assistant message with agent name
    role = entry['role']
    if ":" in role and role.startswith("assistant:"):
        base_role, agent_name = role.split(":", 1)
        print(f"{base_role} ({agent_name}): {entry['content'][:50]}...")
    else:
        print(f"{role}: {entry['content'][:50]}...")

# Get metadata
print(f"Session ID: {history['session_id']}")
print(f"Current Instance: {history['current_instance_uuid']}")
print(f"Number of instances: {len(history['instances'])}")
```

### Default Configuration in Web Applications

In the Advanced Router Web Chat application, caching is enabled by default with these settings:

```python
DEFAULT_CACHE_ENABLED = True
DEFAULT_CACHE_CONFIG = {
    "name": "router_chat",             # Default session name 
    "path": os.path.join(os.getcwd(), ".cache"),  # Default directory
    "include_all_instances": False
}
```

### Command Line Control

The Advanced Router Web Chat application supports command-line options to control caching:

```bash
# Disable caching
python main.py --disable-cache

# Use a custom session name
python main.py --cache-name="customer_support" 

# Use a custom storage directory
python main.py --cache-path="/data/cache"

# Include all instances when retrieving history
python main.py --all-instances
```

### Special History Commands

In the web chat interface, you can use special commands to retrieve history:

```
@history:get         # Get current instance history
@history:get all     # Get all instances' history
```

### Resource Management

The SQLite connection is automatically closed when:
- The application shuts down properly
- The AgentChain object is garbage collected
- The `close_cache()` method is called explicitly

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