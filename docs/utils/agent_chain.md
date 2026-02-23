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

## Per-Agent History Configuration (v0.4.2)

AgentChain now supports fine-grained control over conversation history on a per-agent basis. This powerful feature allows you to optimize token usage, reduce costs, and improve response times by configuring exactly what historical context each agent receives.

### Overview

**Why Per-Agent History Configuration?**

Different agents have different history requirements:
- **Stateless agents** (e.g., calculator, unit converter) don't need conversation history
- **Contextual agents** (e.g., research, writing) benefit from full conversation context
- **Specialized agents** (e.g., code analyzer) may only need specific types of history

Benefits:
- **Token Savings**: Avoid sending unnecessary history to agents that don't need it
- **Cost Reduction**: Lower API costs by reducing prompt token usage
- **Faster Responses**: Smaller prompts = faster processing
- **Better Focus**: Agents receive only relevant context for their task

### Configuration Options

Each agent can have its own history configuration with the following options:

```python
{
    "enabled": bool,                    # Enable/disable history for this agent
    "max_tokens": int,                  # Maximum tokens in history context
    "max_entries": int,                 # Maximum number of history entries
    "truncation_strategy": str,         # How to truncate ("oldest_first", "keep_last")
    "include_types": List[str],         # Only include these message types
    "exclude_sources": List[str]        # Exclude messages from these sources
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | `bool` | `True` | Whether to include history for this agent |
| `max_tokens` | `int` | AgentChain's `max_history_tokens` | Maximum tokens to include in history |
| `max_entries` | `int` | `None` (unlimited) | Maximum number of history entries |
| `truncation_strategy` | `str` | `"oldest_first"` | How to truncate when limits exceeded |
| `include_types` | `List[str]` | `None` (all types) | Filter to only these message types ("user", "assistant") |
| `exclude_sources` | `List[str]` | `None` (no exclusions) | Exclude messages from these agent sources |

### Complete Code Example

```python
from promptchain.utils.agent_chain import AgentChain
from promptchain import PromptChain

# Define specialized agents
terminal_agent = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Execute terminal command: {input}"]
)

research_agent = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Research topic: {input}"]
)

coding_agent = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Generate code: {input}"]
)

writing_agent = PromptChain(
    models=["anthropic/claude-3-sonnet-20240229"],
    instructions=["Write content: {input}"]
)

# Configure per-agent history settings
agent_history_configs = {
    # Terminal agent: No history needed (stateless operations)
    "terminal": {
        "enabled": False
    },

    # Research agent: Full history with generous limits
    "research": {
        "enabled": True,
        "max_tokens": 8000,              # Large context window
        "truncation_strategy": "keep_last",
        "include_types": ["user", "assistant"]  # Only conversation, no tool calls
    },

    # Coding agent: Recent history only
    "coding": {
        "enabled": True,
        "max_entries": 10,               # Last 10 messages only
        "max_tokens": 4000,
        "exclude_sources": ["terminal"]  # Don't include terminal outputs
    },

    # Writing agent: Full conversation context
    "writing": {
        "enabled": True,
        "max_tokens": 6000,
        "include_types": ["user", "assistant"]
    }
}

# Create AgentChain with per-agent history configuration
agent_chain = AgentChain(
    agents={
        "terminal": terminal_agent,
        "research": research_agent,
        "coding": coding_agent,
        "writing": writing_agent
    },
    agent_descriptions={
        "terminal": "Executes terminal commands",
        "research": "Researches topics and gathers information",
        "coding": "Generates and analyzes code",
        "writing": "Creates written content and documentation"
    },
    agent_history_configs=agent_history_configs,
    auto_include_history=False,  # Use per-agent configs instead of global setting
    execution_mode="router",
    router=router_config
)

# Usage
result = await agent_chain.process_input("Write a Python function to calculate Fibonacci")
# Coding agent receives its configured history (last 10 entries, max 4000 tokens)

result = await agent_chain.process_input("ls -la")
# Terminal agent receives NO history (enabled=False)
```

### Configuration Patterns

#### Pattern 1: Disable History for Stateless Agents

```python
agent_history_configs = {
    "calculator": {"enabled": False},
    "unit_converter": {"enabled": False},
    "timestamp": {"enabled": False}
}
```

**Use Case**: Mathematical, conversion, or utility agents that don't need context

**Token Savings**: 100% for these agents (no history sent)

#### Pattern 2: Token-Limited History for Cost Optimization

```python
agent_history_configs = {
    "standard_agent": {
        "enabled": True,
        "max_tokens": 2000  # Instead of default 4000
    },
    "budget_agent": {
        "enabled": True,
        "max_tokens": 1000  # Very constrained
    }
}
```

**Use Case**: Balance context with cost when using expensive models

**Token Savings**: 50-75% reduction in history tokens

#### Pattern 3: Type-Filtered History

```python
agent_history_configs = {
    "chat_agent": {
        "enabled": True,
        "include_types": ["user", "assistant"],  # Only conversation
        "max_tokens": 4000
    },
    "tool_analyzer": {
        "enabled": True,
        "include_types": ["tool_call", "tool_result"],  # Only tool execution
        "max_tokens": 3000
    }
}
```

**Use Case**: Agents that only need specific types of context

**Token Savings**: Varies by conversation structure (typically 30-50%)

#### Pattern 4: Source Exclusion

```python
agent_history_configs = {
    "summary_agent": {
        "enabled": True,
        "exclude_sources": ["debug_agent", "diagnostic_agent"],
        "max_tokens": 5000
    }
}
```

**Use Case**: Exclude noisy or irrelevant agents from context

**Token Savings**: Depends on excluded agent verbosity (10-40%)

#### Pattern 5: Entry-Limited Recent Context

```python
agent_history_configs = {
    "quick_response": {
        "enabled": True,
        "max_entries": 5,  # Last 5 messages only
        "truncation_strategy": "keep_last"
    }
}
```

**Use Case**: Fast agents that only need immediate context

**Token Savings**: 60-80% in long conversations

### Token Savings Examples

#### Example 1: Mixed Agent System

```python
# 100-turn conversation scenario
# Default: All agents get full 4000-token history

# Conversation stats:
# - 100 user messages (avg 50 tokens each) = 5,000 tokens
# - 100 assistant messages (avg 150 tokens each) = 15,000 tokens
# - Total history: 20,000 tokens (truncated to 4000 tokens per agent)

# Without per-agent config:
# - 10 calculator calls × 4000 tokens = 40,000 history tokens
# - 20 research calls × 4000 tokens = 80,000 history tokens
# - 30 coding calls × 4000 tokens = 120,000 history tokens
# Total: 240,000 history tokens sent

# With per-agent config:
agent_history_configs = {
    "calculator": {"enabled": False},           # 0 tokens
    "research": {"max_tokens": 8000},           # 20 × 8000 = 160,000 tokens
    "coding": {"max_entries": 10, "max_tokens": 3000}  # 30 × 3000 = 90,000 tokens
}

# New total: 250,000 history tokens
# Wait, this is MORE? Let's optimize:

agent_history_configs = {
    "calculator": {"enabled": False},           # 0 tokens (was 40,000)
    "research": {
        "max_tokens": 6000,
        "include_types": ["user", "assistant"]  # 20 × 4000 = 80,000 tokens (was 80,000)
    },
    "coding": {
        "max_entries": 8,
        "max_tokens": 2000                      # 30 × 2000 = 60,000 tokens (was 120,000)
    }
}

# Optimized total: 140,000 history tokens
# Savings: 100,000 tokens (42% reduction)
# At $5 per 1M tokens: $0.50 saved per 100-turn session
```

#### Example 2: Stateless vs Stateful Mix

```python
# Session with 50% stateless operations
agent_history_configs = {
    # Stateless agents (no history needed)
    "math": {"enabled": False},
    "convert": {"enabled": False},
    "format": {"enabled": False},
    "validate": {"enabled": False},

    # Stateful agents (need context)
    "research": {"max_tokens": 6000},
    "writer": {"max_tokens": 6000},
    "reviewer": {"max_tokens": 4000}
}

# 100 operations: 50 stateless, 50 stateful
# Without config: 100 × 4000 = 400,000 history tokens
# With config: (0 × 50) + (varying × 50) ≈ 260,000 history tokens
# Savings: 140,000 tokens (35% reduction)
```

### AgenticStepProcessor Isolation

**Important**: AgenticStepProcessor internal reasoning is automatically isolated from per-agent history configuration.

When using `AgenticStepProcessor` within an agent:
- The processor's internal tool calls and reasoning steps are NOT exposed to the agent's conversation history
- Only the final output from the AgenticStepProcessor is added to history
- This prevents history pollution from complex multi-step internal reasoning

```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

# Agent with internal agentic reasoning
research_chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        AgenticStepProcessor(
            objective="Research topic thoroughly",
            max_internal_steps=10  # These 10 steps won't appear in history
        ),
        "Synthesize findings: {input}"
    ]
)

agent_chain = AgentChain(
    agents={"research": research_chain},
    agent_descriptions={"research": "Deep research agent"},
    agent_history_configs={
        "research": {
            "enabled": True,
            "max_tokens": 4000,
            "include_types": ["user", "assistant"]  # Only sees user queries and final outputs
        }
    }
)

# When research agent runs:
# 1. AgenticStepProcessor executes 10 internal reasoning steps (NOT added to history)
# 2. Only the final synthesized output is added to conversation history
# 3. Next call to research agent sees clean history without internal reasoning clutter
```

**Why This Matters**:
- Prevents token waste from internal reasoning traces
- Keeps conversation history clean and focused
- Avoids confusing the agent with its own internal tool calls
- Maintains the abstraction boundary between internal processing and external conversation

### Best Practices

#### 1. Start with Defaults, Optimize Based on Need

```python
# Initially, let all agents use default history
agent_chain = AgentChain(
    agents=agents,
    agent_descriptions=descriptions,
    auto_include_history=True  # Use default for all
)

# After observing behavior, optimize specific agents
agent_chain = AgentChain(
    agents=agents,
    agent_descriptions=descriptions,
    agent_history_configs={
        # Only configure agents that need different settings
        "calculator": {"enabled": False},  # Clearly stateless
        "writer": {"max_tokens": 8000}     # Needs more context
    },
    auto_include_history=False
)
```

#### 2. Monitor Token Usage

```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

# Track history usage
history_mgr = ExecutionHistoryManager(max_tokens=10000)

# After each agent call
stats = history_mgr.get_statistics()
print(f"History tokens: {stats['total_tokens']}")
print(f"Utilization: {stats['utilization_pct']:.1f}%")

# Adjust per-agent configs based on actual usage
if stats['utilization_pct'] > 80:
    print("Consider reducing max_tokens for some agents")
```

#### 3. Use include_types for Focused Context

```python
# Conversational agents: Only need user/assistant messages
agent_history_configs = {
    "chat_agent": {
        "include_types": ["user", "assistant"]
    }
}

# Debugging agents: Need tool execution traces
agent_history_configs = {
    "debug_agent": {
        "include_types": ["tool_call", "tool_result", "error"]
    }
}
```

#### 4. Validate Configuration Keys

```python
# AgentChain validates agent names at initialization
try:
    agent_chain = AgentChain(
        agents={"math": math_agent, "research": research_agent},
        agent_descriptions={...},
        agent_history_configs={
            "math": {"enabled": False},
            "writing": {"max_tokens": 4000}  # ERROR: 'writing' not in agents
        }
    )
except ValueError as e:
    print(f"Configuration error: {e}")
    # ValueError: agent_history_configs contains invalid agent names: {'writing'}
```

#### 5. Consider Conversation Length

```python
# Short conversations: Default settings fine
# Long conversations (100+ turns): Optimize aggressively

agent_history_configs = {
    "frequent_agent": {
        "max_entries": 20,  # Prevent excessive history in long sessions
        "max_tokens": 3000
    }
}
```

#### 6. Document Your Configuration

```python
# Add comments explaining why each agent has specific settings
agent_history_configs = {
    # Calculator: Stateless mathematical operations, no context needed
    "calculator": {"enabled": False},

    # Research: Needs broad context to maintain topic coherence across queries
    "research": {"max_tokens": 8000, "include_types": ["user", "assistant"]},

    # Code reviewer: Only needs recent code-related history
    "code_reviewer": {
        "max_entries": 15,
        "exclude_sources": ["calculator", "weather"]  # Irrelevant to code review
    }
}
```

### Integration with ExecutionHistoryManager

Per-agent history configuration works seamlessly with `ExecutionHistoryManager`:

```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

# Create shared history manager
shared_history = ExecutionHistoryManager(
    max_tokens=10000,
    max_entries=200
)

# AgentChain uses its own internal history management
# But you can use ExecutionHistoryManager for custom tracking
agent_chain = AgentChain(
    agents=agents,
    agent_descriptions=descriptions,
    agent_history_configs=agent_history_configs
)

# Track all interactions in your own history manager
async def process_with_tracking(user_input: str):
    # Add to your history tracker
    shared_history.add_entry("user_input", user_input, source="user")

    # AgentChain handles per-agent history internally
    result = await agent_chain.process_input(user_input)

    # Add result to your tracker
    shared_history.add_entry("agent_output", result, source="agent_chain")

    return result

# Get your custom history view
my_history = shared_history.get_formatted_history(
    max_tokens=2000,
    format_style="chat"
)
```

For complete `ExecutionHistoryManager` documentation, see: [ExecutionHistoryManager.md](../ExecutionHistoryManager.md)

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