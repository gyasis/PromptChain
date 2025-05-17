---
noteId: "4f61910032bc11f0a51a37b27c0392c3"
tags: []

---

# MCP Setup and Prompt Management

## MCP Server Configuration

### Critical Path Requirements
The MCP server configuration requires **absolute paths** for both the Python interpreter and the script. Relative paths currently do not work correctly.

```python
mcp_server_config_list = [{
    "id": "deeplake_server",
    "type": "stdio",
    "command": "/home/gyasis/Documents/code/hello-World/.venv/bin/python",  # Must be absolute path
    "args": ["/home/gyasis/Documents/code/hello-World/main_2.py"],         # Must be absolute path
    "env": {}
}]
```

### Common Issues
1. Using `sys.executable` or relative paths may cause errors
2. Environment paths must be explicitly specified
3. The Python interpreter path must point to the correct virtual environment

## Prompt Directory Structure

### Main Prompts Directory
Location: `/prompts`
Contains:
- Task-specific prompt templates (`.md` files)
- Strategy definitions (in `strategies/` subdirectory)
- System prompts and instructions

### Key Prompt Files
1. `chat_response_instruction.md`: Chat response formatting
2. `retrieve_context_instruction.md`: Context retrieval guidelines
3. `mini_lesson.md`: Lesson generation template

### Strategies Directory
Location: `/prompts/strategies`
Contains JSON files for different prompting strategies:
```
strategies/
├── cod.json      # Chain of Density
├── cot.json      # Chain of Thought
├── ltm.json      # Long-term Memory
├── reflexion.json
├── self-consistent.json
├── self-refine.json
├── standard.json
└── tot.json      # Tree of Thoughts
```

## Token Management

The system implements an 8000 token limit for conversation history, managed through:
1. Complete chain interaction tracking (Q&A pairs)
2. Automatic truncation of oldest interactions
3. Token counting using tiktoken with cl100k_base encoding

```python
# Token management configuration
self.max_history_tokens = 8000  # Total token limit for history
self.encoding = tiktoken.get_encoding("cl100k_base")
```

## Project-Specific Prompts
Location: `/project_prompts`
- Contains project-specific implementations
- Has frontend/backend structure
- Includes custom server implementations

## Best Practices

1. **Path Management**:
   ```python
   WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
   PROMPTS_DIR = os.path.join(WORKSPACE_ROOT, "prompts")
   ```

2. **MCP Configuration**:
   - Always use absolute paths
   - Verify Python interpreter exists
   - Verify target script exists
   - Check environment variables

3. **Prompt Organization**:
   - Keep base prompts in `/prompts`
   - Project-specific prompts in `/project_prompts`
   - Strategy definitions in `/prompts/strategies`

4. **Token Management**:
   - Monitor token usage with logging
   - Truncate by complete interactions
   - Maintain conversation coherence

## Known Limitations

1. **Path Handling**:
   - Relative paths don't work in MCP configuration
   - Must use absolute paths for Python interpreter and script
   - Environment paths must be explicitly set

2. **Prompt Loading**:
   - Standard prompt directories must exist
   - Strategy files must be valid JSON
   - Prompt files must use `.md` extension

3. **Token Management**:
   - Fixed 8000 token limit for history
   - Truncation happens at interaction boundaries
   - No partial interaction preservation 
noteId: "089ea8c019c611f090dc6dad05957594"
tags: []

---

# Router Decision Colored Logging Example

## Purpose
To visually distinguish in the terminal when the router selects a single agent versus a multi-agent plan, use colored logging in your chat orchestration script (e.g., router_chat.py).

## Implementation
Add ANSI color codes and a logging function:

```python
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[92m"
COLOR_CYAN = "\033[96m"
COLOR_MAGENTA = "\033[95m"

def log_router_decision(plan, reasoning=None):
    if isinstance(plan, list) and len(plan) == 1:
        print(f"{COLOR_GREEN}Router selected single agent: {plan[0]}{COLOR_RESET}")
    elif isinstance(plan, list) and len(plan) > 1:
        plan_str = " → ".join(plan)
        print(f"{COLOR_CYAN}Router created a multi-agent plan: {plan_str}{COLOR_RESET}")
    else:
        print(f"{COLOR_MAGENTA}Router output unrecognized plan: {plan}{COLOR_RESET}")
    if reasoning:
        print(f"{COLOR_MAGENTA}Router reasoning: {reasoning}{COLOR_RESET}")
```

## Usage Example
After parsing the router's output (plan and reasoning), call the function before executing the plan:

```python
plan = router_output.get("plan")
reasoning = router_output.get("reasoning")
log_router_decision(plan, reasoning)
```

## Minimal Test Example

```python
examples = [
    {"plan": ["rag_retrieval_agent"], "reasoning": "Only retrieval needed."},
    {"plan": ["rag_retrieval_agent", "library_verification_agent", "engineer_agent"], "reasoning": "Multi-step workflow required."},
    {"plan": [], "reasoning": "No valid agent found."}
]
for example in examples:
    log_router_decision(example["plan"], example["reasoning"])
```

## Notes
- Place the logging call immediately after parsing the router's output and before executing the agent(s).
- This helps with debugging and understanding router behavior in multi-agent chat orchestration. 

---

## Logging and Debugging Conventions for MCP Tool Use (2024-05)

- Use color-coded, step-by-step logs for each agentic step that interacts with MCP tools.
- For each tool call, log:
  - Tool/function name
  - Arguments (as JSON or string)
  - Tool result (truncated if long)
- When appending a tool result message to the LLM history, always include the correct `tool_call_id` (from the tool call object) as required by the OpenAI API.
- This logging style is now the recommended practice for all PromptChain MCP integrations, ensuring transparency, easier debugging, and API compliance. 