---
noteId: "4ef7f9c032bc11f0a51a37b27c0392c3"
tags: []

---

# AgenticStepProcessor: Guide to Agentic Functionality in PromptChain

## Overview

The `AgenticStepProcessor` is a powerful component in the PromptChain framework that enables agentic behavior within a single chain step. Unlike regular chain steps that perform a single operation (like sending a prompt to an LLM), the `AgenticStepProcessor` runs an internal loop to achieve a defined objective, potentially making multiple LLM calls and tool invocations in the process.

## Key Features

- **Internal Agentic Loop**: Executes multiple LLM calls and tool invocations within a single chain step.
- **Tool Integration**: Seamlessly uses tools (both local and MCP) registered on the parent `PromptChain`.
- **Self-contained Model Configuration**: Can (and should) specify its own `model_name` and `model_params`.
- **Objective-driven Processing**: Works to achieve a specific defined objective.

## Usage Examples

### Basic Example (Agentic Step Only)

This shows a chain where the *only* step is the agentic one.

```python
from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

# Define a calculator tool schema
calculator_schema = {
    "type": "function",
    "function": {
        "name": "simple_calculator",
        "description": "Evaluates a simple mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The expression to evaluate."}
            },
            "required": ["expression"]
        }
    }
}

def simple_calculator(expression: str) -> str:
    """Evaluates a mathematical expression safely"""
    try:
        # Create a safe evaluation environment with limited scope
        allowed_names = {"abs": abs, "pow": pow, "round": round}
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Create the PromptChain
# NOTE: models=[] because NO OTHER steps need a model.
chain = PromptChain(
    models=[],
    instructions=[
        AgenticStepProcessor(
            objective="You are a math assistant. Use the calculator tool when needed.",
            max_internal_steps=3,
            model_name="openai/gpt-4o",  # MUST set model here for the agentic step
            model_params={"tool_choice": "auto"}
        )
    ],
    verbose=True
)

# Register the LOCAL tool on the chain
chain.add_tools([calculator_schema])
chain.register_tool_function(simple_calculator)

# Run the chain
result = chain.process_prompt("What is the square root of 144?")
print(result)
```

### Multi-step Example with Mixed Instructions

This shows how an agentic step fits into a larger chain.

```python
from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

# Create a chain with mixed instruction types
chain = PromptChain(
    # Model needed for the *non-agentic* steps (Step 1 and 3)
    models=["openai/gpt-3.5-turbo", "openai/gpt-3.5-turbo"],
    instructions=[
        "Analyze the following request: {input}", # Step 1 (uses model 1)
        AgenticStepProcessor(                   # Step 2 (uses its own model)
            objective="You are a research assistant. Extract key information.",
            max_internal_steps=5,
            model_name="openai/gpt-4o",      # Explicitly set model for this step
            model_params={"temperature": 0.2}
        ),
        "Summarize the findings based on: {input}" # Step 3 (uses model 2)
    ],
    verbose=True
)

# Run the chain
result = chain.process_prompt("Tell me about quantum computing advancements in 2023.")
print(result)
```

### Example with MCP Tools

The agentic step automatically uses MCP tools if they are discovered on the parent chain *before* the step runs. See `experiments/test_context7_mcp_single_step.py` for a full example.

```python
# --- Setup from test_context7_mcp_single_step.py ---
mcp_server_config = [ ... ] # Define MCP server
agentic_objective = "... Use Context7 tools ... {input}"

# Chain needs a model for the SECOND step (non-agentic)
chain = PromptChain(
    models=["openai/gpt-4o"],
    instructions=[
        AgenticStepProcessor(
            objective=agentic_objective,
            max_internal_steps=5,
            model_name="openai/gpt-4o", # Agentic step specifies its own model
            model_params={"tool_choice": "auto", "temperature": 0.2},
        ),
        "Based on the following... {input} ... generate a tutorial script." # This step uses the model from `models` list
    ],
    mcp_servers=mcp_server_config,
    verbose=True
)

# --- Connect MCP tools BEFORE running the chain ---
await chain.mcp_helper.connect_mcp_async()
print("Discovered tools:", [t['function']['name'] for t in chain.tools])

# --- Run the chain ---
# AgenticStepProcessor will now have access to discovered MCP tools
final_output = await chain.process_prompt_async(initial_prompt)
print(final_output)
```

## Implementation Details

The `AgenticStepProcessor` operates through several key mechanisms:

1. **Internal Loop**: Runs a loop for up to `max_internal_steps` iterations.
2. **Tool Integration**: Uses the tools available in the parent `PromptChain`'s `tools` property *at the time the step executes*. This includes local tools and any discovered (prefixed) MCP tools.
3. **LLM Interaction**: Makes sequential calls to its configured LLM (`model_name`, `model_params`), incorporating tool results into the internal message history.
4. **Callbacks**: Relies on callbacks (`llm_runner`, `tool_executor`) passed by `PromptChain` during execution to interact with the LLM and execute tools via the parent chain's mechanisms (including routing to `MCPHelper`).

## Best Practices

1. **Model Configuration**:
   - **Always** specify `model_name` and `model_params` directly on the `AgenticStepProcessor` instance.
   - The `models` list in the main `PromptChain` initialization should *only* contain models for the *other*, non-agentic steps in the chain. If the *entire* chain consists of only `AgenticStepProcessor`(s), then `models=[]` is appropriate for the main `PromptChain`.

2. **Tool Integration**:
   - Register local tools (schemas and functions) on the parent `PromptChain` using `add_tools()` and `register_tool_function()`.
   - For MCP tools, configure `mcp_servers` on the `PromptChain` and ensure `await chain.mcp_helper.connect_mcp_async()` is called *before* running `process_prompt_async`.
   - The `AgenticStepProcessor` automatically gains access to all tools present in `chain.tools` when its step is executed.

3. **Objective Definition**:
   - Write clear, specific objectives for the agent.
   - Include instructions on how to use tools (using their registered/prefixed names) and how to determine when the objective is complete.

4. **Error Handling**:
   - Set a reasonable `max_internal_steps` to prevent infinite loops.
   - Ensure tool functions (local or MCP servers) handle errors gracefully.

## Common Issues and Solutions

1. **Issue**: `AttributeError: 'PromptChain' object has no attribute 'tool_executor_callback'`
   **Solution**: This is an internal callback. Do not attempt to access or override it directly. Ensure the `AgenticStepProcessor` is correctly integrated into the `instructions` list.

2. **Issue**: `AgenticStepProcessor` doesn't seem to use the right model or tools.
   **Solution**: Verify `model_name` is set *on the processor*. Ensure tools (local or MCP) are registered/discovered on the parent `PromptChain` *before* running the chain.

3. **Issue**: `ValueError: Number of models must match number of non-function/non-agentic instructions`
   **Solution**: Remember that `AgenticStepProcessor` does *not* count towards the models needed in the main `PromptChain` `models` list. Only provide models in the main list for regular string instruction steps. Specify the model for the agentic step using its `model_name` parameter.

4. **Issue**: MCP tools not found by `AgenticStepProcessor`.
   **Solution**: Make sure `await chain.mcp_helper.connect_mcp_async()` was called *after* `PromptChain` initialization but *before* `chain.process_prompt_async()`.

## Advanced Usage

### Custom Tool Execution

For specialized tool handling, you can implement custom tool execution logic:

```python
# Define a custom tool executor
async def custom_tool_executor(tool_call):
    function_name = get_function_name_from_tool_call(tool_call)
    # Custom logic for special handling of specific tools
    if function_name == "special_tool":
        # Custom processing
        return "Custom result"
    
    # Fall back to default execution for other tools
    return await default_executor(tool_call)

# Advanced integration with custom executor would require modifying the
# internal implementation of AgenticStepProcessor
```

## Performance Considerations

- Each internal step involves an LLM call, which has latency implications
- Tool calls add additional processing time
- Consider the cost implications of using more expensive models for agentic steps

## Future Development

The `AgenticStepProcessor` will continue to evolve with:

- Enhanced tool calling capabilities
- Improved error handling and recovery
- Memory and context management optimizations
- Multi-agent collaboration capabilities

## Conclusion

The `AgenticStepProcessor` enables sophisticated agentic behaviors. By configuring its model correctly and ensuring tools are available on the parent chain (connecting MCP beforehand if needed), you can create powerful, tool-using steps within a larger `PromptChain` workflow.

## Logging and Debugging Best Practices (2024-05)

- Use color-coded, step-by-step logs for each internal agentic step.
- Log the following for each step:
  - Messages sent to the LLM (role, content, or [No content])
  - List of available tools
  - LLM response (truncated if long)
  - For each tool call: function name, arguments, and result (truncated)
- When appending a tool result message to the LLM history, always include the correct `tool_call_id` (from the tool call object) as required by the OpenAI API.
- This logging style is now recommended for all agentic PromptChain applications to aid in debugging, transparency, and compliance with LLM API requirements.

noteId: "fad4da20287a11f099a79bc8f395ca4b"
tags: []

---

 