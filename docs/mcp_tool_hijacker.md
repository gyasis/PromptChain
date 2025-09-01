# MCP Tool Hijacker Documentation

## Overview

The MCP Tool Hijacker is a powerful feature in PromptChain that enables direct execution of Model Context Protocol (MCP) tools without the overhead of LLM agent processing. This capability significantly improves performance for tool-heavy workflows while maintaining compatibility with the existing PromptChain ecosystem.

## Key Benefits

- **Sub-100ms Tool Execution**: Bypass LLM processing for direct tool calls
- **Static Parameter Management**: Configure default parameters once, use everywhere
- **Parameter Transformation**: Automatic parameter validation and transformation
- **Batch Processing**: Execute multiple tools concurrently with rate limiting
- **Performance Monitoring**: Built-in tracking of execution times and success rates
- **Modular Design**: Non-breaking integration with existing PromptChain functionality

## Architecture

The MCP Tool Hijacker consists of four main components:

### 1. MCPConnectionManager
Handles MCP server connections, tool discovery, and session management.

### 2. ToolParameterManager
Manages static and dynamic parameters with transformation and validation capabilities.

### 3. MCPSchemaValidator
Provides JSON Schema validation for tool parameters ensuring type safety.

### 4. MCPToolHijacker
Main class that orchestrates direct tool execution with all supporting features.

## Installation

The MCP Tool Hijacker is included with PromptChain. Ensure you have the MCP dependencies installed:

```bash
pip install promptchain[mcp]
```

## Basic Usage

### 1. Simple Direct Execution

```python
from promptchain.utils.mcp_tool_hijacker import MCPToolHijacker

# Configure MCP servers
mcp_config = [
    {
        "id": "gemini_server",
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "@google/gemini-mcp@latest"]
    }
]

# Create and connect hijacker
hijacker = MCPToolHijacker(mcp_config, verbose=True)
await hijacker.connect()

# Direct tool execution (no LLM overhead)
result = await hijacker.call_tool(
    "mcp_gemini_server_ask_gemini",
    prompt="What is quantum computing?",
    temperature=0.5
)
print(result)

# Disconnect when done
await hijacker.disconnect()
```

### 2. Using with PromptChain

```python
from promptchain import PromptChain

# Create PromptChain with hijacker enabled
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze: {input}"],
    mcp_servers=mcp_config,
    enable_mcp_hijacker=True,
    hijacker_config={
        "connection_timeout": 45.0,
        "max_retries": 3,
        "parameter_validation": True
    }
)

async with chain:
    # Use traditional chain processing
    result = await chain.process_prompt_async("AI ethics")
    
    # Use hijacker for direct execution
    if chain.mcp_hijacker:
        direct_result = await chain.mcp_hijacker.call_tool(
            "mcp_gemini_server_ask_gemini",
            prompt="Quick fact about AI"
        )
```

## Advanced Features

### Step-to-Step Parameter Chaining (NEW!)

**The Missing Pieces for True Interchangeability**

The hijacker now supports dynamic parameter passing between PromptChain steps, enabling:

1. **Variable substitution from previous step outputs**
2. **JSON parsing and key extraction from MCP responses**

#### Dynamic Parameter Templates

Use outputs from previous steps as inputs for current steps:

```python
# Step 1: Initial search
result1 = await hijacker.call_tool(
    "mcp__deeplake__retrieve_context", 
    query="neural network optimization"
)

# Store step output for chaining
hijacker.store_step_output(1, result1, "search_step")

# Step 2: Use previous result's first document ID
hijacker.param_manager.set_parameter_template(
    "mcp__deeplake__get_document",
    "document_id",
    "{previous.results[0].id}"  # Extract ID from step 1 output
)

# Execute step 2 with automatic template resolution
result2 = await hijacker.call_tool_with_chaining(
    "mcp__deeplake__get_document",
    current_step=2,
    title="{previous_first_title}"  # Another template variable
)
```

#### Template Variable Patterns

**Previous Step Reference:**
```python
"{previous.results[0].id}"          # First result's ID from previous step
"{previous.metadata.title}"         # Title from previous step metadata
"{previous_first_text}"             # Shortcut for first result's text
```

**Numbered Step Reference:**
```python
"{step_1.results[0].metadata.title}"  # Specific step by number
"{step_2.summary}"                     # Output from step 2
```

**Named Step Reference:**
```python
"{search_step.results[0].id}"       # Reference by step name
"{analysis_step.conclusions}"       # Use meaningful names
```

**With Default Values:**
```python
"{previous.missing_field|default}"  # Provide fallback value
"{previous.results|[]}"             # Default to empty array
```

#### JSON Output Parsing

Extract specific values from MCP tool JSON responses:

```python
from promptchain.utils.json_output_parser import JSONOutputParser, CommonExtractions

parser = JSONOutputParser()

# Single value extraction
document_id = parser.extract(mcp_output, "results[0].id")
title = parser.extract(mcp_output, "results[0].metadata.title")

# Multiple extractions at once
extractions = {
    "first_id": "results[0].id",
    "first_title": "results[0].metadata.title", 
    "result_count": "results"
}
extracted = parser.extract_multiple(mcp_output, extractions)

# Common DeepLake patterns
all_ids = CommonExtractions.deeplake_document_ids(mcp_output)
all_titles = CommonExtractions.deeplake_titles(mcp_output)

# Create template variables automatically
template_vars = CommonExtractions.create_template_vars(mcp_output, "previous")
```

#### Complete Workflow Example

```python
async def chained_workflow():
    hijacker = MCPToolHijacker(mcp_config)
    await hijacker.connect()
    
    # Step 1: Search
    search_result = await hijacker.call_tool(
        "mcp__deeplake__retrieve_context",
        query="machine learning optimization"
    )
    hijacker.store_step_output(1, search_result, "search")
    
    # Step 2: Get details using Step 1's first result
    hijacker.param_manager.set_parameter_template(
        "mcp__deeplake__get_document", 
        "document_id",
        "{previous.results[0].id}"
    )
    
    details = await hijacker.call_tool_with_chaining(
        "mcp__deeplake__get_document",
        current_step=2
    )
    hijacker.store_step_output(2, details, "details")
    
    # Step 3: Summary using both previous steps
    summary = await hijacker.call_tool_with_chaining(
        "mcp__deeplake__get_summary",
        current_step=3,
        query="{search.query} {details.title}"  # Combine data
    )
    
    return summary
```

### Static Parameter Management

Configure parameters once and reuse them across multiple calls:

```python
# Set static parameters for a tool
hijacker.set_static_params(
    "mcp_gemini_server_ask_gemini",
    temperature=0.7,
    model="gemini-2.0-flash-001"
)

# Now only provide the varying parameters
questions = ["What is ML?", "Explain AI", "Define NLP"]
for question in questions:
    result = await hijacker.call_tool(
        "mcp_gemini_server_ask_gemini",
        prompt=question  # Only need prompt, other params are static
    )
```

### Parameter Transformation

Add automatic parameter transformation and validation:

```python
from promptchain.utils.tool_parameter_manager import CommonTransformers, CommonValidators

# Add temperature clamping (0.0 - 1.0)
hijacker.add_param_transformer(
    "mcp_gemini_server_ask_gemini",
    "temperature",
    CommonTransformers.clamp_float(0.0, 1.0)
)

# Add prompt length validation
hijacker.add_param_validator(
    "mcp_gemini_server_ask_gemini",
    "prompt",
    CommonValidators.is_string_max_length(1000)
)

# Temperature will be automatically clamped
result = await hijacker.call_tool(
    "mcp_gemini_server_ask_gemini",
    prompt="Test",
    temperature=1.5  # Automatically clamped to 1.0
)
```

### Batch Processing

Execute multiple tool calls concurrently:

```python
batch_calls = [
    {
        "tool_name": "mcp_gemini_server_ask_gemini",
        "params": {"prompt": "What is Python?", "temperature": 0.5}
    },
    {
        "tool_name": "mcp_gemini_server_ask_gemini",
        "params": {"prompt": "What is JavaScript?", "temperature": 0.5}
    },
    {
        "tool_name": "mcp_gemini_server_brainstorm",
        "params": {"topic": "Web development"}
    }
]

# Execute with max 2 concurrent calls
results = await hijacker.call_tool_batch(batch_calls, max_concurrent=2)

for result in results:
    if result["success"]:
        print(f"Tool: {result['tool_name']} - Success")
    else:
        print(f"Tool: {result['tool_name']} - Error: {result['error']}")
```

### Execution Hooks

Add custom logic before and after tool execution:

```python
def pre_execution_hook(tool_name, params):
    print(f"About to execute {tool_name}")
    # Could modify params, log, validate, etc.

def post_execution_hook(tool_name, params, result, execution_time):
    print(f"Executed {tool_name} in {execution_time:.3f}s")
    # Could log metrics, cache results, etc.

hijacker.add_execution_hook(pre_execution_hook, stage="pre")
hijacker.add_execution_hook(post_execution_hook, stage="post")
```

### Performance Monitoring

Track tool execution performance:

```python
# Execute some tools
for i in range(10):
    await hijacker.call_tool("mcp_tool", param=f"test{i}")

# Get performance statistics
stats = hijacker.get_performance_stats("mcp_tool")
print(f"Total calls: {stats['call_count']}")
print(f"Average time: {stats['avg_time']:.3f}s")
print(f"Success rate: {stats['success_rate']:.2%}")

# Get overall statistics
overall = hijacker.get_performance_stats()
print(f"Total tools: {overall['overall']['total_tools']}")
print(f"Total calls: {overall['overall']['total_calls']}")
```

## Parameter Management

### Priority Order

Parameters are merged in the following priority (highest to lowest):
1. **Dynamic parameters** - Provided at call time
2. **Static parameters** - Set via `set_static_params()`
3. **Default parameters** - Tool-specific defaults

### Global Transformers and Validators

Apply transformations/validations to all tools:

```python
# Global temperature clamping for all tools
hijacker.add_global_transformer(
    "temperature",
    CommonTransformers.clamp_float(0.0, 2.0)
)

# Global prompt validation for all tools
hijacker.add_global_validator(
    "prompt",
    CommonValidators.is_non_empty_string()
)
```

### Parameter Templates

Use templates for dynamic parameter substitution:

```python
hijacker.param_manager.set_parameter_template(
    "mcp_tool",
    "message",
    "User {username} says: {content}"
)

# Apply template with variables
result = await hijacker.call_tool(
    "mcp_tool",
    template_vars={"username": "Alice", "content": "Hello"},
    message="placeholder"  # Will be replaced by template
)
```

## Production Configuration

### Using Production Hijacker

```python
from promptchain.utils.mcp_tool_hijacker import create_production_hijacker

# Creates hijacker with production-ready settings
hijacker = create_production_hijacker(mcp_config, verbose=False)

# Includes:
# - 60s connection timeout
# - 5 retry attempts
# - Parameter validation enabled
# - Common transformers and validators
```

### Error Handling

```python
try:
    result = await hijacker.call_tool("mcp_tool", **params)
except ToolNotFoundError as e:
    print(f"Tool not available: {e}")
except ParameterValidationError as e:
    print(f"Invalid parameters: {e}")
except ToolExecutionError as e:
    print(f"Execution failed: {e}")
```

### Connection Management

Use context managers for automatic connection handling:

```python
async with MCPToolHijacker(mcp_config) as hijacker:
    # Connection established automatically
    result = await hijacker.call_tool("mcp_tool", param="value")
    # Connection closed automatically on exit
```

## Performance Comparison

### Traditional MCP (through LLM)
- Latency: 500-2000ms (includes LLM processing)
- Token usage: Variable based on prompt/response
- Cost: LLM API costs per call

### MCP Tool Hijacker (direct)
- Latency: 20-100ms (direct tool execution)
- Token usage: Zero (no LLM involved)
- Cost: Only MCP server costs (if any)

## Common Use Cases

### 1. High-Frequency Tool Calls
When making many repetitive tool calls where LLM reasoning isn't needed.

### 2. Latency-Sensitive Operations
Real-time applications requiring sub-100ms response times.

### 3. Batch Processing
Processing large datasets with the same tool configuration.

### 4. Testing and Development
Direct tool testing without the complexity of LLM prompting.

### 5. Cost Optimization
Reducing LLM token usage for simple tool operations.

## Troubleshooting

### Connection Issues
```python
# Increase timeout for slow connections
hijacker = MCPToolHijacker(
    mcp_config,
    connection_timeout=60.0,  # 60 seconds
    max_retries=5
)
```

### Tool Discovery
```python
# List available tools
tools = hijacker.get_available_tools()
print(f"Available tools: {tools}")

# Get tool schema
schema = hijacker.get_tool_schema("mcp_tool_name")
print(f"Tool schema: {schema}")
```

### Performance Issues
```python
# Enable verbose logging
hijacker = MCPToolHijacker(mcp_config, verbose=True)

# Check connection status
status = hijacker.get_status()
print(f"Connection status: {status}")

# Clear performance stats
hijacker.clear_performance_stats()
```

## API Reference

### MCPToolHijacker

#### Constructor
```python
MCPToolHijacker(
    mcp_servers_config: List[Dict[str, Any]],
    verbose: bool = False,
    connection_timeout: float = 30.0,
    max_retries: int = 3,
    parameter_validation: bool = True
)
```

#### Methods

- `async connect()` - Establish MCP connections
- `async disconnect()` - Close MCP connections
- `async call_tool(tool_name, **kwargs)` - Execute tool directly
- `async call_tool_batch(batch_calls, max_concurrent)` - Batch execution
- `set_static_params(tool_name, **params)` - Set static parameters
- `add_param_transformer(tool_name, param_name, transformer)` - Add transformer
- `add_param_validator(tool_name, param_name, validator)` - Add validator
- `get_performance_stats(tool_name=None)` - Get performance statistics
- `get_available_tools()` - List available tools
- `get_tool_schema(tool_name)` - Get tool schema

#### Step Chaining Methods (NEW!)

- `store_step_output(step_index, output, step_name=None)` - Store step output for chaining
- `create_template_vars_for_current_step(current_step)` - Create template variables from previous steps
- `async call_tool_with_chaining(tool_name, current_step, **params)` - Execute tool with automatic step chaining
- `parse_output_for_chaining(output, parse_config=None)` - Parse tool output for easier chaining
- `get_step_reference_info()` - Get available step references for debugging
- `clear_step_outputs()` - Clear all stored step outputs

### Common Transformers

- `CommonTransformers.clamp_float(min, max)` - Clamp float values
- `CommonTransformers.clamp_int(min, max)` - Clamp integer values
- `CommonTransformers.to_string()` - Convert to string
- `CommonTransformers.to_lowercase()` - Convert to lowercase
- `CommonTransformers.truncate_string(max_length)` - Truncate strings

### Common Validators

- `CommonValidators.is_float_in_range(min, max)` - Validate float range
- `CommonValidators.is_int_in_range(min, max)` - Validate integer range
- `CommonValidators.is_non_empty_string()` - Validate non-empty string
- `CommonValidators.is_string_max_length(max)` - Validate string length
- `CommonValidators.is_in_choices(choices)` - Validate against choices

## Best Practices

1. **Always use context managers** for automatic connection handling
2. **Set static parameters** for frequently used configurations
3. **Add validators** for critical parameters to catch errors early
4. **Monitor performance** to identify bottlenecks
5. **Use batch processing** for multiple similar operations
6. **Handle errors gracefully** with specific exception catching
7. **Clear performance stats** periodically in long-running applications

## Migration Guide

### From Traditional MCP to Hijacker

Before (traditional MCP through LLM):
```python
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Use the tool to: {input}"],
    mcp_servers=mcp_config
)
result = await chain.process_prompt_async("get weather for NYC")
```

After (direct execution with hijacker):
```python
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze: {input}"],
    mcp_servers=mcp_config,
    enable_mcp_hijacker=True
)

# Direct tool call (no LLM)
result = await chain.mcp_hijacker.call_tool(
    "mcp_weather_tool",
    location="NYC"
)
```

## Contributing

The MCP Tool Hijacker is part of the PromptChain project. Contributions are welcome! Please ensure:

1. All tests pass (`pytest tests/test_mcp_tool_hijacker.py`)
2. Code follows existing style conventions
3. Documentation is updated for new features
4. Performance impact is considered

## License

Same as PromptChain project license.