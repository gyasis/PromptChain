# CLI Tools Registry System

Extensible tool registration and discovery system for PromptChain CLI commands and agent interactions.

## Features

- **Decorator-based Registration**: Simple `@registry.register()` decorator for tools
- **Parameter Schema Validation**: JSON schema-based parameter validation with type checking
- **OpenAI Function Calling Format**: Compatible with LiteLLM and existing PromptChain infrastructure
- **Category-based Organization**: Organize tools by category (filesystem, shell, session, agent, etc.)
- **Tag-based Discovery**: Find tools using flexible tag queries
- **Type Safety**: Full type hints and runtime validation

## Quick Start

```python
from promptchain.cli.tools import registry, ToolCategory

# Register a filesystem tool
@registry.register(
    category=ToolCategory.FILESYSTEM,
    description="Read file contents",
    parameters={
        "path": {
            "type": "string",
            "required": True,
            "description": "File path to read"
        },
        "encoding": {
            "type": "string",
            "default": "utf-8",
            "description": "File encoding"
        }
    },
    tags=["io", "files"],
    examples=["fs_read(path='config.json')", "fs_read(path='data.txt', encoding='latin-1')"]
)
def fs_read(path: str, encoding: str = "utf-8") -> str:
    """Read file contents with specified encoding."""
    with open(path, encoding=encoding) as f:
        return f.read()


# Lookup and execute
tool = registry.get("fs_read")
result = tool(path="myfile.txt")  # Automatic parameter validation

# Or execute directly
result = registry.execute("fs_read", path="myfile.txt")
```

## Parameter Schema Types

The registry supports all JSON schema types:

### Basic Types

```python
@registry.register(
    category="utility",
    description="Example with basic types",
    parameters={
        "name": {"type": "string", "required": True, "description": "User name"},
        "age": {"type": "integer", "required": True, "description": "User age"},
        "score": {"type": "number", "description": "Score (float)"},
        "active": {"type": "boolean", "default": True, "description": "Active status"}
    }
)
def basic_example(name: str, age: int, score: float = 0.0, active: bool = True):
    return f"{name} ({age}): {score}"
```

### Enum Constraints

```python
@registry.register(
    category="utility",
    description="File operation with mode",
    parameters={
        "file": {"type": "string", "required": True, "description": "File path"},
        "mode": {
            "type": "string",
            "required": True,
            "description": "Operation mode",
            "enum": ["read", "write", "append"]
        }
    }
)
def file_operation(file: str, mode: str):
    return f"Operating on {file} in {mode} mode"
```

### Nested Objects

```python
from promptchain.cli.tools import ParameterSchema

@registry.register(
    category="utility",
    description="Configure server",
    parameters={
        "config": {
            "type": "object",
            "required": True,
            "description": "Server configuration",
            "properties": {
                "host": ParameterSchema(
                    name="host",
                    type="string",
                    description="Server host",
                    required=True
                ),
                "port": ParameterSchema(
                    name="port",
                    type="integer",
                    description="Server port",
                    required=True,
                    default=8080
                )
            }
        }
    }
)
def configure_server(config: dict):
    return f"Server: {config['host']}:{config['port']}"
```

### Arrays

```python
@registry.register(
    category="utility",
    description="Process list of items",
    parameters={
        "items": {
            "type": "array",
            "required": True,
            "description": "List of items to process",
            "items": ParameterSchema(
                name="item",
                type="string",
                description="Individual item"
            )
        }
    }
)
def process_items(items: list):
    return f"Processing {len(items)} items"
```

## Discovery and Lookup

### By Name

```python
tool = registry.get("fs_read")
if tool:
    result = tool(path="file.txt")
```

### By Category

```python
from promptchain.cli.tools import ToolCategory

# Get all filesystem tools
fs_tools = registry.get_by_category(ToolCategory.FILESYSTEM)

# Or use string
fs_tools = registry.get_by_category("filesystem")

for tool in fs_tools:
    print(f"{tool.name}: {tool.description}")
```

### By Tags

```python
# Match any tag
io_tools = registry.get_by_tags(["io", "network"], match_all=False)

# Match all tags
file_io_tools = registry.get_by_tags(["io", "files"], match_all=True)
```

### List All Tools

```python
# List all registered tool names
all_tools = registry.list_tools()

# List all active categories
categories = registry.list_categories()

# List all tags in use
tags = registry.list_tags()
```

## OpenAI Function Calling Integration

The registry generates schemas compatible with OpenAI function calling format:

```python
# Get schema for a single tool
tool = registry.get("fs_read")
schema = tool.to_openai_schema()
# Returns:
# {
#     "type": "function",
#     "function": {
#         "name": "fs_read",
#         "description": "Read file contents",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "path": {"type": "string", "description": "File path to read"},
#                 "encoding": {"type": "string", "default": "utf-8", "description": "File encoding"}
#             },
#             "required": ["path"]
#         }
#     }
# }

# Get all schemas for a category
fs_schemas = registry.get_openai_schemas(category="filesystem")

# Get all schemas
all_schemas = registry.get_openai_schemas()
```

## Integration with PromptChain

```python
from promptchain import PromptChain
from promptchain.cli.tools import registry

# Register tools
@registry.register(category="analysis", description="Analyze data")
def analyze_data(data: str) -> str:
    return f"Analysis of: {data}"

# Create chain with registered tools
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze the following: {input}"]
)

# Add tool schemas from registry
chain.add_tools(registry.get_openai_schemas(category="analysis"))

# Register tool functions
for tool in registry.get_by_category("analysis"):
    chain.register_tool_function(tool.function)

# Process with tool calling
result = chain.process_prompt("Analyze user behavior data")
```

## Tool Categories

Built-in categories:

- `FILESYSTEM`: File operations (read, write, list, etc.)
- `SHELL`: Shell command execution
- `SESSION`: Session management operations
- `AGENT`: Agent management and control
- `CONTEXT`: Context and memory operations
- `ANALYSIS`: Data analysis and processing
- `UTILITY`: General utility functions
- `CUSTOM`: Custom category for specialized tools

## Error Handling

```python
from promptchain.cli.tools import (
    ToolRegistrationError,
    ToolValidationError,
    ToolNotFoundError
)

try:
    # Registration errors
    @registry.register(category="utility", description="Duplicate")
    def duplicate_tool():
        pass

    @registry.register(category="utility", description="Duplicate")
    def duplicate_tool():  # Raises ToolRegistrationError
        pass

except ToolRegistrationError as e:
    print(f"Registration failed: {e}")

try:
    # Validation errors
    tool = registry.get("my_tool")
    tool(invalid_param="value")  # Raises ToolValidationError

except ToolValidationError as e:
    print(f"Validation failed: {e}")

try:
    # Not found errors
    registry.execute("nonexistent_tool")  # Raises ToolNotFoundError

except ToolNotFoundError as e:
    print(f"Tool not found: {e}")
```

## Advanced Usage

### Custom Parameter Schemas

```python
from promptchain.cli.tools import ParameterSchema

# Create reusable parameter schemas
file_path_param = ParameterSchema(
    name="path",
    type="string",
    description="File path",
    required=True
)

encoding_param = ParameterSchema(
    name="encoding",
    type="string",
    description="Text encoding",
    default="utf-8",
    enum=["utf-8", "ascii", "latin-1"]
)

@registry.register(
    category="filesystem",
    description="Read file",
    parameters={
        "path": file_path_param,
        "encoding": encoding_param
    }
)
def read_file(path: str, encoding: str = "utf-8"):
    pass
```

### Tool Metadata Export

```python
# Export registry metadata (excludes function objects)
metadata = registry.export_metadata()
# Returns:
# {
#     "tools": {
#         "tool_name": {
#             "category": "filesystem",
#             "description": "...",
#             "parameters": {...},
#             "tags": [...],
#             "examples": [...]
#         }
#     },
#     "statistics": {
#         "total_tools": 10,
#         "categories": {"filesystem": 3, "shell": 2, ...},
#         "total_tags": 5
#     }
# }

# Save to file
import json
with open("tools_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

### Tool Unregistration

```python
# Unregister a tool
success = registry.unregister("my_tool")

# Clear all tools
registry.clear()
```

## Best Practices

1. **Use Type Hints**: Always provide type hints in tool functions for clarity
2. **Descriptive Names**: Use clear, descriptive tool names (e.g., `fs_read` not `rd`)
3. **Required vs Optional**: Mark parameters as required only when truly necessary
4. **Provide Defaults**: Supply sensible default values for optional parameters
5. **Add Examples**: Include usage examples for complex tools
6. **Use Tags**: Tag tools for easier discovery and filtering
7. **Validate Early**: Let the registry validate parameters before execution
8. **Category Organization**: Use appropriate categories for logical grouping

## Testing

The registry includes comprehensive tests covering:

- Tool registration via decorator
- Parameter validation (types, required, enum, nested objects, arrays)
- Tool lookup (by name, category, tags)
- OpenAI function calling format
- Error handling
- Tool execution with validation

Run tests:

```bash
pytest tests/cli/tools/test_registry.py -v
```

## Architecture

### Components

- **ToolRegistry**: Central registry with decorator-based registration
- **ToolMetadata**: Complete tool specification (name, category, description, function, schema)
- **ParameterSchema**: Parameter definitions with type info and validation
- **ToolCategory**: Enum of standard tool categories

### Design Patterns

- **Decorator Pattern**: `@registry.register()` for clean registration
- **Validation Strategy**: Parameter validation follows JSON schema specification
- **Index Pattern**: Multiple indexes (category, tag) for efficient lookup
- **OpenAI Compatibility**: Schemas follow OpenAI function calling format

### Integration Points

1. **CLI Command Handler**: Tools registered for slash commands
2. **Agent Chains**: Tools available to agent conversations
3. **PromptChain Integration**: Tools exposed via OpenAI function calling
4. **MCP Tools**: Compatibility with existing MCP tool infrastructure
