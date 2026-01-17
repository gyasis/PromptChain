"""
Comprehensive tests for Tool Registry System

Tests cover:
- Tool registration via decorator
- Parameter schema validation
- Tool lookup (by name, category, tags)
- OpenAI function calling format compatibility
- Error handling and edge cases
"""

import pytest
from typing import Any

from promptchain.cli.tools import (
    ToolRegistry,
    ToolMetadata,
    ToolCategory,
    ParameterSchema,
    ToolRegistrationError,
    ToolValidationError,
    ToolNotFoundError
)


# Test fixtures
@pytest.fixture
def registry():
    """Create fresh registry for each test."""
    return ToolRegistry()


@pytest.fixture
def sample_tool_function():
    """Sample tool function for testing."""
    def fs_read(path: str, encoding: str = "utf-8") -> str:
        """Read file contents."""
        return f"Reading {path} with {encoding}"
    return fs_read


# Test 1: Basic tool registration via decorator
def test_tool_registration_basic(registry, sample_tool_function):
    """Test basic tool registration using decorator."""

    @registry.register(
        category="filesystem",
        description="Read file contents",
        parameters={
            "path": {"type": "string", "required": True, "description": "File path"},
            "encoding": {"type": "string", "default": "utf-8", "description": "Encoding"}
        }
    )
    def fs_read(path: str, encoding: str = "utf-8") -> str:
        return f"Reading {path} with {encoding}"

    # Verify registration
    assert "fs_read" in registry.list_tools()
    tool = registry.get("fs_read")
    assert tool is not None
    assert tool.name == "fs_read"
    assert tool.category == ToolCategory.FILESYSTEM
    assert tool.description == "Read file contents"
    assert len(tool.parameters) == 2


# Test 2: Tool registration with custom name
def test_tool_registration_custom_name(registry):
    """Test tool registration with custom name override."""

    @registry.register(
        category="utility",
        description="Custom named tool",
        name="custom_tool_name"
    )
    def original_function_name():
        return "test"

    assert "custom_tool_name" in registry.list_tools()
    assert "original_function_name" not in registry.list_tools()
    tool = registry.get("custom_tool_name")
    assert tool.name == "custom_tool_name"


# Test 3: Duplicate registration error
def test_duplicate_registration_error(registry):
    """Test that duplicate tool registration raises error."""

    @registry.register(category="utility", description="First tool")
    def duplicate_tool():
        return "first"

    with pytest.raises(ToolRegistrationError, match="already registered"):
        @registry.register(category="utility", description="Second tool")
        def duplicate_tool():
            return "second"


# Test 4: Parameter schema validation - valid parameters
def test_parameter_validation_valid(registry):
    """Test parameter validation with valid inputs."""

    @registry.register(
        category="filesystem",
        description="Test tool",
        parameters={
            "name": {"type": "string", "required": True, "description": "Name"},
            "count": {"type": "integer", "required": True, "description": "Count"},
            "flag": {"type": "boolean", "default": False, "description": "Flag"}
        }
    )
    def test_tool(name: str, count: int, flag: bool = False):
        return f"{name} {count} {flag}"

    tool = registry.get("test_tool")

    # Valid parameters - should not raise
    tool.validate_parameters({"name": "test", "count": 5})
    tool.validate_parameters({"name": "test", "count": 5, "flag": True})


# Test 5: Parameter validation - missing required
def test_parameter_validation_missing_required(registry):
    """Test parameter validation fails for missing required parameters."""

    @registry.register(
        category="utility",
        description="Test tool",
        parameters={
            "required_param": {"type": "string", "required": True, "description": "Required"}
        }
    )
    def test_tool(required_param: str):
        return required_param

    tool = registry.get("test_tool")

    with pytest.raises(ToolValidationError, match="Missing required parameters"):
        tool.validate_parameters({})


# Test 6: Parameter validation - type checking
def test_parameter_validation_type_checking(registry):
    """Test parameter validation enforces types."""

    @registry.register(
        category="utility",
        description="Test tool",
        parameters={
            "number": {"type": "integer", "required": True, "description": "Number"}
        }
    )
    def test_tool(number: int):
        return number

    tool = registry.get("test_tool")

    # Valid type
    tool.validate_parameters({"number": 42})

    # Invalid type
    with pytest.raises(ToolValidationError, match="must be of type"):
        tool.validate_parameters({"number": "not_a_number"})


# Test 7: Parameter validation - enum constraint
def test_parameter_validation_enum(registry):
    """Test parameter validation with enum constraint."""

    @registry.register(
        category="utility",
        description="Test tool",
        parameters={
            "mode": {
                "type": "string",
                "required": True,
                "description": "Mode",
                "enum": ["read", "write", "append"]
            }
        }
    )
    def test_tool(mode: str):
        return mode

    tool = registry.get("test_tool")

    # Valid enum value
    tool.validate_parameters({"mode": "read"})

    # Invalid enum value
    with pytest.raises(ToolValidationError, match="must be one of"):
        tool.validate_parameters({"mode": "invalid"})


# Test 8: Tool lookup by name
def test_tool_lookup_by_name(registry):
    """Test tool lookup by name."""

    @registry.register(category="utility", description="Tool 1")
    def tool1():
        return "1"

    @registry.register(category="utility", description="Tool 2")
    def tool2():
        return "2"

    # Successful lookup
    tool = registry.get("tool1")
    assert tool is not None
    assert tool.name == "tool1"

    # Failed lookup
    tool = registry.get("nonexistent")
    assert tool is None


# Test 9: Tool lookup by category
def test_tool_lookup_by_category(registry):
    """Test tool lookup by category."""

    @registry.register(category="filesystem", description="FS Tool")
    def fs_tool():
        return "fs"

    @registry.register(category="shell", description="Shell Tool")
    def shell_tool():
        return "shell"

    @registry.register(category="filesystem", description="Another FS Tool")
    def fs_tool2():
        return "fs2"

    # Lookup by category
    fs_tools = registry.get_by_category("filesystem")
    assert len(fs_tools) == 2
    assert all(t.category == ToolCategory.FILESYSTEM for t in fs_tools)

    shell_tools = registry.get_by_category(ToolCategory.SHELL)
    assert len(shell_tools) == 1
    assert shell_tools[0].name == "shell_tool"


# Test 10: Tool lookup by tags
def test_tool_lookup_by_tags(registry):
    """Test tool lookup by tags."""

    @registry.register(
        category="utility",
        description="Tool 1",
        tags=["tag1", "tag2"]
    )
    def tool1():
        return "1"

    @registry.register(
        category="utility",
        description="Tool 2",
        tags=["tag2", "tag3"]
    )
    def tool2():
        return "2"

    @registry.register(
        category="utility",
        description="Tool 3",
        tags=["tag1"]
    )
    def tool3():
        return "3"

    # Match any tag
    tools = registry.get_by_tags(["tag1"], match_all=False)
    assert len(tools) == 2
    assert {t.name for t in tools} == {"tool1", "tool3"}

    # Match all tags
    tools = registry.get_by_tags(["tag1", "tag2"], match_all=True)
    assert len(tools) == 1
    assert tools[0].name == "tool1"

    # No matches
    tools = registry.get_by_tags(["nonexistent"], match_all=False)
    assert len(tools) == 0


# Test 11: OpenAI function calling format
def test_openai_schema_format(registry):
    """Test OpenAI function calling schema generation."""

    @registry.register(
        category="filesystem",
        description="Read file contents",
        parameters={
            "path": {"type": "string", "required": True, "description": "File path"},
            "encoding": {"type": "string", "default": "utf-8", "description": "Encoding"}
        }
    )
    def fs_read(path: str, encoding: str = "utf-8"):
        return "content"

    tool = registry.get("fs_read")
    schema = tool.to_openai_schema()

    # Verify schema structure
    assert schema["type"] == "function"
    assert "function" in schema

    func_schema = schema["function"]
    assert func_schema["name"] == "fs_read"
    assert func_schema["description"] == "Read file contents"
    assert "parameters" in func_schema

    params_schema = func_schema["parameters"]
    assert params_schema["type"] == "object"
    assert "properties" in params_schema
    assert "required" in params_schema

    # Verify properties
    props = params_schema["properties"]
    assert "path" in props
    assert "encoding" in props
    assert props["path"]["type"] == "string"
    assert props["encoding"]["default"] == "utf-8"

    # Verify required
    assert params_schema["required"] == ["path"]


# Test 12: Tool execution with validation
def test_tool_execution(registry):
    """Test tool execution through registry."""

    @registry.register(
        category="utility",
        description="Add numbers",
        parameters={
            "a": {"type": "integer", "required": True, "description": "First number"},
            "b": {"type": "integer", "required": True, "description": "Second number"}
        }
    )
    def add(a: int, b: int):
        return a + b

    # Execute via registry
    result = registry.execute("add", a=5, b=3)
    assert result == 8

    # Execute via tool metadata
    tool = registry.get("add")
    result = tool(a=10, b=20)
    assert result == 30


# Test 13: Tool execution errors
def test_tool_execution_errors(registry):
    """Test tool execution error handling."""

    @registry.register(
        category="utility",
        description="Test tool",
        parameters={
            "value": {"type": "integer", "required": True, "description": "Value"}
        }
    )
    def test_tool(value: int):
        return value

    # Tool not found
    with pytest.raises(ToolNotFoundError):
        registry.execute("nonexistent", value=1)

    # Invalid parameters
    with pytest.raises(ToolValidationError):
        registry.execute("test_tool", value="not_an_int")


# Test 14: Tool unregistration
def test_tool_unregistration(registry):
    """Test tool unregistration."""

    @registry.register(category="utility", description="Test tool")
    def test_tool():
        return "test"

    assert "test_tool" in registry.list_tools()

    # Unregister
    result = registry.unregister("test_tool")
    assert result is True
    assert "test_tool" not in registry.list_tools()

    # Unregister nonexistent
    result = registry.unregister("nonexistent")
    assert result is False


# Test 15: Registry clear and metadata export
def test_registry_clear_and_export(registry):
    """Test registry clear and metadata export."""

    @registry.register(
        category="filesystem",
        description="FS Tool",
        tags=["tag1"]
    )
    def fs_tool():
        return "fs"

    @registry.register(
        category="shell",
        description="Shell Tool",
        tags=["tag2"]
    )
    def shell_tool():
        return "shell"

    # Export metadata
    metadata = registry.export_metadata()
    assert "tools" in metadata
    assert "statistics" in metadata
    assert len(metadata["tools"]) == 2
    assert metadata["statistics"]["total_tools"] == 2

    # Clear registry
    registry.clear()
    assert len(registry.list_tools()) == 0
    assert len(registry.list_tags()) == 0


# Test 16: ParameterSchema nested objects
def test_parameter_schema_nested_objects(registry):
    """Test parameter validation with nested object schemas."""

    @registry.register(
        category="utility",
        description="Test nested params",
        parameters={
            "config": {
                "type": "object",
                "required": True,
                "description": "Configuration",
                "properties": {
                    "host": ParameterSchema(
                        name="host",
                        type="string",
                        description="Host",
                        required=True
                    ),
                    "port": ParameterSchema(
                        name="port",
                        type="integer",
                        description="Port",
                        required=True
                    )
                }
            }
        }
    )
    def test_tool(config: dict):
        return config

    tool = registry.get("test_tool")

    # Valid nested object
    tool.validate_parameters({
        "config": {"host": "localhost", "port": 8080}
    })

    # Missing nested required field
    with pytest.raises(ToolValidationError, match="Required property"):
        tool.validate_parameters({
            "config": {"host": "localhost"}
        })


# Test 17: ParameterSchema arrays
def test_parameter_schema_arrays(registry):
    """Test parameter validation with array schemas."""

    @registry.register(
        category="utility",
        description="Test array params",
        parameters={
            "items": {
                "type": "array",
                "required": True,
                "description": "Items",
                "items": ParameterSchema(
                    name="item",
                    type="string",
                    description="Item"
                )
            }
        }
    )
    def test_tool(items: list):
        return items

    tool = registry.get("test_tool")

    # Valid array
    tool.validate_parameters({"items": ["a", "b", "c"]})

    # Invalid array item type
    with pytest.raises(ToolValidationError):
        tool.validate_parameters({"items": ["a", 123, "c"]})


# Test 18: List all categories and tags
def test_list_categories_and_tags(registry):
    """Test listing categories and tags."""

    @registry.register(
        category="filesystem",
        description="FS Tool",
        tags=["file", "io"]
    )
    def fs_tool():
        return "fs"

    @registry.register(
        category="shell",
        description="Shell Tool",
        tags=["cmd", "io"]
    )
    def shell_tool():
        return "shell"

    # List categories
    categories = registry.list_categories()
    assert "filesystem" in categories
    assert "shell" in categories

    # List tags
    tags = registry.list_tags()
    assert "file" in tags
    assert "cmd" in tags
    assert "io" in tags


# Test 19: Get OpenAI schemas with category filter
def test_get_openai_schemas_filtered(registry):
    """Test getting OpenAI schemas with category filter."""

    @registry.register(category="filesystem", description="FS Tool")
    def fs_tool():
        return "fs"

    @registry.register(category="shell", description="Shell Tool")
    def shell_tool():
        return "shell"

    # All schemas
    all_schemas = registry.get_openai_schemas()
    assert len(all_schemas) == 2

    # Filtered by category
    fs_schemas = registry.get_openai_schemas(category="filesystem")
    assert len(fs_schemas) == 1
    assert fs_schemas[0]["function"]["name"] == "fs_tool"


# Test 20: Tool with examples
def test_tool_with_examples(registry):
    """Test tool registration with usage examples."""

    @registry.register(
        category="utility",
        description="Example tool",
        examples=[
            "example_tool(value=1)",
            "example_tool(value=2, flag=True)"
        ]
    )
    def example_tool(value: int, flag: bool = False):
        return f"{value} {flag}"

    tool = registry.get("example_tool")
    assert len(tool.examples) == 2
    assert "example_tool(value=1)" in tool.examples
