"""
Integration tests for ToolExecutor with real-world scenarios.

Tests the complete flow from registration → validation → execution → result.
"""

import asyncio
import pytest

from promptchain.cli.tools.registry import ToolRegistry, ToolCategory
from promptchain.cli.tools.executor import ToolExecutor
from promptchain.cli.tools.safety import SafetyValidator
from promptchain.cli.tools.models import ToolExecutionContext


def test_complete_tool_lifecycle(tmp_path):
    """
    Test complete lifecycle: register → execute → validate → result.

    This simulates a real-world scenario where an LLM agent would:
    1. Discover available tools
    2. Execute a tool with parameters
    3. Get results and handle errors
    """
    # Setup
    registry = ToolRegistry()
    validator = SafetyValidator(project_root=str(tmp_path))
    executor = ToolExecutor(registry, safety_validator=validator)

    # Register a realistic file operation tool
    @registry.register(
        category=ToolCategory.FILESYSTEM,
        description="Write content to a file",
        parameters={
            "path": {
                "type": "string",
                "required": True,
                "description": "File path to write to",
            },
            "content": {
                "type": "string",
                "required": True,
                "description": "Content to write",
            },
        },
    )
    def write_file(path: str, content: str) -> str:
        """Write content to file."""
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {path}"

    # Execute tool
    test_file = tmp_path / "output.txt"
    result = executor.execute(
        "write_file", path=str(test_file), content="Hello, World!"
    )

    # Verify
    assert result.success is True
    assert "Wrote 13 characters" in result.result
    assert test_file.exists()
    assert test_file.read_text() == "Hello, World!"


def test_multi_step_workflow(tmp_path):
    """
    Test multi-step workflow: create dir → write file → read file.

    This simulates an LLM agent performing a complex task with multiple tool calls.
    """
    registry = ToolRegistry()
    executor = ToolExecutor(registry)

    # Register tools
    @registry.register(
        category=ToolCategory.FILESYSTEM,
        description="Create directory",
        parameters={
            "path": {"type": "string", "required": True, "description": "Directory path"}
        },
    )
    def create_dir(path: str) -> str:
        import os

        os.makedirs(path, exist_ok=True)
        return f"Created directory: {path}"

    @registry.register(
        category=ToolCategory.FILESYSTEM,
        description="Write file",
        parameters={
            "path": {"type": "string", "required": True, "description": "File path"},
            "content": {"type": "string", "required": True, "description": "Content"},
        },
    )
    def write_file(path: str, content: str) -> str:
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote to {path}"

    @registry.register(
        category=ToolCategory.FILESYSTEM,
        description="Read file",
        parameters={
            "path": {"type": "string", "required": True, "description": "File path"}
        },
    )
    def read_file(path: str) -> str:
        with open(path, "r") as f:
            return f.read()

    # Execute workflow
    subdir = tmp_path / "subdir"
    file_path = subdir / "data.txt"

    # Step 1: Create directory
    result1 = executor.execute("create_dir", path=str(subdir))
    assert result1.success is True

    # Step 2: Write file
    result2 = executor.execute("write_file", path=str(file_path), content="Test data")
    assert result2.success is True

    # Step 3: Read file
    result3 = executor.execute("read_file", path=str(file_path))
    assert result3.success is True
    assert result3.result == "Test data"


@pytest.mark.asyncio
async def test_async_multi_tool_execution():
    """
    Test async execution of multiple tools in parallel.

    This simulates concurrent tool execution by an async LLM agent.
    """
    registry = ToolRegistry()
    executor = ToolExecutor(registry)

    # Register async tools
    @registry.register(
        category=ToolCategory.UTILITY,
        description="Async computation",
        parameters={
            "value": {"type": "integer", "required": True, "description": "Input value"}
        },
    )
    async def async_compute(value: int) -> int:
        await asyncio.sleep(0.01)  # Simulate async work
        return value * 2

    # Execute multiple tools in parallel
    tasks = [
        executor.execute_async("async_compute", value=i) for i in range(10)
    ]

    results = await asyncio.gather(*tasks)

    # Verify all succeeded
    assert all(r.success for r in results)
    assert [r.result for r in results] == [i * 2 for i in range(10)]


def test_error_recovery_workflow():
    """
    Test error handling and recovery in a workflow.

    This simulates an LLM agent handling errors gracefully.
    """
    registry = ToolRegistry()
    executor = ToolExecutor(registry)

    # Register tools (one that can fail)
    @registry.register(
        category=ToolCategory.UTILITY,
        description="Divide numbers",
        parameters={
            "a": {"type": "number", "required": True, "description": "Numerator"},
            "b": {"type": "number", "required": True, "description": "Denominator"},
        },
    )
    def divide(a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    # Successful execution
    result1 = executor.execute("divide", a=10, b=2)
    assert result1.success is True
    assert result1.result == 5.0

    # Failed execution (error)
    result2 = executor.execute("divide", a=10, b=0)
    assert result2.success is False
    assert result2.error_type == "ValueError"
    assert "divide by zero" in result2.error.lower()

    # Recovery: try again with valid input
    result3 = executor.execute("divide", a=10, b=5)
    assert result3.success is True
    assert result3.result == 2.0


def test_type_coercion_in_workflow():
    """
    Test type coercion with LLM-provided string parameters.

    This simulates an LLM agent providing all parameters as strings
    (common when LLMs generate JSON).
    """
    registry = ToolRegistry()
    executor = ToolExecutor(registry, enable_type_coercion=True)

    @registry.register(
        category=ToolCategory.UTILITY,
        description="Calculate area",
        parameters={
            "width": {"type": "number", "required": True, "description": "Width"},
            "height": {"type": "number", "required": True, "description": "Height"},
        },
    )
    def calculate_area(width: float, height: float) -> float:
        return width * height

    # LLM provides strings (from JSON)
    result = executor.execute(
        "calculate_area",
        width="10.5",  # String
        height="20.0",  # String
    )

    assert result.success is True
    assert result.result == 210.0  # Correctly coerced and calculated


def test_context_tracking_workflow():
    """
    Test execution context tracking through workflow.

    This simulates tracking which agent/session executed which tools.
    """
    registry = ToolRegistry()
    executor = ToolExecutor(registry)

    @registry.register(
        category=ToolCategory.UTILITY,
        description="Echo tool",
        parameters={
            "message": {"type": "string", "required": True, "description": "Message"}
        },
    )
    def echo(message: str) -> str:
        return message

    # Create execution context
    context = ToolExecutionContext(
        session_id="session-123",
        agent_name="coding-agent",
        user_id="user-456",
        project_root="/home/user/project",
    )

    # Execute with context
    result = executor.execute("echo", context=context, message="Hello")

    assert result.success is True
    assert result.result == "Hello"
    # Context should be tracked (used for logging, not returned in result)


def test_performance_at_scale():
    """
    Test executor performance with many tools registered.

    This simulates a real CLI with 50+ tools registered.
    """
    registry = ToolRegistry()
    executor = ToolExecutor(registry)

    # Register many tools
    for i in range(50):

        @registry.register(
            category=ToolCategory.UTILITY,
            description=f"Tool {i}",
            parameters={
                "value": {"type": "integer", "required": True, "description": "Value"}
            },
            name=f"tool_{i}",  # Explicit name to avoid closure issues
        )
        def tool_func(value: int) -> int:
            return value + 1

    # Execute a tool
    result = executor.execute("tool_25", value=100)

    assert result.success is True
    assert result.result == 101

    # Performance should still be good
    assert result.execution_time_ms < 100  # Total time under 100ms


def test_openai_schema_integration():
    """
    Test integration with OpenAI function calling format.

    This verifies that executor works with schemas compatible with LiteLLM.
    """
    registry = ToolRegistry()
    executor = ToolExecutor(registry)

    @registry.register(
        category=ToolCategory.UTILITY,
        description="Format name",
        parameters={
            "first_name": {
                "type": "string",
                "required": True,
                "description": "First name",
            },
            "last_name": {
                "type": "string",
                "required": True,
                "description": "Last name",
            },
        },
    )
    def format_name(first_name: str, last_name: str) -> str:
        return f"{last_name}, {first_name}"

    # Get OpenAI schemas
    schemas = registry.get_openai_schemas()

    # Verify schema format
    assert len(schemas) == 1
    schema = schemas[0]

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "format_name"
    assert "first_name" in schema["function"]["parameters"]["properties"]
    assert "last_name" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["required"] == ["first_name", "last_name"]

    # Execute using schema
    result = executor.execute("format_name", first_name="John", last_name="Doe")

    assert result.success is True
    assert result.result == "Doe, John"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
