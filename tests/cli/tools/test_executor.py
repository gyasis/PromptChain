"""
Tests for ToolExecutor - Safe tool execution with validation

Tests cover:
- Basic synchronous execution
- Asynchronous execution
- Parameter validation
- Type coercion (string to int, bool, etc.)
- Error handling (tool not found, validation errors, execution errors)
- Safety validation integration
- Performance metrics tracking
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch

from promptchain.cli.tools.registry import (
    ToolRegistry,
    ToolCategory,
    ParameterSchema,
    ToolNotFoundError,
    ToolValidationError,
)
from promptchain.cli.tools.executor import ToolExecutor, ToolExecutionError
from promptchain.cli.tools.models import ToolResult, ToolExecutionContext
from promptchain.cli.tools.safety import SafetyValidator, SecurityError


@pytest.fixture
def registry():
    """Create a fresh tool registry for testing."""
    reg = ToolRegistry()

    # Register test tools
    @reg.register(
        category=ToolCategory.UTILITY,
        description="Add two numbers",
        parameters={
            "a": {"type": "integer", "required": True, "description": "First number"},
            "b": {"type": "integer", "required": True, "description": "Second number"},
        },
    )
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @reg.register(
        category=ToolCategory.UTILITY,
        description="Concatenate strings",
        parameters={
            "strings": {
                "type": "array",
                "required": True,
                "description": "Strings to concatenate",
            },
            "separator": {
                "type": "string",
                "required": False,
                "default": "",
                "description": "Separator between strings",
            },
        },
    )
    def concat(strings: list, separator: str = "") -> str:
        """Concatenate strings with separator."""
        return separator.join(strings)

    @reg.register(
        category=ToolCategory.UTILITY,
        description="Async sleep function",
        parameters={
            "seconds": {
                "type": "number",
                "required": True,
                "description": "Seconds to sleep",
            }
        },
    )
    async def async_sleep(seconds: float) -> str:
        """Async sleep function."""
        await asyncio.sleep(seconds)
        return f"Slept for {seconds}s"

    @reg.register(
        category=ToolCategory.UTILITY,
        description="Function that raises error",
        parameters={
            "message": {
                "type": "string",
                "required": True,
                "description": "Error message",
            }
        },
    )
    def error_func(message: str) -> str:
        """Function that always raises an error."""
        raise RuntimeError(message)

    @reg.register(
        category=ToolCategory.FILESYSTEM,
        description="Mock file read",
        parameters={
            "path": {"type": "string", "required": True, "description": "File path"}
        },
    )
    def mock_file_read(path: str) -> str:
        """Mock file read for testing safety validation."""
        return f"Contents of {path}"

    return reg


@pytest.fixture
def executor(registry):
    """Create executor without safety validation."""
    return ToolExecutor(registry)


@pytest.fixture
def safe_executor(registry, tmp_path):
    """Create executor with safety validation."""
    # Use tmp_path which is a real directory created by pytest
    validator = SafetyValidator(project_root=str(tmp_path))
    return ToolExecutor(registry, safety_validator=validator)


# ============================================================================
# Basic Execution Tests
# ============================================================================


def test_execute_simple_tool(executor):
    """Test basic synchronous tool execution."""
    result = executor.execute("add", a=5, b=3)

    assert result.success is True
    assert result.result == 8
    assert result.tool_name == "add"
    assert result.error is None
    assert result.execution_time_ms is not None
    assert result.execution_time_ms > 0


def test_execute_with_default_params(executor):
    """Test execution with default parameter values."""
    result = executor.execute("concat", strings=["hello", "world"])

    assert result.success is True
    assert result.result == "helloworld"

    # With separator
    result = executor.execute("concat", strings=["hello", "world"], separator=" ")

    assert result.success is True
    assert result.result == "hello world"


def test_execute_tool_not_found(executor):
    """Test execution of non-existent tool."""
    result = executor.execute("nonexistent_tool", param="value")

    assert result.success is False
    assert result.error_type == "ToolNotFoundError"
    assert "not found" in result.error.lower()


# ============================================================================
# Async Execution Tests
# ============================================================================


@pytest.mark.asyncio
async def test_execute_async_tool(executor):
    """Test async tool execution."""
    result = await executor.execute_async("async_sleep", seconds=0.01)

    assert result.success is True
    assert "Slept for" in result.result
    assert result.execution_time_ms >= 10  # At least 10ms for 0.01s sleep


@pytest.mark.asyncio
async def test_execute_async_with_sync_tool(executor):
    """Test async execution of sync tool (should work transparently)."""
    result = await executor.execute_async("add", a=10, b=20)

    assert result.success is True
    assert result.result == 30


def test_execute_sync_with_async_tool(executor):
    """Test sync execution of async tool (should work transparently)."""
    result = executor.execute("async_sleep", seconds=0.01)

    assert result.success is True
    assert "Slept for" in result.result


# ============================================================================
# Parameter Validation Tests
# ============================================================================


def test_missing_required_parameter(executor):
    """Test validation error for missing required parameter."""
    result = executor.execute("add", a=5)  # Missing 'b'

    assert result.success is False
    assert result.error_type == "ToolValidationError"
    assert "required" in result.error.lower()
    assert "b" in result.error


def test_extra_parameter_warning(executor, caplog):
    """Test that extra parameters are logged but don't fail."""
    result = executor.execute("add", a=5, b=3, c=10)

    assert result.success is True
    assert result.result == 8  # Extra param ignored


def test_parameter_type_validation(executor):
    """Test parameter type validation."""
    # This should fail if type coercion is disabled
    executor_no_coerce = ToolExecutor(executor.registry, enable_type_coercion=False)

    # String instead of integer
    result = executor_no_coerce.execute("add", a="5", b="3")

    # Without coercion, this should fail validation
    assert result.success is False
    assert result.error_type == "ToolValidationError"


# ============================================================================
# Type Coercion Tests
# ============================================================================


def test_type_coercion_string_to_int(executor):
    """Test string to integer coercion."""
    result = executor.execute("add", a="10", b="20")

    assert result.success is True
    assert result.result == 30


def test_type_coercion_float_to_int(executor):
    """Test float to integer coercion."""
    result = executor.execute("add", a=10.5, b=20.7)

    assert result.success is True
    assert result.result == 30  # 10 + 20


def test_type_coercion_invalid(executor):
    """Test invalid type coercion (should keep original and fail validation)."""
    result = executor.execute("add", a="not_a_number", b=5)

    # Should fail validation after failed coercion
    assert result.success is False


def test_type_coercion_disabled(executor):
    """Test execution with type coercion disabled."""
    executor_no_coerce = ToolExecutor(executor.registry, enable_type_coercion=False)

    result = executor_no_coerce.execute("add", a="10", b="20")

    # Should fail without coercion
    assert result.success is False
    assert result.error_type == "ToolValidationError"


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_execution_error_handling(executor):
    """Test proper error handling when tool execution fails."""
    result = executor.execute("error_func", message="Test error")

    assert result.success is False
    assert result.error_type == "RuntimeError"
    assert "Test error" in result.error
    assert result.execution_time_ms is not None


@pytest.mark.asyncio
async def test_async_execution_error_handling(executor):
    """Test async error handling."""
    result = await executor.execute_async("error_func", message="Async test error")

    assert result.success is False
    assert result.error_type == "RuntimeError"
    assert "Async test error" in result.error


# ============================================================================
# Safety Validation Tests
# ============================================================================


def test_safety_validation_path_traversal(safe_executor):
    """Test path traversal detection."""
    # Try to read file outside project root
    result = safe_executor.execute("mock_file_read", path="/etc/passwd")

    assert result.success is False
    assert result.error_type == "SecurityError"


def test_safety_validation_allowed_path(safe_executor, tmp_path):
    """Test that paths within project root are allowed."""
    # Create a test file in tmp_path
    test_file = tmp_path / "file.txt"
    test_file.write_text("test content")

    result = safe_executor.execute(
        "mock_file_read", path=str(test_file)
    )

    assert result.success is True
    assert "Contents of" in result.result


def test_safety_validation_disabled(executor):
    """Test execution without safety validation."""
    # Should work even with path outside project (no validator)
    result = executor.execute("mock_file_read", path="/etc/passwd")

    assert result.success is True


# ============================================================================
# Performance Metrics Tests
# ============================================================================


def test_performance_metrics_tracking(executor):
    """Test that performance metrics are tracked."""
    # Reset metrics
    executor.reset_metrics()

    # Execute some tools
    executor.execute("add", a=1, b=2)
    executor.execute("add", a=3, b=4)
    executor.execute("concat", strings=["a", "b"])

    stats = executor.get_performance_stats()

    assert stats["total_executions"] == 3
    assert stats["avg_execution_time_ms"] > 0
    assert stats["avg_validation_time_ms"] > 0
    assert stats["avg_overhead_ms"] > 0


def test_execution_overhead_performance(executor):
    """Test that execution overhead is <50ms (acceptance criteria)."""
    # Create a very fast tool
    registry = executor.registry

    @registry.register(
        category=ToolCategory.UTILITY,
        description="Instant return",
        parameters={},
    )
    def instant() -> str:
        return "done"

    # Execute and measure overhead
    result = executor.execute("instant")

    assert result.success is True

    # Check execution time (overhead should be minimal)
    metrics = result.metadata["metrics"]
    validation_time = metrics["validation_time_ms"]

    # Validation overhead should be <50ms (target from requirements)
    assert validation_time < 50, f"Validation overhead {validation_time}ms exceeds 50ms target"


def test_metrics_reset(executor):
    """Test metrics can be reset."""
    executor.execute("add", a=1, b=2)

    stats_before = executor.get_performance_stats()
    assert stats_before["total_executions"] > 0

    executor.reset_metrics()

    stats_after = executor.get_performance_stats()
    assert stats_after["total_executions"] == 0
    assert stats_after["avg_execution_time_ms"] == 0


# ============================================================================
# Result Format Tests
# ============================================================================


def test_result_to_dict(executor):
    """Test ToolResult serialization to dict."""
    result = executor.execute("add", a=5, b=3)

    result_dict = result.to_dict()

    assert isinstance(result_dict, dict)
    assert result_dict["success"] is True
    assert result_dict["tool_name"] == "add"
    assert result_dict["result"] == 8
    assert "execution_time_ms" in result_dict
    assert "timestamp" in result_dict


def test_result_string_representation(executor):
    """Test ToolResult string representation."""
    result_success = executor.execute("add", a=5, b=3)
    result_error = executor.execute("error_func", message="test")

    str_success = str(result_success)
    str_error = str(result_error)

    assert "success=True" in str_success
    assert "add" in str_success

    assert "success=False" in str_error
    assert "RuntimeError" in str_error


# ============================================================================
# Integration Tests
# ============================================================================


def test_execute_multiple_tools_sequence(executor):
    """Test executing multiple tools in sequence."""
    results = []

    results.append(executor.execute("add", a=5, b=3))
    results.append(executor.execute("concat", strings=["hello", "world"], separator=" "))
    results.append(executor.execute("add", a=10, b=20))

    assert all(r.success for r in results)
    assert results[0].result == 8
    assert results[1].result == "hello world"
    assert results[2].result == 30


@pytest.mark.asyncio
async def test_async_execute_parallel(executor):
    """Test parallel async execution of multiple tools."""
    # Execute multiple async operations in parallel
    tasks = [
        executor.execute_async("async_sleep", seconds=0.01),
        executor.execute_async("async_sleep", seconds=0.01),
        executor.execute_async("add", a=1, b=2),
    ]

    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    assert all(r.success for r in results)


def test_execution_context_tracking(executor):
    """Test execution context is properly tracked."""
    context = ToolExecutionContext(
        session_id="test-session",
        agent_name="test-agent",
        project_root="/home/user/project",
    )

    result = executor.execute("add", context=context, a=5, b=3)

    assert result.success is True
    # Context should be used for logging/validation but not affect result


# ============================================================================
# Edge Cases
# ============================================================================


def test_execute_with_empty_params(executor):
    """Test tool with no parameters."""
    registry = executor.registry

    @registry.register(
        category=ToolCategory.UTILITY,
        description="No params tool",
        parameters={},
    )
    def no_params() -> str:
        return "success"

    result = executor.execute("no_params")

    assert result.success is True
    assert result.result == "success"


def test_execute_with_complex_params(executor):
    """Test tool with complex nested parameters."""
    registry = executor.registry

    @registry.register(
        category=ToolCategory.UTILITY,
        description="Complex params",
        parameters={
            "config": {
                "type": "object",
                "required": True,
                "description": "Configuration object",
            }
        },
    )
    def complex_tool(config: dict) -> str:
        return f"Config: {config.get('key')}"

    result = executor.execute("complex_tool", config={"key": "value"})

    assert result.success is True
    assert "value" in result.result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
