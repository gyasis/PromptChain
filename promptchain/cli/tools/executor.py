"""Tool executor engine for safe and efficient tool execution.

Provides parameter validation, type coercion, async/sync execution support,
and comprehensive error handling with performance monitoring.

Architecture:
- ToolExecutor: Main execution engine with validation and safety checks
- Parameter validation against tool schemas
- Type coercion (string "123" → int 123)
- Async/sync transparent execution
- Performance monitoring (<50ms overhead target)

Example:
    >>> from promptchain.cli.tools.registry import ToolRegistry
    >>> from promptchain.cli.tools.executor import ToolExecutor
    >>>
    >>> registry = ToolRegistry()
    >>> executor = ToolExecutor(registry)
    >>>
    >>> # Execute a tool
    >>> result = executor.execute("fs.read", path="/path/to/file.txt")
    >>> if result.success:
    ...     print(result.result)
"""

import asyncio
import inspect
import logging
import time
from typing import Any, Dict, Optional, Union

from promptchain.cli.utils.event_loop_manager import run_async_in_context

from .models import ExecutionMetrics, ToolExecutionContext, ToolResult
from .registry import ToolMetadata, ToolNotFoundError, ToolRegistry, ToolValidationError
from .safety import SafetyValidator, SecurityError


logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Raised when tool execution fails for non-validation reasons."""

    pass


class ToolExecutor:
    """
    Safe tool execution engine with parameter validation and performance monitoring.

    Features:
    - Execute tools by name with parameter validation
    - Type coercion for common conversions (string to int, etc.)
    - Transparent async/sync execution
    - Safety validation (path traversal, command injection)
    - Performance monitoring (<50ms overhead target)
    - Comprehensive error handling and reporting

    Example:
        >>> executor = ToolExecutor(registry, safety_validator)
        >>> result = executor.execute("fs.read", path="file.py")
        >>> print(result.success, result.execution_time_ms)
        True 12.5
    """

    def __init__(
        self,
        registry: ToolRegistry,
        safety_validator: Optional[SafetyValidator] = None,
        enable_type_coercion: bool = True,
        max_execution_time_ms: float = 30000,  # 30 seconds default
    ):
        """
        Initialize tool executor.

        Args:
            registry: Tool registry containing registered tools
            safety_validator: Optional safety validator for security checks
            enable_type_coercion: Enable automatic type coercion
            max_execution_time_ms: Maximum execution time before timeout
        """
        self.registry = registry
        self.safety_validator = safety_validator
        self.enable_type_coercion = enable_type_coercion
        self.max_execution_time_ms = max_execution_time_ms

        # Performance tracking
        self._total_executions = 0
        self._total_execution_time_ms = 0.0
        self._total_validation_time_ms = 0.0

    def execute(
        self,
        tool_name: str,
        context: Optional[ToolExecutionContext] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a tool by name with parameters (synchronous).

        Args:
            tool_name: Name of tool to execute
            context: Optional execution context for logging/security
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution details

        Example:
            >>> result = executor.execute("fs.read", path="/path/to/file.txt")
            >>> if result.success:
            ...     print(f"Read {len(result.result)} characters")
        """
        start_time = time.perf_counter()
        metrics = ExecutionMetrics(parameter_count=len(kwargs))

        try:
            # Get tool metadata
            tool = self._get_tool(tool_name)

            # Validate and coerce parameters
            validation_start = time.perf_counter()
            validated_params = self._validate_and_coerce_parameters(tool, kwargs)
            metrics.validation_time_ms = (time.perf_counter() - validation_start) * 1000

            # Safety validation (if enabled)
            if self.safety_validator:
                self._safety_validate(tool_name, validated_params, context)

            # Execute tool (handle sync/async)
            exec_start = time.perf_counter()
            if asyncio.iscoroutinefunction(tool.function):
                # Use run_async_in_context to safely handle both TUI (loop
                # already running) and CLI (no loop) contexts per FR-002
                result = run_async_in_context(tool.function(**validated_params))
            else:
                # Direct sync execution
                result = tool.function(**validated_params)

            metrics.execution_time_ms = (time.perf_counter() - exec_start) * 1000
            metrics.total_time_ms = (time.perf_counter() - start_time) * 1000

            # Update global metrics
            self._update_metrics(metrics)

            # Log successful execution
            logger.debug(
                f"Tool executed: {tool_name} in {metrics.total_time_ms:.2f}ms "
                f"(validation: {metrics.validation_time_ms:.2f}ms, "
                f"execution: {metrics.execution_time_ms:.2f}ms)"
            )

            return ToolResult(
                success=True,
                tool_name=tool_name,
                result=result,
                execution_time_ms=metrics.total_time_ms,
                metadata={"metrics": metrics.to_dict()},
            )

        except (ToolNotFoundError, ToolValidationError, SecurityError) as e:
            # Expected errors (validation, not found, security)
            metrics.total_time_ms = (time.perf_counter() - start_time) * 1000

            logger.warning(f"Tool execution failed: {tool_name} - {type(e).__name__}: {e}")

            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=metrics.total_time_ms,
                metadata={"metrics": metrics.to_dict()},
            )

        except Exception as e:
            # Unexpected errors during execution
            metrics.total_time_ms = (time.perf_counter() - start_time) * 1000

            logger.error(f"Unexpected error executing tool {tool_name}: {e}", exc_info=True)

            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=metrics.total_time_ms,
                metadata={"metrics": metrics.to_dict()},
            )

    async def execute_async(
        self,
        tool_name: str,
        context: Optional[ToolExecutionContext] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a tool by name with parameters (asynchronous).

        Handles both sync and async tools transparently. Async tools are
        awaited directly, sync tools are executed in a thread pool to avoid
        blocking the event loop.

        Args:
            tool_name: Name of tool to execute
            context: Optional execution context for logging/security
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution details

        Example:
            >>> result = await executor.execute_async("fs.read", path="file.txt")
            >>> print(result.success)
            True
        """
        start_time = time.perf_counter()
        metrics = ExecutionMetrics(parameter_count=len(kwargs))

        try:
            # Get tool metadata
            tool = self._get_tool(tool_name)

            # Validate and coerce parameters
            validation_start = time.perf_counter()
            validated_params = self._validate_and_coerce_parameters(tool, kwargs)
            metrics.validation_time_ms = (time.perf_counter() - validation_start) * 1000

            # Safety validation (if enabled)
            if self.safety_validator:
                self._safety_validate(tool_name, validated_params, context)

            # Execute tool (handle sync/async)
            exec_start = time.perf_counter()
            if asyncio.iscoroutinefunction(tool.function):
                # Await async function directly
                result = await tool.function(**validated_params)
            else:
                # Run sync function in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tool.function(**validated_params))

            metrics.execution_time_ms = (time.perf_counter() - exec_start) * 1000
            metrics.total_time_ms = (time.perf_counter() - start_time) * 1000

            # Update global metrics
            self._update_metrics(metrics)

            # Log successful execution
            logger.debug(
                f"Tool executed (async): {tool_name} in {metrics.total_time_ms:.2f}ms "
                f"(validation: {metrics.validation_time_ms:.2f}ms, "
                f"execution: {metrics.execution_time_ms:.2f}ms)"
            )

            return ToolResult(
                success=True,
                tool_name=tool_name,
                result=result,
                execution_time_ms=metrics.total_time_ms,
                metadata={"metrics": metrics.to_dict()},
            )

        except (ToolNotFoundError, ToolValidationError, SecurityError) as e:
            # Expected errors (validation, not found, security)
            metrics.total_time_ms = (time.perf_counter() - start_time) * 1000

            logger.warning(f"Tool execution failed (async): {tool_name} - {type(e).__name__}: {e}")

            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=metrics.total_time_ms,
                metadata={"metrics": metrics.to_dict()},
            )

        except Exception as e:
            # Unexpected errors during execution
            metrics.total_time_ms = (time.perf_counter() - start_time) * 1000

            logger.error(f"Unexpected error executing tool (async) {tool_name}: {e}", exc_info=True)

            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=metrics.total_time_ms,
                metadata={"metrics": metrics.to_dict()},
            )

    def _get_tool(self, tool_name: str) -> ToolMetadata:
        """
        Get tool metadata from registry.

        Args:
            tool_name: Tool name

        Returns:
            ToolMetadata

        Raises:
            ToolNotFoundError: If tool not found
        """
        tool = self.registry.get(tool_name)
        if not tool:
            available = self.registry.list_tools()
            raise ToolNotFoundError(
                f"Tool '{tool_name}' not found. Available tools: {', '.join(available[:5])}"
                + (f" (and {len(available) - 5} more)" if len(available) > 5 else "")
            )
        return tool

    def _validate_and_coerce_parameters(
        self, tool: ToolMetadata, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and coerce parameters against tool schema.

        Args:
            tool: Tool metadata with parameter schemas
            params: Parameters to validate

        Returns:
            Validated and coerced parameters

        Raises:
            ToolValidationError: If validation fails
        """
        validated = {}

        # Check required parameters
        required = tool.get_required_parameters()
        missing = [name for name in required if name not in params]

        if missing:
            raise ToolValidationError(
                f"Missing required parameters for tool '{tool.name}': {missing}"
            )

        # Validate and coerce each parameter
        for param_name, param_value in params.items():
            if param_name not in tool.parameters:
                # Unknown parameter - log warning but don't fail
                logger.warning(
                    f"Unknown parameter '{param_name}' for tool '{tool.name}', ignoring"
                )
                continue

            param_schema = tool.parameters[param_name]

            # Type coercion (if enabled)
            if self.enable_type_coercion:
                coerced_value = self._coerce_type(param_value, param_schema.type)
            else:
                coerced_value = param_value

            # Validate coerced value
            param_schema.validate_value(coerced_value)

            validated[param_name] = coerced_value

        # Add default values for optional parameters
        for param_name, param_schema in tool.parameters.items():
            if param_name not in validated and param_schema.default is not None:
                validated[param_name] = param_schema.default

        return validated

    def _coerce_type(self, value: Any, target_type: str) -> Any:
        """
        Coerce value to target type.

        Handles common conversions like string to int, string to bool, etc.

        Args:
            value: Value to coerce
            target_type: Target JSON schema type

        Returns:
            Coerced value (or original if no coercion needed/possible)
        """
        # Already correct type
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
        }

        check = type_checks.get(target_type)
        if check and check(value):
            return value

        # Type coercion strategies
        try:
            if target_type == "integer":
                if isinstance(value, str):
                    return int(value)
                elif isinstance(value, float):
                    return int(value)
                elif isinstance(value, bool):
                    return int(value)

            elif target_type == "number":
                if isinstance(value, str):
                    # Try int first, then float
                    try:
                        return int(value)
                    except ValueError:
                        return float(value)
                elif isinstance(value, bool):
                    return float(value)

            elif target_type == "boolean":
                if isinstance(value, str):
                    # Common string representations
                    lower = value.lower()
                    if lower in ("true", "yes", "1", "on"):
                        return True
                    elif lower in ("false", "no", "0", "off"):
                        return False
                elif isinstance(value, (int, float)):
                    return bool(value)

            elif target_type == "string":
                if value is not None:
                    return str(value)

            elif target_type == "array":
                if isinstance(value, str):
                    # Try JSON parsing
                    import json

                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            return parsed
                    except json.JSONDecodeError:
                        pass

            elif target_type == "object":
                if isinstance(value, str):
                    # Try JSON parsing
                    import json

                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass

        except (ValueError, TypeError) as e:
            logger.debug(f"Type coercion failed: {value} -> {target_type}: {e}")

        # Return original value if coercion failed
        return value

    def _safety_validate(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[ToolExecutionContext],
    ) -> None:
        """
        Perform safety validation on tool execution.

        Args:
            tool_name: Tool name
            params: Validated parameters
            context: Execution context

        Raises:
            SecurityError: If safety validation fails
        """
        # Path validation for filesystem tools
        if "path" in params:
            self.safety_validator.validate_path(params["path"])

        # Command validation for shell tools
        if "command" in params:
            cmd = params["command"]
            if isinstance(cmd, str):
                # Parse string command
                import shlex

                cmd_list = shlex.split(cmd)
                self.safety_validator.validate_command(cmd_list)
            elif isinstance(cmd, list):
                self.safety_validator.validate_command(cmd)

        # Add more tool-specific validations as needed
        # This can be extended with validators for different tool categories

    def _update_metrics(self, metrics: ExecutionMetrics) -> None:
        """
        Update global execution metrics.

        Args:
            metrics: Execution metrics from current run
        """
        self._total_executions += 1
        self._total_execution_time_ms += metrics.execution_time_ms
        self._total_validation_time_ms += metrics.validation_time_ms

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get executor performance statistics.

        Returns:
            Dictionary with performance metrics

        Example:
            >>> stats = executor.get_performance_stats()
            >>> print(f"Average overhead: {stats['avg_overhead_ms']:.2f}ms")
        """
        if self._total_executions == 0:
            return {
                "total_executions": 0,
                "avg_execution_time_ms": 0,
                "avg_validation_time_ms": 0,
                "avg_overhead_ms": 0,
            }

        avg_execution = self._total_execution_time_ms / self._total_executions
        avg_validation = self._total_validation_time_ms / self._total_executions
        avg_overhead = avg_validation

        return {
            "total_executions": self._total_executions,
            "avg_execution_time_ms": avg_execution,
            "avg_validation_time_ms": avg_validation,
            "avg_overhead_ms": avg_overhead,
            "total_execution_time_ms": self._total_execution_time_ms,
            "total_validation_time_ms": self._total_validation_time_ms,
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._total_executions = 0
        self._total_execution_time_ms = 0.0
        self._total_validation_time_ms = 0.0
