"""
Observability decorators for tracking LLM calls, task operations, and agent routing.

This module provides the @track_* decorators that integrate with MLflow for
comprehensive observability across PromptChain. Uses the ghost pattern for
zero overhead when disabled, background queue for non-blocking operations,
and ContextVars for async-safe nested run tracking.

Usage:
    from promptchain.observability import track_llm_call, track_task, track_routing, track_session

    @track_session()
    def main():
        init_mlflow()
        try:
            # CLI logic
            ...
        finally:
            shutdown_mlflow()

    @track_llm_call(model_param="model_name", extract_args=["temperature", "max_tokens"])
    async def run_model_async(self, model_name: str, messages: List[Dict], ...):
        # Existing implementation unchanged
        ...

    @track_task(operation_type="CREATE")
    def create_task_list(self, objective: str, tasks: List[Dict]):
        # Existing implementation unchanged
        ...

    @track_routing(extract_decision=True)
    async def _route_to_agent(self):
        # Existing implementation unchanged
        ...
"""

import time
import logging
import inspect
from functools import wraps
from typing import Callable, Optional, List, Any, Dict

from .config import is_enabled, get_experiment_name, get_tracking_uri
from .ghost import conditional_decorator
from .context import run_context, get_current_run
from .queue import (
    queue_log_metric,
    queue_log_param,
    queue_log_params,
    queue_set_tag,
    flush_queue,
    shutdown_background_logger
)
from .extractors import (
    extract_function_args,
    extract_llm_params,
    extract_task_metadata,
    extract_routing_metadata,
    extract_all_metadata,
    sanitize_for_mlflow
)
from .mlflow_adapter import is_available, set_experiment

logger = logging.getLogger(__name__)


def track_llm_call(
    model_param: str = "model_name",
    extract_args: Optional[List[str]] = None
) -> Callable:
    """
    Decorator for tracking LLM calls with automatic parameter extraction.

    Tracks model name, execution time, token usage, and custom parameters.
    Uses background queue for non-blocking MLflow logging and supports
    nested runs via ContextVars.

    Args:
        model_param: Parameter name containing the model identifier (default: "model_name")
        extract_args: Additional parameter names to extract and log (e.g., ["temperature", "max_tokens"])

    Returns:
        Decorator function that tracks LLM execution

    Example:
        @track_llm_call(model_param="model_name", extract_args=["temperature", "max_tokens"])
        async def run_model_async(self, model_name: str, messages: List[Dict], temperature: float = 0.7):
            # Existing implementation unchanged
            response = await litellm.acompletion(model=model_name, messages=messages)
            return response

    Tracked Metrics:
        - execution_time_seconds: Total execution time
        - prompt_tokens: Number of prompt tokens (if available in response)
        - completion_tokens: Number of completion tokens (if available)
        - total_tokens: Total tokens used (if available)

    Tracked Parameters:
        - model: Model identifier (from model_param)
        - temperature, max_tokens, etc.: LLM parameters (auto-extracted)
        - Custom parameters from extract_args

    Error Handling:
        Logs exception type and message to MLflow, then re-raises original exception.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract function arguments using inspect
            params = extract_function_args(func, args, kwargs)

            # Extract LLM-specific parameters
            llm_params = extract_llm_params(params)

            # Get model name from specified parameter
            model_name = params.get(model_param, "unknown")

            # Extract additional custom arguments if specified
            custom_params = {}
            if extract_args:
                for arg_name in extract_args:
                    if arg_name in params:
                        custom_params[arg_name] = params[arg_name]

            # Start nested run under current session
            with run_context(f"llm_call_{model_name}"):
                # Log parameters to MLflow
                queue_log_param("model", sanitize_for_mlflow(model_name))

                # Log LLM parameters
                for key, value in llm_params.items():
                    queue_log_param(key, sanitize_for_mlflow(value))

                # Log custom parameters
                for key, value in custom_params.items():
                    queue_log_param(key, sanitize_for_mlflow(value))

                # Track execution time
                start_time = time.time()

                try:
                    # Execute original function
                    result = await func(*args, **kwargs)

                    # Log execution time
                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "success")

                    # Extract token counts from result if available
                    if hasattr(result, 'usage'):
                        if hasattr(result.usage, 'prompt_tokens'):
                            queue_log_metric("prompt_tokens", float(result.usage.prompt_tokens))
                        if hasattr(result.usage, 'completion_tokens'):
                            queue_log_metric("completion_tokens", float(result.usage.completion_tokens))
                        if hasattr(result.usage, 'total_tokens'):
                            queue_log_metric("total_tokens", float(result.usage.total_tokens))

                    return result

                except Exception as e:
                    # Log exception to MLflow, then re-raise (FR-016)
                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "error")
                    queue_set_tag("error_type", type(e).__name__)
                    queue_log_param("error_message", str(e)[:250])  # Truncate long error messages
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Same logic as async_wrapper but without await
            params = extract_function_args(func, args, kwargs)
            llm_params = extract_llm_params(params)
            model_name = params.get(model_param, "unknown")

            custom_params = {}
            if extract_args:
                for arg_name in extract_args:
                    if arg_name in params:
                        custom_params[arg_name] = params[arg_name]

            with run_context(f"llm_call_{model_name}"):
                queue_log_param("model", sanitize_for_mlflow(model_name))

                for key, value in llm_params.items():
                    queue_log_param(key, sanitize_for_mlflow(value))

                for key, value in custom_params.items():
                    queue_log_param(key, sanitize_for_mlflow(value))

                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "success")

                    if hasattr(result, 'usage'):
                        if hasattr(result.usage, 'prompt_tokens'):
                            queue_log_metric("prompt_tokens", float(result.usage.prompt_tokens))
                        if hasattr(result.usage, 'completion_tokens'):
                            queue_log_metric("completion_tokens", float(result.usage.completion_tokens))
                        if hasattr(result.usage, 'total_tokens'):
                            queue_log_metric("total_tokens", float(result.usage.total_tokens))

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "error")
                    queue_set_tag("error_type", type(e).__name__)
                    queue_log_param("error_message", str(e)[:250])
                    raise

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    # Apply ghost pattern - returns identity decorator if disabled
    return conditional_decorator(decorator)


def track_task(operation_type: str) -> Callable:
    """
    Decorator for tracking task operations (CREATE, UPDATE, STATE_CHANGE).

    Tracks task metadata, execution time, and success/failure status.
    Uses smart extraction to capture task IDs, statuses, and other metadata.

    Args:
        operation_type: Type of task operation (CREATE, UPDATE, DELETE, STATE_CHANGE)

    Returns:
        Decorator function that tracks task operations

    Example:
        @track_task(operation_type="CREATE")
        def create_task_list(self, objective: str, tasks: List[Dict]):
            # Existing implementation unchanged
            task_list = TaskList(objective=objective, tasks=tasks)
            return task_list

    Tracked Metrics:
        - execution_time_seconds: Total execution time
        - task_count: Number of tasks processed (if applicable)

    Tracked Parameters:
        - operation_type: Type of operation
        - task_id: Task identifier (auto-extracted)
        - status: Task status (auto-extracted)
        - priority: Task priority (auto-extracted)
        - Additional metadata from extractors.extract_task_metadata()

    Error Handling:
        Logs exception type and message to MLflow, then re-raises original exception.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract function arguments
            params = extract_function_args(func, args, kwargs)

            # Extract task-specific metadata
            task_metadata = extract_task_metadata(params)

            # Start nested run under current session
            with run_context(f"task_{operation_type.lower()}"):
                # Log operation type
                queue_log_param("operation_type", operation_type)

                # Log task metadata
                for key, value in task_metadata.items():
                    queue_log_param(key, sanitize_for_mlflow(value))

                # Track execution time
                start_time = time.time()

                try:
                    # Execute original function
                    result = await func(*args, **kwargs)

                    # Log execution time
                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "success")

                    # Extract task count from result if it's a list
                    if isinstance(result, (list, tuple)):
                        queue_log_metric("task_count", float(len(result)))

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "error")
                    queue_set_tag("error_type", type(e).__name__)
                    queue_log_param("error_message", str(e)[:250])
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            params = extract_function_args(func, args, kwargs)
            task_metadata = extract_task_metadata(params)

            with run_context(f"task_{operation_type.lower()}"):
                queue_log_param("operation_type", operation_type)

                for key, value in task_metadata.items():
                    queue_log_param(key, sanitize_for_mlflow(value))

                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "success")

                    if isinstance(result, (list, tuple)):
                        queue_log_metric("task_count", float(len(result)))

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "error")
                    queue_set_tag("error_type", type(e).__name__)
                    queue_log_param("error_message", str(e)[:250])
                    raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return conditional_decorator(decorator)


def track_routing(extract_decision: bool = True) -> Callable:
    """
    Decorator for tracking agent routing decisions.

    Tracks which agent was selected, routing strategy, confidence scores,
    and execution time. Uses smart extraction to capture routing metadata.

    Args:
        extract_decision: Whether to extract and log routing decision details (default: True)

    Returns:
        Decorator function that tracks routing operations

    Example:
        @track_routing(extract_decision=True)
        async def _route_to_agent(self):
            # Existing implementation unchanged
            decision = await self._make_routing_decision()
            selected_agent = self.agents[decision['agent_name']]
            return selected_agent, decision

    Tracked Metrics:
        - execution_time_seconds: Total execution time
        - confidence: Routing confidence score (if available)

    Tracked Parameters:
        - selected_agent: Name of selected agent (auto-extracted)
        - routing_strategy: Strategy used for routing (auto-extracted)
        - decision_reason: Reason for selection (auto-extracted)
        - Additional metadata from extractors.extract_routing_metadata()

    Error Handling:
        Logs exception type and message to MLflow, then re-raises original exception.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract function arguments
            params = extract_function_args(func, args, kwargs)

            # Start nested run under current session
            with run_context("agent_routing"):
                # Track execution time
                start_time = time.time()

                try:
                    # Execute original function
                    result = await func(*args, **kwargs)

                    # Log execution time
                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "success")

                    # Extract routing metadata from result if enabled
                    if extract_decision:
                        routing_metadata = extract_routing_metadata(result)

                        # Log routing decision
                        for key, value in routing_metadata.items():
                            if key == "confidence":
                                queue_log_metric(key, float(value))
                            else:
                                queue_log_param(key, sanitize_for_mlflow(value))

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "error")
                    queue_set_tag("error_type", type(e).__name__)
                    queue_log_param("error_message", str(e)[:250])
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            params = extract_function_args(func, args, kwargs)

            with run_context("agent_routing"):
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "success")

                    if extract_decision:
                        routing_metadata = extract_routing_metadata(result)

                        for key, value in routing_metadata.items():
                            if key == "confidence":
                                queue_log_metric(key, float(value))
                            else:
                                queue_log_param(key, sanitize_for_mlflow(value))

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "error")
                    queue_set_tag("error_type", type(e).__name__)
                    queue_log_param("error_message", str(e)[:250])
                    raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return conditional_decorator(decorator)


def track_session() -> Callable:
    """
    Decorator for tracking top-level CLI sessions.

    Creates a parent MLflow run for all nested operations. Should be applied
    to the main() function in CLI entry points.

    Returns:
        Decorator function that creates session-level tracking

    Example:
        @track_session()
        def main():
            init_mlflow()
            try:
                # CLI logic with nested @track_* decorators
                agent_chain = AgentChain(...)
                agent_chain.run_chat()
            finally:
                shutdown_mlflow()

    Tracked Parameters:
        - session_type: Type of session (e.g., "cli", "api")
        - start_time: Session start timestamp

    Tracked Metrics:
        - total_duration_seconds: Total session duration

    Error Handling:
        Logs exception type and message to MLflow, then re-raises original exception.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with run_context("session"):
                # Log session metadata
                queue_log_param("session_type", "cli")
                queue_set_tag("start_time", time.strftime("%Y-%m-%d %H:%M:%S"))

                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)

                    total_duration = time.time() - start_time
                    queue_log_metric("total_duration_seconds", total_duration)
                    queue_set_tag("status", "success")

                    return result

                except Exception as e:
                    total_duration = time.time() - start_time
                    queue_log_metric("total_duration_seconds", total_duration)
                    queue_set_tag("status", "error")
                    queue_set_tag("error_type", type(e).__name__)
                    queue_log_param("error_message", str(e)[:250])
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with run_context("session"):
                queue_log_param("session_type", "cli")
                queue_set_tag("start_time", time.strftime("%Y-%m-%d %H:%M:%S"))

                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    total_duration = time.time() - start_time
                    queue_log_metric("total_duration_seconds", total_duration)
                    queue_set_tag("status", "success")

                    return result

                except Exception as e:
                    total_duration = time.time() - start_time
                    queue_log_metric("total_duration_seconds", total_duration)
                    queue_set_tag("status", "error")
                    queue_set_tag("error_type", type(e).__name__)
                    queue_log_param("error_message", str(e)[:250])
                    raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return conditional_decorator(decorator)


def track_tool(tool_name: Optional[str] = None) -> Callable:
    """
    Decorator for tracking MCP tool calls.

    Optional decorator for tracking external tool invocations via MCP.
    Tracks tool name, execution time, and success/failure status.

    Args:
        tool_name: Name of the tool being called (auto-detected if None)

    Returns:
        Decorator function that tracks tool execution

    Example:
        @track_tool(tool_name="filesystem_read")
        async def call_mcp_tool(self, tool_name: str, arguments: Dict):
            # Existing implementation unchanged
            result = await self.mcp_client.call_tool(tool_name, arguments)
            return result

    Tracked Metrics:
        - execution_time_seconds: Total execution time

    Tracked Parameters:
        - tool_name: Name of the called tool
        - tool_arguments: Tool arguments (sanitized)

    Error Handling:
        Logs exception type and message to MLflow, then re-raises original exception.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract function arguments
            params = extract_function_args(func, args, kwargs)

            # Get tool name from parameter or use provided name
            actual_tool_name = tool_name or params.get("tool_name", "unknown_tool")

            with run_context(f"tool_{actual_tool_name}"):
                queue_log_param("tool_name", actual_tool_name)

                # Log tool arguments if available
                if "arguments" in params:
                    queue_log_param("tool_arguments", sanitize_for_mlflow(params["arguments"]))

                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)

                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "success")

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "error")
                    queue_set_tag("error_type", type(e).__name__)
                    queue_log_param("error_message", str(e)[:250])
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            params = extract_function_args(func, args, kwargs)
            actual_tool_name = tool_name or params.get("tool_name", "unknown_tool")

            with run_context(f"tool_{actual_tool_name}"):
                queue_log_param("tool_name", actual_tool_name)

                if "arguments" in params:
                    queue_log_param("tool_arguments", sanitize_for_mlflow(params["arguments"]))

                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "success")

                    return result

                except Exception as e:
                    execution_time = time.time() - start_time
                    queue_log_metric("execution_time_seconds", execution_time)
                    queue_set_tag("status", "error")
                    queue_set_tag("error_type", type(e).__name__)
                    queue_log_param("error_message", str(e)[:250])
                    raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return conditional_decorator(decorator)


def init_mlflow() -> None:
    """
    Initialize MLflow tracking for the session.

    Called at CLI startup to set up MLflow experiment and verify
    server connectivity. Gracefully handles MLflow unavailability.

    Example:
        @track_session()
        def main():
            init_mlflow()  # Setup MLflow tracking
            try:
                # CLI logic
                ...
            finally:
                shutdown_mlflow()  # Cleanup

    Raises:
        No exceptions - logs warnings if MLflow unavailable
    """
    if not is_enabled():
        logger.info("MLflow tracking disabled via environment variable")
        return

    if not is_available():
        logger.warning("MLflow not available - tracking disabled for this session")
        return

    try:
        # Set experiment
        experiment_name = get_experiment_name()
        set_experiment(experiment_name)

        tracking_uri = get_tracking_uri()
        logger.info(f"MLflow tracking initialized: {tracking_uri} | Experiment: {experiment_name}")
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow: {e}")


def shutdown_mlflow() -> None:
    """
    Gracefully shutdown MLflow tracking.

    Called at CLI shutdown to flush pending operations and close
    MLflow runs. Ensures all metrics are persisted before exit.

    Example:
        @track_session()
        def main():
            init_mlflow()
            try:
                # CLI logic
                ...
            finally:
                shutdown_mlflow()  # Flush queue and close runs

    Raises:
        No exceptions - logs warnings if shutdown fails
    """
    if not is_enabled():
        return

    try:
        logger.info("Shutting down MLflow tracking...")

        # Flush any pending operations
        flush_queue(timeout=10.0)

        # Shutdown background logger
        shutdown_background_logger(timeout=5.0)

        logger.info("MLflow tracking shutdown complete")
    except Exception as e:
        logger.warning(f"Error during MLflow shutdown: {e}")
