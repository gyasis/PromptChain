"""
Smart argument extraction and sanitization for MLflow tracking.

This module provides utilities for automatically extracting function parameters,
LLM configurations, and task metadata for observability tracking. Uses Python's
inspect module to capture function signatures and convert complex types to
MLflow-compatible formats.

FR-015: Automatic parameter extraction for MLflow tags
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Union


def extract_function_args(func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """
    Extract function arguments by inspecting signature.

    Maps positional and keyword arguments to their parameter names using
    Python's inspect module for automatic parameter capture.

    Args:
        func: The function being called
        args: Positional arguments passed to function
        kwargs: Keyword arguments passed to function

    Returns:
        Dictionary mapping parameter names to their values

    Example:
        >>> def example(model: str, temperature: float = 0.7):
        ...     pass
        >>> extract_function_args(example, ("gpt-4",), {"temperature": 0.9})
        {'model': 'gpt-4', 'temperature': 0.9}
    """
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (ValueError, TypeError) as e:
        # Fallback to basic arg extraction if signature binding fails
        return {f"arg_{i}": v for i, v in enumerate(args)} | kwargs


def extract_llm_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract LLM-specific parameters for tracking.

    Filters parameter dictionary to only include relevant LLM configuration
    values and converts them to MLflow-compatible strings.

    Args:
        params: Dictionary of all parameters

    Returns:
        Dictionary of LLM parameters (model, temperature, etc.)

    Example:
        >>> params = {"model": "gpt-4", "temperature": 0.7, "verbose": True}
        >>> extract_llm_params(params)
        {'model': 'gpt-4', 'temperature': '0.7'}
    """
    llm_keys = {
        'model', 'model_name', 'models',
        'temperature', 'max_tokens', 'max_completion_tokens',
        'top_p', 'top_k', 'frequency_penalty', 'presence_penalty',
        'stop', 'stream'
    }

    extracted = {}
    for key in llm_keys:
        if key in params:
            value = params[key]
            # Skip None values and non-serializable types
            if value is not None and not callable(value):
                extracted[key] = sanitize_for_mlflow(value)

    return extracted


def extract_task_metadata(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract task-specific metadata for tracking.

    Filters parameter dictionary to include task operation details like
    objective, task counts, and operation types.

    Args:
        params: Dictionary of all parameters

    Returns:
        Dictionary of task metadata

    Example:
        >>> params = {"objective": "Analyze code", "max_internal_steps": 5}
        >>> extract_task_metadata(params)
        {'objective': 'Analyze code', 'max_internal_steps': '5'}
    """
    task_keys = {
        'objective', 'max_internal_steps', 'max_iterations',
        'operation_type', 'task_count', 'task_type',
        'history_mode', 'verbose'
    }

    extracted = {}
    for key in task_keys:
        if key in params:
            value = params[key]
            if value is not None and not callable(value):
                extracted[key] = sanitize_for_mlflow(value)

    return extracted


def extract_routing_metadata(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract routing decision metadata for tracking.

    Filters parameter dictionary to include agent routing information like
    selected agent, routing strategy, and decision confidence.

    Args:
        params: Dictionary of all parameters

    Returns:
        Dictionary of routing metadata

    Example:
        >>> params = {"selected_agent": "analyzer", "strategy": "router", "confidence": 0.95}
        >>> extract_routing_metadata(params)
        {'selected_agent': 'analyzer', 'strategy': 'router', 'confidence': '0.95'}
    """
    routing_keys = {
        'selected_agent', 'chosen_agent', 'agent_name',
        'strategy', 'execution_mode', 'routing_strategy',
        'confidence', 'decision_confidence',
        'refined_query', 'router_reasoning'
    }

    extracted = {}
    for key in routing_keys:
        if key in params:
            value = params[key]
            if value is not None and not callable(value):
                extracted[key] = sanitize_for_mlflow(value)

    return extracted


def extract_chain_metadata(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract PromptChain-specific metadata for tracking.

    Filters parameter dictionary to include chain configuration like
    instruction count, step storage, and breaking modes.

    Args:
        params: Dictionary of all parameters

    Returns:
        Dictionary of chain metadata
    """
    chain_keys = {
        'instruction_count', 'store_steps',
        'breaking_mode', 'break_on_result',
        'enable_history', 'enable_tools',
        'mcp_servers_count'
    }

    extracted = {}
    for key in chain_keys:
        if key in params:
            value = params[key]
            if value is not None and not callable(value):
                extracted[key] = sanitize_for_mlflow(value)

    return extracted


def sanitize_for_mlflow(value: Any) -> str:
    """
    Convert value to MLflow-compatible string.

    MLflow has strict requirements for tag/parameter values:
    - Must be strings
    - Limited length (500 chars recommended)
    - No complex objects or callables

    Args:
        value: Any value to sanitize

    Returns:
        String representation truncated to 500 characters

    Example:
        >>> sanitize_for_mlflow([1, 2, 3, 4, 5])
        '[1, 2, 3, 4, 5]'
        >>> sanitize_for_mlflow(None)
        'None'
        >>> sanitize_for_mlflow("a" * 600)  # Truncated
        'aaa...aaa...'
    """
    if value is None:
        return "None"

    if isinstance(value, bool):
        return str(value).lower()  # "true" or "false"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str):
        return value[:500] + "..." if len(value) > 500 else value

    if isinstance(value, (list, tuple)):
        str_repr = str(value)
        return str_repr[:500] + "..." if len(str_repr) > 500 else str_repr

    if isinstance(value, dict):
        str_repr = str(value)
        return str_repr[:500] + "..." if len(str_repr) > 500 else str_repr

    # Fallback: convert to string and truncate
    str_repr = str(value)
    return str_repr[:500] + "..." if len(str_repr) > 500 else str_repr


def merge_metadata(*metadata_dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple metadata dictionaries with conflict resolution.

    Later dictionaries override earlier ones for duplicate keys.

    Args:
        *metadata_dicts: Variable number of metadata dictionaries

    Returns:
        Merged dictionary

    Example:
        >>> merge_metadata({"a": 1, "b": 2}, {"b": 3, "c": 4})
        {'a': 1, 'b': 3, 'c': 4}
    """
    merged = {}
    for metadata in metadata_dicts:
        merged.update(metadata)
    return merged


def extract_all_metadata(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all relevant metadata for MLflow tracking.

    Combines LLM params, task metadata, routing metadata, and chain metadata
    into a single dictionary for comprehensive tracking.

    Args:
        params: Dictionary of all parameters

    Returns:
        Combined metadata dictionary

    Example:
        >>> params = {
        ...     "model": "gpt-4",
        ...     "temperature": 0.7,
        ...     "objective": "Analyze code",
        ...     "selected_agent": "analyzer"
        ... }
        >>> extract_all_metadata(params)
        {'model': 'gpt-4', 'temperature': '0.7', 'objective': 'Analyze code', ...}
    """
    return merge_metadata(
        extract_llm_params(params),
        extract_task_metadata(params),
        extract_routing_metadata(params),
        extract_chain_metadata(params)
    )
