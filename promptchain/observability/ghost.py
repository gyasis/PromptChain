"""Ghost decorator pattern for zero-overhead observability when tracking is disabled.

This module implements the ghost decorator pattern which provides:
- Import-time check (not runtime) to determine if tracking is enabled
- <0.1% performance overhead when tracking is disabled
- Graceful degradation when MLflow is not installed

The key insight: check _ENABLED once at module import, not on every function call.

BUG-022 fix: If you change environment variables after import, call reload_config()
to re-evaluate the enabled state. This allows dynamic configuration while maintaining
performance benefits of import-time evaluation by default.
"""

from typing import Callable, Any, Optional, TypeVar
import functools

from .config import is_enabled

# Import-time check - evaluated ONCE when module loads
_ENABLED = is_enabled()

# Check MLflow availability at import time
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    # If MLflow not installed, force tracking to be disabled
    _ENABLED = False

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


def reload_config() -> bool:
    """Re-evaluate tracking configuration from environment variables.

    BUG-022 fix: Call this function if you change PROMPTCHAIN_MLFLOW_ENABLED
    or install MLflow after the module was imported. This allows dynamic
    configuration while maintaining the performance benefit of import-time
    evaluation by default.

    Returns:
        True if tracking is now enabled, False otherwise

    Example:
        ```python
        import os
        from promptchain.observability import ghost

        # Change environment after import
        os.environ['PROMPTCHAIN_MLFLOW_ENABLED'] = 'true'

        # Reload to pick up the change
        ghost.reload_config()

        # Now tracking is enabled
        assert ghost.is_tracking_enabled() == True
        ```
    """
    global _ENABLED, _MLFLOW_AVAILABLE

    # Re-check MLflow availability
    try:
        import mlflow
        _MLFLOW_AVAILABLE = True
    except ImportError:
        _MLFLOW_AVAILABLE = False

    # Re-evaluate enabled state from environment
    _ENABLED = is_enabled()

    # If MLflow not available, force disabled regardless of config
    if not _MLFLOW_AVAILABLE:
        _ENABLED = False

    return _ENABLED


def make_ghost_decorator() -> Callable[[F], F]:
    """Create a ghost decorator that returns the original function unchanged.

    This is used when tracking is disabled to provide zero overhead.
    The decorator literally does nothing - it returns the function as-is.

    Returns:
        A decorator that returns the original function unchanged
    """
    def ghost(func: F) -> F:
        return func
    return ghost


def conditional_decorator(
    tracking_decorator: Callable[[F], F]
) -> Callable[[F], F]:
    """Return either the tracking decorator or a ghost decorator based on _ENABLED.

    This function is evaluated at import time when decorators are applied,
    not at runtime when the decorated function is called. This ensures
    zero overhead when tracking is disabled.

    Args:
        tracking_decorator: The actual tracking decorator to use when enabled

    Returns:
        Either tracking_decorator (if enabled) or ghost decorator (if disabled)

    Example:
        ```python
        @conditional_decorator(track_chain_execution)
        def my_function():
            pass  # If disabled, this is equivalent to no decorator at all
        ```
    """
    if _ENABLED:
        return tracking_decorator
    else:
        return make_ghost_decorator()


def is_tracking_enabled() -> bool:
    """Check if observability tracking is currently enabled.

    This function returns the import-time cached value, not a fresh check.
    Use this when you need to conditionally execute tracking code.

    Returns:
        True if tracking is enabled, False otherwise
    """
    return _ENABLED


def is_mlflow_available() -> bool:
    """Check if MLflow is installed and available.

    Returns:
        True if MLflow is installed, False otherwise
    """
    return _MLFLOW_AVAILABLE


def require_enabled(func: F) -> F:
    """Decorator that makes a function a no-op when tracking is disabled.

    This is useful for utility functions that should only execute when
    tracking is enabled, avoiding any overhead when disabled.

    Args:
        func: Function to conditionally execute

    Returns:
        Either the original function (if enabled) or a no-op version

    Example:
        ```python
        @require_enabled
        def log_metrics(metrics: dict):
            # This entire function is skipped when tracking disabled
            mlflow.log_metrics(metrics)
        ```
    """
    if not _ENABLED:
        @functools.wraps(func)
        def noop(*args, **kwargs):
            return None
        return noop  # type: ignore
    return func
