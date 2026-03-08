"""
PromptChain Observability Package

Provides MLflow-based tracking for LLM calls, task execution, routing decisions,
and session management. Gracefully degrades when MLflow is not installed.

Public API:
    Decorators:
        - track_llm_call: Track LLM API calls with metrics
        - track_task: Track task execution
        - track_routing: Track routing decisions
        - track_session: Track session-level metrics

    Observer Plugin:
        - MLflowObserver: CallbackManager plugin for automatic event tracking

    Lifecycle:
        - init_mlflow: Initialize MLflow tracking
        - shutdown_mlflow: Cleanup MLflow resources
"""

# Always import MLflowObserver (has built-in graceful degradation)
from .mlflow_observer import MLflowObserver

try:
    # Import decorators and lifecycle functions
    from .decorators import (init_mlflow, shutdown_mlflow, track_llm_call,
                             track_routing, track_session, track_task)

    _MLFLOW_AVAILABLE = True
except ImportError:
    # Graceful fallback: Provide no-op stubs when MLflow unavailable or modules not yet created
    _MLFLOW_AVAILABLE = False

    def track_llm_call(*args, **kwargs):  # type: ignore[misc]
        """No-op stub for track_llm_call decorator"""

        def decorator(func):
            return func

        return decorator(args[0]) if args and callable(args[0]) else decorator

    def track_task(*args, **kwargs):  # type: ignore[misc]
        """No-op stub for track_task decorator"""

        def decorator(func):
            return func

        return decorator(args[0]) if args and callable(args[0]) else decorator

    def track_routing(*args, **kwargs):  # type: ignore[misc]
        """No-op stub for track_routing decorator"""

        def decorator(func):
            return func

        return decorator(args[0]) if args and callable(args[0]) else decorator

    def track_session(*args, **kwargs):  # type: ignore[misc]
        """No-op stub for track_session decorator"""

        def decorator(func):
            return func

        return decorator(args[0]) if args and callable(args[0]) else decorator

    def init_mlflow(*args, **kwargs):  # type: ignore[misc]
        """No-op stub for init_mlflow"""
        pass

    def shutdown_mlflow(*args, **kwargs):  # type: ignore[misc]
        """No-op stub for shutdown_mlflow"""
        pass


__all__ = [
    "track_llm_call",
    "track_task",
    "track_routing",
    "track_session",
    "init_mlflow",
    "shutdown_mlflow",
    "MLflowObserver",
]
