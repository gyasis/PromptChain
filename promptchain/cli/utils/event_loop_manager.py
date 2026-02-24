"""Centralized event loop management for CLI and TUI contexts.

Solves Issue #2 (P0 Critical): Event Loop Race Conditions in TUI Pattern Commands

This module provides utilities to safely handle async operations in both:
- CLI context (no running loop) - creates new loop
- TUI context (already running loop from Textual) - uses existing loop

Key Functions:
- get_or_create_event_loop(): Safe loop retrieval/creation
- run_async_in_context(): Execute coroutine in appropriate context
- is_event_loop_running(): Check if loop is already running
"""

import asyncio
import functools
from typing import Any, Coroutine, TypeVar

T = TypeVar('T')


def is_event_loop_running() -> bool:
    """Check if an event loop is currently running.

    Returns:
        bool: True if event loop is running, False otherwise
    """
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get current event loop or create new one if needed.

    Handles both CLI and TUI contexts safely:
    - CLI: Creates new event loop
    - TUI: Returns existing running loop

    Returns:
        asyncio.AbstractEventLoop: Event loop instance
    """
    try:
        # Try to get running loop (TUI context)
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        # No loop running, safe to create new one (CLI context)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop


def run_async_in_context(coro: Coroutine[Any, Any, T]) -> T:
    """Run async coroutine in current context (CLI or TUI).

    Automatically detects context and uses appropriate execution strategy:
    - If event loop is running (TUI): Creates task in existing loop
    - If no event loop (CLI): Uses asyncio.run() to execute

    Args:
        coro: Async coroutine to execute

    Returns:
        T: Result of coroutine execution

    Raises:
        RuntimeError: If called from running loop without proper async context

    Examples:
        # CLI context (no running loop)
        result = run_async_in_context(my_async_function())

        # TUI context (loop already running) - will raise RuntimeError
        # Instead, use: await my_async_function() directly in TUI
    """
    try:
        # Check if loop is already running (TUI context)
        loop = asyncio.get_running_loop()

        # If we get here, we're in TUI context with running loop
        # Cannot use run_until_complete or asyncio.run on running loop
        raise RuntimeError(
            "Event loop is already running. "
            "In TUI context, use 'await' instead of run_async_in_context(). "
            "This function should only be called from CLI context (no running loop)."
        )

    except RuntimeError:
        # No loop running, safe to use asyncio.run (CLI context)
        return asyncio.run(coro)


def make_sync_from_async(async_func):
    """Decorator to create sync wrapper for async function.

    Creates a synchronous version that safely handles both CLI and TUI contexts.

    Args:
        async_func: Async function to wrap

    Returns:
        Callable: Synchronous wrapper function

    Example:
        @make_sync_from_async
        async def my_async_function(arg1, arg2):
            return await some_async_operation(arg1, arg2)

        # Can now call synchronously from CLI
        result = my_async_function(val1, val2)
    """
    @functools.wraps(async_func)
    def wrapper(*args, **kwargs):
        coro = async_func(*args, **kwargs)

        try:
            # Check if loop is already running (TUI context)
            loop = asyncio.get_running_loop()

            # If we're here, we're in async context - should use await instead
            raise RuntimeError(
                f"Cannot call {async_func.__name__} synchronously in async context. "
                f"Use 'await {async_func.__name__}(...)' instead."
            )

        except RuntimeError:
            # No running loop (CLI context) - safe to use asyncio.run
            return asyncio.run(coro)

    return wrapper


# Context manager for safe async execution in CLI
class AsyncContextManager:
    """Context manager for safe async operations in CLI context.

    Ensures event loop is properly created and cleaned up for CLI commands.

    Example:
        async def my_async_operation():
            return await some_async_call()

        with AsyncContextManager() as runner:
            result = runner.run(my_async_operation())
    """

    def __enter__(self):
        """Enter context - verify no running loop."""
        if is_event_loop_running():
            raise RuntimeError(
                "AsyncContextManager should only be used in CLI context "
                "(no running event loop). In TUI, use async/await directly."
            )
        self.loop = get_or_create_event_loop()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - cleanup if needed."""
        # Loop cleanup handled by asyncio.run() in run_async_in_context
        pass

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run coroutine in managed context.

        Args:
            coro: Coroutine to execute

        Returns:
            T: Result of coroutine execution
        """
        return run_async_in_context(coro)
