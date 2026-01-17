# execution_callback.py
"""Callback protocol for PromptChain execution event handling.

This module defines the callback interface for handling execution events,
supporting both synchronous and asynchronous callbacks with optional filtering.
"""

from typing import Protocol, Optional, Set, Union, Callable, Awaitable
from .execution_events import ExecutionEvent, ExecutionEventType


class ExecutionCallback(Protocol):
    """Protocol for execution event callbacks.

    Callbacks can be either synchronous or asynchronous functions that
    receive ExecutionEvent objects. They can optionally filter which
    event types they want to receive.

    Example synchronous callback:
        def my_callback(event: ExecutionEvent) -> None:
            print(f"Event: {event.event_type.name}")

    Example asynchronous callback:
        async def my_async_callback(event: ExecutionEvent) -> None:
            await log_to_database(event)

    Example filtered callback:
        def error_callback(event: ExecutionEvent) -> None:
            if event.event_type in error_callback.event_filter:
                handle_error(event)

        error_callback.event_filter = {
            ExecutionEventType.CHAIN_ERROR,
            ExecutionEventType.STEP_ERROR,
            ExecutionEventType.MODEL_CALL_ERROR
        }
    """

    def __call__(self, event: ExecutionEvent) -> Optional[Awaitable[None]]:
        """Handle an execution event.

        Args:
            event: The execution event to handle

        Returns:
            None for sync callbacks, Awaitable[None] for async callbacks
        """
        ...


# Type aliases for callback functions
SyncCallback = Callable[[ExecutionEvent], None]
AsyncCallback = Callable[[ExecutionEvent], Awaitable[None]]
CallbackFunction = Union[SyncCallback, AsyncCallback]


class FilteredCallback:
    """Wrapper for callbacks with event type filtering.

    This class wraps a callback function and only invokes it for events
    matching the specified event types.

    Attributes:
        callback: The underlying callback function
        event_filter: Set of event types to handle (None = all events)
    """

    def __init__(
        self,
        callback: CallbackFunction,
        event_filter: Optional[Set[ExecutionEventType]] = None
    ):
        """Initialize filtered callback.

        Args:
            callback: Callback function to wrap
            event_filter: Optional set of event types to filter for.
                         If None, all events are passed through.
        """
        self.callback = callback
        self.event_filter = event_filter

    def should_handle(self, event_type: ExecutionEventType) -> bool:
        """Check if this callback should handle the given event type.

        Args:
            event_type: The event type to check

        Returns:
            True if the callback should handle this event type
        """
        if self.event_filter is None:
            return True
        return event_type in self.event_filter

    async def __call__(self, event: ExecutionEvent) -> None:
        """Handle an event if it matches the filter.

        Args:
            event: The execution event to potentially handle
        """
        if not self.should_handle(event.event_type):
            return

        # Check if callback is async
        import asyncio
        import inspect
        import functools

        try:
            if inspect.iscoroutinefunction(self.callback):
                await self.callback(event)
            else:
                # Run sync callback in executor to avoid blocking
                loop = asyncio.get_running_loop()
                wrapped_callback = functools.partial(self.callback, event)
                await loop.run_in_executor(None, wrapped_callback)
        except Exception as e:
            # Log error but don't stop other callbacks from executing
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"Error in callback execution: {e}",
                exc_info=True
            )


class CallbackManager:
    """Manager for execution callbacks.

    This class manages a collection of callbacks and handles event
    distribution to all registered callbacks.
    """

    def __init__(self):
        """Initialize callback manager."""
        self.callbacks: list[FilteredCallback] = []
        self._emitting = False  # Recursion guard

    def register(
        self,
        callback: CallbackFunction,
        event_filter: Optional[Set[ExecutionEventType]] = None
    ) -> None:
        """Register a callback with optional event filtering.

        Args:
            callback: Callback function to register
            event_filter: Optional set of event types to filter for
        """
        filtered_callback = FilteredCallback(callback, event_filter)
        self.callbacks.append(filtered_callback)

    def unregister(self, callback: CallbackFunction) -> bool:
        """Unregister a callback.

        Args:
            callback: Callback function to unregister

        Returns:
            True if callback was found and removed, False otherwise
        """
        for i, filtered_callback in enumerate(self.callbacks):
            if filtered_callback.callback == callback:
                self.callbacks.pop(i)
                return True
        return False

    def clear(self) -> None:
        """Remove all registered callbacks."""
        self.callbacks.clear()

    async def emit(self, event: ExecutionEvent) -> None:
        """Emit an event to all registered callbacks.

        Args:
            event: The event to emit
        """
        # Prevent infinite recursion - if we're already emitting, skip
        if self._emitting:
            return

        import asyncio

        try:
            self._emitting = True

            # Collect all callback tasks
            tasks = []
            for callback in self.callbacks:
                if callback.should_handle(event.event_type):
                    tasks.append(callback(event))

            # Run all callbacks concurrently
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._emitting = False

    def emit_sync(self, event: ExecutionEvent) -> None:
        """Emit an event synchronously (for sync contexts).

        This method wraps the async emit for use in synchronous contexts.
        WARNING: This should NOT be called from within a running event loop.
        Use the async emit() method instead in async contexts.

        Args:
            event: The event to emit

        Raises:
            RuntimeError: If called from within a running event loop
        """
        # Prevent infinite recursion - if we're already emitting, skip
        if self._emitting:
            return

        import asyncio

        try:
            # Check if there's a running event loop
            asyncio.get_running_loop()
            # If we get here, there's a running loop - this is an error
            raise RuntimeError(
                "Cannot call emit_sync from within a running event loop. "
                "Use await callback_manager.emit(event) instead."
            )
        except RuntimeError as e:
            if "Cannot call emit_sync" in str(e):
                raise  # Re-raise our custom error
            # No running loop, safe to use asyncio.run
            asyncio.run(self.emit(event))

    def has_callbacks(self) -> bool:
        """Check if any callbacks are registered.

        Returns:
            True if callbacks are registered, False otherwise
        """
        return len(self.callbacks) > 0
