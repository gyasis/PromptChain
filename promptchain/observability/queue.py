"""
Background Queue for Non-Blocking MLflow Operations

Provides thread-safe queuing of MLflow API calls to prevent TUI blocking.
Targets <5ms overhead and processes 100+ metrics per second.

Architecture:
- Single worker thread processes queue items
- Queue-based producer-consumer pattern
- Graceful shutdown with flush support
- Synchronous fallback when background logging disabled
"""

import queue
import threading
import logging
import time
from typing import Callable, Any, Dict, Optional

from .mlflow_adapter import (
    log_metric,
    log_param,
    log_params,
    set_tag,
    start_run,
    end_run,
    log_artifact
)
from .config import use_background_logging

logger = logging.getLogger(__name__)


class BackgroundLogger:
    """Thread-safe background queue for MLflow operations.

    Attributes:
        queue: Thread-safe queue for operations (maxsize=1000)
        worker: Background worker thread
        shutdown_flag: Event to signal worker shutdown
        enabled: Whether background logging is active
    """

    def __init__(self, maxsize: int = 1000):
        """Initialize background logger.

        Args:
            maxsize: Maximum queue size (default: 1000)
        """
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.shutdown_flag = threading.Event()
        self.enabled = use_background_logging()
        self.worker: Optional[threading.Thread] = None
        self._dropped_count = 0

        if self.enabled:
            self.worker = threading.Thread(
                target=self._worker,
                daemon=True,
                name="MLflowBackgroundLogger"
            )
            self.worker.start()
            logger.info("Background logger started with queue size %d", maxsize)

    def _worker(self) -> None:
        """Worker thread that processes queued operations.

        Runs continuously until shutdown_flag is set.
        Handles exceptions to prevent thread termination.
        """
        while not self.shutdown_flag.is_set():
            try:
                # Wait for operation with timeout to check shutdown flag
                operation, args, kwargs = self.queue.get(timeout=0.1)

                try:
                    operation(*args, **kwargs)
                except Exception as e:
                    logger.error(
                        "Error executing queued operation %s: %s",
                        operation.__name__,
                        e,
                        exc_info=True
                    )
                finally:
                    self.queue.task_done()

            except queue.Empty:
                # Timeout - check shutdown flag and continue
                continue

        logger.info("Background logger worker shutting down")

    def submit(self, operation: Callable, *args, **kwargs) -> None:
        """Submit operation to background queue.

        Args:
            operation: Callable to execute in background
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Note:
            Falls back to synchronous execution if background logging disabled.
            Drops oldest item if queue is full.
        """
        if not self.enabled:
            # Synchronous fallback
            operation(*args, **kwargs)
            return

        try:
            self.queue.put_nowait((operation, args, kwargs))
        except queue.Full:
            # Queue full - log warning and drop this operation
            self._dropped_count += 1
            logger.warning(
                "Background queue full, dropping operation %s (total dropped: %d)",
                operation.__name__,
                self._dropped_count
            )

    def flush(self, timeout: Optional[float] = 10.0) -> bool:
        """Wait for all queued operations to complete.

        Fixes Issue #4: Queue Flush Timeout Ignored in MLflow Observability.
        Now properly respects timeout parameter to prevent indefinite hangs.

        Args:
            timeout: Maximum seconds to wait (None = infinite)

        Returns:
            True if queue emptied, False if timeout occurred
        """
        if not self.enabled:
            return True

        if timeout is None:
            # Infinite timeout - use original blocking join
            try:
                self.queue.join()
                return True
            except Exception as e:
                logger.error("Error flushing queue: %s", e)
                return False

        # Timeout-aware flush implementation
        start_time = time.time()

        try:
            while not self.queue.empty():
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    remaining_items = self.queue.qsize()
                    logger.warning(
                        f"Queue flush timed out after {timeout}s "
                        f"({remaining_items} items remain)"
                    )
                    return False

                # Wait for remaining time or 0.1s, whichever is less
                remaining = timeout - elapsed
                time.sleep(min(0.1, remaining))

            # Queue is empty - success
            logger.debug("Queue flushed successfully")
            return True

        except Exception as e:
            logger.error("Error flushing queue: %s", e)
            return False

    def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shutdown background worker.

        Args:
            timeout: Maximum seconds to wait for worker to finish
        """
        if not self.enabled or self.worker is None:
            return

        logger.info("Shutting down background logger...")

        # Signal shutdown and wait for worker
        self.shutdown_flag.set()
        self.worker.join(timeout=timeout)

        if self.worker.is_alive():
            logger.warning("Background worker did not shutdown within timeout")
        else:
            logger.info("Background logger shutdown complete")

        # Log stats
        if self._dropped_count > 0:
            logger.warning("Total operations dropped: %d", self._dropped_count)

    @property
    def qsize(self) -> int:
        """Current queue size."""
        return self.queue.qsize()


# Singleton instance
_background_logger: BackgroundLogger = BackgroundLogger()


# Convenience functions that delegate to singleton

def queue_log_metric(key: str, value: float, step: Optional[int] = None) -> None:
    """Queue metric logging operation."""
    _background_logger.submit(log_metric, key, value, step)


def queue_log_param(key: str, value: Any) -> None:
    """Queue parameter logging operation."""
    _background_logger.submit(log_param, key, value)


def queue_log_params(params: Dict[str, Any]) -> None:
    """Queue batch parameter logging operation."""
    _background_logger.submit(log_params, params)


def queue_set_tag(key: str, value: str) -> None:
    """Queue tag setting operation."""
    _background_logger.submit(set_tag, key, value)


def flush_queue(timeout: Optional[float] = 10.0) -> bool:
    """Flush all queued operations."""
    return _background_logger.flush(timeout)


def shutdown_background_logger(timeout: float = 5.0) -> None:
    """Shutdown background logger."""
    _background_logger.shutdown(timeout)


def get_queue_size() -> int:
    """Get current queue size."""
    return _background_logger.qsize
