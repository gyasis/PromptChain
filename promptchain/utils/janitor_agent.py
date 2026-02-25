"""
JanitorAgent: Background async task that monitors conversation history
and triggers compression when token usage exceeds threshold.

FR-010: Non-blocking background monitoring via asyncio.Task.

Design notes
------------
The threshold check is performed inside JanitorAgent itself — it computes
    current_tokens / max_tokens >= compression_threshold
using history_manager attributes directly (_current_token_count and
max_tokens).  This keeps the agent self-contained and avoids coupling to
any particular ContextDistiller.should_distill() implementation, which
lets mock distillers in tests control only distill() behaviour without
accidentally short-circuiting the threshold logic.
"""

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JanitorAgent:
    """Background agent that monitors history token usage and triggers
    distillation when compression_threshold is exceeded.

    The agent runs an asyncio.Task that wakes up every ``check_interval``
    seconds, computes the current token-usage ratio, and calls
    ``distiller.distill(history_manager)`` when the ratio is at or above
    ``compression_threshold``.

    Attributes
    ----------
    distiller : Any
        Object with an async ``distill(history_manager)`` coroutine.
    history_manager : Any
        Object exposing ``_current_token_count`` (int) and
        ``max_tokens`` (int | None) attributes.
    check_interval : float
        Seconds between monitoring ticks.  Defaults to 5.0.
    compression_threshold : float
        Fraction of max_tokens (0.0–1.0) that triggers distillation.
        Defaults to 0.7 (70 %).
    _task : asyncio.Task | None
        The background monitoring task.  Exposed so callers can inspect
        cancellation status; also accessible as ``_monitor_task`` and
        ``background_task`` aliases for compatibility.

    Usage
    -----
    >>> janitor = JanitorAgent(distiller, history_manager,
    ...                        check_interval=1.0,
    ...                        compression_threshold=0.7)
    >>> await janitor.start()
    >>> # ... runs in background ...
    >>> await janitor.stop()
    """

    def __init__(
        self,
        distiller: Any,
        history_manager: Any,
        check_interval: float = 5.0,
        compression_threshold: float = 0.7,
    ) -> None:
        """Initialise JanitorAgent.

        Parameters
        ----------
        distiller:
            An object with an async ``distill(history_manager)``
            coroutine method.
        history_manager:
            An ExecutionHistoryManager (or compatible mock) that exposes
            ``_current_token_count`` and ``max_tokens``.
        check_interval:
            Polling interval in seconds.
        compression_threshold:
            Trigger distillation when token usage ratio reaches this
            fraction of the configured ``max_tokens`` limit.
        """
        self.distiller = distiller
        self.history_manager = history_manager
        self.check_interval = check_interval
        self.compression_threshold = compression_threshold
        self._task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Aliases — expose the same task object under alternative attribute
    # names so that test code can use any of the accepted names.
    # ------------------------------------------------------------------

    @property
    def _monitor_task(self) -> Optional[asyncio.Task]:  # type: ignore[type-arg]
        """Alias for _task (compatibility)."""
        return self._task

    @property
    def background_task(self) -> Optional[asyncio.Task]:  # type: ignore[type-arg]
        """Alias for _task (compatibility)."""
        return self._task

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background monitoring loop.

        Creates an asyncio.Task that runs :meth:`_monitor_loop`.  Calling
        ``start()`` while already running is a no-op (the existing task
        keeps running).
        """
        if self._task is not None and not self._task.done():
            logger.debug("JanitorAgent.start() called but task is already running.")
            return
        self._task = asyncio.create_task(self._monitor_loop())
        logger.debug("JanitorAgent started (check_interval=%.2fs, threshold=%.0f%%).",
                     self.check_interval, self.compression_threshold * 100)

    async def stop(self) -> None:
        """Cancel and await the background monitoring task.

        Safe to call even if the agent was never started or has already
        stopped.  After this method returns, ``_task.done()`` is ``True``.
        """
        if self._task is None or self._task.done():
            self._task = None
            return

        self._task.cancel()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        finally:
            # _task reference is kept so callers can inspect .done()
            logger.debug("JanitorAgent stopped.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _exceeds_threshold(self) -> bool:
        """Return True when token-usage ratio is at or above the threshold.

        Computes ``current_tokens / max_tokens`` from the history manager's
        attributes directly.  Returns False when ``max_tokens`` is None or
        zero (no limit configured).
        """
        max_tokens: Optional[int] = getattr(self.history_manager, "max_tokens", None)
        if not max_tokens:  # None or 0 → no limit
            return False

        current_tokens: int = getattr(
            self.history_manager, "_current_token_count", 0
        )
        ratio = current_tokens / max_tokens
        return ratio >= self.compression_threshold

    async def _monitor_loop(self) -> None:
        """Background loop: sleep, check threshold, trigger distillation.

        Runs indefinitely until cancelled via :meth:`stop`.  Any exception
        other than ``asyncio.CancelledError`` is logged as a warning and
        the loop continues — this prevents a transient error from silently
        killing the background agent.
        """
        while True:
            try:
                await asyncio.sleep(self.check_interval)

                if self._exceeds_threshold():
                    logger.debug(
                        "JanitorAgent: token threshold exceeded "
                        "(%.0f%% threshold). Triggering distillation.",
                        self.compression_threshold * 100,
                    )
                    await self.distiller.distill(self.history_manager)
                else:
                    logger.debug(
                        "JanitorAgent: token usage below threshold (%.0f%%). Skipping.",
                        self.compression_threshold * 100,
                    )

            except asyncio.CancelledError:
                logger.debug("JanitorAgent._monitor_loop cancelled.")
                break
            except Exception as exc:  # pragma: no cover
                logger.warning("JanitorAgent._monitor_loop error: %s", exc)
