"""
Unit tests for JanitorAgent background monitoring.

T025: test_janitor_compresses_at_threshold
T026: test_janitor_stop_cancels_task

Discovery results (2026-02-25):
  - janitor_agent.py      : NOT FOUND — JanitorAgent does not exist yet
  - context_distiller.py  : NOT FOUND — ContextDistiller does not exist yet

These tests are RED (expected to fail) because:
  1. promptchain/utils/janitor_agent.py does not exist
  2. JanitorAgent class with start() / stop() / _monitor_loop() does not exist

Expected interface (to be implemented in T028/T029):

    class JanitorAgent:
        def __init__(
            self,
            distiller: ContextDistiller,
            history_manager: ExecutionHistoryManager,
            check_interval: float = 5.0,
            compression_threshold: float = 0.7,
        ): ...

        async def start(self) -> None:
            \"\"\"Launch background monitoring loop.\"\"\"

        async def stop(self) -> None:
            \"\"\"Cancel background task and await its completion.\"\"\"

        async def _monitor_loop(self) -> None:
            \"\"\"
            Periodically checks token usage ratio:
              if current_tokens / max_tokens >= compression_threshold:
                  await self.distiller.distill(self.history_manager)
            \"\"\"

Tests will turn GREEN once T028/T029 implement the above.

Branch: 006-promptchain-improvements (US2)
"""

import asyncio
import warnings
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Import guard — collected as ERRORS (not skips) so CI stays RED until impl
# ---------------------------------------------------------------------------

def _import_janitor_agent():
    """
    Attempt to import JanitorAgent.  Raises ImportError if not implemented yet.
    Tests use this in a try/except to produce clean FAIL messages.
    """
    from promptchain.utils.janitor_agent import JanitorAgent  # noqa: F401
    return JanitorAgent


def _import_context_distiller():
    """Attempt to import ContextDistiller."""
    from promptchain.utils.context_distiller import ContextDistiller  # noqa: F401
    return ContextDistiller


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_mock_history_manager(
    current_tokens: int = 600,
    max_tokens: int = 1000,
):
    """Return a mock ExecutionHistoryManager with token-usage attributes."""
    hm = MagicMock()
    hm.max_tokens = max_tokens
    hm._current_token_count = current_tokens
    return hm


def _make_mock_distiller():
    """Return a mock ContextDistiller with an async distill() method."""
    distiller = MagicMock()
    distiller.distill = AsyncMock(return_value="compressed")
    distiller.should_distill = MagicMock(return_value=True)
    return distiller


# ---------------------------------------------------------------------------
# T025: JanitorAgent compresses history when usage exceeds threshold
# ---------------------------------------------------------------------------

class TestJanitorAgent:
    """Tests for JanitorAgent background monitoring and compression."""

    @pytest.mark.asyncio
    async def test_janitor_compresses_at_threshold(self):
        """
        T025 — RED until T028/T029 implement JanitorAgent.

        Arrange:
          - compression_threshold=0.5
          - history filled to 60% of max_tokens (above threshold)
          - check_interval=0.05 seconds (fast for testing)

        Act:
          - Start JanitorAgent
          - Wait 2 × check_interval (0.1 s) to allow at least one monitor tick
          - Stop JanitorAgent

        Assert:
          - mock_distiller.distill() was called at least once
        """
        JanitorAgent = _import_janitor_agent()  # raises ImportError if missing

        history_manager = _make_mock_history_manager(
            current_tokens=600, max_tokens=1000  # 60% usage
        )
        mock_distiller = _make_mock_distiller()

        janitor = JanitorAgent(
            distiller=mock_distiller,
            history_manager=history_manager,
            check_interval=0.05,        # 50 ms ticks
            compression_threshold=0.5,  # trigger at 50%
        )

        await janitor.start()
        # Wait for 2 full check cycles
        await asyncio.sleep(0.12)
        await janitor.stop()

        mock_distiller.distill.assert_called()

    @pytest.mark.asyncio
    async def test_janitor_stop_cancels_task(self):
        """
        T026 — RED until T028/T029 implement JanitorAgent.

        Arrange:
          - JanitorAgent with a long check_interval (it will not fire naturally)

        Act:
          - Start the janitor (launches background asyncio task)
          - Immediately call stop()
          - Wait up to 5 seconds for cancellation to complete

        Assert:
          - The background asyncio.Task is done (cancelled or finished)
            within 5 seconds of stop() returning
        """
        JanitorAgent = _import_janitor_agent()  # raises ImportError if missing

        history_manager = _make_mock_history_manager(
            current_tokens=100, max_tokens=1000  # below threshold — no fire
        )
        mock_distiller = _make_mock_distiller()

        janitor = JanitorAgent(
            distiller=mock_distiller,
            history_manager=history_manager,
            check_interval=60.0,        # very long — must be cancelled explicitly
            compression_threshold=0.7,
        )

        await janitor.start()

        # The janitor must expose its background task so we can inspect it.
        # Accepted attribute names: _task, _monitor_task, background_task
        task = (
            getattr(janitor, "_task", None)
            or getattr(janitor, "_monitor_task", None)
            or getattr(janitor, "background_task", None)
        )

        assert task is not None, (
            "JanitorAgent must expose its background asyncio.Task as one of: "
            "_task, _monitor_task, or background_task"
        )
        assert not task.done(), (
            "Background task should still be running before stop() is called"
        )

        await janitor.stop()

        # Give the event loop a moment to propagate cancellation
        deadline = asyncio.get_event_loop().time() + 5.0
        while not task.done() and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)

        assert task.done(), (
            "Background asyncio.Task was not cancelled within 5 seconds of "
            "JanitorAgent.stop() returning"
        )

    @pytest.mark.asyncio
    async def test_janitor_does_not_compress_below_threshold(self):
        """
        Complementary guard: when usage is below compression_threshold,
        distill() must NOT be called during the monitor window.
        """
        JanitorAgent = _import_janitor_agent()

        history_manager = _make_mock_history_manager(
            current_tokens=300, max_tokens=1000  # 30% usage
        )
        mock_distiller = _make_mock_distiller()

        janitor = JanitorAgent(
            distiller=mock_distiller,
            history_manager=history_manager,
            check_interval=0.05,        # fast ticks
            compression_threshold=0.7,  # trigger only at 70%
        )

        await janitor.start()
        await asyncio.sleep(0.12)  # 2+ cycles at 50 ms
        await janitor.stop()

        mock_distiller.distill.assert_not_called()
