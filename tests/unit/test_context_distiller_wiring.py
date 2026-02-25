"""
Unit tests for ContextDistiller wiring into EnhancedAgenticStepProcessor.

T020: test_distiller_triggered_at_threshold
T021: test_distiller_not_triggered_below_threshold
T022: test_distiller_llm_failure_leaves_history_unchanged

Discovery results (2026-02-25):
  - context_distiller.py  : NOT FOUND - ContextDistiller class does not exist yet
  - janitor_agent.py      : NOT FOUND - JanitorAgent class does not exist yet
  - memo_store.py         : EXISTS and fully wired into EnhancedAgenticStepProcessor

These tests are RED (expected to fail) because:
  1. ContextDistiller class does not exist in promptchain/utils/context_distiller.py
  2. EnhancedAgenticStepProcessor does not accept a context_distiller parameter
  3. No distill() call path exists in the processor

Tests will turn GREEN when T027/T028/T029 implement:
  - promptchain/utils/context_distiller.py  (ContextDistiller class)
  - Wiring of context_distiller param into EnhancedAgenticStepProcessor.__init__
  - Distill trigger logic at summarize_token_threshold (default 0.7 = 70%)

Branch: 006-promptchain-improvements (US2)
"""

import asyncio
import pytest
import warnings
from unittest.mock import AsyncMock, MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm_runner(response_text: str = "step result"):
    """Return a coroutine that mimics a minimal LiteLLM response object."""
    async def llm_runner(messages, tools=None, tool_choice=None):
        mock_msg = MagicMock()
        mock_msg.content = response_text
        mock_msg.tool_calls = None
        return mock_msg
    return llm_runner


async def _noop_tool_executor(tool_call):
    return "{}"


def _make_available_tools():
    return []


def _build_processor_with_mock_distiller(mock_distiller, max_context_tokens=1000):
    """
    Attempt to build an EnhancedAgenticStepProcessor that accepts a
    context_distiller parameter.

    This will raise ImportError / TypeError when ContextDistiller does not
    exist yet — that is the expected RED state.
    """
    from promptchain.utils.enhanced_agentic_step_processor import (
        EnhancedAgenticStepProcessor,
    )

    # Suppress the DeprecationWarning about 'minimal' history mode during
    # processor construction so test output is clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        processor = EnhancedAgenticStepProcessor(
            objective="test objective for distiller wiring",
            max_internal_steps=1,
            max_context_tokens=max_context_tokens,
            enable_rag_verification=False,
            enable_gemini_augmentation=False,
            enable_memo_store=False,
            enable_interrupt_queue=False,
            # This parameter does NOT exist yet — wire-up target for T027
            context_distiller=mock_distiller,
        )
    return processor


def _fill_conversation_history(processor, fraction: float):
    """
    Artificially fill processor.conversation_history so that
    current token usage = fraction * max_context_tokens.

    Uses a simple character-count approximation consistent with the base
    processor's token counting (1 token ≈ 4 chars).
    """
    max_tokens = processor.max_context_tokens or 1000
    target_tokens = int(max_tokens * fraction)
    # 1 token ≈ 4 chars; build a single large message to hit target
    filler = "x" * (target_tokens * 4)
    processor.conversation_history = [
        {"role": "user", "content": filler}
    ]


# ---------------------------------------------------------------------------
# T020: distiller triggered at threshold (75% >= 70% default threshold)
# ---------------------------------------------------------------------------

class TestContextDistillerWiring:
    """Tests for ContextDistiller wiring into EnhancedAgenticStepProcessor."""

    @pytest.mark.asyncio
    async def test_distiller_triggered_at_threshold(self):
        """
        T020 — RED until T027 implements ContextDistiller wiring.

        Arrange:
          - Mock ContextDistiller with async distill() coroutine
          - Construct processor with context_distiller=mock_distiller
          - Fill conversation history to 75% of max_context_tokens

        Act:
          - Run one agentic step

        Assert:
          - mock_distiller.distill() was called at least once
        """
        mock_distiller = MagicMock()
        mock_distiller.distill = AsyncMock(return_value="compressed history")
        mock_distiller.should_distill = MagicMock(return_value=True)

        # Build processor — will raise TypeError until parameter is wired
        processor = _build_processor_with_mock_distiller(
            mock_distiller, max_context_tokens=1000
        )

        # Fill to 75% token capacity (above default 70% threshold)
        _fill_conversation_history(processor, fraction=0.75)

        await processor.run_async(
            initial_input="run one step",
            available_tools=_make_available_tools(),
            llm_runner=_make_mock_llm_runner("done"),
            tool_executor=_noop_tool_executor,
        )

        mock_distiller.distill.assert_called()

    @pytest.mark.asyncio
    async def test_distiller_not_triggered_below_threshold(self):
        """
        T021 — RED until T027 implements ContextDistiller wiring.

        Arrange:
          - Mock ContextDistiller
          - Fill conversation history to only 50% of max_context_tokens

        Act:
          - Run one agentic step

        Assert:
          - mock_distiller.distill() was NOT called
        """
        mock_distiller = MagicMock()
        mock_distiller.distill = AsyncMock(return_value="compressed history")
        mock_distiller.should_distill = MagicMock(return_value=False)

        processor = _build_processor_with_mock_distiller(
            mock_distiller, max_context_tokens=1000
        )

        # Fill to 50% token capacity (below default 70% threshold)
        _fill_conversation_history(processor, fraction=0.50)

        await processor.run_async(
            initial_input="run one step",
            available_tools=_make_available_tools(),
            llm_runner=_make_mock_llm_runner("done"),
            tool_executor=_noop_tool_executor,
        )

        mock_distiller.distill.assert_not_called()

    @pytest.mark.asyncio
    async def test_distiller_llm_failure_leaves_history_unchanged(self):
        """
        T022 — RED until T027 implements ContextDistiller wiring.

        Arrange:
          - Mock ContextDistiller whose distill() raises an exception
          - Fill conversation history to 75% (above threshold)
          - Snapshot history content before the step

        Act:
          - Run one agentic step (should not crash due to exception in distill)

        Assert:
          - Processor does not raise an exception
          - The user message originally in history is still present afterwards
            (distill failure must leave history in a valid, non-corrupted state)
        """
        mock_distiller = MagicMock()
        mock_distiller.distill = AsyncMock(
            side_effect=RuntimeError("Simulated LLM failure during distillation")
        )
        mock_distiller.should_distill = MagicMock(return_value=True)

        processor = _build_processor_with_mock_distiller(
            mock_distiller, max_context_tokens=1000
        )

        # Fill to 75% (above threshold)
        _fill_conversation_history(processor, fraction=0.75)

        # Snapshot the original message content before the step
        original_content = processor.conversation_history[0]["content"]

        # Must NOT raise even though distill() throws
        await processor.run_async(
            initial_input="run one step",
            available_tools=_make_available_tools(),
            llm_runner=_make_mock_llm_runner("done"),
            tool_executor=_noop_tool_executor,
        )

        # History must still contain the original user message unchanged
        assert any(
            entry.get("content") == original_content
            for entry in processor.conversation_history
        ), (
            "Expected original conversation history entry to survive a failed "
            "distillation, but it was missing or altered."
        )
