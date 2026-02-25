"""
Unit tests for MemoStore integration with EnhancedAgenticStepProcessor.

T023: test_memo_injected_into_context_before_llm_call
T024: test_successful_task_stored_as_memo

Discovery results (2026-02-25):
  - memo_store.py         : EXISTS — MemoStore fully implemented
  - EnhancedAgenticStepProcessor wires MemoStore via run_async_with_verification()
    at lines 979-993 (inject before LLM) and 1121-1153 (store after execution)

These tests are GREEN because the MemoStore is already wired.

The tests verify:
  T023: Memo content appears in the enhanced_input passed to the parent run_async
  T024: store_memo() is called with outcome="success" after a successful step

Branch: 006-promptchain-improvements (US2)
"""

import asyncio
import os
import tempfile
import warnings
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm_runner(response_text: str = "task completed successfully"):
    """Return a coroutine that mimics a LiteLLM response with no tool calls."""
    async def llm_runner(messages, tools=None, tool_choice=None):
        mock_msg = MagicMock()
        mock_msg.content = response_text
        mock_msg.tool_calls = None
        return mock_msg
    return llm_runner


async def _noop_tool_executor(tool_call):
    return "{}"


def _build_enhanced_processor(memo_store, objective: str = "test objective"):
    """Build EnhancedAgenticStepProcessor with given memo_store, all other
    heavyweight features disabled so tests remain fast and isolated."""
    from promptchain.utils.enhanced_agentic_step_processor import (
        EnhancedAgenticStepProcessor,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        processor = EnhancedAgenticStepProcessor(
            objective=objective,
            max_internal_steps=1,
            enable_rag_verification=False,
            enable_gemini_augmentation=False,
            enable_memo_store=True,
            enable_interrupt_queue=False,
            memo_store=memo_store,
        )
    return processor


def _make_mock_memo_store(relevant_memo_text: str = "Use approach X for tasks like this"):
    """Return a MagicMock that mimics the MemoStore API."""
    from promptchain.utils.memo_store import Memo

    mock_store = MagicMock()

    # retrieve_relevant_memos returns one mock memo
    mock_memo = MagicMock(spec=Memo)
    mock_memo.task_description = "similar past task"
    mock_memo.solution = relevant_memo_text
    mock_memo.outcome = "success"
    mock_memo.relevance_score = 0.95
    mock_memo.metadata = None

    mock_store.retrieve_relevant_memos = MagicMock(return_value=[mock_memo])

    # format_memos_for_context returns a formatted string that includes the solution
    formatted_context = (
        "=== Relevant Lessons Learned from Past Tasks ===\n"
        f"  Solution: {relevant_memo_text}\n"
        "=== End of Lessons Learned ===\n"
    )
    mock_store.format_memos_for_context = MagicMock(return_value=formatted_context)

    # store_memo returns a fake memo_id
    mock_store.store_memo = MagicMock(return_value=1)

    return mock_store, formatted_context


# ---------------------------------------------------------------------------
# T023: Memo injected into context before LLM call
# ---------------------------------------------------------------------------

class TestMemoStoreIntegration:
    """Tests for MemoStore wiring into EnhancedAgenticStepProcessor."""

    @pytest.mark.asyncio
    async def test_memo_injected_into_context_before_llm_call(self):
        """
        T023 — GREEN: memo_store is already wired.

        Arrange:
          - Create a mock MemoStore that returns one relevant memo
          - Create EnhancedAgenticStepProcessor with that memo_store

        Act:
          - Call run_async_with_verification() with a spy on the parent
            run_async so we can inspect the initial_input it receives

        Assert:
          - The initial_input passed down to the parent contains the memo
            content returned by format_memos_for_context()
        """
        relevant_memo_text = "Use approach X for tasks like this"
        mock_store, formatted_context = _make_mock_memo_store(relevant_memo_text)

        processor = _build_enhanced_processor(
            memo_store=mock_store,
            objective="write a function that processes data",
        )

        # We capture what initial_input the parent's run_async receives
        captured_inputs = []
        original_run_async = processor.__class__.__bases__[0].run_async

        async def capturing_run_async(self_inner, initial_input, **kwargs):
            captured_inputs.append(initial_input)
            # Return a minimal string so downstream logic doesn't crash
            return "task completed successfully"

        with patch.object(
            processor.__class__.__bases__[0],
            "run_async",
            new=capturing_run_async,
        ):
            await processor.run_async_with_verification(
                initial_input="process my data",
                available_tools=[],
                llm_runner=_make_mock_llm_runner(),
                tool_executor=_noop_tool_executor,
                mcp_helper=None,
            )

        assert len(captured_inputs) == 1, (
            "Expected parent run_async to be called exactly once"
        )
        injected_input = captured_inputs[0]
        assert relevant_memo_text in injected_input, (
            f"Expected memo content '{relevant_memo_text}' to be present in the "
            f"enhanced_input passed to the LLM, but got:\n{injected_input!r}"
        )

    @pytest.mark.asyncio
    async def test_successful_task_stored_as_memo(self):
        """
        T024 — GREEN: store_memo is already called after successful execution.

        Arrange:
          - Create a mock MemoStore with no pre-existing relevant memos
            (so injection path is a no-op and we can isolate the store path)
          - LLM runner returns a response without "error" or "failed" text

        Act:
          - Call run_async_with_verification()

        Assert:
          - mock_store.store_memo() was called with outcome="success"
        """
        mock_store = MagicMock()
        mock_store.retrieve_relevant_memos = MagicMock(return_value=[])
        mock_store.format_memos_for_context = MagicMock(return_value="")
        mock_store.store_memo = MagicMock(return_value=42)

        processor = _build_enhanced_processor(
            memo_store=mock_store,
            objective="complete a successful task",
        )

        # Parent run_async is patched to return a clean success string
        async def mock_parent_run_async(self_inner, initial_input, **kwargs):
            return "Task completed. All steps executed successfully."

        with patch.object(
            processor.__class__.__bases__[0],
            "run_async",
            new=mock_parent_run_async,
        ):
            result = await processor.run_async_with_verification(
                initial_input="do the task",
                available_tools=[],
                llm_runner=_make_mock_llm_runner("Task completed successfully."),
                tool_executor=_noop_tool_executor,
                mcp_helper=None,
            )

        # store_memo must have been called
        mock_store.store_memo.assert_called()

        # The call must include outcome="success"
        call_kwargs_list = [c.kwargs for c in mock_store.store_memo.call_args_list]
        call_args_list = [c.args for c in mock_store.store_memo.call_args_list]

        success_calls = [
            (args, kwargs)
            for args, kwargs in zip(call_args_list, call_kwargs_list)
            if kwargs.get("outcome") == "success"
            or (len(args) >= 3 and args[2] == "success")
        ]

        assert len(success_calls) >= 1, (
            f"Expected store_memo() to be called with outcome='success', "
            f"but actual calls were: {mock_store.store_memo.call_args_list}"
        )

    @pytest.mark.asyncio
    async def test_memo_store_disabled_skips_all_memo_operations(self):
        """
        Regression guard: when enable_memo_store=False, neither retrieve nor
        store_memo should be called — verifies the feature flag is honoured.
        """
        from promptchain.utils.enhanced_agentic_step_processor import (
            EnhancedAgenticStepProcessor,
        )

        mock_store = MagicMock()
        mock_store.retrieve_relevant_memos = MagicMock(return_value=[])
        mock_store.store_memo = MagicMock(return_value=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            processor = EnhancedAgenticStepProcessor(
                objective="disabled memo store test",
                max_internal_steps=1,
                enable_rag_verification=False,
                enable_gemini_augmentation=False,
                enable_memo_store=False,
                enable_interrupt_queue=False,
                memo_store=mock_store,
            )

        async def mock_parent_run_async(self_inner, initial_input, **kwargs):
            return "completed"

        with patch.object(
            processor.__class__.__bases__[0],
            "run_async",
            new=mock_parent_run_async,
        ):
            await processor.run_async_with_verification(
                initial_input="test input",
                available_tools=[],
                llm_runner=_make_mock_llm_runner("completed"),
                tool_executor=_noop_tool_executor,
                mcp_helper=None,
            )

        mock_store.retrieve_relevant_memos.assert_not_called()
        mock_store.store_memo.assert_not_called()
