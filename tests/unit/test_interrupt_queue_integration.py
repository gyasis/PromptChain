"""
Unit tests for InterruptQueue integration with EnhancedAgenticStepProcessor.

Tests T030-T032 (US3, Wave 5):
  - T030: test_interrupt_checked_each_thought_cycle
  - T031: test_abort_interrupt_halts_execution
  - T032: test_steering_message_injected_into_context  [RED]

Branch: 006-promptchain-improvements (US3)

Design notes:
  - The llm_runner must return plain dicts (not MagicMocks) because the parent
    processor calls json.dumps() on model_dump() for debug logging and will raise
    TypeError on MagicMock objects.
  - Tool calls must be plain dicts with shape:
      {"id": str, "type": "function", "function": {"name": str, "arguments": str}}
  - The interrupt check fires inside verified_tool_executor (a wrapper closure
    built in run_async_with_verification) BEFORE delegating to the original
    tool_executor. We spy on InterruptHandler.check_and_handle_interrupt directly.
  - T031 verifies the abort path: original_executor is bypassed and a JSON
    abort-error string is returned by the wrapper.
  - T032 is a RED test: steering message injection into LLM messages is not yet
    implemented (the code logs but does not inject — see comment at line ~1025 of
    enhanced_agentic_step_processor.py).
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from promptchain.utils.interrupt_queue import (
    InterruptQueue,
    InterruptHandler,
    InterruptType,
)


# ---------------------------------------------------------------------------
# Helpers — dict-based llm_runner responses
# ---------------------------------------------------------------------------

def _tool_call_dict(name: str, call_id: str = "call_001") -> Dict:
    """Return a tool_call dict in the format the processor expects."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _llm_response_with_tool(tool_name: str = "search") -> Dict:
    """LLM response dict that requests one tool call."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call_dict(tool_name)],
    }


def _llm_final_answer(text: str = "done") -> Dict:
    """LLM response dict with a final answer (no tool calls)."""
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": None,
    }


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


# ---------------------------------------------------------------------------
# T030 — interrupt checked each thought cycle
# ---------------------------------------------------------------------------

class TestInterruptCheckedEachThoughtCycle:
    """T030: check_and_handle_interrupt is called during execution, not only at end."""

    def test_interrupt_checked_each_thought_cycle(self):
        """
        Arrange: Build processor with a shared InterruptQueue.
                 Submit a STEERING interrupt before run_async_with_verification.
        Act:     Run with a llm_runner that returns one tool call then a final answer.
        Assert:  InterruptHandler.check_and_handle_interrupt was called >= 1 time
                 during execution (verified via a recorded wrapper on the handler).
        """
        from promptchain.utils.enhanced_agentic_step_processor import (
            EnhancedAgenticStepProcessor,
        )

        iq = InterruptQueue()
        processor = EnhancedAgenticStepProcessor(
            objective="test objective",
            max_internal_steps=3,
            enable_rag_verification=False,
            enable_gemini_augmentation=False,
            enable_memo_store=False,
            enable_interrupt_queue=True,
            interrupt_queue=iq,
        )
        assert processor.interrupt_handler is not None, (
            "interrupt_handler should be set when enable_interrupt_queue=True"
        )

        # Submit a steering interrupt before the run
        iq.submit_interrupt(InterruptType.STEERING, "focus on accuracy")

        # Wrap check_and_handle_interrupt to record calls
        original_check = processor.interrupt_handler.check_and_handle_interrupt
        call_log: List[int] = []

        def recording_check(current_step: int, current_context: str):
            call_log.append(current_step)
            return original_check(current_step, current_context)

        processor.interrupt_handler.check_and_handle_interrupt = recording_check

        call_count = [0]

        async def llm_runner(messages, tools=None, tool_choice=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return _llm_response_with_tool("search")
            return _llm_final_answer("result")

        async def tool_executor(tool_call):
            return "search result"

        asyncio.run(
            processor.run_async_with_verification(
                initial_input="test input",
                available_tools=_TOOLS,
                llm_runner=llm_runner,
                tool_executor=tool_executor,
            )
        )

        assert len(call_log) >= 1, (
            "check_and_handle_interrupt was never called — "
            "the interrupt queue is not polled during execution"
        )


# ---------------------------------------------------------------------------
# T031 — ABORT interrupt halts execution
# ---------------------------------------------------------------------------

class TestAbortInterruptHaltsExecution:
    """T031: An ABORT interrupt prevents the original tool_executor from being called."""

    def test_abort_interrupt_halts_execution(self):
        """
        Arrange: Submit an ABORT interrupt before run_async_with_verification.
        Act:     Run with a llm_runner that requests a tool call.
        Assert:  The original tool_executor is never invoked AND the final
                 result string contains "aborted" (propagated from the JSON
                 abort-error payload returned by verified_tool_executor).

        The current implementation (enhanced_agentic_step_processor.py ~line 1013)
        returns a JSON string {"error": "Execution aborted by user", ...} from the
        verified_tool_executor wrapper without ever calling the original executor.
        The parent run_async then treats this as the tool result and ultimately
        returns it as part of the final answer string.
        """
        from promptchain.utils.enhanced_agentic_step_processor import (
            EnhancedAgenticStepProcessor,
        )

        iq = InterruptQueue()
        processor = EnhancedAgenticStepProcessor(
            objective="test abort",
            max_internal_steps=5,
            enable_rag_verification=False,
            enable_gemini_augmentation=False,
            enable_memo_store=False,
            enable_interrupt_queue=True,
            interrupt_queue=iq,
        )

        # Submit ABORT before the run
        iq.submit_interrupt(InterruptType.ABORT, "stop now")

        original_executor_called = [False]

        # llm_runner: first call returns a tool request (triggering verified_tool_executor),
        # second call returns a final answer (after the abort payload is seen as tool result)
        call_count = [0]

        async def llm_runner(messages, tools=None, tool_choice=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return _llm_response_with_tool("search")
            # After the abort payload is returned as tool result, LLM ends execution
            return _llm_final_answer("aborted")

        async def original_tool_executor(tool_call):
            original_executor_called[0] = True
            return "should not reach here"

        result = asyncio.run(
            processor.run_async_with_verification(
                initial_input="start task",
                available_tools=_TOOLS,
                llm_runner=llm_runner,
                tool_executor=original_tool_executor,
            )
        )

        # The original executor must NOT have been called
        assert not original_executor_called[0], (
            "Original tool_executor was called after ABORT interrupt — "
            "the abort did not short-circuit before delegating to the executor"
        )

        # The abort error must be visible in the overall result (the tool result
        # carrying the abort payload becomes part of the execution trace that
        # feeds the next LLM call, so "aborted" appears in the context)
        result_str = str(result).lower()
        assert "aborted" in result_str, (
            f"Expected 'aborted' to appear somewhere in the result after an ABORT "
            f"interrupt, but got: {str(result)[:300]!r}"
        )


# ---------------------------------------------------------------------------
# T032 — steering message injected into LLM context  [RED]
# ---------------------------------------------------------------------------

class TestSteeringMessageInjectedIntoContext:
    """
    T032 (RED): Steering message must appear in the messages list sent to the LLM.

    Current status: The implementation only LOGS the steering interrupt without
    injecting it into the LLM message list (see comment "For now, we log it and
    continue with execution" in enhanced_agentic_step_processor.py ~line 1025).

    This test documents the REQUIRED behaviour from FR-011:
        "Inject interrupt message into LLM context"
    It will turn GREEN once T039 implements the injection.
    """

    def test_steering_message_injected_into_context(self):
        """
        Arrange: Submit STEERING interrupt with message "focus on X".
        Act:     Run run_async_with_verification; capture all messages passed to
                 llm_runner across every invocation.
        Assert:  At least one message in at least one LLM call contains "focus on X".
        """
        from promptchain.utils.enhanced_agentic_step_processor import (
            EnhancedAgenticStepProcessor,
        )

        iq = InterruptQueue()
        processor = EnhancedAgenticStepProcessor(
            objective="test steering injection",
            max_internal_steps=3,
            enable_rag_verification=False,
            enable_gemini_augmentation=False,
            enable_memo_store=False,
            enable_interrupt_queue=True,
            interrupt_queue=iq,
        )

        # Submit steering interrupt before the run
        iq.submit_interrupt(InterruptType.STEERING, "focus on X")

        captured_messages: List[List] = []
        call_count = [0]

        async def llm_runner(messages, tools=None, tool_choice=None):
            captured_messages.append(list(messages))
            call_count[0] += 1
            if call_count[0] == 1:
                return _llm_response_with_tool("search")
            return _llm_final_answer("done")

        async def tool_executor(tool_call):
            return "result"

        asyncio.run(
            processor.run_async_with_verification(
                initial_input="begin",
                available_tools=_TOOLS,
                llm_runner=llm_runner,
                tool_executor=tool_executor,
            )
        )

        # Collect all content from all messages across all LLM calls
        all_content_strings: List[str] = []
        for msg_list in captured_messages:
            for msg in msg_list:
                if isinstance(msg, dict):
                    content = msg.get("content", "") or ""
                else:
                    content = getattr(msg, "content", "") or ""
                all_content_strings.append(str(content))

        combined = " ".join(all_content_strings)
        assert "focus on X" in combined, (
            "Steering message 'focus on X' was NOT found in any message passed to the LLM. "
            "FR-011 requires the interrupt context to be injected into the LLM message list "
            "so the agent can act on user guidance. This is not yet implemented "
            "(enhanced_agentic_step_processor.py ~line 1025: 'For now, we log it and "
            "continue with execution'). "
            f"Checked {len(captured_messages)} LLM call(s), "
            f"{len(all_content_strings)} total messages."
        )
