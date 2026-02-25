"""
Integration tests for 006-promptchain-improvements: Real-Time Steering (US3).

Tests T033-T036 (Wave 5):
  - T033: test_micro_checkpoint_saved_after_tool_call
  - T034: test_rewind_to_checkpoint_restores_state
  - T035: test_global_override_replaces_prompt
  - T036: test_tui_interrupt_command_enqueues_without_blocking

Branch: 006-promptchain-improvements
Success Criteria: SC-005 (interrupt ack within 2s)

Design notes:
  - T033/T034 are RED: _micro_checkpoints, _save_micro_checkpoint, and
    rewind_to_last_checkpoint are not yet implemented in
    EnhancedAgenticStepProcessor (they are planned in T039/FR-013).
  - T035 is RED: send_global_override() and PubSubBus topic routing are not
    yet added to MessageBus (planned in T039/FR-014).
  - T036 is RED: handle_interrupt_command() is not yet implemented in the
    TUI app (planned in T040/FR-012) — however, the underlying
    InterruptQueue.submit_interrupt() IS non-blocking, so we can verify
    the latency guarantee using the queue directly if the TUI method
    is absent, and fail with a clear message if the method is missing.
"""

import asyncio
import copy
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptchain.utils.interrupt_queue import (
    InterruptQueue,
    InterruptType,
    get_global_interrupt_queue,
)
from promptchain.utils.checkpoint_manager import CheckpointManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response_with_tool(tool_name: str = "tool_a") -> MagicMock:
    tc = MagicMock()
    tc.function.name = tool_name
    msg = MagicMock()
    msg.tool_calls = [tc]
    msg.content = ""
    return msg


def _make_final_answer_response(text: str = "done") -> MagicMock:
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = f"FINAL_ANSWER: {text}"
    return msg


def _make_available_tools(name: str = "tool_a") -> List[Dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"A tool named {name}",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


# ---------------------------------------------------------------------------
# T033 — micro-checkpoint saved after tool call
# ---------------------------------------------------------------------------

class TestMicroCheckpoints:
    """Tests for micro-checkpoint save/rewind functionality (FR-013)."""

    def test_micro_checkpoint_saved_after_tool_call(self):
        """
        T033 (RED): After a successful tool call inside run_async_with_verification,
        the processor must store a micro-checkpoint with tool_call_index=0 in
        self._micro_checkpoints.

        RED because _save_micro_checkpoint and _micro_checkpoints are not yet
        implemented in EnhancedAgenticStepProcessor.
        """
        from promptchain.utils.enhanced_agentic_step_processor import (
            EnhancedAgenticStepProcessor,
        )

        processor = EnhancedAgenticStepProcessor(
            objective="checkpoint test",
            max_internal_steps=3,
            enable_rag_verification=False,
            enable_gemini_augmentation=False,
            enable_memo_store=False,
            enable_interrupt_queue=False,
        )

        # Verify the attribute exists (will raise AttributeError if not implemented)
        assert hasattr(processor, "_micro_checkpoints"), (
            "_micro_checkpoints attribute not found on EnhancedAgenticStepProcessor. "
            "FR-013 micro-checkpoint feature is not yet implemented."
        )

        call_count = [0]

        async def llm_runner(messages, tools=None, tool_choice=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_llm_response_with_tool("tool_a")
            return _make_final_answer_response("done")

        async def tool_executor(tool_call):
            return "tool result"

        asyncio.run(
            processor.run_async_with_verification(
                initial_input="run it",
                available_tools=_make_available_tools("tool_a"),
                llm_runner=llm_runner,
                tool_executor=tool_executor,
            )
        )

        checkpoints = processor._micro_checkpoints
        assert len(checkpoints) >= 1, (
            "No micro-checkpoints were saved after a tool call. "
            "_save_micro_checkpoint was not called during tool execution."
        )

        # The first checkpoint must record tool_call_index=0
        first_cp = checkpoints[0]
        tool_call_index = (
            first_cp.get("tool_call_index", None)
            if isinstance(first_cp, dict)
            else getattr(first_cp, "tool_call_index", None)
        )
        assert tool_call_index == 0, (
            f"Expected tool_call_index=0 on first micro-checkpoint, got: {tool_call_index}"
        )

    def test_rewind_to_checkpoint_restores_state(self):
        """
        T034 (RED): After saving a micro-checkpoint and then modifying conversation
        history, rewind_to_last_checkpoint() must restore conversation_history to
        the state at the time of the checkpoint.

        RED because rewind_to_last_checkpoint is not yet implemented.
        """
        from promptchain.utils.enhanced_agentic_step_processor import (
            EnhancedAgenticStepProcessor,
        )

        processor = EnhancedAgenticStepProcessor(
            objective="rewind test",
            max_internal_steps=3,
            enable_rag_verification=False,
            enable_gemini_augmentation=False,
            enable_memo_store=False,
            enable_interrupt_queue=False,
        )

        # Verify method exists before testing behaviour
        assert hasattr(processor, "rewind_to_last_checkpoint"), (
            "rewind_to_last_checkpoint() method not found on "
            "EnhancedAgenticStepProcessor. FR-013 rewind is not yet implemented."
        )
        assert hasattr(processor, "_save_micro_checkpoint"), (
            "_save_micro_checkpoint() method not found on "
            "EnhancedAgenticStepProcessor. FR-013 checkpoint saving is not yet implemented."
        )

        # Arrange: seed a known conversation state and save a checkpoint
        original_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        processor.conversation_history = copy.deepcopy(original_history)

        snapshot_id = f"snap_{int(time.time())}"
        processor._save_micro_checkpoint(
            checkpoint_id=snapshot_id,
            step_number=1,
            tool_call_index=0,
            conversation_snapshot=copy.deepcopy(processor.conversation_history),
        )

        # Modify conversation history (simulating further execution)
        processor.conversation_history.append(
            {"role": "user", "content": "new message after checkpoint"}
        )
        assert len(processor.conversation_history) == 3

        # Act: rewind
        rewound = processor.rewind_to_last_checkpoint()

        # Assert: rewound must be truthy (not None) and state restored
        assert rewound is not None, (
            "rewind_to_last_checkpoint() returned None — no checkpoint to restore from."
        )
        assert processor.conversation_history == original_history, (
            f"After rewind, conversation_history was not restored to checkpoint state. "
            f"Expected {original_history}, got {processor.conversation_history}"
        )


# ---------------------------------------------------------------------------
# T035 — global override replaces prompt
# ---------------------------------------------------------------------------

class TestGlobalOverride:
    """Tests for global prompt override via MessageBus topic (FR-014)."""

    def test_global_override_replaces_prompt(self):
        """
        T035 (RED): Subscribing an AgenticStepProcessor to "agent.global_override"
        topic and publishing an override message should update the processor's active
        prompt at the next thought cycle.

        RED because send_global_override() and the PubSubBus topic system are not
        yet added to MessageBus / AgenticStepProcessor (FR-014, T039).

        The test uses whatever publish/subscribe API is available on MessageBus.
        If neither send_global_override nor a subscribe method exists, it fails
        with a clear message indicating what needs to be implemented.
        """
        from promptchain.cli.communication.message_bus import MessageBus
        from promptchain.utils.enhanced_agentic_step_processor import (
            EnhancedAgenticStepProcessor,
        )

        bus = MessageBus(session_id="test-override-session")
        processor = EnhancedAgenticStepProcessor(
            objective="original objective",
            max_internal_steps=3,
            enable_rag_verification=False,
            enable_gemini_augmentation=False,
            enable_memo_store=False,
            enable_interrupt_queue=False,
        )

        # Verify send_global_override exists on MessageBus
        assert hasattr(bus, "send_global_override"), (
            "MessageBus.send_global_override() not found. "
            "FR-014 global override is not yet implemented in message_bus.py."
        )

        # Verify processor can subscribe to global override topic
        assert hasattr(processor, "_handle_override") or hasattr(
            processor, "subscribe_to_override"
        ), (
            "AgenticStepProcessor has no _handle_override or subscribe_to_override "
            "method. FR-014 processor subscription is not yet implemented."
        )

        # Subscribe processor to override topic
        if hasattr(processor, "subscribe_to_override"):
            processor.subscribe_to_override(bus)
        else:
            # Direct subscription via bus if bus has subscribe method
            assert hasattr(bus, "subscribe"), (
                "MessageBus has no subscribe() method. Cannot wire up global override."
            )
            asyncio.run(
                bus.subscribe("agent.global_override", processor._handle_override)
            )

        # Publish override
        asyncio.run(
            bus.send_global_override(
                new_prompt="updated objective: do something else",
                sender_id="test-tui"
            )
        )

        # The processor's active prompt (objective) must be updated
        active_prompt = getattr(
            processor, "_active_prompt",
            getattr(processor, "objective", None)
        )
        assert active_prompt == "updated objective: do something else", (
            f"Active prompt was not updated after global override. "
            f"Got: {active_prompt!r}"
        )


# ---------------------------------------------------------------------------
# T036 — TUI interrupt command enqueues without blocking
# ---------------------------------------------------------------------------

class TestTUIInterruptCommand:
    """Tests for non-blocking TUI interrupt command handling (FR-012)."""

    def test_tui_interrupt_command_enqueues_without_blocking(self):
        """
        T036: handle_interrupt_command("steering", "msg") must return in < 10ms.

        If the method exists on the TUI app, we call it and time it.
        If it does not exist, the test is RED with a clear message.

        The underlying guarantee (submit_interrupt is non-blocking) is verified
        by also directly timing InterruptQueue.submit_interrupt().
        """
        # First verify the underlying queue is non-blocking
        iq = InterruptQueue()
        start = time.perf_counter()
        iq.submit_interrupt(InterruptType.STEERING, "test message")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10, (
            f"InterruptQueue.submit_interrupt() took {elapsed_ms:.2f}ms — "
            f"expected < 10ms for non-blocking enqueue"
        )

        # Now verify the TUI method exists and is also non-blocking
        try:
            # Import without triggering Textual TUI rendering
            from promptchain.cli.tui.app import PromptChainApp  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"TUI app not importable in headless environment: {exc}")

        # Check for handle_interrupt_command on the class (not instance, to avoid
        # starting the full Textual app)
        from promptchain.cli.tui.app import PromptChainApp

        assert hasattr(PromptChainApp, "handle_interrupt_command"), (
            "PromptChainApp.handle_interrupt_command() not found. "
            "FR-012 TUI interrupt wiring is not yet implemented in app.py."
        )

        # Create a minimal mock of the app to call the method without a real TUI
        # We patch asyncio to avoid the Textual event loop
        app_mock = MagicMock(spec=PromptChainApp)
        app_mock.handle_interrupt_command = PromptChainApp.handle_interrupt_command.__get__(
            app_mock, PromptChainApp
        )

        # Attach a real InterruptQueue so the method can actually enqueue
        app_mock.interrupt_queue = InterruptQueue()

        start = time.perf_counter()
        app_mock.handle_interrupt_command("steering", "focus on X")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10, (
            f"handle_interrupt_command took {elapsed_ms:.2f}ms — "
            f"must return in < 10ms (non-blocking, FR-012)"
        )

        # Verify the interrupt was actually enqueued
        assert app_mock.interrupt_queue.has_pending_interrupts(), (
            "handle_interrupt_command did not enqueue an interrupt — "
            "the queue is empty after the call"
        )


# ---------------------------------------------------------------------------
# SC-005: Interrupt acknowledgment latency <= 2s
# ---------------------------------------------------------------------------

class TestInterruptLatency:
    """SC-005 validation: interrupt acknowledgment latency <= 2s."""

    def test_interrupt_ack_latency_under_2s(self):
        """
        SC-005: From the moment submit_interrupt() is called to the moment
        check_for_interrupt() returns the interrupt, elapsed time must be <= 2s.

        This measures the round-trip through InterruptQueue in isolation.
        In production the bound is dominated by the polling interval between
        thought cycles; here we verify the queue itself adds negligible overhead.
        """
        iq = InterruptQueue()
        start = time.perf_counter()
        iq.submit_interrupt(InterruptType.STEERING, "urgent redirect")
        interrupt = iq.check_for_interrupt(timeout=0.0)
        elapsed_s = time.perf_counter() - start

        assert interrupt is not None, "Interrupt was not immediately retrievable"
        assert elapsed_s <= 2.0, (
            f"Interrupt round-trip took {elapsed_s:.4f}s — "
            f"SC-005 requires <= 2s acknowledgment"
        )
        assert interrupt.interrupt_type == InterruptType.STEERING
        assert interrupt.message == "urgent redirect"
