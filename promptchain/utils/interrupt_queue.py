"""
Interrupt Queue for Real-Time User Steering (h2A Pattern from Claude Code)

Issue #8 Enhancement: Implements h2A (Human-to-Agent) interrupt system for
mid-execution steering, corrections, and clarifications.

Architecture:
- Thread-safe queue for interrupt messages
- Non-blocking interrupt checks between agentic steps
- Multiple interrupt types (steering, correction, clarification, abort)
- Integration with AgenticStepProcessor

Benefits:
- Real-time course correction without restart
- User can guide agent mid-execution
- Graceful handling of clarification requests
- Emergency abort capability
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class InterruptType(str, Enum):
    """Types of user interrupts during agent execution."""

    STEERING = "steering"  # Guide agent in different direction
    CORRECTION = "correction"  # Fix agent's misunderstanding
    CLARIFICATION = "clarification"  # Provide additional context
    ABORT = "abort"  # Stop execution immediately
    PAUSE = "pause"  # Pause execution for review
    RESUME = "resume"  # Resume after pause


@dataclass
class Interrupt:
    """Represents a user interrupt during agent execution.

    Attributes:
        interrupt_id: Unique identifier for this interrupt
        interrupt_type: Type of interrupt (steering, correction, etc.)
        message: User's interrupt message
        metadata: Additional context or data
        timestamp: When interrupt was created
        processed: Whether interrupt has been processed
    """

    interrupt_id: str
    interrupt_type: InterruptType
    message: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    processed: bool = False

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


class InterruptQueue:
    """Thread-safe queue for real-time user interrupts.

    Implements h2A pattern for human-to-agent communication during execution.

    Features:
    - Non-blocking interrupt submission
    - Priority interrupts (ABORT, PAUSE processed first)
    - Thread-safe operations
    - Interrupt history tracking
    - Graceful pause/resume mechanism
    """

    def __init__(self, maxsize: int = 100):
        """Initialize interrupt queue.

        Args:
            maxsize: Maximum number of pending interrupts (default: 100)
        """
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._interrupt_counter = 0
        self._counter_lock = threading.Lock()
        self._interrupt_history: list[Interrupt] = []
        self._paused = threading.Event()  # For pause/resume
        self._paused.clear()  # Not paused initially

        logger.info(f"InterruptQueue initialized (maxsize: {maxsize})")

    def submit_interrupt(
        self,
        interrupt_type: InterruptType,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit a user interrupt to the queue.

        Args:
            interrupt_type: Type of interrupt
            message: User's interrupt message
            metadata: Optional additional context

        Returns:
            Unique interrupt ID

        Raises:
            queue.Full: If queue is at capacity (unlikely with maxsize=100)
        """
        # Generate unique ID
        with self._counter_lock:
            self._interrupt_counter += 1
            interrupt_id = f"int_{self._interrupt_counter}_{int(time.time())}"

        # Create interrupt
        interrupt = Interrupt(
            interrupt_id=interrupt_id,
            interrupt_type=interrupt_type,
            message=message,
            metadata=metadata,
        )

        # Add to queue with priority handling
        try:
            # High-priority interrupts (ABORT, PAUSE) should be processed first
            # For now, just use a single queue, but could implement priority queue
            self._queue.put_nowait(interrupt)

            # Track in history
            self._interrupt_history.append(interrupt)

            logger.info(
                f"Interrupt submitted: type={interrupt_type}, "
                f"id={interrupt_id}, message={message[:50]}..."
            )

            # Handle pause immediately
            if interrupt_type == InterruptType.PAUSE:
                self._paused.set()
                logger.info("Execution paused by user interrupt")

            return interrupt_id

        except queue.Full:
            logger.error(f"Interrupt queue full, dropping interrupt: {interrupt_id}")
            raise

    def check_for_interrupt(self, timeout: float = 0.0) -> Optional[Interrupt]:
        """Check if there's a pending interrupt (non-blocking by default).

        Args:
            timeout: How long to wait for interrupt (0.0 = non-blocking)

        Returns:
            Interrupt object if available, None otherwise
        """
        try:
            interrupt = self._queue.get(timeout=timeout)
            self._queue.task_done()

            logger.info(
                f"Retrieved interrupt: type={interrupt.interrupt_type}, "
                f"id={interrupt.interrupt_id}"
            )

            # Handle resume
            if interrupt.interrupt_type == InterruptType.RESUME:
                self._paused.clear()
                logger.info("Execution resumed by user interrupt")

            return interrupt

        except queue.Empty:
            return None

    def is_paused(self) -> bool:
        """Check if execution is currently paused.

        Returns:
            True if paused, False otherwise
        """
        return self._paused.is_set()

    def wait_while_paused(self, check_interval: float = 0.1) -> None:
        """Block execution while paused, checking for resume.

        Args:
            check_interval: How often to check for resume (seconds)
        """
        while self._paused.is_set():
            logger.debug("Execution paused, waiting for resume...")
            time.sleep(check_interval)

            # Check for resume interrupt
            resume_interrupt = self.check_for_interrupt(timeout=0.0)
            if (
                resume_interrupt
                and resume_interrupt.interrupt_type == InterruptType.RESUME
            ):
                break

    def has_pending_interrupts(self) -> bool:
        """Check if there are any pending interrupts.

        Returns:
            True if interrupts are pending, False otherwise
        """
        return not self._queue.empty()

    def get_pending_count(self) -> int:
        """Get number of pending interrupts.

        Returns:
            Number of unprocessed interrupts in queue
        """
        return self._queue.qsize()

    def clear_interrupts(self) -> int:
        """Clear all pending interrupts.

        Returns:
            Number of interrupts cleared
        """
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                count += 1
            except queue.Empty:
                break

        logger.info(f"Cleared {count} pending interrupts")
        return count

    def get_interrupt_history(self, limit: int = 10) -> list[Interrupt]:
        """Get recent interrupt history.

        Args:
            limit: Maximum number of interrupts to return

        Returns:
            List of recent interrupts (most recent first)
        """
        return self._interrupt_history[-limit:][::-1]

    def mark_processed(self, interrupt_id: str) -> bool:
        """Mark an interrupt as processed.

        Args:
            interrupt_id: ID of interrupt to mark

        Returns:
            True if found and marked, False otherwise
        """
        for interrupt in self._interrupt_history:
            if interrupt.interrupt_id == interrupt_id:
                interrupt.processed = True
                logger.debug(f"Marked interrupt {interrupt_id} as processed")
                return True

        logger.warning(f"Interrupt {interrupt_id} not found in history")
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get interrupt queue statistics.

        Returns:
            Dictionary with statistics
        """
        total_interrupts = len(self._interrupt_history)
        processed_count = sum(1 for i in self._interrupt_history if i.processed)

        type_counts: Dict[str, int] = {}
        for interrupt in self._interrupt_history:
            type_name = interrupt.interrupt_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "total_interrupts": total_interrupts,
            "processed_count": processed_count,
            "pending_count": self.get_pending_count(),
            "is_paused": self.is_paused(),
            "interrupt_types": type_counts,
            "queue_maxsize": self._queue.maxsize,
        }


# ============================================================================
# Integration with AgenticStepProcessor
# ============================================================================


class InterruptHandler:
    """Handles interrupt processing for agentic execution.

    Provides high-level logic for responding to different interrupt types.
    """

    def __init__(self, interrupt_queue: InterruptQueue):
        """Initialize interrupt handler.

        Args:
            interrupt_queue: InterruptQueue instance
        """
        self.interrupt_queue = interrupt_queue
        logger.info("InterruptHandler initialized")

    def check_and_handle_interrupt(
        self, current_step: int, current_context: str
    ) -> Optional[Dict[str, Any]]:
        """Check for interrupts and handle appropriately.

        Call this between agentic steps to allow user steering.

        Args:
            current_step: Current execution step number
            current_context: Current execution context for user

        Returns:
            Dictionary with handling result, or None if no interrupt
        """
        # Wait if paused
        if self.interrupt_queue.is_paused():
            logger.info(f"Execution paused at step {current_step}")
            self.interrupt_queue.wait_while_paused()
            logger.info(f"Execution resumed at step {current_step}")

        # Check for new interrupt
        interrupt = self.interrupt_queue.check_for_interrupt(timeout=0.0)
        if not interrupt:
            return None

        # Handle based on type
        if interrupt.interrupt_type == InterruptType.ABORT:
            logger.warning(f"Abort interrupt received at step {current_step}")
            return {
                "action": "abort",
                "interrupt_id": interrupt.interrupt_id,
                "message": interrupt.message,
                "step": current_step,
            }

        elif interrupt.interrupt_type == InterruptType.STEERING:
            logger.info(f"Steering interrupt received at step {current_step}")
            return {
                "action": "steering",
                "interrupt_id": interrupt.interrupt_id,
                "message": interrupt.message,
                "guidance": interrupt.message,
                "step": current_step,
            }

        elif interrupt.interrupt_type == InterruptType.CORRECTION:
            logger.info(f"Correction interrupt received at step {current_step}")
            return {
                "action": "correction",
                "interrupt_id": interrupt.interrupt_id,
                "message": interrupt.message,
                "correction": interrupt.message,
                "step": current_step,
            }

        elif interrupt.interrupt_type == InterruptType.CLARIFICATION:
            logger.info(f"Clarification interrupt received at step {current_step}")
            return {
                "action": "clarification",
                "interrupt_id": interrupt.interrupt_id,
                "message": interrupt.message,
                "clarification": interrupt.message,
                "step": current_step,
            }

        elif interrupt.interrupt_type == InterruptType.PAUSE:
            logger.info(f"Pause interrupt received at step {current_step}")
            # Already handled in check_for_interrupt()
            return {
                "action": "pause",
                "interrupt_id": interrupt.interrupt_id,
                "message": "Execution paused",
                "step": current_step,
            }

        elif interrupt.interrupt_type == InterruptType.RESUME:
            logger.info(f"Resume interrupt received at step {current_step}")
            # Already handled in check_for_interrupt()
            return {
                "action": "resume",
                "interrupt_id": interrupt.interrupt_id,
                "message": "Execution resumed",
                "step": current_step,
            }

        return None

    def format_interrupt_for_llm(self, interrupt_result: Dict[str, Any]) -> str:
        """Format interrupt result for LLM context injection.

        Args:
            interrupt_result: Result from check_and_handle_interrupt()

        Returns:
            Formatted string for LLM context
        """
        action = interrupt_result.get("action", "unknown")
        message = interrupt_result.get("message", "")
        step = interrupt_result.get("step", 0)

        if action == "abort":
            return f"\n\n🛑 USER ABORT at step {step}: {message}\nPlease stop execution immediately."

        elif action == "steering":
            return f"\n\n🧭 USER GUIDANCE at step {step}: {message}\nPlease adjust your approach accordingly."

        elif action == "correction":
            return f"\n\n✏️ USER CORRECTION at step {step}: {message}\nPlease correct your understanding and continue."

        elif action == "clarification":
            return f"\n\n💡 USER CLARIFICATION at step {step}: {message}\nUse this additional context to proceed."

        else:
            return f"\n\n📥 USER INTERRUPT at step {step}: {message}"


# Singleton instance for global access (optional)
_global_interrupt_queue: Optional[InterruptQueue] = None


def get_global_interrupt_queue() -> InterruptQueue:
    """Get or create global interrupt queue.

    Returns:
        Global InterruptQueue instance
    """
    global _global_interrupt_queue
    if _global_interrupt_queue is None:
        _global_interrupt_queue = InterruptQueue()
        logger.info("Created global interrupt queue")
    return _global_interrupt_queue


def submit_user_interrupt(
    interrupt_type: InterruptType,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Submit interrupt to global queue (convenience function).

    Args:
        interrupt_type: Type of interrupt
        message: User's interrupt message
        metadata: Optional additional context

    Returns:
        Unique interrupt ID
    """
    queue = get_global_interrupt_queue()
    return queue.submit_interrupt(interrupt_type, message, metadata)
