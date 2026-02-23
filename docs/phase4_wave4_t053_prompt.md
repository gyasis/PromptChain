# T053: Reasoning Step Logging to ExecutionHistoryManager

## Objective
Integrate AgenticStepProcessor reasoning steps into ExecutionHistoryManager for comprehensive conversation history tracking and token-aware truncation.

## Context Files
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/execution_history_manager.py` (extend this)
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agentic_step_processor.py` (log source)
- `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agent_chain.py` (integration point)

## Requirements

### Part 1: Add Reasoning Entry Types to ExecutionHistoryManager

**Extend `promptchain/utils/execution_history_manager.py`**:

```python
from typing import Literal, Optional, Dict, Any
from dataclasses import dataclass, field
import time


# Add new entry types for agentic reasoning
EntryType = Literal[
    "user_input",
    "agent_output",
    "tool_call",
    "tool_result",
    "error",
    "system",
    "reasoning_start",      # NEW: AgenticStepProcessor started
    "reasoning_step",       # NEW: Individual reasoning step
    "reasoning_complete",   # NEW: Reasoning objective achieved
    "reasoning_failed"      # NEW: Reasoning failed/exhausted
]


@dataclass
class HistoryEntry:
    """Extended with reasoning metadata."""
    entry_type: EntryType
    content: str
    source: str
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0
    metadata: Optional[Dict[str, Any]] = None  # NEW: For reasoning context

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ExecutionHistoryManager:
    """Extended with reasoning step tracking."""

    def add_reasoning_start(
        self,
        objective: str,
        max_steps: int,
        source: str = "agentic_processor"
    ):
        """Log start of agentic reasoning.

        Args:
            objective: Reasoning objective
            max_steps: Maximum internal steps
            source: Source identifier
        """
        content = f"Starting multi-hop reasoning: {objective}"
        self.add_entry(
            entry_type="reasoning_start",
            content=content,
            source=source,
            metadata={
                "objective": objective,
                "max_steps": max_steps,
                "reasoning_session_id": self._generate_reasoning_session_id()
            }
        )

    def add_reasoning_step(
        self,
        step_num: int,
        activity: str,
        step_output: Optional[str] = None,
        tool_calls: Optional[list] = None,
        source: str = "agentic_processor",
        session_id: Optional[str] = None
    ):
        """Log individual reasoning step.

        Args:
            step_num: Step number (1-indexed)
            activity: Brief description of step activity
            step_output: Full LLM output for this step (optional)
            tool_calls: List of tool calls made in this step (optional)
            source: Source identifier
            session_id: Reasoning session ID for grouping steps
        """
        content = f"Step {step_num}: {activity}"

        metadata = {
            "step_num": step_num,
            "activity": activity,
            "reasoning_session_id": session_id or self._get_latest_reasoning_session_id(),
            "has_tool_calls": bool(tool_calls),
            "tool_count": len(tool_calls) if tool_calls else 0
        }

        # Optionally include full output (can be large)
        if step_output:
            metadata["full_output"] = step_output[:500]  # Truncate for metadata

        if tool_calls:
            metadata["tool_calls"] = [
                {"name": tc.get("name"), "args": str(tc.get("args", {})[:100])}
                for tc in tool_calls[:3]  # Limit to 3 tools
            ]

        self.add_entry(
            entry_type="reasoning_step",
            content=content,
            source=source,
            metadata=metadata
        )

    def add_reasoning_complete(
        self,
        final_result: str,
        steps_taken: int,
        source: str = "agentic_processor",
        session_id: Optional[str] = None
    ):
        """Log successful reasoning completion.

        Args:
            final_result: Final synthesis/result
            steps_taken: Number of steps taken to complete
            source: Source identifier
            session_id: Reasoning session ID
        """
        content = f"Reasoning complete ({steps_taken} steps): {final_result[:100]}..."

        self.add_entry(
            entry_type="reasoning_complete",
            content=content,
            source=source,
            metadata={
                "steps_taken": steps_taken,
                "final_result": final_result[:1000],  # Truncate for metadata
                "reasoning_session_id": session_id or self._get_latest_reasoning_session_id()
            }
        )

    def add_reasoning_failed(
        self,
        error_message: str,
        steps_taken: int,
        source: str = "agentic_processor",
        session_id: Optional[str] = None
    ):
        """Log failed reasoning attempt.

        Args:
            error_message: Error description
            steps_taken: Steps taken before failure
            source: Source identifier
            session_id: Reasoning session ID
        """
        content = f"Reasoning failed after {steps_taken} steps: {error_message}"

        self.add_entry(
            entry_type="reasoning_failed",
            content=content,
            source=source,
            metadata={
                "steps_taken": steps_taken,
                "error_message": error_message,
                "reasoning_session_id": session_id or self._get_latest_reasoning_session_id()
            }
        )

    def _generate_reasoning_session_id(self) -> str:
        """Generate unique session ID for reasoning workflow."""
        import uuid
        return f"reasoning_{uuid.uuid4().hex[:8]}"

    def _get_latest_reasoning_session_id(self) -> Optional[str]:
        """Get session ID from most recent reasoning_start entry."""
        for entry in reversed(self.entries):
            if entry.entry_type == "reasoning_start":
                return entry.metadata.get("reasoning_session_id")
        return None

    def get_reasoning_sessions(self) -> Dict[str, list]:
        """Get all reasoning sessions grouped by session_id.

        Returns:
            Dict mapping session_id to list of related entries
        """
        sessions = {}

        for entry in self.entries:
            if entry.entry_type.startswith("reasoning_"):
                session_id = entry.metadata.get("reasoning_session_id")
                if session_id:
                    if session_id not in sessions:
                        sessions[session_id] = []
                    sessions[session_id].append(entry)

        return sessions

    def format_reasoning_session(self, session_id: str) -> str:
        """Format specific reasoning session for display.

        Args:
            session_id: Session ID to format

        Returns:
            Formatted string representation of reasoning session
        """
        sessions = self.get_reasoning_sessions()
        if session_id not in sessions:
            return f"No reasoning session found: {session_id}"

        entries = sessions[session_id]
        output = []

        for entry in entries:
            if entry.entry_type == "reasoning_start":
                obj = entry.metadata.get("objective", "N/A")
                max_s = entry.metadata.get("max_steps", "?")
                output.append(f"🔍 Started Reasoning: {obj} (max {max_s} steps)")

            elif entry.entry_type == "reasoning_step":
                step_num = entry.metadata.get("step_num", "?")
                activity = entry.metadata.get("activity", "N/A")
                has_tools = entry.metadata.get("has_tool_calls", False)
                tool_indicator = " 🔧" if has_tools else ""
                output.append(f"  Step {step_num}: {activity}{tool_indicator}")

            elif entry.entry_type == "reasoning_complete":
                steps = entry.metadata.get("steps_taken", "?")
                output.append(f"✅ Completed in {steps} steps")

            elif entry.entry_type == "reasoning_failed":
                steps = entry.metadata.get("steps_taken", "?")
                error = entry.metadata.get("error_message", "Unknown error")
                output.append(f"❌ Failed after {steps} steps: {error}")

        return "\n".join(output)
```

### Part 2: Integrate with AgenticStepProcessor

**Modify `promptchain/utils/agentic_step_processor.py`**:

```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from typing import Optional


class AgenticStepProcessor:
    def __init__(
        self,
        # ... existing params
        history_manager: Optional[ExecutionHistoryManager] = None
    ):
        """
        Args:
            history_manager: ExecutionHistoryManager for logging reasoning steps
        """
        self.history_manager = history_manager
        self.reasoning_session_id: Optional[str] = None

    async def process_async(self, user_input: str) -> str:
        """Process with history logging."""
        # Log reasoning start
        if self.history_manager:
            self.history_manager.add_reasoning_start(
                objective=self.objective,
                max_steps=self.max_internal_steps
            )
            self.reasoning_session_id = self.history_manager._get_latest_reasoning_session_id()

        try:
            for step_num in range(1, self.max_internal_steps + 1):
                # Execute step
                response = await self._execute_reasoning_step(user_input, step_num)

                # Log step
                if self.history_manager:
                    activity = self._extract_step_summary(response)
                    tool_calls = self._extract_tool_calls(response)
                    self.history_manager.add_reasoning_step(
                        step_num=step_num,
                        activity=activity,
                        step_output=response if self.store_detailed_steps else None,
                        tool_calls=tool_calls,
                        session_id=self.reasoning_session_id
                    )

                # Check completion
                if self._is_objective_complete(response):
                    final_result = self._extract_final_synthesis(response)

                    # Log completion
                    if self.history_manager:
                        self.history_manager.add_reasoning_complete(
                            final_result=final_result,
                            steps_taken=step_num,
                            session_id=self.reasoning_session_id
                        )

                    return final_result

            # Max steps exhausted
            if self.history_manager:
                self.history_manager.add_reasoning_failed(
                    error_message="Max steps exhausted",
                    steps_taken=self.max_internal_steps,
                    session_id=self.reasoning_session_id
                )

            raise MaxStepsExhaustedError(...)

        except Exception as e:
            # Log failure
            if self.history_manager:
                self.history_manager.add_reasoning_failed(
                    error_message=str(e),
                    steps_taken=len(self.step_history),
                    session_id=self.reasoning_session_id
                )
            raise

    def _extract_tool_calls(self, llm_response: str) -> list:
        """Extract tool call information from LLM response.

        Returns:
            List of tool call dicts with name and args
        """
        # Parse tool calls from response (implementation depends on response format)
        # This is a simplified version
        import re
        tool_calls = []

        # Look for tool_calls in response structure
        if hasattr(self, 'last_tool_calls'):
            for tc in self.last_tool_calls:
                tool_calls.append({
                    "name": tc.get("name"),
                    "args": tc.get("arguments", {})
                })

        return tool_calls
```

### Part 3: Connect to AgentChain

**Modify `promptchain/utils/promptchaining.py`** (extend T049):

```python
async def _execute_agentic_step(self, processor, step_index):
    """Execute AgenticStepProcessor with history logging."""
    # Share history manager with processor
    if hasattr(self, 'history_manager') and self.history_manager:
        processor.history_manager = self.history_manager

    # Execute processor (now logs to history)
    result = await processor.process_async(self.current_input)

    return result
```

### Success Criteria
- Reasoning steps logged to ExecutionHistoryManager
- New entry types (reasoning_start, reasoning_step, etc.) work
- Session grouping allows filtering by reasoning workflow
- Token counting includes reasoning entries
- History formatting includes reasoning sessions
- No performance degradation with logging

## Testing

**Unit Test** (`tests/cli/unit/test_reasoning_history_logging.py`):

```python
import pytest
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


@pytest.mark.asyncio
async def test_reasoning_logged_to_history():
    """AgenticStepProcessor logs reasoning steps to history manager."""
    history_manager = ExecutionHistoryManager(max_tokens=10000)

    processor = AgenticStepProcessor(
        objective="Test logging",
        max_internal_steps=3,
        model_name="openai/gpt-4o-mini",
        history_manager=history_manager
    )

    result = await processor.process_async("Test input")

    # Verify logging
    reasoning_entries = [
        e for e in history_manager.entries
        if e.entry_type.startswith("reasoning_")
    ]

    assert len(reasoning_entries) >= 2  # Start + steps/complete
    assert any(e.entry_type == "reasoning_start" for e in reasoning_entries)
    assert any(
        e.entry_type in ("reasoning_complete", "reasoning_failed")
        for e in reasoning_entries
    )


def test_reasoning_session_grouping():
    """Reasoning sessions grouped by session_id."""
    history_manager = ExecutionHistoryManager()

    # Simulate two reasoning sessions
    history_manager.add_reasoning_start("Task 1", 5)
    session_id_1 = history_manager._get_latest_reasoning_session_id()
    history_manager.add_reasoning_step(1, "Step 1", session_id=session_id_1)
    history_manager.add_reasoning_complete("Done", 1, session_id=session_id_1)

    history_manager.add_reasoning_start("Task 2", 3)
    session_id_2 = history_manager._get_latest_reasoning_session_id()
    history_manager.add_reasoning_step(1, "Step 1", session_id=session_id_2)

    # Verify grouping
    sessions = history_manager.get_reasoning_sessions()
    assert len(sessions) == 2
    assert session_id_1 in sessions
    assert session_id_2 in sessions
    assert len(sessions[session_id_1]) == 3  # start + step + complete
    assert len(sessions[session_id_2]) == 2  # start + step
```

## Validation
- Run unit tests: `pytest tests/cli/unit/test_reasoning_history_logging.py -v`
- Manual test with TUI, verify history includes reasoning steps
- Check token counting includes reasoning entries

## Deliverable
- Extended `ExecutionHistoryManager` with reasoning entry types
- Integration in `AgenticStepProcessor`
- Session grouping and formatting
- Unit tests for history logging
