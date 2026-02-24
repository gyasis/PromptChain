# API Contract: Real-Time Steering (FR-011 to FR-014)
**Branch**: `006-promptchain-improvements`

---

## FR-011: Interrupt Queue Integration

**Files**:
- `promptchain/utils/interrupt_queue.py` (existing — complete)
- `promptchain/utils/enhanced_agentic_step_processor.py` (wiring)

### Contract: InterruptHandler.check_and_handle_interrupt()
```python
def check_and_handle_interrupt(
    current_step: int,
    current_context: str
) -> Optional[Dict[str, Any]]:
    """
    Non-blocking check at start of each thought cycle.
    Returns action dict on interrupt, None if clear.

    Action dict keys:
        "action": "abort" | "steering" | "correction" | "clarification" | "pause" | "resume"
        "interrupt_id": str
        "message": str
        "step": int
    """
```

### Contract: AgenticStepProcessor with interrupt
```python
# At start of every thought cycle:
if self.interrupt_handler:
    result = self.interrupt_handler.check_and_handle_interrupt(step, context)
    if result and result["action"] == "abort":
        return AgenticStepResult(status="aborted", ...)
    elif result and result["action"] in ("steering", "correction", "clarification"):
        # Inject interrupt message into LLM context
        context += self.interrupt_handler.format_interrupt_for_llm(result)
```

---

## FR-012: TUI Interrupt Enqueueing

**File**: `promptchain/cli/tui/app.py` (wiring)

### Contract
```python
# TUI input handler — when user types /interrupt <message> or !<message>:
def handle_interrupt_command(self, interrupt_type: str, message: str) -> None:
    """
    Enqueues interrupt without blocking TUI event loop.
    interrupt_type: "steering" | "correction" | "abort" | "pause"
    Returns immediately — fire-and-forget.
    """
    queue = get_global_interrupt_queue()
    queue.submit_interrupt(InterruptType[interrupt_type.upper()], message)
```

**TUI responsiveness guarantee**: `submit_interrupt()` is non-blocking
(`queue.put_nowait()`). The TUI event loop is never blocked.

---

## FR-013: Micro-Checkpoint After Tool Call

**File**: `promptchain/utils/enhanced_agentic_step_processor.py`
**Checkpoint class**: `promptchain/utils/checkpoint_manager.py` (existing)

### Contract
```python
# After each successful tool call in _execute_step():
checkpoint = MicroCheckpoint(
    checkpoint_id=f"{agent_id}_{step}_{tool_idx}_{int(time.time())}",
    agent_id=agent_id,
    step_number=step,
    tool_call_index=tool_idx,
    conversation_snapshot=copy.deepcopy(self.conversation_history),
    tool_results_snapshot=copy.deepcopy(current_tool_results),
    timestamp=time.time()
)
self._save_micro_checkpoint(checkpoint)
```

### Contract: Rewind on redirect
```python
def rewind_to_last_checkpoint(self) -> Optional[MicroCheckpoint]:
    """
    Returns most recent MicroCheckpoint and restores conversation state.
    Returns None if no checkpoints saved.
    Does NOT re-execute tool calls.
    """
```

**Retention**: Max 10 micro-checkpoints per session; oldest evicted first.

---

## FR-014: Global Override Signal

**File**: `promptchain/cli/communication/message_bus.py`

### Contract
```python
async def send_global_override(new_prompt: str, sender_id: str) -> None:
    """
    Publishes override to topic "agent.global_override".
    All subscribed agents receive the new prompt on next thought cycle.
    """

# AgenticStepProcessor subscribes to "agent.global_override":
async def _handle_override(topic: str, payload: Dict[str, Any]) -> None:
    """
    Replaces active prompt with payload["new_prompt"].
    Logs override event.
    Takes effect at next thought cycle start.
    """
```
