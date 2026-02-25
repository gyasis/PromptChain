# API Contract: Non-Blocking Async Execution (FR-015 to FR-017)
**Branch**: `006-promptchain-improvements`

---

## FR-015: Yielding LLM I/O

**File**: `promptchain/utils/agentic_step_processor.py` and
`promptchain/utils/enhanced_agentic_step_processor.py`

### Contract
```python
async def _call_llm_async(self, messages, model, **kwargs) -> str:
    """
    All LLM calls MUST use async variants (litellm.acompletion).
    NEVER calls litellm.completion() in an async context.
    Yields control during network I/O via await.
    """
    response = await litellm.acompletion(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content
```

**Rule**: Any sync path that exists is a wrapper calling `asyncio.run()` only
from a non-async entrypoint. The internal implementation uses `acompletion`.

---

## FR-016: AsyncAgentInbox

**File**: `promptchain/utils/async_agent_inbox.py` (new)

### Full API Contract
```python
class AsyncAgentInbox:
    def __init__(self, agent_id: str, maxsize: int = 100) -> None: ...

    async def send(self, message: InboxMessage) -> None:
        """Non-blocking send. Raises QueueFull if at capacity."""

    async def receive(self) -> InboxMessage:
        """Blocking receive. Yields control while waiting."""

    async def try_receive(self) -> Optional[InboxMessage]:
        """Non-blocking receive. Returns None if empty."""

    def qsize(self) -> int:
        """Current queue depth."""

    def empty(self) -> bool:
        """True if no messages pending."""
```

### Priority levels
```python
PRIORITY_INTERRUPT = 0     # interrupt / abort signals
PRIORITY_NORMAL    = 1     # regular agent messages
PRIORITY_BACKGROUND = 2    # background info / notifications
```

---

## FR-017: PubSubBus Topic Fan-out

**File**: `promptchain/cli/communication/message_bus.py` (extension)

### Full API Contract
```python
class PubSubBus:
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[str, Any], Awaitable[None]]
    ) -> None:
        """Register async callback for topic. Idempotent."""

    async def unsubscribe(
        self,
        topic: str,
        callback: Callable
    ) -> None:
        """Remove callback for topic. No-op if not registered."""

    async def publish(self, topic: str, payload: Any) -> None:
        """
        Delivers payload to all subscribers concurrently via asyncio.gather().
        Per-subscriber exceptions are caught, logged, and do NOT propagate.
        Returns when all subscribers have been triggered (not completed).
        """

    def publish_sync(self, topic: str, payload: Any) -> None:
        """
        Synchronous wrapper. Safe to call from non-async context only.
        Uses asyncio.run() internally.
        """
```

### Standard Topics
| Topic | Published by | Payload |
|-------|-------------|---------|
| `"agent.tool_complete"` | AgenticStepProcessor | `{"agent_id", "tool_name", "result"}` |
| `"agent.interrupt"` | InterruptQueue | `{"interrupt_id", "type", "message"}` |
| `"agent.global_override"` | TUI / MessageBus | `{"new_prompt", "sender_id"}` |
| `"session.context_distilled"` | ContextDistiller | `{"summary_tokens", "replaced_tokens"}` |
| `"session.memo_stored"` | MemoStore | `{"memo_id", "task_description"}` |
