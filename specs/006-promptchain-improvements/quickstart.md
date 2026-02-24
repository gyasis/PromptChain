# Quickstart: PromptChain Improvements (006)
**Branch**: `006-promptchain-improvements`
**Date**: 2026-02-24

---

## Setup

```bash
git checkout -b 006-promptchain-improvements
pip install -e ".[dev]"
```

---

## 1. Bug Fixes: Gemini MCP Tool Calls (FR-001)

After the patch, Gemini tools work without parameter errors:

```python
from promptchain.utils.enhanced_agentic_step_processor import EnhancedAgenticStepProcessor

processor = EnhancedAgenticStepProcessor(model="openai/gpt-4o-mini", tools=[...])
# gemini_debug, gemini_brainstorm, ask_gemini now use correct param names
result = processor.run("Debug this Python error: ...")
```

---

## 2. Real-Time Steering: Interrupt Queue (FR-011)

```python
from promptchain.utils.interrupt_queue import (
    InterruptQueue, InterruptType, InterruptHandler, get_global_interrupt_queue
)
from promptchain.utils.enhanced_agentic_step_processor import EnhancedAgenticStepProcessor

# Create queue and wire it in
queue = InterruptQueue()
processor = EnhancedAgenticStepProcessor(
    model="openai/gpt-4o-mini",
    interrupt_queue=queue,    # new optional parameter
)

# In a separate thread, submit a steering interrupt mid-execution:
import threading
def submit_later():
    import time; time.sleep(2)
    queue.submit_interrupt(InterruptType.STEERING, "Focus on error handling only")

t = threading.Thread(target=submit_later)
t.start()

# Agent will acknowledge interrupt at next thought cycle boundary
result = processor.run("Implement a complete REST API framework...")
```

---

## 3. Context Memory: Memo Store (FR-008/009)

```python
from promptchain.utils.memo_store import MemoStore, inject_relevant_memos

# Auto-persists to ~/.promptchain/memos.db
store = MemoStore(similarity_threshold=0.7)

# Store a lesson learned
store.store_memo(
    task_description="Fix Python async event loop in TUI application",
    solution="Use run_async_in_context() instead of asyncio.run() inside Textual",
    outcome="success"
)

# Later in a new session — retrieve relevant knowledge:
relevant = store.retrieve_relevant_memos("Handle async operations in Textual TUI")
for memo in relevant:
    print(f"[{memo.relevance_score:.2f}] {memo.task_description}")
    print(f"  -> {memo.solution}")
```

---

## 4. Context Distillation (FR-007)

```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager, ContextDistiller

history = ExecutionHistoryManager(max_tokens=8000)
distiller = ContextDistiller(distillation_threshold=0.7)

processor = EnhancedAgenticStepProcessor(
    model="openai/gpt-4o-mini",
    history_manager=history,
    context_distiller=distiller,  # new optional parameter
)

# At 70% token capacity, distiller automatically summarizes older messages
# and replaces them with a "Current State of Knowledge" entry.
```

---

## 5. Background Janitor (FR-010)

```python
import asyncio
from promptchain.utils.janitor_agent import JanitorAgent

history = ExecutionHistoryManager(max_tokens=8000)
distiller = ContextDistiller()

janitor = JanitorAgent(
    history_manager=history,
    context_distiller=distiller,
    check_interval=30.0,      # check every 30 seconds
    compression_threshold=0.8  # compress at 80% token usage
)

async def main():
    await janitor.start()   # runs as background asyncio.Task
    # ... your agent logic ...
    await janitor.stop()

asyncio.run(main())
```

---

## 6. Pub/Sub Pipeline (FR-017)

```python
from promptchain.cli.communication.message_bus import PubSubBus

bus = PubSubBus()

async def on_tool_complete(topic: str, payload: dict) -> None:
    print(f"Tool done: {payload['tool_name']} by {payload['agent_id']}")

async def main():
    await bus.subscribe("agent.tool_complete", on_tool_complete)
    # Multiple subscribers receive events concurrently
    await bus.publish("agent.tool_complete", {
        "agent_id": "research_agent",
        "tool_name": "web_search",
        "result": "..."
    })

asyncio.run(main())
```

---

## 7. Async Agent Inbox (FR-016)

```python
from promptchain.utils.async_agent_inbox import AsyncAgentInbox, InboxMessage

inbox = AsyncAgentInbox(agent_id="analyst_agent")

async def agent_loop():
    while True:
        msg = await inbox.receive()   # yields control while waiting
        print(f"[{msg.priority}] {msg.topic}: {msg.payload}")

# Another coroutine can send without blocking:
async def send_messages():
    await inbox.send(InboxMessage(priority=1, topic="task.assigned", payload={"task": "..."}))
```

---

## Running Tests

```bash
# All bug fix tests
pytest tests/integration/test_006_bug_fixes.py -v

# Steering / interrupt tests
pytest tests/integration/test_006_steering_flow.py -v

# Unit tests for new components
pytest tests/unit/test_memo_store_integration.py \
       tests/unit/test_interrupt_queue_integration.py \
       tests/unit/test_pubsub_bus.py \
       tests/unit/test_async_agent_inbox.py \
       tests/unit/test_context_distiller_wiring.py -v
```
