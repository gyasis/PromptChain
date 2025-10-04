# Event System Guide

This guide covers the comprehensive event system introduced in PromptChain (v0.4.1d-f).

## Overview

The event system provides real-time visibility into PromptChain execution through callbacks. You can subscribe to specific events and react to them as they occur.

### Key Features

- **33+ event types** covering complete execution lifecycle
- **Event filtering** to subscribe to specific event types
- **Async and sync callbacks** both supported
- **Concurrent execution** with error isolation
- **Zero overhead** when callbacks not registered

## Core Concepts

### ExecutionEvent

Every event is an instance of `ExecutionEvent`:

```python
@dataclass
class ExecutionEvent:
    event_type: ExecutionEventType  # Type of event
    timestamp: datetime             # When it occurred
    step_number: Optional[int]      # Step in chain (if applicable)
    step_instruction: Optional[str] # Instruction being executed
    model_name: Optional[str]       # Model being used
    metadata: Dict[str, Any]        # Event-specific data
```

### ExecutionEventType

All available event types:

```python
class ExecutionEventType(Enum):
    # Chain lifecycle
    CHAIN_START, CHAIN_END, CHAIN_ERROR

    # Step execution
    STEP_START, STEP_END, STEP_ERROR, STEP_SKIPPED

    # Model calls
    MODEL_CALL_START, MODEL_CALL_END, MODEL_CALL_ERROR

    # Tool calls
    TOOL_CALL_START, TOOL_CALL_END, TOOL_CALL_ERROR

    # Function calls
    FUNCTION_CALL_START, FUNCTION_CALL_END, FUNCTION_CALL_ERROR

    # Agentic steps
    AGENTIC_STEP_START, AGENTIC_STEP_END, AGENTIC_STEP_ERROR
    AGENTIC_INTERNAL_STEP

    # Chain control
    CHAIN_BREAK

    # History management
    HISTORY_TRUNCATED

    # MCP events (see MCP Events Guide)
    MCP_CONNECT_START, MCP_CONNECT_END
    MCP_DISCONNECT_START, MCP_DISCONNECT_END
    MCP_TOOL_DISCOVERED, MCP_ERROR

    # Model management
    MODEL_LOAD, MODEL_UNLOAD
```

## Basic Usage

### 1. Define a Callback

Callbacks are simple functions that accept an `ExecutionEvent`:

```python
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

def my_callback(event: ExecutionEvent):
    """Simple callback that logs events."""
    print(f"[{event.event_type.name}] at {event.timestamp}")
    print(f"  Step: {event.step_number}")
    print(f"  Model: {event.model_name}")
    print(f"  Metadata: {event.metadata}")
```

### 2. Register Callback

```python
from promptchain import PromptChain

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze: {input}", "Summarize: {input}"]
)

# Register callback (receives ALL events)
chain.register_callback(my_callback)

# Run chain - events will be fired
result = chain.process_prompt("Explain quantum entanglement")
```

### 3. Unregister Callback

```python
# Remove callback when no longer needed
chain.unregister_callback(my_callback)
```

## Event Filtering

Subscribe to specific event types only:

### Filter by Single Event Type

```python
def error_callback(event: ExecutionEvent):
    """Only called for error events."""
    print(f"ERROR: {event.metadata.get('error')}")

# Register with filter
chain.register_callback(
    error_callback,
    event_filter=ExecutionEventType.CHAIN_ERROR
)
```

### Filter by Multiple Event Types

```python
def performance_callback(event: ExecutionEvent):
    """Track performance metrics."""
    if "execution_time_ms" in event.metadata:
        print(f"{event.event_type.name}: {event.metadata['execution_time_ms']}ms")

# Filter for START and END events
chain.register_callback(
    performance_callback,
    event_filter=[
        ExecutionEventType.CHAIN_START,
        ExecutionEventType.CHAIN_END,
        ExecutionEventType.STEP_START,
        ExecutionEventType.STEP_END
    ]
)
```

## Event Metadata

Each event type includes specific metadata:

### Chain Events

```python
# CHAIN_START
{
    "total_instructions": 3,
    "models": ["openai/gpt-4"],
    "initial_input": "User's question..."
}

# CHAIN_END
{
    "execution_time_ms": 1234.5,
    "total_steps": 3,
    "final_output": "Final response..."
}

# CHAIN_ERROR
{
    "error": "Error message",
    "error_type": "ModelError",
    "step_number": 2
}
```

### Step Events

```python
# STEP_START
{
    "instruction": "Analyze: {input}",
    "instruction_type": "string",  # or "function", "agentic"
    "step_number": 0
}

# STEP_END
{
    "execution_time_ms": 456.7,
    "output_length": 500,
    "tokens_used": 250
}

# STEP_ERROR
{
    "error": "Model API error",
    "step_number": 1,
    "instruction": "The failing instruction"
}
```

### Model Call Events

```python
# MODEL_CALL_START
{
    "model": "openai/gpt-4",
    "prompt_length": 150,
    "temperature": 0.7
}

# MODEL_CALL_END
{
    "execution_time_ms": 1200.0,
    "tokens_used": 450,
    "prompt_tokens": 150,
    "completion_tokens": 300,
    "response_length": 800
}

# MODEL_CALL_ERROR
{
    "error": "Rate limit exceeded",
    "model": "openai/gpt-4",
    "retry_attempt": 2
}
```

### Tool Call Events

```python
# TOOL_CALL_START
{
    "tool_name": "search_database",
    "tool_args": {"query": "SELECT * FROM users"},
    "is_mcp_tool": False
}

# TOOL_CALL_END
{
    "execution_time_ms": 89.2,
    "result_length": 1500,
    "success": True
}

# TOOL_CALL_ERROR
{
    "error": "Database connection failed",
    "tool_name": "search_database",
    "error_type": "ConnectionError"
}
```

### Agentic Step Events

```python
# AGENTIC_STEP_START
{
    "objective": "Find and analyze patterns",
    "max_steps": 5,
    "history_mode": "minimal"
}

# AGENTIC_INTERNAL_STEP
{
    "step_number": 3,
    "reasoning": "Calling search tool to find data",
    "tool_calls": [{"name": "search", "args": {...}}]
}

# AGENTIC_STEP_END
{
    "total_steps": 4,
    "objective_achieved": True,
    "total_tools_called": 7,
    "execution_time_ms": 3456.8
}
```

## Async Callbacks

Both sync and async callbacks are supported:

```python
import asyncio

async def async_callback(event: ExecutionEvent):
    """Async callback for database logging."""
    await log_to_database(event)
    print(f"Logged event: {event.event_type.name}")

# Register async callback (works automatically)
chain.register_callback(async_callback)
```

The callback manager automatically detects async functions and executes them appropriately.

## Real-World Examples

### Performance Monitoring

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def __call__(self, event: ExecutionEvent):
        if event.event_type == ExecutionEventType.STEP_END:
            self.metrics.append({
                "step": event.step_number,
                "time_ms": event.metadata.get("execution_time_ms"),
                "tokens": event.metadata.get("tokens_used")
            })

    def report(self):
        total_time = sum(m["time_ms"] for m in self.metrics)
        total_tokens = sum(m["tokens"] for m in self.metrics)
        print(f"Total time: {total_time}ms")
        print(f"Total tokens: {total_tokens}")
        print(f"Avg time/step: {total_time / len(self.metrics)}ms")

monitor = PerformanceMonitor()
chain.register_callback(monitor)

result = chain.process_prompt("Analyze this data...")
monitor.report()
```

### Error Tracking

```python
class ErrorTracker:
    def __init__(self):
        self.errors = []

    def __call__(self, event: ExecutionEvent):
        if event.event_type.name.endswith("_ERROR"):
            self.errors.append({
                "type": event.event_type.name,
                "error": event.metadata.get("error"),
                "timestamp": event.timestamp,
                "context": {
                    "step": event.step_number,
                    "model": event.model_name
                }
            })

    def has_errors(self):
        return len(self.errors) > 0

    def get_error_summary(self):
        return {
            "total_errors": len(self.errors),
            "error_types": [e["type"] for e in self.errors],
            "latest_error": self.errors[-1] if self.errors else None
        }

tracker = ErrorTracker()
chain.register_callback(
    tracker,
    event_filter=[
        ExecutionEventType.CHAIN_ERROR,
        ExecutionEventType.STEP_ERROR,
        ExecutionEventType.MODEL_CALL_ERROR,
        ExecutionEventType.TOOL_CALL_ERROR
    ]
)

result = chain.process_prompt("Your input...")

if tracker.has_errors():
    print(f"Error summary: {tracker.get_error_summary()}")
```

### Structured Logging

```python
import json
import logging

logger = logging.getLogger(__name__)

def structured_logger(event: ExecutionEvent):
    """Log events in structured JSON format."""
    log_entry = {
        "event_type": event.event_type.name,
        "timestamp": event.timestamp.isoformat(),
        "step": event.step_number,
        "model": event.model_name,
        "metadata": event.metadata
    }

    if event.event_type.name.endswith("_ERROR"):
        logger.error(json.dumps(log_entry))
    elif event.event_type.name.endswith("_START"):
        logger.info(json.dumps(log_entry))
    elif event.event_type.name.endswith("_END"):
        logger.info(json.dumps(log_entry))

chain.register_callback(structured_logger)
```

### Live Dashboard Updates

```python
import asyncio
from typing import Optional

class DashboardUpdater:
    def __init__(self, websocket=None):
        self.websocket = websocket
        self.current_step = 0
        self.status = "idle"

    async def __call__(self, event: ExecutionEvent):
        """Async callback for WebSocket updates."""
        update = {
            "event": event.event_type.name,
            "step": event.step_number,
            "timestamp": event.timestamp.isoformat()
        }

        if event.event_type == ExecutionEventType.CHAIN_START:
            self.status = "running"
            update["status"] = "started"

        elif event.event_type == ExecutionEventType.STEP_END:
            self.current_step = event.step_number
            update["progress"] = self.current_step

        elif event.event_type == ExecutionEventType.CHAIN_END:
            self.status = "completed"
            update["status"] = "completed"

        # Send to WebSocket dashboard
        if self.websocket:
            await self.websocket.send_json(update)

# Usage with WebSocket
dashboard = DashboardUpdater(websocket=ws)
chain.register_callback(dashboard)
```

## Error Handling

Callback errors are isolated and don't affect execution:

```python
def buggy_callback(event: ExecutionEvent):
    raise ValueError("Oops!")  # This won't stop the chain

chain.register_callback(buggy_callback)

# Chain still executes successfully
result = chain.process_prompt("Your input...")  # Works fine
```

Callback errors are logged but execution continues.

## Performance Considerations

### Zero Overhead Without Callbacks

```python
# No callbacks registered = zero overhead
chain = PromptChain(models=["openai/gpt-4"], instructions=[...])
result = chain.process_prompt("...")  # Fast, no event overhead
```

### Minimal Overhead With Callbacks

- Events are emitted via async tasks (non-blocking)
- Callbacks run concurrently with `asyncio.gather`
- Typical overhead: <1% of execution time

### Best Practices

1. **Use event filtering** to reduce callback invocations
2. **Keep callbacks lightweight** - offload heavy work to background tasks
3. **Unregister callbacks** when no longer needed
4. **Use async callbacks** for I/O operations (DB, network)

## Callback Manager API

Advanced usage: access the CallbackManager directly:

```python
from promptchain.utils.execution_callback import CallbackManager

# Get callback manager from chain
callback_manager = chain.callback_manager

# Check if callbacks registered
if callback_manager.has_callbacks():
    print("Callbacks active")

# Emit custom events (advanced)
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType
custom_event = ExecutionEvent(
    event_type=ExecutionEventType.CHAIN_START,
    metadata={"custom_field": "value"}
)
await callback_manager.emit(custom_event)
```

## Event Flow Diagram

```
User Code
    ↓
PromptChain.process_prompt()
    ↓
[CHAIN_START] ──→ Callbacks
    ↓
For each instruction (step):
    ↓
    [STEP_START] ──→ Callbacks
    ↓
    If string instruction:
        [MODEL_CALL_START] ──→ Callbacks
        → LLM execution
        [MODEL_CALL_END] ──→ Callbacks
    ↓
    If function instruction:
        [FUNCTION_CALL_START] ──→ Callbacks
        → Function execution
        [FUNCTION_CALL_END] ──→ Callbacks
    ↓
    If agentic instruction:
        [AGENTIC_STEP_START] ──→ Callbacks
        → Internal reasoning loop:
            [AGENTIC_INTERNAL_STEP] ──→ Callbacks (each step)
            → Tool calls:
                [TOOL_CALL_START] ──→ Callbacks
                [TOOL_CALL_END] ──→ Callbacks
        [AGENTIC_STEP_END] ──→ Callbacks
    ↓
    [STEP_END] ──→ Callbacks
    ↓
[CHAIN_END] ──→ Callbacks
```

## Next Steps

- Check out the [MCP Events Guide](mcp-events.md) for MCP-specific events
- See [examples/basic_callbacks.py](../../examples/observability/basic_callbacks.py) for working code
- Review [examples/event_filtering.py](../../examples/observability/event_filtering.py) for filtering patterns
- Read [Best Practices](best-practices.md) for production recommendations
