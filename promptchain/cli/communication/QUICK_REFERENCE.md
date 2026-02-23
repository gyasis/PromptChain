# Communication Handlers - Quick Reference

## Import

```python
from promptchain.cli.communication import (
    cli_communication_handler,
    MessageType,
    MessageBus,
    HandlerRegistry,
    get_handler_registry
)
```

## Message Types

```python
MessageType.REQUEST      # "request"    - Ask for data/action
MessageType.RESPONSE     # "response"   - Reply to request
MessageType.BROADCAST    # "broadcast"  - Message to all
MessageType.DELEGATION   # "delegation" - Assign task
MessageType.STATUS       # "status"     - Progress update
```

## Decorator Syntax

```python
@cli_communication_handler(
    type=MessageType.REQUEST,           # Single type (optional)
    types=[MessageType.REQUEST, ...],   # Multiple types (optional)
    sender="agent1",                    # Single sender (optional)
    senders=["a1", "a2"],              # Multiple senders (optional)
    receiver="agent2",                  # Single receiver (optional)
    receivers=["r1", "r2"],            # Multiple receivers (optional)
    priority=0,                         # Execution order (default: 0)
    name="custom_name"                 # Handler name (default: function name)
)
def handler_function(payload, sender, receiver):
    """
    payload: Dict[str, Any] - Message data
    sender: str - Sending agent name
    receiver: str - Receiving agent name
    """
    return result  # Can be any value
```

## Common Patterns

### Match All of Type

```python
@cli_communication_handler(type=MessageType.REQUEST)
def handle_all_requests(payload, sender, receiver):
    return "processed"
```

### Specific Route

```python
@cli_communication_handler(
    type=MessageType.DELEGATION,
    sender="supervisor",
    receiver="worker"
)
def handle_delegation(payload, sender, receiver):
    return {"status": "completed"}
```

### Multiple Senders

```python
@cli_communication_handler(
    type=MessageType.REQUEST,
    senders=["supervisor", "manager", "coordinator"]
)
def handle_leadership_requests(payload, sender, receiver):
    return "handled"
```

### Broadcast Listener

```python
@cli_communication_handler(type=MessageType.BROADCAST)
def handle_broadcasts(payload, sender, receiver):
    return "received"
```

### High Priority

```python
@cli_communication_handler(
    type=MessageType.REQUEST,
    priority=100  # Runs before priority 0
)
def critical_handler(payload, sender, receiver):
    return "priority_handled"
```

### Async Handler

```python
@cli_communication_handler(type=MessageType.STATUS)
async def async_handler(payload, sender, receiver):
    await async_operation()
    return "async_result"
```

## Registry Operations

### Get Registry

```python
registry = get_handler_registry()
# OR
registry = HandlerRegistry()  # Same instance
```

### List Handlers

```python
for handler in registry.handlers:
    print(f"{handler.name}: priority={handler.priority}")
```

### Find Matching Handlers

```python
matches = registry.get_matching_handlers(
    MessageType.REQUEST,
    sender="agent1",
    receiver="agent2"
)
```

### Dispatch Message

```python
results = await registry.dispatch(
    MessageType.REQUEST,
    sender="agent1",
    receiver="agent2",
    payload={"data": "value"}
)
# results = [handler1_result, handler2_result, ...]
```

### Reset (Testing)

```python
HandlerRegistry.reset()  # Clear all handlers
```

## MessageBus Usage

### Create Bus

```python
bus = MessageBus(
    session_id="my_session",
    activity_logger=lambda event: log(event)  # Optional
)
```

### Send Messages

```python
# Generic send
message = await bus.send(
    sender="agent1",
    receiver="agent2",
    message_type=MessageType.REQUEST,
    payload={"query": "data"}
)

# Convenience methods
await bus.request(sender, receiver, payload)
await bus.respond(sender, receiver, payload)
await bus.delegate(sender, receiver, payload)
await bus.status_update(sender, receiver, payload)
await bus.broadcast(sender, payload)  # receiver="*"
```

### Get History

```python
# All messages
history = bus.get_history()

# Filtered
history = bus.get_history(
    sender="agent1",
    receiver="agent2",
    message_type=MessageType.REQUEST,
    limit=50
)
```

### Clear History

```python
count = bus.clear_history()  # Returns number cleared
```

## Function Signatures

### Handler Function

```python
def sync_handler(payload: Dict[str, Any], sender: str, receiver: str) -> Any:
    return result

async def async_handler(payload: Dict[str, Any], sender: str, receiver: str) -> Any:
    return result
```

### Activity Logger Callback

```python
def activity_logger(event: Dict[str, Any]) -> None:
    # event contains:
    # - event_type: str ("message_sent", "message_delivered")
    # - session_id: str
    # - timestamp: str (ISO format)
    # - message_id: str
    # - sender: str
    # - receiver: str
    # - type: str (message type)
    # - payload_keys: List[str]
    # - handlers_invoked: int (for delivered events)
    # - delivered: bool (for delivered events)
    pass
```

## Error Handling

Handlers can raise exceptions - system continues:

```python
@cli_communication_handler(type=MessageType.REQUEST)
def handler(payload, sender, receiver):
    if not valid(payload):
        raise ValueError("Invalid payload")
    return "processed"

# Exception caught, result = {"error": "Invalid payload", "handler": "handler"}
# Other handlers still execute
```

## Priority Guidelines

- **100+**: Critical (validation, auth)
- **50-99**: Important business logic
- **0-49**: Normal handlers (default: 0)
- **Negative**: Cleanup/logging

## Filter Semantics

- **Empty filter = match all**: `senders=set()` matches any sender
- **Multiple values = OR**: `senders={"a1", "a2"}` matches a1 OR a2
- **Multiple filters = AND**: Must match type AND sender AND receiver

## Testing Pattern

```python
import pytest
from promptchain.cli.communication import HandlerRegistry

@pytest.fixture(autouse=True)
def reset_handlers():
    HandlerRegistry.reset()
    yield
    HandlerRegistry.reset()

def test_handler():
    @cli_communication_handler(type=MessageType.REQUEST)
    def my_handler(payload, sender, receiver):
        return "test"

    registry = get_handler_registry()
    assert len(registry.handlers) == 1
```

## Performance Notes

- Registration: O(n log n) - handlers sorted by priority
- Matching: O(n) - checks all handlers
- Dispatch: O(m) - invokes matching handlers
- Filter check: O(1) - set membership

Typical usage (10-50 handlers): excellent performance

## Complete Example

```python
from promptchain.cli.communication import (
    cli_communication_handler,
    MessageType,
    MessageBus
)

# Register handlers
@cli_communication_handler(
    type=MessageType.DELEGATION,
    sender="supervisor",
    receiver="worker"
)
def handle_work(payload, sender, receiver):
    task = payload["task"]
    return {"status": "completed", "result": process(task)}

@cli_communication_handler(type=MessageType.BROADCAST)
def monitor(payload, sender, receiver):
    log_message(sender, payload)
    return "logged"

# Use message bus
async def run():
    bus = MessageBus(session_id="example")

    # Send delegation
    msg = await bus.delegate(
        sender="supervisor",
        receiver="worker",
        payload={"task": "analyze_data"}
    )

    # Send broadcast
    await bus.broadcast(
        sender="coordinator",
        payload={"announcement": "System update"}
    )

    # Get history
    history = bus.get_history(limit=10)
    for msg in history:
        print(f"{msg.sender} -> {msg.receiver}: {msg.type.value}")
```

## See Also

- **Full Documentation**: `USAGE_EXAMPLES.md`
- **Implementation**: `handlers.py`
- **MessageBus**: `message_bus.py`
- **Tests**: `/tests/cli/unit/test_communication_handlers.py`
- **Verification**: `/verify_communication_handlers.py`
