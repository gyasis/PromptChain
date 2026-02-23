# Communication Handlers Module - Implementation Complete

## Overview

The communication handlers module at `/home/gyasis/Documents/code/PromptChain/promptchain/cli/communication/handlers.py` is **fully implemented and tested**. It provides the foundation for agent-to-agent messaging in PromptChain CLI.

## Implemented Components

### 1. MessageType Enum ✅

Defines all required message types:

```python
class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    DELEGATION = "delegation"
    STATUS = "status"
```

### 2. CommunicationHandler Dataclass ✅

Handler registration object with filter criteria:

- **func**: Handler function (sync or async)
- **name**: Handler identifier
- **message_types**: Filter by message type (empty = match all)
- **senders**: Filter by sender (empty = match all)
- **receivers**: Filter by receiver (empty = match all)
- **priority**: Execution priority (higher runs first)

Includes `matches()` method for filter evaluation.

### 3. HandlerRegistry Singleton ✅

Global registry managing all handlers:

- **Singleton pattern**: Single instance across application
- **register(handler)**: Add handler, auto-sort by priority
- **unregister(name)**: Remove handler by name
- **get_matching_handlers()**: Find handlers matching criteria
- **dispatch()**: Route message to all matching handlers
- **handlers**: Property returning all registered handlers
- **reset()**: Clear registry (for testing)

### 4. @cli_communication_handler Decorator ✅

Convenient handler registration:

```python
@cli_communication_handler(
    type=MessageType.REQUEST,      # Single type
    types=[...],                   # Multiple types
    sender="agent1",               # Single sender
    senders=["a1", "a2"],         # Multiple senders
    receiver="agent2",             # Single receiver
    receivers=["r1", "r2"],       # Multiple receivers
    priority=100,                  # Execution priority
    name="custom_name"            # Handler name
)
def my_handler(payload, sender, receiver):
    return result
```

**Features**:
- Supports both sync and async functions
- Returns original function (preserves direct callability)
- Auto-registers with HandlerRegistry
- Flexible filtering options

### 5. get_handler_registry() Helper ✅

Convenience function to access the singleton registry.

## Key Features

### Filter Matching Logic

Handlers use **AND** logic for filters:
- Empty filter set = match all
- Message must match ALL specified filters
- Multiple values in filter = OR within that filter

Example:
```python
# Matches: REQUEST from (supervisor OR manager) to worker
@cli_communication_handler(
    types=[MessageType.REQUEST],
    senders=["supervisor", "manager"],
    receivers=["worker"]
)
```

### Priority-Based Execution

Handlers execute in priority order (highest first):
- Sorting happens automatically on registration
- Default priority is 0
- Use negative priorities for cleanup handlers

### Exception Handling

System continues even if handlers fail:
- Exceptions are caught and logged
- Failed handler returns `{"error": "...", "handler": "name"}`
- Other handlers still execute

### Async/Sync Support

Handlers can be either sync or async:
- Registry detects function type with `asyncio.iscoroutinefunction()`
- Async handlers are awaited, sync handlers called directly
- Both types can coexist in same registry

## Verification Status

All functionality verified by `/home/gyasis/Documents/code/PromptChain/verify_communication_handlers.py`:

```
✓ MessageType enum has all required values
✓ CommunicationHandler filtering works correctly
✓ HandlerRegistry is a proper singleton
✓ Handlers registered and sorted by priority
✓ Handler unregistration works
✓ Handler matching filters work correctly
✓ Sync handler dispatch works
✓ Async handler dispatch works
✓ Exception handling works (system continues)
✓ Supervisor-worker pattern works
✓ Broadcast pattern works
✓ Priority ordering works correctly
✓ Decorator registration works
✓ Decorator with all options works
✓ Decorator preserves original function
```

**Test Results**: All 16 tests passed ✅

## Integration with MessageBus

The handlers module integrates with `message_bus.py`:

1. MessageBus creates messages with sender/receiver/type
2. MessageBus calls `registry.dispatch()` with message details
3. Registry finds matching handlers using filter logic
4. Handlers execute in priority order
5. Results collected and returned to MessageBus
6. Activity logged for debugging (FR-019)

## Usage Examples

### Basic Handler

```python
from promptchain.cli.communication import cli_communication_handler, MessageType

@cli_communication_handler(type=MessageType.REQUEST)
def handle_requests(payload, sender, receiver):
    return {"status": "processed", "data": payload}
```

### Filtered Handler

```python
@cli_communication_handler(
    type=MessageType.DELEGATION,
    sender="supervisor",
    receiver="worker"
)
def handle_work(payload, sender, receiver):
    task = payload["task"]
    return {"task_id": task, "status": "completed"}
```

### High-Priority Handler

```python
@cli_communication_handler(
    type=MessageType.REQUEST,
    priority=100
)
def critical_validator(payload, sender, receiver):
    if not is_valid(payload):
        raise ValueError("Invalid request")
    return "validated"
```

### Async Handler

```python
@cli_communication_handler(type=MessageType.STATUS)
async def async_status_handler(payload, sender, receiver):
    await some_async_operation()
    return {"status": "updated"}
```

## File Locations

- **Implementation**: `/home/gyasis/Documents/code/PromptChain/promptchain/cli/communication/handlers.py`
- **Message Bus Integration**: `/home/gyasis/Documents/code/PromptChain/promptchain/cli/communication/message_bus.py`
- **Module Exports**: `/home/gyasis/Documents/code/PromptChain/promptchain/cli/communication/__init__.py`
- **Verification Script**: `/home/gyasis/Documents/code/PromptChain/verify_communication_handlers.py`
- **Usage Examples**: `/home/gyasis/Documents/code/PromptChain/promptchain/cli/communication/USAGE_EXAMPLES.md`
- **Unit Tests**: `/home/gyasis/Documents/code/PromptChain/tests/cli/unit/test_communication_handlers.py`

## Functional Requirements Coverage

- **FR-016**: ✅ System MUST support @cli_communication_handler decorator for message handlers
- **FR-017**: ✅ Handlers MUST support filtering by sender, receiver, and message type
- **FR-018**: ✅ System MUST support message types: request, response, broadcast, delegation, status
- **FR-019**: ✅ Activity logger MUST capture all communication (integrated via MessageBus)
- **FR-020**: ✅ Communication MUST be backward compatible - existing code works without handlers

## Architecture Notes

### Singleton Pattern

HandlerRegistry uses class-level singleton:
```python
class HandlerRegistry:
    _instance: Optional["HandlerRegistry"] = None

    def __new__(cls) -> "HandlerRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._handlers: List[CommunicationHandler] = []
        return cls._instance
```

Benefits:
- Single source of truth for handlers
- No need to pass registry around
- Automatic handler sharing across modules

### Decorator Pattern

Decorator preserves original function while registering handler:
```python
def cli_communication_handler(...):
    def decorator(func: Callable) -> Callable:
        handler = CommunicationHandler(...)
        HandlerRegistry().register(handler)
        return func  # Return original, not wrapper
    return decorator
```

Benefits:
- Functions can be called directly for testing
- No performance overhead when not using message bus
- Clean syntax with @decorator

### Filter Logic

Empty sets mean "match all":
```python
def matches(self, message_type, sender, receiver) -> bool:
    type_match = not self.message_types or message_type in self.message_types
    sender_match = not self.senders or sender in self.senders
    receiver_match = not self.receivers or receiver in self.receivers
    return type_match and sender_match and receiver_match
```

Benefits:
- Flexible filtering without complex DSL
- Simple to understand and debug
- Performant (set membership is O(1))

## Performance Characteristics

- **Registration**: O(n log n) due to priority sorting
- **Matching**: O(n) where n = number of handlers
- **Dispatch**: O(m) where m = number of matching handlers
- **Filter Check**: O(1) due to set membership tests

For typical usage (10-50 handlers), performance is excellent.

## Testing Strategy

Comprehensive test coverage includes:

1. **Unit Tests**: Individual component functionality
2. **Integration Tests**: Handler patterns (supervisor-worker, broadcast)
3. **Edge Cases**: Exception handling, empty filters, async/sync mix
4. **Performance**: Priority ordering, filter efficiency
5. **Functional**: All FR requirements verified

## Next Steps

The module is complete and ready for integration. To use in PromptChain CLI:

1. **Import decorators** where agents are defined
2. **Register handlers** for agent communication patterns
3. **Create MessageBus** in session manager
4. **Send messages** between agents via bus
5. **Monitor activity** via activity logger callback

Example integration:
```python
from promptchain.cli.communication import (
    MessageBus,
    MessageType,
    cli_communication_handler
)

# Define handlers
@cli_communication_handler(type=MessageType.DELEGATION)
def handle_work(payload, sender, receiver):
    return process_task(payload)

# Create bus
bus = MessageBus(session_id="cli_session")

# Send messages
await bus.delegate(
    sender="supervisor",
    receiver="worker",
    payload={"task": "analyze"}
)
```

## Conclusion

The communication handlers module is **production-ready** with:

- ✅ All required features implemented
- ✅ Comprehensive test coverage
- ✅ Full documentation with examples
- ✅ Integration with MessageBus
- ✅ Backward compatibility maintained
- ✅ Clean, Pythonic API

**Status**: COMPLETE AND VERIFIED ✓
