# Communication Handlers - Usage Examples

This document demonstrates how to use the `@cli_communication_handler` decorator and `MessageBus` for agent-to-agent communication in PromptChain CLI.

## Table of Contents

- [Basic Concepts](#basic-concepts)
- [Handler Registration](#handler-registration)
- [Message Types](#message-types)
- [Filtering Patterns](#filtering-patterns)
- [Integration Patterns](#integration-patterns)
- [Best Practices](#best-practices)

## Basic Concepts

The communication system has three main components:

1. **MessageType**: Enum defining message categories (request, response, broadcast, delegation, status)
2. **@cli_communication_handler**: Decorator for registering handler functions
3. **MessageBus**: Central routing system that dispatches messages to handlers

## Handler Registration

### Simple Handler

Register a handler for a specific message type:

```python
from promptchain.cli.communication import cli_communication_handler, MessageType

@cli_communication_handler(type=MessageType.REQUEST)
def handle_requests(payload, sender, receiver):
    """Handle all incoming requests."""
    return {"status": "processed", "data": payload}
```

### Handler with Filters

Filter by sender, receiver, or both:

```python
@cli_communication_handler(
    type=MessageType.DELEGATION,
    sender="supervisor",
    receiver="worker"
)
def handle_work_delegation(payload, sender, receiver):
    """Handle work delegated from supervisor to worker."""
    task = payload["task"]
    # Process task...
    return {"task_id": task["id"], "status": "completed"}
```

### Multiple Filters

Handle messages from multiple sources:

```python
@cli_communication_handler(
    types=[MessageType.REQUEST, MessageType.DELEGATION],
    senders=["supervisor", "manager", "coordinator"],
    receivers=["worker1", "worker2", "worker3"]
)
def handle_team_messages(payload, sender, receiver):
    """Handle messages from any team lead to any team member."""
    return f"Received from {sender} for {receiver}"
```

### Custom Priority

Control execution order with priority (higher executes first):

```python
@cli_communication_handler(
    type=MessageType.REQUEST,
    priority=100  # Runs before default priority (0)
)
def high_priority_handler(payload, sender, receiver):
    """Process critical requests first."""
    return "urgent_handling"

@cli_communication_handler(
    type=MessageType.REQUEST,
    priority=0  # Default priority
)
def normal_handler(payload, sender, receiver):
    """Process normal requests."""
    return "normal_handling"
```

### Async Handlers

Handlers can be async functions:

```python
@cli_communication_handler(type=MessageType.STATUS)
async def async_status_handler(payload, sender, receiver):
    """Async handler for status updates."""
    await some_async_operation()
    return {"status": "updated"}
```

## Message Types

### REQUEST

Agent asks another agent for information or action:

```python
@cli_communication_handler(type=MessageType.REQUEST)
def handle_data_request(payload, sender, receiver):
    query = payload.get("query")
    data = database.query(query)
    return {"data": data, "count": len(data)}
```

### RESPONSE

Agent responds to a previous request:

```python
@cli_communication_handler(type=MessageType.RESPONSE)
def handle_response(payload, sender, receiver):
    original_request_id = payload.get("request_id")
    result = payload.get("result")
    # Store or process response
    return "response_acknowledged"
```

### BROADCAST

Message to all agents (receiver = "*"):

```python
@cli_communication_handler(type=MessageType.BROADCAST)
def handle_global_announcement(payload, sender, receiver):
    announcement = payload.get("message")
    log_announcement(announcement)
    return "announcement_received"
```

### DELEGATION

Agent delegates task to another agent:

```python
@cli_communication_handler(
    type=MessageType.DELEGATION,
    receiver="specialist_agent"
)
def handle_delegated_task(payload, sender, receiver):
    task = payload["task"]
    result = perform_specialized_work(task)
    return {"task_id": task["id"], "result": result}
```

### STATUS

Status update or progress report:

```python
@cli_communication_handler(type=MessageType.STATUS)
def handle_status_update(payload, sender, receiver):
    progress = payload.get("progress", 0)
    stage = payload.get("stage")
    update_dashboard(sender, progress, stage)
    return "status_logged"
```

## Filtering Patterns

### Broadcast Receiver

Handle all broadcast messages:

```python
@cli_communication_handler(type=MessageType.BROADCAST)
def broadcast_listener(payload, sender, receiver):
    """Receives all broadcasts from any sender."""
    return f"Broadcast from {sender}: {payload}"
```

### Specific Agent Communication

Only handle messages between two specific agents:

```python
@cli_communication_handler(
    type=MessageType.REQUEST,
    sender="agent_a",
    receiver="agent_b"
)
def handle_a_to_b(payload, sender, receiver):
    """Only handles messages from agent_a to agent_b."""
    return "specific_routing"
```

### Wildcard Handlers

Match all messages of a type:

```python
@cli_communication_handler(type=MessageType.REQUEST)
def catch_all_requests(payload, sender, receiver):
    """Handles ANY request from ANY sender to ANY receiver."""
    return f"Request from {sender} to {receiver}"
```

### Multiple Message Types

Handle different types with same logic:

```python
@cli_communication_handler(
    types=[MessageType.REQUEST, MessageType.RESPONSE, MessageType.STATUS]
)
def logging_handler(payload, sender, receiver):
    """Log all communication."""
    logger.log(f"{sender} -> {receiver}: {payload}")
    return "logged"
```

## Integration Patterns

### Supervisor-Worker Pattern

Supervisor delegates work, workers report back:

```python
# Worker side
@cli_communication_handler(
    type=MessageType.DELEGATION,
    sender="supervisor",
    receiver="worker"
)
def handle_work(payload, sender, receiver):
    task = payload["task"]
    result = process_task(task)
    # Worker sends response back
    return {"status": "completed", "result": result}

# Supervisor side
@cli_communication_handler(
    type=MessageType.RESPONSE,
    sender="worker",
    receiver="supervisor"
)
def handle_worker_response(payload, sender, receiver):
    task_id = payload["task_id"]
    result = payload["result"]
    mark_task_complete(task_id, result)
    return "acknowledged"
```

### Request-Response Cycle

Agent makes request, receives response:

```python
# Data agent handles requests
@cli_communication_handler(
    type=MessageType.REQUEST,
    receiver="data_agent"
)
def handle_data_request(payload, sender, receiver):
    query = payload["query"]
    data = fetch_data(query)
    # Response is sent back via message bus
    return {"data": data}

# Requester handles responses
@cli_communication_handler(
    type=MessageType.RESPONSE,
    sender="data_agent"
)
def handle_data_response(payload, sender, receiver):
    data = payload["data"]
    process_data(data)
    return "data_processed"
```

### Multi-Agent Collaboration

Multiple agents working together:

```python
# Coordinator broadcasts task
@cli_communication_handler(
    type=MessageType.BROADCAST,
    sender="coordinator"
)
def handle_coordinator_broadcast(payload, sender, receiver):
    task_type = payload["task_type"]
    if can_handle(task_type):
        claim_task(task_type)
    return "broadcast_processed"

# Specialist handles delegations
@cli_communication_handler(
    type=MessageType.DELEGATION,
    receiver="specialist"
)
def handle_specialist_work(payload, sender, receiver):
    result = perform_specialized_work(payload["task"])
    return {"result": result}

# Status monitoring
@cli_communication_handler(type=MessageType.STATUS)
def monitor_status(payload, sender, receiver):
    update_dashboard(sender, payload)
    return "status_updated"
```

### Chain of Responsibility

Multiple handlers with priority:

```python
@cli_communication_handler(
    type=MessageType.REQUEST,
    priority=100
)
def validation_handler(payload, sender, receiver):
    """First: Validate request."""
    if not is_valid(payload):
        raise ValueError("Invalid request")
    return "validated"

@cli_communication_handler(
    type=MessageType.REQUEST,
    priority=50
)
def authentication_handler(payload, sender, receiver):
    """Second: Check authentication."""
    if not is_authenticated(sender):
        raise PermissionError("Not authenticated")
    return "authenticated"

@cli_communication_handler(
    type=MessageType.REQUEST,
    priority=0
)
def processing_handler(payload, sender, receiver):
    """Third: Process request."""
    return process_request(payload)
```

## Best Practices

### 1. Handler Function Signature

Always accept three parameters:

```python
@cli_communication_handler(type=MessageType.REQUEST)
def my_handler(payload, sender, receiver):
    # payload: Dict[str, Any] - message content
    # sender: str - sending agent name
    # receiver: str - receiving agent name
    return result
```

### 2. Return Values

Handlers can return any value, which becomes part of dispatch results:

```python
@cli_communication_handler(type=MessageType.REQUEST)
def handler_with_result(payload, sender, receiver):
    return {"status": "success", "data": [...]}

# Caller receives list of all handler results
results = await message_bus.send(...)
# results = [{"status": "success", "data": [...]}, ...]
```

### 3. Error Handling

The system continues on handler exceptions:

```python
@cli_communication_handler(type=MessageType.REQUEST)
def safe_handler(payload, sender, receiver):
    try:
        return process(payload)
    except Exception as e:
        logger.error(f"Handler failed: {e}")
        # Exception is caught by registry, system continues
        raise  # Re-raise to be logged
```

### 4. Naming Handlers

Use descriptive names for debugging:

```python
@cli_communication_handler(
    type=MessageType.DELEGATION,
    name="supervisor_to_worker_delegation_handler"
)
def handle_delegation(payload, sender, receiver):
    return "processed"
```

### 5. Avoid Side Effects in Filters

Keep filter logic simple - complex logic belongs in handler:

```python
# GOOD: Simple filters
@cli_communication_handler(
    type=MessageType.REQUEST,
    sender="supervisor"
)
def handle(payload, sender, receiver):
    if payload.get("urgent"):
        # Complex logic here
        pass

# AVOID: Complex filter logic
# (Not possible with current API, but conceptually)
```

### 6. Priority Guidelines

Use priority sparingly and consistently:

- 100+: Critical system handlers (validation, auth)
- 50-99: Important business logic
- 0-49: Normal handlers
- Negative: Cleanup/logging handlers

```python
@cli_communication_handler(type=MessageType.REQUEST, priority=100)
def critical_validator(payload, sender, receiver):
    return "validated"

@cli_communication_handler(type=MessageType.REQUEST, priority=0)
def normal_processor(payload, sender, receiver):
    return "processed"

@cli_communication_handler(type=MessageType.REQUEST, priority=-10)
def logging_handler(payload, sender, receiver):
    log_message(payload, sender, receiver)
    return "logged"
```

### 7. Testing Handlers

Use HandlerRegistry.reset() for test isolation:

```python
import pytest
from promptchain.cli.communication import HandlerRegistry

@pytest.fixture(autouse=True)
def reset_handlers():
    HandlerRegistry.reset()
    yield
    HandlerRegistry.reset()

def test_my_handler():
    @cli_communication_handler(type=MessageType.REQUEST)
    def test_handler(payload, sender, receiver):
        return "test"

    registry = HandlerRegistry()
    assert len(registry.handlers) == 1
```

## Advanced Usage

### Dynamic Handler Registration

Register handlers at runtime:

```python
from promptchain.cli.communication.handlers import CommunicationHandler, HandlerRegistry

def create_dynamic_handler(agent_name):
    def handler(payload, sender, receiver):
        return f"{agent_name} processed"

    registry = HandlerRegistry()
    registry.register(CommunicationHandler(
        func=handler,
        name=f"{agent_name}_handler",
        message_types={MessageType.REQUEST},
        receivers={agent_name}
    ))

# Create handlers dynamically
create_dynamic_handler("worker1")
create_dynamic_handler("worker2")
```

### Handler Introspection

Inspect registered handlers:

```python
from promptchain.cli.communication import get_handler_registry

registry = get_handler_registry()

# List all handlers
for handler in registry.handlers:
    print(f"{handler.name}: priority={handler.priority}")
    print(f"  Types: {handler.message_types}")
    print(f"  Senders: {handler.senders or 'any'}")
    print(f"  Receivers: {handler.receivers or 'any'}")

# Find handlers for specific criteria
matches = registry.get_matching_handlers(
    MessageType.REQUEST,
    "supervisor",
    "worker"
)
print(f"Found {len(matches)} matching handlers")
```

## Complete Example: Multi-Agent System

```python
from promptchain.cli.communication import (
    cli_communication_handler,
    MessageType,
    MessageBus
)

# Supervisor agent
@cli_communication_handler(
    type=MessageType.RESPONSE,
    sender="worker",
    receiver="supervisor"
)
def supervisor_handle_response(payload, sender, receiver):
    task_id = payload["task_id"]
    result = payload["result"]
    print(f"Task {task_id} completed: {result}")
    return "acknowledged"

# Worker agent
@cli_communication_handler(
    type=MessageType.DELEGATION,
    sender="supervisor",
    receiver="worker"
)
def worker_handle_task(payload, sender, receiver):
    task = payload["task"]
    result = f"Processed {task}"
    return {"task_id": task, "result": result}

# Monitor agent (broadcast receiver)
@cli_communication_handler(type=MessageType.BROADCAST)
def monitor_all_messages(payload, sender, receiver):
    print(f"Monitor: {sender} -> {receiver}: {payload}")
    return "logged"

# Status tracker
@cli_communication_handler(type=MessageType.STATUS)
def track_status(payload, sender, receiver):
    progress = payload.get("progress", 0)
    print(f"Status from {sender}: {progress}%")
    return "status_recorded"

# Usage
async def run_multi_agent_workflow():
    bus = MessageBus(session_id="example_session")

    # Supervisor delegates work
    await bus.delegate(
        sender="supervisor",
        receiver="worker",
        payload={"task": "analyze_data"}
    )

    # Worker sends status update
    await bus.status_update(
        sender="worker",
        receiver="supervisor",
        payload={"progress": 50, "stage": "processing"}
    )

    # Coordinator broadcasts to all
    await bus.broadcast(
        sender="coordinator",
        payload={"message": "System maintenance in 1 hour"}
    )
```

## See Also

- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/communication/handlers.py` - Handler implementation
- `/home/gyasis/Documents/code/PromptChain/promptchain/cli/communication/message_bus.py` - MessageBus implementation
- `/home/gyasis/Documents/code/PromptChain/verify_communication_handlers.py` - Comprehensive tests
