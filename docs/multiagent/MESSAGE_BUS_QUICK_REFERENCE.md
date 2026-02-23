# MessageBus Quick Reference

**Location**: `promptchain/cli/communication/message_bus.py`

## Quick Start

```python
import asyncio
from promptchain.cli.communication import MessageBus, MessageType, cli_communication_handler

# 1. Create message bus
bus = MessageBus(session_id="my-session")

# 2. Register handlers
@cli_communication_handler(type=MessageType.REQUEST, receiver="worker")
async def handle_work(payload, sender, receiver):
    return {"status": "done"}

# 3. Send messages
async def main():
    msg = await bus.request("boss", "worker", {"task": "job1"})
    print(f"Delivered: {msg.delivered}")

asyncio.run(main())
```

## Core API

### MessageBus Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `send()` | `async send(sender, receiver, message_type, payload)` | Send message to specific agent |
| `broadcast()` | `async broadcast(sender, payload, message_type=BROADCAST)` | Send to all agents (receiver="*") |
| `request()` | `async request(sender, receiver, payload)` | Send REQUEST message |
| `respond()` | `async respond(sender, receiver, payload)` | Send RESPONSE message |
| `delegate()` | `async delegate(sender, receiver, payload)` | Send DELEGATION message |
| `status_update()` | `async status_update(sender, receiver, payload)` | Send STATUS message |
| `get_history()` | `get_history(sender=None, receiver=None, message_type=None, limit=100)` | Get message history |
| `clear_history()` | `clear_history()` | Clear all messages |

### Message Fields

```python
@dataclass
class Message:
    message_id: str          # Auto-generated UUID
    sender: str              # Sending agent
    receiver: str            # Receiving agent ("*" for broadcast)
    type: MessageType        # Message type enum
    payload: Dict[str, Any]  # Message content
    timestamp: float         # Creation timestamp
    delivered: bool          # Was delivered to handlers
```

### MessageType Enum

```python
MessageType.REQUEST     # Request for action/data
MessageType.RESPONSE    # Response to request
MessageType.BROADCAST   # Announcement to all
MessageType.DELEGATION  # Task delegation
MessageType.STATUS      # Status update
```

## Handler Registration

### Basic Handler

```python
@cli_communication_handler(type=MessageType.REQUEST)
async def my_handler(payload, sender, receiver):
    # Process message
    return {"result": "success"}
```

### Filtered Handler

```python
@cli_communication_handler(
    type=MessageType.REQUEST,
    sender="agent1",           # Only from agent1
    receiver="agent2",         # Only to agent2
    priority=10                # Higher priority (default 0)
)
async def filtered_handler(payload, sender, receiver):
    return {"status": "handled"}
```

### Multiple Types

```python
@cli_communication_handler(
    types=[MessageType.REQUEST, MessageType.DELEGATION]
)
async def multi_type_handler(payload, sender, receiver):
    return {"processed": True}
```

## Usage Patterns

### Direct Messaging

```python
# Send request
request_msg = await bus.request(
    sender="coordinator",
    receiver="analyzer",
    payload={"task": "analyze", "data": "dataset.csv"}
)

# Send response
response_msg = await bus.respond(
    sender="analyzer",
    receiver="coordinator",
    payload={"status": "complete", "result": {...}}
)
```

### Broadcasting

```python
# Broadcast to all agents
broadcast_msg = await bus.broadcast(
    sender="coordinator",
    payload={"message": "Starting workflow", "phase": 1}
)

# All handlers with type=BROADCAST will receive this
```

### Task Delegation

```python
# Delegate task to another agent
delegate_msg = await bus.delegate(
    sender="coordinator",
    receiver="worker",
    payload={
        "task": "process_data",
        "input": "file.csv",
        "deadline": "5min"
    }
)
```

### Status Updates

```python
# Send status update
status_msg = await bus.status_update(
    sender="worker",
    receiver="coordinator",
    payload={
        "status": "in_progress",
        "completion": 0.5,
        "eta": "2min"
    }
)
```

## History Management

### Get All History

```python
# Get recent messages (newest first)
history = bus.get_history(limit=50)

for msg in history:
    print(f"{msg.sender} → {msg.receiver}: {msg.type.value}")
```

### Filter History

```python
# Messages from specific sender
from_alice = bus.get_history(sender="alice")

# Messages to specific receiver
to_bob = bus.get_history(receiver="bob")

# Messages of specific type
requests = bus.get_history(message_type=MessageType.REQUEST)

# Combined filters
alice_requests = bus.get_history(
    sender="alice",
    message_type=MessageType.REQUEST,
    limit=10
)
```

### Clear History

```python
# Clear all messages
count = bus.clear_history()
print(f"Cleared {count} messages")
```

## Activity Logging

### Setup Logging

```python
# Define callback
def log_activity(entry):
    print(f"{entry['event_type']}: {entry['sender']} → {entry['receiver']}")
    # Save to database, file, etc.

# Create bus with logging
bus = MessageBus(
    session_id="my-session",
    activity_logger=log_activity
)
```

### Activity Log Structure

```python
{
    "event_type": "message_sent",      # or "message_delivered"
    "session_id": "my-session",
    "timestamp": "2025-11-28T10:30:00",
    "message_id": "abc123...",
    "sender": "agent1",
    "receiver": "agent2",
    "type": "request",
    "payload_keys": ["task", "data"]  # Keys from payload
}
```

## Message Serialization

### Save/Load Messages

```python
# Serialize message
msg = await bus.request("a", "b", {"data": "test"})
data = msg.to_dict()

# Save to JSON
import json
with open("message.json", "w") as f:
    json.dump(data, f)

# Load from JSON
with open("message.json") as f:
    loaded_data = json.load(f)

# Deserialize
from promptchain.cli.communication.message_bus import Message
restored_msg = Message.from_dict(loaded_data)
```

### Save History

```python
# Get all history
history = bus.get_history()

# Serialize all messages
serialized = [msg.to_dict() for msg in history]

# Save to JSONL
import json
with open("history.jsonl", "w") as f:
    for msg in serialized:
        f.write(json.dumps(msg) + "\n")
```

## Error Handling

### Fail-Safe Design

```python
# Handlers that raise exceptions don't crash the system
@cli_communication_handler(type=MessageType.REQUEST)
async def faulty_handler(payload, sender, receiver):
    raise ValueError("Something went wrong")

# Message still sent, system continues
msg = await bus.request("a", "b", {"test": "data"})
# Error logged, but system keeps running
```

### Activity Logger Errors

```python
# Even if activity logger fails, messages still work
def faulty_logger(entry):
    raise RuntimeError("Logger error")

bus = MessageBus(session_id="test", activity_logger=faulty_logger)

# Still works, error logged but doesn't crash
msg = await bus.request("a", "b", {"data": "test"})
```

## Best Practices

### 1. Use Specific Message Types

```python
# Good - Clear intent
await bus.request("a", "b", {"task": "analyze"})
await bus.respond("b", "a", {"result": "done"})

# Avoid - Generic send
await bus.send("a", "b", MessageType.REQUEST, {"task": "analyze"})
```

### 2. Filter Handlers Specifically

```python
# Good - Specific handler
@cli_communication_handler(type=MessageType.REQUEST, receiver="analyzer")
async def handle_analysis(payload, sender, receiver):
    pass

# Avoid - Too broad
@cli_communication_handler()  # Receives ALL messages
async def handle_all(payload, sender, receiver):
    pass
```

### 3. Structure Payloads Consistently

```python
# Good - Consistent structure
await bus.request("a", "b", {
    "task": "analyze",
    "input": {...},
    "options": {...}
})

# Avoid - Inconsistent payloads
await bus.request("a", "b", {"x": 1, "y": 2})  # What does this mean?
```

### 4. Use Activity Logging

```python
# Good - Track all communication
def log_to_session(entry):
    session.add_activity_log(entry)

bus = MessageBus(session_id=session.id, activity_logger=log_to_session)

# Avoid - No logging (harder to debug)
bus = MessageBus(session_id=session.id)
```

### 5. Clean Up History Periodically

```python
# In long-running sessions
if len(bus.get_history()) > 1000:
    # Archive old messages
    archive = bus.get_history(limit=500)
    save_to_file(archive)

    # Clear from memory
    bus.clear_history()
```

## Common Patterns

### Request-Response Pattern

```python
# Agent A requests work
request_msg = await bus.request(
    sender="coordinator",
    receiver="worker",
    payload={"task": "job1", "request_id": "req-123"}
)

# Agent B processes and responds
@cli_communication_handler(type=MessageType.REQUEST, receiver="worker")
async def handle_work(payload, sender, receiver):
    result = process_task(payload["task"])

    # Send response
    await bus.respond(
        sender="worker",
        receiver=sender,  # Back to original sender
        payload={
            "request_id": payload["request_id"],
            "result": result,
            "status": "complete"
        }
    )
```

### Broadcast-Acknowledge Pattern

```python
# Coordinator broadcasts
broadcast = await bus.broadcast(
    sender="coordinator",
    payload={"command": "start", "workflow_id": "wf-123"}
)

# All agents acknowledge
@cli_communication_handler(type=MessageType.BROADCAST)
async def acknowledge_broadcast(payload, sender, receiver):
    await bus.status_update(
        sender="agent",  # This agent's name
        receiver=sender,  # Back to broadcaster
        payload={
            "workflow_id": payload["workflow_id"],
            "status": "acknowledged"
        }
    )
```

### Multi-Hop Delegation

```python
# A delegates to B
await bus.delegate("agent_a", "agent_b", {"task": "step1"})

# B processes and delegates to C
@cli_communication_handler(type=MessageType.DELEGATION, receiver="agent_b")
async def handle_step1(payload, sender, receiver):
    result = process_step1(payload)

    await bus.delegate(
        sender="agent_b",
        receiver="agent_c",
        payload={"task": "step2", "input": result}
    )
```

## Performance Tips

1. **Use History Filtering**: Don't load full history if you only need specific messages
   ```python
   # Good - Filter at query time
   recent_requests = bus.get_history(message_type=MessageType.REQUEST, limit=10)

   # Avoid - Load all then filter
   all = bus.get_history()
   requests = [m for m in all if m.type == MessageType.REQUEST]
   ```

2. **Clear Old History**: Prevent unbounded memory growth
   ```python
   # Periodically clear old messages
   if bus.get_history().__len__() > 1000:
       bus.clear_history()
   ```

3. **Use Priority for Critical Handlers**: Ensure important handlers run first
   ```python
   @cli_communication_handler(type=MessageType.REQUEST, priority=100)
   async def critical_handler(payload, sender, receiver):
       pass
   ```

## Troubleshooting

### Messages Not Delivered

```python
# Check if handlers are registered
from promptchain.cli.communication.handlers import get_handler_registry
registry = get_handler_registry()
print(f"Registered handlers: {len(registry.handlers)}")

# Check handler filters
msg = await bus.request("a", "b", {"test": "data"})
if not msg.delivered:
    print("No matching handlers found")
    print(f"Looking for: type=REQUEST, receiver=b")
```

### Handler Not Called

```python
# Verify handler registration
@cli_communication_handler(type=MessageType.REQUEST, receiver="worker")
async def my_handler(payload, sender, receiver):
    print("Handler called!")  # Add debug print
    return {"status": "ok"}

# Check if message matches filters
# Handler filters: receiver="worker"
# Your message: receiver="worker" ✅
# Your message: receiver="other" ❌
```

### Activity Logger Not Working

```python
# Verify callback is set
bus = MessageBus(
    session_id="test",
    activity_logger=lambda e: print(f"LOG: {e}")  # Should print
)

# Check for exceptions in logger
def safe_logger(entry):
    try:
        your_logging_function(entry)
    except Exception as e:
        print(f"Logger error: {e}")

bus = MessageBus(session_id="test", activity_logger=safe_logger)
```

## Integration Example

```python
# Complete integration with SessionManager
class SessionManager:
    def __init__(self):
        self.session_id = "session-123"
        self.activity_log = []
        self.message_bus = MessageBus(
            session_id=self.session_id,
            activity_logger=self._log_activity
        )

    def _log_activity(self, entry):
        """Save activity to session log."""
        self.activity_log.append(entry)
        # Could also save to database, file, etc.

    async def save_session(self):
        """Save session including message history."""
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.message_bus.get_history()],
            "activity_log": self.activity_log
        }

    async def load_session(self, data):
        """Restore session from saved data."""
        # Restore messages (optional - history is usually session-specific)
        for msg_data in data.get("messages", []):
            msg = Message.from_dict(msg_data)
            # Could add to history if needed

        self.activity_log = data.get("activity_log", [])
```

---

**See Also**:
- `MESSAGE_BUS_IMPLEMENTATION_SUMMARY.md` - Detailed architecture guide
- `MESSAGE_BUS_COMPLETION_REPORT.md` - Implementation status
- `examples/message_bus_example.py` - Working examples
