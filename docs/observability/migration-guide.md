# Migration Guide: v0.4.0 → v0.4.1h

This guide helps you migrate from PromptChain v0.4.0 and earlier to the new observability system in v0.4.1h.

## Quick Summary

**What Changed**:
- ExecutionHistoryManager now has public APIs (no more private attributes)
- AgentChain and AgenticStepProcessor return rich metadata with `return_metadata=True`
- New event system with 33+ event types and callback support
- MCP operations now emit events

**Breaking Changes**: **NONE** - All changes are backward compatible!

**Deprecation Timeline**:
- v0.4.1h: Private attributes still work (with warnings)
- v0.5.0: Private attributes will be removed (Q2 2025)

## Migration Steps

### 1. ExecutionHistoryManager: Private → Public API

#### Before (v0.4.0 and earlier)
```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

history = ExecutionHistoryManager(max_tokens=4000)

# ❌ Using private attributes (deprecated)
token_count = history._current_tokens
entries = history._history
truncation_count = history._truncation_count
```

#### After (v0.4.1a+)
```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

history = ExecutionHistoryManager(max_tokens=4000)

# ✅ Using public API (stable)
token_count = history.current_token_count
entries = history.history
stats = history.get_statistics()
truncation_count = stats['truncation_count']
```

#### Migration Checklist
- [ ] Replace `history._current_tokens` with `history.current_token_count`
- [ ] Replace `history._history` with `history.history`
- [ ] Replace `len(history._history)` with `history.history_size`
- [ ] Use `history.get_statistics()` for detailed stats

### 2. Adding Execution Metadata

#### Before (v0.4.0 and earlier)
```python
from promptchain.utils.agent_chain import AgentChain

agent_chain = AgentChain(
    agents={"analyzer": analyzer_agent},
    execution_mode="router"
)

# Only got string response
result = agent_chain.process_input("Analyze this data")
print(result)  # Just a string
```

#### After (v0.4.1b+)
```python
from promptchain.utils.agent_chain import AgentChain

agent_chain = AgentChain(
    agents={"analyzer": analyzer_agent},
    execution_mode="router"
)

# Get rich metadata
result = agent_chain.process_input(
    "Analyze this data",
    return_metadata=True  # New parameter
)

# Now it's an AgentExecutionResult dataclass
print(f"Response: {result.response}")
print(f"Agent: {result.agent_name}")
print(f"Execution time: {result.execution_time_ms}ms")
print(f"Tools called: {len(result.tools_called)}")

# Or get just string (backward compatible)
result_str = agent_chain.process_input("Analyze this data")  # return_metadata=False
print(result_str)  # Still works as before
```

#### Migration Checklist
- [ ] Identify where you need execution metadata
- [ ] Add `return_metadata=True` to `process_input()` calls
- [ ] Update code to handle `AgentExecutionResult` instead of string
- [ ] Keep `return_metadata=False` (default) where metadata not needed

### 3. Adding AgenticStepProcessor Metadata

#### Before (v0.4.0 and earlier)
```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

agentic_step = AgenticStepProcessor(
    objective="Find patterns in data",
    max_internal_steps=5
)

# Only got final answer
result = await agentic_step.run_async("Search logs")
print(result)  # Just a string
```

#### After (v0.4.1c+)
```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

agentic_step = AgenticStepProcessor(
    objective="Find patterns in data",
    max_internal_steps=5
)

# Get rich metadata
result = await agentic_step.run_async(
    "Search logs",
    return_metadata=True  # New parameter
)

# Now it's an AgenticStepResult dataclass
print(f"Final answer: {result.final_answer}")
print(f"Total steps: {result.total_steps}")
print(f"Tools called: {result.total_tools_called}")

# Step-by-step details
for step in result.steps:
    print(f"Step {step.step_number}: {len(step.tool_calls)} tools, {step.execution_time_ms}ms")

# Or get just string (backward compatible)
result_str = await agentic_step.run_async("Search logs")  # return_metadata=False
print(result_str)  # Still works as before
```

#### Migration Checklist
- [ ] Identify agentic steps that need detailed metrics
- [ ] Add `return_metadata=True` to `run_async()` calls
- [ ] Update code to handle `AgenticStepResult` instead of string
- [ ] Keep `return_metadata=False` (default) for simple cases

### 4. Adding Event Callbacks

This is entirely new functionality - nothing to migrate, just add where needed.

#### New Feature (v0.4.1d+)
```python
from promptchain import PromptChain
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

# Define callback
def performance_monitor(event: ExecutionEvent):
    if "execution_time_ms" in event.metadata:
        print(f"{event.event_type.name}: {event.metadata['execution_time_ms']}ms")

# Register callback
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze: {input}", "Summarize: {input}"]
)

chain.register_callback(performance_monitor)

# Events will fire during execution
result = chain.process_prompt("Your input")
```

#### Usage Checklist
- [ ] Identify where you need execution monitoring
- [ ] Write callback functions (sync or async)
- [ ] Register callbacks with optional event filtering
- [ ] Unregister callbacks when no longer needed

### 5. MCP Event Monitoring

New feature for MCP server operations (v0.4.1f+).

```python
from promptchain import PromptChain
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

def mcp_monitor(event: ExecutionEvent):
    if event.event_type == ExecutionEventType.MCP_CONNECT_END:
        server_id = event.metadata["server_id"]
        status = event.metadata["status"]
        print(f"MCP {server_id}: {status}")

mcp_config = [{
    "id": "filesystem",
    "type": "stdio",
    "command": "mcp-server-filesystem",
    "args": ["--root", "./project"]
}]

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Read files: {input}"],
    mcp_servers=mcp_config
)

# Monitor MCP operations
chain.register_callback(
    mcp_monitor,
    event_filter=[
        ExecutionEventType.MCP_CONNECT_START,
        ExecutionEventType.MCP_CONNECT_END,
        ExecutionEventType.MCP_TOOL_DISCOVERED,
        ExecutionEventType.MCP_ERROR
    ]
)

result = chain.process_prompt("Read config.json")
```

## Complete Migration Example

### Before (v0.4.0)
```python
from promptchain import PromptChain
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain.utils.agent_chain import AgentChain

# ExecutionHistoryManager usage
history = ExecutionHistoryManager(max_tokens=4000)
history.add_entry("user_input", "Hello", source="user")

# ❌ Private attribute access
print(f"Tokens: {history._current_tokens}")
print(f"History: {history._history}")

# AgentChain usage
agent_chain = AgentChain(
    agents={"analyzer": analyzer_agent},
    execution_mode="router"
)

# Only string response
result = agent_chain.process_input("Analyze data")
print(result)  # String only

# No monitoring capabilities
# No metadata available
# No event system
```

### After (v0.4.1h)
```python
from promptchain import PromptChain
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

# ExecutionHistoryManager with public API
history = ExecutionHistoryManager(max_tokens=4000)
history.add_entry("user_input", "Hello", source="user")

# ✅ Public API access
print(f"Tokens: {history.current_token_count}")
print(f"History size: {history.history_size}")
stats = history.get_statistics()
print(f"Stats: {stats}")

# AgentChain with metadata
agent_chain = AgentChain(
    agents={"analyzer": analyzer_agent},
    execution_mode="router"
)

# Rich metadata
result = agent_chain.process_input(
    "Analyze data",
    return_metadata=True
)

print(f"Response: {result.response}")
print(f"Agent: {result.agent_name}")
print(f"Execution time: {result.execution_time_ms}ms")
print(f"Tools called: {len(result.tools_called)}")

# Event monitoring
def monitor_callback(event: ExecutionEvent):
    print(f"Event: {event.event_type.name} at {event.timestamp}")

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze: {input}"]
)

chain.register_callback(monitor_callback)

# Full observability
result = chain.process_prompt("Your input")
```

## Common Migration Patterns

### Pattern 1: Logging Execution Metrics

#### Before
```python
import time

start = time.time()
result = chain.process_prompt("Input")
end = time.time()

print(f"Execution time: {(end - start) * 1000}ms")
# No token count, no tool info, no step details
```

#### After
```python
# Option 1: Use metadata
result = agent_chain.process_input("Input", return_metadata=True)
print(f"Execution time: {result.execution_time_ms}ms")
print(f"Tokens: {result.total_tokens}")
print(f"Tools: {len(result.tools_called)}")

# Option 2: Use callbacks
def log_metrics(event: ExecutionEvent):
    if event.event_type == ExecutionEventType.CHAIN_END:
        print(f"Time: {event.metadata['execution_time_ms']}ms")
        print(f"Steps: {event.metadata['total_steps']}")

chain.register_callback(log_metrics, event_filter=ExecutionEventType.CHAIN_END)
```

### Pattern 2: Error Tracking

#### Before
```python
try:
    result = chain.process_prompt("Input")
except Exception as e:
    print(f"Error: {e}")
    # Limited error context
```

#### After
```python
# Option 1: Check metadata errors
result = agent_chain.process_input("Input", return_metadata=True)
if result.errors:
    print(f"Errors: {result.errors}")

# Option 2: Use error callbacks
def track_errors(event: ExecutionEvent):
    if event.event_type.name.endswith("_ERROR"):
        print(f"Error in {event.event_type.name}: {event.metadata.get('error')}")

chain.register_callback(
    track_errors,
    event_filter=[
        ExecutionEventType.CHAIN_ERROR,
        ExecutionEventType.STEP_ERROR,
        ExecutionEventType.MODEL_CALL_ERROR,
        ExecutionEventType.TOOL_CALL_ERROR
    ]
)
```

### Pattern 3: Debugging Agentic Steps

#### Before
```python
# No way to see internal steps
result = await agentic_step.run_async("Find patterns")
print(result)  # Just final answer
# No visibility into reasoning process
```

#### After
```python
result = await agentic_step.run_async("Find patterns", return_metadata=True)

print(f"Final answer: {result.final_answer}")
print(f"Took {result.total_steps} steps")

# See each step's details
for step in result.steps:
    print(f"\nStep {step.step_number}:")
    print(f"  Tools: {[tc['name'] for tc in step.tool_calls]}")
    print(f"  Time: {step.execution_time_ms}ms")
    print(f"  Tokens: {step.tokens_used}")
```

## Deprecation Warnings

Starting in v0.4.1h, you'll see warnings when using deprecated patterns:

```python
# This will show a deprecation warning
token_count = history._current_tokens

# Warning message:
# DeprecationWarning: Accessing _current_tokens is deprecated.
# Use history.current_token_count instead.
# Private attributes will be removed in v0.5.0.
```

## Testing Your Migration

### Step 1: Update Dependencies
```bash
pip install promptchain>=0.4.1h
```

### Step 2: Run Tests
```bash
# Run your existing tests - they should pass (backward compatible)
pytest

# Check for deprecation warnings
pytest -W default
```

### Step 3: Update Code Gradually
1. Fix deprecation warnings one by one
2. Add metadata collection where needed
3. Add callbacks for monitoring
4. Test thoroughly

### Step 4: Validate
```bash
# Should have no deprecation warnings
pytest -W error::DeprecationWarning
```

## Rollback Plan

If you encounter issues:

```bash
# Rollback to v0.4.0
pip install promptchain==0.4.0

# Or pin to last known good version
pip install promptchain==0.4.0
```

All v0.4.1h features are opt-in, so you can roll back safely.

## Support & Resources

- **Migration Issues**: [GitHub Issues](https://github.com/yourusername/promptchain/issues)
- **Examples**: [examples/observability/migration_example.py](../../examples/observability/migration_example.py)
- **API Docs**: [Public APIs Guide](public-apis.md)
- **Events Guide**: [Event System Guide](event-system.md)

## FAQ

**Q: Will my code break if I upgrade to v0.4.1h?**
A: No! All changes are backward compatible. Existing code will work unchanged.

**Q: When will private attributes be removed?**
A: Planned for v0.5.0 (Q2 2025). You'll have several months to migrate.

**Q: Do I have to use the new features?**
A: No! They're all opt-in. Use them only where you need enhanced observability.

**Q: Is there a performance impact?**
A: Minimal. Public APIs have zero overhead. Metadata and callbacks add ~1-2% when enabled.

**Q: Can I use some features and not others?**
A: Yes! Mix and match. Use public APIs without callbacks, or callbacks without metadata.

## Next Steps

1. Review [Public APIs Guide](public-apis.md) for API details
2. Check [Event System Guide](event-system.md) for callback patterns
3. Try [examples/migration_example.py](../../examples/observability/migration_example.py)
4. Read [Best Practices](best-practices.md) for recommendations
