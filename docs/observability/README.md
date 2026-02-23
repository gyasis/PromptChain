# PromptChain Observability System

**Version**: 0.4.1h
**Status**: Production Ready

## Overview

PromptChain's observability system provides comprehensive visibility into execution lifecycle, enabling monitoring, debugging, and performance analysis. The system consists of three main components:

1. **Public APIs**: Access execution state and statistics
2. **Event System**: React to execution events in real-time
3. **Execution Metadata**: Rich metadata from execution results

## Quick Start

### Basic Callback Usage

```python
from promptchain import PromptChain
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

# Define a callback function
def log_events(event: ExecutionEvent):
    print(f"[{event.event_type.name}] Step {event.step_number}: {event.model_name}")
    if "execution_time_ms" in event.metadata:
        print(f"  Time: {event.metadata['execution_time_ms']}ms")

# Create chain with callback
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze: {input}", "Summarize: {input}"],
    verbose=True
)

# Register callback
chain.register_callback(log_events)

# Execute and observe events
result = chain.process_prompt("Tell me about quantum computing")
```

### Getting Execution Metadata

```python
from promptchain.utils.agent_chain import AgentChain

# Create agent chain
agent_chain = AgentChain(
    agents={"analyzer": analyzer_agent, "writer": writer_agent},
    execution_mode="router"
)

# Get detailed metadata
result = agent_chain.process_input(
    "Analyze this data and write a report",
    return_metadata=True
)

# Access metadata
print(f"Agent: {result.agent_name}")
print(f"Execution time: {result.execution_time_ms}ms")
print(f"Tools called: {len(result.tools_called)}")
print(f"Total tokens: {result.total_tokens}")
```

### Using History Manager Public API

```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

# Create history manager
history = ExecutionHistoryManager(max_tokens=4000, max_entries=100)

# Use public properties (no more private attributes!)
print(f"Current tokens: {history.current_token_count}")
print(f"History size: {history.history_size}")

# Get statistics
stats = history.get_statistics()
print(f"Total entries: {stats['total_entries']}")
print(f"Memory usage: {stats['memory_usage_bytes']} bytes")
```

## Key Features

### 1. Public APIs (v0.4.1a)
- **ExecutionHistoryManager**: Public properties for token count, history access, statistics
- **No more private attributes**: Stable API that won't break
- **Thread-safe**: All properties are thread-safe

### 2. Execution Metadata (v0.4.1b-c)
- **AgentExecutionResult**: Complete metadata from AgentChain execution
- **AgenticStepResult**: Detailed agentic reasoning metadata
- **StepExecutionMetadata**: Per-step execution details
- **return_metadata parameter**: Opt-in metadata collection

### 3. Event System (v0.4.1d-f)
- **33+ event types**: Complete lifecycle coverage
- **Event filtering**: Subscribe to specific event types
- **Async/sync callbacks**: Both patterns supported
- **MCP events**: Monitor MCP server lifecycle and tool discovery

### 4. Backward Compatibility (v0.4.1g)
- **Opt-in design**: All features are optional
- **No breaking changes**: Existing code works unchanged
- **Deprecation warnings**: Clear migration path for private attributes

## Documentation Structure

- **[Public APIs Guide](public-apis.md)**: ExecutionHistoryManager, AgentChain, AgenticStepProcessor
- **[Event System Guide](event-system.md)**: Events, callbacks, and filtering
- **[MCP Events Guide](mcp-events.md)**: MCP server monitoring
- **[Migration Guide](migration-guide.md)**: Migrate from private to public APIs
- **[Best Practices](best-practices.md)**: Patterns and recommendations

## Examples

All examples are in `examples/observability/`:

- **[basic_callbacks.py](../../examples/observability/basic_callbacks.py)**: Simple callback usage
- **[event_filtering.py](../../examples/observability/event_filtering.py)**: Filtering specific events
- **[execution_metadata.py](../../examples/observability/execution_metadata.py)**: Using return_metadata
- **[monitoring_dashboard.py](../../examples/observability/monitoring_dashboard.py)**: Real-world monitoring
- **[migration_example.py](../../examples/observability/migration_example.py)**: Old → new migration

## Event Types

### Chain Lifecycle
- `CHAIN_START`, `CHAIN_END`, `CHAIN_ERROR`

### Step Execution
- `STEP_START`, `STEP_END`, `STEP_ERROR`, `STEP_SKIPPED`

### Model Calls
- `MODEL_CALL_START`, `MODEL_CALL_END`, `MODEL_CALL_ERROR`

### Tool Calls
- `TOOL_CALL_START`, `TOOL_CALL_END`, `TOOL_CALL_ERROR`

### Agentic Steps
- `AGENTIC_STEP_START`, `AGENTIC_STEP_END`, `AGENTIC_STEP_ERROR`
- `AGENTIC_INTERNAL_STEP`

### MCP Operations
- `MCP_CONNECT_START`, `MCP_CONNECT_END`
- `MCP_DISCONNECT_START`, `MCP_DISCONNECT_END`
- `MCP_TOOL_DISCOVERED`, `MCP_ERROR`

### Other Events
- `CHAIN_BREAK`, `HISTORY_TRUNCATED`
- `MODEL_LOAD`, `MODEL_UNLOAD`

## Performance

- **Zero overhead without callbacks**: Early return when no callbacks registered
- **Concurrent callback execution**: Callbacks run in parallel with `asyncio.gather`
- **Error isolation**: Callback errors don't affect execution
- **Lightweight events**: Minimal memory footprint

## Use Cases

1. **Production Monitoring**: Track execution metrics, errors, performance
2. **Debugging**: Detailed event traces for troubleshooting
3. **Analytics**: Collect data for usage analysis and optimization
4. **Logging**: Custom logging integration with event callbacks
5. **Testing**: Validate execution flow in automated tests

## Version History

- **0.4.1a**: ExecutionHistoryManager public API
- **0.4.1b**: AgentChain execution metadata
- **0.4.1c**: AgenticStepProcessor metadata tracking
- **0.4.1d**: PromptChain callback system
- **0.4.1e**: Event integration throughout PromptChain
- **0.4.1f**: MCPHelper event callbacks
- **0.4.1g**: Backward compatibility validation
- **0.4.1h**: Documentation and examples (current)

## Next Steps

1. Read the [Public APIs Guide](public-apis.md) to understand available APIs
2. Review the [Event System Guide](event-system.md) to learn about callbacks
3. Check out [examples/observability/](../../examples/observability/) for working code
4. Consult the [Migration Guide](migration-guide.md) if upgrading from older versions

## Support

For questions and issues:
- GitHub Issues: [PromptChain Issues](https://github.com/yourusername/promptchain/issues)
- Documentation: [Full Documentation](../index.md)
- Examples: [Working Examples](../../examples/)
