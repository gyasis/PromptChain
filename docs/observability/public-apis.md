# Public APIs Guide

This guide covers the public APIs introduced in PromptChain's observability system (v0.4.1a-h).

## ExecutionHistoryManager Public API (v0.4.1a)

### Overview

ExecutionHistoryManager now provides public properties and methods for accessing execution history state. **No more private attribute access!**

### Public Properties

#### `current_token_count`
```python
@property
def current_token_count(self) -> int:
    """Get current total token count in history.

    Returns:
        Total tokens across all history entries
    """
```

**Example**:
```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

history = ExecutionHistoryManager(max_tokens=4000)
history.add_entry("user_input", "Hello, world!", source="user")

# Access token count (public API)
print(f"Current tokens: {history.current_token_count}")  # ✅ Correct

# Old way (deprecated)
# print(f"Tokens: {history._current_tokens}")  # ❌ Don't do this
```

#### `history`
```python
@property
def history(self) -> List[Dict[str, Any]]:
    """Get copy of current history entries.

    Returns:
        List of history entry dictionaries (safe copy, not reference)
    """
```

**Example**:
```python
# Get history safely
for entry in history.history:
    print(f"[{entry['entry_type']}] {entry['content'][:50]}")

# Safe to modify (it's a copy)
history_copy = history.history
history_copy[0]["content"] = "Modified"  # Doesn't affect original
```

#### `history_size`
```python
@property
def history_size(self) -> int:
    """Get number of entries in history.

    Returns:
        Total number of history entries
    """
```

**Example**:
```python
print(f"History has {history.history_size} entries")

# Old way (deprecated)
# print(f"Size: {len(history._history)}")  # ❌ Don't do this
```

### Public Methods

#### `get_statistics()`
```python
def get_statistics(self) -> Dict[str, Any]:
    """Get comprehensive statistics about history manager state.

    Returns:
        Dictionary with statistics:
        - total_entries: Number of entries
        - current_token_count: Total tokens
        - max_tokens: Maximum token limit
        - max_entries: Maximum entry limit
        - truncation_strategy: Current strategy
        - memory_usage_bytes: Estimated memory usage
        - entries_by_type: Count of entries by type
    """
```

**Example**:
```python
stats = history.get_statistics()

print(f"Total entries: {stats['total_entries']}")
print(f"Token usage: {stats['current_token_count']}/{stats['max_tokens']}")
print(f"Memory usage: {stats['memory_usage_bytes']} bytes")
print(f"Entry types: {stats['entries_by_type']}")

# Example output:
# {
#     "total_entries": 15,
#     "current_token_count": 3245,
#     "max_tokens": 4000,
#     "max_entries": 100,
#     "truncation_strategy": "oldest_first",
#     "memory_usage_bytes": 125000,
#     "entries_by_type": {
#         "user_input": 5,
#         "agent_output": 5,
#         "tool_call": 3,
#         "tool_result": 2
#     }
# }
```

### Migration from Private Attributes

**Before (v0.4.0 and earlier)**:
```python
# ❌ Using private attributes (will break in future versions)
token_count = history._current_tokens
history_list = history._history
entry_count = len(history._history)
```

**After (v0.4.1a+)**:
```python
# ✅ Using public API (stable, won't break)
token_count = history.current_token_count
history_list = history.history
entry_count = history.history_size
```

## AgentChain Execution Metadata (v0.4.1b)

### Overview

AgentChain now returns rich execution metadata when `return_metadata=True` is passed to `process_input()`.

### AgentExecutionResult

```python
@dataclass
class AgentExecutionResult:
    """Complete execution metadata from AgentChain."""

    # Response data
    response: str
    agent_name: str

    # Execution metadata
    execution_time_ms: float
    start_time: datetime
    end_time: datetime

    # Routing information
    router_decision: Optional[Dict[str, Any]] = None
    router_steps: int = 0
    fallback_used: bool = False

    # Agent execution details
    agent_execution_metadata: Optional[Dict[str, Any]] = None
    tools_called: List[Dict[str, Any]] = field(default_factory=list)

    # Token usage
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # Cache information
    cache_hit: bool = False
    cache_key: Optional[str] = None

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
```

### Usage

```python
from promptchain.utils.agent_chain import AgentChain

# Create agent chain
agent_chain = AgentChain(
    agents={"analyzer": analyzer_agent, "writer": writer_agent},
    execution_mode="router"
)

# Get metadata
result = agent_chain.process_input(
    "Analyze quarterly sales data",
    return_metadata=True  # Enable metadata
)

# Access execution metadata
print(f"Agent: {result.agent_name}")
print(f"Execution time: {result.execution_time_ms}ms")
print(f"Router decision: {result.router_decision}")
print(f"Tools called: {len(result.tools_called)}")

# Check for errors
if result.errors:
    print(f"Errors encountered: {result.errors}")

# Get summary
summary = result.to_summary_dict()
print(f"Summary: {summary}")
```

### Router Decision Metadata

When using router mode, `router_decision` contains:

```python
{
    "chosen_agent": "analyzer",
    "refined_query": "Analyze Q3 2024 sales trends",
    "confidence": 0.95,
    "reasoning": "Sales analysis requires data analysis capabilities",
    "alternatives_considered": ["writer", "summarizer"]
}
```

### Tool Call Metadata

Each tool call in `tools_called` includes:

```python
{
    "name": "search_database",
    "args": {"query": "SELECT * FROM sales WHERE quarter = 'Q3'"},
    "result": "Found 1500 records",
    "execution_time_ms": 234.5,
    "success": True
}
```

## AgenticStepProcessor Metadata (v0.4.1c)

### Overview

AgenticStepProcessor provides detailed step-by-step execution metadata when `return_metadata=True`.

### AgenticStepResult

```python
@dataclass
class AgenticStepResult:
    """Complete execution result from AgenticStepProcessor."""

    # Final output
    final_answer: str

    # Execution summary
    total_steps: int
    max_steps_reached: bool
    objective_achieved: bool

    # Step-by-step details
    steps: List[StepExecutionMetadata] = field(default_factory=list)

    # Overall statistics
    total_tools_called: int = 0
    total_tokens_used: int = 0
    total_execution_time_ms: float = 0.0

    # Configuration used
    history_mode: str = "minimal"
    max_internal_steps: int = 5
    model_name: Optional[str] = None

    # Errors/warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
```

### StepExecutionMetadata

```python
@dataclass
class StepExecutionMetadata:
    """Metadata for a single agentic step."""

    step_number: int
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tokens_used: int = 0
    execution_time_ms: float = 0.0
    clarification_attempts: int = 0
    error: Optional[str] = None
```

### Usage

```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

# Create agentic step with metadata
agentic_step = AgenticStepProcessor(
    objective="Find and analyze error patterns in logs",
    max_internal_steps=8,
    model_name="openai/gpt-4"
)

# Run with metadata
result = await agentic_step.run_async(
    initial_input="Search error logs from last 24 hours",
    return_metadata=True
)

# Access step-by-step details
print(f"Total steps executed: {result.total_steps}")
print(f"Objective achieved: {result.objective_achieved}")
print(f"Tools called: {result.total_tools_called}")
print(f"Total tokens: {result.total_tokens_used}")

# Analyze each step
for step in result.steps:
    print(f"\nStep {step.step_number}:")
    print(f"  Tools: {len(step.tool_calls)}")
    print(f"  Tokens: {step.tokens_used}")
    print(f"  Time: {step.execution_time_ms}ms")

    for tool_call in step.tool_calls:
        print(f"  - {tool_call['name']}: {tool_call['result'][:50]}")
```

### History Modes

AgenticStepProcessor supports three history modes (v0.4.0+):

```python
# Minimal mode (default): Only current step context
agentic_step = AgenticStepProcessor(
    objective="...",
    history_mode="minimal"
)

# Accumulative mode: Full conversation history
agentic_step = AgenticStepProcessor(
    objective="...",
    history_mode="accumulative"
)

# Summary mode: Summarized history
agentic_step = AgenticStepProcessor(
    objective="...",
    history_mode="summary"
)
```

Metadata includes which mode was used: `result.history_mode`

## Best Practices

### 1. Use Public APIs Only

```python
# ✅ Good: Use public properties
token_count = history.current_token_count
entries = history.history

# ❌ Bad: Use private attributes
token_count = history._current_tokens  # Will break in future
entries = history._history  # Not stable
```

### 2. Enable Metadata Selectively

```python
# Enable metadata for analysis/debugging
result = agent_chain.process_input(prompt, return_metadata=True)

# Disable for production (slight performance improvement)
result = agent_chain.process_input(prompt, return_metadata=False)
```

### 3. Check for Errors

```python
result = agent_chain.process_input(prompt, return_metadata=True)

if result.errors:
    print(f"Execution had errors: {result.errors}")

if result.warnings:
    print(f"Warnings: {result.warnings}")
```

### 4. Use Summary for Logging

```python
# Full metadata (for debugging)
full_data = result.to_dict()
logger.debug(f"Full execution data: {full_data}")

# Summary only (for production logging)
summary = result.to_summary_dict()
logger.info(f"Execution summary: {summary}")
```

## Performance Impact

- **ExecutionHistoryManager public properties**: Zero overhead (direct attribute access)
- **Execution metadata**: Minimal overhead when `return_metadata=True`
  - Adds ~1-2% execution time
  - Additional memory for metadata storage
- **Without metadata**: No performance impact (`return_metadata=False` or omitted)

## Thread Safety

- **ExecutionHistoryManager properties**: Thread-safe (return copies)
- **AgentExecutionResult**: Immutable dataclass (thread-safe)
- **AgenticStepResult**: Immutable dataclass (thread-safe)

## Next Steps

- Read the [Event System Guide](event-system.md) for callback-based observability
- Check [examples/execution_metadata.py](../../examples/observability/execution_metadata.py) for working examples
- Review the [Migration Guide](migration-guide.md) for upgrading existing code
