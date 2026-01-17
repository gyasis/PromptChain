# ExecutionHistoryManager

**Version:** v0.4.2
**Module:** `promptchain.utils.execution_history_manager`

## Overview

The `ExecutionHistoryManager` is a sophisticated history management system designed for complex agentic workflows. It provides structured, token-aware conversation history tracking with automatic truncation, filtering capabilities, and a public API for monitoring and statistics.

**Why ExecutionHistoryManager exists:**
- **Token Management**: Prevents context window overflow by tracking token usage with tiktoken
- **Structured Tracking**: Organizes conversation events by type (user_input, agent_output, tool_call, etc.)
- **Flexible Retrieval**: Filter and format history based on use case requirements
- **Production Ready**: Built-in statistics, monitoring, and observability support

Unlike simple list-based history storage, ExecutionHistoryManager provides:
- Accurate token counting (via tiktoken)
- Automatic truncation when limits are reached
- Type-safe entry classification
- Multiple output formats (chat, JSON, content-only)
- Integration with CallbackManager for observability events

## Key Features

### 1. Token-Aware History Tracking
- Uses `tiktoken` (cl100k_base encoding) for accurate token counting
- Falls back to character estimation if tiktoken unavailable
- Tracks running token count for performance

### 2. Structured Entry Types
Seven predefined entry types for comprehensive tracking:

| Entry Type | Description | Common Source |
|-----------|-------------|---------------|
| `user_input` | User messages and queries | User, Application |
| `agent_output` | Agent responses | PromptChain, AgentChain |
| `tool_call` | Tool invocation requests | LLM, AgenticStep |
| `tool_result` | Tool execution results | Tool Functions, MCP |
| `system_message` | System notifications | Framework |
| `error` | Error messages | Exception Handlers |
| `custom` | Application-specific events | Custom Code |

### 3. Automatic Truncation Strategies

**Current Strategy:**
- `oldest_first`: Removes oldest entries first when limits exceeded

**Future Strategies (Placeholder):**
- `keep_last`: Keep most recent N entries
- `priority_based`: Keep high-priority entries
- `semantic_compression`: Compress similar entries

### 4. Filtering Capabilities
- Filter by entry type (include/exclude)
- Filter by source identifier (include/exclude)
- Limit by number of entries
- Limit by token count
- Multiple filters can be combined

### 5. Multiple Format Styles
- `chat`: Human-readable chat format with role labels
- `full_json`: Complete JSON representation with metadata
- `content_only`: Plain text content without metadata

### 6. Public API for Statistics and Monitoring
Introduced in v0.4.1a, provides production-ready observability:
- Current token count
- Entry count and distribution
- Utilization percentage
- Truncation events via CallbackManager

## Initialization & Configuration

### Basic Initialization

```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

# Minimal setup - unlimited history
history_manager = ExecutionHistoryManager()

# With token limit only
history_manager = ExecutionHistoryManager(
    max_tokens=4000  # Keeps ~4000 tokens of history
)

# With entry limit only
history_manager = ExecutionHistoryManager(
    max_entries=50  # Keeps last 50 entries
)

# Full configuration
history_manager = ExecutionHistoryManager(
    max_entries=50,              # Max 50 entries
    max_tokens=4000,             # Max 4000 tokens
    truncation_strategy="oldest_first",  # Remove oldest first
    callback_manager=my_callback_mgr     # For observability events
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_entries` | `Optional[int]` | `None` | Maximum number of history entries to keep. `None` = unlimited |
| `max_tokens` | `Optional[int]` | `None` | Maximum total tokens allowed in history. `None` = unlimited |
| `truncation_strategy` | `TruncationStrategy` | `"oldest_first"` | Method used when truncating history |
| `callback_manager` | `Optional[CallbackManager]` | `None` | For emitting HISTORY_TRUNCATED events |

**Truncation Priority:**
1. Token limits are checked and applied first
2. Entry limits are checked and applied second
3. Both can be active simultaneously

## Core Methods Documentation

### add_entry()

Adds a new entry to the history and applies truncation rules.

```python
def add_entry(
    entry_type: HistoryEntryType,
    content: Any,
    source: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None
```

**Parameters:**
- `entry_type`: One of the predefined entry types (see Entry Types Reference)
- `content`: The actual content (string, dict, list, etc.)
- `source`: Identifier of the source (e.g., "agent_chain", "math_agent", "user")
- `metadata`: Additional structured data for the entry

**Example:**
```python
# User input
history_manager.add_entry(
    entry_type="user_input",
    content="What is 2 + 2?",
    source="user"
)

# Agent response
history_manager.add_entry(
    entry_type="agent_output",
    content="The answer is 4.",
    source="math_agent",
    metadata={"confidence": 1.0, "execution_time_ms": 234}
)

# Tool call
history_manager.add_entry(
    entry_type="tool_call",
    content='{"function": "calculate", "args": {"expr": "2+2"}}',
    source="agentic_step",
    metadata={"tool_name": "calculator"}
)

# Error
history_manager.add_entry(
    entry_type="error",
    content="Connection timeout after 30s",
    source="mcp_helper",
    metadata={"error_code": "TIMEOUT", "retry_count": 3}
)
```

### get_history()

Returns the complete current history as a list of entry dictionaries.

```python
def get_history() -> List[Dict[str, Any]]
```

**Returns:** Copy of the internal history list (safe from external modification)

**Example:**
```python
full_history = history_manager.get_history()

for entry in full_history:
    print(f"{entry['type']} from {entry['source']}: {entry['content'][:50]}...")
    print(f"  Timestamp: {entry['timestamp']}")
    print(f"  Metadata: {entry['metadata']}")
```

### get_formatted_history()

Retrieves and formats history with filtering and limits.

```python
def get_formatted_history(
    include_types: Optional[List[HistoryEntryType]] = None,
    include_sources: Optional[List[str]] = None,
    exclude_types: Optional[List[HistoryEntryType]] = None,
    exclude_sources: Optional[List[str]] = None,
    max_entries: Optional[int] = None,
    max_tokens: Optional[int] = None,
    format_style: Literal['chat', 'full_json', 'content_only'] = 'chat'
) -> str
```

**Filtering Logic (Applied in Order):**
1. **Type/Source Filters**: Include/exclude filters applied
2. **max_entries Filter**: Takes most recent N entries
3. **max_tokens Filter**: Iterates backwards from recent entries until limit

**Parameters:**
- `include_types`: Only include entries with these types
- `include_sources`: Only include entries from these sources
- `exclude_types`: Exclude entries with these types
- `exclude_sources`: Exclude entries from these sources
- `max_entries`: Maximum number of most recent entries
- `max_tokens`: Maximum token count for formatted output
- `format_style`: Output format (`'chat'`, `'full_json'`, `'content_only'`)

**Example:**
```python
# Get last 10 user/assistant exchanges in chat format
formatted = history_manager.get_formatted_history(
    include_types=["user_input", "agent_output"],
    max_entries=10,
    format_style="chat"
)
print(formatted)
# Output:
# User: What is 2 + 2?
# Agent (math_agent): The answer is 4.
# User: What about 5 * 5?
# Agent (math_agent): That equals 25.

# Get all tool-related entries as JSON
tool_history = history_manager.get_formatted_history(
    include_types=["tool_call", "tool_result"],
    format_style="full_json"
)

# Get recent context with token limit
context = history_manager.get_formatted_history(
    max_tokens=2000,
    exclude_types=["error"],  # Skip errors
    format_style="content_only"
)
```

### clear_history()

Clears all entries from the history.

```python
def clear_history() -> None
```

**Example:**
```python
history_manager.clear_history()
print(f"Entries after clear: {len(history_manager)}")  # 0
```

### get_statistics()

Returns comprehensive statistics about the history. **New in v0.4.1a**

```python
def get_statistics() -> Dict[str, Any]
```

**Returns:**
```python
{
    "total_tokens": 3456,          # Current total token count
    "total_entries": 42,            # Number of entries
    "max_tokens": 4000,             # Token limit (None if unlimited)
    "max_entries": 50,              # Entry limit (None if unlimited)
    "utilization_pct": 86.4,        # Percentage of token limit used
    "entry_types": {                # Distribution by type
        "user_input": 12,
        "agent_output": 12,
        "tool_call": 8,
        "tool_result": 8,
        "error": 2
    },
    "truncation_strategy": "oldest_first"
}
```

**Example:**
```python
stats = history_manager.get_statistics()

print(f"Using {stats['total_tokens']} of {stats['max_tokens']} tokens")
print(f"Utilization: {stats['utilization_pct']:.1f}%")

if stats['utilization_pct'] > 90:
    print("WARNING: Approaching token limit!")

# Log entry distribution
for entry_type, count in stats['entry_types'].items():
    print(f"  {entry_type}: {count}")
```

### Public Properties

**New in v0.4.1a** - Read-only properties for monitoring:

```python
# Current token count
token_count = history_manager.current_token_count
print(f"Current tokens: {token_count}")

# Number of entries
num_entries = history_manager.history_size
print(f"History size: {num_entries}")

# Access history (read-only copy)
entries = history_manager.history
for entry in entries:
    print(entry['type'], entry['timestamp'])

# Can also use len()
print(f"Length: {len(history_manager)}")
```

## Usage Examples

### Example 1: Basic Usage with Single Agent

```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain import PromptChain

# Initialize with token limit
history_manager = ExecutionHistoryManager(
    max_tokens=4000,
    truncation_strategy="oldest_first"
)

# Create agent
agent = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze: {input}", "Summarize: {input}"]
)

# Process user input
user_query = "Explain quantum computing"
history_manager.add_entry("user_input", user_query, source="user")

response = agent.process_prompt(user_query)
history_manager.add_entry("agent_output", response, source="agent")

# Get formatted conversation
conversation = history_manager.get_formatted_history(
    include_types=["user_input", "agent_output"],
    format_style="chat"
)
print(conversation)

# Check statistics
stats = history_manager.get_statistics()
print(f"Tokens used: {stats['total_tokens']} / {stats['max_tokens']}")
```

### Example 2: Integration with AgentChain

```python
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

# Create history manager for router context
router_history = ExecutionHistoryManager(
    max_tokens=8000,  # Larger limit for complex conversations
    max_entries=100
)

# AgentChain with multiple specialized agents
agent_chain = AgentChain(
    agents={"math": math_agent, "code": code_agent, "research": research_agent},
    agent_descriptions={...},
    execution_mode="router",
    router=router_config
)

async def process_with_history(user_input: str):
    # Add user input to history
    router_history.add_entry("user_input", user_input, source="user")

    # Get formatted history for router context
    context = router_history.get_formatted_history(
        max_tokens=2000,  # Limit context for router
        include_types=["user_input", "agent_output"],
        format_style="chat"
    )

    # Process with AgentChain
    result = await agent_chain.process_input(user_input)

    # Add result to history
    router_history.add_entry(
        "agent_output",
        result,
        source="agent_chain",
        metadata={"mode": "router"}
    )

    return result

# Monitor history utilization
stats = router_history.get_statistics()
if stats['utilization_pct'] > 80:
    print(f"WARNING: History at {stats['utilization_pct']:.0f}% capacity")
```

### Example 3: Using Filters and Truncation

```python
# Track a complex workflow with multiple agents and tools
workflow_history = ExecutionHistoryManager(
    max_tokens=10000,
    max_entries=200
)

# Add various entries
workflow_history.add_entry("user_input", "Process invoice #12345", source="api")
workflow_history.add_entry("agent_output", "Validating invoice...", source="validator_agent")
workflow_history.add_entry("tool_call", '{"function": "db_query", "args": {...}}', source="validator_agent")
workflow_history.add_entry("tool_result", "Invoice found: {...}", source="database")
workflow_history.add_entry("agent_output", "Invoice valid, proceeding...", source="validator_agent")

# Get only the user-facing conversation (exclude internal tools)
user_view = workflow_history.get_formatted_history(
    include_types=["user_input", "agent_output"],
    exclude_sources=["system"],
    format_style="chat"
)

# Get detailed tool execution log
tool_log = workflow_history.get_formatted_history(
    include_types=["tool_call", "tool_result"],
    format_style="full_json"
)

# Get last 5 entries only
recent = workflow_history.get_formatted_history(
    max_entries=5,
    format_style="chat"
)
```

### Example 4: Token Limit Management

```python
# Create history manager with strict token limit
limited_history = ExecutionHistoryManager(
    max_tokens=2000,
    truncation_strategy="oldest_first"
)

# Simulate conversation
for i in range(20):
    limited_history.add_entry("user_input", f"Query {i}", source="user")
    limited_history.add_entry("agent_output", "Response..." * 100, source="agent")

    # Check if truncation occurred
    stats = limited_history.get_statistics()
    print(f"Turn {i}: {stats['total_entries']} entries, "
          f"{stats['total_tokens']} tokens ({stats['utilization_pct']:.0f}%)")

# Verify we stayed under limit
final_stats = limited_history.get_statistics()
assert final_stats['total_tokens'] <= 2000, "Token limit exceeded!"
print(f"Final: {final_stats['total_entries']} entries kept in {final_stats['total_tokens']} tokens")
```

### Example 5: Observability Integration

```python
from promptchain.utils.callback_manager import CallbackManager
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

# Create callback manager for observability
callback_mgr = CallbackManager()

# Register handler for truncation events
def handle_truncation(event: ExecutionEvent):
    if event.event_type == ExecutionEventType.HISTORY_TRUNCATED:
        print(f"⚠️  History truncated!")
        print(f"   Reason: {event.metadata['reason']}")
        print(f"   Entries removed: {event.metadata['entries_removed']}")
        print(f"   Tokens removed: {event.metadata['tokens_removed']}")
        print(f"   Final count: {event.metadata.get('final_token_count', 'N/A')}")

callback_mgr.register_handler(ExecutionEventType.HISTORY_TRUNCATED, handle_truncation)

# Create history manager with callback
history = ExecutionHistoryManager(
    max_tokens=1000,
    callback_manager=callback_mgr
)

# Add entries - will trigger truncation
for i in range(50):
    history.add_entry("user_input", "Test message " * 20, source="test")
    # Will see truncation warnings when limit exceeded
```

## Entry Types Reference Table

| Type | Purpose | When to Use | Common Metadata |
|------|---------|-------------|-----------------|
| `user_input` | User messages | Every user query/command | `{"user_id": str, "session_id": str}` |
| `agent_output` | Agent responses | After agent processing | `{"agent_name": str, "confidence": float, "execution_time_ms": int}` |
| `tool_call` | Tool invocations | Before calling a tool | `{"tool_name": str, "function": str, "args": dict}` |
| `tool_result` | Tool outputs | After tool execution | `{"tool_name": str, "success": bool, "execution_time_ms": int}` |
| `system_message` | System notifications | Internal events | `{"event_type": str, "severity": str}` |
| `error` | Error messages | Exception handling | `{"error_code": str, "stack_trace": str, "retry_count": int}` |
| `custom` | Application-specific | Custom tracking needs | Application-defined |

## Truncation Strategies

### Current Strategies

#### oldest_first (Default)
- Removes entries from the beginning of history
- Preserves most recent conversation context
- Applied when `max_tokens` or `max_entries` exceeded

```python
# Example: oldest_first with token limit
history = ExecutionHistoryManager(
    max_tokens=1000,
    truncation_strategy="oldest_first"
)

# Add 100 entries
for i in range(100):
    history.add_entry("user_input", f"Message {i}", source="user")

# Oldest entries removed to stay under 1000 tokens
stats = history.get_statistics()
print(f"Kept {stats['total_entries']} entries in {stats['total_tokens']} tokens")
```

### Future Strategies (Planned)

#### keep_last
- Keep most recent N entries, remove everything older
- Useful for fixed-window context

#### priority_based
- Assign priority scores to entries
- Remove lowest priority entries first
- Example: Keep errors and user inputs, remove intermediate tool calls

#### semantic_compression
- Use embeddings to identify similar/redundant entries
- Compress or remove duplicates
- Preserve semantic diversity

## Best Practices

### 1. Choose Appropriate Limits

```python
# For chat applications
chat_history = ExecutionHistoryManager(
    max_tokens=4000,  # ~4000 tokens = 16K characters
    max_entries=50    # Last ~25 turns of conversation
)

# For long-running workflows
workflow_history = ExecutionHistoryManager(
    max_tokens=10000,  # Larger limit for complex processes
    max_entries=None   # No entry limit
)

# For debugging/development
debug_history = ExecutionHistoryManager(
    max_tokens=None,  # Unlimited for full capture
    max_entries=None
)
```

### 2. Use Type Filters for Different Contexts

```python
# Get only user-facing messages for UI
ui_history = history.get_formatted_history(
    include_types=["user_input", "agent_output"],
    max_tokens=2000,
    format_style="chat"
)

# Get only errors for debugging
error_log = history.get_formatted_history(
    include_types=["error"],
    format_style="full_json"
)

# Get tool execution trace
tool_trace = history.get_formatted_history(
    include_types=["tool_call", "tool_result"],
    format_style="content_only"
)
```

### 3. Monitor Statistics Regularly

```python
def check_history_health(history_mgr: ExecutionHistoryManager):
    stats = history_mgr.get_statistics()

    # Warn if approaching limits
    if stats['max_tokens'] and stats['utilization_pct'] > 90:
        print(f"⚠️  WARNING: {stats['utilization_pct']:.0f}% token capacity used!")

    # Check entry distribution
    if 'error' in stats['entry_types'] and stats['entry_types']['error'] > 10:
        print(f"⚠️  High error count: {stats['entry_types']['error']}")

    return stats

# Check after major operations
stats = check_history_health(history_manager)
```

### 4. Use Source Identifiers Consistently

```python
# Consistent source naming
history.add_entry("user_input", query, source="user")
history.add_entry("agent_output", response, source="math_agent")
history.add_entry("tool_call", tool_args, source="math_agent")
history.add_entry("tool_result", tool_output, source="calculator_tool")

# Enables filtering by source
math_agent_history = history.get_formatted_history(
    include_sources=["math_agent", "calculator_tool"]
)
```

### 5. Leverage Metadata for Rich Context

```python
# Add structured metadata for better tracking
history.add_entry(
    "agent_output",
    response,
    source="research_agent",
    metadata={
        "query_complexity": "high",
        "sources_consulted": 5,
        "confidence_score": 0.87,
        "execution_time_ms": 3421,
        "tokens_used": 1234,
        "model": "gpt-4"
    }
)

# Later analysis
full_history = history.get_history()
high_confidence = [
    e for e in full_history
    if e['metadata'].get('confidence_score', 0) > 0.8
]
```

### 6. Integrate with CallbackManager for Production

```python
from promptchain.utils.callback_manager import CallbackManager

# Production setup
callback_mgr = CallbackManager()

# Log all truncation events
callback_mgr.register_handler(
    ExecutionEventType.HISTORY_TRUNCATED,
    lambda e: logger.warning(f"History truncated: {e.metadata}")
)

history = ExecutionHistoryManager(
    max_tokens=4000,
    callback_manager=callback_mgr
)
```

### 7. Consider Per-Agent History Configs

When using with AgentChain, configure history separately per agent:

```python
# See AgentChain documentation for per-agent history configuration
# Different agents may need different history contexts
```

## Cross-References

- **[AgentChain Documentation](utils/agent_chain.md)**: See per-agent history configuration in v0.4.2
- **[CallbackManager Documentation](observability/callback_manager.md)**: Event system integration
- **[PromptChain Documentation](core/promptchain.md)**: Using history with prompt chains

## Version History

- **v0.4.2**: Per-agent history configuration support (see AgentChain docs)
- **v0.4.1a**: Added public API properties and get_statistics() method
- **v0.4.0**: Initial release with token-aware tracking and truncation

## Related Documentation

- [AgentChain Per-Agent History](utils/agent_chain.md#per-agent-history-configuration-v042)
- [Observability System](observability/overview.md)
- [Token Management Best Practices](guides/token-management.md)
