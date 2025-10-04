# Best Practices for PromptChain Observability

This guide provides patterns and recommendations for using PromptChain's observability features effectively.

## General Principles

### 1. Use Public APIs Only

**Always** use public APIs instead of private attributes.

```python
# ✅ Good: Public API (stable, won't break)
token_count = history.current_token_count
entries = history.history
stats = history.get_statistics()

# ❌ Bad: Private attributes (will be removed in v0.5.0)
token_count = history._current_tokens
entries = history._history
```

**Why**: Public APIs are stable, tested, and won't change without deprecation warnings.

### 2. Enable Metadata Selectively

Use `return_metadata=True` only when you need it.

```python
# Development/debugging: Enable metadata
result = agent_chain.process_input(prompt, return_metadata=True)
log_detailed_metrics(result)

# Production: Disable for performance
result = agent_chain.process_input(prompt)  # return_metadata=False (default)
```

**Impact**: Metadata adds ~1-2% overhead. Use it where value justifies the cost.

### 3. Filter Events Aggressively

Don't subscribe to all events - filter for what you need.

```python
# ✅ Good: Filter specific events
chain.register_callback(
    error_handler,
    event_filter=[
        ExecutionEventType.CHAIN_ERROR,
        ExecutionEventType.MODEL_CALL_ERROR,
        ExecutionEventType.TOOL_CALL_ERROR
    ]
)

# ❌ Bad: Receive all events (unnecessary overhead)
chain.register_callback(error_handler)  # Gets ALL events
```

## ExecutionHistoryManager Patterns

### Pattern 1: Monitor Memory Usage

```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

history = ExecutionHistoryManager(max_tokens=8000, max_entries=200)

def check_memory_usage():
    stats = history.get_statistics()
    usage_percent = (stats['current_token_count'] / stats['max_tokens']) * 100

    if usage_percent > 80:
        print(f"Warning: History at {usage_percent:.1f}% capacity")
        print(f"Consider truncating or increasing max_tokens")

    return stats

# Periodic check
stats = check_memory_usage()
```

### Pattern 2: Analyze History Distribution

```python
def analyze_history_distribution(history: ExecutionHistoryManager):
    stats = history.get_statistics()
    entries_by_type = stats['entries_by_type']

    print("History Distribution:")
    for entry_type, count in entries_by_type.items():
        percentage = (count / stats['total_entries']) * 100
        print(f"  {entry_type}: {count} ({percentage:.1f}%)")

    # Check for imbalance
    if entries_by_type.get('tool_result', 0) > entries_by_type.get('tool_call', 0):
        print("Warning: More tool results than tool calls (data inconsistency?)")

analyze_history_distribution(history)
```

### Pattern 3: Smart Truncation Strategy

```python
# For chat applications: Keep recent messages
chat_history = ExecutionHistoryManager(
    max_tokens=4000,
    truncation_strategy="oldest_first"
)

# For analysis: Keep complete context
analysis_history = ExecutionHistoryManager(
    max_tokens=16000,
    truncation_strategy="fifo"  # First In, First Out
)

# Monitor truncation
def on_history_truncated(event: ExecutionEvent):
    metadata = event.metadata
    print(f"History truncated:")
    print(f"  Removed: {metadata['entries_removed']} entries")
    print(f"  Freed: {metadata['tokens_freed']} tokens")
    print(f"  Remaining: {metadata['remaining_entries']} entries")

chain.register_callback(
    on_history_truncated,
    event_filter=ExecutionEventType.HISTORY_TRUNCATED
)
```

## Callback Patterns

### Pattern 1: Performance Monitoring

```python
from collections import defaultdict
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)

    def __call__(self, event: ExecutionEvent):
        if event.event_type == ExecutionEventType.MODEL_CALL_END:
            self.metrics['model_latency'].append(
                event.metadata.get('execution_time_ms', 0)
            )
            self.metrics['tokens_used'].append(
                event.metadata.get('tokens_used', 0)
            )

        elif event.event_type == ExecutionEventType.TOOL_CALL_END:
            self.metrics['tool_latency'].append(
                event.metadata.get('execution_time_ms', 0)
            )

    def get_summary(self):
        return {
            'avg_model_latency': sum(self.metrics['model_latency']) / len(self.metrics['model_latency']) if self.metrics['model_latency'] else 0,
            'avg_tokens': sum(self.metrics['tokens_used']) / len(self.metrics['tokens_used']) if self.metrics['tokens_used'] else 0,
            'avg_tool_latency': sum(self.metrics['tool_latency']) / len(self.metrics['tool_latency']) if self.metrics['tool_latency'] else 0,
            'total_model_calls': len(self.metrics['model_latency']),
            'total_tool_calls': len(self.metrics['tool_latency'])
        }

# Usage
monitor = PerformanceMonitor()
chain.register_callback(
    monitor,
    event_filter=[
        ExecutionEventType.MODEL_CALL_END,
        ExecutionEventType.TOOL_CALL_END
    ]
)

result = chain.process_prompt("Your input")
summary = monitor.get_summary()
print(f"Performance: {summary}")
```

### Pattern 2: Error Tracking and Alerting

```python
import logging
from datetime import datetime, timedelta

class ErrorTracker:
    def __init__(self, alert_threshold=5, time_window_minutes=5):
        self.errors = []
        self.alert_threshold = alert_threshold
        self.time_window = timedelta(minutes=time_window_minutes)
        self.logger = logging.getLogger(__name__)

    def __call__(self, event: ExecutionEvent):
        if event.event_type.name.endswith('_ERROR'):
            error_info = {
                'type': event.event_type.name,
                'error': event.metadata.get('error'),
                'timestamp': event.timestamp,
                'context': {
                    'step': event.step_number,
                    'model': event.model_name,
                    'instruction': event.step_instruction
                }
            }
            self.errors.append(error_info)

            # Check for error spike
            self._check_error_spike()

            # Log error
            self.logger.error(f"Execution error: {error_info}")

    def _check_error_spike(self):
        now = datetime.now()
        recent_errors = [
            e for e in self.errors
            if now - e['timestamp'] < self.time_window
        ]

        if len(recent_errors) >= self.alert_threshold:
            self._send_alert(recent_errors)

    def _send_alert(self, recent_errors):
        # Send to monitoring system (e.g., Sentry, Datadog)
        alert_msg = f"Error spike detected: {len(recent_errors)} errors in {self.time_window.seconds/60} minutes"
        self.logger.critical(alert_msg)
        # send_to_slack(alert_msg)
        # send_to_pagerduty(alert_msg)

# Usage
tracker = ErrorTracker(alert_threshold=5, time_window_minutes=5)
chain.register_callback(
    tracker,
    event_filter=[
        ExecutionEventType.CHAIN_ERROR,
        ExecutionEventType.STEP_ERROR,
        ExecutionEventType.MODEL_CALL_ERROR,
        ExecutionEventType.TOOL_CALL_ERROR
    ]
)
```

### Pattern 3: Structured Logging

```python
import json
import logging

class StructuredLogger:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def __call__(self, event: ExecutionEvent):
        log_entry = {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type.name,
            'step_number': event.step_number,
            'model_name': event.model_name,
            'metadata': event.metadata
        }

        # Determine log level based on event type
        if event.event_type.name.endswith('_ERROR'):
            self.logger.error(json.dumps(log_entry))
        elif event.event_type in [ExecutionEventType.CHAIN_START, ExecutionEventType.CHAIN_END]:
            self.logger.info(json.dumps(log_entry))
        else:
            self.logger.debug(json.dumps(log_entry))

# Usage
structured_logger = StructuredLogger(log_level=logging.DEBUG)
chain.register_callback(structured_logger)  # Logs all events in JSON format
```

## Metadata Usage Patterns

### Pattern 1: Execution Analysis

```python
def analyze_execution(result: AgentExecutionResult):
    """Analyze execution result for insights."""

    # Performance analysis
    if result.execution_time_ms > 5000:
        print(f"⚠️  Slow execution: {result.execution_time_ms}ms")

        # Identify bottleneck
        if result.router_steps > 3:
            print(f"  - Router took {result.router_steps} steps (consider simplifying logic)")

        if len(result.tools_called) > 10:
            print(f"  - Many tool calls: {len(result.tools_called)} (consider batching)")

    # Token analysis
    if result.total_tokens and result.total_tokens > 8000:
        print(f"⚠️  High token usage: {result.total_tokens} tokens")
        print(f"  - Prompt: {result.prompt_tokens}, Completion: {result.completion_tokens}")

    # Error analysis
    if result.errors:
        print(f"❌ Errors encountered: {len(result.errors)}")
        for error in result.errors:
            print(f"  - {error}")

    # Cache analysis
    if result.cache_hit:
        print(f"✓ Cache hit (saved {result.execution_time_ms}ms)")
    else:
        print(f"  Cache miss - key: {result.cache_key}")

# Usage
result = agent_chain.process_input("Your input", return_metadata=True)
analyze_execution(result)
```

### Pattern 2: AgenticStepProcessor Debugging

```python
def debug_agentic_steps(result: AgenticStepResult):
    """Debug agentic step execution."""

    print(f"Agentic Step Analysis:")
    print(f"  Objective: {'✓ Achieved' if result.objective_achieved else '✗ Failed'}")
    print(f"  Steps: {result.total_steps}/{result.max_internal_steps}")
    print(f"  Tools called: {result.total_tools_called}")
    print(f"  Execution time: {result.total_execution_time_ms}ms")
    print(f"  Tokens used: {result.total_tokens_used}")

    # Step-by-step analysis
    print(f"\nStep-by-Step Breakdown:")
    for step in result.steps:
        print(f"  Step {step.step_number}:")
        print(f"    Tools: {len(step.tool_calls)}")
        print(f"    Time: {step.execution_time_ms}ms")
        print(f"    Tokens: {step.tokens_used}")

        if step.error:
            print(f"    ❌ Error: {step.error}")

        if step.clarification_attempts > 0:
            print(f"    ⚠️  Clarification attempts: {step.clarification_attempts}")

    # Efficiency analysis
    if result.total_steps == result.max_internal_steps:
        print(f"\n⚠️  Warning: Reached max steps limit")
        print(f"  Consider increasing max_internal_steps or simplifying objective")

    avg_time_per_step = result.total_execution_time_ms / result.total_steps
    if avg_time_per_step > 2000:
        print(f"\n⚠️  Warning: Slow steps (avg {avg_time_per_step:.0f}ms/step)")

# Usage
result = await agentic_step.run_async("Your input", return_metadata=True)
debug_agentic_steps(result)
```

### Pattern 3: Tool Usage Analytics

```python
from collections import Counter

def analyze_tool_usage(result: AgentExecutionResult):
    """Analyze tool usage patterns."""

    if not result.tools_called:
        print("No tools called")
        return

    # Tool frequency
    tool_names = [tc['name'] for tc in result.tools_called]
    tool_freq = Counter(tool_names)

    print(f"Tool Usage Analysis:")
    print(f"  Total calls: {len(result.tools_called)}")
    print(f"  Unique tools: {len(tool_freq)}")

    print(f"\n  Frequency:")
    for tool, count in tool_freq.most_common():
        print(f"    {tool}: {count} calls")

    # Performance analysis
    tool_times = {}
    for tc in result.tools_called:
        tool_name = tc['name']
        time_ms = tc.get('time_ms', 0)
        if tool_name not in tool_times:
            tool_times[tool_name] = []
        tool_times[tool_name].append(time_ms)

    print(f"\n  Performance:")
    for tool, times in tool_times.items():
        avg_time = sum(times) / len(times)
        print(f"    {tool}: {avg_time:.1f}ms avg ({min(times)}-{max(times)}ms range)")

    # Success rate
    successes = sum(1 for tc in result.tools_called if tc.get('success', True))
    success_rate = (successes / len(result.tools_called)) * 100
    print(f"\n  Success rate: {success_rate:.1f}%")

# Usage
result = agent_chain.process_input("Your input", return_metadata=True)
analyze_tool_usage(result)
```

## MCP Monitoring Patterns

### Pattern 1: Connection Health Monitoring

```python
class MCPHealthMonitor:
    def __init__(self):
        self.servers = {}

    def __call__(self, event: ExecutionEvent):
        if event.event_type == ExecutionEventType.MCP_CONNECT_END:
            server_id = event.metadata['server_id']
            status = event.metadata['status']

            self.servers[server_id] = {
                'status': status,
                'last_check': event.timestamp,
                'connection_time_ms': event.metadata.get('execution_time_ms')
            }

            if status == 'failed':
                self._alert_connection_failure(server_id, event.metadata.get('error'))

    def _alert_connection_failure(self, server_id, error):
        print(f"🚨 MCP server connection failed: {server_id}")
        print(f"   Error: {error}")
        # Send alert to monitoring system

    def get_health_status(self):
        healthy = sum(1 for s in self.servers.values() if s['status'] == 'connected')
        total = len(self.servers)
        return {
            'healthy_servers': healthy,
            'total_servers': total,
            'health_percentage': (healthy / total * 100) if total > 0 else 0,
            'servers': self.servers
        }

# Usage
health_monitor = MCPHealthMonitor()
chain.register_callback(
    health_monitor,
    event_filter=ExecutionEventType.MCP_CONNECT_END
)
```

### Pattern 2: Tool Discovery Tracking

```python
class ToolDiscoveryTracker:
    def __init__(self):
        self.tools_by_server = {}
        self.all_tools = {}

    def __call__(self, event: ExecutionEvent):
        if event.event_type == ExecutionEventType.MCP_TOOL_DISCOVERED:
            server_id = event.metadata['server_id']
            tool_name = event.metadata['tool_name']
            prefixed_name = event.metadata['mcp_prefixed_name']

            # Track by server
            if server_id not in self.tools_by_server:
                self.tools_by_server[server_id] = []
            self.tools_by_server[server_id].append(tool_name)

            # Track all tools
            self.all_tools[prefixed_name] = {
                'original_name': tool_name,
                'server_id': server_id,
                'schema': event.metadata.get('schema')
            }

    def get_server_tools(self, server_id: str):
        return self.tools_by_server.get(server_id, [])

    def get_all_tools(self):
        return list(self.all_tools.keys())

    def find_tool(self, search_term: str):
        """Find tools matching search term."""
        return [
            name for name in self.all_tools.keys()
            if search_term.lower() in name.lower()
        ]

# Usage
tracker = ToolDiscoveryTracker()
chain.register_callback(
    tracker,
    event_filter=ExecutionEventType.MCP_TOOL_DISCOVERED
)

# After chain initialization
print(f"Available tools: {tracker.get_all_tools()}")
print(f"Filesystem tools: {tracker.get_server_tools('filesystem')}")
print(f"Search 'read': {tracker.find_tool('read')}")
```

## Production Recommendations

### 1. Layered Monitoring

```python
# Layer 1: Always-on error tracking
error_tracker = ErrorTracker(alert_threshold=10, time_window_minutes=5)
chain.register_callback(
    error_tracker,
    event_filter=[
        ExecutionEventType.CHAIN_ERROR,
        ExecutionEventType.MODEL_CALL_ERROR
    ]
)

# Layer 2: Performance monitoring (sampled)
import random

def sampled_performance_monitor(event: ExecutionEvent):
    if random.random() < 0.1:  # 10% sampling
        performance_monitor(event)

chain.register_callback(
    sampled_performance_monitor,
    event_filter=[
        ExecutionEventType.CHAIN_END,
        ExecutionEventType.MODEL_CALL_END
    ]
)

# Layer 3: Detailed debugging (on-demand)
if DEBUG_MODE:
    chain.register_callback(detailed_logger)  # All events
```

### 2. Resource Management

```python
# Unregister callbacks when not needed
def temporary_monitoring(chain, duration_seconds=60):
    monitor = PerformanceMonitor()
    chain.register_callback(monitor)

    try:
        # Run with monitoring
        time.sleep(duration_seconds)
    finally:
        # Always cleanup
        chain.unregister_callback(monitor)

# Context manager pattern
from contextlib import contextmanager

@contextmanager
def monitored_execution(chain, callback):
    chain.register_callback(callback)
    try:
        yield
    finally:
        chain.unregister_callback(callback)

# Usage
with monitored_execution(chain, performance_monitor):
    result = chain.process_prompt("Your input")
```

### 3. Async Best Practices

```python
import asyncio

# ✅ Good: Use async callbacks for I/O
async def async_logger(event: ExecutionEvent):
    await write_to_database(event)
    await send_to_monitoring_service(event)

chain.register_callback(async_logger)

# ✅ Good: Lightweight sync callbacks
def quick_counter(event: ExecutionEvent):
    counter[event.event_type] += 1

chain.register_callback(quick_counter)

# ❌ Bad: Blocking I/O in sync callback
def blocking_logger(event: ExecutionEvent):
    requests.post(monitoring_url, json=event.to_dict())  # Blocks execution!

# Better: Offload to background
async def non_blocking_logger(event: ExecutionEvent):
    asyncio.create_task(
        send_to_monitoring(event)  # Fire and forget
    )
```

## Testing Patterns

### Pattern 1: Test with Mock Callbacks

```python
from unittest.mock import Mock

def test_chain_execution():
    mock_callback = Mock()

    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Test: {input}"]
    )
    chain.register_callback(mock_callback)

    result = chain.process_prompt("Test input")

    # Assert callbacks were called
    assert mock_callback.call_count > 0

    # Check specific events
    events = [call[0][0] for call in mock_callback.call_args_list]
    event_types = [e.event_type for e in events]

    assert ExecutionEventType.CHAIN_START in event_types
    assert ExecutionEventType.CHAIN_END in event_types
```

### Pattern 2: Validate Metadata

```python
def test_execution_metadata():
    result = agent_chain.process_input("Test", return_metadata=True)

    # Validate metadata structure
    assert isinstance(result, AgentExecutionResult)
    assert result.agent_name is not None
    assert result.execution_time_ms > 0
    assert isinstance(result.tools_called, list)

    # Validate timing
    assert result.end_time > result.start_time

    # Validate consistency
    if result.errors:
        assert len(result.errors) > 0
```

## Summary

**Key Takeaways**:

1. ✅ Use public APIs only
2. ✅ Enable metadata selectively
3. ✅ Filter events aggressively
4. ✅ Use async callbacks for I/O
5. ✅ Monitor errors and performance
6. ✅ Clean up callbacks when done
7. ✅ Test your observability code

**Performance Tips**:

- Public APIs: Zero overhead
- Metadata: ~1-2% overhead when enabled
- Callbacks: <1% overhead with filtering
- Sampling: Use for low-priority metrics

**Next Steps**:

- Review [examples/observability/](../../examples/observability/) for working code
- Check [Public APIs Guide](public-apis.md) for API details
- Read [Event System Guide](event-system.md) for callback patterns
