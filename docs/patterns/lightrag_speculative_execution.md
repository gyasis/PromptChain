# LightRAG Speculative Execution Pattern

## Overview

The Speculative Execution Pattern predicts and pre-executes likely LightRAG queries based on conversation patterns, caching results to reduce latency. This pattern analyzes query history to identify frequent or sequential patterns, then proactively executes predicted queries before they're actually requested.

## Key Concepts

### Prediction Models

**Frequency-based**: Analyzes query frequency in conversation history
- Tracks how often queries appear
- Considers recency of queries
- Confidence = (frequency * 0.7) + (recency * 0.3)

**Pattern-based**: Detects sequential patterns in query modes
- Identifies mode continuation patterns (e.g., consecutive hybrid queries)
- Predicts follow-up questions based on conversation flow
- Useful for predictable conversation structures

### Cache Management

- **TTL (Time-to-Live)**: Cached results expire after configured duration
- **Automatic Cleanup**: Expired entries are automatically removed
- **Cache Keys**: Computed from query text + mode (local/global/hybrid)
- **Hit Tracking**: Records cache hits/misses for performance analysis

### Confidence Scoring

Predictions are scored based on:
- **Frequency Score**: How often the query pattern appears
- **Recency Score**: How recently the pattern was observed
- **Pattern Strength**: How well the pattern matches known sequences

Only predictions exceeding `min_confidence` threshold are executed.

## Architecture

```
User Query → History Analysis → Predictions → Speculative Execution
                                    ↓                    ↓
                              Confidence Filter    Async Execution
                                    ↓                    ↓
                            High-Confidence Queries   Cache Results
                                                         ↓
                              Actual Query → Cache Lookup → Result
```

## Components

### ToolPrediction

Represents a predicted query with metadata:

```python
@dataclass
class ToolPrediction:
    prediction_id: str           # Unique identifier
    query: str                   # Predicted query text
    mode: str                    # Query mode (local/global/hybrid)
    confidence: float            # Prediction confidence (0.0-1.0)
    pattern_matched: str         # Pattern that triggered prediction
    context_hash: str            # Hash of context used for prediction
```

### SpeculativeResult

Represents a cached speculative execution result:

```python
@dataclass
class SpeculativeResult:
    prediction_id: str           # ID of prediction that generated result
    query: str                   # Executed query
    result: Any                  # Query result (LightRAG response)
    cached_at: datetime          # When result was cached
    ttl_seconds: float           # Time-to-live for cache
    hit: bool                    # Whether result was used (cache hit)
```

### SpeculativeConfig

Configuration for speculative execution:

```python
@dataclass
class SpeculativeConfig(PatternConfig):
    min_confidence: float = 0.7       # Minimum confidence for execution
    max_concurrent: int = 3           # Max concurrent speculative executions
    default_ttl: float = 60.0         # Default cache TTL in seconds
    prediction_model: str = "frequency"  # "frequency" or "pattern"
    history_window: int = 20          # Number of queries to track
```

### LightRAGSpeculativeExecutor

Main pattern implementation:

```python
class LightRAGSpeculativeExecutor(BasePattern):
    def __init__(
        self,
        lightrag_core: LightRAGIntegration,
        config: Optional[SpeculativeConfig] = None
    )

    # Core methods
    async def execute(self, context: str) -> SpeculativeExecutionResult
    def record_call(self, query: str, mode: str) -> None
    def predict_next_queries(self, context: str) -> List[ToolPrediction]
    def check_cache(self, query: str, mode: str) -> Optional[SpeculativeResult]
    def cleanup() -> None
```

## Usage Examples

### Basic Speculative Execution

```python
from promptchain.integrations.lightrag import (
    LightRAGIntegration,
    LightRAGSpeculativeExecutor
)
from promptchain.integrations.lightrag.speculative import SpeculativeConfig

# Initialize LightRAG
lightrag = LightRAGIntegration()

# Initialize speculative executor
config = SpeculativeConfig(
    min_confidence=0.7,
    max_concurrent=3,
    default_ttl=120.0  # 2 minutes
)
executor = LightRAGSpeculativeExecutor(lightrag, config)

# Execute queries and build history
queries = [
    "What is machine learning?",
    "What is deep learning?",
    "What is machine learning?"  # Repeated query
]

for query in queries:
    # Check cache first
    cached = executor.check_cache(query, "hybrid")

    if cached:
        print(f"Cache hit! Using cached result")
        result = cached.result
    else:
        # Execute actual query
        result = await lightrag.hybrid_query(query)

    # Record in history
    executor.record_call(query, "hybrid")

    # Execute speculative predictions
    spec_result = await executor.execute(context=query)
    print(f"Generated {len(spec_result.predictions)} predictions")
```

### Frequency-Based Prediction

```python
# Configure for frequency-based prediction
config = SpeculativeConfig(
    prediction_model="frequency",
    min_confidence=0.6,
    history_window=20
)
executor = LightRAGSpeculativeExecutor(lightrag, config)

# Build query history
for i in range(5):
    executor.record_call("What is AI?", "hybrid")

executor.record_call("What is ML?", "local")

# Generate predictions
predictions = executor.predict_next_queries("Current context")

for pred in predictions:
    print(f"Query: {pred.query}")
    print(f"Confidence: {pred.confidence:.2f}")
    print(f"Pattern: {pred.pattern_matched}")
```

### Pattern-Based Prediction

```python
# Configure for pattern-based prediction
config = SpeculativeConfig(
    prediction_model="pattern",
    min_confidence=0.7
)
executor = LightRAGSpeculativeExecutor(lightrag, config)

# Create pattern with consecutive same-mode queries
executor.record_call("Overview of AI", "global")
executor.record_call("Overview of ML", "global")

# Pattern will predict another global query
predictions = executor.predict_next_queries("Overview context")
# Predictions will include mode="global" continuation
```

### Cache Management

```python
# Configure short TTL for testing
config = SpeculativeConfig(default_ttl=30.0)  # 30 seconds
executor = LightRAGSpeculativeExecutor(lightrag, config)

# Execute and cache
executor.record_call("Test query", "hybrid")
await executor.execute(context="Test")

print(f"Cache size: {len(executor.cache)}")

# Wait for expiration
await asyncio.sleep(35)

# Cleanup expired entries
executor.cleanup()
print(f"Cache size after cleanup: {len(executor.cache)}")
```

### Event Monitoring

```python
# Configure with event emission
config = SpeculativeConfig(emit_events=True)
executor = LightRAGSpeculativeExecutor(lightrag, config)

# Add event handler
def event_handler(event_type: str, data: dict):
    print(f"Event: {event_type}")
    if "num_predictions" in data:
        print(f"  Predictions: {data['num_predictions']}")

executor.add_event_handler(event_handler)

# Execute with event tracking
await executor.execute(context="Test")
```

## Performance Metrics

### Statistics Tracking

```python
stats = executor.get_stats()

# Core metrics
stats["total_predictions"]      # Total predictions generated
stats["total_executions"]       # Total speculative executions
stats["cache_hits"]             # Number of cache hits
stats["cache_misses"]           # Number of cache misses
stats["cache_hit_rate"]         # Hit rate (0.0-1.0)
stats["cache_size"]             # Current cache entries
stats["history_size"]           # Current history entries
```

### Latency Savings

When a cache hit occurs, latency savings can be significant:

```python
# Without speculative execution
time_to_execute = 2000ms  # LightRAG query execution time

# With speculative execution (cache hit)
time_to_execute = 10ms    # Cache lookup time
latency_saved = 1990ms    # 99.5% reduction
```

### Memory Usage

Memory consumption scales with:
- **History size**: `history_window * entry_size (~200 bytes)`
- **Cache size**: `num_predictions * result_size (varies)`

Example: 20 history entries + 10 cached results ≈ 4KB + result data

## Events Emitted

The pattern emits events for monitoring and debugging:

| Event Type | Data | Description |
|------------|------|-------------|
| `pattern.speculative.started` | `context_length` | Execution started |
| `pattern.speculative.predicted` | `num_predictions`, `avg_confidence` | Predictions generated |
| `pattern.speculative.executed` | `num_executed`, `num_failed` | Speculative executions completed |
| `pattern.speculative.cache_hit` | `query`, `mode` | Cache hit occurred (via check_cache) |
| `pattern.speculative.cache_miss` | `query`, `mode` | Cache miss occurred (via check_cache) |
| `pattern.speculative.completed` | `execution_time_ms`, `num_cached` | Execution completed |

## Best Practices

### Configuration Tuning

**High-Confidence Mode** (conservative):
```python
config = SpeculativeConfig(
    min_confidence=0.9,      # Very high threshold
    max_concurrent=2,        # Limit resource usage
    default_ttl=60.0         # Short cache lifetime
)
```

**Aggressive Mode** (maximize cache hits):
```python
config = SpeculativeConfig(
    min_confidence=0.5,      # Lower threshold
    max_concurrent=5,        # More concurrent executions
    default_ttl=300.0        # Longer cache lifetime
)
```

**Balanced Mode** (recommended):
```python
config = SpeculativeConfig(
    min_confidence=0.7,      # Moderate threshold
    max_concurrent=3,        # Moderate concurrency
    default_ttl=120.0        # 2-minute cache
)
```

### Query Pattern Analysis

For best results, analyze your query patterns:

```python
# Analyze frequency patterns
patterns = executor._analyze_frequency_patterns()

for key, stats in patterns.items():
    print(f"Query: {stats['query'][:50]}")
    print(f"  Count: {stats['count']}")
    print(f"  Last seen: {stats['last_seen']}")
    print(f"  Mode: {stats['mode']}")
```

### Resource Management

Monitor resource usage:

```python
# Check cache size periodically
if len(executor.cache) > 100:
    executor.cleanup()  # Force cleanup

# Monitor memory usage
import sys
cache_size_bytes = sys.getsizeof(executor.cache)
history_size_bytes = sys.getsizeof(executor.history)
```

### Integration with MessageBus

```python
from promptchain.cli.models import MessageBus

# Create message bus
bus = MessageBus()

# Connect executor
executor.connect_messagebus(bus)

# Subscribe to speculative events
def on_prediction(event_type: str, data: dict):
    print(f"New prediction: {data['num_predictions']}")

bus.subscribe("pattern.speculative.predicted", on_prediction)
```

## Common Patterns

### Warm-Up Cache

Pre-populate cache with common queries:

```python
common_queries = [
    ("What is AI?", "hybrid"),
    ("What is ML?", "hybrid"),
    ("What is DL?", "local")
]

for query, mode in common_queries:
    executor.record_call(query, mode)

# Execute speculative predictions
await executor.execute(context="Initialization")
```

### Adaptive Confidence

Adjust confidence based on cache performance:

```python
stats = executor.get_stats()

if stats["cache_hit_rate"] < 0.3:
    # Low hit rate - be more aggressive
    executor.config.min_confidence = 0.5
elif stats["cache_hit_rate"] > 0.8:
    # High hit rate - be more conservative
    executor.config.min_confidence = 0.9
```

## Error Handling

```python
try:
    result = await executor.execute(context="Test")

    if not result.success:
        print(f"Execution failed: {result.errors}")

except asyncio.TimeoutError:
    print("Execution timed out")

except Exception as e:
    print(f"Unexpected error: {e}")
```

## Limitations

- **Prediction Accuracy**: Depends on query pattern consistency
- **Memory Usage**: Large history windows increase memory consumption
- **Cache Staleness**: Cached results may become outdated
- **Concurrent Execution**: Limited by `max_concurrent` setting
- **Context Sensitivity**: Predictions don't consider full conversation context

## Future Enhancements

- **ML-based Prediction**: Use ML models for better prediction accuracy
- **Context-Aware Caching**: Consider full conversation context in predictions
- **Dynamic TTL**: Adjust TTL based on query characteristics
- **Negative Caching**: Cache "query not found" results to avoid repeated failures
- **Cache Warming**: Pre-populate cache based on common patterns
- **Multi-Level Caching**: Separate caches for different query modes

## See Also

- [BasePattern Documentation](./base_pattern.md)
- [LightRAG Integration](../integrations/lightrag.md)
- [Multi-Hop Pattern](./lightrag_multi_hop.md)
- [Query Expansion Pattern](./lightrag_query_expansion.md)
