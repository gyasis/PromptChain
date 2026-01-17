# Speculative Execution Pattern

Predict and pre-execute likely LightRAG queries based on conversation patterns, caching results to reduce latency.

## Overview

The Speculative Execution pattern reduces query latency by:
1. Analyzing conversation history for patterns
2. Predicting likely next queries with confidence scores
3. Pre-executing high-confidence predictions
4. Caching results for immediate retrieval
5. Serving cached results on cache hits

This pattern is particularly effective for conversational interfaces with predictable query patterns.

## Installation

```bash
pip install git+https://github.com/gyasis/hybridrag.git
```

## Basic Usage

```python
from promptchain.integrations.lightrag import (
    LightRAGIntegration,
    LightRAGSpeculativeExecutor,
    SpeculativeConfig
)

# Initialize
integration = LightRAGIntegration(working_dir="./lightrag_data")

# Create executor
executor = LightRAGSpeculativeExecutor(
    lightrag_core=integration,
    config=SpeculativeConfig(
        min_confidence=0.7,
        max_concurrent=3,
        default_ttl=60.0
    )
)

# Record query patterns
executor.record_call("What is ML?", mode="hybrid")
executor.record_call("What is deep learning?", mode="hybrid")

# Generate predictions and cache
result = await executor.execute(context="User asking about AI")

print(f"Predictions: {len(result.predictions)}")
print(f"Executed: {len(result.executed)}")
print(f"Cached: {len(result.cached_results)}")

# Check cache
cached = executor.check_cache("What is deep learning?", mode="hybrid")
if cached:
    print(f"Cache hit! Result: {cached.result}")
    print(f"Latency saved: {result.latency_saved_ms}ms")

# Get statistics
stats = executor.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

## Configuration

```python
SpeculativeConfig(
    min_confidence=0.7,       # Min confidence for execution
    max_concurrent=3,         # Max speculative queries
    default_ttl=60.0,         # Cache TTL in seconds
    prediction_model="frequency",  # "frequency" or "pattern"
    history_window=20         # Query history size
)
```

## Prediction Models

### Frequency-Based

```python
config = SpeculativeConfig(prediction_model="frequency")
```

Predicts based on query frequency and recency.

### Pattern-Based

```python
config = SpeculativeConfig(prediction_model="pattern")
```

Detects follow-up question patterns.

## Result Structure

```python
@dataclass
class SpeculativeExecutionResult:
    predictions: List[ToolPrediction]
    executed: List[ToolPrediction]
    cached_results: List[SpeculativeResult]
    actual_call: Optional[str]
    hit: bool
    latency_saved_ms: float
```

## Events

```python
"pattern.speculative.started"
"pattern.speculative.predicted"
"pattern.speculative.executed"
"pattern.speculative.completed"
```

## Best Practices

1. **Record All Queries**: Build history for better predictions
2. **Tune Confidence Threshold**: Balance cache utilization and waste
3. **Monitor Hit Rate**: Aim for >30% hit rate
4. **Set Appropriate TTL**: Balance freshness and cache efficiency
5. **Clean Expired Entries**: Call executor.cleanup() periodically

## Performance Metrics

```python
stats = executor.get_stats()

print(f"Total predictions: {stats['total_predictions']}")
print(f"Total executions: {stats['total_executions']}")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
print(f"Hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Cache size: {stats['cache_size']}")
```

## Limitations

1. **Pattern Dependency**: Requires predictable query patterns
2. **Memory Usage**: Caching increases memory consumption
3. **Staleness**: Cached results may become outdated
4. **Waste**: Low-confidence predictions waste resources

## Related Patterns

- Multi-Hop: Speculate on likely next hops
- Query Expansion: Speculate on likely expansions
