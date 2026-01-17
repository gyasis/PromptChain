# Sharded Retrieval Pattern

Parallel retrieval across multiple LightRAG database shards with aggregation and fault tolerance.

## Overview

The Sharded Retrieval pattern enables distributed knowledge base querying by:
1. Registering multiple LightRAG shards (databases)
2. Querying shards in parallel or sequentially
3. Aggregating results with score normalization
4. Handling shard failures gracefully

This pattern is essential for large-scale distributed systems with geographic distribution or multi-tenant architectures.

## Installation

```bash
pip install git+https://github.com/gyasis/hybridrag.git
```

## Basic Usage

```python
from promptchain.integrations.lightrag import (
    LightRAGShardRegistry,
    LightRAGShardedRetriever,
    ShardConfig,
    ShardType,
    ShardedRetrievalConfig
)

# Create shard registry
registry = LightRAGShardRegistry()

# Register shards
registry.register_shard(ShardConfig(
    shard_id="shard_us_east",
    shard_type=ShardType.LIGHTRAG,
    working_dir="./shards/us_east",
    priority=1
))

registry.register_shard(ShardConfig(
    shard_id="shard_us_west",
    shard_type=ShardType.LIGHTRAG,
    working_dir="./shards/us_west",
    priority=0
))

# Create retriever
retriever = LightRAGShardedRetriever(
    registry=registry,
    config=ShardedRetrievalConfig(
        parallel=True,
        aggregate_top_k=10
    )
)

# Query across all shards
result = await retriever.execute(query="What is machine learning?")

print(f"Queried {result.shards_queried} shards")
print(f"Found {len(result.aggregated_results)} results")
print(f"Failures: {result.shards_failed}")
```

## Configuration

### ShardConfig

```python
ShardConfig(
    shard_id="shard_001",
    shard_type=ShardType.LIGHTRAG,
    working_dir="./shard_001",
    priority=0,              # Priority for score weighting
    timeout_seconds=10.0,    # Per-shard timeout
    enabled=True
)
```

### ShardedRetrievalConfig

```python
ShardedRetrievalConfig(
    parallel=True,           # Query shards in parallel
    fail_partial=True,       # Continue if some shards fail
    aggregate_top_k=10,      # Top results to return
    normalize_scores=True    # Normalize scores across shards
)
```

## Events

```python
"pattern.sharded.started"        # Query started
"pattern.sharded.shard_queried"  # Individual shard completed
"pattern.sharded.shard_failed"   # Individual shard failed
"pattern.sharded.completed"      # All shards queried
```

## Best Practices

1. **Set Per-Shard Timeouts**: Prevent slow shards from blocking
2. **Use Priority Weighting**: Higher priority for authoritative shards
3. **Enable Partial Failure**: Don't fail entire query on single shard failure
4. **Monitor Shard Health**: Use registry.health_check()

## Related Patterns

- Query Expansion: Apply expansion before sharded retrieval
- Hybrid Search: Use sharding with fusion algorithms
