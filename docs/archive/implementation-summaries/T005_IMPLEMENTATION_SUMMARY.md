# T005: LightRAG Sharded Retrieval Pattern - Implementation Summary

## Overview

Implemented a complete sharded retrieval pattern for LightRAG that enables parallel querying across multiple database shards with advanced features including fault tolerance, priority-based ranking, and score normalization.

## Files Created

### 1. Core Implementation
**File:** `/home/gyasis/Documents/code/PromptChain/promptchain/integrations/lightrag/sharded.py`

**Components:**
- `ShardType` enum: Defines shard types (LIGHTRAG, VECTOR_DB, DOCUMENT_DB, API)
- `ShardConfig` dataclass: Configuration for individual shards with priority and timeout settings
- `ShardResult` dataclass: Results from individual shard queries with timing information
- `ShardedRetrievalConfig` dataclass: Pattern configuration extending PatternConfig
- `ShardedRetrievalResult` dataclass: Aggregated results with shard-level statistics
- `LightRAGShardRegistry` class: Manages shard registration, lifecycle, and health checking
- `LightRAGShardedRetriever` class: Main pattern implementation extending BasePattern

**Key Features:**
- Parallel or sequential shard querying
- Per-shard timeout handling
- Graceful error handling with partial failure support
- Priority-based result weighting
- Score normalization across shards
- Top-k result aggregation
- Event emission for monitoring (pattern.sharded.*)

### 2. Comprehensive Tests
**File:** `/home/gyasis/Documents/code/PromptChain/tests/integrations/lightrag/test_sharded.py`

**Test Coverage (23 tests, 100% passing):**

**ShardConfig Tests:**
- Default and custom configuration values

**ShardResult Tests:**
- Successful and failed shard results
- Success property based on error state

**LightRAGShardRegistry Tests:**
- Registry initialization
- Shard registration (LIGHTRAG and non-LIGHTRAG types)
- Duplicate shard detection
- Shard retrieval and configuration access
- Health checking for enabled/disabled shards
- ImportError handling when hybridrag unavailable

**LightRAGShardedRetriever Tests:**
- Execution with no shards
- Single and parallel shard execution
- Timeout handling with graceful degradation
- Error handling with fail_partial flag
- Priority weighting in result aggregation
- Score normalization
- Top-k result limiting
- Selective shard querying
- Score extraction from various result formats
- Event emission integration

**Testing Approach:**
- Used `mock_hybridrag` fixture to inject mocked hybridrag module into sys.modules
- Mocked HybridLightRAGCore and async query methods
- Tested both success and failure paths
- Verified timeout and error handling behavior
- Confirmed event emission and statistics tracking

### 3. Usage Examples
**File:** `/home/gyasis/Documents/code/PromptChain/examples/lightrag_sharded_example.py`

**Examples Included:**
1. **Basic Example**: Query across multiple LIGHTRAG shards with different priorities
2. **Selective Shard Example**: Query specific shards only (subset of all registered)
3. **Fault Tolerance Example**: Handling shard failures with fail_partial flag
4. **Health Check Example**: Pre-query health checking of shard availability
5. **Mixed Shard Types Example**: Registry with different shard types (VECTOR_DB, API)

## Architecture Highlights

### Shard Registry Pattern
```python
registry = LightRAGShardRegistry()

# Register LIGHTRAG shard
registry.register_shard(ShardConfig(
    shard_id="tech_docs",
    shard_type=ShardType.LIGHTRAG,
    working_dir="./data/tech",
    priority=5,
    timeout_seconds=10.0
))

# Health check
health = registry.health_check()
```

### Parallel Retrieval with Aggregation
```python
retriever = LightRAGShardedRetriever(
    registry=registry,
    config=ShardedRetrievalConfig(
        parallel=True,  # asyncio.gather for concurrent queries
        fail_partial=True,  # Continue if some shards fail
        aggregate_top_k=10,  # Top 10 after aggregation
        normalize_scores=True  # Min-max normalization
    )
)

result = await retriever.execute(query="AI trends")
```

### Result Structure
```python
ShardedRetrievalResult(
    success=True,
    query="AI trends",
    shard_results=[
        ShardResult(shard_id="shard1", results=[...], query_time_ms=150.0),
        ShardResult(shard_id="shard2", results=[...], query_time_ms=120.0),
    ],
    aggregated_results=[...],  # Top-k after normalization
    shards_queried=2,
    shards_failed=0,
    execution_time_ms=155.0
)
```

## Technical Details

### Async/Parallel Execution
- Uses `asyncio.gather()` for parallel shard queries
- Per-shard timeout handling with `asyncio.wait_for()`
- Graceful degradation on timeout or errors

### Score Aggregation Algorithm
1. Extract scores from result objects (supports `.score`, `["score"]`, `["relevance"]`)
2. Apply priority weighting: `weighted_score = score * (1.0 + priority * 0.1)`
3. Optional min-max normalization across all results
4. Sort by weighted score (descending)
5. Return top-k results

### Event Emission
```python
# Events emitted during execution:
- pattern.sharded.started
- pattern.sharded.shard_queried  (per shard)
- pattern.sharded.shard_failed   (on errors)
- pattern.sharded.completed
```

### Error Handling
- **Per-Shard Timeouts**: Individual shards timeout independently
- **Partial Failures**: With `fail_partial=True`, returns results from successful shards
- **Complete Failures**: With `fail_partial=False`, fails if any shard fails
- **Error Tracking**: Each `ShardResult` includes optional error message

## Integration with BasePattern

Extends `promptchain.patterns.base.BasePattern`:
- Event emission to MessageBus
- State sharing via Blackboard (optional)
- Timeout handling at pattern level
- Statistics tracking (execution count, average time)
- Standard `PatternResult` structure

## ImportError Handling

Gracefully handles missing `hybridrag` dependency:
```python
if not LIGHTRAG_AVAILABLE:
    raise ImportError(
        "hybridrag is not installed. Install with: "
        "pip install git+https://github.com/gyasis/hybridrag.git"
    )
```

## Use Cases

1. **Distributed Knowledge Retrieval**: Query across domain-specific knowledge shards (tech, business, research)
2. **Geographic Sharding**: Route queries to region-specific shards
3. **Temporal Sharding**: Query recent vs historical data shards
4. **Fault Tolerance**: Continue operation when some shards are unavailable
5. **Load Balancing**: Distribute query load across multiple shards
6. **Hybrid Search**: Combine LIGHTRAG with vector DB and API sources

## Performance Characteristics

- **Parallel Execution**: Reduces total query time to max(shard_times) instead of sum(shard_times)
- **Token Efficiency**: Aggregates top-k results to minimize token usage
- **Timeout Protection**: Per-shard timeouts prevent slow shards from blocking queries
- **Graceful Degradation**: Returns partial results when some shards fail

## Future Enhancements

Possible extensions:
- Caching layer for frequent queries
- Dynamic shard selection based on query routing
- Shard-level result reranking strategies
- Federated search across heterogeneous sources
- Query performance analytics and optimization

## Compliance with Requirements

✅ All requirements from T005 specification met:
- [x] Import from promptchain.patterns.base
- [x] ShardType enum with 4 values
- [x] ShardConfig dataclass with all fields
- [x] ShardResult dataclass with success property
- [x] ShardedRetrievalConfig extending PatternConfig
- [x] ShardedRetrievalResult extending PatternResult
- [x] LightRAGShardRegistry with all methods
- [x] LightRAGShardedRetriever extending BasePattern
- [x] Async execute() implementation
- [x] Parallel shard querying
- [x] Timeout and error handling
- [x] Result aggregation with normalization
- [x] Event emission (4 event types)
- [x] ImportError handling for hybridrag
- [x] Code style matching base.py
- [x] Comprehensive test coverage

## Testing Results

```
======================== 23 passed, 45 warnings in 2.32s ========================
```

All tests passing with comprehensive coverage of:
- Configuration and data models
- Registry operations
- Parallel execution
- Error handling
- Timeout behavior
- Result aggregation
- Event emission
