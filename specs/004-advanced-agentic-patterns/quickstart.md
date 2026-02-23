# Quickstart: Advanced Agentic Patterns

**Feature**: 004-advanced-agentic-patterns
**Date**: 2025-11-29

## Overview

This feature adds 6 advanced agentic patterns to PromptChain, completing 100% coverage of the 14 pillars of production-grade agentic AI.

## Installation

```bash
# Already installed with PromptChain
pip install -e .
```

## Quick Examples

### 1. Branching Thoughts (US1)

Generate multiple hypotheses and let a judge select the best path:

```python
from promptchain.patterns import BranchingThoughts

# Initialize with default models
brancher = BranchingThoughts()

# Generate and evaluate hypotheses
result = await brancher.execute(
    problem="How can we reduce API latency while maintaining data consistency?"
)

print(f"Selected approach: {result.selected_hypothesis.approach}")
print(f"Reasoning: {result.selection_reasoning}")
print(f"Score: {result.scores[0].score:.2f}")
```

### 2. Parallel Query Expansion (US2)

Expand queries and search in parallel for better recall:

```python
from promptchain.patterns import QueryExpander, ExpansionStrategy

expander = QueryExpander(
    strategies=[
        ExpansionStrategy.SYNONYM,
        ExpansionStrategy.SEMANTIC,
        ExpansionStrategy.ACRONYM
    ]
)

# Define your search function
async def my_search(query: str) -> list:
    return your_search_implementation(query)

# Expand and search
result = await expander.execute(
    query="ML model performance optimization",
    searcher=my_search
)

print(f"Original found: {len(result.search_results) - result.unique_results_found}")
print(f"Unique from expansion: {result.unique_results_found}")
```

### 3. Sharded Retrieval (US3)

Query across multiple data sources:

```python
from promptchain.patterns import ShardRegistry, ShardedRetriever, ShardConfig, ShardType

# Set up shard registry
registry = ShardRegistry()

# Register data sources
registry.register_shard(ShardConfig(
    shard_id="docs_db",
    shard_type=ShardType.VECTOR_DB,
    connection_config={"endpoint": "http://localhost:8000"}
))

registry.register_shard(ShardConfig(
    shard_id="code_db",
    shard_type=ShardType.DOCUMENT_DB,
    connection_config={"endpoint": "http://localhost:9200"}
))

# Query all shards in parallel
retriever = ShardedRetriever(registry)
result = await retriever.query("authentication implementation")

print(f"Queried {result.shards_queried} shards")
print(f"Found {len(result.aggregated_results)} results")
if result.shards_failed > 0:
    print(f"Warnings: {result.warnings}")
```

### 4. Multi-Hop Retrieval (US4)

Decompose complex questions for comprehensive answers:

```python
from promptchain.patterns import MultiHopRetriever

retriever = MultiHopRetriever(
    retriever=your_base_retriever,
    max_hops=3
)

# Complex question requiring multiple lookups
result = await retriever.execute(
    question="What is the company's revenue model and how does it compare to competitors in the same market segment?"
)

print(f"Decomposed into {len(result.sub_questions)} sub-questions")
print(f"Executed {result.hops_executed} hops")
print(f"Answer: {result.unified_answer}")

if result.unanswered_aspects:
    print(f"Gaps: {result.unanswered_aspects}")
```

### 5. Hybrid Search Fusion (US5)

Combine multiple search techniques:

```python
from promptchain.patterns import (
    HybridSearcher,
    SearchTechnique,
    FusionAlgorithm
)

searcher = HybridSearcher(
    techniques=[
        SearchTechnique.EMBEDDING,
        SearchTechnique.KEYWORD,
        SearchTechnique.BM25
    ],
    fusion_algorithm=FusionAlgorithm.RRF
)

result = await searcher.hybrid_search(
    query="machine learning deployment best practices"
)

print(f"Fused {len(result.technique_results)} techniques")
print(f"Contribution: {result.technique_contributions}")
for item, score in zip(result.fused_results[:3], result.fused_scores[:3]):
    print(f"  {score:.3f}: {item}")
```

### 6. Speculative Execution (US6)

Pre-execute predicted tool calls for lower latency:

```python
from promptchain.patterns import SpeculativeExecutor

executor = SpeculativeExecutor(
    tool_registry=your_tool_registry,
    min_confidence=0.7
)

# Run speculative execution cycle
result = await executor.execute(
    context="User is asking about weather in NYC..."
)

print(f"Predicted {len(result.predictions)} tools")
print(f"Executed {len(result.executed)} speculatively")

if result.hit:
    print(f"Cache hit! Saved {result.latency_saved_ms:.0f}ms")
```

## Integration with 003 Infrastructure

All patterns integrate with the multi-agent communication infrastructure:

```python
from promptchain.cli.models import MessageBus, Blackboard
from promptchain.patterns import BranchingThoughts

# Set up infrastructure
bus = MessageBus(session_id="my-session")
blackboard = Blackboard(session_id="my-session")

# Pattern with integration
brancher = BranchingThoughts(
    emit_events=True,      # Emit to MessageBus
    use_blackboard=True    # Share state via Blackboard
)

# Subscribe to pattern events
bus.subscribe("pattern.branching.*", lambda event: print(f"Event: {event}"))

# Execute
result = await brancher.execute(problem="Optimize database queries")
```

## Configuration

### Global Pattern Config

```python
from promptchain.patterns import PatternConfig

config = PatternConfig(
    pattern_id="my-branching",
    enabled=True,
    timeout_seconds=30.0,
    emit_events=True,
    use_blackboard=False
)
```

### Per-Pattern Configuration

Each pattern has specialized configuration:

```python
# Branching Thoughts
BranchingConfig(
    hypothesis_count=3,
    diversity_threshold=0.3,
    record_outcomes=True
)

# Query Expansion
QueryExpansionConfig(
    strategies=[ExpansionStrategy.SYNONYM, ExpansionStrategy.SEMANTIC],
    max_expansions_per_strategy=3,
    deduplicate=True
)

# Speculative Execution
SpeculativeConfig(
    min_confidence=0.7,
    max_concurrent=3,
    default_ttl=60.0
)
```

## Testing

Run pattern tests:

```bash
# Unit tests
pytest tests/unit/patterns/

# Integration tests
pytest tests/integration/patterns/

# Specific pattern
pytest tests/unit/patterns/test_branching_thoughts.py -v
```

## Next Steps

1. See [spec.md](./spec.md) for full requirements
2. See [data-model.md](./data-model.md) for entity definitions
3. See [contracts/](./contracts/) for API specifications
4. See [research.md](./research.md) for design decisions
