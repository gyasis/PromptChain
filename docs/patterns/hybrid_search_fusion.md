# LightRAG Hybrid Search Fusion Pattern

**Pattern 7/14: Hybrid Search Fusion**

The Hybrid Search Fusion pattern combines multiple LightRAG retrieval techniques (local, global, hybrid, naive, mix) using sophisticated fusion algorithms to produce optimal search results.

## Overview

This pattern addresses the challenge that different search techniques excel at different types of queries:

- **Local queries**: Best for entity-specific, detailed retrieval
- **Global queries**: Best for high-level, conceptual retrieval
- **Hybrid queries**: Combines local and global approaches
- **Naive queries**: Simple baseline retrieval
- **Mix queries**: Equal blend of local and global

By executing multiple techniques in parallel and fusing their results using algorithms like Reciprocal Rank Fusion (RRF), Linear Combination, or Borda Count, this pattern achieves more robust and comprehensive retrieval than any single technique alone.

## Key Components

### SearchTechnique Enum

```python
class SearchTechnique(Enum):
    LOCAL = "local"      # Entity-specific retrieval
    GLOBAL = "global"    # Conceptual overview retrieval
    HYBRID = "hybrid"    # Combined local+global
    NAIVE = "naive"      # Baseline retrieval
    MIX = "mix"         # Equal blend of local+global
```

### FusionAlgorithm Enum

```python
class FusionAlgorithm(Enum):
    RRF = "rrf"          # Reciprocal Rank Fusion
    LINEAR = "linear"    # Weighted linear combination
    BORDA = "borda"      # Borda count voting
```

### HybridSearchConfig

```python
@dataclass
class HybridSearchConfig(PatternConfig):
    techniques: List[SearchTechnique] = [LOCAL, GLOBAL]
    fusion_algorithm: FusionAlgorithm = RRF
    rrf_k: int = 60                    # RRF constant (standard: 60)
    top_k: int = 10                    # Number of results to return
    normalize_scores: bool = True       # Normalize scores before fusion
```

### HybridSearchResult

```python
@dataclass
class HybridSearchResult(PatternResult):
    query: str                                  # Original query
    technique_results: List[TechniqueResult]    # Per-technique results
    fused_results: List[Any]                    # Final fused results
    fused_scores: List[float]                   # Fusion scores
    technique_contributions: Dict[str, int]     # Contribution counts
```

## Fusion Algorithms

### Reciprocal Rank Fusion (RRF)

**Formula**: `score(doc) = Σ 1 / (k + rank_in_technique)`

RRF is robust and parameter-free (k=60 is standard). It assigns higher scores to documents that appear high in multiple rankings.

**Best for**: General-purpose fusion when you want equal weighting of techniques

```python
config = HybridSearchConfig(
    techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
    fusion_algorithm=FusionAlgorithm.RRF,
    rrf_k=60,  # Standard value from literature
)
```

### Linear Fusion

**Formula**: `score(doc) = Σ weight_i × normalized_score_i`

Linear fusion combines scores using weighted averages. Scores can be normalized to [0, 1] range.

**Best for**: When you want to weight certain techniques higher than others

```python
config = HybridSearchConfig(
    techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
    fusion_algorithm=FusionAlgorithm.LINEAR,
    normalize_scores=True,  # Normalize before weighting
)
```

### Borda Count

**Formula**: `score(doc) = Σ (num_results - rank)`

Borda count uses rank-based voting where higher-ranked documents get more points.

**Best for**: Democratic fusion where all techniques vote equally

```python
config = HybridSearchConfig(
    techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
    fusion_algorithm=FusionAlgorithm.BORDA,
)
```

## Usage Examples

### Basic RRF Fusion

```python
from promptchain.integrations.lightrag.core import LightRAGIntegration
from promptchain.integrations.lightrag.hybrid_search import (
    LightRAGHybridSearcher,
    HybridSearchConfig,
    SearchTechnique,
    FusionAlgorithm,
)

# Initialize LightRAG
integration = LightRAGIntegration()

# Configure hybrid search
config = HybridSearchConfig(
    techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
    fusion_algorithm=FusionAlgorithm.RRF,
    top_k=10,
)

# Create searcher
searcher = LightRAGHybridSearcher(integration, config=config)

# Execute search
result = await searcher.execute(query="What is machine learning?")

print(f"Found {len(result.fused_results)} results")
print(f"Technique contributions: {result.technique_contributions}")
```

### Multi-Technique with Linear Fusion

```python
config = HybridSearchConfig(
    techniques=[
        SearchTechnique.LOCAL,
        SearchTechnique.GLOBAL,
        SearchTechnique.HYBRID
    ],
    fusion_algorithm=FusionAlgorithm.LINEAR,
    normalize_scores=True,
    top_k=15,
)

searcher = LightRAGHybridSearcher(integration, config=config)
result = await searcher.execute(query="Explain neural networks")

# Access individual technique results
for tech_result in result.technique_results:
    print(f"{tech_result.technique.value}: {len(tech_result.results)} results")
```

### Dynamic top_k Override

```python
# Configure with default top_k=10
config = HybridSearchConfig(
    techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
    fusion_algorithm=FusionAlgorithm.RRF,
    top_k=10,
)

searcher = LightRAGHybridSearcher(integration, config=config)

# Override at execution time
result = await searcher.execute(
    query="Computer vision",
    top_k=5  # Override to get only top 5
)
```

### Event Tracking

```python
config = HybridSearchConfig(
    techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
    fusion_algorithm=FusionAlgorithm.RRF,
    emit_events=True,  # Enable events
)

searcher = LightRAGHybridSearcher(integration, config=config)

# Add event handler
def track_events(event_type: str, data: dict):
    print(f"Event: {event_type}")
    if "num_techniques" in data:
        print(f"  Techniques: {data['num_techniques']}")

searcher.add_event_handler(track_events)

result = await searcher.execute(query="Deep learning")
```

## Events

The pattern emits the following events:

- `pattern.hybrid_search.started`: Search initiated
- `pattern.hybrid_search.technique_completed`: Individual technique finished
- `pattern.hybrid_search.technique_failed`: Technique encountered error
- `pattern.hybrid_search.fusing`: Beginning fusion process
- `pattern.hybrid_search.completed`: Search finished
- `pattern.hybrid_search.timeout`: Execution timed out
- `pattern.hybrid_search.error`: Error occurred

## Integration with 003 Infrastructure

### MessageBus Integration

```python
from promptchain.cli.models import MessageBus

bus = MessageBus()
searcher.connect_messagebus(bus)

# Subscribe to events
def handle_completion(event_type: str, data: dict):
    print(f"Search completed in {data['execution_time_ms']}ms")

bus.subscribe("pattern.hybrid_search.completed", handle_completion)
```

### Blackboard Integration

```python
from promptchain.cli.models import Blackboard

blackboard = Blackboard()
searcher.connect_blackboard(blackboard)

# Share results
config.use_blackboard = True
result = await searcher.execute(query="Machine learning")

# Results are automatically shared to blackboard
cached_results = blackboard.read("hybrid_search_results")
```

## Performance Characteristics

### Parallel Execution

All configured techniques execute in parallel using `asyncio.gather()`:

```python
# These run concurrently, not sequentially:
# - local_query("query")
# - global_query("query")
# - hybrid_query("query")
```

**Speedup**: ~2-3x faster than sequential execution for 3 techniques

### Fusion Complexity

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| RRF       | O(n log n)     | O(n)            |
| Linear    | O(n log n)     | O(n)            |
| Borda     | O(n log n)     | O(n)            |

Where n = total unique documents across all techniques.

### Typical Execution Times

- 2 techniques (local + global): ~200-500ms
- 3 techniques (local + global + hybrid): ~300-700ms
- Fusion overhead: ~5-20ms (negligible)

## Best Practices

### Technique Selection

```python
# For factual queries → use LOCAL + GLOBAL
config = HybridSearchConfig(
    techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
    fusion_algorithm=FusionAlgorithm.RRF,
)

# For comprehensive coverage → use all techniques
config = HybridSearchConfig(
    techniques=[
        SearchTechnique.LOCAL,
        SearchTechnique.GLOBAL,
        SearchTechnique.HYBRID
    ],
    fusion_algorithm=FusionAlgorithm.LINEAR,
)

# For simple baseline → use single technique
config = HybridSearchConfig(
    techniques=[SearchTechnique.HYBRID],
    fusion_algorithm=FusionAlgorithm.RRF,  # Still works with 1 technique
)
```

### Fusion Algorithm Selection

```python
# DEFAULT: Use RRF for most cases
# - Robust and parameter-free
# - Works well across different query types
fusion_algorithm=FusionAlgorithm.RRF

# Use LINEAR when you want custom weighting
# - Enable normalize_scores=True for fair comparison
fusion_algorithm=FusionAlgorithm.LINEAR

# Use BORDA for democratic voting
# - All techniques contribute equally
fusion_algorithm=FusionAlgorithm.BORDA
```

### Parameter Tuning

```python
# RRF k parameter (default: 60)
rrf_k=60   # Standard value (recommended)
rrf_k=100  # Less emphasis on rank differences
rrf_k=30   # More emphasis on rank differences

# Top-k selection
top_k=10   # Default, good for most queries
top_k=5    # Quick overview
top_k=20   # Comprehensive results
```

## Error Handling

The pattern handles failures gracefully:

```python
# If one technique fails, others continue
config = HybridSearchConfig(
    techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
)

# Even if LOCAL fails, GLOBAL results are still returned
result = await searcher.execute(query="test")

# Check which techniques succeeded
successful = [tr.technique.value for tr in result.technique_results]
print(f"Successful techniques: {successful}")
```

## Limitations

1. **Duplicate Detection**: Results are deduplicated using `str(result)` - complex objects may need custom hashing
2. **Score Normalization**: Only available for LINEAR fusion
3. **Technique Failures**: Failed techniques are logged but don't stop execution
4. **Memory**: Stores all results from all techniques in memory before fusion

## See Also

- [LightRAG Integration Core](./lightrag_core.md)
- [Multi-Hop Retrieval Pattern](./multi_hop.md)
- [Query Expansion Pattern](./query_expansion.md)
- [BasePattern API](./base_pattern.md)

## References

- RRF Paper: [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- Borda Count: [Social Choice Theory](https://en.wikipedia.org/wiki/Borda_count)
- LightRAG: [Graph-based RAG](https://github.com/HKUDS/LightRAG)
