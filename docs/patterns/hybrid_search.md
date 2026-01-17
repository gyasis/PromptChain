# Hybrid Search Fusion Pattern

Combine multiple search techniques (local, global, hybrid) with explicit fusion algorithms (RRF, linear, Borda) to produce optimal retrieval results.

## Overview

The Hybrid Search Fusion pattern maximizes retrieval quality by:
1. Executing multiple search techniques in parallel
2. Collecting ranked results from each technique
3. Fusing results using sophisticated ranking algorithms
4. Returning top-k unified results

This pattern is ideal for production search systems requiring maximum accuracy.

## Installation

```bash
pip install git+https://github.com/gyasis/hybridrag.git
```

## Basic Usage

```python
from promptchain.integrations.lightrag import (
    LightRAGIntegration,
    LightRAGHybridSearcher,
    HybridSearchConfig,
    SearchTechnique,
    FusionAlgorithm
)

# Initialize
integration = LightRAGIntegration(working_dir="./lightrag_data")

# Create searcher
searcher = LightRAGHybridSearcher(
    lightrag_integration=integration,
    config=HybridSearchConfig(
        techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
        fusion_algorithm=FusionAlgorithm.RRF,
        top_k=10
    )
)

# Execute
result = await searcher.execute(query="What is deep learning?")

print(f"Fused {len(result.fused_results)} results")
print(f"Contributions: {result.technique_contributions}")

# Review technique results
for tech_result in result.technique_results:
    print(f"
{tech_result.technique.value}: {len(tech_result.results)} results")
```

## Search Techniques

- **LOCAL**: Entity-focused search with specific details
- **GLOBAL**: Concept-focused search with high-level patterns
- **HYBRID**: Balanced combination of entity and concept
- **NAIVE**: Basic retrieval (fallback)
- **MIX**: Explicit local+global combination

## Fusion Algorithms

### Reciprocal Rank Fusion (RRF)

```python
config = HybridSearchConfig(
    fusion_algorithm=FusionAlgorithm.RRF,
    rrf_k=60  # Standard value from literature
)
```

Formula: `score(doc) = sum(1 / (k + rank))`

**Best for**: Rank-based fusion, standard approach

### Linear Fusion

```python
config = HybridSearchConfig(
    fusion_algorithm=FusionAlgorithm.LINEAR,
    normalize_scores=True
)
```

**Best for**: Score-based fusion with normalization

### Borda Count

```python
config = HybridSearchConfig(
    fusion_algorithm=FusionAlgorithm.BORDA
)
```

**Best for**: Voting-based fusion, position-aware

## Configuration

```python
HybridSearchConfig(
    techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
    fusion_algorithm=FusionAlgorithm.RRF,
    rrf_k=60,
    top_k=10,
    normalize_scores=True
)
```

## Events

```python
"pattern.hybrid_search.started"
"pattern.hybrid_search.technique_completed"
"pattern.hybrid_search.technique_failed"
"pattern.hybrid_search.fusing"
"pattern.hybrid_search.completed"
```

## Best Practices

1. **Use RRF for Production**: Most robust fusion algorithm
2. **Combine LOCAL + GLOBAL**: Best coverage
3. **Monitor Technique Performance**: Track which techniques contribute most
4. **Set Appropriate top_k**: 10-20 for most applications

## Related Patterns

- Query Expansion: Expand before hybrid search
- Sharded Retrieval: Apply fusion within each shard
