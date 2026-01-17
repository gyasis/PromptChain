# Research: Advanced Agentic Patterns

**Feature**: 004-advanced-agentic-patterns
**Date**: 2025-11-29
**Status**: Complete (Updated with LightRAG Integration)

## Key Finding: LightRAG Integration Path

**Decision**: Integrate existing `/home/gyasis/Documents/code/hybridrag` implementation into PromptChain instead of building patterns from scratch.

**Rationale**:
- HybridRAG project already implements graph-based retrieval with LightRAG
- Has multi-mode queries (local, global, hybrid, naive, mix)
- Already integrates with PromptChain's AgenticStepProcessor
- Has FastMCP 2.0 server for MCP tool access
- Production-tested with document ingestion, folder watching, etc.

**Architecture Decision**: Create `promptchain.integrations.lightrag` module that wraps hybridrag components as PromptChain patterns.

---

## LightRAG Deep Research (via Gemini)

### LightRAG Framework Overview

**Graph-Enhanced RAG System**:
- Overcomes limitations of traditional flat-representation RAG
- Uses dual-level retrieval: low-level (entity-specific) + high-level (conceptual)
- Creates comprehensive knowledge graph from documents using LLM extraction
- Supports incremental updates without full graph regeneration

**Key Advantages over GraphRAG**:
- Lighter weight, faster, more affordable
- Fewer API calls and lower token consumption
- Cost-effective for frequent data updates
- KV data structures offer more precise retrieval than embedding-only RAG

**Architecture**:
```
Document → LLM Entity Extraction → Knowledge Graph
                ↓
    Vector Storage + Key-Value Storage + Graph Storage
                ↓
    Query → Dual-Level Retrieval → Answer Generation
```

---

## Existing HybridRAG Components to Integrate

### 1. LightRAG Core (`hybridrag/src/lightrag_core.py`)

**Already Implements**:
- `HybridLightRAGCore` class with query modes
- `local_query()` - Entity-specific retrieval
- `global_query()` - High-level overviews
- `hybrid_query()` - Combined approach
- `extract_context()` - Raw context extraction
- Async/sync parity

**Maps to 004 Patterns**:
- **Sharded Retrieval (US3)**: Multi-source querying already supported
- **Hybrid Search Fusion (US5)**: Local+global fusion = hybrid mode

### 2. Search Interface (`hybridrag/src/search_interface.py`)

**Already Implements**:
- `SearchInterface` class
- `simple_search()` - Direct LightRAG queries
- `agentic_search()` - Multi-hop reasoning with AgenticStepProcessor
- `multi_query_search()` - Parallel queries + synthesis
- Tool registration for PromptChain

**Maps to 004 Patterns**:
- **Multi-Hop Retrieval (US4)**: `agentic_search()` with max_steps
- **Query Expansion (US2)**: `multi_query_search()` with query variations

### 3. MCP Server (`hybridrag/hybridrag_mcp_server.py`)

**Already Implements**:
- FastMCP 2.0 compliant server
- `lightrag_local_query` tool
- `lightrag_global_query` tool
- `lightrag_hybrid_query` tool
- `get_database_info` tool

**Integration Approach**: Use MCP server as external tool or import core directly

---

## Pattern Implementation via LightRAG

### US1: Branching Thoughts

**Implementation**:
```python
# Use LightRAG multi-mode queries as hypothesis generators
async def generate_hypotheses(problem: str, count: int = 3):
    hypotheses = await asyncio.gather(
        lightrag.local_query(problem),   # Entity-focused hypothesis
        lightrag.global_query(problem),  # Overview hypothesis
        lightrag.hybrid_query(problem)   # Balanced hypothesis
    )
    return [Hypothesis(approach=h.result, mode=m)
            for h, m in zip(hypotheses, ["local", "global", "hybrid"])]
```

**Judge**: Use AgenticStepProcessor to evaluate and score hypotheses

### US2: Parallel Query Expansion

**Implementation**:
```python
# Already implemented in SearchInterface.multi_query_search()
# Extend with expansion strategies
async def expand_query(query: str, strategies: List[ExpansionStrategy]):
    expanded = []
    if ExpansionStrategy.SEMANTIC in strategies:
        # Use LightRAG to generate semantic variations
        context = await lightrag.extract_context(query)
        variations = generate_semantic_variations(query, context)
        expanded.extend(variations)
    return expanded
```

### US3: Sharded Retrieval

**Implementation**:
```python
# Register multiple LightRAG databases as shards
class ShardRegistry:
    def register_lightrag_shard(self, name: str, working_dir: str):
        core = HybridLightRAGCore(config_with_working_dir(working_dir))
        self.shards[name] = core

    async def query_all_shards(self, query: str):
        results = await asyncio.gather(*[
            shard.hybrid_query(query) for shard in self.shards.values()
        ])
        return aggregate_results(results)
```

### US4: Multi-Hop Retrieval

**Implementation**:
```python
# Already implemented in SearchInterface.agentic_search()
# Wrap with question decomposition
async def multi_hop_retrieve(question: str):
    # Decompose using AgenticStepProcessor
    processor = AgenticStepProcessor(
        objective=f"Answer: {question}",
        max_internal_steps=5
    )
    # Register LightRAG tools
    chain.add_tools(lightrag_tools)
    return await chain.process_prompt_async(question)
```

### US5: Hybrid Search Fusion

**Implementation**:
```python
# LightRAG hybrid mode IS the fusion
# Extend with RRF for explicit control
async def hybrid_search_with_rrf(query: str):
    local_result = await lightrag.local_query(query)
    global_result = await lightrag.global_query(query)
    # Apply RRF fusion
    return reciprocal_rank_fusion([local_result, global_result])
```

### US6: Speculative Execution

**Implementation**:
```python
# Predict next LightRAG query based on conversation
class LightRAGPredictor:
    def predict_next_query(self, context: str) -> List[ToolPrediction]:
        # Analyze conversation for likely next queries
        patterns = self.analyze_patterns(context)
        return [ToolPrediction(
            tool_name="lightrag_hybrid_query",
            tool_args={"query": p.predicted_query},
            confidence=p.confidence
        ) for p in patterns]
```

---

## Integration Architecture

```
PromptChain/
├── promptchain/
│   ├── patterns/                      # 004 Pattern Implementations
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── ...
│   └── integrations/
│       └── lightrag/                  # NEW: LightRAG integration
│           ├── __init__.py
│           ├── core.py                # Wrapper for HybridLightRAGCore
│           ├── search.py              # Wrapper for SearchInterface
│           └── patterns.py            # Pattern implementations via LightRAG

HybridRAG/ (external project - reference/import)
├── src/
│   ├── lightrag_core.py
│   ├── search_interface.py
│   └── ...
└── hybridrag_mcp_server.py
```

---

## Performance Benchmarks (From Gemini Research)

| Pattern | LightRAG Approach | Expected Performance |
|---------|-------------------|---------------------|
| Multi-Hop | Graph traversal | 74%+ improvement on complex reasoning |
| Hybrid Search | Local+Global fusion | Superior to single-mode retrieval |
| Query Expansion | Context-aware generation | 30-50% recall improvement |
| Speculative | Pattern-based prediction | 60%+ hit rate achievable |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LightRAG dependency | Abstract behind integration layer |
| Version compatibility | Pin LightRAG version, test matrix |
| Performance overhead | Lazy initialization, caching |
| API changes | Interface contracts, adapter pattern |

---

## Development Approach

**Wave-Based Parallelization**:
1. **Wave 1**: Create integration layer (imports, wrappers)
2. **Wave 2**: Implement pattern adapters (parallel - no file conflicts)
3. **Wave 3**: Integration tests with MessageBus/Blackboard
4. **Wave 4**: CLI integration and documentation

**File Locking Strategy**:
- Each pattern file owned by one agent
- Core integration files sequential
- Tests can run parallel per pattern

---

## Dependencies

- `lightrag>=0.1.0` - Core graph RAG
- `promptchain` - Existing (this project)
- `hybridrag` - Can be: (a) pip installed or (b) path import

**Installation Options**:
```bash
# Option A: Install hybridrag from git (RECOMMENDED)
pip install git+https://github.com/gyasis/hybridrag.git

# Option B: Install from local path (development)
pip install -e /home/gyasis/Documents/code/hybridrag

# Option C: Path-based import (fallback)
sys.path.insert(0, '/home/gyasis/Documents/code/hybridrag/src')
```

**Note**: HybridRAG has its own git repository. Install from git for production use.

---

## Conclusion

**Recommendation**: Leverage existing hybridrag project as the implementation backbone for 004 patterns. This provides:
- Battle-tested LightRAG integration
- Existing PromptChain integration via AgenticStepProcessor
- MCP server for external tool access
- Reduces implementation effort by ~60%

**Next Step**: Create Wave-parallelized tasks.md with file locking
