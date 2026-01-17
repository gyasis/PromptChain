# Data Model: Advanced Agentic Patterns

**Feature**: 004-advanced-agentic-patterns
**Date**: 2025-11-29
**Status**: Complete

## Core Entities

### 1. BasePattern

Base class for all pattern implementations.

```python
@dataclass
class PatternConfig:
    """Configuration for pattern execution."""
    pattern_id: str                    # Unique pattern identifier
    enabled: bool = True               # Whether pattern is active
    timeout_seconds: float = 30.0      # Maximum execution time
    emit_events: bool = True           # Emit to MessageBus
    use_blackboard: bool = False       # Use shared Blackboard

@dataclass
class PatternResult:
    """Result from pattern execution."""
    pattern_id: str                    # Pattern that generated result
    success: bool                      # Execution succeeded
    result: Any                        # Pattern-specific result
    metadata: Dict[str, Any]           # Execution metadata
    timing_ms: float                   # Execution time
    events_emitted: List[str]          # Event types emitted
```

---

### 2. Branching Thoughts (US1)

```python
@dataclass
class Hypothesis:
    """A potential solution path."""
    hypothesis_id: str                 # Unique identifier
    approach: str                      # Description of approach
    reasoning: str                     # Why this approach
    confidence: float                  # Self-assessed confidence 0-1
    metadata: Dict[str, Any]           # Additional context

@dataclass
class HypothesisScore:
    """Judge evaluation of a hypothesis."""
    hypothesis_id: str                 # Which hypothesis scored
    score: float                       # Normalized score 0-1
    reasoning: str                     # Justification for score
    strengths: List[str]               # Identified strengths
    weaknesses: List[str]              # Identified weaknesses
    judge_model: str                   # Model used for judging

@dataclass
class BranchingConfig(PatternConfig):
    """Branching thoughts configuration."""
    hypothesis_count: int = 3          # Number of hypotheses to generate
    generator_model: str = None        # Model for hypothesis generation
    judge_model: str = None            # Model for judging
    diversity_threshold: float = 0.3   # Minimum difference between hypotheses
    record_outcomes: bool = True       # Track for learning

@dataclass
class BranchingResult(PatternResult):
    """Result from branching thoughts execution."""
    hypotheses: List[Hypothesis]       # Generated hypotheses
    scores: List[HypothesisScore]      # Judge evaluations
    selected_hypothesis: Hypothesis    # Chosen path
    selection_reasoning: str           # Why selected
```

---

### 3. Query Expansion (US2)

```python
class ExpansionStrategy(Enum):
    """Query expansion strategies."""
    SYNONYM = "synonym"                # WordNet/embedding synonyms
    SEMANTIC = "semantic"              # LLM semantic variations
    ACRONYM = "acronym"                # Acronym expansion/contraction
    REFORMULATION = "reformulation"    # Query rephrasing

@dataclass
class ExpandedQuery:
    """A query variation."""
    query_id: str                      # Unique identifier
    original_query: str                # Original query text
    expanded_query: str                # Expanded variation
    strategy: ExpansionStrategy        # Strategy used
    similarity_score: float            # Similarity to original 0-1
    metadata: Dict[str, Any]           # Strategy-specific metadata

@dataclass
class QueryExpansionConfig(PatternConfig):
    """Query expansion configuration."""
    strategies: List[ExpansionStrategy] # Strategies to use
    max_expansions_per_strategy: int = 3
    min_similarity: float = 0.5        # Minimum semantic similarity
    deduplicate: bool = True           # Remove duplicate expansions
    parallel_search: bool = True       # Search expanded in parallel

@dataclass
class QueryExpansionResult(PatternResult):
    """Result from query expansion."""
    original_query: str                # Input query
    expanded_queries: List[ExpandedQuery]
    search_results: List[Any]          # Fused search results
    unique_results_found: int          # Results not in original
```

---

### 4. Sharded Retrieval (US3)

```python
class ShardType(Enum):
    """Types of data shards."""
    VECTOR_DB = "vector_db"            # Embedding-based (Pinecone, Weaviate)
    SQL_DB = "sql_db"                  # Relational database
    DOCUMENT_DB = "document_db"        # MongoDB, Elasticsearch
    API = "api"                        # External API endpoint
    FILE_SYSTEM = "file_system"        # Local/network files

@dataclass
class ShardConfig:
    """Configuration for a data shard."""
    shard_id: str                      # Unique identifier
    shard_type: ShardType              # Type of shard
    connection_config: Dict[str, Any]  # Connection details
    query_capabilities: List[str]      # Supported query types
    priority: int = 0                  # Query priority (higher first)
    timeout_seconds: float = 10.0      # Per-shard timeout
    enabled: bool = True               # Whether shard is active

@dataclass
class ShardHealth:
    """Health status of a shard."""
    shard_id: str
    available: bool
    latency_ms: float
    last_checked: datetime
    error_message: Optional[str]

@dataclass
class ShardResult:
    """Result from a single shard query."""
    shard_id: str                      # Source shard
    results: List[Any]                 # Retrieved items
    score_normalization: float         # Factor for score normalization
    query_time_ms: float               # Query execution time
    error: Optional[str]               # Error if failed

@dataclass
class ShardedRetrievalConfig(PatternConfig):
    """Sharded retrieval configuration."""
    shards: List[ShardConfig]          # Registered shards
    parallel: bool = True              # Query in parallel
    fail_partial: bool = True          # Return partial on failures
    aggregate_top_k: int = 10          # Top results after aggregation
    normalize_scores: bool = True      # Normalize across shards

@dataclass
class ShardedRetrievalResult(PatternResult):
    """Result from sharded retrieval."""
    query: str                         # Input query
    shard_results: List[ShardResult]   # Per-shard results
    aggregated_results: List[Any]      # Merged results
    shards_queried: int                # Number of shards queried
    shards_failed: int                 # Number of failures
    warnings: List[str]                # Failure warnings
```

---

### 5. Multi-Hop Retrieval (US4)

```python
@dataclass
class SubQuestion:
    """A decomposed sub-question."""
    question_id: str                   # Unique identifier
    question_text: str                 # Sub-question text
    parent_question: str               # Original question
    dependencies: List[str]            # IDs of dependent questions
    rationale: str                     # Why this sub-question
    answer: Optional[str] = None       # Answer when resolved
    retrieval_context: List[str] = None # Supporting evidence

@dataclass
class QuestionDependencyGraph:
    """Graph of sub-question dependencies."""
    nodes: Dict[str, SubQuestion]      # question_id -> SubQuestion
    edges: List[Tuple[str, str]]       # (from_id, to_id) dependencies
    execution_order: List[List[str]]   # Waves of parallel execution

@dataclass
class MultiHopConfig(PatternConfig):
    """Multi-hop retrieval configuration."""
    max_hops: int = 3                  # Maximum decomposition depth
    max_sub_questions: int = 5         # Max sub-questions per level
    detect_dependencies: bool = True   # Analyze dependencies
    retriever: Any = None              # Retrieval system to use
    synthesizer_model: str = None      # Model for answer synthesis

@dataclass
class MultiHopResult(PatternResult):
    """Result from multi-hop retrieval."""
    original_question: str             # Input question
    sub_questions: List[SubQuestion]   # Decomposed questions
    dependency_graph: QuestionDependencyGraph
    sub_answers: Dict[str, str]        # question_id -> answer
    unified_answer: str                # Synthesized answer
    hops_executed: int                 # Number of retrieval hops
    unanswered_aspects: List[str]      # Gaps in answer
```

---

### 6. Hybrid Search Fusion (US5)

```python
class SearchTechnique(Enum):
    """Search technique types."""
    EMBEDDING = "embedding"            # Vector similarity search
    KEYWORD = "keyword"                # Keyword/full-text search
    BM25 = "bm25"                      # BM25 ranking
    METADATA = "metadata"              # Metadata filtering

class FusionAlgorithm(Enum):
    """Result fusion algorithms."""
    RRF = "rrf"                        # Reciprocal Rank Fusion
    LINEAR = "linear"                  # Weighted linear combination
    BORDA = "borda"                    # Borda count

@dataclass
class TechniqueConfig:
    """Configuration for a search technique."""
    technique_type: SearchTechnique    # Type of technique
    weight: float = 1.0                # Fusion weight
    enabled: bool = True               # Whether to use
    params: Dict[str, Any] = None      # Technique-specific params

@dataclass
class TechniqueResult:
    """Result from a single search technique."""
    technique: SearchTechnique         # Which technique
    results: List[Any]                 # Retrieved items
    scores: List[float]                # Raw scores
    query_time_ms: float               # Execution time

@dataclass
class HybridSearchConfig(PatternConfig):
    """Hybrid search configuration."""
    techniques: List[TechniqueConfig]  # Techniques to use
    fusion_algorithm: FusionAlgorithm = FusionAlgorithm.RRF
    rrf_k: int = 60                    # RRF constant
    top_k: int = 10                    # Final result count
    normalize_scores: bool = True      # Normalize before fusion
    dynamic_weighting: bool = False    # Adjust weights per query

@dataclass
class HybridSearchResult(PatternResult):
    """Result from hybrid search."""
    query: str                         # Input query
    technique_results: List[TechniqueResult]
    fused_results: List[Any]           # Final fused results
    fused_scores: List[float]          # Final scores
    technique_contributions: Dict[str, int]  # Results per technique
```

---

### 7. Speculative Execution (US6)

```python
@dataclass
class ToolPrediction:
    """A predicted tool call."""
    prediction_id: str                 # Unique identifier
    tool_name: str                     # Predicted tool
    tool_args: Dict[str, Any]          # Predicted arguments
    confidence: float                  # Prediction confidence 0-1
    pattern_matched: str               # Pattern that triggered
    context_hash: str                  # Hash of prediction context

@dataclass
class SpeculativeResult:
    """Result from speculative execution."""
    prediction_id: str                 # Which prediction
    tool_name: str                     # Tool executed
    result: Any                        # Execution result
    cached_at: datetime                # When cached
    ttl_seconds: float                 # Time-to-live
    hit: bool = False                  # Was this result used?

@dataclass
class SpeculativeConfig(PatternConfig):
    """Speculative execution configuration."""
    min_confidence: float = 0.7        # Minimum to execute
    max_concurrent: int = 3            # Max parallel speculative calls
    default_ttl: float = 60.0          # Default cache TTL seconds
    prediction_model: str = "frequency" # "frequency" or "sequence"
    history_window: int = 20           # Tool calls to analyze

@dataclass
class SpeculativeExecutionResult(PatternResult):
    """Result from speculative execution cycle."""
    predictions: List[ToolPrediction]  # What was predicted
    executed: List[ToolPrediction]     # What was speculatively run
    cached_results: List[SpeculativeResult]
    actual_call: Optional[str]         # What was actually called
    hit: bool                          # Did prediction match?
    latency_saved_ms: float            # Time saved if hit
```

---

## Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                    PatternConfig (Base)                      │
│  ┌──────────────────┬──────────────────┬──────────────────┐ │
│  │ BranchingConfig  │ QueryExpConfig   │ ShardedConfig    │ │
│  │ MultiHopConfig   │ HybridConfig     │ SpeculativeConfig│ │
│  └──────────────────┴──────────────────┴──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PatternResult (Base)                      │
│  ┌──────────────────┬──────────────────┬──────────────────┐ │
│  │ BranchingResult  │ QueryExpResult   │ ShardedResult    │ │
│  │ MultiHopResult   │ HybridResult     │ SpeculativeResult│ │
│  └──────────────────┴──────────────────┴──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Integration Points                        │
│  ┌──────────────────┬──────────────────┬──────────────────┐ │
│  │    MessageBus    │    Blackboard    │ CapabilityReg.   │ │
│  │   (from 003)     │    (from 003)    │   (from 003)     │ │
│  └──────────────────┴──────────────────┴──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Validation Rules

### Hypothesis
- `confidence` must be 0.0 ≤ x ≤ 1.0
- `approach` must not be empty
- `hypothesis_id` must be unique within generation batch

### ExpandedQuery
- `similarity_score` must be 0.0 ≤ x ≤ 1.0
- `expanded_query` must differ from `original_query`
- `strategy` must be valid ExpansionStrategy

### ShardConfig
- `shard_id` must be unique
- `timeout_seconds` must be > 0
- `connection_config` must have required fields for shard_type

### SubQuestion
- No circular dependencies allowed
- `dependencies` must reference valid question_ids
- `question_text` must not be empty

### TechniqueConfig
- `weight` must be > 0
- At least one technique must be enabled

### ToolPrediction
- `confidence` must be 0.0 ≤ x ≤ 1.0
- `tool_name` must exist in tool registry
