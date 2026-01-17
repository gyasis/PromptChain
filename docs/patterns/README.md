# LightRAG Pattern Adapters

Advanced agentic patterns for PromptChain powered by [HybridRAG](https://github.com/gyasis/hybridrag) integration with LightRAG.

## Overview

PromptChain provides six sophisticated agentic patterns that leverage LightRAG's knowledge graph capabilities to implement state-of-the-art retrieval and reasoning strategies. These patterns integrate seamlessly with the 003-multi-agent-communication infrastructure (MessageBus, Blackboard) for distributed AI systems.

### Available Patterns

1. **[Branching Thoughts](./branching.md)** - Generate and evaluate multiple hypotheses using different query modes
2. **[Query Expansion](./query_expansion.md)** - Expand queries with synonyms, semantics, and reformulations
3. **[Sharded Retrieval](./sharded.md)** - Parallel retrieval across multiple LightRAG database shards
4. **[Multi-Hop Retrieval](./multi_hop.md)** - Iterative question decomposition and reasoning
5. **[Hybrid Search Fusion](./hybrid_search.md)** - Combine multiple search techniques with fusion algorithms
6. **[Speculative Execution](./speculative.md)** - Predict and pre-execute likely queries to reduce latency

## Installation

### Prerequisites

```bash
# Install PromptChain
pip install -e .

# Install HybridRAG (required dependency)
pip install git+https://github.com/gyasis/hybridrag.git

# Install LiteLLM (for LLM capabilities)
pip install litellm
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here  # Optional
GOOGLE_API_KEY=your_google_key_here        # Optional
```

## Quick Start

### Basic Pattern Usage

```python
from promptchain.integrations.lightrag import (
    LightRAGIntegration,
    LightRAGBranchingThoughts,
    BranchingConfig
)

# Initialize LightRAG integration
integration = LightRAGIntegration(working_dir="./lightrag_data")

# Create a pattern instance
branching = LightRAGBranchingThoughts(
    lightrag_core=integration,
    config=BranchingConfig(hypothesis_count=3)
)

# Execute the pattern
result = await branching.execute(
    problem="What are the main causes of climate change?"
)

print(f"Selected hypothesis: {result.selected_hypothesis.reasoning}")
print(f"Confidence: {result.selected_hypothesis.confidence}")
```

### Multi-Agent Integration

All patterns integrate with PromptChain's MessageBus and Blackboard:

```python
from promptchain.cli.models import MessageBus, Blackboard
from promptchain.integrations.lightrag import LightRAGQueryExpander

# Create infrastructure components
message_bus = MessageBus()
blackboard = Blackboard()

# Create pattern with multi-agent capabilities
expander = LightRAGQueryExpander(
    lightrag_integration=integration,
    config=QueryExpansionConfig(
        emit_events=True,      # Enable event emission
        use_blackboard=True    # Enable state sharing
    )
)

# Connect infrastructure
expander.connect_messagebus(message_bus)
expander.connect_blackboard(blackboard)

# Subscribe to pattern events
def handle_expansion_event(event_type: str, data: dict):
    print(f"Event: {event_type}, Data: {data}")

message_bus.subscribe("pattern.query_expansion.*", handle_expansion_event)

# Execute with event broadcasting and state sharing
result = await expander.execute(query="machine learning algorithms")

# Read shared state
shared_expansions = blackboard.read("expanded_queries")
```

## Pattern Selection Guide

Choose the right pattern for your use case:

### Branching Thoughts
**Best for:**
- Complex problems requiring multiple perspectives
- Situations where you need confidence scoring
- Problems with multiple valid approaches

**Example use cases:**
- Strategic decision making
- Research question analysis
- Comparing alternative solutions

### Query Expansion
**Best for:**
- Improving recall in retrieval tasks
- Handling ambiguous or underspecified queries
- Comprehensive information gathering

**Example use cases:**
- Exploratory search
- Synonym-aware retrieval
- Broadening search scope

### Sharded Retrieval
**Best for:**
- Large-scale distributed knowledge bases
- High-availability retrieval systems
- Queries across multiple data sources

**Example use cases:**
- Multi-tenant systems
- Geographic data distribution
- Fault-tolerant retrieval

### Multi-Hop Retrieval
**Best for:**
- Complex questions requiring step-by-step reasoning
- Questions with dependencies between sub-questions
- Comprehensive question answering

**Example use cases:**
- Research paper analysis
- Technical troubleshooting
- Chain-of-thought reasoning

### Hybrid Search Fusion
**Best for:**
- Maximizing retrieval quality
- Combining entity and concept search
- Balancing precision and recall

**Example use cases:**
- Production search systems
- High-quality QA systems
- Comprehensive knowledge retrieval

### Speculative Execution
**Best for:**
- Low-latency interactive systems
- Predictable query patterns
- Conversational interfaces

**Example use cases:**
- Chat interfaces
- Interactive assistants
- Real-time retrieval systems

## Configuration Overview

All patterns inherit from `BasePattern` and support common configuration options:

```python
from promptchain.patterns.base import PatternConfig

config = PatternConfig(
    pattern_id="custom_id",        # Unique identifier
    enabled=True,                  # Whether pattern is active
    timeout_seconds=30.0,          # Maximum execution time
    emit_events=True,              # Enable MessageBus events
    use_blackboard=False           # Enable Blackboard state sharing
)
```

Pattern-specific configurations extend `PatternConfig`:

```python
from promptchain.integrations.lightrag import BranchingConfig

branching_config = BranchingConfig(
    # Base config
    pattern_id="branching_001",
    timeout_seconds=60.0,

    # Pattern-specific
    hypothesis_count=5,
    generator_model="openai/gpt-4o",
    judge_model="openai/gpt-4o",
    diversity_threshold=0.3
)
```

## Event System

All patterns emit events to the MessageBus for monitoring and coordination:

### Event Types

```python
# Pattern lifecycle events
"pattern.{pattern_name}.started"      # Execution started
"pattern.{pattern_name}.completed"    # Execution completed
"pattern.{pattern_name}.timeout"      # Execution timed out
"pattern.{pattern_name}.error"        # Execution failed

# Pattern-specific events
"pattern.branching.hypothesis_generated"
"pattern.query_expansion.expanded"
"pattern.sharded.shard_queried"
"pattern.multi_hop.hop_completed"
"pattern.hybrid_search.fusing"
"pattern.speculative.predicted"
```

### Event Handling

```python
from promptchain.cli.models import MessageBus

bus = MessageBus()
pattern.connect_messagebus(bus)

# Subscribe to all events from a pattern
bus.subscribe("pattern.branching.*", lambda evt, data: print(data))

# Subscribe to specific events across all patterns
bus.subscribe("*.completed", lambda evt, data: print(f"Pattern completed: {data}"))
```

## Blackboard State Sharing

Patterns can share results via the Blackboard for multi-agent coordination:

```python
from promptchain.cli.models import Blackboard

blackboard = Blackboard()
pattern.connect_blackboard(blackboard)
pattern.config.use_blackboard = True

# Pattern automatically shares results
await pattern.execute(query="test")

# Other agents can read shared state
result = blackboard.read(f"branching.{pattern.config.pattern_id}.selected")
```

## Performance Considerations

### Token Efficiency

Patterns use token-aware history management:

```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

history = ExecutionHistoryManager(
    max_tokens=4000,
    max_entries=50,
    truncation_strategy="oldest_first"
)

# Track pattern executions
history.add_entry("user_input", query, source="user")
result = await pattern.execute(query=query)
history.add_entry("agent_output", result.result, source=pattern.config.pattern_id)
```

### Parallel Execution

Most patterns support parallel execution:

```python
# Query Expansion - parallel search
config = QueryExpansionConfig(parallel_search=True)

# Sharded Retrieval - parallel shard queries
config = ShardedRetrievalConfig(parallel=True)

# Hybrid Search - parallel technique execution
# (Always parallel by design)
```

### Caching

Speculative Execution pattern provides built-in caching:

```python
executor = LightRAGSpeculativeExecutor(
    lightrag_core=integration,
    config=SpeculativeConfig(
        default_ttl=60.0,           # Cache for 60 seconds
        max_concurrent=3            # Max 3 speculative queries
    )
)

# Check cache hit rate
stats = executor.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

## Error Handling

All patterns use consistent error handling:

```python
result = await pattern.execute(query="test")

if not result.success:
    print(f"Pattern failed: {result.errors}")
    print(f"Execution time: {result.execution_time_ms}ms")
else:
    print(f"Success! Result: {result.result}")
```

### Timeout Handling

Patterns respect timeout configurations:

```python
config = PatternConfig(timeout_seconds=5.0)
pattern = SomePattern(config=config)

# Automatically times out after 5 seconds
result = await pattern.execute_with_timeout(query="test")

if "Execution timed out" in result.errors:
    print("Pattern execution exceeded timeout")
```

## Best Practices

### 1. Choose Appropriate Models

Different patterns have different model requirements:

```python
# Branching Thoughts - needs strong reasoning for judging
BranchingConfig(
    generator_model="openai/gpt-4o",      # Query generation
    judge_model="openai/gpt-4o"           # Hypothesis evaluation
)

# Multi-Hop - needs good decomposition
MultiHopConfig(
    synthesizer_model="openai/gpt-4o-mini"  # Faster for synthesis
)
```

### 2. Configure Timeouts Appropriately

Set timeouts based on pattern complexity:

```python
# Simple patterns - short timeout
QueryExpansionConfig(timeout_seconds=10.0)

# Complex patterns - longer timeout
MultiHopConfig(timeout_seconds=60.0)

# Sharded retrieval - per-shard timeout
ShardConfig(shard_id="shard1", timeout_seconds=5.0)
```

### 3. Monitor Events for Debugging

Use event handlers to track pattern execution:

```python
def debug_handler(event_type: str, data: dict):
    import json
    print(f"[DEBUG] {event_type}")
    print(json.dumps(data, indent=2))

pattern.add_event_handler(debug_handler)
```

### 4. Leverage State Sharing for Multi-Agent Systems

Share results between patterns:

```python
# Pattern A shares results
pattern_a.config.use_blackboard = True
result_a = await pattern_a.execute(query="test")

# Pattern B reads shared state
pattern_b.config.use_blackboard = True
shared_data = pattern_b.read_shared(f"pattern_a_result_{pattern_a.config.pattern_id}")
```

### 5. Use Statistics for Optimization

Track pattern performance:

```python
stats = pattern.get_stats()
print(f"Executions: {stats['execution_count']}")
print(f"Avg time: {stats['average_execution_time_ms']:.2f}ms")

# For Speculative Execution
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

## Examples

### Complete Research Workflow

See [examples/patterns/research_workflow.py](../../examples/patterns/research_workflow.py) for a complete example combining multiple patterns:

1. Query Expansion for comprehensive search
2. Sharded Retrieval across multiple knowledge bases
3. Multi-Hop Retrieval for complex reasoning
4. Hybrid Search Fusion for result quality
5. Branching Thoughts for hypothesis evaluation

### Quick Start Example

See [examples/patterns/quick_start.py](../../examples/patterns/quick_start.py) for a minimal getting started example.

## API Reference

### Base Pattern Interface

All patterns implement:

```python
class BasePattern(ABC):
    async def execute(self, **kwargs) -> PatternResult
    async def execute_with_timeout(self, **kwargs) -> PatternResult

    # Infrastructure integration
    def connect_messagebus(self, bus: MessageBus) -> None
    def connect_blackboard(self, blackboard: Blackboard) -> None

    # Event system
    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None
    def add_event_handler(self, handler: Callable) -> None

    # State sharing
    def share_result(self, key: str, value: Any) -> None
    def read_shared(self, key: str) -> Optional[Any]

    # Statistics
    def get_stats(self) -> Dict[str, Any]
```

### Pattern Results

All patterns return `PatternResult` (or subclass):

```python
@dataclass
class PatternResult:
    pattern_id: str                    # Pattern instance ID
    success: bool                      # Whether execution succeeded
    result: Any                        # Pattern-specific result
    execution_time_ms: float           # Execution time
    metadata: Dict[str, Any]           # Additional metadata
    errors: List[str]                  # Error messages if any
    timestamp: datetime                # Execution timestamp
```

## Further Reading

- [Branching Thoughts Pattern](./branching.md)
- [Query Expansion Pattern](./query_expansion.md)
- [Sharded Retrieval Pattern](./sharded.md)
- [Multi-Hop Retrieval Pattern](./multi_hop.md)
- [Hybrid Search Fusion Pattern](./hybrid_search.md)
- [Speculative Execution Pattern](./speculative.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/gyasis/PromptChain
- HybridRAG: https://github.com/gyasis/hybridrag

## License

Same as PromptChain project license.
