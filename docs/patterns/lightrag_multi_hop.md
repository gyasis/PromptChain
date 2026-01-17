# LightRAG Multi-Hop Retrieval Pattern

## Overview

The Multi-Hop Retrieval Pattern implements complex question answering through iterative retrieval and reasoning. It decomposes complex questions into sub-questions and uses LightRAG's agentic search capabilities to perform multi-hop reasoning across a knowledge graph.

## Key Features

- **Question Decomposition**: Automatically breaks down complex questions into answerable sub-questions
- **Multi-Hop Reasoning**: Executes multiple retrieval steps to build comprehensive answers
- **Answer Synthesis**: Combines sub-answers into coherent unified responses
- **Progress Tracking**: Emits events at each stage of execution
- **State Sharing**: Integrates with Blackboard for cross-agent collaboration
- **Error Handling**: Gracefully handles retrieval failures and partial results

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  LightRAGMultiHop Pattern                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Question Decomposition (optional)                       │
│     └─> Uses LiteLLM to generate sub-questions             │
│                                                             │
│  2. Agentic Search Execution                                │
│     └─> Calls SearchInterface.agentic_search()             │
│     └─> Tracks hops/steps executed                         │
│                                                             │
│  3. Answer Synthesis                                        │
│     └─> Uses LiteLLM to combine sub-answers                │
│     └─> Identifies unanswered aspects                      │
│                                                             │
│  4. Result Packaging                                        │
│     └─> Creates MultiHopResult with all metadata           │
│     └─> Shares to Blackboard if enabled                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Usage

```python
from promptchain.integrations.lightrag.core import LightRAGIntegration
from promptchain.integrations.lightrag.multi_hop import (
    LightRAGMultiHop,
    MultiHopConfig,
)

# Initialize LightRAG
integration = LightRAGIntegration(
    config=LightRAGConfig(working_dir="./lightrag_data")
)

# Create pattern
pattern = LightRAGMultiHop(
    search_interface=integration.search,
    config=MultiHopConfig(
        max_hops=5,
        decompose_first=True,
    ),
)

# Execute
result = await pattern.execute(
    question="What are the key differences between transformers and RNNs?"
)

print(result.unified_answer)
print(f"Hops: {result.hops_executed}")
```

### With Question Decomposition

```python
config = MultiHopConfig(
    max_hops=7,
    max_sub_questions=5,
    decompose_first=True,  # Enable decomposition
    synthesizer_model="openai/gpt-4o-mini",
)

pattern = LightRAGMultiHop(
    search_interface=integration.search,
    config=config,
)

result = await pattern.execute(
    question="How has attention mechanism evolved from seq2seq to transformers?"
)

# Access sub-questions
for sq in result.sub_questions:
    print(f"Q: {sq.question_text}")
    print(f"A: {sq.answer}")
    print(f"Dependencies: {sq.dependencies}")
```

### Event Tracking

```python
def event_handler(event_type: str, data: dict):
    if event_type == "pattern.multi_hop.decomposed":
        print(f"Generated {data['num_sub_questions']} sub-questions")
    elif event_type == "pattern.multi_hop.hop_completed":
        print(f"Executed {data['hops_executed']} hops")

pattern.add_event_handler(event_handler)
result = await pattern.execute(question="...")
```

### Blackboard Integration

```python
from promptchain.cli.tools.library.blackboard_tools import (
    set_session_manager,
    get_session_manager,
)

# Create pattern with Blackboard
config = MultiHopConfig(use_blackboard=True)
pattern = LightRAGMultiHop(
    search_interface=integration.search,
    config=config,
)

# Connect to Blackboard (usually done by session manager)
pattern.connect_blackboard(blackboard)

# Results are automatically shared
result = await pattern.execute(question="...")
# Read from Blackboard: multi_hop_result_{pattern_id}
```

## Configuration

### MultiHopConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_hops` | int | 5 | Maximum retrieval hops to execute |
| `max_sub_questions` | int | 5 | Maximum sub-questions to generate |
| `synthesizer_model` | str | None | Model for answer synthesis (uses default if None) |
| `decompose_first` | bool | True | Whether to decompose question before retrieval |
| `emit_events` | bool | True | Enable event emission |
| `use_blackboard` | bool | False | Enable Blackboard state sharing |
| `timeout_seconds` | float | 30.0 | Maximum execution time |

## Data Models

### SubQuestion

Represents a decomposed sub-question:

```python
@dataclass
class SubQuestion:
    question_id: str              # Unique identifier (e.g., "sq1")
    question_text: str            # The actual question
    parent_question: str          # Original complex question
    dependencies: List[str]       # IDs of prerequisite sub-questions
    rationale: str                # Why this sub-question is needed
    answer: Optional[str]         # Answer (populated after retrieval)
    retrieval_context: List[str]  # Retrieved context chunks
```

### MultiHopResult

Execution result with comprehensive metadata:

```python
@dataclass
class MultiHopResult(PatternResult):
    original_question: str              # Input question
    sub_questions: List[SubQuestion]    # Generated sub-questions
    sub_answers: Dict[str, str]         # question_id -> answer mapping
    unified_answer: str                 # Synthesized final answer
    hops_executed: int                  # Number of retrieval hops
    unanswered_aspects: List[str]       # Questions without answers
```

## Events

The pattern emits the following events:

| Event Type | Data | Description |
|------------|------|-------------|
| `pattern.multi_hop.started` | `question`, `decompose_first` | Execution started |
| `pattern.multi_hop.decomposed` | `num_sub_questions`, `sub_questions` | Question decomposed |
| `pattern.multi_hop.hop_completed` | `hops_executed`, `num_sub_answers` | Hop completed |
| `pattern.multi_hop.synthesizing` | `num_sub_answers` | Starting synthesis |
| `pattern.multi_hop.completed` | `hops_executed`, `num_unanswered`, `execution_time_ms` | Execution finished |

## Integration with 003-Multi-Agent Communication

The pattern integrates with PromptChain's 003 infrastructure:

### MessageBus Integration

```python
from promptchain.cli.models.message import MessageBus

bus = MessageBus()
pattern.connect_messagebus(bus)

# Events are published to MessageBus
bus.subscribe("pattern.multi_hop.*", my_handler)
```

### Blackboard Integration

```python
from promptchain.cli.models.blackboard import BlackboardEntry

# Pattern automatically shares results
pattern.config.use_blackboard = True
pattern.connect_blackboard(blackboard)

# Results accessible by other agents
result = blackboard.read(f"multi_hop_result_{pattern_id}")
```

## Performance Considerations

### Token Efficiency

- **Decomposition Overhead**: ~500-1000 tokens for question decomposition
- **Synthesis Overhead**: ~300-800 tokens per synthesis call
- **Per-Hop Cost**: Varies based on LightRAG configuration

**Optimization Tips**:
- Set `decompose_first=False` for simple questions
- Reduce `max_sub_questions` for faster execution
- Use `max_hops` to control depth vs. speed trade-off

### Execution Time

Typical execution times:
- Without decomposition: 2-5 seconds (3-5 hops)
- With decomposition: 5-10 seconds (includes LLM calls)
- Complex questions: 10-20 seconds (7+ hops)

## Error Handling

The pattern handles errors gracefully:

```python
result = await pattern.execute(question="...")

if not result.success:
    print(f"Errors: {result.errors}")
else:
    if result.unanswered_aspects:
        print(f"Partial results. Unanswered: {result.unanswered_aspects}")
    else:
        print(f"Complete answer: {result.unified_answer}")
```

## Examples

See `/home/gyasis/Documents/code/PromptChain/examples/lightrag_multi_hop_example.py` for comprehensive examples including:
- Basic multi-hop retrieval
- Question decomposition
- Blackboard integration
- Custom event tracking

## Testing

Run tests with:

```bash
pytest tests/test_multi_hop_pattern.py -v
```

Test coverage includes:
- Pattern initialization
- Question decomposition
- Agentic search execution
- Answer synthesis
- Event emission
- Blackboard integration
- Error handling

## Dependencies

Required:
- `promptchain.patterns.base`: Base pattern infrastructure
- `hybridrag`: LightRAG implementation (SearchInterface)
- `litellm`: LLM calls for decomposition and synthesis

Optional:
- `promptchain.cli.models`: MessageBus and Blackboard (003 infrastructure)

## Future Enhancements

Potential improvements:
1. **Parallel Sub-Question Execution**: Execute independent sub-questions concurrently
2. **Adaptive Hop Limit**: Dynamically adjust max_hops based on question complexity
3. **Context Pruning**: Intelligently filter retrieval context to reduce token usage
4. **Sub-Answer Extraction**: Parse synthesized answers to extract individual sub-answers
5. **Dependency-Aware Execution**: Execute sub-questions respecting dependency graph
6. **Caching**: Cache sub-question answers for reuse across similar questions

## See Also

- [BasePattern Documentation](./base_pattern.md)
- [LightRAG Integration](./lightrag_integration.md)
- [003-Multi-Agent Communication](../003-multi-agent-communication.md)
- [Agentic Patterns Overview](./agentic_patterns.md)
