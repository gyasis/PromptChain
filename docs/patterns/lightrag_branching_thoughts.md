# LightRAG Branching Thoughts Pattern

## Overview

The Branching Thoughts pattern generates multiple hypotheses using LightRAG's different query modes (local, global, hybrid) and uses an LLM judge to evaluate and select the best one. This pattern explores diverse reasoning paths in parallel and systematically evaluates them for quality.

## Pattern Type

**Branching/Divergent Reasoning** - Generates multiple solution hypotheses and selects the best one.

## How It Works

1. **Hypothesis Generation**: Query LightRAG using different modes to generate diverse hypotheses:
   - **Local mode**: Entity-specific hypothesis focusing on concrete details
   - **Global mode**: High-level conceptual hypothesis
   - **Hybrid mode**: Balanced combination of entity details and concepts

2. **Parallel Execution**: All query modes run concurrently using `asyncio.gather` for efficiency

3. **LLM Judging**: An LLM evaluates each hypothesis based on:
   - Completeness of answer
   - Relevance to the problem
   - Internal consistency
   - Supporting evidence quality

4. **Selection**: The highest-scored hypothesis is selected and returned with detailed reasoning

## Architecture

```
Problem Input
     |
     v
+----+----+----+
|    |    |    |
v    v    v    v
Local Global Hybrid
Query Query  Query
|    |    |    |
+----+----+----+
     |
     v
Extract Hypotheses
     |
     v
LLM Judge Evaluation
     |
     v
Score & Select Best
     |
     v
Selected Hypothesis
```

## Usage

### Basic Usage

```python
from promptchain.integrations.lightrag import LightRAGIntegration, LightRAGBranchingThoughts
from promptchain.integrations.lightrag.branching import BranchingConfig

# Create LightRAG integration
integration = LightRAGIntegration()

# Configure the pattern
config = BranchingConfig(
    hypothesis_count=3,
    judge_model="openai/gpt-4o",
    diversity_threshold=0.3,
    emit_events=True,
)

# Create the pattern
branching = LightRAGBranchingThoughts(
    lightrag_core=integration,
    config=config
)

# Execute
result = await branching.execute(
    problem="What are the key factors in climate change?"
)

# Access results
print(f"Selected hypothesis: {result.selected_hypothesis.reasoning}")
print(f"Mode used: {result.selected_hypothesis.mode}")
print(f"Selection reasoning: {result.selection_reasoning}")
```

### With Event Monitoring

```python
# Add event handler to monitor progress
def event_handler(event_type: str, data: dict):
    if event_type == "pattern.branching.hypothesis_generated":
        print(f"Generated hypothesis using {data['mode']} mode")
    elif event_type == "pattern.branching.judging":
        print(f"Judging {data['hypothesis_count']} hypotheses")
    elif event_type == "pattern.branching.selected":
        print(f"Selected hypothesis with score {data['score']:.2f}")

branching.add_event_handler(event_handler)
result = await branching.execute(problem="Your problem here")
```

### With MessageBus Integration

```python
from promptchain.cli.models import MessageBus

# Create MessageBus
bus = MessageBus()

# Connect pattern to bus
branching.connect_messagebus(bus)

# Subscribe to events
def on_hypothesis(event_type: str, data: dict):
    print(f"New hypothesis: {data}")

bus.subscribe("pattern.branching.hypothesis_generated", on_hypothesis)

# Execute
result = await branching.execute(problem="Your problem here")
```

### With Blackboard State Sharing

```python
from promptchain.cli.models import Blackboard

# Create Blackboard
blackboard = Blackboard()

# Enable Blackboard in config
config = BranchingConfig(use_blackboard=True)
branching = LightRAGBranchingThoughts(
    lightrag_core=integration,
    config=config
)

# Connect to Blackboard
branching.connect_blackboard(blackboard)

# Execute - results will be shared via Blackboard
result = await branching.execute(problem="Your problem here")

# Other patterns can read the shared result
selected = blackboard.read(f"branching.{branching.config.pattern_id}.selected")
```

## Configuration

### BranchingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hypothesis_count` | int | 3 | Number of hypotheses to generate |
| `generator_model` | str | None | Model for hypothesis generation (uses LightRAG default if None) |
| `judge_model` | str | "openai/gpt-4o" | Model to use for judging hypotheses |
| `diversity_threshold` | float | 0.3 | Minimum diversity between hypotheses (0.0-1.0) |
| `record_outcomes` | bool | True | Whether to record outcomes for future learning |
| `pattern_id` | str | auto | Unique identifier for this pattern instance |
| `enabled` | bool | True | Whether the pattern is active |
| `timeout_seconds` | float | 30.0 | Maximum execution time before timeout |
| `emit_events` | bool | True | Whether to emit events to MessageBus |
| `use_blackboard` | bool | False | Whether to use Blackboard for state sharing |

## Result Structure

### BranchingResult

```python
@dataclass
class BranchingResult(PatternResult):
    hypotheses: List[Hypothesis]           # All generated hypotheses
    scores: List[HypothesisScore]          # Evaluation scores
    selected_hypothesis: Optional[Hypothesis]  # Best hypothesis
    selection_reasoning: str                # Why it was selected
```

### Hypothesis

```python
@dataclass
class Hypothesis:
    hypothesis_id: str      # Unique ID
    approach: str           # Reasoning approach description
    reasoning: str          # Detailed reasoning
    confidence: float       # Confidence score (0.0-1.0)
    mode: str              # Query mode ("local", "global", "hybrid")
```

### HypothesisScore

```python
@dataclass
class HypothesisScore:
    hypothesis_id: str      # ID of scored hypothesis
    score: float            # Numerical score (0.0-1.0)
    reasoning: str          # Score explanation
    strengths: List[str]    # Identified strengths
    weaknesses: List[str]   # Identified weaknesses
```

## Events Emitted

| Event | Data | Description |
|-------|------|-------------|
| `pattern.branching.started` | `problem`, `hypothesis_count` | Pattern execution started |
| `pattern.branching.generating` | `modes`, `count` | Beginning hypothesis generation |
| `pattern.branching.hypothesis_generated` | `hypothesis_id`, `mode`, `confidence` | Single hypothesis generated |
| `pattern.branching.judging` | `hypothesis_count` | Starting LLM judging phase |
| `pattern.branching.selected` | `hypothesis_id`, `mode`, `score` | Best hypothesis selected |
| `pattern.branching.completed` | `selected_hypothesis_id`, `execution_time_ms` | Pattern execution completed |
| `pattern.branching.error` | `error`, `elapsed_ms` | Execution error occurred |
| `pattern.branching.timeout` | `elapsed_ms` | Execution timed out |

## Query Mode Characteristics

### Local Mode
- **Focus**: Entity-specific details
- **Best For**: Questions about specific entities, facts, relationships
- **Example**: "What products does Company X manufacture?"

### Global Mode
- **Focus**: High-level conceptual overview
- **Best For**: Broad understanding, conceptual questions
- **Example**: "What are the main trends in renewable energy?"

### Hybrid Mode
- **Focus**: Balanced combination
- **Best For**: Complex questions requiring both details and overview
- **Example**: "How is AI transforming healthcare delivery?"

## Best Practices

### Choosing Hypothesis Count

```python
# For quick decisions: 3 hypotheses (one per mode)
config = BranchingConfig(hypothesis_count=3)

# For thorough analysis: 6+ hypotheses (multiple per mode)
config = BranchingConfig(hypothesis_count=6)

# For critical decisions: 9+ hypotheses with strict judging
config = BranchingConfig(
    hypothesis_count=9,
    judge_model="openai/gpt-4o",
    diversity_threshold=0.5
)
```

### Selecting Judge Models

```python
# Fast judging (less accurate)
config = BranchingConfig(judge_model="openai/gpt-4o-mini")

# Balanced judging (recommended)
config = BranchingConfig(judge_model="openai/gpt-4o")

# High-quality judging (slower, more expensive)
config = BranchingConfig(judge_model="anthropic/claude-3-opus-20240229")
```

### Error Handling

```python
result = await branching.execute(problem="Your problem here")

if result.success:
    print(f"Selected: {result.selected_hypothesis.reasoning}")
else:
    print("Pattern failed:")
    for error in result.errors:
        print(f"  - {error}")
```

### Timeout Configuration

```python
# For complex problems requiring more time
config = BranchingConfig(timeout_seconds=60.0)

# For time-critical applications
config = BranchingConfig(timeout_seconds=10.0)
```

## Advanced Usage

### Custom Hypothesis Extraction

```python
class CustomBranching(LightRAGBranchingThoughts):
    def _extract_hypothesis(self, mode: str, result: Any, index: int) -> Hypothesis:
        # Custom logic to extract hypothesis from result
        hypothesis = super()._extract_hypothesis(mode, result, index)

        # Add custom processing
        hypothesis.confidence = self._calculate_custom_confidence(result)

        return hypothesis
```

### Recording Outcomes for Learning

```python
# Enable outcome recording
config = BranchingConfig(record_outcomes=True)

# Access recorded outcomes
result = await branching.execute(problem="Your problem")

# Store for future analysis
outcomes.append({
    "problem": problem,
    "selected_mode": result.selected_hypothesis.mode,
    "score": max(s.score for s in result.scores),
    "timestamp": result.timestamp,
})
```

## Performance Considerations

### Token Usage

- **Hypothesis Generation**: ~500-2000 tokens per hypothesis (depends on LightRAG data)
- **Judge Prompt**: ~1000-3000 tokens (depends on hypothesis length)
- **Total**: Approximately (hypothesis_count * 1500) + 2000 tokens

### Execution Time

- **Hypothesis Generation**: 2-5 seconds (parallel execution)
- **LLM Judging**: 3-10 seconds (depends on judge model)
- **Total**: Typically 5-15 seconds for 3 hypotheses

### Cost Optimization

```python
# Use cheaper models for generation
config = BranchingConfig(
    generator_model="openai/gpt-4o-mini",
    judge_model="openai/gpt-4o-mini",
    hypothesis_count=3,
)

# Use expensive models only for judging
config = BranchingConfig(
    generator_model=None,  # Use LightRAG default
    judge_model="openai/gpt-4o",  # High-quality judging
    hypothesis_count=3,
)
```

## Integration with Other Patterns

### With Multi-Hop Pattern

```python
from promptchain.integrations.lightrag import LightRAGMultiHop

# Use branching to generate initial hypothesis
branching_result = await branching.execute(problem="Your problem")

# Use multi-hop to refine the selected hypothesis
multi_hop = LightRAGMultiHop(lightrag_core=integration)
refined_result = await multi_hop.execute(
    query=branching_result.selected_hypothesis.reasoning,
    max_hops=3
)
```

### With Speculative Execution

```python
from promptchain.integrations.lightrag import LightRAGSpeculativeExecutor

# Use branching hypotheses as speculative candidates
speculative = LightRAGSpeculativeExecutor(lightrag_core=integration)

branching_result = await branching.execute(problem="Your problem")
hypotheses_text = [h.reasoning for h in branching_result.hypotheses]

speculative_result = await speculative.execute(
    candidates=hypotheses_text,
    verification_strategy="parallel"
)
```

## Troubleshooting

### No Hypotheses Generated

**Problem**: Pattern returns empty hypothesis list

**Solutions**:
- Verify LightRAG has documents inserted
- Check API keys are configured
- Increase timeout if queries are slow
- Verify network connectivity

### Judge Fails to Parse Output

**Problem**: Judge output cannot be parsed into scores

**Solutions**:
- Use more structured judge model (GPT-4 over GPT-3.5)
- Increase judge temperature for more consistent formatting
- Check judge prompt for clarity
- Pattern falls back to default confidence scores

### Low Diversity in Hypotheses

**Problem**: All hypotheses are very similar

**Solutions**:
- Increase `diversity_threshold` in config
- Ensure LightRAG has diverse document sources
- Use different query modes explicitly
- Consider query reformulation

## References

- [Base Pattern Documentation](./base_pattern.md)
- [LightRAG Integration](./lightrag_integration.md)
- [MessageBus Documentation](../003-multi-agent-communication/message_bus.md)
- [Blackboard Documentation](../003-multi-agent-communication/blackboard.md)

## Example Projects

See `/home/gyasis/Documents/code/PromptChain/examples/lightrag_branching_demo.py` for a complete working example.
