# LightRAG Branching Thoughts - Quick Reference

## Import

```python
from promptchain.integrations.lightrag import LightRAGIntegration, LightRAGBranchingThoughts
from promptchain.integrations.lightrag.branching import BranchingConfig
```

## Basic Usage

```python
# Setup
integration = LightRAGIntegration()
config = BranchingConfig(hypothesis_count=3)
branching = LightRAGBranchingThoughts(lightrag_core=integration, config=config)

# Execute
result = await branching.execute(problem="Your question here")

# Access results
selected = result.selected_hypothesis
print(f"Mode: {selected.mode}")
print(f"Reasoning: {selected.reasoning}")
print(f"Score: {max(s.score for s in result.scores):.2f}")
```

## Configuration

```python
BranchingConfig(
    hypothesis_count=3,           # Number of hypotheses to generate
    judge_model="openai/gpt-4o",  # Model for judging
    diversity_threshold=0.3,      # Min diversity (0.0-1.0)
    timeout_seconds=30.0,         # Max execution time
    emit_events=True,             # Emit events to MessageBus
    use_blackboard=False          # Use Blackboard for sharing
)
```

## Query Modes

| Mode | Focus | Best For |
|------|-------|----------|
| **local** | Entity-specific details | Questions about specific entities/facts |
| **global** | High-level conceptual | Broad understanding, concepts |
| **hybrid** | Balanced combination | Complex questions needing both |

## Result Structure

```python
result = BranchingResult(
    hypotheses=[Hypothesis, ...],      # All generated hypotheses
    scores=[HypothesisScore, ...],     # Evaluation scores
    selected_hypothesis=Hypothesis,     # Best hypothesis
    selection_reasoning=str,            # Why it was selected
    success=bool,                       # Execution success
    execution_time_ms=float            # Time taken
)
```

## Events

| Event | When | Data |
|-------|------|------|
| `pattern.branching.started` | Execution begins | `problem`, `hypothesis_count` |
| `pattern.branching.hypothesis_generated` | Hypothesis created | `hypothesis_id`, `mode`, `confidence` |
| `pattern.branching.judging` | Judging starts | `hypothesis_count` |
| `pattern.branching.selected` | Best selected | `hypothesis_id`, `mode`, `score` |
| `pattern.branching.completed` | Execution done | `selected_hypothesis_id`, `execution_time_ms` |

## With Event Handling

```python
def event_handler(event_type: str, data: dict):
    if "hypothesis_generated" in event_type:
        print(f"Generated: {data['mode']}")

branching.add_event_handler(event_handler)
result = await branching.execute(problem="Your question")
```

## With MessageBus

```python
from promptchain.cli.models import MessageBus

bus = MessageBus()
branching.connect_messagebus(bus)
bus.subscribe("pattern.branching.*", lambda e, d: print(e))
result = await branching.execute(problem="Your question")
```

## With Blackboard

```python
from promptchain.cli.models import Blackboard

blackboard = Blackboard()
config = BranchingConfig(use_blackboard=True)
branching = LightRAGBranchingThoughts(integration, config)
branching.connect_blackboard(blackboard)

result = await branching.execute(problem="Your question")

# Read shared result
shared = blackboard.read(f"branching.{branching.config.pattern_id}.selected")
```

## Performance Tips

### Token Optimization
```python
# Use cheaper models
config = BranchingConfig(
    generator_model="openai/gpt-4o-mini",
    judge_model="openai/gpt-4o-mini",
    hypothesis_count=3
)
```

### Speed Optimization
```python
# Reduce hypothesis count
config = BranchingConfig(hypothesis_count=2)

# Shorter timeout
config = BranchingConfig(timeout_seconds=15.0)
```

### Quality Optimization
```python
# More hypotheses, better judge
config = BranchingConfig(
    hypothesis_count=6,
    judge_model="openai/gpt-4o",
    diversity_threshold=0.5
)
```

## Error Handling

```python
result = await branching.execute(problem="Your question")

if not result.success:
    print("Pattern failed:")
    for error in result.errors:
        print(f"  - {error}")
elif not result.selected_hypothesis:
    print("No hypothesis could be selected")
else:
    print(f"Success: {result.selected_hypothesis.reasoning}")
```

## Integration Examples

### With Multi-Hop
```python
from promptchain.integrations.lightrag import LightRAGMultiHop

# Get initial hypothesis
branching_result = await branching.execute(problem="Your question")

# Refine with multi-hop
multi_hop = LightRAGMultiHop(lightrag_core=integration)
refined = await multi_hop.execute(
    query=branching_result.selected_hypothesis.reasoning,
    max_hops=3
)
```

### With Speculative Execution
```python
from promptchain.integrations.lightrag import LightRAGSpeculativeExecutor

# Generate hypotheses
branching_result = await branching.execute(problem="Your question")

# Use as speculative candidates
speculative = LightRAGSpeculativeExecutor(lightrag_core=integration)
hypotheses_text = [h.reasoning for h in branching_result.hypotheses]
result = await speculative.execute(candidates=hypotheses_text)
```

## Common Patterns

### Quick Decision
```python
config = BranchingConfig(
    hypothesis_count=3,
    judge_model="openai/gpt-4o-mini",
    timeout_seconds=10.0
)
```

### Thorough Analysis
```python
config = BranchingConfig(
    hypothesis_count=9,
    judge_model="openai/gpt-4o",
    diversity_threshold=0.5,
    timeout_seconds=60.0
)
```

### Cost-Conscious
```python
config = BranchingConfig(
    hypothesis_count=2,
    judge_model="openai/gpt-4o-mini",
    record_outcomes=False
)
```

## Troubleshooting

### Empty Hypotheses
- Verify LightRAG has documents inserted
- Check API keys configured
- Increase timeout if queries slow

### Judge Parse Errors
- Pattern falls back to default scores
- Use more structured model (GPT-4)
- Check judge prompt clarity

### Low Diversity
- Increase `diversity_threshold`
- Ensure diverse document sources
- Use different query modes

## See Also

- [Full Documentation](./lightrag_branching_thoughts.md)
- [Base Pattern](./base_pattern.md)
- [LightRAG Integration](./lightrag_integration.md)
- [Example Code](../../examples/lightrag_branching_demo.py)
