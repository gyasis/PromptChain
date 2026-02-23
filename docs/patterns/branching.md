# Branching Thoughts Pattern

Generate and evaluate multiple hypotheses using different LightRAG query modes (local, global, hybrid) and select the best one using an LLM judge.

## Overview

The Branching Thoughts pattern implements a sophisticated reasoning approach where:
1. Multiple hypotheses are generated in parallel using different query modes
2. Each hypothesis explores a different reasoning path
3. An LLM judge evaluates all hypotheses based on completeness, relevance, consistency, and evidence
4. The best hypothesis is selected and returned with confidence scoring

This pattern is particularly effective for complex problems that benefit from exploring multiple perspectives before converging on a solution.

## Use Cases

- **Strategic Decision Making**: Evaluate multiple strategic options before choosing
- **Research Question Analysis**: Explore different angles of a research question
- **Problem Solving**: Generate and compare alternative solutions
- **Comparative Analysis**: Assess multiple competing explanations
- **Risk Assessment**: Evaluate scenarios from different perspectives

## Installation

```bash
# Install HybridRAG (required)
pip install git+https://github.com/gyasis/hybridrag.git

# Install LiteLLM (required for LLM judge)
pip install litellm
```

## Basic Usage

```python
from promptchain.integrations.lightrag import (
    LightRAGIntegration,
    LightRAGBranchingThoughts,
    BranchingConfig
)

# Initialize LightRAG
integration = LightRAGIntegration(working_dir="./lightrag_data")

# Create pattern instance
branching = LightRAGBranchingThoughts(
    lightrag_core=integration,
    config=BranchingConfig(
        hypothesis_count=3,
        judge_model="openai/gpt-4o"
    )
)

# Execute pattern
result = await branching.execute(
    problem="What are the key factors driving renewable energy adoption?"
)

# Access results
print(f"Selected Hypothesis: {result.selected_hypothesis.reasoning}")
print(f"Approach: {result.selected_hypothesis.approach}")
print(f"Confidence: {result.selected_hypothesis.confidence}")
print(f"Mode: {result.selected_hypothesis.mode}")

# Review all hypotheses
for i, hypothesis in enumerate(result.hypotheses, 1):
    print(f"\nHypothesis {i} ({hypothesis.mode}):")
    print(f"  Approach: {hypothesis.approach}")
    print(f"  Reasoning: {hypothesis.reasoning[:100]}...")

# Review scoring
for score in result.scores:
    print(f"\nHypothesis {score.hypothesis_id}:")
    print(f"  Score: {score.score}")
    print(f"  Strengths: {score.strengths}")
    print(f"  Weaknesses: {score.weaknesses}")
```

## Configuration

### BranchingConfig Options

```python
from promptchain.integrations.lightrag import BranchingConfig

config = BranchingConfig(
    # Base pattern config
    pattern_id="branching_001",        # Unique identifier
    enabled=True,                      # Enable/disable pattern
    timeout_seconds=60.0,              # Max execution time
    emit_events=True,                  # Enable event emission
    use_blackboard=False,              # Enable state sharing

    # Branching-specific config
    hypothesis_count=3,                # Number of hypotheses to generate
    generator_model=None,              # Model for hypothesis generation (uses LightRAG default)
    judge_model="openai/gpt-4o",      # Model for judging hypotheses
    diversity_threshold=0.3,           # Min diversity between hypotheses (0.0-1.0)
    record_outcomes=True               # Track outcomes for learning
)
```

### Configuration Guidelines

**hypothesis_count**:
- **3** (default): Good balance of coverage and cost
- **5-7**: More comprehensive exploration, higher cost
- **1-2**: Fast evaluation for simple problems

**judge_model**:
- **gpt-4o**: Best quality judging, recommended for production
- **gpt-4o-mini**: Faster, lower cost, good for development
- **claude-3-opus**: Alternative high-quality option

**diversity_threshold**:
- **0.3** (default): Moderate diversity requirement
- **0.5-0.7**: Enforce high diversity between hypotheses
- **0.0-0.2**: Allow similar hypotheses

## Query Modes

The pattern uses different LightRAG query modes to generate diverse hypotheses:

### Local Query Mode
- **Focus**: Entity-specific, concrete details
- **Best for**: Questions about specific entities, relationships, or facts
- **Example**: "What are Tesla's key technological innovations?"

### Global Query Mode
- **Focus**: High-level concepts, patterns, themes
- **Best for**: Conceptual questions, big-picture analysis
- **Example**: "What are the major trends in renewable energy?"

### Hybrid Query Mode
- **Focus**: Balanced combination of entities and concepts
- **Best for**: Complex questions requiring both specific details and conceptual understanding
- **Example**: "How do Tesla's innovations contribute to renewable energy trends?"

## Result Structure

### BranchingResult

```python
@dataclass
class BranchingResult(PatternResult):
    hypotheses: List[Hypothesis]                  # All generated hypotheses
    scores: List[HypothesisScore]                 # Judge scores for each
    selected_hypothesis: Optional[Hypothesis]     # Best hypothesis
    selection_reasoning: str                      # Why it was selected

    # Inherited from PatternResult
    pattern_id: str
    success: bool
    result: Any                                   # Selected hypothesis reasoning
    execution_time_ms: float
    metadata: Dict[str, Any]
    errors: List[str]
    timestamp: datetime
```

### Hypothesis

```python
@dataclass
class Hypothesis:
    hypothesis_id: str        # Unique identifier
    approach: str             # Description of reasoning approach
    reasoning: str            # Detailed reasoning/answer
    confidence: float         # Initial confidence (0.0-1.0)
    mode: str                 # Query mode used ("local", "global", "hybrid")
```

### HypothesisScore

```python
@dataclass
class HypothesisScore:
    hypothesis_id: str        # ID of scored hypothesis
    score: float              # Judge score (0.0-1.0)
    reasoning: str            # Explanation of score
    strengths: List[str]      # Identified strengths
    weaknesses: List[str]     # Identified weaknesses
```

## Events

The pattern emits the following events:

```python
# Lifecycle events
"pattern.branching.started"              # Execution started
"pattern.branching.completed"            # Execution completed
"pattern.branching.timeout"              # Execution timed out
"pattern.branching.error"                # Execution failed

# Process events
"pattern.branching.generating"           # Hypothesis generation started
"pattern.branching.hypothesis_generated" # Single hypothesis generated
"pattern.branching.judging"              # Judging started
"pattern.branching.selected"             # Best hypothesis selected
```

### Event Handling Example

```python
from promptchain.cli.models import MessageBus

bus = MessageBus()
branching.connect_messagebus(bus)

# Track hypothesis generation
def on_hypothesis_generated(event_type: str, data: dict):
    print(f"Generated hypothesis using {data['mode']} mode")
    print(f"Confidence: {data['confidence']}")

bus.subscribe("pattern.branching.hypothesis_generated", on_hypothesis_generated)

# Track selection
def on_selected(event_type: str, data: dict):
    print(f"Selected hypothesis from {data['mode']} mode")
    print(f"Score: {data['score']}")

bus.subscribe("pattern.branching.selected", on_selected)
```

## Advanced Usage

### Custom Judge Prompt

The pattern uses a sophisticated judge prompt by default. You can customize evaluation criteria by subclassing:

```python
class CustomBranchingThoughts(LightRAGBranchingThoughts):
    def _create_judge_prompt(self, problem: str, hypotheses: List[Hypothesis]) -> str:
        # Custom prompt emphasizing specific criteria
        prompt = f"""Evaluate hypotheses for: {problem}

Evaluation Criteria:
1. Technical Accuracy (40%)
2. Practical Applicability (30%)
3. Innovation (20%)
4. Clarity (10%)

HYPOTHESES:
"""
        for h in hypotheses:
            prompt += f"\n{h.hypothesis_id}: {h.reasoning}\n"

        prompt += "\nProvide JSON scores for each hypothesis."
        return prompt
```

### Multi-Agent Coordination

Share branching results via Blackboard:

```python
from promptchain.cli.models import Blackboard

blackboard = Blackboard()
branching.connect_blackboard(blackboard)
branching.config.use_blackboard = True

# Execute and auto-share
result = await branching.execute(problem="Complex problem")

# Another agent reads the result
shared_result = blackboard.read(f"branching.{branching.config.pattern_id}.selected")
print(f"Shared hypothesis: {shared_result['hypothesis']['reasoning']}")
```

### Parallel Branching for Multiple Problems

```python
import asyncio

problems = [
    "What drives renewable energy adoption?",
    "How can we reduce carbon emissions?",
    "What are the economic impacts of climate change?"
]

# Create pattern instances
branchings = [
    LightRAGBranchingThoughts(integration, config=BranchingConfig(pattern_id=f"branch_{i}"))
    for i in range(len(problems))
]

# Execute in parallel
results = await asyncio.gather(*[
    branching.execute(problem=problem)
    for branching, problem in zip(branchings, problems)
])

# Compare selected hypotheses
for problem, result in zip(problems, results):
    print(f"\nProblem: {problem}")
    print(f"Selected: {result.selected_hypothesis.reasoning[:100]}...")
```

## Best Practices

### 1. Choose Appropriate Hypothesis Count

```python
# Simple problems - fewer hypotheses
BranchingConfig(hypothesis_count=2)

# Complex multi-faceted problems - more hypotheses
BranchingConfig(hypothesis_count=5)

# Research or critical decisions - comprehensive exploration
BranchingConfig(hypothesis_count=7)
```

### 2. Use Strong Judge Model

The judge quality directly impacts selection accuracy:

```python
# Production - use best available model
BranchingConfig(judge_model="openai/gpt-4o")

# Development - balance cost and quality
BranchingConfig(judge_model="openai/gpt-4o-mini")
```

### 3. Monitor Execution Time

Branching can be expensive - set appropriate timeouts:

```python
# Standard problems
BranchingConfig(timeout_seconds=30.0)

# Complex research questions
BranchingConfig(timeout_seconds=120.0)
```

### 4. Review All Hypotheses

Don't just use the selected hypothesis - review alternatives:

```python
result = await branching.execute(problem="...")

# Log all hypotheses for analysis
for h in result.hypotheses:
    logger.info(f"Mode {h.mode}: {h.reasoning[:100]}")

# Consider top 2-3 hypotheses for critical decisions
top_scores = sorted(result.scores, key=lambda s: s.score, reverse=True)[:3]
```

### 5. Track Pattern Performance

```python
stats = branching.get_stats()
print(f"Total executions: {stats['execution_count']}")
print(f"Average time: {stats['average_execution_time_ms']:.2f}ms")

# Branching-specific metrics via metadata
if result.success:
    print(f"Hypotheses generated: {len(result.hypotheses)}")
    print(f"Selection score: {max(s.score for s in result.scores):.2f}")
```

## Limitations

1. **Cost**: Generates multiple LLM queries + judge call per execution
2. **Latency**: Parallel generation helps but still slower than single query
3. **Diversity**: Limited by LightRAG query mode differences
4. **Judge Bias**: Judge model quality affects selection accuracy

## Troubleshooting

### No Hypotheses Generated

```python
# Check if LightRAG has indexed data
integration = LightRAGIntegration(working_dir="./lightrag_data")
# Ensure you've called integration.insert_documents() with your data

# Verify query modes work
local_result = await integration.local_query("test query")
print(f"Local query result: {local_result}")
```

### Judge Failures

```python
# Check judge model configuration
config = BranchingConfig(
    judge_model="openai/gpt-4o",  # Verify model name is correct
)

# Monitor judge failures via events
def on_error(event_type: str, data: dict):
    if "judge" in str(data.get("error", "")):
        print(f"Judge error: {data}")

branching.add_event_handler(on_error)
```

### Low Score Diversity

```python
# Increase diversity threshold
config = BranchingConfig(
    diversity_threshold=0.5,  # Higher = more diverse
    hypothesis_count=5         # More hypotheses = better coverage
)
```

## Performance Optimization

### Reduce Cost

```python
# Use fewer hypotheses
config = BranchingConfig(hypothesis_count=2)

# Use cheaper judge model
config = BranchingConfig(judge_model="openai/gpt-4o-mini")

# Disable recording if not needed
config = BranchingConfig(record_outcomes=False)
```

### Reduce Latency

```python
# Hypothesis generation is already parallel
# Focus on faster judge model
config = BranchingConfig(judge_model="openai/gpt-4o-mini")

# Set aggressive timeout
config = BranchingConfig(timeout_seconds=15.0)
```

## Related Patterns

- **Query Expansion**: Use before branching to improve hypothesis generation
- **Multi-Hop Retrieval**: Use branching for each hop's hypothesis selection
- **Hybrid Search**: Use as input to generate diverse hypotheses

## Further Reading

- [Pattern Selection Guide](./README.md#pattern-selection-guide)
- [Multi-Agent Integration](./README.md#multi-agent-integration)
- [LightRAG Documentation](https://github.com/gyasis/hybridrag)
