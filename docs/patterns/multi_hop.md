# Multi-Hop Retrieval Pattern

Iterative question decomposition and multi-hop reasoning using LightRAG's agentic search capabilities.

## Overview

The Multi-Hop Retrieval pattern implements complex question answering through:
1. Decomposing complex questions into sub-questions
2. Executing multi-hop retrieval via agentic search
3. Tracking reasoning hops and intermediate results
4. Synthesizing unified answers from sub-answers

This pattern excels at questions requiring step-by-step reasoning and information synthesis.

## Installation

```bash
pip install git+https://github.com/gyasis/hybridrag.git
pip install litellm  # For question decomposition
```

## Basic Usage

```python
from promptchain.integrations.lightrag import (
    LightRAGIntegration,
    LightRAGMultiHop,
    MultiHopConfig
)

# Initialize
integration = LightRAGIntegration(working_dir="./lightrag_data")

# Create pattern
multi_hop = LightRAGMultiHop(
    search_interface=integration.search,
    config=MultiHopConfig(
        max_hops=5,
        decompose_first=True
    )
)

# Execute
result = await multi_hop.execute(
    question="What are the key differences between transformers and RNNs in NLP?"
)

print(f"Unified Answer: {result.unified_answer}")
print(f"Hops Executed: {result.hops_executed}")
print(f"Sub-questions: {len(result.sub_questions)}")

# Review reasoning steps
for sq in result.sub_questions:
    print(f"
Q: {sq.question_text}")
    print(f"A: {sq.answer}")
```

## Configuration

```python
MultiHopConfig(
    max_hops=5,                  # Maximum reasoning hops
    max_sub_questions=5,         # Max sub-questions to generate
    synthesizer_model="openai/gpt-4o-mini",
    decompose_first=True         # Decompose before retrieval
)
```

## Result Structure

```python
@dataclass
class MultiHopResult:
    original_question: str
    sub_questions: List[SubQuestion]
    sub_answers: Dict[str, str]
    unified_answer: str
    hops_executed: int
    unanswered_aspects: List[str]
```

## Events

```python
"pattern.multi_hop.started"
"pattern.multi_hop.decomposed"
"pattern.multi_hop.hop_completed"
"pattern.multi_hop.synthesizing"
"pattern.multi_hop.completed"
```

## Best Practices

1. **Enable Decomposition**: Set decompose_first=True for complex questions
2. **Set Appropriate Hop Limit**: 5-7 hops for most questions
3. **Use Good Synthesizer**: Quality model for answer synthesis
4. **Review Unanswered Aspects**: Check what couldn't be answered

## Related Patterns

- Branching Thoughts: Use for hypothesis evaluation at each hop
- Query Expansion: Expand each sub-question for better retrieval
