"""Demo of LightRAG Branching Thoughts Pattern.

This example demonstrates how to use the Branching Thoughts pattern to
generate multiple hypotheses using different LightRAG query modes and
select the best one using an LLM judge.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if hybridrag is available
try:
    from promptchain.integrations.lightrag import (
        LightRAGIntegration,
        LightRAGBranchingThoughts,
        LIGHTRAG_AVAILABLE,
    )
    from promptchain.integrations.lightrag.branching import BranchingConfig
except ImportError as e:
    print(f"Error importing LightRAG components: {e}")
    print("Install hybridrag with: pip install git+https://github.com/gyasis/hybridrag.git")
    exit(1)


async def main():
    """Demonstrate the Branching Thoughts pattern."""

    if not LIGHTRAG_AVAILABLE:
        print("hybridrag is not installed. Install with:")
        print("pip install git+https://github.com/gyasis/hybridrag.git")
        return

    print("=" * 70)
    print("LightRAG Branching Thoughts Pattern Demo")
    print("=" * 70)
    print()

    # Create LightRAG integration
    print("Initializing LightRAG integration...")
    integration = LightRAGIntegration()

    # Note: You would need to insert some documents first
    # integration.insert_documents([...])

    # Create Branching Thoughts pattern
    config = BranchingConfig(
        hypothesis_count=3,
        judge_model="openai/gpt-4o",
        diversity_threshold=0.3,
        emit_events=True,
    )

    branching = LightRAGBranchingThoughts(
        lightrag_core=integration,
        config=config
    )

    # Add event handler to see what's happening
    def event_handler(event_type: str, data: dict):
        print(f"Event: {event_type}")
        if "hypothesis_id" in data:
            print(f"  Hypothesis ID: {data['hypothesis_id']}")
        if "mode" in data:
            print(f"  Mode: {data['mode']}")
        if "score" in data:
            print(f"  Score: {data['score']:.2f}")
        print()

    branching.add_event_handler(event_handler)

    # Execute the pattern
    problem = "What are the key factors contributing to climate change?"

    print(f"Problem: {problem}")
    print()
    print("Generating hypotheses using different query modes...")
    print()

    result = await branching.execute(problem=problem)

    # Display results
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print()

    if result.success:
        print(f"Generated {len(result.hypotheses)} hypotheses:")
        print()

        for i, hypothesis in enumerate(result.hypotheses, 1):
            print(f"Hypothesis {i} (Mode: {hypothesis.mode})")
            print(f"  ID: {hypothesis.hypothesis_id}")
            print(f"  Approach: {hypothesis.approach}")
            print(f"  Confidence: {hypothesis.confidence:.2f}")
            print(f"  Reasoning: {hypothesis.reasoning[:200]}...")
            print()

        print("-" * 70)
        print("Hypothesis Scores:")
        print()

        for score in result.scores:
            print(f"Hypothesis: {score.hypothesis_id}")
            print(f"  Score: {score.score:.2f}")
            print(f"  Reasoning: {score.reasoning[:150]}...")
            print(f"  Strengths: {', '.join(score.strengths[:2])}")
            print(f"  Weaknesses: {', '.join(score.weaknesses[:2])}")
            print()

        print("-" * 70)
        print("Selected Hypothesis:")
        print()

        if result.selected_hypothesis:
            print(f"Mode: {result.selected_hypothesis.mode}")
            print(f"Approach: {result.selected_hypothesis.approach}")
            print(f"Reasoning: {result.selected_hypothesis.reasoning}")
            print()
            print(f"Selection Reasoning: {result.selection_reasoning}")
        else:
            print("No hypothesis selected")

        print()
        print(f"Total execution time: {result.execution_time_ms:.2f}ms")

    else:
        print("Pattern execution failed:")
        for error in result.errors:
            print(f"  - {error}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
