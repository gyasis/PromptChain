"""Quick Start Example for LightRAG Patterns

Minimal example demonstrating basic pattern usage with LightRAG integration.

Prerequisites:
    pip install git+https://github.com/gyasis/hybridrag.git
    pip install litellm

Environment:
    Set OPENAI_API_KEY in .env file
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LightRAG patterns
from promptchain.integrations.lightrag import (
    LightRAGIntegration,
    LightRAGBranchingThoughts,
    LightRAGQueryExpander,
    BranchingConfig,
    QueryExpansionConfig,
    ExpansionStrategy,
)


async def main():
    """Run quick start examples for LightRAG patterns."""

    # Setup: Create working directory for LightRAG
    working_dir = Path("./lightrag_quickstart_data")
    working_dir.mkdir(exist_ok=True)

    print("=== LightRAG Patterns Quick Start ===\n")

    # Initialize LightRAG integration
    print("1. Initializing LightRAG...")
    integration = LightRAGIntegration(working_dir=str(working_dir))

    # Insert some sample documents (required for patterns to work)
    print("2. Indexing sample documents...")
    sample_docs = [
        """
        Machine Learning is a subset of artificial intelligence that focuses on
        building systems that learn from data. Deep Learning is a specialized
        branch of machine learning that uses neural networks with multiple layers.
        """,
        """
        Renewable energy sources include solar power, wind energy, and hydroelectric
        power. These are sustainable alternatives to fossil fuels and help reduce
        carbon emissions contributing to climate change.
        """,
        """
        Climate change is driven by greenhouse gas emissions, primarily from burning
        fossil fuels. This leads to global warming, rising sea levels, and extreme
        weather events.
        """,
    ]

    await integration.insert_documents(sample_docs)
    print("   ✓ Documents indexed\n")

    # Example 1: Branching Thoughts Pattern
    print("=" * 60)
    print("Example 1: Branching Thoughts Pattern")
    print("=" * 60)

    branching = LightRAGBranchingThoughts(
        lightrag_core=integration,
        config=BranchingConfig(
            hypothesis_count=3,
            judge_model="openai/gpt-4o-mini",  # Use mini for quick start
        ),
    )

    print("\nQuery: 'What are the main causes of climate change?'\n")
    branching_result = await branching.execute(
        problem="What are the main causes of climate change?"
    )

    if branching_result.success:
        print(f"✓ Generated {len(branching_result.hypotheses)} hypotheses")
        print(f"\nSelected Hypothesis ({branching_result.selected_hypothesis.mode} mode):")
        print(f"  {branching_result.selected_hypothesis.reasoning}\n")

        print("All hypotheses:")
        for i, h in enumerate(branching_result.hypotheses, 1):
            score = next(
                (s for s in branching_result.scores if s.hypothesis_id == h.hypothesis_id),
                None
            )
            print(f"\n  {i}. [{h.mode.upper()}] Score: {score.score if score else 'N/A':.2f}")
            print(f"     {h.reasoning[:100]}...")
    else:
        print(f"✗ Failed: {branching_result.errors}")

    print("\n" + "=" * 60)
    print("Example 2: Query Expansion Pattern")
    print("=" * 60)

    expander = LightRAGQueryExpander(
        lightrag_integration=integration,
        config=QueryExpansionConfig(
            strategies=[ExpansionStrategy.SEMANTIC, ExpansionStrategy.REFORMULATION],
            max_expansions_per_strategy=2,
        ),
    )

    print("\nQuery: 'renewable energy'\n")
    expansion_result = await expander.execute(query="renewable energy")

    if expansion_result.success:
        print(f"✓ Original query: {expansion_result.original_query}")
        print(f"✓ Generated {len(expansion_result.expanded_queries)} expansions\n")

        print("Expanded queries:")
        for eq in expansion_result.expanded_queries:
            print(f"\n  [{eq.strategy.value}] {eq.expanded_query}")
            print(f"  Similarity: {eq.similarity_score:.2f}")

        print(f"\n✓ Found {expansion_result.unique_results_found} unique results")
    else:
        print(f"✗ Failed: {expansion_result.errors}")

    print("\n" + "=" * 60)
    print("Example 3: Pattern Statistics")
    print("=" * 60)

    branching_stats = branching.get_stats()
    print("\nBranching Thoughts Statistics:")
    print(f"  Executions: {branching_stats['execution_count']}")
    print(f"  Avg time: {branching_stats['average_execution_time_ms']:.2f}ms")

    expansion_stats = expander.get_stats()
    print("\nQuery Expansion Statistics:")
    print(f"  Executions: {expansion_stats['execution_count']}")
    print(f"  Avg time: {expansion_stats['average_execution_time_ms']:.2f}ms")

    print("\n" + "=" * 60)
    print("✓ Quick Start Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Try other patterns: Multi-Hop, Hybrid Search, Sharded, Speculative")
    print("  2. See research_workflow.py for advanced multi-pattern integration")
    print("  3. Check docs/patterns/ for detailed documentation")


if __name__ == "__main__":
    asyncio.run(main())
