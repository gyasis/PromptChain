"""Example usage of LightRAG Hybrid Search Fusion Pattern.

This example demonstrates how to use the hybrid search pattern to combine
multiple retrieval techniques with sophisticated fusion algorithms.
"""

import asyncio
from promptchain.integrations.lightrag.core import LightRAGIntegration, LightRAGConfig
from promptchain.integrations.lightrag.hybrid_search import (
    LightRAGHybridSearcher,
    HybridSearchConfig,
    SearchTechnique,
    FusionAlgorithm,
)


async def example_rrf_fusion():
    """Example: Use Reciprocal Rank Fusion to combine local and global queries."""
    print("\n=== Example 1: RRF Fusion ===\n")

    # Initialize LightRAG
    config = LightRAGConfig(
        working_dir="./lightrag_data",
        model_name="openai/gpt-4o-mini"
    )
    integration = LightRAGIntegration(config=config)

    # Configure hybrid search with RRF
    search_config = HybridSearchConfig(
        techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
        fusion_algorithm=FusionAlgorithm.RRF,
        rrf_k=60,  # Standard RRF constant
        top_k=10,
    )

    # Create searcher
    searcher = LightRAGHybridSearcher(integration, config=search_config)

    # Execute hybrid search
    result = await searcher.execute(query="What is machine learning?")

    print(f"Query: {result.query}")
    print(f"Success: {result.success}")
    print(f"Execution time: {result.execution_time_ms:.2f}ms")
    print(f"\nTechnique contributions: {result.technique_contributions}")
    print(f"\nTop {len(result.fused_results)} results:")
    for i, (res, score) in enumerate(zip(result.fused_results, result.fused_scores)):
        print(f"  {i+1}. Score: {score:.4f} - {res[:100]}...")


async def example_linear_fusion():
    """Example: Use weighted linear combination for fusion."""
    print("\n=== Example 2: Linear Fusion with Score Normalization ===\n")

    integration = LightRAGIntegration()

    # Configure hybrid search with linear fusion
    search_config = HybridSearchConfig(
        techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL, SearchTechnique.HYBRID],
        fusion_algorithm=FusionAlgorithm.LINEAR,
        top_k=5,
        normalize_scores=True,  # Normalize scores before fusion
    )

    searcher = LightRAGHybridSearcher(integration, config=search_config)

    result = await searcher.execute(query="Explain neural networks")

    print(f"Query: {result.query}")
    print(f"Techniques used: {[tr.technique.value for tr in result.technique_results]}")
    print(f"Technique contributions: {result.technique_contributions}")


async def example_borda_fusion():
    """Example: Use Borda count voting for democratic fusion."""
    print("\n=== Example 3: Borda Count Fusion ===\n")

    integration = LightRAGIntegration()

    # Configure hybrid search with Borda count
    search_config = HybridSearchConfig(
        techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
        fusion_algorithm=FusionAlgorithm.BORDA,
        top_k=8,
    )

    searcher = LightRAGHybridSearcher(integration, config=search_config)

    result = await searcher.execute(query="Deep learning architectures")

    print(f"Query: {result.query}")
    print(f"Total results from techniques: {sum(len(tr.results) for tr in result.technique_results)}")
    print(f"Fused to top {len(result.fused_results)} results")
    print(f"Borda scores: {result.fused_scores}")


async def example_event_tracking():
    """Example: Track search events using event handlers."""
    print("\n=== Example 4: Event Tracking ===\n")

    integration = LightRAGIntegration()

    search_config = HybridSearchConfig(
        techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
        fusion_algorithm=FusionAlgorithm.RRF,
        top_k=5,
        emit_events=True,  # Enable event emission
    )

    searcher = LightRAGHybridSearcher(integration, config=search_config)

    # Add event handler
    def event_handler(event_type: str, data: dict):
        print(f"[EVENT] {event_type}: {data.get('num_techniques', data.get('query', 'N/A'))}")

    searcher.add_event_handler(event_handler)

    # Execute search
    result = await searcher.execute(query="Transformer models")

    print(f"\nFinal result: {len(result.fused_results)} documents retrieved")


async def example_custom_top_k():
    """Example: Override top_k at execution time."""
    print("\n=== Example 5: Dynamic top_k Override ===\n")

    integration = LightRAGIntegration()

    # Default config with top_k=10
    search_config = HybridSearchConfig(
        techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
        fusion_algorithm=FusionAlgorithm.RRF,
        top_k=10,  # Default
    )

    searcher = LightRAGHybridSearcher(integration, config=search_config)

    # Override at execution time
    result = await searcher.execute(
        query="Computer vision techniques",
        top_k=3  # Override to get only top 3 results
    )

    print(f"Configured top_k: 10")
    print(f"Actual results returned: {len(result.fused_results)} (overridden to 3)")


async def main():
    """Run all examples."""
    print("LightRAG Hybrid Search Fusion Pattern Examples")
    print("=" * 60)

    try:
        await example_rrf_fusion()
        await example_linear_fusion()
        await example_borda_fusion()
        await example_event_tracking()
        await example_custom_top_k()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nTo run these examples, install hybridrag:")
        print("  pip install git+https://github.com/gyasis/hybridrag.git")


if __name__ == "__main__":
    asyncio.run(main())
