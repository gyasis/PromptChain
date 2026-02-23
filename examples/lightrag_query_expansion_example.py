"""Example: LightRAG Query Expansion Pattern

Demonstrates how to use the query expansion pattern to expand queries using
multiple strategies and execute parallel searches for comprehensive results.
"""

import asyncio
from promptchain.integrations.lightrag import (
    LightRAGIntegration,
    LightRAGQueryExpander,
    LIGHTRAG_AVAILABLE,
)
from promptchain.integrations.lightrag.query_expansion import (
    ExpansionStrategy,
    QueryExpansionConfig,
)


async def basic_query_expansion():
    """Example 1: Basic query expansion with default settings."""
    print("\n=== Example 1: Basic Query Expansion ===\n")

    if not LIGHTRAG_AVAILABLE:
        print("hybridrag not installed. Install with:")
        print("pip install git+https://github.com/gyasis/hybridrag.git")
        return

    # Initialize LightRAG integration
    integration = LightRAGIntegration()

    # Create query expander with default config (semantic expansion)
    expander = LightRAGQueryExpander(lightrag_integration=integration)

    # Execute query expansion
    result = await expander.execute(query="What is machine learning?")

    print(f"Original query: {result.original_query}")
    print(f"Success: {result.success}")
    print(f"Expanded queries: {len(result.expanded_queries)}")
    print(f"Unique results found: {result.unique_results_found}")
    print(f"Execution time: {result.execution_time_ms:.2f}ms")

    # Show expanded query variations
    print("\nExpanded query variations:")
    for i, eq in enumerate(result.expanded_queries, 1):
        print(f"  {i}. [{eq.strategy.value}] {eq.expanded_query}")
        print(f"     Similarity: {eq.similarity_score:.2f}")


async def multi_strategy_expansion():
    """Example 2: Query expansion with multiple strategies."""
    print("\n=== Example 2: Multi-Strategy Expansion ===\n")

    if not LIGHTRAG_AVAILABLE:
        print("hybridrag not installed.")
        return

    integration = LightRAGIntegration()

    # Configure expander with multiple strategies
    config = QueryExpansionConfig(
        strategies=[
            ExpansionStrategy.SEMANTIC,
            ExpansionStrategy.SYNONYM,
            ExpansionStrategy.REFORMULATION,
        ],
        max_expansions_per_strategy=3,
        min_similarity=0.6,
        deduplicate=True,
        parallel_search=True,
    )

    expander = LightRAGQueryExpander(
        lightrag_integration=integration,
        config=config
    )

    # Execute with multiple strategies
    result = await expander.execute(query="ML algorithms")

    print(f"Original query: {result.original_query}")
    print(f"Total expanded queries: {len(result.expanded_queries)}")

    # Group by strategy
    by_strategy = {}
    for eq in result.expanded_queries:
        strategy = eq.strategy.value
        if strategy not in by_strategy:
            by_strategy[strategy] = []
        by_strategy[strategy].append(eq)

    print("\nExpansions by strategy:")
    for strategy, queries in by_strategy.items():
        print(f"\n{strategy.upper()} ({len(queries)} expansions):")
        for eq in queries:
            print(f"  - {eq.expanded_query}")
            print(f"    Similarity: {eq.similarity_score:.2f}")


async def custom_similarity_threshold():
    """Example 3: Using custom similarity threshold."""
    print("\n=== Example 3: Custom Similarity Threshold ===\n")

    if not LIGHTRAG_AVAILABLE:
        print("hybridrag not installed.")
        return

    integration = LightRAGIntegration()

    # High similarity threshold (stricter filtering)
    strict_config = QueryExpansionConfig(
        strategies=[ExpansionStrategy.SEMANTIC],
        min_similarity=0.9,  # Very strict
    )

    # Low similarity threshold (more permissive)
    permissive_config = QueryExpansionConfig(
        strategies=[ExpansionStrategy.SEMANTIC],
        min_similarity=0.3,  # Very permissive
    )

    strict_expander = LightRAGQueryExpander(
        lightrag_integration=integration,
        config=strict_config
    )
    permissive_expander = LightRAGQueryExpander(
        lightrag_integration=integration,
        config=permissive_config
    )

    query = "neural networks"

    strict_result = await strict_expander.execute(query=query)
    permissive_result = await permissive_expander.execute(query=query)

    print(f"Query: {query}")
    print(f"\nStrict (>0.9): {len(strict_result.expanded_queries)} expansions")
    print(f"Permissive (>0.3): {len(permissive_result.expanded_queries)} expansions")


async def sequential_vs_parallel_search():
    """Example 4: Sequential vs parallel search execution."""
    print("\n=== Example 4: Sequential vs Parallel Search ===\n")

    if not LIGHTRAG_AVAILABLE:
        print("hybridrag not installed.")
        return

    integration = LightRAGIntegration()

    # Parallel search configuration
    parallel_config = QueryExpansionConfig(
        strategies=[ExpansionStrategy.SEMANTIC],
        max_expansions_per_strategy=5,
        parallel_search=True,
    )

    # Sequential search configuration
    sequential_config = QueryExpansionConfig(
        strategies=[ExpansionStrategy.SEMANTIC],
        max_expansions_per_strategy=5,
        parallel_search=False,
    )

    parallel_expander = LightRAGQueryExpander(
        lightrag_integration=integration,
        config=parallel_config
    )
    sequential_expander = LightRAGQueryExpander(
        lightrag_integration=integration,
        config=sequential_config
    )

    query = "deep learning architectures"

    # Time parallel execution
    parallel_result = await parallel_expander.execute(query=query)

    # Time sequential execution
    sequential_result = await sequential_expander.execute(query=query)

    print(f"Query: {query}")
    print(f"\nParallel search: {parallel_result.execution_time_ms:.2f}ms")
    print(f"  Unique results: {parallel_result.unique_results_found}")
    print(f"\nSequential search: {sequential_result.execution_time_ms:.2f}ms")
    print(f"  Unique results: {sequential_result.unique_results_found}")
    print(f"\nSpeedup: {sequential_result.execution_time_ms / parallel_result.execution_time_ms:.2f}x")


async def with_event_monitoring():
    """Example 5: Monitor events during query expansion."""
    print("\n=== Example 5: Event Monitoring ===\n")

    if not LIGHTRAG_AVAILABLE:
        print("hybridrag not installed.")
        return

    integration = LightRAGIntegration()

    # Create expander with event tracking
    expander = LightRAGQueryExpander(lightrag_integration=integration)

    # Track events
    events = []

    def event_handler(event_type: str, data: dict):
        events.append((event_type, data))
        print(f"[EVENT] {event_type}")
        if "query" in data:
            print(f"  Query: {data['query']}")
        if "expansion_count" in data:
            print(f"  Expansions: {data['expansion_count']}")
        if "unique_results" in data:
            print(f"  Results: {data['unique_results']}")

    expander.add_event_handler(event_handler)

    # Execute with event monitoring
    result = await expander.execute(query="artificial intelligence")

    print(f"\nTotal events: {len(events)}")
    print(f"Event types: {[e[0] for e in events]}")


async def acronym_expansion_example():
    """Example 6: Acronym expansion."""
    print("\n=== Example 6: Acronym Expansion ===\n")

    if not LIGHTRAG_AVAILABLE:
        print("hybridrag not installed.")
        return

    integration = LightRAGIntegration()

    # Configure for acronym expansion
    config = QueryExpansionConfig(
        strategies=[ExpansionStrategy.ACRONYM],
        max_expansions_per_strategy=3,
    )

    expander = LightRAGQueryExpander(
        lightrag_integration=integration,
        config=config
    )

    # Queries with acronyms
    queries = ["What is NLP?", "ML vs DL", "AI and AGI"]

    for query in queries:
        result = await expander.execute(query=query)
        print(f"\nQuery: {query}")
        print(f"Acronym expansions: {len(result.expanded_queries)}")
        for eq in result.expanded_queries:
            print(f"  - {eq.expanded_query}")


async def reformulation_example():
    """Example 7: Query reformulation."""
    print("\n=== Example 7: Query Reformulation ===\n")

    if not LIGHTRAG_AVAILABLE:
        print("hybridrag not installed.")
        return

    integration = LightRAGIntegration()

    # Configure for reformulation
    config = QueryExpansionConfig(
        strategies=[ExpansionStrategy.REFORMULATION],
        max_expansions_per_strategy=5,
    )

    expander = LightRAGQueryExpander(
        lightrag_integration=integration,
        config=config
    )

    # Simple statement to reformulate
    result = await expander.execute(query="machine learning")

    print(f"Original: {result.original_query}")
    print(f"\nReformulations:")
    for eq in result.expanded_queries:
        print(f"  - {eq.expanded_query}")


async def main():
    """Run all examples."""
    examples = [
        basic_query_expansion,
        multi_strategy_expansion,
        custom_similarity_threshold,
        sequential_vs_parallel_search,
        with_event_monitoring,
        acronym_expansion_example,
        reformulation_example,
    ]

    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
