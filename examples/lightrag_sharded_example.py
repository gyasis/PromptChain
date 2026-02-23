"""Example: LightRAG Sharded Retrieval Pattern

This example demonstrates how to use the LightRAG Sharded Retrieval Pattern
to query across multiple LightRAG database shards in parallel for improved
performance and fault tolerance.
"""

import asyncio
from promptchain.integrations.lightrag.sharded import (
    ShardType,
    ShardConfig,
    ShardedRetrievalConfig,
    LightRAGShardRegistry,
    LightRAGShardedRetriever,
)


async def basic_example():
    """Basic example: Query across multiple LIGHTRAG shards."""
    print("=== Basic Sharded Retrieval Example ===\n")

    # Create shard registry
    registry = LightRAGShardRegistry()

    # Register multiple LightRAG shards
    # In a real scenario, each shard would have different working directories
    # containing domain-specific knowledge
    try:
        registry.register_shard(ShardConfig(
            shard_id="tech_shard",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./lightrag_data/tech",
            priority=5,  # Higher priority for tech queries
            timeout_seconds=15.0
        ))

        registry.register_shard(ShardConfig(
            shard_id="business_shard",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./lightrag_data/business",
            priority=3,
            timeout_seconds=15.0
        ))

        registry.register_shard(ShardConfig(
            shard_id="research_shard",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./lightrag_data/research",
            priority=4,
            timeout_seconds=15.0
        ))

        print(f"Registered {len(registry.list_shards())} shards:")
        for shard_id in registry.list_shards():
            config = registry.get_config(shard_id)
            print(f"  - {shard_id}: {config.shard_type.value}, priority={config.priority}")

    except ImportError as e:
        print(f"Note: {e}")
        print("This example requires hybridrag to be installed.")
        print("Install with: pip install git+https://github.com/gyasis/hybridrag.git\n")
        return

    # Create retriever with configuration
    config = ShardedRetrievalConfig(
        parallel=True,  # Query shards in parallel
        fail_partial=True,  # Continue if some shards fail
        aggregate_top_k=10,  # Return top 10 results overall
        normalize_scores=True,  # Normalize scores across shards
        timeout_seconds=30.0
    )

    retriever = LightRAGShardedRetriever(registry=registry, config=config)

    # Execute query across all shards
    print("\nQuerying: 'What are the latest developments in AI?'\n")

    result = await retriever.execute(
        query="What are the latest developments in AI?"
    )

    # Display results
    print(f"Query Status: {'✓ Success' if result.success else '✗ Failed'}")
    print(f"Shards Queried: {result.shards_queried}")
    print(f"Shards Failed: {result.shards_failed}")
    print(f"Results Retrieved: {len(result.aggregated_results)}")
    print(f"Execution Time: {result.execution_time_ms:.2f}ms\n")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    # Display shard-level results
    print("\nShard-level Results:")
    for shard_result in result.shard_results:
        status = "✓" if shard_result.success else "✗"
        print(f"  {status} {shard_result.shard_id}: "
              f"{len(shard_result.results)} results in {shard_result.query_time_ms:.2f}ms")
        if shard_result.error:
            print(f"    Error: {shard_result.error}")


async def selective_shard_example():
    """Example: Query specific shards only."""
    print("\n=== Selective Shard Query Example ===\n")

    registry = LightRAGShardRegistry()

    try:
        # Register shards
        for shard_id in ["shard1", "shard2", "shard3"]:
            registry.register_shard(ShardConfig(
                shard_id=shard_id,
                shard_type=ShardType.LIGHTRAG,
                working_dir=f"./lightrag_data/{shard_id}"
            ))

        retriever = LightRAGShardedRetriever(
            registry=registry,
            config=ShardedRetrievalConfig(parallel=True)
        )

        # Query only specific shards
        print("Querying only shard1 and shard3...\n")

        result = await retriever.execute(
            query="Machine learning concepts",
            shard_ids=["shard1", "shard3"]  # Query only these shards
        )

        print(f"Queried {result.shards_queried} of {len(registry.list_shards())} shards")
        print(f"Results: {len(result.aggregated_results)}")

    except ImportError:
        print("This example requires hybridrag to be installed.")


async def fault_tolerance_example():
    """Example: Handling shard failures gracefully."""
    print("\n=== Fault Tolerance Example ===\n")

    registry = LightRAGShardRegistry()

    try:
        # Register shards with different configurations
        registry.register_shard(ShardConfig(
            shard_id="stable_shard",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./lightrag_data/stable"
        ))

        registry.register_shard(ShardConfig(
            shard_id="slow_shard",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./lightrag_data/slow",
            timeout_seconds=0.1  # Very short timeout (will likely fail)
        ))

        # With fail_partial=True (default), continue even if some shards fail
        retriever = LightRAGShardedRetriever(
            registry=registry,
            config=ShardedRetrievalConfig(
                parallel=True,
                fail_partial=True  # Continue despite failures
            )
        )

        result = await retriever.execute(query="Test query")

        print(f"Total shards: {result.shards_queried}")
        print(f"Failed shards: {result.shards_failed}")
        print(f"Success: {result.success}")  # Still succeeds with partial results

        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  {warning}")

    except ImportError:
        print("This example requires hybridrag to be installed.")


async def health_check_example():
    """Example: Check shard health before querying."""
    print("\n=== Health Check Example ===\n")

    registry = LightRAGShardRegistry()

    # Register some shards
    registry.register_shard(ShardConfig(
        shard_id="active_shard",
        shard_type=ShardType.LIGHTRAG,
        working_dir="./lightrag_data/active",
        enabled=True
    ))

    registry.register_shard(ShardConfig(
        shard_id="disabled_shard",
        shard_type=ShardType.LIGHTRAG,
        working_dir="./lightrag_data/disabled",
        enabled=False  # Shard is disabled
    ))

    # Check health
    health = registry.health_check()

    print("Shard Health Status:")
    for shard_id, is_healthy in health.items():
        status = "✓ Healthy" if is_healthy else "✗ Unhealthy"
        print(f"  {shard_id}: {status}")


async def mixed_shard_types_example():
    """Example: Registry with different shard types."""
    print("\n=== Mixed Shard Types Example ===\n")

    registry = LightRAGShardRegistry()

    # Register different types of shards
    registry.register_shard(ShardConfig(
        shard_id="vector_db",
        shard_type=ShardType.VECTOR_DB,
        connection_config={"host": "localhost", "port": 6333}
    ))

    registry.register_shard(ShardConfig(
        shard_id="api_service",
        shard_type=ShardType.API,
        connection_config={"endpoint": "https://api.example.com/search"}
    ))

    print("Registered shards:")
    for shard_id in registry.list_shards():
        config = registry.get_config(shard_id)
        print(f"  - {shard_id}: {config.shard_type.value}")

    print("\nNote: Non-LIGHTRAG shard types require custom query implementations.")


async def main():
    """Run all examples."""
    await basic_example()
    await selective_shard_example()
    await fault_tolerance_example()
    await health_check_example()
    await mixed_shard_types_example()

    print("\n=== Examples Complete ===")
    print("\nKey Features Demonstrated:")
    print("  ✓ Parallel shard querying")
    print("  ✓ Result aggregation with score normalization")
    print("  ✓ Priority-based ranking")
    print("  ✓ Fault tolerance and partial failures")
    print("  ✓ Selective shard querying")
    print("  ✓ Health checking")
    print("  ✓ Mixed shard types")


if __name__ == "__main__":
    asyncio.run(main())
