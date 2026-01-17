"""Example demonstrating LightRAG Speculative Execution Pattern.

This example shows how to use speculative execution to reduce latency
in LightRAG query processing by predicting and pre-executing likely
follow-up queries.
"""

import asyncio
from promptchain.integrations.lightrag import (
    LightRAGIntegration,
    LightRAGSpeculativeExecutor,
    LIGHTRAG_AVAILABLE
)
from promptchain.integrations.lightrag.core import LightRAGConfig
from promptchain.integrations.lightrag.speculative import SpeculativeConfig


async def basic_speculative_execution():
    """Demonstrate basic speculative execution workflow."""
    print("=== Basic Speculative Execution ===\n")

    if not LIGHTRAG_AVAILABLE:
        print("Error: hybridrag not installed")
        print("Install with: pip install git+https://github.com/gyasis/hybridrag.git")
        return

    # Initialize LightRAG
    lightrag_config = LightRAGConfig(
        working_dir="./lightrag_data",
        model_name="openai/gpt-4o-mini"
    )
    lightrag = LightRAGIntegration(config=lightrag_config)

    # Initialize speculative executor
    speculative_config = SpeculativeConfig(
        min_confidence=0.7,
        max_concurrent=3,
        default_ttl=120.0,  # 2 minutes cache
        prediction_model="frequency"
    )
    executor = LightRAGSpeculativeExecutor(
        lightrag_core=lightrag,
        config=speculative_config
    )

    # Simulate conversation with query patterns
    queries = [
        ("What is machine learning?", "hybrid"),
        ("What is deep learning?", "hybrid"),
        ("What is machine learning?", "hybrid"),  # Repeated query
        ("What is neural networks?", "local"),
        ("What is machine learning?", "hybrid"),  # Frequent query
    ]

    for query, mode in queries:
        print(f"\nQuery: {query} (mode={mode})")

        # Check cache first
        cached = executor.check_cache(query, mode)
        if cached:
            print(f"✓ Cache HIT! Saved latency: ~{executor.config.default_ttl}s")
            print(f"  Cached result: {str(cached.result)[:100]}...")
        else:
            print("✗ Cache MISS - executing query...")
            # Execute actual query
            if mode == "local":
                result = await lightrag.local_query(query)
            elif mode == "global":
                result = await lightrag.global_query(query)
            else:
                result = await lightrag.hybrid_query(query)

            print(f"  Query result: {str(result)[:100]}...")

        # Record in history for pattern learning
        executor.record_call(query, mode)

        # Execute speculative predictions for next queries
        print("\n  Running speculative execution...")
        spec_result = await executor.execute(context=query)

        if spec_result.success:
            print(f"  Generated {len(spec_result.predictions)} predictions")
            print(f"  Executed {len(spec_result.executed)} high-confidence predictions")

            for pred in spec_result.executed:
                print(f"    - '{pred.query[:50]}...' (confidence={pred.confidence:.2f})")

    # Show statistics
    print("\n=== Execution Statistics ===")
    stats = executor.get_stats()
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Total executions: {stats['total_executions']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"Cache size: {stats['cache_size']} entries")


async def pattern_based_prediction():
    """Demonstrate pattern-based prediction model."""
    print("\n\n=== Pattern-Based Prediction ===\n")

    if not LIGHTRAG_AVAILABLE:
        print("Error: hybridrag not installed")
        return

    # Initialize with pattern-based model
    lightrag_config = LightRAGConfig(working_dir="./lightrag_data")
    lightrag = LightRAGIntegration(config=lightrag_config)

    speculative_config = SpeculativeConfig(
        min_confidence=0.6,
        max_concurrent=2,
        prediction_model="pattern",  # Use pattern-based prediction
        history_window=10
    )
    executor = LightRAGSpeculativeExecutor(
        lightrag_core=lightrag,
        config=speculative_config
    )

    # Simulate conversation with clear patterns
    conversation = [
        ("Overview of AI", "global"),
        ("Overview of machine learning", "global"),  # Same mode pattern
        ("Overview of deep learning", "global"),  # Continuation pattern
    ]

    for query, mode in conversation:
        print(f"\nQuery: {query} (mode={mode})")
        executor.record_call(query, mode)

        # Generate predictions based on pattern
        spec_result = await executor.execute(context=query)

        if spec_result.predictions:
            print(f"  Pattern predictions:")
            for pred in spec_result.predictions:
                print(f"    - Pattern: {pred.pattern_matched}")
                print(f"      Query: '{pred.query[:50]}...'")
                print(f"      Mode: {pred.mode}, Confidence: {pred.confidence:.2f}")


async def cache_management_demo():
    """Demonstrate cache TTL and cleanup."""
    print("\n\n=== Cache Management Demo ===\n")

    if not LIGHTRAG_AVAILABLE:
        print("Error: hybridrag not installed")
        return

    lightrag_config = LightRAGConfig(working_dir="./lightrag_data")
    lightrag = LightRAGIntegration(config=lightrag_config)

    # Short TTL for demonstration
    speculative_config = SpeculativeConfig(
        default_ttl=5.0,  # 5 seconds
        max_concurrent=2
    )
    executor = LightRAGSpeculativeExecutor(
        lightrag_core=lightrag,
        config=speculative_config
    )

    # Add some entries
    print("Adding cache entries...")
    for i in range(3):
        query = f"Test query {i}"
        executor.record_call(query, "hybrid")

    result = await executor.execute(context="Initial context")
    print(f"Cache size: {len(executor.cache)} entries")

    # Wait for expiration
    print("\nWaiting 6 seconds for cache expiration...")
    await asyncio.sleep(6)

    # Cleanup expired entries
    print("Running cleanup...")
    executor.cleanup()
    print(f"Cache size after cleanup: {len(executor.cache)} entries")


async def event_monitoring_demo():
    """Demonstrate event emission and monitoring."""
    print("\n\n=== Event Monitoring Demo ===\n")

    if not LIGHTRAG_AVAILABLE:
        print("Error: hybridrag not installed")
        return

    lightrag_config = LightRAGConfig(working_dir="./lightrag_data")
    lightrag = LightRAGIntegration(config=lightrag_config)

    speculative_config = SpeculativeConfig(
        emit_events=True,  # Enable event emission
        max_concurrent=2
    )
    executor = LightRAGSpeculativeExecutor(
        lightrag_core=lightrag,
        config=speculative_config
    )

    # Add event handler
    events_received = []

    def event_handler(event_type: str, data: dict):
        events_received.append((event_type, data))
        print(f"Event: {event_type}")
        if "num_predictions" in data:
            print(f"  Predictions: {data['num_predictions']}")
        if "num_executed" in data:
            print(f"  Executed: {data['num_executed']}")

    executor.add_event_handler(event_handler)

    # Execute with event tracking
    print("Executing with event monitoring...")
    executor.record_call("Test query", "hybrid")
    await executor.execute(context="Test context")

    print(f"\nTotal events received: {len(events_received)}")
    for event_type, _ in events_received:
        print(f"  - {event_type}")


async def main():
    """Run all examples."""
    try:
        await basic_speculative_execution()
        await pattern_based_prediction()
        await cache_management_demo()
        await event_monitoring_demo()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
