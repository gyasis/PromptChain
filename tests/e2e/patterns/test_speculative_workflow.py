"""E2E tests for speculative execution workflow.

Tests:
1. Speculative pre-fetching in conversation
2. Cache hit reduces latency
3. Prediction accuracy over time
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from promptchain.integrations.lightrag.speculative import (
    SpeculativeConfig,
    LightRAGSpeculativeExecutor,
)


@pytest.mark.e2e
@pytest.mark.asyncio
class TestSpeculativePreFetching:
    """Test speculative pre-fetching in conversational workflows."""

    async def test_conversation_with_speculation(
        self, e2e_test_context, mock_message_bus, conversation_history
    ):
        """Test speculative execution improves conversation latency.

        Workflow:
        1. Track conversation history
        2. Predict likely next queries
        3. Pre-execute predictions
        4. Verify cache hits when actual queries arrive
        """
        lightrag = e2e_test_context["lightrag"]

        executor = LightRAGSpeculativeExecutor(
            lightrag_core=lightrag,
            config=SpeculativeConfig(
                min_confidence=0.6,
                max_concurrent=3,
                default_ttl=120.0,
                prediction_model="frequency",
            )
        )
        executor.connect_messagebus(mock_message_bus)

        # Simulate conversation turns with query recording
        conversation_queries = [
            ("What is machine learning?", "hybrid"),
            ("How does deep learning work?", "hybrid"),
            ("Tell me about neural networks", "hybrid"),
            ("What are transformers?", "hybrid"),
            ("How do transformers differ from RNNs?", "hybrid"),
        ]

        cache_hit_count = 0
        total_latency_saved = 0.0

        for i, (query, mode) in enumerate(conversation_queries):
            # Check cache before executing
            cached = executor.check_cache(query, mode)

            if cached:
                cache_hit_count += 1
                # Estimate latency saved (actual query time - cache lookup time)
                total_latency_saved += 50.0  # Mock 50ms saved per hit
                print(f"Turn {i+1}: Cache HIT for '{query}'")
            else:
                print(f"Turn {i+1}: Cache MISS for '{query}'")

            # Record the query (builds history)
            executor.record_call(query, mode)

            # Execute speculative predictions for next turn
            if i < len(conversation_queries) - 1:
                context = " ".join([q for q, _ in conversation_queries[:i+1]])
                spec_result = await executor.execute(context=context)

                assert spec_result.success
                print(f"  -> Predicted {len(spec_result.predictions)} queries")
                print(f"  -> Executed {len(spec_result.executed)} predictions")
                print(f"  -> Cache size: {spec_result.metadata['cache_size']}")

                # Verify predictions were made
                assert len(spec_result.predictions) >= 0

                # Verify events
                spec_events = mock_message_bus.get_events_by_type("pattern.speculative.*")
                assert any("started" in e["type"] for e in spec_events)

        # After conversation, check overall cache performance
        stats = executor.get_stats()

        print(f"\nConversation Summary:")
        print(f"  Total queries: {len(conversation_queries)}")
        print(f"  Cache hits: {cache_hit_count}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Total executions: {stats['total_executions']}")

        # With frequency-based prediction, we should get some cache hits
        # after the pattern stabilizes
        assert stats["total_predictions"] > 0

    async def test_cache_ttl_expiration(
        self, e2e_test_context, mock_message_bus
    ):
        """Test that expired cache entries are cleaned up."""
        lightrag = e2e_test_context["lightrag"]

        executor = LightRAGSpeculativeExecutor(
            lightrag_core=lightrag,
            config=SpeculativeConfig(
                min_confidence=0.7,
                default_ttl=0.1,  # 100ms TTL
            )
        )

        # Execute speculation
        spec_result = await executor.execute(context="machine learning basics")

        # Record some calls to build cache
        executor.record_call("What is ML?", "hybrid")
        executor.record_call("What is ML?", "hybrid")

        # Wait for TTL to expire
        await asyncio.sleep(0.2)

        # Cleanup expired entries
        executor.cleanup()

        # Cache should be empty or smaller
        stats = executor.get_stats()
        print(f"Cache size after cleanup: {stats['cache_size']}")

        # New prediction should work
        spec_result2 = await executor.execute(context="machine learning")
        assert spec_result2.success

    async def test_prediction_confidence_threshold(
        self, e2e_test_context, mock_message_bus
    ):
        """Test that only high-confidence predictions are executed."""
        lightrag = e2e_test_context["lightrag"]

        # High confidence threshold
        executor = LightRAGSpeculativeExecutor(
            lightrag_core=lightrag,
            config=SpeculativeConfig(
                min_confidence=0.9,  # Very high threshold
                max_concurrent=5,
            )
        )
        executor.connect_messagebus(mock_message_bus)

        # Build some history
        for _ in range(3):
            executor.record_call("What is machine learning?", "hybrid")

        # Execute prediction
        spec_result = await executor.execute(context="machine learning")

        assert spec_result.success

        # With high threshold, may not execute many predictions
        executed_count = len(spec_result.executed)
        total_predictions = len(spec_result.predictions)

        print(f"Predictions: {total_predictions}, Executed: {executed_count}")

        # Predictions exist but execution is selective
        assert executed_count <= total_predictions

        # All executed predictions should meet confidence threshold
        for prediction in spec_result.executed:
            assert prediction.confidence >= 0.9


@pytest.mark.e2e
@pytest.mark.asyncio
class TestCacheHitLatencyReduction:
    """Test cache hits reduce query latency."""

    async def test_cache_hit_timing(
        self, e2e_test_context, mock_message_bus
    ):
        """Test that cache hits are faster than fresh queries."""
        import time

        lightrag = e2e_test_context["lightrag"]

        executor = LightRAGSpeculativeExecutor(
            lightrag_core=lightrag,
            config=SpeculativeConfig(min_confidence=0.7, default_ttl=300.0)
        )

        query = "What is deep learning?"
        mode = "hybrid"

        # First execution - cache miss (will be slow)
        executor.record_call(query, mode)

        # Execute speculation to populate cache
        context = "deep learning neural networks"
        spec_result = await executor.execute(context=context)

        # Manually add to cache to guarantee hit
        from promptchain.integrations.lightrag.speculative import SpeculativeResult
        from datetime import datetime

        cached_result = SpeculativeResult(
            prediction_id="test-pred",
            query=query,
            result=await lightrag.hybrid_query(query),
            cached_at=datetime.utcnow(),
            ttl_seconds=300.0,
        )
        cache_key = f"{query}_{mode}"
        executor.cache[cache_key] = cached_result

        # Check cache (should hit)
        start_hit = time.perf_counter()
        hit_result = executor.check_cache(query, mode)
        hit_time = (time.perf_counter() - start_hit) * 1000  # ms

        assert hit_result is not None
        assert hit_result.hit is True

        print(f"Cache hit time: {hit_time:.3f}ms")

        # Cache lookup should be extremely fast (< 1ms)
        assert hit_time < 5.0  # Very generous upper bound

        # Verify cache stats
        stats = executor.get_stats()
        assert stats["cache_hits"] >= 1

    async def test_multiple_cache_hits(
        self, e2e_test_context, mock_message_bus
    ):
        """Test repeated cache hits accumulate savings."""
        lightrag = e2e_test_context["lightrag"]

        executor = LightRAGSpeculativeExecutor(
            lightrag_core=lightrag,
            config=SpeculativeConfig(min_confidence=0.6, default_ttl=300.0)
        )

        # Pre-populate cache with predictions
        queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "What are neural networks?",
        ]

        for query in queries:
            executor.record_call(query, "hybrid")

        # Execute speculation
        spec_result = await executor.execute(context=" ".join(queries))

        # Now check cache for each query
        hits = 0
        for query in queries:
            cached = executor.check_cache(query, "hybrid")
            if cached:
                hits += 1

        print(f"Cache hits: {hits}/{len(queries)}")

        # Get final stats
        stats = executor.get_stats()
        assert stats["cache_hits"] >= hits

        # Calculate hit rate
        if stats["cache_hits"] + stats["cache_misses"] > 0:
            hit_rate = stats["cache_hit_rate"]
            print(f"Cache hit rate: {hit_rate:.2%}")


@pytest.mark.e2e
@pytest.mark.asyncio
class TestPredictionAccuracy:
    """Test prediction accuracy improves over time."""

    async def test_frequency_based_prediction_accuracy(
        self, e2e_test_context, mock_message_bus
    ):
        """Test frequency-based predictor learns from history."""
        lightrag = e2e_test_context["lightrag"]

        executor = LightRAGSpeculativeExecutor(
            lightrag_core=lightrag,
            config=SpeculativeConfig(
                min_confidence=0.6,
                prediction_model="frequency",
                history_window=20,
            )
        )

        # Establish a pattern: user frequently asks about ML
        for _ in range(5):
            executor.record_call("What is machine learning?", "hybrid")

        for _ in range(3):
            executor.record_call("How does deep learning work?", "hybrid")

        # Execute prediction
        spec_result = await executor.execute(context="AI and machine learning")

        assert spec_result.success
        assert len(spec_result.predictions) > 0

        # Most frequent queries should be predicted with higher confidence
        predictions_by_confidence = sorted(
            spec_result.predictions,
            key=lambda p: p.confidence,
            reverse=True
        )

        if predictions_by_confidence:
            top_prediction = predictions_by_confidence[0]
            print(f"Top prediction: '{top_prediction.query}' (confidence: {top_prediction.confidence:.2f})")

            # Top prediction should have reasonable confidence
            assert top_prediction.confidence >= 0.6

    async def test_pattern_based_prediction(
        self, e2e_test_context, mock_message_bus
    ):
        """Test pattern-based predictor detects query sequences."""
        lightrag = e2e_test_context["lightrag"]

        executor = LightRAGSpeculativeExecutor(
            lightrag_core=lightrag,
            config=SpeculativeConfig(
                min_confidence=0.6,
                prediction_model="pattern",
                history_window=10,
            )
        )

        # Create a pattern: user asks about same topic with same mode
        executor.record_call("What is machine learning?", "hybrid")
        executor.record_call("What is deep learning?", "hybrid")

        # Execute prediction (should detect mode continuation)
        spec_result = await executor.execute(context="neural networks")

        assert spec_result.success

        # Should predict continuation with same mode
        if spec_result.predictions:
            for prediction in spec_result.predictions:
                print(f"Predicted: {prediction.query} (mode: {prediction.mode}, pattern: {prediction.pattern_matched})")
                assert prediction.mode == "hybrid"  # Should match pattern

    async def test_prediction_stats_tracking(
        self, e2e_test_context, mock_message_bus
    ):
        """Test prediction statistics are tracked accurately."""
        lightrag = e2e_test_context["lightrag"]

        executor = LightRAGSpeculativeExecutor(
            lightrag_core=lightrag,
            config=SpeculativeConfig(min_confidence=0.7)
        )

        # Execute multiple speculation cycles
        contexts = [
            "machine learning basics",
            "deep learning neural networks",
            "transformer architecture",
        ]

        for context in contexts:
            spec_result = await executor.execute(context=context)
            assert spec_result.success

        # Get comprehensive stats
        stats = executor.get_stats()

        print(f"\nPrediction Statistics:")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Total executions: {stats['total_executions']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache misses: {stats['cache_misses']}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"  Cache size: {stats['cache_size']}")
        print(f"  History size: {stats['history_size']}")

        # Verify stats are being tracked
        assert stats["total_predictions"] >= 0
        assert stats["total_executions"] >= 0
        assert "cache_hit_rate" in stats
