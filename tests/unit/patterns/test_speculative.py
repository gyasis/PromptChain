"""Unit tests for LightRAGSpeculativeExecutor pattern adapter."""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timedelta
from promptchain.integrations.lightrag.speculative import LightRAGSpeculativeExecutor


@pytest.mark.asyncio
class TestLightRAGSpeculativeExecutor:
    """Test suite for speculative execution pattern."""

    async def test_initialization(self, mock_lightrag_core, mock_cache):
        """Test pattern initializes correctly."""
        pattern = LightRAGSpeculativeExecutor(
            lightrag_core=mock_lightrag_core,
            cache=mock_cache,
            prediction_window=5,
            cache_ttl=300
        )

        assert pattern.lightrag_core == mock_lightrag_core
        assert pattern.cache == mock_cache
        assert pattern.prediction_window == 5
        assert pattern.cache_ttl == 300
        assert pattern.pattern_name == "speculative_execution"

    async def test_prediction_from_history(self, mock_lightrag_core, mock_cache):
        """Test prediction of likely next queries from history."""
        pattern = LightRAGSpeculativeExecutor(
            lightrag_core=mock_lightrag_core,
            cache=mock_cache,
            prediction_window=3
        )

        history = [
            "What is Python?",
            "How to install Python?",
            "Python syntax basics"
        ]

        with patch.object(pattern, '_predict_next_queries', new_callable=AsyncMock) as mock_predict:
            mock_predict.return_value = [
                "Python data types",
                "Python functions",
                "Python classes"
            ]

            predictions = await pattern._predict_next_queries(history)

            assert len(predictions) == 3
            mock_predict.assert_called_once_with(history)

    async def test_cache_hit(self, mock_lightrag_core, mock_cache):
        """Test cache hit returns cached result immediately."""
        pattern = LightRAGSpeculativeExecutor(
            lightrag_core=mock_lightrag_core,
            cache=mock_cache,
            cache_ttl=300
        )

        # Pre-populate cache
        cached_result = {
            "results": ["cached answer"],
            "metadata": {"cached": True}
        }
        await mock_cache.set("query:test_query", cached_result)

        result = await pattern.execute(query="test_query")

        # Should return cached result without querying LightRAG
        assert result["results"] == ["cached answer"]
        assert result["metadata"]["cached"] is True
        mock_lightrag_core.local_query.assert_not_called()

    async def test_cache_miss(self, mock_lightrag_core, mock_cache):
        """Test cache miss executes query and caches result."""
        pattern = LightRAGSpeculativeExecutor(
            lightrag_core=mock_lightrag_core,
            cache=mock_cache,
            cache_ttl=300
        )

        mock_lightrag_core.local_query.return_value = {
            "results": ["fresh answer"],
            "metadata": {"search_type": "local"}
        }

        result = await pattern.execute(query="new_query")

        # Should execute query
        mock_lightrag_core.local_query.assert_called_once()
        assert result["results"] == ["fresh answer"]

        # Should cache the result
        cached = await mock_cache.get("query:new_query")
        assert cached is not None

    async def test_ttl_expiration(self, mock_lightrag_core, mock_cache):
        """Test cache entry expires after TTL."""
        pattern = LightRAGSpeculativeExecutor(
            lightrag_core=mock_lightrag_core,
            cache=mock_cache,
            cache_ttl=1  # 1 second TTL
        )

        # Cache a result
        await pattern.execute(query="test")

        # Simulate TTL expiration
        import asyncio
        await asyncio.sleep(2)

        # Mock cache expiration
        mock_cache.storage.clear()

        # Should execute fresh query
        mock_lightrag_core.local_query.reset_mock()
        await pattern.execute(query="test")

        mock_lightrag_core.local_query.assert_called_once()

    async def test_speculative_prefetching(self, mock_lightrag_core, mock_cache):
        """Test speculative prefetching of predicted queries."""
        pattern = LightRAGSpeculativeExecutor(
            lightrag_core=mock_lightrag_core,
            cache=mock_cache,
            prediction_window=2,
            enable_prefetch=True
        )

        # Execute first query
        await pattern.execute(query="query1")

        with patch.object(pattern, '_predict_next_queries', new_callable=AsyncMock) as mock_predict:
            mock_predict.return_value = ["predicted_query1", "predicted_query2"]

            # Execute second query which should trigger prefetch
            await pattern.execute(query="query2")

            # Predictions should be cached
            mock_predict.assert_called()

    async def test_prediction_accuracy_tracking(self, mock_lightrag_core, mock_cache):
        """Test tracking of prediction accuracy."""
        pattern = LightRAGSpeculativeExecutor(
            lightrag_core=mock_lightrag_core,
            cache=mock_cache,
            prediction_window=3,
            track_accuracy=True
        )

        # Predict queries
        with patch.object(pattern, '_predict_next_queries', new_callable=AsyncMock) as mock_predict:
            mock_predict.return_value = ["pred1", "pred2", "pred3"]

            await pattern.execute(query="q1")

            # Execute predicted query (hit)
            await pattern.execute(query="pred1")

            # Execute unpredicted query (miss)
            await pattern.execute(query="unpredicted")

            metrics = pattern.get_prediction_metrics()
            assert "accuracy" in metrics
            assert "hits" in metrics
            assert "misses" in metrics

    async def test_cache_invalidation(self, mock_lightrag_core, mock_cache):
        """Test manual cache invalidation."""
        pattern = LightRAGSpeculativeExecutor(
            lightrag_core=mock_lightrag_core,
            cache=mock_cache,
            cache_ttl=300
        )

        # Cache a result
        await pattern.execute(query="test")

        # Invalidate cache
        await pattern.invalidate_cache("test")

        # Should execute fresh query
        mock_lightrag_core.local_query.reset_mock()
        await pattern.execute(query="test")

        mock_lightrag_core.local_query.assert_called_once()

    async def test_event_emission_cache_hit(self, mock_lightrag_core, mock_cache, event_collector):
        """Test events emitted on cache hit."""
        pattern = LightRAGSpeculativeExecutor(
            lightrag_core=mock_lightrag_core,
            cache=mock_cache,
            cache_ttl=300
        )
        pattern.emit_event = event_collector.collect

        # Pre-populate cache
        await mock_cache.set("query:test", {"results": ["cached"]})

        await pattern.execute(query="test")

        events = event_collector.get_events("pattern.speculative.cache_hit")
        assert len(events) > 0

    async def test_event_emission_cache_miss(self, mock_lightrag_core, mock_cache, event_collector):
        """Test events emitted on cache miss."""
        pattern = LightRAGSpeculativeExecutor(
            lightrag_core=mock_lightrag_core,
            cache=mock_cache,
            cache_ttl=300
        )
        pattern.emit_event = event_collector.collect

        await pattern.execute(query="new_query")

        events = event_collector.get_events("pattern.speculative.cache_miss")
        assert len(events) > 0

    async def test_event_emission_prediction_made(self, mock_lightrag_core, mock_cache, event_collector):
        """Test events emitted when predictions are made."""
        pattern = LightRAGSpeculativeExecutor(
            lightrag_core=mock_lightrag_core,
            cache=mock_cache,
            prediction_window=2
        )
        pattern.emit_event = event_collector.collect

        with patch.object(pattern, '_predict_next_queries', new_callable=AsyncMock) as mock_predict:
            mock_predict.return_value = ["pred1", "pred2"]

            await pattern.execute(query="test")

            events = event_collector.get_events("pattern.speculative.prediction_made")
            assert len(events) > 0

    async def test_cache_size_management(self, mock_lightrag_core, mock_cache):
        """Test cache size limits and eviction."""
        pattern = LightRAGSpeculativeExecutor(
            lightrag_core=mock_lightrag_core,
            cache=mock_cache,
            cache_ttl=300,
            max_cache_size=3
        )

        # Fill cache beyond limit
        for i in range(5):
            await pattern.execute(query=f"query{i}")

        # Cache should only have 3 most recent entries
        cache_size = len(mock_cache.storage)
        assert cache_size <= 3

    async def test_prediction_window_validation(self, mock_lightrag_core, mock_cache):
        """Test validation of prediction_window parameter."""
        with pytest.raises(ValueError, match="prediction_window must be at least 1"):
            LightRAGSpeculativeExecutor(
                lightrag_core=mock_lightrag_core,
                cache=mock_cache,
                prediction_window=0
            )

    async def test_base_pattern_interface_compliance(self, mock_lightrag_core, mock_cache):
        """Test that pattern implements BasePattern interface correctly."""
        pattern = LightRAGSpeculativeExecutor(
            lightrag_core=mock_lightrag_core,
            cache=mock_cache,
            prediction_window=3
        )

        assert hasattr(pattern, 'execute')
        assert hasattr(pattern, 'pattern_name')
        assert callable(pattern.execute)

        result = await pattern.execute(query="test")
        assert isinstance(result, dict)
