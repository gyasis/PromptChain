"""Tests for LightRAG Speculative Execution Pattern."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from promptchain.integrations.lightrag.speculative import (
    LightRAGSpeculativeExecutor,
    SpeculativeConfig,
    ToolPrediction,
    SpeculativeResult,
    SpeculativeExecutionResult,
    LIGHTRAG_AVAILABLE
)


@pytest.fixture
def mock_lightrag_core():
    """Create a mock LightRAGIntegration instance."""
    mock_core = MagicMock()
    mock_core.local_query = AsyncMock(return_value={"result": "local_data"})
    mock_core.global_query = AsyncMock(return_value={"result": "global_data"})
    mock_core.hybrid_query = AsyncMock(return_value={"result": "hybrid_data"})
    return mock_core


@pytest.fixture
def executor(mock_lightrag_core):
    """Create a SpeculativeExecutor instance."""
    config = SpeculativeConfig(
        min_confidence=0.6,
        max_concurrent=2,
        default_ttl=60.0,
        history_window=10
    )
    with patch("promptchain.integrations.lightrag.speculative.LIGHTRAG_AVAILABLE", True):
        return LightRAGSpeculativeExecutor(mock_lightrag_core, config)


class TestToolPrediction:
    """Test ToolPrediction dataclass."""

    def test_creation(self):
        """Test creating a ToolPrediction."""
        prediction = ToolPrediction(
            prediction_id="pred_001",
            query="What is ML?",
            mode="hybrid",
            confidence=0.85,
            pattern_matched="frequency",
            context_hash="abc123"
        )

        assert prediction.prediction_id == "pred_001"
        assert prediction.query == "What is ML?"
        assert prediction.mode == "hybrid"
        assert prediction.confidence == 0.85
        assert prediction.pattern_matched == "frequency"
        assert prediction.context_hash == "abc123"


class TestSpeculativeResult:
    """Test SpeculativeResult dataclass."""

    def test_creation(self):
        """Test creating a SpeculativeResult."""
        result = SpeculativeResult(
            prediction_id="pred_001",
            query="What is ML?",
            result={"answer": "Machine Learning"},
            cached_at=datetime.utcnow(),
            ttl_seconds=60.0,
            hit=False
        )

        assert result.prediction_id == "pred_001"
        assert result.query == "What is ML?"
        assert result.result == {"answer": "Machine Learning"}
        assert not result.hit
        assert result.ttl_seconds == 60.0

    def test_is_expired(self):
        """Test checking if result is expired."""
        from datetime import timedelta

        # Not expired
        result = SpeculativeResult(
            prediction_id="pred_001",
            query="test",
            result={},
            cached_at=datetime.utcnow(),
            ttl_seconds=60.0
        )
        assert not result.is_expired()

        # Expired
        result_old = SpeculativeResult(
            prediction_id="pred_002",
            query="test",
            result={},
            cached_at=datetime.utcnow() - timedelta(seconds=120),
            ttl_seconds=60.0
        )
        assert result_old.is_expired()


class TestSpeculativeConfig:
    """Test SpeculativeConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SpeculativeConfig()

        assert config.min_confidence == 0.7
        assert config.max_concurrent == 3
        assert config.default_ttl == 60.0
        assert config.prediction_model == "frequency"
        assert config.history_window == 20

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SpeculativeConfig(
            min_confidence=0.8,
            max_concurrent=5,
            default_ttl=120.0,
            prediction_model="pattern",
            history_window=30
        )

        assert config.min_confidence == 0.8
        assert config.max_concurrent == 5
        assert config.default_ttl == 120.0
        assert config.prediction_model == "pattern"
        assert config.history_window == 30


class TestLightRAGSpeculativeExecutor:
    """Test LightRAGSpeculativeExecutor class."""

    def test_initialization(self, mock_lightrag_core):
        """Test executor initialization."""
        config = SpeculativeConfig(history_window=15)

        with patch("promptchain.integrations.lightrag.speculative.LIGHTRAG_AVAILABLE", True):
            executor = LightRAGSpeculativeExecutor(mock_lightrag_core, config)

            assert executor.lightrag_core == mock_lightrag_core
            assert executor.config.history_window == 15
            assert len(executor.history) == 0
            assert len(executor.cache) == 0

    def test_initialization_without_lightrag(self, mock_lightrag_core):
        """Test that initialization fails without LightRAG."""
        with patch("promptchain.integrations.lightrag.speculative.LIGHTRAG_AVAILABLE", False):
            with pytest.raises(ImportError, match="hybridrag is not installed"):
                LightRAGSpeculativeExecutor(mock_lightrag_core)

    def test_record_call(self, executor):
        """Test recording a query in history."""
        executor.record_call("What is AI?", "hybrid")

        assert len(executor.history) == 1
        assert executor.history[0]["query"] == "What is AI?"
        assert executor.history[0]["mode"] == "hybrid"
        assert "timestamp" in executor.history[0]

    def test_history_window_limit(self, mock_lightrag_core):
        """Test that history respects window limit."""
        # Create executor with custom history window
        config = SpeculativeConfig(history_window=3)

        with patch("promptchain.integrations.lightrag.speculative.LIGHTRAG_AVAILABLE", True):
            executor = LightRAGSpeculativeExecutor(mock_lightrag_core, config)

        for i in range(5):
            executor.record_call(f"Query {i}", "hybrid")

        assert len(executor.history) == 3
        assert executor.history[0]["query"] == "Query 2"
        assert executor.history[-1]["query"] == "Query 4"

    def test_compute_context_hash(self, executor):
        """Test context hash computation."""
        hash1 = executor._compute_context_hash("test context")
        hash2 = executor._compute_context_hash("test context")
        hash3 = executor._compute_context_hash("different context")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

    def test_analyze_frequency_patterns(self, executor):
        """Test frequency pattern analysis."""
        executor.record_call("What is AI?", "hybrid")
        executor.record_call("What is AI?", "hybrid")
        executor.record_call("What is ML?", "local")

        patterns = executor._analyze_frequency_patterns()

        assert len(patterns) >= 2
        # Check that "What is AI?" appears twice
        ai_pattern = [p for p in patterns.values() if "AI" in p["query"]]
        assert len(ai_pattern) == 1
        assert ai_pattern[0]["count"] == 2

    def test_predict_next_queries_frequency(self, executor):
        """Test frequency-based query prediction."""
        executor.config.prediction_model = "frequency"
        executor.config.min_confidence = 0.3

        # Add some history
        for i in range(5):
            executor.record_call("What is AI?", "hybrid")
        executor.record_call("What is ML?", "local")

        predictions = executor.predict_next_queries("Current context")

        assert len(predictions) > 0
        # Most frequent query should be predicted
        assert any("AI" in p.query for p in predictions)
        # Predictions should be sorted by confidence
        confidences = [p.confidence for p in predictions]
        assert confidences == sorted(confidences, reverse=True)

    def test_predict_next_queries_pattern(self, executor):
        """Test pattern-based query prediction."""
        executor.config.prediction_model = "pattern"

        # Add consecutive queries with same mode
        executor.record_call("Query 1", "hybrid")
        executor.record_call("Query 2", "hybrid")

        predictions = executor.predict_next_queries("Current context")

        assert len(predictions) > 0
        assert predictions[0].pattern_matched == "mode_continuation"
        assert predictions[0].mode == "hybrid"

    def test_check_cache_hit(self, executor):
        """Test cache hit."""
        # Add entry to cache
        cached_result = SpeculativeResult(
            prediction_id="pred_001",
            query="What is AI?",
            result={"answer": "Artificial Intelligence"},
            cached_at=datetime.utcnow(),
            ttl_seconds=60.0
        )
        executor.cache["What is AI?_hybrid"] = cached_result

        # Check cache
        result = executor.check_cache("What is AI?", "hybrid")

        assert result is not None
        assert result.query == "What is AI?"
        assert result.hit
        assert executor._cache_hits == 1

    def test_check_cache_miss(self, executor):
        """Test cache miss."""
        result = executor.check_cache("Unknown query", "hybrid")

        assert result is None
        assert executor._cache_misses == 1

    def test_check_cache_expired(self, executor):
        """Test that expired entries are removed."""
        from datetime import timedelta

        # Add expired entry
        expired_result = SpeculativeResult(
            prediction_id="pred_001",
            query="Old query",
            result={},
            cached_at=datetime.utcnow() - timedelta(seconds=120),
            ttl_seconds=60.0
        )
        executor.cache["Old query_hybrid"] = expired_result

        # Check cache
        result = executor.check_cache("Old query", "hybrid")

        assert result is None
        assert "Old query_hybrid" not in executor.cache
        assert executor._cache_misses == 1

    @pytest.mark.asyncio
    async def test_execute_prediction_local(self, executor, mock_lightrag_core):
        """Test executing a local query prediction."""
        prediction = ToolPrediction(
            prediction_id="pred_001",
            query="What is AI?",
            mode="local",
            confidence=0.9,
            pattern_matched="frequency",
            context_hash="abc123"
        )

        result = await executor._execute_prediction(prediction)

        assert result.query == "What is AI?"
        assert result.result == {"result": "local_data"}
        mock_lightrag_core.local_query.assert_called_once_with("What is AI?")

        # Check cache
        cache_key = "What is AI?_local"
        assert cache_key in executor.cache

    @pytest.mark.asyncio
    async def test_execute_prediction_global(self, executor, mock_lightrag_core):
        """Test executing a global query prediction."""
        prediction = ToolPrediction(
            prediction_id="pred_002",
            query="What is ML?",
            mode="global",
            confidence=0.85,
            pattern_matched="frequency",
            context_hash="def456"
        )

        result = await executor._execute_prediction(prediction)

        assert result.query == "What is ML?"
        assert result.result == {"result": "global_data"}
        mock_lightrag_core.global_query.assert_called_once_with("What is ML?")

    @pytest.mark.asyncio
    async def test_execute_prediction_hybrid(self, executor, mock_lightrag_core):
        """Test executing a hybrid query prediction."""
        prediction = ToolPrediction(
            prediction_id="pred_003",
            query="What is NLP?",
            mode="hybrid",
            confidence=0.95,
            pattern_matched="frequency",
            context_hash="ghi789"
        )

        result = await executor._execute_prediction(prediction)

        assert result.query == "What is NLP?"
        assert result.result == {"result": "hybrid_data"}
        mock_lightrag_core.hybrid_query.assert_called_once_with("What is NLP?")

    @pytest.mark.asyncio
    async def test_execute(self, executor):
        """Test full execute workflow."""
        # Add some history to generate predictions
        executor.config.min_confidence = 0.3
        for i in range(3):
            executor.record_call("What is AI?", "hybrid")

        result = await executor.execute(context="Tell me about AI")

        assert isinstance(result, SpeculativeExecutionResult)
        assert result.success
        assert len(result.predictions) > 0
        assert "cache_size" in result.metadata
        assert "cache_hit_rate" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_with_max_concurrent(self, executor):
        """Test that max_concurrent limit is respected."""
        executor.config.max_concurrent = 2
        executor.config.min_confidence = 0.1

        # Add diverse history to generate multiple predictions
        executor.record_call("Query 1", "hybrid")
        executor.record_call("Query 2", "local")
        executor.record_call("Query 3", "global")

        result = await executor.execute(context="Test context")

        assert result.success
        # Should have limited predictions
        assert len(result.predictions) <= 2

    def test_cleanup(self, executor):
        """Test cleanup of expired cache entries."""
        from datetime import timedelta

        # Add some cached results
        executor.cache["active_1"] = SpeculativeResult(
            prediction_id="pred_001",
            query="Active query 1",
            result={},
            cached_at=datetime.utcnow(),
            ttl_seconds=60.0
        )

        executor.cache["expired_1"] = SpeculativeResult(
            prediction_id="pred_002",
            query="Expired query 1",
            result={},
            cached_at=datetime.utcnow() - timedelta(seconds=120),
            ttl_seconds=60.0
        )

        executor.cache["expired_2"] = SpeculativeResult(
            prediction_id="pred_003",
            query="Expired query 2",
            result={},
            cached_at=datetime.utcnow() - timedelta(seconds=180),
            ttl_seconds=60.0
        )

        # Run cleanup
        executor.cleanup()

        # Only active entry should remain
        assert len(executor.cache) == 1
        assert "active_1" in executor.cache
        assert "expired_1" not in executor.cache
        assert "expired_2" not in executor.cache

    def test_get_stats(self, executor):
        """Test statistics collection."""
        # Execute some operations
        executor._total_predictions = 10
        executor._total_executions = 8
        executor._cache_hits = 5
        executor._cache_misses = 3

        stats = executor.get_stats()

        assert stats["total_predictions"] == 10
        assert stats["total_executions"] == 8
        assert stats["cache_hits"] == 5
        assert stats["cache_misses"] == 3
        assert stats["cache_hit_rate"] == 5 / 8  # 5 hits out of 8 total queries
        assert "cache_size" in stats
        assert "history_size" in stats
