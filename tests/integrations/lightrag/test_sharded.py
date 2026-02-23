"""Tests for LightRAG Sharded Retrieval Pattern."""

import asyncio
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from promptchain.integrations.lightrag.sharded import (
    ShardType,
    ShardConfig,
    ShardResult,
    ShardedRetrievalConfig,
    ShardedRetrievalResult,
    LightRAGShardRegistry,
    LightRAGShardedRetriever,
)


@dataclass
class MockQueryResult:
    """Mock result object with score."""
    content: str
    score: float


@pytest.fixture(scope="function")
def mock_hybridrag():
    """Mock the hybridrag module for tests."""
    # Create mock module structure
    mock_lightrag_core = MagicMock()
    mock_lightrag_core_module = MagicMock()
    mock_lightrag_core_module.HybridLightRAGCore = MagicMock

    mock_src = MagicMock()
    mock_src.lightrag_core = mock_lightrag_core_module

    mock_hybridrag = MagicMock()
    mock_hybridrag.src = mock_src

    # Inject into sys.modules
    sys.modules['hybridrag'] = mock_hybridrag
    sys.modules['hybridrag.src'] = mock_src
    sys.modules['hybridrag.src.lightrag_core'] = mock_lightrag_core_module

    yield mock_lightrag_core_module

    # Clean up
    sys.modules.pop('hybridrag', None)
    sys.modules.pop('hybridrag.src', None)
    sys.modules.pop('hybridrag.src.lightrag_core', None)


class TestShardConfig:
    """Test ShardConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ShardConfig(
            shard_id="test_shard",
            shard_type=ShardType.LIGHTRAG
        )

        assert config.shard_id == "test_shard"
        assert config.shard_type == ShardType.LIGHTRAG
        assert config.working_dir == ""
        assert config.connection_config is None
        assert config.priority == 0
        assert config.timeout_seconds == 10.0
        assert config.enabled is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ShardConfig(
            shard_id="custom_shard",
            shard_type=ShardType.VECTOR_DB,
            working_dir="/data/shard1",
            connection_config={"host": "localhost"},
            priority=5,
            timeout_seconds=20.0,
            enabled=False
        )

        assert config.shard_id == "custom_shard"
        assert config.shard_type == ShardType.VECTOR_DB
        assert config.working_dir == "/data/shard1"
        assert config.connection_config == {"host": "localhost"}
        assert config.priority == 5
        assert config.timeout_seconds == 20.0
        assert config.enabled is False


class TestShardResult:
    """Test ShardResult dataclass."""

    def test_successful_result(self):
        """Test successful shard result."""
        result = ShardResult(
            shard_id="shard1",
            results=[{"content": "test"}],
            query_time_ms=100.0
        )

        assert result.shard_id == "shard1"
        assert len(result.results) == 1
        assert result.query_time_ms == 100.0
        assert result.error is None
        assert result.success is True

    def test_failed_result(self):
        """Test failed shard result."""
        result = ShardResult(
            shard_id="shard2",
            results=[],
            query_time_ms=50.0,
            error="Connection timeout"
        )

        assert result.shard_id == "shard2"
        assert len(result.results) == 0
        assert result.error == "Connection timeout"
        assert result.success is False


class TestLightRAGShardRegistry:
    """Test LightRAGShardRegistry."""

    def test_init(self):
        """Test registry initialization."""
        registry = LightRAGShardRegistry()

        assert len(registry.list_shards()) == 0
        assert registry.health_check() == {}

    def test_register_non_lightrag_shard(self):
        """Test registering non-LIGHTRAG shard."""
        registry = LightRAGShardRegistry()

        config = ShardConfig(
            shard_id="vector_shard",
            shard_type=ShardType.VECTOR_DB,
            connection_config={"host": "localhost"}
        )

        registry.register_shard(config)

        assert "vector_shard" in registry.list_shards()
        assert registry.get_shard("vector_shard") == {"host": "localhost"}
        assert registry.get_config("vector_shard") == config

    def test_register_duplicate_shard(self):
        """Test registering duplicate shard raises error."""
        registry = LightRAGShardRegistry()

        config = ShardConfig(
            shard_id="duplicate",
            shard_type=ShardType.API
        )

        registry.register_shard(config)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_shard(config)

    def test_get_nonexistent_shard(self):
        """Test getting non-existent shard raises error."""
        registry = LightRAGShardRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get_shard("nonexistent")

    def test_health_check(self):
        """Test health check for multiple shards."""
        registry = LightRAGShardRegistry()

        # Register enabled shard
        registry.register_shard(ShardConfig(
            shard_id="enabled",
            shard_type=ShardType.API,
            enabled=True
        ))

        # Register disabled shard
        registry.register_shard(ShardConfig(
            shard_id="disabled",
            shard_type=ShardType.API,
            enabled=False
        ))

        health = registry.health_check()

        assert health["enabled"] is True
        assert health["disabled"] is False

    @patch("promptchain.integrations.lightrag.sharded.LIGHTRAG_AVAILABLE", True)
    def test_register_lightrag_shard(self, mock_hybridrag):
        """Test registering LIGHTRAG shard."""
        registry = LightRAGShardRegistry()

        mock_core = MagicMock()
        mock_hybridrag.HybridLightRAGCore = MagicMock(return_value=mock_core)

        config = ShardConfig(
            shard_id="lightrag_shard",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test_data",
            connection_config={"model_name": "gpt-4"}
        )

        registry.register_shard(config)

        assert "lightrag_shard" in registry.list_shards()
        assert registry.get_shard("lightrag_shard") == mock_core
        mock_hybridrag.HybridLightRAGCore.assert_called_once_with(
            working_dir="./test_data",
            model_name="gpt-4"
        )

    @patch("promptchain.integrations.lightrag.sharded.LIGHTRAG_AVAILABLE", False)
    def test_register_lightrag_unavailable(self):
        """Test registering LIGHTRAG shard when unavailable."""
        registry = LightRAGShardRegistry()

        config = ShardConfig(
            shard_id="lightrag_shard",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test_data"
        )

        with pytest.raises(ImportError, match="hybridrag is not installed"):
            registry.register_shard(config)


class TestLightRAGShardedRetriever:
    """Test LightRAGShardedRetriever pattern."""

    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        return LightRAGShardRegistry()

    @pytest.fixture
    def retriever(self, registry):
        """Create a test retriever."""
        return LightRAGShardedRetriever(
            registry=registry,
            config=ShardedRetrievalConfig(
                parallel=True,
                aggregate_top_k=5
            )
        )

    @pytest.mark.asyncio
    async def test_execute_no_shards(self, retriever):
        """Test execution with no registered shards."""
        result = await retriever.execute(query="test query")

        assert result.success is False
        assert "No enabled shards available" in result.errors
        assert result.shards_queried == 0

    @pytest.mark.asyncio
    @patch("promptchain.integrations.lightrag.sharded.LIGHTRAG_AVAILABLE", True)
    async def test_execute_single_shard(self, mock_hybridrag, retriever, registry):
        """Test execution with single shard."""
        # Setup mock shard
        mock_core = MagicMock()
        mock_core.hybrid_query = AsyncMock(return_value=[
            MockQueryResult(content="result1", score=0.9),
            MockQueryResult(content="result2", score=0.8)
        ])
        mock_hybridrag.HybridLightRAGCore = MagicMock(return_value=mock_core)

        registry.register_shard(ShardConfig(
            shard_id="shard1",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test"
        ))

        result = await retriever.execute(query="test query")

        assert result.success is True
        assert result.shards_queried == 1
        assert result.shards_failed == 0
        assert len(result.aggregated_results) == 2
        mock_core.hybrid_query.assert_called_once()

    @pytest.mark.asyncio
    @patch("promptchain.integrations.lightrag.sharded.LIGHTRAG_AVAILABLE", True)
    async def test_execute_parallel_shards(self, mock_hybridrag, retriever, registry):
        """Test parallel execution with multiple shards."""
        # Setup mock shards
        mock_core1 = MagicMock()
        mock_core1.hybrid_query = AsyncMock(return_value=[
            MockQueryResult(content="shard1_result", score=0.9)
        ])

        mock_core2 = MagicMock()
        mock_core2.hybrid_query = AsyncMock(return_value=[
            MockQueryResult(content="shard2_result", score=0.95)
        ])

        mock_hybridrag.HybridLightRAGCore = MagicMock(side_effect=[mock_core1, mock_core2])

        registry.register_shard(ShardConfig(
            shard_id="shard1",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test1"
        ))
        registry.register_shard(ShardConfig(
            shard_id="shard2",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test2"
        ))

        result = await retriever.execute(query="test query")

        assert result.success is True
        assert result.shards_queried == 2
        assert result.shards_failed == 0
        assert len(result.aggregated_results) == 2

    @pytest.mark.asyncio
    @patch("promptchain.integrations.lightrag.sharded.LIGHTRAG_AVAILABLE", True)
    async def test_execute_with_timeout(self, mock_hybridrag, retriever, registry):
        """Test handling of shard timeout."""
        # Setup mock shard that times out
        mock_core = MagicMock()

        async def slow_query(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return []

        mock_core.hybrid_query = slow_query
        mock_hybridrag.HybridLightRAGCore = MagicMock(return_value=mock_core)

        registry.register_shard(ShardConfig(
            shard_id="slow_shard",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test",
            timeout_seconds=0.1  # Very short timeout
        ))

        result = await retriever.execute(query="test query")

        # With fail_partial=True (default), should still succeed
        assert result.success is True
        assert result.shards_queried == 1
        assert result.shards_failed == 1
        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    @patch("promptchain.integrations.lightrag.sharded.LIGHTRAG_AVAILABLE", True)
    async def test_execute_with_error(self, mock_hybridrag, retriever, registry):
        """Test handling of shard error."""
        # Setup mock shard that raises error
        mock_core = MagicMock()
        mock_core.hybrid_query = AsyncMock(side_effect=Exception("Query failed"))
        mock_hybridrag.HybridLightRAGCore = MagicMock(return_value=mock_core)

        registry.register_shard(ShardConfig(
            shard_id="error_shard",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test"
        ))

        result = await retriever.execute(query="test query")

        assert result.success is True  # With fail_partial=True
        assert result.shards_failed == 1

    @pytest.mark.asyncio
    @patch("promptchain.integrations.lightrag.sharded.LIGHTRAG_AVAILABLE", True)
    async def test_execute_fail_partial_false(self, mock_hybridrag, retriever, registry):
        """Test fail_partial=False fails on any shard error."""
        retriever.config.fail_partial = False

        # Setup mock shard that raises error
        mock_core = MagicMock()
        mock_core.hybrid_query = AsyncMock(side_effect=Exception("Query failed"))
        mock_hybridrag.HybridLightRAGCore = MagicMock(return_value=mock_core)

        registry.register_shard(ShardConfig(
            shard_id="error_shard",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test"
        ))

        result = await retriever.execute(query="test query")

        assert result.success is False
        assert "shard(s) failed" in result.errors[0]

    @pytest.mark.asyncio
    @patch("promptchain.integrations.lightrag.sharded.LIGHTRAG_AVAILABLE", True)
    async def test_priority_weighting(self, mock_hybridrag, retriever, registry):
        """Test priority weighting in result aggregation."""
        # Setup two shards with different priorities
        mock_core1 = MagicMock()
        mock_core1.hybrid_query = AsyncMock(return_value=[
            MockQueryResult(content="low_priority", score=0.8)
        ])

        mock_core2 = MagicMock()
        mock_core2.hybrid_query = AsyncMock(return_value=[
            MockQueryResult(content="high_priority", score=0.8)
        ])

        mock_hybridrag.HybridLightRAGCore = MagicMock(side_effect=[mock_core1, mock_core2])

        # Low priority shard
        registry.register_shard(ShardConfig(
            shard_id="shard1",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test1",
            priority=0
        ))

        # High priority shard
        registry.register_shard(ShardConfig(
            shard_id="shard2",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test2",
            priority=10
        ))

        result = await retriever.execute(query="test query")

        # High priority result should come first
        assert result.aggregated_results[0].content == "high_priority"

    @pytest.mark.asyncio
    @patch("promptchain.integrations.lightrag.sharded.LIGHTRAG_AVAILABLE", True)
    async def test_score_normalization(self, mock_hybridrag, retriever, registry):
        """Test score normalization across shards."""
        retriever.config.normalize_scores = True

        mock_core = MagicMock()
        mock_core.hybrid_query = AsyncMock(return_value=[
            MockQueryResult(content="result1", score=10.0),
            MockQueryResult(content="result2", score=5.0)
        ])
        mock_hybridrag.HybridLightRAGCore = MagicMock(return_value=mock_core)

        registry.register_shard(ShardConfig(
            shard_id="shard1",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test"
        ))

        result = await retriever.execute(query="test query")

        assert result.success is True
        # Scores should be normalized, but results should still be ordered correctly

    @pytest.mark.asyncio
    @patch("promptchain.integrations.lightrag.sharded.LIGHTRAG_AVAILABLE", True)
    async def test_aggregate_top_k(self, mock_hybridrag, retriever, registry):
        """Test aggregate_top_k limits results."""
        retriever.config.aggregate_top_k = 2

        mock_core = MagicMock()
        mock_core.hybrid_query = AsyncMock(return_value=[
            MockQueryResult(content="result1", score=0.9),
            MockQueryResult(content="result2", score=0.8),
            MockQueryResult(content="result3", score=0.7)
        ])
        mock_hybridrag.HybridLightRAGCore = MagicMock(return_value=mock_core)

        registry.register_shard(ShardConfig(
            shard_id="shard1",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test"
        ))

        result = await retriever.execute(query="test query")

        assert len(result.aggregated_results) == 2

    @pytest.mark.asyncio
    @patch("promptchain.integrations.lightrag.sharded.LIGHTRAG_AVAILABLE", True)
    async def test_specific_shard_ids(self, mock_hybridrag, retriever, registry):
        """Test querying specific shard IDs."""
        mock_core1 = MagicMock()
        mock_core1.hybrid_query = AsyncMock(return_value=[
            MockQueryResult(content="shard1", score=0.9)
        ])

        mock_core2 = MagicMock()
        mock_core2.hybrid_query = AsyncMock(return_value=[
            MockQueryResult(content="shard2", score=0.9)
        ])

        mock_hybridrag.HybridLightRAGCore = MagicMock(side_effect=[mock_core1, mock_core2])

        registry.register_shard(ShardConfig(
            shard_id="shard1",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test1"
        ))
        registry.register_shard(ShardConfig(
            shard_id="shard2",
            shard_type=ShardType.LIGHTRAG,
            working_dir="./test2"
        ))

        # Query only shard1
        result = await retriever.execute(query="test", shard_ids=["shard1"])

        assert result.shards_queried == 1
        assert len(result.aggregated_results) == 1
        assert result.aggregated_results[0].content == "shard1"

    def test_extract_score_from_object(self, retriever):
        """Test score extraction from various result formats."""
        # Object with score attribute
        result_obj = MockQueryResult(content="test", score=0.9)
        assert retriever._extract_score(result_obj) == 0.9

        # Dict with score key
        result_dict = {"content": "test", "score": 0.8}
        assert retriever._extract_score(result_dict) == 0.8

        # Dict with relevance key
        result_relevance = {"content": "test", "relevance": 0.7}
        assert retriever._extract_score(result_relevance) == 0.7

        # No score available
        result_no_score = {"content": "test"}
        assert retriever._extract_score(result_no_score) == 1.0

    @pytest.mark.asyncio
    async def test_event_emission(self, retriever, registry):
        """Test event emission during execution."""
        events = []

        def handler(event_type, data):
            events.append((event_type, data))

        retriever.add_event_handler(handler)

        # Register a non-LIGHTRAG shard to avoid mocking
        registry.register_shard(ShardConfig(
            shard_id="api_shard",
            shard_type=ShardType.API
        ))

        await retriever.execute(query="test")

        # Check events were emitted
        event_types = [e[0] for e in events]
        assert "pattern.sharded.started" in event_types
        assert "pattern.sharded.completed" in event_types
