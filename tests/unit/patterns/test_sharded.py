"""Unit tests for LightRAGShardedRetriever and LightRAGShardRegistry."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from promptchain.integrations.lightrag.sharded import (
    LightRAGShardedRetriever,
    LightRAGShardRegistry,
    ShardConfig,
    ShardType,
    ShardedRetrievalConfig
)


@pytest.mark.asyncio
class TestLightRAGShardRegistry:
    """Test suite for LightRAGShardRegistry."""

    def test_registry_initialization(self):
        """Test registry initializes empty."""
        registry = LightRAGShardRegistry()
        assert len(registry.list_shards()) == 0

    def test_shard_registration_non_lightrag(self):
        """Test registering non-LIGHTRAG shards."""
        registry = LightRAGShardRegistry()

        config = ShardConfig(
            shard_id="shard1",
            shard_type=ShardType.VECTOR_DB,
            connection_config={"host": "localhost"}
        )

        registry.register_shard(config)

        assert len(registry.list_shards()) == 1
        assert "shard1" in registry.list_shards()

    def test_shard_retrieval(self):
        """Test retrieving shard by ID."""
        registry = LightRAGShardRegistry()

        config = ShardConfig(
            shard_id="test_shard",
            shard_type=ShardType.API,
            connection_config={"url": "http://test"}
        )

        registry.register_shard(config)
        retrieved_config = registry.get_config("test_shard")

        assert retrieved_config.shard_id == "test_shard"

    def test_shard_not_found(self):
        """Test retrieving non-existent shard."""
        registry = LightRAGShardRegistry()

        with pytest.raises(KeyError):
            registry.get_shard("nonexistent")

    def test_multiple_shards(self):
        """Test managing multiple shards."""
        registry = LightRAGShardRegistry()

        configs = [
            ShardConfig(
                shard_id=f"shard{i}",
                shard_type=ShardType.VECTOR_DB,
                connection_config={"index": i}
            )
            for i in range(5)
        ]

        for config in configs:
            registry.register_shard(config)

        assert len(registry.list_shards()) == 5

        for i in range(5):
            config = registry.get_config(f"shard{i}")
            assert config.connection_config["index"] == i

    def test_duplicate_registration(self):
        """Test registering shard with duplicate ID raises error."""
        registry = LightRAGShardRegistry()

        config1 = ShardConfig(shard_id="dup", shard_type=ShardType.API)
        config2 = ShardConfig(shard_id="dup", shard_type=ShardType.API)

        registry.register_shard(config1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_shard(config2)

    def test_health_check(self):
        """Test health check returns correct status."""
        registry = LightRAGShardRegistry()

        config_enabled = ShardConfig(
            shard_id="enabled",
            shard_type=ShardType.API,
            enabled=True
        )
        config_disabled = ShardConfig(
            shard_id="disabled",
            shard_type=ShardType.API,
            enabled=False
        )

        registry.register_shard(config_enabled)
        registry.register_shard(config_disabled)

        health = registry.health_check()

        assert health["enabled"] is True
        assert health["disabled"] is False


@pytest.mark.asyncio
class TestLightRAGShardedRetriever:
    """Test suite for sharded retriever pattern."""

    async def test_initialization(self):
        """Test pattern initializes correctly."""
        registry = LightRAGShardRegistry()
        config = ShardedRetrievalConfig(parallel=True)
        pattern = LightRAGShardedRetriever(registry=registry, config=config)

        assert pattern.registry == registry
        assert pattern.config.parallel is True

    async def test_parallel_query_execution(self, mock_lightrag_core):
        """Test parallel execution across all shards."""
        registry = LightRAGShardRegistry()

        # Register mock shards (non-LIGHTRAG to avoid import issues)
        config1 = ShardConfig(
            shard_id="shard1",
            shard_type=ShardType.VECTOR_DB,
            connection_config={"topic": "science"}
        )
        config2 = ShardConfig(
            shard_id="shard2",
            shard_type=ShardType.VECTOR_DB,
            connection_config={"topic": "history"}
        )

        registry.register_shard(config1)
        registry.register_shard(config2)

        pattern = LightRAGShardedRetriever(
            registry=registry,
            config=ShardedRetrievalConfig(parallel=True)
        )

        result = await pattern.execute(query="test query")

        # Should query both shards
        assert result.shards_queried == 2
        assert result.success

    async def test_empty_shard_registry(self):
        """Test handling of empty shard registry."""
        registry = LightRAGShardRegistry()

        pattern = LightRAGShardedRetriever(
            registry=registry,
            config=ShardedRetrievalConfig()
        )

        result = await pattern.execute(query="test")

        assert result.success is False
        assert result.shards_queried == 0
        assert "No enabled shards" in str(result.errors)

    async def test_event_emission_started(self, event_collector):
        """Test events emitted when query starts."""
        registry = LightRAGShardRegistry()
        config = ShardConfig(
            shard_id="s1",
            shard_type=ShardType.API,
            enabled=True
        )
        registry.register_shard(config)

        pattern = LightRAGShardedRetriever(
            registry=registry,
            config=ShardedRetrievalConfig()
        )
        pattern.emit_event = event_collector.collect

        await pattern.execute(query="test")

        events = event_collector.get_events("pattern.sharded.started")
        assert len(events) > 0

    async def test_event_emission_completed(self, event_collector):
        """Test events emitted after completion."""
        registry = LightRAGShardRegistry()
        config = ShardConfig(shard_id="s1", shard_type=ShardType.API)
        registry.register_shard(config)

        pattern = LightRAGShardedRetriever(
            registry=registry,
            config=ShardedRetrievalConfig()
        )
        pattern.emit_event = event_collector.collect

        await pattern.execute(query="test")

        events = event_collector.get_events("pattern.sharded.completed")
        assert len(events) > 0

    async def test_base_pattern_interface_compliance(self):
        """Test that pattern implements BasePattern interface correctly."""
        registry = LightRAGShardRegistry()
        pattern = LightRAGShardedRetriever(
            registry=registry,
            config=ShardedRetrievalConfig()
        )

        assert hasattr(pattern, 'execute')
        assert callable(pattern.execute)

        result = await pattern.execute(query="test")
        assert hasattr(result, 'success')
        assert hasattr(result, 'execution_time_ms')
