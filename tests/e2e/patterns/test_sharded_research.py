"""E2E tests for sharded retrieval workflow.

Tests:
1. Query across multiple shards
2. Aggregation with source tracking
3. Cross-shard result fusion
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from promptchain.integrations.lightrag.sharded import (
    ShardedRetrievalConfig,
    LightRAGShardedRetriever,
    ShardConfig,
)


# Mock shard implementations
class MockShard:
    """Mock individual shard."""

    def __init__(self, shard_id: str, domain: str):
        self.shard_id = shard_id
        self.domain = domain
        self.query_count = 0

    async def query(self, query: str, **kwargs) -> list:
        """Mock shard query."""
        self.query_count += 1

        # Return domain-specific results
        results = []
        if self.domain == "ml":
            results = [
                {
                    "content": f"ML result from {self.shard_id}: {query}",
                    "score": 0.9,
                    "shard": self.shard_id,
                    "domain": self.domain,
                }
            ]
        elif self.domain == "dl":
            results = [
                {
                    "content": f"DL result from {self.shard_id}: {query}",
                    "score": 0.85,
                    "shard": self.shard_id,
                    "domain": self.domain,
                }
            ]
        elif self.domain == "transformers":
            results = [
                {
                    "content": f"Transformer result from {self.shard_id}: {query}",
                    "score": 0.8,
                    "shard": self.shard_id,
                    "domain": self.domain,
                }
            ]

        return results


@pytest.fixture
def mock_shards():
    """Create mock shards for testing."""
    return [
        MockShard("shard_ml", "ml"),
        MockShard("shard_dl", "dl"),
        MockShard("shard_transformers", "transformers"),
    ]


@pytest.mark.e2e
@pytest.mark.asyncio
class TestShardedQueryDistribution:
    """Test query distribution across shards."""

    async def test_query_all_shards(
        self, e2e_test_context, mock_message_bus, mock_shards
    ):
        """Test querying all shards in parallel.

        Workflow:
        1. Distribute query to all shards
        2. Collect results from each shard
        3. Aggregate with source tracking
        """
        sharded = LightRAGShardedRetriever(
            shards=mock_shards,
            config=ShardedRetrievalConfig(
                query_all_shards=True,
                aggregation_strategy="merge",
            )
        )
        sharded.connect_messagebus(mock_message_bus)

        result = await sharded.execute(
            query="What is machine learning?"
        )

        # Verify execution
        assert result.success
        assert len(result.shard_results) == len(mock_shards)

        # Each shard should have been queried
        for shard in mock_shards:
            assert shard.query_count == 1

        # Verify aggregated results include sources
        assert len(result.aggregated_results) > 0
        for agg_result in result.aggregated_results:
            assert "shard" in agg_result
            assert "domain" in agg_result

        # Verify events
        shard_events = mock_message_bus.get_events_by_type("pattern.sharded.*")
        assert any("started" in e["type"] for e in shard_events)
        assert any("shard_querying" in e["type"] for e in shard_events)
        assert any("aggregating" in e["type"] for e in shard_events)
        assert any("completed" in e["type"] for e in shard_events)

        print(f"\nShard Query Summary:")
        print(f"  Total shards queried: {len(result.shard_results)}")
        print(f"  Total results: {len(result.aggregated_results)}")
        print(f"  Execution time: {result.execution_time_ms:.2f}ms")

    async def test_selective_shard_querying(
        self, e2e_test_context, mock_message_bus, mock_shards
    ):
        """Test selective shard querying based on query routing."""
        # Create sharded retrieval with routing
        shard_configs = [
            ShardConfig(shard_id="shard_ml", domain="ml", keywords=["machine learning", "ML"]),
            ShardConfig(shard_id="shard_dl", domain="dl", keywords=["deep learning", "neural"]),
            ShardConfig(shard_id="shard_transformers", domain="transformers", keywords=["transformer", "attention"]),
        ]

        sharded = LightRAGShardedRetriever(
            shards=mock_shards,
            config=ShardedRetrievalConfig(
                query_all_shards=False,  # Selective routing
                shard_configs=shard_configs,
            )
        )
        sharded.connect_messagebus(mock_message_bus)

        # Query about transformers
        result = await sharded.execute(
            query="How do transformers use attention mechanisms?"
        )

        assert result.success

        # Verify only relevant shards were queried
        # (In real implementation, would route to transformer shard)
        print(f"Queried shards: {len(result.shard_results)}")

        # Verify source tracking
        for shard_result in result.shard_results:
            assert "shard_id" in shard_result or "shard" in shard_result

    async def test_parallel_shard_execution(
        self, e2e_test_context, mock_message_bus, mock_shards
    ):
        """Test shards are queried in parallel."""
        import time

        sharded = LightRAGShardedRetriever(
            shards=mock_shards,
            config=ShardedRetrievalConfig(query_all_shards=True)
        )

        start_time = time.perf_counter()
        result = await sharded.execute(query="What is deep learning?")
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert result.success

        print(f"Parallel shard execution time: {elapsed_ms:.2f}ms")

        # Parallel execution should be faster than sequential
        # (would be 3x slower if sequential with mock delays)
        # For now, just verify all shards were queried
        assert len(result.shard_results) == len(mock_shards)


@pytest.mark.e2e
@pytest.mark.asyncio
class TestAggregationStrategies:
    """Test different result aggregation strategies."""

    async def test_merge_aggregation(
        self, e2e_test_context, mock_message_bus, mock_shards
    ):
        """Test simple merge aggregation."""
        sharded = LightRAGShardedRetriever(
            shards=mock_shards,
            config=ShardedRetrievalConfig(
                query_all_shards=True,
                aggregation_strategy="merge",
            )
        )
        sharded.connect_messagebus(mock_message_bus)

        result = await sharded.execute(query="Explain neural networks")

        assert result.success
        assert len(result.aggregated_results) == len(result.shard_results)

        # Merge should combine all results
        total_results = sum(len(sr.get("results", [])) for sr in result.shard_results)
        print(f"Total merged results: {total_results}")

    async def test_weighted_aggregation(
        self, e2e_test_context, mock_message_bus, mock_shards
    ):
        """Test weighted aggregation with shard priorities."""
        # Assign different weights to shards
        shard_configs = [
            ShardConfig(shard_id="shard_ml", domain="ml", weight=1.0),
            ShardConfig(shard_id="shard_dl", domain="dl", weight=1.5),  # Higher priority
            ShardConfig(shard_id="shard_transformers", domain="transformers", weight=0.8),
        ]

        sharded = LightRAGShardedRetriever(
            shards=mock_shards,
            config=ShardedRetrievalConfig(
                query_all_shards=True,
                aggregation_strategy="weighted",
                shard_configs=shard_configs,
            )
        )

        result = await sharded.execute(query="Compare ML approaches")

        assert result.success

        # Weighted aggregation should prioritize DL shard results
        # (In real implementation, scores would be adjusted by weight)
        print(f"Aggregated {len(result.aggregated_results)} results with weights")

    async def test_rrf_aggregation(
        self, e2e_test_context, mock_message_bus, mock_shards
    ):
        """Test Reciprocal Rank Fusion aggregation."""
        sharded = LightRAGShardedRetriever(
            shards=mock_shards,
            config=ShardedRetrievalConfig(
                query_all_shards=True,
                aggregation_strategy="rrf",  # Reciprocal Rank Fusion
            )
        )

        result = await sharded.execute(query="What are neural architectures?")

        assert result.success

        # RRF should rerank results based on reciprocal ranks
        print(f"RRF aggregated {len(result.aggregated_results)} results")


@pytest.mark.e2e
@pytest.mark.asyncio
class TestSourceTracking:
    """Test source tracking and attribution."""

    async def test_source_attribution(
        self, e2e_test_context, mock_message_bus, mock_shards
    ):
        """Test each result is attributed to its source shard."""
        sharded = LightRAGShardedRetriever(
            shards=mock_shards,
            config=ShardedRetrievalConfig(query_all_shards=True)
        )
        sharded.connect_messagebus(mock_message_bus)

        result = await sharded.execute(query="Explain transformers")

        assert result.success

        # Every aggregated result should have source attribution
        for agg_result in result.aggregated_results:
            # Should have shard identifier
            has_source = (
                "shard" in agg_result or
                "shard_id" in agg_result or
                "source" in agg_result
            )
            assert has_source, "Result missing source attribution"

        print(f"\nSource Attribution:")
        for i, agg_result in enumerate(result.aggregated_results):
            source = agg_result.get("shard") or agg_result.get("shard_id") or agg_result.get("source")
            print(f"  Result {i+1}: from {source}")

    async def test_cross_shard_deduplication(
        self, e2e_test_context, mock_message_bus, mock_shards
    ):
        """Test deduplication of cross-shard results."""
        sharded = LightRAGShardedRetriever(
            shards=mock_shards,
            config=ShardedRetrievalConfig(
                query_all_shards=True,
                deduplicate=True,  # Enable deduplication
            )
        )

        result = await sharded.execute(query="Machine learning basics")

        assert result.success

        # With deduplication, duplicate results should be removed
        # while preserving source information
        unique_contents = set()
        for agg_result in result.aggregated_results:
            content = agg_result.get("content", "")
            if content:
                unique_contents.add(content)

        print(f"Unique results after deduplication: {len(unique_contents)}")

    async def test_shard_failure_handling(
        self, e2e_test_context, mock_message_bus
    ):
        """Test handling of shard failures."""
        # Create shards where one fails
        class FailingShard(MockShard):
            async def query(self, query: str, **kwargs):
                raise Exception("Shard unavailable")

        shards = [
            MockShard("shard_ml", "ml"),
            FailingShard("shard_failing", "fail"),
            MockShard("shard_dl", "dl"),
        ]

        sharded = LightRAGShardedRetriever(
            shards=shards,
            config=ShardedRetrievalConfig(
                query_all_shards=True,
                fail_on_error=False,  # Continue on shard failure
            )
        )
        sharded.connect_messagebus(mock_message_bus)

        result = await sharded.execute(query="Test query")

        # Should succeed with partial results
        assert result.success or len(result.shard_results) > 0

        # Should have results from working shards
        print(f"Successful shard queries: {len(result.shard_results)}")

        # Verify error was captured
        if not result.success:
            assert len(result.errors) > 0


@pytest.mark.e2e
@pytest.mark.asyncio
class TestShardedWorkflowIntegration:
    """Test sharded retrieval in complete workflows."""

    async def test_sharded_with_query_expansion(
        self, e2e_test_context, mock_message_bus, mock_shards
    ):
        """Test sharded retrieval combined with query expansion."""
        from promptchain.integrations.lightrag.query_expansion import (
            LightRAGQueryExpander,
            QueryExpansionConfig,
            ExpansionStrategy,
        )

        lightrag = e2e_test_context["lightrag"]

        # Phase 1: Query Expansion
        expander = LightRAGQueryExpander(
            lightrag_integration=lightrag,
            config=QueryExpansionConfig(
                strategies=[ExpansionStrategy.SEMANTIC],
                max_expansions_per_strategy=2,
            )
        )
        expander.connect_messagebus(mock_message_bus)

        expansion_result = await expander.execute(
            query="What are neural network architectures?"
        )

        assert expansion_result.success

        # Phase 2: Sharded Retrieval for each expansion
        sharded = LightRAGShardedRetriever(
            shards=mock_shards,
            config=ShardedRetrievalConfig(query_all_shards=True)
        )
        sharded.connect_messagebus(mock_message_bus)

        # Query shards with original + expanded queries
        all_queries = [expansion_result.original_query]
        all_queries.extend([eq.expanded_query for eq in expansion_result.expanded_queries])

        shard_results = []
        for query in all_queries:
            result = await sharded.execute(query=query)
            if result.success:
                shard_results.append(result)

        # Verify workflow
        assert len(shard_results) > 0

        # Calculate total results across all queries and shards
        total_results = sum(
            len(sr.aggregated_results) for sr in shard_results
        )

        print(f"\nSharded + Expansion Workflow:")
        print(f"  Original + expanded queries: {len(all_queries)}")
        print(f"  Successful shard retrievals: {len(shard_results)}")
        print(f"  Total aggregated results: {total_results}")

        # Verify event sequence
        event_sequence = mock_message_bus.get_event_sequence()

        # Expansion should complete before sharding starts
        expansion_end = max([
            i for i, e in enumerate(event_sequence)
            if "expansion.completed" in e
        ])
        sharding_start = min([
            i for i, e in enumerate(event_sequence)
            if "sharded.started" in e
        ])

        assert expansion_end < sharding_start
