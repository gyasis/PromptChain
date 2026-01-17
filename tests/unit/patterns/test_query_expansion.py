"""Unit tests for LightRAGQueryExpander pattern adapter."""

import pytest
from unittest.mock import AsyncMock, patch
from promptchain.integrations.lightrag.query_expansion import (
    LightRAGQueryExpander,
    ExpansionStrategy
)


@pytest.mark.asyncio
class TestLightRAGQueryExpander:
    """Test suite for query expansion pattern."""

    async def test_initialization(self, mock_search_interface):
        """Test pattern initializes correctly."""
        pattern = LightRAGQueryExpander(
            search_interface=mock_search_interface,
            expansion_strategy=ExpansionStrategy.SYNONYM,
            num_expansions=3
        )

        assert pattern.search_interface == mock_search_interface
        assert pattern.expansion_strategy == ExpansionStrategy.SYNONYM
        assert pattern.num_expansions == 3
        assert pattern.pattern_name == "query_expansion"

    async def test_synonym_expansion_strategy(self, mock_search_interface):
        """Test synonym-based query expansion."""
        pattern = LightRAGQueryExpander(
            search_interface=mock_search_interface,
            expansion_strategy=ExpansionStrategy.SYNONYM,
            num_expansions=3
        )

        with patch.object(pattern, '_expand_synonyms', new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = [
                "original query",
                "synonym query 1",
                "synonym query 2"
            ]

            result = await pattern.execute(query="test query")

            mock_expand.assert_called_once_with("test query", 3)
            assert "expanded_queries" in result

    async def test_semantic_expansion_strategy(self, mock_search_interface):
        """Test semantic-based query expansion."""
        pattern = LightRAGQueryExpander(
            search_interface=mock_search_interface,
            expansion_strategy=ExpansionStrategy.SEMANTIC,
            num_expansions=5
        )

        with patch.object(pattern, '_expand_semantic', new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = [
                "original query",
                "semantic variant 1",
                "semantic variant 2",
                "semantic variant 3",
                "semantic variant 4"
            ]

            result = await pattern.execute(query="machine learning")

            mock_expand.assert_called_once_with("machine learning", 5)
            assert len(result["expanded_queries"]) == 5

    async def test_acronym_expansion_strategy(self, mock_search_interface):
        """Test acronym expansion strategy."""
        pattern = LightRAGQueryExpander(
            search_interface=mock_search_interface,
            expansion_strategy=ExpansionStrategy.ACRONYM,
            num_expansions=2
        )

        with patch.object(pattern, '_expand_acronyms', new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = [
                "AI artificial intelligence",
                "AI automated intelligence"
            ]

            result = await pattern.execute(query="AI applications")

            mock_expand.assert_called_once()
            assert "expanded_queries" in result

    async def test_multi_query_search_delegation(self, mock_search_interface):
        """Test delegation to multi_query_search method."""
        pattern = LightRAGQueryExpander(
            search_interface=mock_search_interface,
            expansion_strategy=ExpansionStrategy.SYNONYM,
            num_expansions=3
        )

        expanded = ["query1", "query2", "query3"]
        with patch.object(pattern, '_expand_synonyms', new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = expanded

            result = await pattern.execute(query="test")

            mock_search_interface.multi_query_search.assert_called_once()
            call_args = mock_search_interface.multi_query_search.call_args
            assert call_args[1]["queries"] == expanded

    async def test_result_deduplication(self, mock_search_interface, sample_search_results):
        """Test deduplication of search results from multiple queries."""
        pattern = LightRAGQueryExpander(
            search_interface=mock_search_interface,
            expansion_strategy=ExpansionStrategy.SYNONYM,
            num_expansions=3
        )

        # Mock duplicate results
        duplicate_results = sample_search_results + [sample_search_results[0]]
        mock_search_interface.multi_query_search.return_value = {
            "results": duplicate_results
        }

        result = await pattern.execute(query="test")

        # Check deduplication by ID
        unique_ids = set(r["id"] for r in result["results"])
        assert len(unique_ids) == len(sample_search_results)

    async def test_result_ranking_preservation(self, mock_search_interface):
        """Test that result ranking is preserved after expansion."""
        pattern = LightRAGQueryExpander(
            search_interface=mock_search_interface,
            expansion_strategy=ExpansionStrategy.SEMANTIC,
            num_expansions=2
        )

        ranked_results = [
            {"text": "best", "score": 0.95, "id": "1"},
            {"text": "good", "score": 0.85, "id": "2"},
            {"text": "ok", "score": 0.75, "id": "3"}
        ]

        mock_search_interface.multi_query_search.return_value = {
            "results": ranked_results
        }

        result = await pattern.execute(query="test")

        # Verify scores are descending
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    async def test_expansion_count_validation(self, mock_search_interface):
        """Test validation of num_expansions parameter."""
        with pytest.raises(ValueError, match="num_expansions must be at least 1"):
            LightRAGQueryExpander(
                search_interface=mock_search_interface,
                expansion_strategy=ExpansionStrategy.SYNONYM,
                num_expansions=0
            )

    async def test_empty_expansion_handling(self, mock_search_interface):
        """Test handling of empty expansion results."""
        pattern = LightRAGQueryExpander(
            search_interface=mock_search_interface,
            expansion_strategy=ExpansionStrategy.SYNONYM,
            num_expansions=3
        )

        with patch.object(pattern, '_expand_synonyms', new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = []

            result = await pattern.execute(query="test")

            # Should fall back to original query
            assert "test" in str(result.get("expanded_queries", []))

    async def test_event_emission_expansion_complete(self, mock_search_interface, event_collector):
        """Test events emitted when expansion completes."""
        pattern = LightRAGQueryExpander(
            search_interface=mock_search_interface,
            expansion_strategy=ExpansionStrategy.SYNONYM,
            num_expansions=3
        )

        pattern.emit_event = event_collector.collect

        await pattern.execute(query="test")

        events = event_collector.get_events("pattern.expansion.complete")
        assert len(events) > 0

    async def test_metadata_tracking(self, mock_search_interface):
        """Test that expansion metadata is tracked correctly."""
        pattern = LightRAGQueryExpander(
            search_interface=mock_search_interface,
            expansion_strategy=ExpansionStrategy.SEMANTIC,
            num_expansions=4
        )

        result = await pattern.execute(query="test query")

        assert "metadata" in result
        metadata = result["metadata"]
        assert "expansion_strategy" in metadata
        assert metadata["expansion_strategy"] == "SEMANTIC"
        assert "num_expansions" in metadata

    async def test_base_pattern_interface_compliance(self, mock_search_interface):
        """Test that pattern implements BasePattern interface correctly."""
        pattern = LightRAGQueryExpander(
            search_interface=mock_search_interface,
            expansion_strategy=ExpansionStrategy.SYNONYM,
            num_expansions=3
        )

        assert hasattr(pattern, 'execute')
        assert hasattr(pattern, 'pattern_name')
        assert callable(pattern.execute)

        result = await pattern.execute(query="test")
        assert isinstance(result, dict)

    async def test_all_expansion_strategies(self, mock_search_interface):
        """Test all expansion strategies are valid."""
        strategies = [
            ExpansionStrategy.SYNONYM,
            ExpansionStrategy.SEMANTIC,
            ExpansionStrategy.ACRONYM
        ]

        for strategy in strategies:
            pattern = LightRAGQueryExpander(
                search_interface=mock_search_interface,
                expansion_strategy=strategy,
                num_expansions=2
            )

            result = await pattern.execute(query="test")
            assert isinstance(result, dict)
            assert "results" in result or "expanded_queries" in result
