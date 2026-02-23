"""Unit tests for LightRAGHybridSearcher pattern adapter."""

import pytest
from unittest.mock import AsyncMock, patch
from promptchain.integrations.lightrag.hybrid_search import (
    LightRAGHybridSearcher,
    FusionAlgorithm
)


@pytest.mark.asyncio
class TestLightRAGHybridSearcher:
    """Test suite for hybrid search pattern."""

    async def test_initialization(self, mock_lightrag_core):
        """Test pattern initializes correctly."""
        pattern = LightRAGHybridSearcher(
            lightrag_core=mock_lightrag_core,
            fusion_algorithm=FusionAlgorithm.RRF,
            techniques=["local", "global"]
        )

        assert pattern.lightrag_core == mock_lightrag_core
        assert pattern.fusion_algorithm == FusionAlgorithm.RRF
        assert pattern.techniques == ["local", "global"]
        assert pattern.pattern_name == "hybrid_search"

    async def test_rrf_fusion_algorithm(self, mock_lightrag_core):
        """Test Reciprocal Rank Fusion algorithm."""
        pattern = LightRAGHybridSearcher(
            lightrag_core=mock_lightrag_core,
            fusion_algorithm=FusionAlgorithm.RRF,
            techniques=["local", "global"]
        )

        # Mock different results from each technique
        mock_lightrag_core.local_query.return_value = {
            "results": [
                {"text": "doc1", "score": 0.9, "id": "1"},
                {"text": "doc2", "score": 0.8, "id": "2"}
            ]
        }

        mock_lightrag_core.global_query.return_value = {
            "results": [
                {"text": "doc2", "score": 0.85, "id": "2"},
                {"text": "doc3", "score": 0.7, "id": "3"}
            ]
        }

        result = await pattern.execute(query="test query")

        # RRF should combine rankings
        assert "results" in result
        assert len(result["results"]) >= 2

        # doc2 appears in both, should rank high
        doc2 = next((r for r in result["results"] if r["id"] == "2"), None)
        assert doc2 is not None

    async def test_linear_fusion_algorithm(self, mock_lightrag_core):
        """Test linear score combination."""
        pattern = LightRAGHybridSearcher(
            lightrag_core=mock_lightrag_core,
            fusion_algorithm=FusionAlgorithm.LINEAR,
            techniques=["local", "global"],
            weights={"local": 0.6, "global": 0.4}
        )

        mock_lightrag_core.local_query.return_value = {
            "results": [{"text": "doc1", "score": 1.0, "id": "1"}]
        }

        mock_lightrag_core.global_query.return_value = {
            "results": [{"text": "doc1", "score": 0.5, "id": "1"}]
        }

        result = await pattern.execute(query="test")

        # Linear: 0.6 * 1.0 + 0.4 * 0.5 = 0.8
        doc1 = result["results"][0]
        assert abs(doc1["score"] - 0.8) < 0.01

    async def test_borda_fusion_algorithm(self, mock_lightrag_core):
        """Test Borda count fusion."""
        pattern = LightRAGHybridSearcher(
            lightrag_core=mock_lightrag_core,
            fusion_algorithm=FusionAlgorithm.BORDA,
            techniques=["local", "global"]
        )

        mock_lightrag_core.local_query.return_value = {
            "results": [
                {"text": "doc1", "score": 0.9, "id": "1"},
                {"text": "doc2", "score": 0.8, "id": "2"},
                {"text": "doc3", "score": 0.7, "id": "3"}
            ]
        }

        mock_lightrag_core.global_query.return_value = {
            "results": [
                {"text": "doc3", "score": 0.95, "id": "3"},
                {"text": "doc1", "score": 0.85, "id": "1"},
                {"text": "doc2", "score": 0.75, "id": "2"}
            ]
        }

        result = await pattern.execute(query="test")

        # Borda assigns points based on rank position
        assert "results" in result
        assert len(result["results"]) == 3

    async def test_technique_contribution_tracking(self, mock_lightrag_core):
        """Test tracking which techniques contributed to results."""
        pattern = LightRAGHybridSearcher(
            lightrag_core=mock_lightrag_core,
            fusion_algorithm=FusionAlgorithm.RRF,
            techniques=["local", "global", "naive"]
        )

        result = await pattern.execute(query="test")

        metadata = result.get("metadata", {})
        assert "technique_contributions" in metadata

        contributions = metadata["technique_contributions"]
        assert "local" in contributions
        assert "global" in contributions
        assert "naive" in contributions

    async def test_all_techniques_queried(self, mock_lightrag_core):
        """Test that all specified techniques are queried."""
        pattern = LightRAGHybridSearcher(
            lightrag_core=mock_lightrag_core,
            fusion_algorithm=FusionAlgorithm.RRF,
            techniques=["local", "global", "naive"]
        )

        await pattern.execute(query="test")

        mock_lightrag_core.local_query.assert_called_once()
        mock_lightrag_core.global_query.assert_called_once()
        mock_lightrag_core.naive_query.assert_called_once()

    async def test_partial_technique_failure(self, mock_lightrag_core):
        """Test handling when one technique fails."""
        pattern = LightRAGHybridSearcher(
            lightrag_core=mock_lightrag_core,
            fusion_algorithm=FusionAlgorithm.RRF,
            techniques=["local", "global"]
        )

        # Local succeeds
        mock_lightrag_core.local_query.return_value = {
            "results": [{"text": "doc1", "score": 0.9, "id": "1"}]
        }

        # Global fails
        mock_lightrag_core.global_query.side_effect = Exception("Query failed")

        result = await pattern.execute(query="test")

        # Should still return results from local
        assert "results" in result
        assert len(result["results"]) > 0

    async def test_weight_validation(self, mock_lightrag_core):
        """Test validation of technique weights."""
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            LightRAGHybridSearcher(
                lightrag_core=mock_lightrag_core,
                fusion_algorithm=FusionAlgorithm.LINEAR,
                techniques=["local", "global"],
                weights={"local": 0.7, "global": 0.5}  # Sum > 1.0
            )

    async def test_invalid_technique(self, mock_lightrag_core):
        """Test error handling for invalid technique."""
        with pytest.raises(ValueError, match="Invalid technique"):
            LightRAGHybridSearcher(
                lightrag_core=mock_lightrag_core,
                fusion_algorithm=FusionAlgorithm.RRF,
                techniques=["local", "invalid_technique"]
            )

    async def test_empty_results_handling(self, mock_lightrag_core):
        """Test handling when all techniques return empty results."""
        pattern = LightRAGHybridSearcher(
            lightrag_core=mock_lightrag_core,
            fusion_algorithm=FusionAlgorithm.RRF,
            techniques=["local", "global"]
        )

        mock_lightrag_core.local_query.return_value = {"results": []}
        mock_lightrag_core.global_query.return_value = {"results": []}

        result = await pattern.execute(query="test")

        assert result["results"] == []

    async def test_event_emission_technique_queried(self, mock_lightrag_core, event_collector):
        """Test events emitted for each technique query."""
        pattern = LightRAGHybridSearcher(
            lightrag_core=mock_lightrag_core,
            fusion_algorithm=FusionAlgorithm.RRF,
            techniques=["local", "global"]
        )
        pattern.emit_event = event_collector.collect

        await pattern.execute(query="test")

        events = event_collector.get_events("pattern.hybrid.technique_queried")
        assert len(events) == 2  # One per technique

    async def test_event_emission_fusion_complete(self, mock_lightrag_core, event_collector):
        """Test events emitted after fusion completes."""
        pattern = LightRAGHybridSearcher(
            lightrag_core=mock_lightrag_core,
            fusion_algorithm=FusionAlgorithm.RRF,
            techniques=["local"]
        )
        pattern.emit_event = event_collector.collect

        await pattern.execute(query="test")

        events = event_collector.get_events("pattern.hybrid.fusion_complete")
        assert len(events) > 0

    async def test_result_deduplication(self, mock_lightrag_core):
        """Test deduplication of results across techniques."""
        pattern = LightRAGHybridSearcher(
            lightrag_core=mock_lightrag_core,
            fusion_algorithm=FusionAlgorithm.RRF,
            techniques=["local", "global"]
        )

        # Same doc in both results
        same_doc = {"text": "duplicate", "score": 0.9, "id": "dup"}

        mock_lightrag_core.local_query.return_value = {"results": [same_doc]}
        mock_lightrag_core.global_query.return_value = {"results": [same_doc]}

        result = await pattern.execute(query="test")

        # Should only appear once
        dup_results = [r for r in result["results"] if r["id"] == "dup"]
        assert len(dup_results) == 1

    async def test_base_pattern_interface_compliance(self, mock_lightrag_core):
        """Test that pattern implements BasePattern interface correctly."""
        pattern = LightRAGHybridSearcher(
            lightrag_core=mock_lightrag_core,
            fusion_algorithm=FusionAlgorithm.RRF,
            techniques=["local"]
        )

        assert hasattr(pattern, 'execute')
        assert hasattr(pattern, 'pattern_name')
        assert callable(pattern.execute)

        result = await pattern.execute(query="test")
        assert isinstance(result, dict)
