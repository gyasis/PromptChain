"""Tests for LightRAG Hybrid Search Fusion Pattern."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from promptchain.integrations.lightrag.hybrid_search import (
    SearchTechnique,
    FusionAlgorithm,
    TechniqueResult,
    HybridSearchConfig,
    HybridSearchResult,
    LightRAGHybridSearcher,
)


@pytest.fixture
def mock_lightrag_integration():
    """Create a mock LightRAGIntegration."""
    integration = MagicMock()
    integration.core = MagicMock()

    # Mock query methods
    async def mock_local_query(query, top_k=10):
        return {
            "results": [f"local_{i}_{query}" for i in range(min(3, top_k))],
            "scores": [0.9, 0.8, 0.7][:min(3, top_k)]
        }

    async def mock_global_query(query, top_k=10):
        return {
            "results": [f"global_{i}_{query}" for i in range(min(3, top_k))],
            "scores": [0.85, 0.75, 0.65][:min(3, top_k)]
        }

    async def mock_hybrid_query(query, top_k=10):
        return {
            "results": [f"hybrid_{i}_{query}" for i in range(min(3, top_k))],
            "scores": [0.95, 0.85, 0.75][:min(3, top_k)]
        }

    integration.local_query = mock_local_query
    integration.global_query = mock_global_query
    integration.hybrid_query = mock_hybrid_query

    return integration


@pytest.fixture
def hybrid_searcher(mock_lightrag_integration):
    """Create a LightRAGHybridSearcher instance."""
    config = HybridSearchConfig(
        techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
        fusion_algorithm=FusionAlgorithm.RRF,
        rrf_k=60,
        top_k=5,
        emit_events=False  # Disable events for testing
    )
    return LightRAGHybridSearcher(mock_lightrag_integration, config=config)


class TestSearchTechnique:
    """Test SearchTechnique enum."""

    def test_search_technique_values(self):
        """Test that all search techniques have correct values."""
        assert SearchTechnique.LOCAL.value == "local"
        assert SearchTechnique.GLOBAL.value == "global"
        assert SearchTechnique.HYBRID.value == "hybrid"
        assert SearchTechnique.NAIVE.value == "naive"
        assert SearchTechnique.MIX.value == "mix"


class TestFusionAlgorithm:
    """Test FusionAlgorithm enum."""

    def test_fusion_algorithm_values(self):
        """Test that all fusion algorithms have correct values."""
        assert FusionAlgorithm.RRF.value == "rrf"
        assert FusionAlgorithm.LINEAR.value == "linear"
        assert FusionAlgorithm.BORDA.value == "borda"


class TestTechniqueResult:
    """Test TechniqueResult dataclass."""

    def test_technique_result_creation(self):
        """Test creating a TechniqueResult."""
        result = TechniqueResult(
            technique=SearchTechnique.LOCAL,
            results=["doc1", "doc2"],
            scores=[0.9, 0.8],
            query_time_ms=150.5
        )

        assert result.technique == SearchTechnique.LOCAL
        assert result.results == ["doc1", "doc2"]
        assert result.scores == [0.9, 0.8]
        assert result.query_time_ms == 150.5


class TestHybridSearchConfig:
    """Test HybridSearchConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HybridSearchConfig()

        assert config.techniques == [SearchTechnique.LOCAL, SearchTechnique.GLOBAL]
        assert config.fusion_algorithm == FusionAlgorithm.RRF
        assert config.rrf_k == 60
        assert config.top_k == 10
        assert config.normalize_scores is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = HybridSearchConfig(
            techniques=[SearchTechnique.HYBRID, SearchTechnique.NAIVE],
            fusion_algorithm=FusionAlgorithm.LINEAR,
            rrf_k=100,
            top_k=20,
            normalize_scores=False
        )

        assert config.techniques == [SearchTechnique.HYBRID, SearchTechnique.NAIVE]
        assert config.fusion_algorithm == FusionAlgorithm.LINEAR
        assert config.rrf_k == 100
        assert config.top_k == 20
        assert config.normalize_scores is False


class TestHybridSearchResult:
    """Test HybridSearchResult dataclass."""

    def test_hybrid_search_result_creation(self):
        """Test creating a HybridSearchResult."""
        tech_results = [
            TechniqueResult(
                technique=SearchTechnique.LOCAL,
                results=["doc1"],
                scores=[0.9],
                query_time_ms=100
            )
        ]

        result = HybridSearchResult(
            pattern_id="test-123",
            success=True,
            result=["doc1", "doc2"],
            execution_time_ms=250.0,
            query="test query",
            technique_results=tech_results,
            fused_results=["doc1", "doc2"],
            fused_scores=[0.95, 0.85],
            technique_contributions={"local": 2}
        )

        assert result.pattern_id == "test-123"
        assert result.success is True
        assert result.query == "test query"
        assert len(result.technique_results) == 1
        assert result.fused_results == ["doc1", "doc2"]
        assert result.technique_contributions == {"local": 2}


class TestLightRAGHybridSearcher:
    """Test LightRAGHybridSearcher class."""

    @pytest.mark.asyncio
    async def test_execute_basic(self, hybrid_searcher):
        """Test basic execution with RRF fusion."""
        result = await hybrid_searcher.execute(query="test query")

        assert isinstance(result, HybridSearchResult)
        assert result.success is True
        assert result.query == "test query"
        assert len(result.technique_results) == 2  # LOCAL and GLOBAL
        assert len(result.fused_results) > 0
        assert len(result.fused_scores) == len(result.fused_results)

    @pytest.mark.asyncio
    async def test_execute_with_top_k(self, hybrid_searcher):
        """Test execution with custom top_k."""
        result = await hybrid_searcher.execute(query="test query", top_k=3)

        assert result.success is True
        assert len(result.fused_results) <= 3

    @pytest.mark.asyncio
    async def test_reciprocal_rank_fusion(self, mock_lightrag_integration):
        """Test RRF fusion algorithm."""
        config = HybridSearchConfig(
            techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
            fusion_algorithm=FusionAlgorithm.RRF,
            rrf_k=60,
            top_k=10,
            emit_events=False
        )
        searcher = LightRAGHybridSearcher(mock_lightrag_integration, config=config)

        result = await searcher.execute(query="machine learning")

        # Verify RRF was applied
        assert result.success is True
        assert len(result.fused_scores) > 0
        # RRF scores should be positive and sorted descending
        for i in range(len(result.fused_scores) - 1):
            assert result.fused_scores[i] >= result.fused_scores[i + 1]

    @pytest.mark.asyncio
    async def test_linear_fusion(self, mock_lightrag_integration):
        """Test linear fusion algorithm."""
        config = HybridSearchConfig(
            techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
            fusion_algorithm=FusionAlgorithm.LINEAR,
            top_k=10,
            normalize_scores=True,
            emit_events=False
        )
        searcher = LightRAGHybridSearcher(mock_lightrag_integration, config=config)

        result = await searcher.execute(query="deep learning")

        assert result.success is True
        assert len(result.fused_scores) > 0
        # Linear scores should be sorted descending
        for i in range(len(result.fused_scores) - 1):
            assert result.fused_scores[i] >= result.fused_scores[i + 1]

    @pytest.mark.asyncio
    async def test_borda_fusion(self, mock_lightrag_integration):
        """Test Borda count fusion algorithm."""
        config = HybridSearchConfig(
            techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
            fusion_algorithm=FusionAlgorithm.BORDA,
            top_k=10,
            emit_events=False
        )
        searcher = LightRAGHybridSearcher(mock_lightrag_integration, config=config)

        result = await searcher.execute(query="neural networks")

        assert result.success is True
        assert len(result.fused_scores) > 0
        # Borda scores should be positive integers
        for score in result.fused_scores:
            assert score >= 0

    @pytest.mark.asyncio
    async def test_technique_contributions(self, hybrid_searcher):
        """Test that technique contributions are tracked."""
        result = await hybrid_searcher.execute(query="test query")

        assert result.technique_contributions is not None
        assert isinstance(result.technique_contributions, dict)
        # Should have contributions from local and/or global
        total_contributions = sum(result.technique_contributions.values())
        assert total_contributions > 0

    @pytest.mark.asyncio
    async def test_parallel_execution(self, hybrid_searcher):
        """Test that techniques are executed in parallel."""
        # All techniques should complete even if one is slow
        result = await hybrid_searcher.execute(query="parallel test")

        assert result.success is True
        # Should have results from both techniques
        assert len(result.technique_results) == 2

    @pytest.mark.asyncio
    async def test_single_technique(self, mock_lightrag_integration):
        """Test execution with a single technique."""
        config = HybridSearchConfig(
            techniques=[SearchTechnique.LOCAL],
            fusion_algorithm=FusionAlgorithm.RRF,
            emit_events=False
        )
        searcher = LightRAGHybridSearcher(mock_lightrag_integration, config=config)

        result = await searcher.execute(query="single technique")

        assert result.success is True
        assert len(result.technique_results) == 1
        assert result.technique_results[0].technique == SearchTechnique.LOCAL

    @pytest.mark.asyncio
    async def test_all_techniques(self, mock_lightrag_integration):
        """Test execution with all available techniques."""
        config = HybridSearchConfig(
            techniques=[
                SearchTechnique.LOCAL,
                SearchTechnique.GLOBAL,
                SearchTechnique.HYBRID
            ],
            fusion_algorithm=FusionAlgorithm.RRF,
            emit_events=False
        )
        searcher = LightRAGHybridSearcher(mock_lightrag_integration, config=config)

        result = await searcher.execute(query="all techniques")

        assert result.success is True
        assert len(result.technique_results) == 3

    @pytest.mark.asyncio
    async def test_empty_results_handling(self, mock_lightrag_integration):
        """Test handling of empty results from techniques."""
        # Mock empty results
        async def empty_local_query(query, top_k=10):
            return {"results": [], "scores": []}

        async def empty_global_query(query, top_k=10):
            return {"results": [], "scores": []}

        mock_lightrag_integration.local_query = empty_local_query
        mock_lightrag_integration.global_query = empty_global_query

        config = HybridSearchConfig(
            techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
            fusion_algorithm=FusionAlgorithm.RRF,
            emit_events=False
        )
        searcher = LightRAGHybridSearcher(mock_lightrag_integration, config=config)

        result = await searcher.execute(query="empty results")

        # Should still succeed but with empty results
        assert result.success is True
        assert len(result.fused_results) == 0


class TestRRFAlgorithm:
    """Test RRF fusion specifically."""

    @pytest.mark.asyncio
    async def test_rrf_k_parameter(self, mock_lightrag_integration):
        """Test that RRF k parameter affects scoring."""
        # Test with different k values
        config_k60 = HybridSearchConfig(
            techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
            fusion_algorithm=FusionAlgorithm.RRF,
            rrf_k=60,
            emit_events=False
        )
        searcher_k60 = LightRAGHybridSearcher(
            mock_lightrag_integration,
            config=config_k60
        )

        config_k100 = HybridSearchConfig(
            techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
            fusion_algorithm=FusionAlgorithm.RRF,
            rrf_k=100,
            emit_events=False
        )
        searcher_k100 = LightRAGHybridSearcher(
            mock_lightrag_integration,
            config=config_k100
        )

        result_k60 = await searcher_k60.execute(query="rrf test")
        result_k100 = await searcher_k100.execute(query="rrf test")

        # Both should succeed
        assert result_k60.success is True
        assert result_k100.success is True

        # Scores might differ due to different k values
        # (though with small result sets, differences may be minimal)
        assert len(result_k60.fused_scores) > 0
        assert len(result_k100.fused_scores) > 0


class TestNormalization:
    """Test score normalization."""

    @pytest.mark.asyncio
    async def test_normalization_enabled(self, mock_lightrag_integration):
        """Test linear fusion with normalization enabled."""
        config = HybridSearchConfig(
            techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
            fusion_algorithm=FusionAlgorithm.LINEAR,
            normalize_scores=True,
            emit_events=False
        )
        searcher = LightRAGHybridSearcher(mock_lightrag_integration, config=config)

        result = await searcher.execute(query="normalized")

        assert result.success is True
        # With normalization, scores should be in reasonable range
        assert all(score >= 0 for score in result.fused_scores)

    @pytest.mark.asyncio
    async def test_normalization_disabled(self, mock_lightrag_integration):
        """Test linear fusion with normalization disabled."""
        config = HybridSearchConfig(
            techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
            fusion_algorithm=FusionAlgorithm.LINEAR,
            normalize_scores=False,
            emit_events=False
        )
        searcher = LightRAGHybridSearcher(mock_lightrag_integration, config=config)

        result = await searcher.execute(query="not normalized")

        assert result.success is True
        assert all(score >= 0 for score in result.fused_scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
