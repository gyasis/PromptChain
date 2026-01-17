"""Unit tests for LightRAGBranchingThoughts pattern adapter."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from promptchain.integrations.lightrag.branching import LightRAGBranchingThoughts


@pytest.mark.asyncio
class TestLightRAGBranchingThoughts:
    """Test suite for branching thoughts pattern."""

    async def test_initialization(self, mock_lightrag_core):
        """Test pattern initializes correctly."""
        pattern = LightRAGBranchingThoughts(
            lightrag_core=mock_lightrag_core,
            num_branches=3,
            search_mode="hybrid"
        )

        assert pattern.lightrag_core == mock_lightrag_core
        assert pattern.num_branches == 3
        assert pattern.search_mode == "hybrid"
        assert pattern.pattern_name == "branching_thoughts"

    async def test_hypothesis_generation_local(self, mock_lightrag_core):
        """Test hypothesis generation with local search mode."""
        pattern = LightRAGBranchingThoughts(
            lightrag_core=mock_lightrag_core,
            num_branches=3,
            search_mode="local"
        )

        with patch.object(pattern, '_generate_hypotheses', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = [
                "hypothesis 1",
                "hypothesis 2",
                "hypothesis 3"
            ]

            result = await pattern.execute(query="test query")

            mock_gen.assert_called_once_with("test query")
            assert "hypotheses" in result
            assert len(result["hypotheses"]) == 3

    async def test_hypothesis_generation_global(self, mock_lightrag_core):
        """Test hypothesis generation with global search mode."""
        pattern = LightRAGBranchingThoughts(
            lightrag_core=mock_lightrag_core,
            num_branches=2,
            search_mode="global"
        )

        result = await pattern.execute(query="global test query")

        mock_lightrag_core.global_query.assert_called()
        assert "results" in result

    async def test_hypothesis_generation_hybrid(self, mock_lightrag_core):
        """Test hypothesis generation with hybrid search mode."""
        pattern = LightRAGBranchingThoughts(
            lightrag_core=mock_lightrag_core,
            num_branches=3,
            search_mode="hybrid"
        )

        result = await pattern.execute(query="hybrid test query")

        mock_lightrag_core.hybrid_query.assert_called()
        assert "results" in result

    async def test_llm_judge_scoring(self, mock_lightrag_core, mock_llm):
        """Test LLM-based hypothesis scoring."""
        pattern = LightRAGBranchingThoughts(
            lightrag_core=mock_lightrag_core,
            num_branches=3,
            search_mode="local",
            use_llm_judge=True
        )

        hypotheses = [
            {"text": "hyp1", "results": ["r1"]},
            {"text": "hyp2", "results": ["r2"]},
            {"text": "hyp3", "results": ["r3"]}
        ]

        with patch.object(pattern, '_score_with_llm', new_callable=AsyncMock) as mock_score:
            mock_score.return_value = [0.9, 0.7, 0.8]

            scores = await pattern._score_hypotheses(hypotheses, "test query")

            assert len(scores) == 3
            assert scores[0] == 0.9
            mock_score.assert_called_once()

    async def test_best_hypothesis_selection(self, mock_lightrag_core):
        """Test selection of best hypothesis based on scoring."""
        pattern = LightRAGBranchingThoughts(
            lightrag_core=mock_lightrag_core,
            num_branches=3,
            search_mode="local"
        )

        hypotheses = [
            {"text": "low score", "score": 0.5, "results": ["r1"]},
            {"text": "high score", "score": 0.95, "results": ["r2"]},
            {"text": "medium score", "score": 0.7, "results": ["r3"]}
        ]

        best = pattern._select_best_hypothesis(hypotheses)

        assert best["text"] == "high score"
        assert best["score"] == 0.95

    async def test_event_emission_hypothesis_generation(self, mock_lightrag_core, event_collector):
        """Test events emitted during hypothesis generation."""
        pattern = LightRAGBranchingThoughts(
            lightrag_core=mock_lightrag_core,
            num_branches=2,
            search_mode="local"
        )

        # Mock event emission
        pattern.emit_event = event_collector.collect

        with patch.object(pattern, '_generate_hypotheses', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = ["hyp1", "hyp2"]

            await pattern.execute(query="test")

            events = event_collector.get_events("pattern.branching.hypotheses_generated")
            assert len(events) > 0

    async def test_event_emission_scoring(self, mock_lightrag_core, event_collector):
        """Test events emitted during hypothesis scoring."""
        pattern = LightRAGBranchingThoughts(
            lightrag_core=mock_lightrag_core,
            num_branches=2,
            search_mode="local",
            use_llm_judge=True
        )

        pattern.emit_event = event_collector.collect

        with patch.object(pattern, '_score_with_llm', new_callable=AsyncMock) as mock_score:
            mock_score.return_value = [0.9, 0.7]

            await pattern.execute(query="test")

            events = event_collector.get_events("pattern.branching.scored")
            assert len(events) > 0

    async def test_event_emission_best_selected(self, mock_lightrag_core, event_collector):
        """Test events emitted when best hypothesis is selected."""
        pattern = LightRAGBranchingThoughts(
            lightrag_core=mock_lightrag_core,
            num_branches=2,
            search_mode="local"
        )

        pattern.emit_event = event_collector.collect

        await pattern.execute(query="test")

        events = event_collector.get_events("pattern.branching.best_selected")
        assert len(events) > 0

    async def test_invalid_search_mode(self, mock_lightrag_core):
        """Test error handling for invalid search mode."""
        with pytest.raises(ValueError, match="Invalid search_mode"):
            LightRAGBranchingThoughts(
                lightrag_core=mock_lightrag_core,
                num_branches=3,
                search_mode="invalid_mode"
            )

    async def test_num_branches_validation(self, mock_lightrag_core):
        """Test validation of num_branches parameter."""
        with pytest.raises(ValueError, match="num_branches must be at least 2"):
            LightRAGBranchingThoughts(
                lightrag_core=mock_lightrag_core,
                num_branches=1,
                search_mode="local"
            )

    async def test_empty_hypotheses_handling(self, mock_lightrag_core):
        """Test handling of empty hypothesis generation."""
        pattern = LightRAGBranchingThoughts(
            lightrag_core=mock_lightrag_core,
            num_branches=3,
            search_mode="local"
        )

        with patch.object(pattern, '_generate_hypotheses', new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = []

            result = await pattern.execute(query="test")

            assert result["hypotheses"] == []
            assert "error" in result or "warning" in result

    async def test_base_pattern_interface_compliance(self, mock_lightrag_core):
        """Test that pattern implements BasePattern interface correctly."""
        pattern = LightRAGBranchingThoughts(
            lightrag_core=mock_lightrag_core,
            num_branches=3,
            search_mode="local"
        )

        # Check required methods exist
        assert hasattr(pattern, 'execute')
        assert hasattr(pattern, 'pattern_name')
        assert callable(pattern.execute)

        # Check execute returns dict
        result = await pattern.execute(query="test")
        assert isinstance(result, dict)
