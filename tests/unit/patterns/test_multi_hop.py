"""Unit tests for LightRAGMultiHop pattern adapter."""

import pytest
from unittest.mock import AsyncMock, patch
from promptchain.integrations.lightrag.multi_hop import LightRAGMultiHop


@pytest.mark.asyncio
class TestLightRAGMultiHop:
    """Test suite for multi-hop reasoning pattern."""

    async def test_initialization(self, mock_lightrag_core):
        """Test pattern initializes correctly."""
        pattern = LightRAGMultiHop(
            lightrag_core=mock_lightrag_core,
            max_hops=3
        )

        assert pattern.lightrag_core == mock_lightrag_core
        assert pattern.max_hops == 3
        assert pattern.pattern_name == "multi_hop_reasoning"

    async def test_agentic_search_delegation(self, mock_lightrag_core):
        """Test delegation to agentic_search method."""
        pattern = LightRAGMultiHop(
            lightrag_core=mock_lightrag_core,
            max_hops=5
        )

        mock_lightrag_core.agentic_search.return_value = {
            "results": ["hop1 result", "hop2 result"],
            "metadata": {"hops": 2, "reasoning_chain": ["q1", "q2"]}
        }

        result = await pattern.execute(query="complex question requiring multiple hops")

        mock_lightrag_core.agentic_search.assert_called_once()
        assert "results" in result
        assert result["metadata"]["hops"] == 2

    async def test_question_decomposition(self, mock_lightrag_core):
        """Test decomposition of complex questions into sub-questions."""
        pattern = LightRAGMultiHop(
            lightrag_core=mock_lightrag_core,
            max_hops=3
        )

        with patch.object(pattern, '_decompose_question', new_callable=AsyncMock) as mock_decompose:
            mock_decompose.return_value = [
                "sub-question 1",
                "sub-question 2",
                "sub-question 3"
            ]

            result = await pattern.execute(
                query="What is the capital of the country where the inventor of the telephone was born?"
            )

            mock_decompose.assert_called_once()
            assert "sub_questions" in result or "reasoning_chain" in result.get("metadata", {})

    async def test_hop_tracking(self, mock_lightrag_core):
        """Test tracking of reasoning hops."""
        pattern = LightRAGMultiHop(
            lightrag_core=mock_lightrag_core,
            max_hops=4
        )

        mock_lightrag_core.agentic_search.return_value = {
            "results": ["final answer"],
            "metadata": {
                "hops": 3,
                "reasoning_chain": [
                    "Where was Alexander Graham Bell born?",
                    "What country is Scotland in?",
                    "What is the capital of the United Kingdom?"
                ]
            }
        }

        result = await pattern.execute(query="test")

        metadata = result.get("metadata", {})
        assert "hops" in metadata
        assert metadata["hops"] <= pattern.max_hops
        assert "reasoning_chain" in metadata

    async def test_max_hops_limit(self, mock_lightrag_core):
        """Test that max_hops limit is respected."""
        pattern = LightRAGMultiHop(
            lightrag_core=mock_lightrag_core,
            max_hops=2
        )

        # Mock exceeding max hops
        mock_lightrag_core.agentic_search.return_value = {
            "results": ["partial result"],
            "metadata": {
                "hops": 2,
                "max_hops_reached": True,
                "reasoning_chain": ["q1", "q2"]
            }
        }

        result = await pattern.execute(query="very complex question")

        assert result["metadata"]["hops"] <= 2
        assert result["metadata"].get("max_hops_reached") is True

    async def test_intermediate_results_storage(self, mock_lightrag_core):
        """Test storage of intermediate hop results."""
        pattern = LightRAGMultiHop(
            lightrag_core=mock_lightrag_core,
            max_hops=3,
            store_intermediate=True
        )

        mock_lightrag_core.agentic_search.return_value = {
            "results": ["final"],
            "metadata": {
                "hops": 2,
                "intermediate_results": [
                    {"hop": 1, "query": "q1", "result": "r1"},
                    {"hop": 2, "query": "q2", "result": "r2"}
                ]
            }
        }

        result = await pattern.execute(query="test")

        assert "intermediate_results" in result.get("metadata", {})
        intermediate = result["metadata"]["intermediate_results"]
        assert len(intermediate) == 2

    async def test_single_hop_fallback(self, mock_lightrag_core):
        """Test fallback to single-hop for simple questions."""
        pattern = LightRAGMultiHop(
            lightrag_core=mock_lightrag_core,
            max_hops=3
        )

        with patch.object(pattern, '_needs_multi_hop', new_callable=AsyncMock) as mock_needs:
            mock_needs.return_value = False

            # Should use simple search instead of agentic
            mock_lightrag_core.local_query.return_value = {
                "results": ["simple answer"],
                "metadata": {"hops": 1}
            }

            result = await pattern.execute(query="simple question")

            mock_needs.assert_called_once()
            mock_lightrag_core.local_query.assert_called_once()

    async def test_event_emission_hop_started(self, mock_lightrag_core, event_collector):
        """Test events emitted when hop starts."""
        pattern = LightRAGMultiHop(
            lightrag_core=mock_lightrag_core,
            max_hops=3
        )
        pattern.emit_event = event_collector.collect

        await pattern.execute(query="test")

        events = event_collector.get_events("pattern.multihop.hop_started")
        assert len(events) > 0

    async def test_event_emission_hop_completed(self, mock_lightrag_core, event_collector):
        """Test events emitted when hop completes."""
        pattern = LightRAGMultiHop(
            lightrag_core=mock_lightrag_core,
            max_hops=3
        )
        pattern.emit_event = event_collector.collect

        await pattern.execute(query="test")

        events = event_collector.get_events("pattern.multihop.hop_completed")
        assert len(events) > 0

    async def test_event_emission_reasoning_complete(self, mock_lightrag_core, event_collector):
        """Test events emitted when full reasoning completes."""
        pattern = LightRAGMultiHop(
            lightrag_core=mock_lightrag_core,
            max_hops=3
        )
        pattern.emit_event = event_collector.collect

        await pattern.execute(query="test")

        events = event_collector.get_events("pattern.multihop.complete")
        assert len(events) > 0

    async def test_max_hops_validation(self, mock_lightrag_core):
        """Test validation of max_hops parameter."""
        with pytest.raises(ValueError, match="max_hops must be at least 1"):
            LightRAGMultiHop(
                lightrag_core=mock_lightrag_core,
                max_hops=0
            )

    async def test_circular_reasoning_detection(self, mock_lightrag_core):
        """Test detection of circular reasoning patterns."""
        pattern = LightRAGMultiHop(
            lightrag_core=mock_lightrag_core,
            max_hops=5,
            detect_circular=True
        )

        mock_lightrag_core.agentic_search.return_value = {
            "results": ["result"],
            "metadata": {
                "hops": 3,
                "reasoning_chain": ["q1", "q2", "q1"],  # Circular
                "circular_detected": True
            }
        }

        result = await pattern.execute(query="test")

        assert result["metadata"].get("circular_detected") is True

    async def test_base_pattern_interface_compliance(self, mock_lightrag_core):
        """Test that pattern implements BasePattern interface correctly."""
        pattern = LightRAGMultiHop(
            lightrag_core=mock_lightrag_core,
            max_hops=3
        )

        assert hasattr(pattern, 'execute')
        assert hasattr(pattern, 'pattern_name')
        assert callable(pattern.execute)

        result = await pattern.execute(query="test")
        assert isinstance(result, dict)
