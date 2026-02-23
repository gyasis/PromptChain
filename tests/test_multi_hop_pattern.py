"""Tests for LightRAG Multi-Hop Retrieval Pattern.

Tests cover:
1. Pattern initialization with/without hybridrag
2. Question decomposition
3. Multi-hop execution with agentic_search
4. Answer synthesis
5. Unanswered aspect identification
6. Event emission
7. Blackboard integration
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any

from promptchain.patterns.base import PatternConfig
from promptchain.integrations.lightrag.multi_hop import (
    LightRAGMultiHop,
    MultiHopConfig,
    MultiHopResult,
    SubQuestion,
    HYBRIDRAG_AVAILABLE,
)


# Mock SearchInterface for testing without hybridrag
class MockSearchInterface:
    """Mock SearchInterface for testing."""

    async def agentic_search(self, query: str, objective: str, max_steps: int, **kwargs):
        """Mock agentic search that returns a simple result."""
        return {
            "answer": f"Mock answer for: {query}",
            "steps_taken": min(max_steps, 3),
            "context": ["Mock context 1", "Mock context 2"],
        }


@pytest.fixture
def mock_search_interface():
    """Provide a mock SearchInterface."""
    return MockSearchInterface()


@pytest.fixture
def multi_hop_config():
    """Provide a default MultiHopConfig."""
    return MultiHopConfig(
        max_hops=5,
        max_sub_questions=5,
        synthesizer_model="openai/gpt-4o-mini",
        decompose_first=True,
        emit_events=True,
        timeout_seconds=30.0,
    )


class TestMultiHopInitialization:
    """Test pattern initialization."""

    @pytest.mark.skipif(not HYBRIDRAG_AVAILABLE, reason="hybridrag not installed")
    def test_init_with_real_search_interface(self):
        """Test initialization with real SearchInterface."""
        from hybridrag.src.search_interface import SearchInterface
        from hybridrag.src.lightrag_core import HybridLightRAGCore

        core = HybridLightRAGCore(working_dir="./test_lightrag_data")
        search = SearchInterface(lightrag_core=core)

        pattern = LightRAGMultiHop(search_interface=search)

        assert pattern.search_interface is search
        assert isinstance(pattern.config, PatternConfig)
        assert pattern.multi_hop_config.max_hops == 5

    def test_init_with_mock_search_interface(self, mock_search_interface, multi_hop_config):
        """Test initialization with mock SearchInterface."""
        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=multi_hop_config,
            )

            assert pattern.search_interface is mock_search_interface
            assert pattern.multi_hop_config.max_hops == 5
            assert pattern.multi_hop_config.decompose_first is True

    def test_init_without_hybridrag_raises_error(self, mock_search_interface):
        """Test that initialization fails gracefully without hybridrag."""
        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", False):
            with pytest.raises(ImportError, match="hybridrag is not installed"):
                LightRAGMultiHop(search_interface=mock_search_interface)


class TestQuestionDecomposition:
    """Test question decomposition functionality."""

    @pytest.mark.asyncio
    async def test_decompose_question_with_litellm(self, mock_search_interface, multi_hop_config):
        """Test question decomposition using litellm."""
        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=multi_hop_config,
            )

            mock_response = MagicMock()
            mock_response.choices[0].message.content = """
            {
              "sub_questions": [
                {
                  "question_id": "sq1",
                  "question_text": "What are transformers?",
                  "rationale": "Need to understand transformers first",
                  "dependencies": []
                },
                {
                  "question_id": "sq2",
                  "question_text": "What are RNNs?",
                  "rationale": "Need to understand RNNs for comparison",
                  "dependencies": []
                },
                {
                  "question_id": "sq3",
                  "question_text": "How do they differ?",
                  "rationale": "Direct comparison",
                  "dependencies": ["sq1", "sq2"]
                }
              ]
            }
            """

            with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
                sub_questions = await pattern._decompose_question(
                    "What are the key differences between transformers and RNNs?"
                )

                assert len(sub_questions) == 3
                assert sub_questions[0].question_id == "sq1"
                assert sub_questions[0].question_text == "What are transformers?"
                assert sub_questions[2].dependencies == ["sq1", "sq2"]

    @pytest.mark.asyncio
    async def test_decompose_question_without_litellm(self, mock_search_interface, multi_hop_config):
        """Test question decomposition fails gracefully without litellm."""
        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=multi_hop_config,
            )

            # Mock the import to fail
            import sys
            with patch.dict(sys.modules, {'litellm': None}):
                sub_questions = await pattern._decompose_question("Test question?")
                assert sub_questions == []

    @pytest.mark.asyncio
    async def test_decompose_question_respects_max_sub_questions(self, mock_search_interface):
        """Test that decomposition respects max_sub_questions limit."""
        config = MultiHopConfig(max_sub_questions=2)

        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=config,
            )

            mock_response = MagicMock()
            mock_response.choices[0].message.content = """
            {
              "sub_questions": [
                {"question_id": "sq1", "question_text": "Q1", "rationale": "R1", "dependencies": []},
                {"question_id": "sq2", "question_text": "Q2", "rationale": "R2", "dependencies": []},
                {"question_id": "sq3", "question_text": "Q3", "rationale": "R3", "dependencies": []},
                {"question_id": "sq4", "question_text": "Q4", "rationale": "R4", "dependencies": []}
              ]
            }
            """

            with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
                sub_questions = await pattern._decompose_question("Complex question?")
                assert len(sub_questions) == 2  # Limited to max_sub_questions


class TestAgenticSearch:
    """Test agentic search execution."""

    @pytest.mark.asyncio
    async def test_execute_agentic_search_basic(self, mock_search_interface, multi_hop_config):
        """Test basic agentic search execution."""
        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=multi_hop_config,
            )

            result = await pattern._execute_agentic_search(
                question="Test question?",
                sub_questions=[],
                max_hops=5,
            )

            assert result["hops_executed"] == 3  # MockSearchInterface returns min(max_steps, 3)
            assert "Mock answer" in result["main_answer"]

    @pytest.mark.asyncio
    async def test_execute_agentic_search_with_sub_questions(self, mock_search_interface, multi_hop_config):
        """Test agentic search with sub-questions."""
        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=multi_hop_config,
            )

            sub_questions = [
                SubQuestion(
                    question_id="sq1",
                    question_text="What is X?",
                    parent_question="Test question?",
                ),
                SubQuestion(
                    question_id="sq2",
                    question_text="What is Y?",
                    parent_question="Test question?",
                ),
            ]

            result = await pattern._execute_agentic_search(
                question="Test question?",
                sub_questions=sub_questions,
                max_hops=5,
            )

            assert result["hops_executed"] == 3
            assert len(result["sub_answers"]) == 2  # One answer per sub-question

    @pytest.mark.asyncio
    async def test_execute_agentic_search_handles_errors(self, multi_hop_config):
        """Test that agentic search handles errors gracefully."""
        failing_search = AsyncMock(side_effect=Exception("Search failed"))
        failing_search.agentic_search = AsyncMock(side_effect=Exception("Search failed"))

        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=failing_search,
                config=multi_hop_config,
            )

            result = await pattern._execute_agentic_search(
                question="Test question?",
                sub_questions=[],
                max_hops=5,
            )

            assert result["hops_executed"] == 0
            assert result["sub_answers"] == {}
            assert result["main_answer"] == ""


class TestAnswerSynthesis:
    """Test answer synthesis functionality."""

    @pytest.mark.asyncio
    async def test_synthesize_single_answer(self, mock_search_interface, multi_hop_config):
        """Test synthesis with single answer returns it directly."""
        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=multi_hop_config,
            )

            sub_questions = [
                SubQuestion(question_id="sq1", question_text="Q1", parent_question="Main"),
            ]
            sub_answers = {"sq1": "Single answer"}

            result = await pattern._synthesize_answer(
                question="Main question?",
                sub_questions=sub_questions,
                sub_answers=sub_answers,
            )

            assert result == "Single answer"

    @pytest.mark.asyncio
    async def test_synthesize_multiple_answers_with_litellm(self, mock_search_interface, multi_hop_config):
        """Test synthesis with multiple answers using litellm."""
        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=multi_hop_config,
            )

            sub_questions = [
                SubQuestion(question_id="sq1", question_text="Q1", parent_question="Main"),
                SubQuestion(question_id="sq2", question_text="Q2", parent_question="Main"),
            ]
            sub_answers = {
                "sq1": "Answer to Q1",
                "sq2": "Answer to Q2",
            }

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Synthesized comprehensive answer"

            with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
                result = await pattern._synthesize_answer(
                    question="Main question?",
                    sub_questions=sub_questions,
                    sub_answers=sub_answers,
                )

                assert result == "Synthesized comprehensive answer"

    @pytest.mark.asyncio
    async def test_synthesize_empty_answers(self, mock_search_interface, multi_hop_config):
        """Test synthesis with no answers."""
        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=multi_hop_config,
            )

            result = await pattern._synthesize_answer(
                question="Test?",
                sub_questions=[],
                sub_answers={},
            )

            assert "Unable to generate answer" in result


class TestUnansweredAspects:
    """Test identification of unanswered aspects."""

    def test_identify_all_answered(self, mock_search_interface, multi_hop_config):
        """Test when all sub-questions are answered."""
        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=multi_hop_config,
            )

            sub_questions = [
                SubQuestion(question_id="sq1", question_text="Q1", parent_question="Main"),
                SubQuestion(question_id="sq2", question_text="Q2", parent_question="Main"),
            ]
            sub_answers = {
                "sq1": "Answer 1",
                "sq2": "Answer 2",
            }

            unanswered = pattern._identify_unanswered_aspects(sub_questions, sub_answers)
            assert unanswered == []

    def test_identify_partially_answered(self, mock_search_interface, multi_hop_config):
        """Test when some sub-questions are unanswered."""
        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=multi_hop_config,
            )

            sub_questions = [
                SubQuestion(question_id="sq1", question_text="Q1", parent_question="Main"),
                SubQuestion(question_id="sq2", question_text="Q2", parent_question="Main"),
                SubQuestion(question_id="sq3", question_text="Q3", parent_question="Main"),
            ]
            sub_answers = {
                "sq1": "Answer 1",
                "sq3": "",  # Empty answer
            }

            unanswered = pattern._identify_unanswered_aspects(sub_questions, sub_answers)
            assert len(unanswered) == 2
            assert "Q2" in unanswered
            assert "Q3" in unanswered


class TestFullExecution:
    """Test full execution flow."""

    @pytest.mark.asyncio
    async def test_execute_without_decomposition(self, mock_search_interface):
        """Test execution without question decomposition."""
        config = MultiHopConfig(decompose_first=False, emit_events=False)

        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=config,
            )

            result = await pattern.execute(question="Simple question?")

            assert result.success is True
            assert result.hops_executed == 3
            assert len(result.sub_questions) == 0
            assert "Mock answer" in result.unified_answer

    @pytest.mark.asyncio
    async def test_execute_with_decomposition(self, mock_search_interface):
        """Test full execution with decomposition."""
        config = MultiHopConfig(decompose_first=True, emit_events=False)

        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=config,
            )

            # Mock decomposition
            mock_response = MagicMock()
            mock_response.choices[0].message.content = """
            {
              "sub_questions": [
                {"question_id": "sq1", "question_text": "Q1", "rationale": "R1", "dependencies": []}
              ]
            }
            """

            with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
                result = await pattern.execute(question="Complex question?")

                assert result.success is True
                assert len(result.sub_questions) == 1
                assert result.hops_executed == 3

    @pytest.mark.asyncio
    async def test_execute_handles_errors(self):
        """Test that execute handles errors gracefully."""
        failing_search = AsyncMock(side_effect=Exception("Fatal error"))
        failing_search.agentic_search = AsyncMock(side_effect=Exception("Fatal error"))

        # Disable decomposition to avoid litellm calls
        config = MultiHopConfig(decompose_first=False, emit_events=False)

        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=failing_search,
                config=config,
            )

            result = await pattern.execute(question="Test?")

            # Since agentic_search fails, we should still succeed but with no answers
            # The pattern catches the error and returns empty results
            assert result.success is True  # Pattern handles errors gracefully
            assert result.hops_executed == 0
            assert "Unable to generate answer" in result.unified_answer


class TestEventEmission:
    """Test event emission."""

    @pytest.mark.asyncio
    async def test_events_emitted_during_execution(self, mock_search_interface):
        """Test that events are emitted at each stage."""
        config = MultiHopConfig(emit_events=True, decompose_first=False)
        events_captured = []

        def event_handler(event_type: str, data: dict):
            events_captured.append((event_type, data))

        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=config,
            )
            pattern.add_event_handler(event_handler)

            await pattern.execute(question="Test?")

            event_types = [e[0] for e in events_captured]
            assert "pattern.multi_hop.started" in event_types
            assert "pattern.multi_hop.hop_completed" in event_types
            assert "pattern.multi_hop.synthesizing" in event_types
            assert "pattern.multi_hop.completed" in event_types


class TestBlackboardIntegration:
    """Test Blackboard integration."""

    @pytest.mark.asyncio
    async def test_result_shared_to_blackboard(self, mock_search_interface):
        """Test that results are shared via Blackboard when enabled."""
        # Create a simple mock Blackboard
        class MockBlackboard:
            def __init__(self):
                self._data = {}

            def write(self, key: str, value: Any, source: str = "unknown"):
                self._data[key] = value

            def read(self, key: str):
                return self._data.get(key)

        config = MultiHopConfig(use_blackboard=True, emit_events=False, decompose_first=False)
        blackboard = MockBlackboard()

        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(
                search_interface=mock_search_interface,
                config=config,
            )
            pattern.connect_blackboard(blackboard)

            result = await pattern.execute(question="Test?")

            # Check that result was written to blackboard
            shared_key = f"multi_hop_result_{pattern.config.pattern_id}"
            shared_result = blackboard.read(shared_key)

            assert shared_result is not None
            assert shared_result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
