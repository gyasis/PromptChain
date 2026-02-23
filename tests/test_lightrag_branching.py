"""Tests for LightRAG Branching Thoughts Pattern."""

import asyncio
import json
import pytest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from promptchain.integrations.lightrag.branching import (
    BranchingConfig,
    Hypothesis,
    HypothesisScore,
    BranchingResult,
    LightRAGBranchingThoughts,
)
from promptchain.patterns.base import PatternConfig


# Mock LightRAG components
class MockLightRAGCore:
    """Mock HybridLightRAGCore for testing."""

    async def local_query(self, query: str, **kwargs) -> str:
        """Mock local query."""
        return f"Local query result: {query} - Focuses on specific entities and details."

    async def global_query(self, query: str, **kwargs) -> str:
        """Mock global query."""
        return f"Global query result: {query} - Provides high-level conceptual overview."

    async def hybrid_query(self, query: str, **kwargs) -> str:
        """Mock hybrid query."""
        return f"Hybrid query result: {query} - Combines entity details with concepts."


@pytest.fixture
def mock_lightrag():
    """Create a mock LightRAG core."""
    return MockLightRAGCore()


@pytest.fixture
def branching_config():
    """Create a default branching config."""
    return BranchingConfig(
        hypothesis_count=3,
        judge_model="openai/gpt-4o-mini",
        timeout_seconds=30.0,
    )


@pytest.fixture
def branching_pattern(mock_lightrag, branching_config):
    """Create a branching pattern instance."""
    with patch("promptchain.integrations.lightrag.branching.LIGHTRAG_AVAILABLE", True):
        pattern = LightRAGBranchingThoughts(
            lightrag_core=mock_lightrag,
            config=branching_config
        )
        return pattern


class TestBranchingConfig:
    """Test BranchingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BranchingConfig()
        assert config.hypothesis_count == 3
        assert config.generator_model is None
        assert config.judge_model == "openai/gpt-4o"
        assert config.diversity_threshold == 0.3
        assert config.record_outcomes is True
        assert config.enabled is True
        assert config.emit_events is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BranchingConfig(
            hypothesis_count=5,
            judge_model="anthropic/claude-3-opus-20240229",
            diversity_threshold=0.5,
            timeout_seconds=60.0,
        )
        assert config.hypothesis_count == 5
        assert config.judge_model == "anthropic/claude-3-opus-20240229"
        assert config.diversity_threshold == 0.5
        assert config.timeout_seconds == 60.0


class TestHypothesis:
    """Test Hypothesis dataclass."""

    def test_hypothesis_creation(self):
        """Test creating a hypothesis."""
        hypothesis = Hypothesis(
            hypothesis_id="test-id",
            approach="Test approach",
            reasoning="Test reasoning",
            confidence=0.85,
            mode="hybrid",
        )
        assert hypothesis.hypothesis_id == "test-id"
        assert hypothesis.approach == "Test approach"
        assert hypothesis.reasoning == "Test reasoning"
        assert hypothesis.confidence == 0.85
        assert hypothesis.mode == "hybrid"

    def test_hypothesis_to_dict(self):
        """Test hypothesis serialization."""
        hypothesis = Hypothesis(
            hypothesis_id="test-id",
            approach="Test approach",
            reasoning="Test reasoning",
            confidence=0.85,
            mode="hybrid",
        )
        data = hypothesis.to_dict()

        assert data["hypothesis_id"] == "test-id"
        assert data["approach"] == "Test approach"
        assert data["reasoning"] == "Test reasoning"
        assert data["confidence"] == 0.85
        assert data["mode"] == "hybrid"


class TestHypothesisScore:
    """Test HypothesisScore dataclass."""

    def test_score_creation(self):
        """Test creating a hypothesis score."""
        score = HypothesisScore(
            hypothesis_id="test-id",
            score=0.9,
            reasoning="Excellent hypothesis",
            strengths=["Complete", "Well-reasoned"],
            weaknesses=["Could be more concise"],
        )
        assert score.hypothesis_id == "test-id"
        assert score.score == 0.9
        assert score.reasoning == "Excellent hypothesis"
        assert len(score.strengths) == 2
        assert len(score.weaknesses) == 1

    def test_score_to_dict(self):
        """Test score serialization."""
        score = HypothesisScore(
            hypothesis_id="test-id",
            score=0.9,
            reasoning="Excellent hypothesis",
            strengths=["Complete"],
            weaknesses=["Verbose"],
        )
        data = score.to_dict()

        assert data["hypothesis_id"] == "test-id"
        assert data["score"] == 0.9
        assert data["reasoning"] == "Excellent hypothesis"
        assert data["strengths"] == ["Complete"]
        assert data["weaknesses"] == ["Verbose"]


class TestBranchingResult:
    """Test BranchingResult dataclass."""

    def test_result_creation(self):
        """Test creating a branching result."""
        hypothesis = Hypothesis(
            hypothesis_id="h1",
            approach="Test",
            reasoning="Test reasoning",
            confidence=0.8,
            mode="hybrid",
        )
        score = HypothesisScore(
            hypothesis_id="h1",
            score=0.85,
            reasoning="Good hypothesis",
            strengths=["Clear"],
            weaknesses=["Brief"],
        )

        result = BranchingResult(
            pattern_id="test-pattern",
            success=True,
            result="Selected hypothesis reasoning",
            execution_time_ms=1000.0,
            hypotheses=[hypothesis],
            scores=[score],
            selected_hypothesis=hypothesis,
            selection_reasoning="Best score",
        )

        assert result.success is True
        assert len(result.hypotheses) == 1
        assert len(result.scores) == 1
        assert result.selected_hypothesis == hypothesis
        assert result.selection_reasoning == "Best score"

    def test_result_to_dict(self):
        """Test result serialization."""
        hypothesis = Hypothesis(
            hypothesis_id="h1",
            approach="Test",
            reasoning="Test reasoning",
            confidence=0.8,
            mode="hybrid",
        )

        result = BranchingResult(
            pattern_id="test-pattern",
            success=True,
            result="Selected hypothesis reasoning",
            execution_time_ms=1000.0,
            hypotheses=[hypothesis],
            scores=[],
            selected_hypothesis=hypothesis,
            selection_reasoning="Best score",
        )

        data = result.to_dict()
        assert data["success"] is True
        assert len(data["hypotheses"]) == 1
        assert data["selected_hypothesis"]["hypothesis_id"] == "h1"
        assert data["selection_reasoning"] == "Best score"


class TestLightRAGBranchingThoughts:
    """Test LightRAGBranchingThoughts pattern."""

    def test_initialization(self, mock_lightrag, branching_config):
        """Test pattern initialization."""
        with patch("promptchain.integrations.lightrag.branching.LIGHTRAG_AVAILABLE", True):
            pattern = LightRAGBranchingThoughts(
                lightrag_core=mock_lightrag,
                config=branching_config
            )
            assert pattern._lightrag == mock_lightrag
            assert pattern.config.hypothesis_count == 3

    def test_initialization_without_lightrag(self, mock_lightrag, branching_config):
        """Test initialization fails without hybridrag."""
        with patch("promptchain.integrations.lightrag.branching.LIGHTRAG_AVAILABLE", False):
            with pytest.raises(ImportError, match="hybridrag is not installed"):
                LightRAGBranchingThoughts(
                    lightrag_core=mock_lightrag,
                    config=branching_config
                )

    @pytest.mark.asyncio
    async def test_query_with_mode_local(self, branching_pattern):
        """Test querying in local mode."""
        result = await branching_pattern._query_with_mode("test query", "local")
        assert "Local query result" in result
        assert "specific entities" in result

    @pytest.mark.asyncio
    async def test_query_with_mode_global(self, branching_pattern):
        """Test querying in global mode."""
        result = await branching_pattern._query_with_mode("test query", "global")
        assert "Global query result" in result
        assert "conceptual overview" in result

    @pytest.mark.asyncio
    async def test_query_with_mode_hybrid(self, branching_pattern):
        """Test querying in hybrid mode."""
        result = await branching_pattern._query_with_mode("test query", "hybrid")
        assert "Hybrid query result" in result
        assert "Combines" in result

    def test_extract_hypothesis(self, branching_pattern):
        """Test extracting hypothesis from query result."""
        result = "This is a test result about climate change."
        hypothesis = branching_pattern._extract_hypothesis("local", result, 0)

        assert hypothesis.mode == "local"
        assert hypothesis.reasoning == result
        assert "Entity-focused" in hypothesis.approach
        assert hypothesis.confidence == 0.7

    def test_extract_hypothesis_from_dict(self, branching_pattern):
        """Test extracting hypothesis from dict result."""
        result = {"response": "Climate change is caused by many factors."}
        hypothesis = branching_pattern._extract_hypothesis("global", result, 1)

        assert hypothesis.mode == "global"
        assert "Climate change" in hypothesis.reasoning
        assert "High-level" in hypothesis.approach

    @pytest.mark.asyncio
    async def test_generate_hypotheses(self, branching_pattern):
        """Test generating multiple hypotheses."""
        hypotheses = await branching_pattern._generate_hypotheses("test problem", 3)

        assert len(hypotheses) == 3
        assert hypotheses[0].mode == "local"
        assert hypotheses[1].mode == "global"
        assert hypotheses[2].mode == "hybrid"

        for h in hypotheses:
            assert h.hypothesis_id is not None
            assert len(h.reasoning) > 0
            assert h.confidence > 0

    @pytest.mark.asyncio
    async def test_generate_hypotheses_custom_count(self, branching_pattern):
        """Test generating custom number of hypotheses."""
        hypotheses = await branching_pattern._generate_hypotheses("test problem", 2)
        assert len(hypotheses) == 2

        hypotheses = await branching_pattern._generate_hypotheses("test problem", 5)
        assert len(hypotheses) == 5

    def test_create_judge_prompt(self, branching_pattern):
        """Test creating judge prompt."""
        hypotheses = [
            Hypothesis("h1", "Approach 1", "Reasoning 1", 0.7, "local"),
            Hypothesis("h2", "Approach 2", "Reasoning 2", 0.8, "global"),
        ]

        prompt = branching_pattern._create_judge_prompt("test problem", hypotheses)

        assert "test problem" in prompt
        assert "h1" in prompt
        assert "h2" in prompt
        assert "local" in prompt
        assert "global" in prompt
        assert "Completeness" in prompt
        assert "Relevance" in prompt

    def test_parse_judge_output_valid_json(self, branching_pattern):
        """Test parsing valid judge output."""
        hypotheses = [
            Hypothesis("h1", "Approach 1", "Reasoning 1", 0.7, "local"),
        ]

        judge_output = json.dumps({
            "evaluations": [
                {
                    "hypothesis_id": "h1",
                    "score": 0.85,
                    "reasoning": "Well-reasoned hypothesis",
                    "strengths": ["Clear", "Complete"],
                    "weaknesses": ["Could be more detailed"],
                }
            ]
        })

        scores = branching_pattern._parse_judge_output(judge_output, hypotheses)

        assert len(scores) == 1
        assert scores[0].hypothesis_id == "h1"
        assert scores[0].score == 0.85
        assert scores[0].reasoning == "Well-reasoned hypothesis"
        assert len(scores[0].strengths) == 2
        assert len(scores[0].weaknesses) == 1

    def test_parse_judge_output_invalid_json(self, branching_pattern):
        """Test parsing invalid judge output falls back to defaults."""
        hypotheses = [
            Hypothesis("h1", "Approach 1", "Reasoning 1", 0.7, "local"),
        ]

        judge_output = "This is not valid JSON"
        scores = branching_pattern._parse_judge_output(judge_output, hypotheses)

        # Should fall back to default confidence scores
        assert len(scores) == 1
        assert scores[0].hypothesis_id == "h1"
        assert scores[0].score == 0.7  # Uses hypothesis confidence

    def test_select_best_hypothesis(self, branching_pattern):
        """Test selecting the best hypothesis."""
        hypotheses = [
            Hypothesis("h1", "Approach 1", "Reasoning 1", 0.7, "local"),
            Hypothesis("h2", "Approach 2", "Reasoning 2", 0.8, "global"),
            Hypothesis("h3", "Approach 3", "Reasoning 3", 0.6, "hybrid"),
        ]

        scores = [
            HypothesisScore("h1", 0.75, "Good", ["Clear"], ["Brief"]),
            HypothesisScore("h2", 0.92, "Excellent", ["Complete"], ["Verbose"]),
            HypothesisScore("h3", 0.68, "Okay", ["Fast"], ["Incomplete"]),
        ]

        selected, reasoning = branching_pattern._select_best_hypothesis(hypotheses, scores)

        assert selected is not None
        assert selected.hypothesis_id == "h2"  # Highest score
        assert selected.mode == "global"
        assert "0.92" in reasoning
        assert "Excellent" in reasoning

    def test_select_best_hypothesis_no_scores(self, branching_pattern):
        """Test selection with no scores."""
        hypotheses = [Hypothesis("h1", "A", "R", 0.7, "local")]
        scores = []

        selected, reasoning = branching_pattern._select_best_hypothesis(hypotheses, scores)

        assert selected is None
        assert "No scores" in reasoning

    @pytest.mark.asyncio
    async def test_execute_full_pattern(self, branching_pattern):
        """Test full pattern execution."""
        # Mock LiteLLM to return evaluations matching generated hypothesis IDs
        async def mock_completion(*args, **kwargs):
            # First, let the pattern generate hypotheses
            # We need to extract hypothesis IDs from the judge prompt
            messages = kwargs.get("messages", [])
            judge_prompt = messages[-1]["content"] if messages else ""

            # Extract hypothesis IDs from the prompt
            import re
            hypothesis_ids = re.findall(r'ID: ([a-f0-9-]+)', judge_prompt)

            # Create evaluations for each hypothesis
            evaluations = [
                {
                    "hypothesis_id": hid,
                    "score": 0.7 + (i * 0.1),  # Ascending scores
                    "reasoning": f"Hypothesis {i+1} evaluation",
                    "strengths": ["Clear", "Complete"],
                    "weaknesses": ["Could be more detailed"],
                }
                for i, hid in enumerate(hypothesis_ids)
            ]

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = json.dumps({
                "evaluations": evaluations
            })
            return mock_response

        with patch.object(branching_pattern._litellm, "acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = mock_completion

            result = await branching_pattern.execute(problem="What causes climate change?")

            assert result.success is True
            assert len(result.hypotheses) == 3
            assert len(result.scores) == 3
            assert result.selected_hypothesis is not None
            assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_with_events(self, branching_pattern):
        """Test pattern execution emits events."""
        events_emitted = []

        def event_handler(event_type: str, data: dict):
            events_emitted.append(event_type)

        branching_pattern.add_event_handler(event_handler)

        # Mock LiteLLM for judging
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"evaluations": []}'

        with patch.object(branching_pattern._litellm, "acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response

            result = await branching_pattern.execute(problem="test")

            # Check events were emitted
            assert "pattern.branching.started" in events_emitted
            assert "pattern.branching.generating" in events_emitted
            assert events_emitted.count("pattern.branching.hypothesis_generated") >= 2

    @pytest.mark.asyncio
    async def test_execute_with_custom_count(self, branching_pattern):
        """Test execution with custom hypothesis count."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"evaluations": []}'

        with patch.object(branching_pattern._litellm, "acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response

            result = await branching_pattern.execute(
                problem="test",
                hypothesis_count=5
            )

            assert len(result.hypotheses) == 5

    @pytest.mark.asyncio
    async def test_execute_handles_errors(self, branching_pattern):
        """Test pattern handles execution errors gracefully."""
        # Make query methods raise an exception
        async def failing_query(*args, **kwargs):
            raise ValueError("Query failed")

        branching_pattern._lightrag.local_query = failing_query
        branching_pattern._lightrag.global_query = failing_query
        branching_pattern._lightrag.hybrid_query = failing_query

        # Mock LiteLLM to avoid real API call
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"evaluations": []}'

        with patch.object(branching_pattern._litellm, "acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response

            result = await branching_pattern.execute(problem="test")

            # Pattern should succeed but with empty hypotheses
            # (errors in individual queries are caught and logged)
            assert result.success is True
            assert len(result.hypotheses) == 0  # All queries failed
            assert result.selected_hypothesis is None

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, branching_config, mock_lightrag):
        """Test pattern respects timeout."""
        # Create pattern with very short timeout
        config = BranchingConfig(timeout_seconds=0.001)

        with patch("promptchain.integrations.lightrag.branching.LIGHTRAG_AVAILABLE", True):
            pattern = LightRAGBranchingThoughts(
                lightrag_core=mock_lightrag,
                config=config
            )

            # Make queries take a long time
            async def slow_query(*args, **kwargs):
                await asyncio.sleep(1.0)
                return "result"

            pattern._lightrag.local_query = slow_query
            pattern._lightrag.global_query = slow_query
            pattern._lightrag.hybrid_query = slow_query

            result = await pattern.execute_with_timeout(problem="test")

            # Should timeout
            assert result.success is False
            # Check for timeout in errors
            assert len(result.errors) > 0
            # The timeout error message comes from base pattern
            assert "0.001" in str(result.errors) or "timed out" in str(result.errors).lower()
