"""
Unit tests for DryRunPredictor class (Tool Dry Run Prediction).

Tests cover:
- Prediction generation and parsing
- Confidence bounds and validation
- Prediction accuracy comparison
- Statistics tracking
- JSON parsing (success and fallback cases)
- Edge cases and error handling
"""

import pytest
import json
from promptchain.utils.dry_run import DryRunPredictor, DryRunPrediction


class MockLLMRunner:
    """Mock LLM runner for testing dry run prediction."""

    def __init__(self, response_content: str):
        self.response_content = response_content
        self.call_count = 0
        self.last_messages = None

    async def __call__(self, messages, model=None):
        """Mock LLM call."""
        self.call_count += 1
        self.last_messages = messages
        return {"content": self.response_content}


class TestDryRunPredictorInitialization:
    """Test DryRunPredictor initialization."""

    def test_init_with_model(self):
        """Test initialization with model name."""
        mock_runner = MockLLMRunner("")
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        assert predictor.llm_runner == mock_runner
        assert predictor.model_name == "openai/gpt-4o-mini"
        assert predictor.prediction_count == 0
        assert predictor.accuracy_history == []


class TestPredictionGeneration:
    """Test prediction generation."""

    @pytest.mark.asyncio
    async def test_predict_outcome_success(self):
        """Test successful prediction generation."""
        prediction_response = json.dumps({
            "predicted_output": "Database query will return 10 user records",
            "confidence": 0.85,
            "reasoning": "Query is well-formed and database is accessible"
        })

        mock_runner = MockLLMRunner(prediction_response)
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = await predictor.predict_outcome(
            tool_name="query_database",
            tool_args={"query": "SELECT * FROM users LIMIT 10"},
            context="User requested to list users"
        )

        assert prediction.tool_name == "query_database"
        assert prediction.predicted_output == "Database query will return 10 user records"
        assert prediction.confidence == 0.85
        assert "well-formed" in prediction.reasoning
        assert predictor.prediction_count == 1

    @pytest.mark.asyncio
    async def test_predict_outcome_with_markdown_json(self):
        """Test prediction parsing with JSON in markdown fences."""
        prediction_response = f"""```json
{json.dumps({
    "predicted_output": "File will be created successfully",
    "confidence": 0.9,
    "reasoning": "Valid file path and write permissions"
})}
```"""

        mock_runner = MockLLMRunner(prediction_response)
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = await predictor.predict_outcome(
            tool_name="write_file",
            tool_args={"path": "/tmp/test.txt", "content": "Hello"},
            context=""
        )

        assert prediction.predicted_output == "File will be created successfully"
        assert prediction.confidence == 0.9

    @pytest.mark.asyncio
    async def test_predict_outcome_fallback_on_error(self):
        """Test fallback behavior when prediction fails."""

        async def failing_runner(messages, model=None):
            raise RuntimeError("LLM call failed")

        predictor = DryRunPredictor(failing_runner, "openai/gpt-4o-mini")

        prediction = await predictor.predict_outcome(
            tool_name="test_tool",
            tool_args={},
            context=""
        )

        # Should return low-confidence fallback prediction
        assert prediction.tool_name == "test_tool"
        assert prediction.confidence == 0.0
        assert "Could not predict outcome" in prediction.predicted_output
        assert "error" in prediction.reasoning.lower()


class TestConfidenceBounds:
    """Test confidence value clamping."""

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_max(self):
        """Test confidence is clamped to 1.0."""
        prediction_response = json.dumps({
            "predicted_output": "Result",
            "confidence": 1.5,  # Invalid: > 1.0
            "reasoning": "High confidence"
        })

        mock_runner = MockLLMRunner(prediction_response)
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = await predictor.predict_outcome("tool", {}, "")

        assert prediction.confidence == 1.0  # Clamped to max

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_min(self):
        """Test confidence is clamped to 0.0."""
        prediction_response = json.dumps({
            "predicted_output": "Result",
            "confidence": -0.3,  # Invalid: < 0.0
            "reasoning": "Low confidence"
        })

        mock_runner = MockLLMRunner(prediction_response)
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = await predictor.predict_outcome("tool", {}, "")

        assert prediction.confidence == 0.0  # Clamped to min


class TestJSONParsing:
    """Test JSON response parsing."""

    @pytest.mark.asyncio
    async def test_parse_clean_json(self):
        """Test parsing clean JSON response."""
        prediction_response = json.dumps({
            "predicted_output": "Test output",
            "confidence": 0.75,
            "reasoning": "Test reasoning"
        })

        mock_runner = MockLLMRunner(prediction_response)
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = await predictor.predict_outcome("tool", {}, "")

        assert prediction.predicted_output == "Test output"
        assert prediction.confidence == 0.75
        assert prediction.reasoning == "Test reasoning"

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_json(self):
        """Test fallback behavior when JSON parsing fails."""
        prediction_response = "This is not valid JSON at all"

        mock_runner = MockLLMRunner(prediction_response)
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = await predictor.predict_outcome("tool", {}, "")

        # Should fallback to using response as predicted output
        assert "This is not valid JSON" in prediction.predicted_output
        assert prediction.confidence == 0.3  # Default fallback confidence
        assert "Parsing failed" in prediction.reasoning


class TestPredictionComparison:
    """Test prediction accuracy comparison."""

    def test_compare_exact_substring_match(self):
        """Test comparison with exact substring match."""
        mock_runner = MockLLMRunner("")
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = DryRunPrediction(
            tool_name="test",
            predicted_output="User found in database",
            confidence=0.8,
            reasoning=""
        )

        actual = "Query successful: User found in database with ID 123"

        similarity = predictor.compare_prediction_to_actual(prediction, actual)

        # Exact substring match should give high similarity
        assert similarity == 0.9

    def test_compare_word_overlap(self):
        """Test comparison with word overlap."""
        mock_runner = MockLLMRunner("")
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = DryRunPrediction(
            tool_name="test",
            predicted_output="database query successful users",
            confidence=0.8,
            reasoning=""
        )

        actual = "Query to database returned users successfully"

        similarity = predictor.compare_prediction_to_actual(prediction, actual)

        # Should have reasonable similarity based on word overlap
        assert 0.0 < similarity < 1.0

    def test_compare_no_overlap(self):
        """Test comparison with no word overlap."""
        mock_runner = MockLLMRunner("")
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = DryRunPrediction(
            tool_name="test",
            predicted_output="apple orange banana",
            confidence=0.8,
            reasoning=""
        )

        actual = "zebra elephant giraffe"

        similarity = predictor.compare_prediction_to_actual(prediction, actual)

        assert similarity == 0.0


class TestAccuracyTracking:
    """Test prediction accuracy tracking."""

    def test_accuracy_history_tracking(self):
        """Test that comparisons are tracked in history."""
        mock_runner = MockLLMRunner("")
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = DryRunPrediction("test", "output", 0.8, "")

        # Make several comparisons
        similarity1 = predictor.compare_prediction_to_actual(prediction, "output result")
        similarity2 = predictor.compare_prediction_to_actual(prediction, "output success")
        similarity3 = predictor.compare_prediction_to_actual(prediction, "output done")

        # Verify comparisons returned values
        assert similarity1 is not None
        assert similarity2 is not None
        assert similarity3 is not None

        # Verify history was updated (all had word "output" in common)
        assert len(predictor.accuracy_history) == 3

    def test_accuracy_history_limit(self):
        """Test that accuracy history is limited to last 100 entries."""
        mock_runner = MockLLMRunner("")
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = DryRunPrediction("test", "output", 0.8, "")

        # Make 150 comparisons
        for _ in range(150):
            predictor.compare_prediction_to_actual(prediction, "output")

        # Should keep only last 100
        assert len(predictor.accuracy_history) == 100


class TestStatistics:
    """Test statistics gathering."""

    def test_get_stats_empty(self):
        """Test stats with no predictions."""
        mock_runner = MockLLMRunner("")
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        stats = predictor.get_accuracy_stats()

        assert stats["prediction_count"] == 0
        assert stats["comparisons_made"] == 0
        assert stats["average_accuracy"] == 0.0
        assert stats["model"] == "openai/gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_get_stats_with_predictions(self):
        """Test stats after making predictions."""
        prediction_response = json.dumps({
            "predicted_output": "Result",
            "confidence": 0.8,
            "reasoning": "Test"
        })

        mock_runner = MockLLMRunner(prediction_response)
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        # Make predictions
        await predictor.predict_outcome("tool1", {}, "")
        await predictor.predict_outcome("tool2", {}, "")
        await predictor.predict_outcome("tool3", {}, "")

        stats = predictor.get_accuracy_stats()

        assert stats["prediction_count"] == 3
        assert stats["model"] == "openai/gpt-4o-mini"

    def test_get_stats_with_comparisons(self):
        """Test stats with prediction comparisons."""
        mock_runner = MockLLMRunner("")
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = DryRunPrediction("test", "output", 0.8, "")

        # Make comparisons
        sim1 = predictor.compare_prediction_to_actual(prediction, "output result")
        sim2 = predictor.compare_prediction_to_actual(prediction, "output success")

        # Verify comparisons worked
        assert sim1 is not None
        assert sim2 is not None

        stats = predictor.get_accuracy_stats()

        assert stats["comparisons_made"] == 2
        assert "average_accuracy" in stats
        assert "min_accuracy" in stats
        assert "max_accuracy" in stats

    def test_reset_stats(self):
        """Test statistics reset."""
        mock_runner = MockLLMRunner("")
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = DryRunPrediction("test", "output", 0.8, "")
        predictor.compare_prediction_to_actual(prediction, "output")

        # Reset
        predictor.reset_stats()

        assert predictor.prediction_count == 0
        assert predictor.accuracy_history == []


class TestPromptBuilding:
    """Test prediction prompt building."""

    @pytest.mark.asyncio
    async def test_prompt_includes_context(self):
        """Test that prediction prompt includes context."""
        mock_runner = MockLLMRunner(json.dumps({
            "predicted_output": "Result",
            "confidence": 0.8,
            "reasoning": "Test"
        }))
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        await predictor.predict_outcome(
            tool_name="test_tool",
            tool_args={"arg1": "value1"},
            context="Important context information"
        )

        # Check that prompt includes context
        prompt = mock_runner.last_messages[0]["content"]
        assert "Important context information" in prompt
        assert "test_tool" in prompt
        assert "arg1" in prompt

    @pytest.mark.asyncio
    async def test_long_context_truncated(self):
        """Test that very long context is truncated."""
        long_context = "A" * 5000  # 5000 characters

        mock_runner = MockLLMRunner(json.dumps({
            "predicted_output": "Result",
            "confidence": 0.8,
            "reasoning": "Test"
        }))
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        await predictor.predict_outcome(
            tool_name="test_tool",
            tool_args={},
            context=long_context
        )

        # Verify context was truncated in prompt
        prompt = mock_runner.last_messages[0]["content"]
        assert "(truncated)" in prompt


class TestReprMethod:
    """Test string representation."""

    def test_repr_empty(self):
        """Test __repr__ with no predictions."""
        mock_runner = MockLLMRunner("")
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        repr_str = repr(predictor)

        assert "DryRunPredictor" in repr_str
        assert "openai/gpt-4o-mini" in repr_str
        assert "predictions=0" in repr_str

    @pytest.mark.asyncio
    async def test_repr_with_predictions(self):
        """Test __repr__ with predictions made."""
        prediction_response = json.dumps({
            "predicted_output": "Result",
            "confidence": 0.8,
            "reasoning": "Test"
        })

        mock_runner = MockLLMRunner(prediction_response)
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        await predictor.predict_outcome("tool", {}, "")
        await predictor.predict_outcome("tool", {}, "")

        repr_str = repr(predictor)

        assert "DryRunPredictor" in repr_str
        assert "predictions=2" in repr_str


class TestDryRunPredictionDataclass:
    """Test DryRunPrediction dataclass."""

    def test_prediction_creation(self):
        """Test creating DryRunPrediction instance."""
        prediction = DryRunPrediction(
            tool_name="test_tool",
            predicted_output="Expected result",
            confidence=0.85,
            reasoning="Based on analysis"
        )

        assert prediction.tool_name == "test_tool"
        assert prediction.predicted_output == "Expected result"
        assert prediction.confidence == 0.85
        assert prediction.reasoning == "Based on analysis"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_tool_args(self):
        """Test prediction with empty tool arguments."""
        prediction_response = json.dumps({
            "predicted_output": "Result",
            "confidence": 0.5,
            "reasoning": "No arguments provided"
        })

        mock_runner = MockLLMRunner(prediction_response)
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = await predictor.predict_outcome(
            tool_name="tool",
            tool_args={},
            context=""
        )

        assert prediction is not None
        assert prediction.tool_name == "tool"

    @pytest.mark.asyncio
    async def test_empty_context(self):
        """Test prediction with empty context."""
        prediction_response = json.dumps({
            "predicted_output": "Result",
            "confidence": 0.5,
            "reasoning": "No context provided"
        })

        mock_runner = MockLLMRunner(prediction_response)
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = await predictor.predict_outcome(
            tool_name="tool",
            tool_args={"arg": "value"},
            context=""
        )

        assert prediction is not None

    def test_compare_empty_strings(self):
        """Test comparison with empty strings."""
        mock_runner = MockLLMRunner("")
        predictor = DryRunPredictor(mock_runner, "openai/gpt-4o-mini")

        prediction = DryRunPrediction("test", "", 0.5, "")

        similarity = predictor.compare_prediction_to_actual(prediction, "")

        # Empty strings have no words, so similarity should be 0.0
        # (the implementation checks len(pred_words) == 0 or len(actual_words) == 0)
        assert similarity == 0.0


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
