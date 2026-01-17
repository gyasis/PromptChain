"""
Tool Dry Run Prediction for AgenticStepProcessor

This module implements predictive validation where the LLM predicts tool outcomes
before execution. This enables:
1. Transparent reasoning: Agent explains expected results before acting
2. Prediction accuracy tracking: Compare predicted vs actual outcomes
3. Confidence calibration: Track prediction quality over time
4. Debugging assistance: Understand why agent chose specific tools

Key Concepts:
- Dry Run: LLM predicts what a tool will return before execution
- Prediction Comparison: Measure similarity between predicted and actual results
- Confidence Tracking: Monitor prediction accuracy to calibrate agent confidence

Usage:
    predictor = DryRunPredictor(llm_runner, model_name="openai/gpt-4o-mini")

    # Predict outcome before execution
    prediction = await predictor.predict_outcome(
        tool_name="search_database",
        tool_args={"query": "SELECT * FROM users LIMIT 10"},
        context="User requested to list users"
    )

    # Execute tool
    actual_result = execute_tool(...)

    # Compare prediction to actual
    similarity = predictor.compare_prediction_to_actual(prediction, actual_result)
    logger.info(f"Prediction accuracy: {similarity:.2f}")

Research Basis:
- Enhances transparency by forcing agent to articulate expectations
- Helps detect tool misuse (prediction diverges from reality)
- Low overhead (~10-15%) when using fast models for prediction
- Particularly useful for debugging complex agentic workflows
"""

from dataclasses import dataclass
from typing import Dict, Any, Callable, Awaitable
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DryRunPrediction:
    """
    Predicted outcome of tool execution.

    Attributes:
        tool_name: Name of the tool being predicted
        predicted_output: Expected result from tool execution
        confidence: Confidence level in prediction (0.0 to 1.0)
        reasoning: Explanation of why this output is expected
    """
    tool_name: str
    predicted_output: str
    confidence: float
    reasoning: str


class DryRunPredictor:
    """
    Predicts tool outcomes before execution for transparency and validation.

    The predictor asks the LLM to anticipate what a tool will return based
    on the tool's purpose, arguments, and current context. This provides:
    - Transparent reasoning (agent explains expectations)
    - Prediction accuracy metrics (compare predicted vs actual)
    - Debugging insights (understand agent's tool selection logic)
    """

    def __init__(
        self,
        llm_runner: Callable[..., Awaitable[Any]],
        model_name: str
    ):
        """
        Initialize dry run predictor.

        Args:
            llm_runner: Async function that calls LLM (same signature as AgenticStepProcessor.llm_runner)
            model_name: Model to use for prediction (recommend fast/cheap model like gpt-4o-mini)
        """
        self.llm_runner = llm_runner
        self.model_name = model_name
        self.prediction_count = 0
        self.accuracy_history = []
        logger.info(f"[DryRun] Initialized with model: {model_name}")

    async def predict_outcome(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: str
    ) -> DryRunPrediction:
        """
        Predict what a tool will return before execution.

        Asks LLM to anticipate the tool's output based on:
        1. Tool name and purpose
        2. Arguments being passed
        3. Current execution context
        4. Agent's understanding of system state

        Args:
            tool_name: Name of tool to predict
            tool_args: Arguments for the tool call
            context: Current context (e.g., Blackboard state or history)

        Returns:
            DryRunPrediction with expected output, confidence, and reasoning
        """
        logger.debug(f"[DryRun] Predicting outcome for: {tool_name}")

        # Build prediction prompt
        prediction_prompt = self._build_prediction_prompt(
            tool_name=tool_name,
            tool_args=tool_args,
            context=context
        )

        try:
            # Call LLM for prediction
            response = await self.llm_runner(
                messages=[{"role": "user", "content": prediction_prompt}],
                model=self.model_name
            )

            # Extract response content
            response_content = self._extract_response_content(response)

            # Parse prediction result
            prediction = self._parse_prediction_response(response_content, tool_name)

            self.prediction_count += 1
            logger.info(
                f"[DryRun] Predicted {tool_name}: confidence={prediction.confidence:.2f}"
            )

            return prediction

        except Exception as e:
            logger.error(f"[DryRun] Prediction failed: {e}", exc_info=True)
            # Fallback: return low-confidence prediction
            return DryRunPrediction(
                tool_name=tool_name,
                predicted_output="Could not predict outcome",
                confidence=0.0,
                reasoning=f"Prediction error: {str(e)}"
            )

    def _build_prediction_prompt(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: str
    ) -> str:
        """
        Build prediction prompt for LLM.

        Args:
            tool_name: Tool name
            tool_args: Tool arguments
            context: Execution context

        Returns:
            Formatted prediction prompt
        """
        # Truncate context if too long
        max_context_chars = 1500
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "... (truncated)"

        prompt = f"""You are about to execute a tool. Predict what it will return.

CURRENT CONTEXT:
{context}

TOOL TO EXECUTE:
Name: {tool_name}
Arguments: {json.dumps(tool_args, indent=2)}

PREDICTION TASK:
Based on the context and tool arguments, predict what this tool will return.
Consider:
1. What is this tool designed to do?
2. What do the arguments tell you about the expected behavior?
3. Based on the current context, what result makes sense?
4. How confident are you in this prediction? (0.0 = not confident, 1.0 = very confident)

Respond in valid JSON format:
{{
    "predicted_output": "your prediction of the tool's return value",
    "confidence": 0.0 to 1.0,
    "reasoning": "explain why you expect this output"
}}

IMPORTANT: Respond ONLY with valid JSON, no additional text."""

        return prompt

    def _extract_response_content(self, response: Any) -> str:
        """
        Extract content from LLM response.

        Args:
            response: LLM response (dict or object)

        Returns:
            Response content as string
        """
        if isinstance(response, dict):
            return response.get("content", str(response))
        else:
            return getattr(response, "content", str(response))

    def _parse_prediction_response(
        self,
        response_content: str,
        tool_name: str
    ) -> DryRunPrediction:
        """
        Parse prediction response from LLM.

        Attempts to parse JSON response. If parsing fails, falls back to
        creating a low-confidence prediction.

        Args:
            response_content: LLM response content
            tool_name: Tool name being predicted

        Returns:
            DryRunPrediction
        """
        try:
            # Try to extract JSON from response
            json_str = response_content.strip()

            # Remove markdown code fences if present
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else json_str
                json_str = json_str.replace("```json", "").replace("```", "").strip()

            # Parse JSON
            result_dict = json.loads(json_str)

            # Validate and extract fields
            predicted_output = result_dict.get("predicted_output", "Unknown")
            confidence = float(result_dict.get("confidence", 0.5))

            # Clamp confidence to [0.0, 1.0]
            confidence = max(0.0, min(1.0, confidence))

            reasoning = result_dict.get("reasoning", "No reasoning provided")

            return DryRunPrediction(
                tool_name=tool_name,
                predicted_output=predicted_output,
                confidence=confidence,
                reasoning=reasoning
            )

        except json.JSONDecodeError as e:
            logger.warning(f"[DryRun] Failed to parse prediction response as JSON: {e}")
            logger.debug(f"[DryRun] Response content: {response_content[:200]}")

            # Fallback: use response as predicted output
            return DryRunPrediction(
                tool_name=tool_name,
                predicted_output=response_content[:500],
                confidence=0.3,
                reasoning="Parsing failed, used raw response as prediction"
            )

        except Exception as e:
            logger.error(f"[DryRun] Unexpected error parsing prediction: {e}", exc_info=True)

            # Fallback: return minimal prediction
            return DryRunPrediction(
                tool_name=tool_name,
                predicted_output="Could not predict",
                confidence=0.0,
                reasoning=f"Parsing error: {str(e)}"
            )

    def compare_prediction_to_actual(
        self,
        prediction: DryRunPrediction,
        actual_output: str
    ) -> float:
        """
        Compare prediction to actual result.

        Calculates a similarity score between predicted and actual outputs.
        This is a simple implementation using string matching; could be
        enhanced with embeddings for semantic similarity.

        Args:
            prediction: The prediction made before execution
            actual_output: The actual tool result

        Returns:
            Similarity score 0.0-1.0 (1.0 = perfect match)
        """
        pred_lower = prediction.predicted_output.lower()
        actual_lower = actual_output.lower()

        # Handle empty strings first
        if not pred_lower or not actual_lower:
            similarity = 0.0
            self.accuracy_history.append(similarity)
            if len(self.accuracy_history) > 100:
                self.accuracy_history = self.accuracy_history[-100:]
            return similarity

        # Exact substring match (high confidence)
        if pred_lower in actual_lower or actual_lower in pred_lower:
            similarity = 0.9
            logger.debug(f"[DryRun] High similarity (substring match): {similarity:.2f}")
            # Track this comparison too
            self.accuracy_history.append(similarity)
            if len(self.accuracy_history) > 100:
                self.accuracy_history = self.accuracy_history[-100:]
            return similarity

        # Count word overlap (medium confidence)
        pred_words = set(pred_lower.split())
        actual_words = set(actual_lower.split())

        # Remove common words that don't add meaning
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        pred_words -= stop_words
        actual_words -= stop_words

        if len(pred_words) == 0 or len(actual_words) == 0:
            similarity = 0.0
            self.accuracy_history.append(similarity)
            if len(self.accuracy_history) > 100:
                self.accuracy_history = self.accuracy_history[-100:]
            return similarity

        overlap = len(pred_words & actual_words)
        union = len(pred_words | actual_words)

        similarity = overlap / union if union > 0 else 0.0

        logger.debug(f"[DryRun] Word overlap similarity: {similarity:.2f}")

        # Track accuracy history
        self.accuracy_history.append(similarity)

        # Keep last 100 predictions
        if len(self.accuracy_history) > 100:
            self.accuracy_history = self.accuracy_history[-100:]

        return similarity

    def get_accuracy_stats(self) -> Dict[str, Any]:
        """
        Get prediction accuracy statistics.

        Returns:
            Dict with accuracy metrics
        """
        if not self.accuracy_history:
            return {
                "prediction_count": self.prediction_count,
                "comparisons_made": 0,
                "average_accuracy": 0.0,
                "model": self.model_name
            }

        avg_accuracy = sum(self.accuracy_history) / len(self.accuracy_history)

        return {
            "prediction_count": self.prediction_count,
            "comparisons_made": len(self.accuracy_history),
            "average_accuracy": round(avg_accuracy, 3),
            "min_accuracy": round(min(self.accuracy_history), 3),
            "max_accuracy": round(max(self.accuracy_history), 3),
            "model": self.model_name
        }

    def reset_stats(self):
        """Reset accuracy tracking statistics."""
        self.prediction_count = 0
        self.accuracy_history = []
        logger.info("[DryRun] Reset statistics")

    def __repr__(self) -> str:
        """String representation for debugging."""
        stats = self.get_accuracy_stats()
        return (
            f"DryRunPredictor("
            f"model={self.model_name}, "
            f"predictions={stats['prediction_count']}, "
            f"avg_accuracy={stats['average_accuracy']:.2f})"
        )
