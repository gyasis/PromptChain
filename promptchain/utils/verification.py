"""
Chain of Verification (CoVe) for AgenticStepProcessor

This module implements pre-execution verification that asks the LLM to validate
tool calls before execution. Research shows this pattern can reduce errors by 50%
by catching invalid assumptions, risky parameters, and low-confidence actions.

Key Concepts:
- Chain of Verification (CoVe): Multi-step verification before tool execution
- Epistemic Awareness: LLM explicitly states assumptions and confidence
- Risk Assessment: Identifies potential failure modes before execution
- Suggested Modifications: LLM can propose safer parameter values

Usage:
    verifier = CoVeVerifier(llm_runner, model_name="openai/gpt-4o-mini")

    result = await verifier.verify_tool_call(
        tool_name="delete_file",
        tool_args={"path": "/important/data.db"},
        context="User asked to clean up temporary files",
        available_tools=[...]
    )

    if result.should_execute and result.confidence >= 0.7:
        # Execute tool
        pass
    else:
        # Skip or modify tool call
        logger.warning(f"Skipped {tool_name}: {result.verification_reasoning}")

Research Basis:
- CoVe paper: "Chain-of-Verification Reduces Hallucination in Large Language Models"
- Achieves 40-50% error reduction with only ~10% overhead
- Particularly effective for:
  - File operations (delete, overwrite)
  - API calls with side effects
  - Database modifications
  - System commands
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """
    Result of Chain of Verification (CoVe) check.

    Attributes:
        should_execute: Whether tool should be executed
        confidence: Confidence level (0.0 to 1.0)
        assumptions: List of assumptions being made
        risks: List of potential risks or failure modes
        verification_reasoning: Explanation of verification decision
        suggested_modifications: Optional dict of parameter modifications
    """
    should_execute: bool
    confidence: float
    assumptions: List[str]
    risks: List[str]
    verification_reasoning: str
    suggested_modifications: Optional[Dict[str, Any]] = None


class CoVeVerifier:
    """
    Chain of Verification - pre-execution validation.

    Implements the CoVe pattern where the LLM explicitly verifies tool calls
    before execution by considering:
    1. What assumptions am I making?
    2. What could go wrong?
    3. How confident am I?
    4. Should we proceed?

    This reduces errors by catching invalid assumptions and risky operations
    before they execute.
    """

    def __init__(self, llm_runner: Callable, model_name: str):
        """
        Initialize CoVe verifier.

        Args:
            llm_runner: Async function that calls LLM (same signature as AgenticStepProcessor.llm_runner)
            model_name: Model to use for verification (can be cheaper/faster than primary model)
        """
        self.llm_runner = llm_runner
        self.model_name = model_name
        logger.info(f"[CoVe] Initialized with model: {model_name}")

    async def verify_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: str,
        available_tools: List[Dict]
    ) -> VerificationResult:
        """
        Verify tool call before execution.

        Asks LLM to consider:
        1. What assumptions are being made about inputs?
        2. What could go wrong with this execution?
        3. Is this the right tool for the objective?
        4. How confident am I this will succeed? (0.0 to 1.0)

        Args:
            tool_name: Name of tool to verify
            tool_args: Arguments for the tool call
            context: Current context (e.g., Blackboard state or history)
            available_tools: List of tool schemas

        Returns:
            VerificationResult with decision and reasoning
        """
        logger.debug(f"[CoVe] Verifying tool call: {tool_name}")

        # Find tool schema
        tool_schema = self._find_tool_schema(tool_name, available_tools)

        # Build verification prompt
        verification_prompt = self._build_verification_prompt(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_schema=tool_schema,
            context=context
        )

        try:
            # Call LLM for verification
            response = await self.llm_runner(
                messages=[{"role": "user", "content": verification_prompt}],
                model=self.model_name
            )

            # Extract response content
            response_content = self._extract_response_content(response)

            # Parse verification result
            result = self._parse_verification_response(response_content)

            logger.info(
                f"[CoVe] {tool_name}: should_execute={result.should_execute}, "
                f"confidence={result.confidence:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"[CoVe] Verification failed: {e}", exc_info=True)
            # Fallback: allow execution but log warning
            return VerificationResult(
                should_execute=True,
                confidence=0.5,
                assumptions=[],
                risks=[f"Verification error: {str(e)}"],
                verification_reasoning="Verification failed, allowing execution by default"
            )

    def _build_verification_prompt(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_schema: Dict,
        context: str
    ) -> str:
        """
        Build verification prompt for LLM.

        Args:
            tool_name: Tool name
            tool_args: Tool arguments
            tool_schema: Tool schema/signature
            context: Current execution context

        Returns:
            Formatted verification prompt
        """
        # Truncate context if too long
        max_context_chars = 2000
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "... (truncated)"

        prompt = f"""You are about to execute a tool. Before execution, carefully verify this is the right action.

CURRENT CONTEXT:
{context}

TOOL TO EXECUTE:
Name: {tool_name}
Arguments: {json.dumps(tool_args, indent=2)}
Schema: {json.dumps(tool_schema, indent=2)}

VERIFICATION CHECKLIST (answer each question):
1. What assumptions am I making about the inputs or current state?
2. What could go wrong with this execution? What are the risks?
3. Is this the right tool for achieving the objective?
4. How confident am I that this will succeed? (0.0 = not confident, 1.0 = very confident)
5. Should we proceed with execution?

Think through each question carefully, then respond in valid JSON format:
{{
    "should_execute": true or false,
    "confidence": 0.0 to 1.0,
    "assumptions": ["assumption 1", "assumption 2", ...],
    "risks": ["risk 1", "risk 2", ...],
    "reasoning": "clear explanation of your decision",
    "suggested_modifications": {{"param_name": "safer_value"}} or null
}}

IMPORTANT: Respond ONLY with valid JSON, no additional text."""

        return prompt

    def _find_tool_schema(self, tool_name: str, tools: List[Dict]) -> Dict:
        """
        Find tool schema from available tools.

        Args:
            tool_name: Tool name to find
            tools: List of tool definitions

        Returns:
            Tool schema dict (or empty dict if not found)
        """
        for tool in tools:
            if tool.get("function", {}).get("name") == tool_name:
                return tool.get("function", {})

        logger.warning(f"[CoVe] Tool schema not found for: {tool_name}")
        return {}

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

    def _parse_verification_response(self, response_content: str) -> VerificationResult:
        """
        Parse verification response from LLM.

        Attempts to parse JSON response. If parsing fails, falls back to
        allowing execution with low confidence.

        Args:
            response_content: LLM response content

        Returns:
            VerificationResult
        """
        try:
            # Try to extract JSON from response
            # Look for JSON block between ``` markers or parse directly
            json_str = response_content.strip()

            # Remove markdown code fences if present
            if json_str.startswith("```"):
                # Extract content between ``` markers
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else json_str
                json_str = json_str.replace("```json", "").replace("```", "").strip()

            # Parse JSON
            result_dict = json.loads(json_str)

            # Validate and extract fields
            should_execute = bool(result_dict.get("should_execute", True))
            confidence = float(result_dict.get("confidence", 0.5))

            # Clamp confidence to [0.0, 1.0]
            confidence = max(0.0, min(1.0, confidence))

            assumptions = result_dict.get("assumptions", [])
            if not isinstance(assumptions, list):
                assumptions = [str(assumptions)]

            risks = result_dict.get("risks", [])
            if not isinstance(risks, list):
                risks = [str(risks)]

            reasoning = result_dict.get("reasoning", "No reasoning provided")
            suggested_modifications = result_dict.get("suggested_modifications")

            return VerificationResult(
                should_execute=should_execute,
                confidence=confidence,
                assumptions=assumptions,
                risks=risks,
                verification_reasoning=reasoning,
                suggested_modifications=suggested_modifications
            )

        except json.JSONDecodeError as e:
            logger.warning(f"[CoVe] Failed to parse verification response as JSON: {e}")
            logger.debug(f"[CoVe] Response content: {response_content[:200]}")

            # Fallback: use heuristics on response text
            response_lower = response_content.lower()

            # Check for negative indicators
            should_execute = not any(
                phrase in response_lower
                for phrase in ["do not execute", "should not", "dangerous", "risky", "abort"]
            )

            # Default to medium-low confidence when parsing fails
            confidence = 0.4

            return VerificationResult(
                should_execute=should_execute,
                confidence=confidence,
                assumptions=[],
                risks=["Could not parse verification response"],
                verification_reasoning=f"Parsing failed, used heuristics. Response: {response_content[:100]}"
            )

        except Exception as e:
            logger.error(f"[CoVe] Unexpected error parsing verification: {e}", exc_info=True)

            # Fallback: allow execution but with low confidence
            return VerificationResult(
                should_execute=True,
                confidence=0.3,
                assumptions=[],
                risks=[f"Parsing error: {str(e)}"],
                verification_reasoning="Error during parsing, allowing execution by default"
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get verification statistics.

        Returns:
            Dict with verification stats (for future extension)
        """
        # Future: track verification counts, approval rates, etc.
        return {
            "model": self.model_name,
            "status": "active"
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"CoVeVerifier(model={self.model_name})"
