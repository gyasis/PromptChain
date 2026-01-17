"""LightRAG Branching Thoughts Pattern.

This pattern generates multiple hypotheses using LightRAG's local/global/hybrid
query modes and uses an LLM judge to evaluate and select the best one.

The pattern explores different reasoning paths in parallel:
- Local query: Entity-specific hypothesis focusing on concrete details
- Global query: High-level conceptual hypothesis
- Hybrid query: Balanced combination of entity and concept

An LLM judge evaluates each hypothesis based on:
- Completeness of answer
- Relevance to the problem
- Internal consistency
- Supporting evidence quality
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union
import asyncio
import json
import logging
import time
import uuid

from promptchain.patterns.base import BasePattern, PatternConfig, PatternResult
from promptchain.integrations.lightrag import LIGHTRAG_AVAILABLE
from promptchain.observability import track_llm_call

if TYPE_CHECKING:
    from hybridrag.src.lightrag_core import HybridLightRAGCore
    from promptchain.integrations.lightrag.core import LightRAGIntegration

logger = logging.getLogger(__name__)


@dataclass
class BranchingConfig(PatternConfig):
    """Configuration for Branching Thoughts pattern.

    Attributes:
        hypothesis_count: Number of hypotheses to generate (default: 3).
        generator_model: Model to use for hypothesis generation (None = use LightRAG default).
        judge_model: Model to use for judging hypotheses (default: "openai/gpt-4o").
        diversity_threshold: Minimum diversity between hypotheses (0.0-1.0).
        record_outcomes: Whether to record outcomes for future learning.
    """
    hypothesis_count: int = 3
    generator_model: Optional[str] = None
    judge_model: str = "openai/gpt-4o"
    diversity_threshold: float = 0.3
    record_outcomes: bool = True


@dataclass
class Hypothesis:
    """A single hypothesis generated during branching.

    Attributes:
        hypothesis_id: Unique identifier for this hypothesis.
        approach: Description of the reasoning approach.
        reasoning: Detailed reasoning for this hypothesis.
        confidence: Confidence score (0.0-1.0).
        mode: Query mode used ("local", "global", "hybrid").
    """
    hypothesis_id: str
    approach: str
    reasoning: str
    confidence: float
    mode: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "approach": self.approach,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "mode": self.mode,
        }


@dataclass
class HypothesisScore:
    """Score and evaluation for a hypothesis.

    Attributes:
        hypothesis_id: ID of the hypothesis being scored.
        score: Numerical score (0.0-1.0).
        reasoning: Explanation of the score.
        strengths: List of identified strengths.
        weaknesses: List of identified weaknesses.
    """
    hypothesis_id: str
    score: float
    reasoning: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "score": self.score,
            "reasoning": self.reasoning,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
        }


@dataclass
class BranchingResult(PatternResult):
    """Result from Branching Thoughts pattern execution.

    Attributes:
        hypotheses: List of generated hypotheses.
        scores: List of hypothesis scores.
        selected_hypothesis: The hypothesis chosen as best.
        selection_reasoning: Explanation of why it was selected.
    """
    hypotheses: List[Hypothesis] = field(default_factory=list)
    scores: List[HypothesisScore] = field(default_factory=list)
    selected_hypothesis: Optional[Hypothesis] = None
    selection_reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "scores": [s.to_dict() for s in self.scores],
            "selected_hypothesis": self.selected_hypothesis.to_dict() if self.selected_hypothesis else None,
            "selection_reasoning": self.selection_reasoning,
        })
        return base_dict


class LightRAGBranchingThoughts(BasePattern):
    """Branching Thoughts pattern using LightRAG query modes.

    This pattern generates multiple hypotheses by querying LightRAG in
    different modes (local, global, hybrid) to explore diverse reasoning
    paths. An LLM judge evaluates each hypothesis and selects the best one.

    Example:
        >>> from promptchain.integrations.lightrag import LightRAGIntegration
        >>> integration = LightRAGIntegration()
        >>> branching = LightRAGBranchingThoughts(
        ...     lightrag_core=integration,
        ...     config=BranchingConfig(hypothesis_count=3)
        ... )
        >>> result = await branching.execute(
        ...     problem="What are the key factors in climate change?"
        ... )
        >>> print(result.selected_hypothesis.reasoning)
    """

    def __init__(
        self,
        lightrag_core: Union["HybridLightRAGCore", "LightRAGIntegration"],
        config: Optional[BranchingConfig] = None
    ):
        """Initialize the Branching Thoughts pattern.

        Args:
            lightrag_core: Either HybridLightRAGCore or LightRAGIntegration instance.
            config: Configuration for the pattern.

        Raises:
            ImportError: If hybridrag is not installed.
        """
        if not LIGHTRAG_AVAILABLE:
            raise ImportError(
                "hybridrag is not installed. Install with: "
                "pip install git+https://github.com/gyasis/hybridrag.git"
            )

        super().__init__(config or BranchingConfig())
        self.config: BranchingConfig = self.config  # type: ignore
        self._lightrag = lightrag_core

        # Import LiteLLM for judge
        try:
            import litellm
            self._litellm = litellm
        except ImportError:
            raise ImportError(
                "litellm is required for hypothesis judging. "
                "Install with: pip install litellm"
            )

    async def execute(
        self,
        problem: str,
        hypothesis_count: Optional[int] = None
    ) -> BranchingResult:
        """Execute the Branching Thoughts pattern.

        Generates multiple hypotheses using different LightRAG query modes
        and selects the best one using an LLM judge.

        Args:
            problem: The problem to solve or question to answer.
            hypothesis_count: Number of hypotheses to generate (overrides config).

        Returns:
            BranchingResult with hypotheses, scores, and selected hypothesis.
        """
        start_time = time.perf_counter()
        count = hypothesis_count or self.config.hypothesis_count

        self.emit_event("pattern.branching.started", {
            "problem": problem,
            "hypothesis_count": count,
        })

        try:
            # Generate hypotheses using different query modes
            hypotheses = await self._generate_hypotheses(problem, count)

            # Judge and score each hypothesis
            scores = await self._judge_hypotheses(problem, hypotheses)

            # Select the best hypothesis
            selected, reasoning = self._select_best_hypothesis(hypotheses, scores)

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            self.emit_event("pattern.branching.completed", {
                "selected_hypothesis_id": selected.hypothesis_id if selected else None,
                "execution_time_ms": execution_time_ms,
            })

            # Share result via Blackboard if enabled
            if selected:
                self.share_result(f"branching.{self.config.pattern_id}.selected", {
                    "hypothesis": selected.to_dict(),
                    "reasoning": reasoning,
                })

            return BranchingResult(
                pattern_id=self.config.pattern_id,
                success=True,
                result=selected.reasoning if selected else "",
                execution_time_ms=execution_time_ms,
                hypotheses=hypotheses,
                scores=scores,
                selected_hypothesis=selected,
                selection_reasoning=reasoning,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Branching pattern failed: {e}")

            return BranchingResult(
                pattern_id=self.config.pattern_id,
                success=False,
                result="",
                execution_time_ms=execution_time_ms,
                errors=[str(e)],
            )

    async def _generate_hypotheses(
        self,
        problem: str,
        count: int
    ) -> List[Hypothesis]:
        """Generate hypotheses using different query modes.

        Args:
            problem: The problem to solve.
            count: Number of hypotheses to generate.

        Returns:
            List of generated hypotheses.
        """
        # Determine which query modes to use
        modes = ["local", "global", "hybrid"]
        if count < 3:
            modes = modes[:count]
        elif count > 3:
            # Repeat modes to reach desired count
            modes = modes * (count // 3 + 1)
            modes = modes[:count]

        # Run queries in parallel
        tasks = []
        for mode in modes:
            tasks.append(self._query_with_mode(problem, mode))

        self.emit_event("pattern.branching.generating", {
            "modes": modes,
            "count": len(tasks),
        })

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert results to hypotheses
        hypotheses = []
        for i, (mode, result) in enumerate(zip(modes, results)):
            if isinstance(result, Exception):
                logger.warning(f"Query mode {mode} failed: {result}")
                continue

            hypothesis = self._extract_hypothesis(mode, result, i)
            hypotheses.append(hypothesis)

            self.emit_event("pattern.branching.hypothesis_generated", {
                "hypothesis_id": hypothesis.hypothesis_id,
                "mode": mode,
                "confidence": hypothesis.confidence,
            })

        return hypotheses

    async def _query_with_mode(self, problem: str, mode: str) -> Any:
        """Execute a query using the specified mode.

        Args:
            problem: The problem to query.
            mode: Query mode ("local", "global", "hybrid").

        Returns:
            Query result from LightRAG.
        """
        # Check if we have a LightRAGIntegration or HybridLightRAGCore
        if hasattr(self._lightrag, "local_query"):
            # Direct HybridLightRAGCore or LightRAGIntegration
            if mode == "local":
                return await self._lightrag.local_query(problem)
            elif mode == "global":
                return await self._lightrag.global_query(problem)
            else:  # hybrid
                return await self._lightrag.hybrid_query(problem)
        else:
            # Assume it's a core object with query method
            return await self._lightrag.query(problem, mode=mode)

    def _extract_hypothesis(
        self,
        mode: str,
        result: Any,
        index: int
    ) -> Hypothesis:
        """Extract a hypothesis from a query result.

        Args:
            mode: Query mode used.
            result: Result from LightRAG query.
            index: Index of this hypothesis.

        Returns:
            Hypothesis object.
        """
        # Extract the reasoning from result
        # LightRAG results can be strings or dicts
        if isinstance(result, str):
            reasoning = result
        elif isinstance(result, dict):
            reasoning = result.get("response", result.get("answer", str(result)))
        else:
            reasoning = str(result)

        # Generate approach description
        approach_map = {
            "local": "Entity-focused analysis examining specific details",
            "global": "High-level conceptual overview",
            "hybrid": "Balanced combination of entity details and concepts",
        }

        return Hypothesis(
            hypothesis_id=str(uuid.uuid4()),
            approach=approach_map.get(mode, f"Query mode: {mode}"),
            reasoning=reasoning,
            confidence=0.7,  # Default confidence, will be adjusted by judge
            mode=mode,
        )

    @track_llm_call(
        model_param="judge_model",
        extract_args=["temperature"]
    )
    async def _judge_hypotheses(
        self,
        problem: str,
        hypotheses: List[Hypothesis]
    ) -> List[HypothesisScore]:
        """Judge and score each hypothesis using an LLM.

        Args:
            problem: The original problem.
            hypotheses: List of hypotheses to judge.

        Returns:
            List of HypothesisScore objects.
        """
        self.emit_event("pattern.branching.judging", {
            "hypothesis_count": len(hypotheses),
        })

        # Create judge prompt
        judge_prompt = self._create_judge_prompt(problem, hypotheses)

        # Call LLM judge
        try:
            response = await self._litellm.acompletion(
                model=self.config.judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert judge evaluating different hypotheses. "
                                   "Analyze each hypothesis carefully and provide detailed scoring."
                    },
                    {
                        "role": "user",
                        "content": judge_prompt
                    }
                ],
                temperature=0.2,  # Low temperature for consistent judging
            )

            judge_output = response.choices[0].message.content

            # Parse judge output into scores
            scores = self._parse_judge_output(judge_output, hypotheses)

            return scores

        except Exception as e:
            logger.error(f"LLM judge failed: {e}")
            # Fallback to simple scoring
            return [
                HypothesisScore(
                    hypothesis_id=h.hypothesis_id,
                    score=h.confidence,
                    reasoning="LLM judge unavailable, using default confidence",
                    strengths=["Generated successfully"],
                    weaknesses=["Not evaluated by LLM judge"],
                )
                for h in hypotheses
            ]

    def _create_judge_prompt(
        self,
        problem: str,
        hypotheses: List[Hypothesis]
    ) -> str:
        """Create a prompt for the LLM judge.

        Args:
            problem: The original problem.
            hypotheses: Hypotheses to judge.

        Returns:
            Judge prompt string.
        """
        prompt = f"""You are evaluating different hypotheses for the following problem:

PROBLEM: {problem}

Please evaluate each hypothesis below based on:
1. Completeness: How fully does it answer the problem?
2. Relevance: How directly does it address the problem?
3. Consistency: Is the reasoning internally consistent?
4. Evidence: How well is it supported by evidence?

HYPOTHESES:

"""
        for i, h in enumerate(hypotheses, 1):
            prompt += f"""
Hypothesis {i} (ID: {h.hypothesis_id}, Mode: {h.mode})
Approach: {h.approach}
Reasoning: {h.reasoning}

---
"""

        prompt += """
For each hypothesis, provide:
1. A score from 0.0 to 1.0
2. Reasoning for the score
3. 2-3 key strengths
4. 2-3 key weaknesses

Format your response as JSON:
{
  "evaluations": [
    {
      "hypothesis_id": "...",
      "score": 0.85,
      "reasoning": "...",
      "strengths": ["...", "...", "..."],
      "weaknesses": ["...", "...", "..."]
    },
    ...
  ]
}
"""
        return prompt

    def _parse_judge_output(
        self,
        output: str,
        hypotheses: List[Hypothesis]
    ) -> List[HypothesisScore]:
        """Parse the judge's output into scores.

        Args:
            output: LLM judge output.
            hypotheses: Original hypotheses.

        Returns:
            List of HypothesisScore objects.
        """
        try:
            # Try to extract JSON from output
            # Look for JSON block
            start = output.find("{")
            end = output.rfind("}") + 1
            if start != -1 and end > start:
                json_str = output[start:end]
                data = json.loads(json_str)

                scores = []
                for eval_data in data.get("evaluations", []):
                    scores.append(HypothesisScore(
                        hypothesis_id=eval_data["hypothesis_id"],
                        score=float(eval_data["score"]),
                        reasoning=eval_data.get("reasoning", ""),
                        strengths=eval_data.get("strengths", []),
                        weaknesses=eval_data.get("weaknesses", []),
                    ))
                return scores

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse judge output: {e}")

        # Fallback to default scores
        return [
            HypothesisScore(
                hypothesis_id=h.hypothesis_id,
                score=h.confidence,
                reasoning="Failed to parse judge output",
                strengths=[],
                weaknesses=[],
            )
            for h in hypotheses
        ]

    def _select_best_hypothesis(
        self,
        hypotheses: List[Hypothesis],
        scores: List[HypothesisScore]
    ) -> tuple[Optional[Hypothesis], str]:
        """Select the best hypothesis based on scores.

        Args:
            hypotheses: List of hypotheses.
            scores: List of scores.

        Returns:
            Tuple of (selected_hypothesis, selection_reasoning).
        """
        if not scores:
            return None, "No scores available"

        # Create mapping of ID to score
        score_map = {s.hypothesis_id: s for s in scores}

        # Find hypothesis with highest score
        best_score = max(scores, key=lambda s: s.score)
        best_hypothesis = next(
            (h for h in hypotheses if h.hypothesis_id == best_score.hypothesis_id),
            None
        )

        if best_hypothesis is None:
            return None, "Could not find matching hypothesis"

        reasoning = (
            f"Selected hypothesis from {best_hypothesis.mode} mode "
            f"with score {best_score.score:.2f}. "
            f"Reasoning: {best_score.reasoning}"
        )

        self.emit_event("pattern.branching.selected", {
            "hypothesis_id": best_hypothesis.hypothesis_id,
            "mode": best_hypothesis.mode,
            "score": best_score.score,
        })

        return best_hypothesis, reasoning
