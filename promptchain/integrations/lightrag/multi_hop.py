"""Multi-hop retrieval pattern for LightRAG.

This pattern decomposes complex questions into sub-questions and uses multi-hop
reasoning via LightRAG's agentic_search to iteratively build comprehensive answers.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from promptchain.observability import track_llm_call
from promptchain.patterns.base import BasePattern, PatternConfig, PatternResult

logger = logging.getLogger(__name__)

# Check if hybridrag is available
try:
    from hybridrag.src.search_interface import SearchInterface

    HYBRIDRAG_AVAILABLE = True
except ImportError:
    HYBRIDRAG_AVAILABLE = False
    SearchInterface = None  # type: ignore


@dataclass
class SubQuestion:
    """Represents a sub-question in multi-hop reasoning.

    Attributes:
        question_id: Unique identifier for this sub-question.
        question_text: The actual question text.
        parent_question: ID of the parent question (if any).
        dependencies: IDs of sub-questions this depends on.
        rationale: Why this sub-question is needed.
        answer: The answer to this sub-question (populated after retrieval).
        retrieval_context: Retrieved context chunks used to answer this question.
    """

    question_id: str
    question_text: str
    parent_question: str
    dependencies: List[str] = field(default_factory=list)
    rationale: str = ""
    answer: Optional[str] = None
    retrieval_context: List[str] = field(default_factory=list)


@dataclass
class MultiHopConfig(PatternConfig):
    """Configuration for multi-hop retrieval pattern.

    Attributes:
        max_hops: Maximum number of retrieval hops to perform.
        max_sub_questions: Maximum number of sub-questions to generate.
        synthesizer_model: Model to use for answer synthesis (uses default if None).
        decompose_first: Whether to decompose the question before retrieval.
    """

    max_hops: int = 5
    max_sub_questions: int = 5
    synthesizer_model: Optional[str] = None
    decompose_first: bool = True


@dataclass
class MultiHopResult(PatternResult):
    """Result from multi-hop retrieval execution.

    Attributes:
        original_question: The original complex question.
        sub_questions: List of generated sub-questions.
        sub_answers: Dictionary mapping question_id to answer.
        unified_answer: Final synthesized answer.
        hops_executed: Number of retrieval hops actually executed.
        unanswered_aspects: List of aspects that couldn't be answered.
    """

    original_question: str = ""
    sub_questions: List[SubQuestion] = field(default_factory=list)
    sub_answers: Dict[str, str] = field(default_factory=dict)
    unified_answer: str = ""
    hops_executed: int = 0
    unanswered_aspects: List[str] = field(default_factory=list)


class LightRAGMultiHop(BasePattern):
    """Multi-hop retrieval pattern using LightRAG's agentic search.

    This pattern implements complex question answering by:
    1. Decomposing complex questions into sub-questions (optional)
    2. Executing multi-hop retrieval via SearchInterface.agentic_search()
    3. Tracking reasoning hops and intermediate results
    4. Synthesizing a unified answer from sub-answers

    Example:
        >>> from hybridrag.src.search_interface import SearchInterface
        >>> from promptchain.integrations.lightrag.core import LightRAGIntegration
        >>>
        >>> integration = LightRAGIntegration()
        >>> pattern = LightRAGMultiHop(
        ...     search_interface=integration.search,
        ...     config=MultiHopConfig(max_hops=5, decompose_first=True)
        ... )
        >>> result = await pattern.execute(
        ...     question="What are the key differences between transformers and RNNs?"
        ... )
        >>> print(result.unified_answer)
    """

    def __init__(
        self,
        search_interface: "SearchInterface",
        config: Optional[MultiHopConfig] = None,
    ):
        """Initialize the multi-hop retrieval pattern.

        Args:
            search_interface: SearchInterface instance from hybridrag.
            config: Multi-hop configuration. Uses defaults if not provided.

        Raises:
            ImportError: If hybridrag is not installed.
        """
        if not HYBRIDRAG_AVAILABLE:
            raise ImportError(
                "hybridrag is not installed. Install with: "
                "pip install git+https://github.com/gyasis/hybridrag.git"
            )

        super().__init__(config or MultiHopConfig())
        self.search_interface = search_interface
        self.multi_hop_config = self.config if isinstance(self.config, MultiHopConfig) else MultiHopConfig()

    async def execute(self, question: str, **kwargs) -> MultiHopResult:
        """Execute multi-hop retrieval for a complex question.

        Args:
            question: The complex question to answer.
            **kwargs: Additional arguments (e.g., override max_hops).

        Returns:
            MultiHopResult with sub-questions, answers, and unified response.
        """
        start_time = time.perf_counter()

        self.emit_event("pattern.multi_hop.started", {
            "question": question,
            "decompose_first": self.multi_hop_config.decompose_first,
        })

        try:
            # Step 1: Optionally decompose question into sub-questions
            sub_questions = []
            if self.multi_hop_config.decompose_first:
                sub_questions = await self._decompose_question(question)
                self.emit_event("pattern.multi_hop.decomposed", {
                    "num_sub_questions": len(sub_questions),
                    "sub_questions": [sq.question_text for sq in sub_questions],
                })

            # Step 2: Execute multi-hop retrieval via agentic_search
            max_hops = kwargs.get("max_hops", self.multi_hop_config.max_hops)
            agentic_result = await self._execute_agentic_search(
                question=question,
                sub_questions=sub_questions,
                max_hops=max_hops,
            )

            # Step 3: Extract hops executed and sub-answers
            hops_executed = agentic_result.get("hops_executed", 0)
            sub_answers = agentic_result.get("sub_answers", {})

            self.emit_event("pattern.multi_hop.hop_completed", {
                "hops_executed": hops_executed,
                "num_sub_answers": len(sub_answers),
            })

            # Step 4: Synthesize unified answer
            self.emit_event("pattern.multi_hop.synthesizing", {
                "num_sub_answers": len(sub_answers),
            })

            unified_answer = await self._synthesize_answer(
                question=question,
                sub_questions=sub_questions,
                sub_answers=sub_answers,
            )

            # Step 5: Identify unanswered aspects
            unanswered_aspects = self._identify_unanswered_aspects(
                sub_questions=sub_questions,
                sub_answers=sub_answers,
            )

            # Update sub-questions with answers
            for sq in sub_questions:
                if sq.question_id in sub_answers:
                    sq.answer = sub_answers[sq.question_id]

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            self.emit_event("pattern.multi_hop.completed", {
                "hops_executed": hops_executed,
                "num_unanswered": len(unanswered_aspects),
                "execution_time_ms": execution_time_ms,
            })

            result = MultiHopResult(
                pattern_id=self.config.pattern_id,
                success=True,
                result=unified_answer,
                execution_time_ms=execution_time_ms,
                original_question=question,
                sub_questions=sub_questions,
                sub_answers=sub_answers,
                unified_answer=unified_answer,
                hops_executed=hops_executed,
                unanswered_aspects=unanswered_aspects,
                metadata={
                    "decompose_first": self.multi_hop_config.decompose_first,
                    "max_hops_configured": max_hops,
                },
            )

            # Share result via Blackboard if enabled
            if self.config.use_blackboard:
                self.share_result(f"multi_hop_result_{self.config.pattern_id}", result)

            return result

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Multi-hop retrieval failed: {e}")

            return MultiHopResult(
                pattern_id=self.config.pattern_id,
                success=False,
                result=None,
                execution_time_ms=execution_time_ms,
                original_question=question,
                errors=[str(e)],
            )

    @track_llm_call(
        model_param="model_name",
        extract_args=["temperature", "max_tokens"]
    )
    async def _decompose_question(self, question: str) -> List[SubQuestion]:
        """Decompose a complex question into sub-questions using LLM.

        Args:
            question: The complex question to decompose.

        Returns:
            List of SubQuestion instances.
        """
        try:
            from litellm import acompletion
        except ImportError:
            logger.warning("litellm not installed, skipping decomposition")
            return []

        decomposition_prompt = f"""Decompose the following complex question into 3-5 sub-questions that would help answer it comprehensively.

For each sub-question, provide:
1. The question text
2. A brief rationale for why it's needed
3. Any dependencies on other sub-questions

Complex question: {question}

Return the sub-questions in JSON format:
[
  {{
    "question_id": "sq1",
    "question_text": "What is X?",
    "rationale": "Need to understand X before Y",
    "dependencies": []
  }},
  ...
]
"""

        try:
            model = self.multi_hop_config.synthesizer_model or "openai/gpt-4o-mini"
            response = await acompletion(
                model=model,
                messages=[{"role": "user", "content": decomposition_prompt}],
                response_format={"type": "json_object"},
            )

            import json
            sub_questions_data = json.loads(response.choices[0].message.content)

            # Handle both list and dict with "sub_questions" key
            if isinstance(sub_questions_data, dict):
                sub_questions_data = sub_questions_data.get("sub_questions", [])

            sub_questions = []
            for idx, sq_data in enumerate(sub_questions_data[:self.multi_hop_config.max_sub_questions]):
                sub_questions.append(
                    SubQuestion(
                        question_id=sq_data.get("question_id", f"sq{idx+1}"),
                        question_text=sq_data.get("question_text", ""),
                        parent_question=question,
                        dependencies=sq_data.get("dependencies", []),
                        rationale=sq_data.get("rationale", ""),
                    )
                )

            return sub_questions

        except Exception as e:
            logger.warning(f"Question decomposition failed: {e}, proceeding without decomposition")
            return []

    async def _execute_agentic_search(
        self,
        question: str,
        sub_questions: List[SubQuestion],
        max_hops: int,
    ) -> Dict[str, Any]:
        """Execute agentic search via SearchInterface.

        Args:
            question: The main question to answer.
            sub_questions: List of sub-questions (may be empty).
            max_hops: Maximum number of reasoning hops.

        Returns:
            Dictionary with hops_executed and sub_answers.
        """
        objective = f"Comprehensively answer: {question}"
        if sub_questions:
            objective += "\n\nSub-questions to address:\n"
            for sq in sub_questions:
                objective += f"- {sq.question_text}\n"

        try:
            # Call SearchInterface.agentic_search
            agentic_result = await self.search_interface.agentic_search(
                query=question,
                objective=objective,
                max_steps=max_hops,
            )

            # Extract hop information from result
            # The actual structure depends on hybridrag's implementation
            # Assuming it returns a dict with 'answer' and 'steps_taken'
            hops_executed = agentic_result.get("steps_taken", 0)
            main_answer = agentic_result.get("answer", "")

            # Map sub-questions to answers (simplified - would need more sophisticated parsing)
            sub_answers = {}
            if main_answer:
                if sub_questions:
                    # For now, map the main answer to all sub-questions
                    # In a real implementation, would parse the answer to extract sub-answers
                    for sq in sub_questions:
                        sub_answers[sq.question_id] = main_answer
                else:
                    # No sub-questions, but we have an answer - create a synthetic entry
                    sub_answers["main"] = main_answer

            return {
                "hops_executed": hops_executed,
                "sub_answers": sub_answers,
                "main_answer": main_answer,
            }

        except Exception as e:
            logger.error(f"Agentic search failed: {e}")
            return {
                "hops_executed": 0,
                "sub_answers": {},
                "main_answer": "",
            }

    async def _synthesize_answer(
        self,
        question: str,
        sub_questions: List[SubQuestion],
        sub_answers: Dict[str, str],
    ) -> str:
        """Synthesize a unified answer from sub-answers.

        Args:
            question: The original question.
            sub_questions: List of sub-questions.
            sub_answers: Dictionary of sub-question answers.

        Returns:
            Synthesized unified answer.
        """
        if not sub_answers:
            return "Unable to generate answer due to retrieval failure."

        # If we only have one answer (from agentic_search), return it directly
        unique_answers = set(sub_answers.values())
        if len(unique_answers) == 1:
            return list(unique_answers)[0]

        # Otherwise, use LLM to synthesize multiple sub-answers
        try:
            from litellm import acompletion
        except ImportError:
            # Fallback: concatenate sub-answers
            return "\n\n".join(sub_answers.values())

        synthesis_prompt = f"""Synthesize a comprehensive answer to the following question using the provided sub-answers.

Original question: {question}

Sub-answers:
"""
        for sq in sub_questions:
            if sq.question_id in sub_answers:
                synthesis_prompt += f"\nQ: {sq.question_text}\nA: {sub_answers[sq.question_id]}\n"

        synthesis_prompt += "\nProvide a unified, coherent answer that integrates all sub-answers:"

        try:
            model = self.multi_hop_config.synthesizer_model or "openai/gpt-4o-mini"
            response = await acompletion(
                model=model,
                messages=[{"role": "user", "content": synthesis_prompt}],
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.warning(f"Answer synthesis failed: {e}, using concatenated answers")
            return "\n\n".join(sub_answers.values())

    def _identify_unanswered_aspects(
        self,
        sub_questions: List[SubQuestion],
        sub_answers: Dict[str, str],
    ) -> List[str]:
        """Identify aspects of the question that weren't answered.

        Args:
            sub_questions: List of sub-questions.
            sub_answers: Dictionary of sub-question answers.

        Returns:
            List of unanswered question texts.
        """
        unanswered = []
        for sq in sub_questions:
            if sq.question_id not in sub_answers or not sub_answers[sq.question_id].strip():
                unanswered.append(sq.question_text)

        return unanswered
