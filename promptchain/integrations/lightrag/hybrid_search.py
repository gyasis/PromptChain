"""LightRAG Hybrid Search Fusion Pattern.

Combines multiple search techniques (local, global, hybrid, naive, mix) with
explicit fusion algorithms (RRF, linear, Borda) to produce optimal retrieval results.

Pattern 7/14: Hybrid Search Fusion
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from promptchain.patterns.base import BasePattern, PatternConfig, PatternResult

if TYPE_CHECKING:
    from promptchain.integrations.lightrag.core import LightRAGIntegration

logger = logging.getLogger(__name__)


class SearchTechnique(Enum):
    """Available search techniques in LightRAG."""
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    NAIVE = "naive"
    MIX = "mix"


class FusionAlgorithm(Enum):
    """Result fusion algorithms."""
    RRF = "rrf"  # Reciprocal Rank Fusion
    LINEAR = "linear"  # Weighted linear combination
    BORDA = "borda"  # Borda count voting


@dataclass
class TechniqueResult:
    """Result from a single search technique execution.

    Attributes:
        technique: The search technique used.
        results: List of retrieved results.
        scores: Corresponding scores for each result.
        query_time_ms: Time taken to execute the query in milliseconds.
    """
    technique: SearchTechnique
    results: List[Any]
    scores: List[float]
    query_time_ms: float


@dataclass
class HybridSearchConfig(PatternConfig):
    """Configuration for hybrid search fusion.

    Attributes:
        techniques: List of search techniques to execute.
        fusion_algorithm: Algorithm to use for result fusion.
        rrf_k: K parameter for RRF algorithm (default: 60).
        top_k: Number of top results to return after fusion.
        normalize_scores: Whether to normalize scores before fusion.
    """
    techniques: List[SearchTechnique] = field(
        default_factory=lambda: [SearchTechnique.LOCAL, SearchTechnique.GLOBAL]
    )
    fusion_algorithm: FusionAlgorithm = FusionAlgorithm.RRF
    rrf_k: int = 60
    top_k: int = 10
    normalize_scores: bool = True


@dataclass
class HybridSearchResult(PatternResult):
    """Result from hybrid search fusion.

    Attributes:
        query: The original search query.
        technique_results: Results from each individual technique.
        fused_results: Final fused and ranked results.
        fused_scores: Scores assigned by the fusion algorithm.
        technique_contributions: Count of results contributed by each technique.
    """
    query: str = ""
    technique_results: List[TechniqueResult] = field(default_factory=list)
    fused_results: List[Any] = field(default_factory=list)
    fused_scores: List[float] = field(default_factory=list)
    technique_contributions: Dict[str, int] = field(default_factory=dict)


class LightRAGHybridSearcher(BasePattern):
    """Hybrid search pattern combining multiple LightRAG techniques.

    Executes multiple search techniques in parallel and fuses results using
    sophisticated ranking algorithms (RRF, linear, Borda).

    Example:
        >>> from promptchain.integrations.lightrag.core import LightRAGIntegration
        >>> integration = LightRAGIntegration()
        >>> config = HybridSearchConfig(
        ...     techniques=[SearchTechnique.LOCAL, SearchTechnique.GLOBAL],
        ...     fusion_algorithm=FusionAlgorithm.RRF,
        ...     top_k=10
        ... )
        >>> searcher = LightRAGHybridSearcher(integration.core, config=config)
        >>> result = await searcher.execute(query="What is machine learning?")
    """

    def __init__(
        self,
        lightrag_integration: "LightRAGIntegration",
        config: Optional[HybridSearchConfig] = None
    ):
        """Initialize hybrid searcher.

        Args:
            lightrag_integration: LightRAGIntegration instance for queries.
            config: Configuration for hybrid search. Uses defaults if not provided.
        """
        super().__init__(config or HybridSearchConfig())
        self.config: HybridSearchConfig = self.config  # Type narrowing
        self.integration = lightrag_integration

        # Validate technique availability
        try:
            # This will raise ImportError if hybridrag not available
            _ = self.integration.core
        except ImportError as e:
            logger.error(f"LightRAG not available: {e}")
            raise

    async def execute(self, query: str, **kwargs) -> HybridSearchResult:
        """Execute hybrid search with fusion.

        Args:
            query: The search query.
            **kwargs: Additional arguments (e.g., override top_k).

        Returns:
            HybridSearchResult with fused results and metadata.
        """
        start_time = time.perf_counter()

        self.emit_event("pattern.hybrid_search.started", {
            "query": query,
            "techniques": [t.value for t in self.config.techniques],
            "fusion_algorithm": self.config.fusion_algorithm.value,
        })

        # Execute all techniques in parallel
        technique_results = await self._execute_techniques(query, **kwargs)

        self.emit_event("pattern.hybrid_search.fusing", {
            "num_techniques": len(technique_results),
            "total_results": sum(len(tr.results) for tr in technique_results),
        })

        # Fuse results using configured algorithm (respect top_k override)
        top_k = kwargs.get("top_k", self.config.top_k)
        fused_results, fused_scores, contributions = await self._fuse_results(
            technique_results,
            top_k=top_k
        )

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        self.emit_event("pattern.hybrid_search.completed", {
            "query": query,
            "num_fused_results": len(fused_results),
            "execution_time_ms": execution_time_ms,
        })

        return HybridSearchResult(
            pattern_id=self.config.pattern_id,
            success=True,
            result=fused_results,
            execution_time_ms=execution_time_ms,
            query=query,
            technique_results=technique_results,
            fused_results=fused_results,
            fused_scores=fused_scores,
            technique_contributions=contributions,
        )

    async def _execute_techniques(
        self,
        query: str,
        **kwargs
    ) -> List[TechniqueResult]:
        """Execute all configured search techniques in parallel.

        Args:
            query: Search query.
            **kwargs: Additional arguments passed to techniques.

        Returns:
            List of TechniqueResult objects.
        """
        tasks = []
        for technique in self.config.techniques:
            tasks.append(self._execute_single_technique(technique, query, **kwargs))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed techniques
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Technique {self.config.techniques[i].value} failed: {result}"
                )
                self.emit_event("pattern.hybrid_search.technique_failed", {
                    "technique": self.config.techniques[i].value,
                    "error": str(result),
                })
            else:
                successful_results.append(result)

        return successful_results

    async def _execute_single_technique(
        self,
        technique: SearchTechnique,
        query: str,
        **kwargs
    ) -> TechniqueResult:
        """Execute a single search technique.

        Args:
            technique: The search technique to use.
            query: Search query.
            **kwargs: Additional arguments.

        Returns:
            TechniqueResult with results and scores.
        """
        start_time = time.perf_counter()
        top_k = kwargs.get("top_k", self.config.top_k)

        # Execute the appropriate query method
        if technique == SearchTechnique.LOCAL:
            raw_result = await self.integration.local_query(query, top_k=top_k)
        elif technique == SearchTechnique.GLOBAL:
            raw_result = await self.integration.global_query(query, top_k=top_k)
        elif technique == SearchTechnique.HYBRID:
            raw_result = await self.integration.hybrid_query(query, top_k=top_k)
        elif technique == SearchTechnique.NAIVE:
            # Fallback to basic retrieval if naive is explicitly supported
            raw_result = await self.integration.local_query(query, top_k=top_k)
        elif technique == SearchTechnique.MIX:
            # Mix technique combines local and global equally
            local_res = await self.integration.local_query(query, top_k=top_k // 2)
            global_res = await self.integration.global_query(query, top_k=top_k // 2)
            raw_result = {"results": [local_res, global_res]}
        else:
            raise ValueError(f"Unknown technique: {technique}")

        query_time_ms = (time.perf_counter() - start_time) * 1000

        # Extract results and scores from raw result
        # The structure depends on what HybridLightRAGCore returns
        # Assuming it returns a dict with 'results' and optionally 'scores'
        if isinstance(raw_result, dict):
            results = raw_result.get("results", [])
            scores = raw_result.get("scores", [1.0] * len(results))
        elif isinstance(raw_result, list):
            results = raw_result
            scores = [1.0] * len(results)
        else:
            # Single result
            results = [raw_result]
            scores = [1.0]

        self.emit_event("pattern.hybrid_search.technique_completed", {
            "technique": technique.value,
            "num_results": len(results),
            "query_time_ms": query_time_ms,
        })

        return TechniqueResult(
            technique=technique,
            results=results,
            scores=scores,
            query_time_ms=query_time_ms,
        )

    async def _fuse_results(
        self,
        technique_results: List[TechniqueResult],
        top_k: Optional[int] = None
    ) -> tuple[List[Any], List[float], Dict[str, int]]:
        """Fuse results from multiple techniques.

        Args:
            technique_results: Results from each technique.
            top_k: Number of top results to return (overrides config).

        Returns:
            Tuple of (fused_results, fused_scores, technique_contributions).
        """
        final_top_k = top_k if top_k is not None else self.config.top_k

        if self.config.fusion_algorithm == FusionAlgorithm.RRF:
            return self._reciprocal_rank_fusion(technique_results, k=self.config.rrf_k, top_k=final_top_k)
        elif self.config.fusion_algorithm == FusionAlgorithm.LINEAR:
            return self._linear_fusion(technique_results, top_k=final_top_k)
        elif self.config.fusion_algorithm == FusionAlgorithm.BORDA:
            return self._borda_fusion(technique_results, top_k=final_top_k)
        else:
            raise ValueError(f"Unknown fusion algorithm: {self.config.fusion_algorithm}")

    def _reciprocal_rank_fusion(
        self,
        technique_results: List[TechniqueResult],
        k: int = 60,
        top_k: Optional[int] = None
    ) -> tuple[List[Any], List[float], Dict[str, int]]:
        """Fuse results using Reciprocal Rank Fusion (RRF).

        RRF formula: score(doc) = sum_over_techniques(1 / (k + rank_in_technique))

        Args:
            technique_results: Results from each technique.
            k: Constant for RRF (default: 60, standard value from literature).
            top_k: Number of top results to return.

        Returns:
            Tuple of (fused_results, fused_scores, technique_contributions).
        """
        final_top_k = top_k if top_k is not None else self.config.top_k

        # Build unified ranking using RRF
        result_scores: Dict[str, float] = {}
        result_objects: Dict[str, Any] = {}
        result_sources: Dict[str, List[str]] = {}

        for tech_result in technique_results:
            technique_name = tech_result.technique.value

            for rank, result in enumerate(tech_result.results):
                # Use result content as key (assuming results are strings or have str representation)
                result_key = str(result)

                # RRF score contribution
                rrf_score = 1.0 / (k + rank + 1)

                if result_key not in result_scores:
                    result_scores[result_key] = 0.0
                    result_objects[result_key] = result
                    result_sources[result_key] = []

                result_scores[result_key] += rrf_score
                result_sources[result_key].append(technique_name)

        # Sort by RRF score
        sorted_results = sorted(
            result_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:final_top_k]

        # Extract final results
        fused_results = [result_objects[key] for key, _ in sorted_results]
        fused_scores = [score for _, score in sorted_results]

        # Count contributions
        contributions = {}
        for result_key, _ in sorted_results:
            for source in result_sources[result_key]:
                contributions[source] = contributions.get(source, 0) + 1

        return fused_results, fused_scores, contributions

    def _linear_fusion(
        self,
        technique_results: List[TechniqueResult],
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None
    ) -> tuple[List[Any], List[float], Dict[str, int]]:
        """Fuse results using weighted linear combination.

        Args:
            technique_results: Results from each technique.
            weights: Weight for each technique. Equal weights if not provided.
            top_k: Number of top results to return.

        Returns:
            Tuple of (fused_results, fused_scores, technique_contributions).
        """
        final_top_k = top_k if top_k is not None else self.config.top_k

        if weights is None:
            weights = [1.0 / len(technique_results)] * len(technique_results)

        result_scores: Dict[str, float] = {}
        result_objects: Dict[str, Any] = {}
        result_sources: Dict[str, List[str]] = {}

        for tech_result, weight in zip(technique_results, weights):
            technique_name = tech_result.technique.value

            # Normalize scores if configured
            scores = tech_result.scores
            if self.config.normalize_scores and scores:
                max_score = max(scores) if scores else 1.0
                scores = [s / max_score if max_score > 0 else 0.0 for s in scores]

            for result, score in zip(tech_result.results, scores):
                result_key = str(result)

                if result_key not in result_scores:
                    result_scores[result_key] = 0.0
                    result_objects[result_key] = result
                    result_sources[result_key] = []

                result_scores[result_key] += score * weight
                result_sources[result_key].append(technique_name)

        # Sort by weighted score
        sorted_results = sorted(
            result_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:final_top_k]

        fused_results = [result_objects[key] for key, _ in sorted_results]
        fused_scores = [score for _, score in sorted_results]

        contributions = {}
        for result_key, _ in sorted_results:
            for source in result_sources[result_key]:
                contributions[source] = contributions.get(source, 0) + 1

        return fused_results, fused_scores, contributions

    def _borda_fusion(
        self,
        technique_results: List[TechniqueResult],
        top_k: Optional[int] = None
    ) -> tuple[List[Any], List[float], Dict[str, int]]:
        """Fuse results using Borda count voting.

        Each result gets points based on its rank: (total_results - rank).

        Args:
            technique_results: Results from each technique.
            top_k: Number of top results to return.

        Returns:
            Tuple of (fused_results, fused_scores, technique_contributions).
        """
        final_top_k = top_k if top_k is not None else self.config.top_k

        result_scores: Dict[str, float] = {}
        result_objects: Dict[str, Any] = {}
        result_sources: Dict[str, List[str]] = {}

        for tech_result in technique_results:
            technique_name = tech_result.technique.value
            num_results = len(tech_result.results)

            for rank, result in enumerate(tech_result.results):
                result_key = str(result)

                # Borda count: points = (num_results - rank)
                borda_points = num_results - rank

                if result_key not in result_scores:
                    result_scores[result_key] = 0.0
                    result_objects[result_key] = result
                    result_sources[result_key] = []

                result_scores[result_key] += borda_points
                result_sources[result_key].append(technique_name)

        # Sort by Borda points
        sorted_results = sorted(
            result_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:final_top_k]

        fused_results = [result_objects[key] for key, _ in sorted_results]
        fused_scores = [score for _, score in sorted_results]

        contributions = {}
        for result_key, _ in sorted_results:
            for source in result_sources[result_key]:
                contributions[source] = contributions.get(source, 0) + 1

        return fused_results, fused_scores, contributions
