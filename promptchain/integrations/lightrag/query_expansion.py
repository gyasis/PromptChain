"""LightRAG Query Expansion Pattern.

Expands queries using multiple strategies (synonym, semantic, acronym, reformulation)
and searches in parallel to retrieve more comprehensive results.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from promptchain.patterns.base import BasePattern, PatternConfig, PatternResult

if TYPE_CHECKING:
    from hybridrag.src.search_interface import SearchInterface

    from promptchain.integrations.lightrag.core import LightRAGIntegration

logger = logging.getLogger(__name__)


class ExpansionStrategy(Enum):
    """Query expansion strategies."""

    SYNONYM = "synonym"  # Expand using synonyms and related terms
    SEMANTIC = "semantic"  # Expand using semantic similarity
    ACRONYM = "acronym"  # Expand acronyms and abbreviations
    REFORMULATION = "reformulation"  # Rephrase query in different ways


@dataclass
class QueryExpansionConfig(PatternConfig):
    """Configuration for query expansion pattern.

    Attributes:
        strategies: List of expansion strategies to use.
        max_expansions_per_strategy: Maximum variations per strategy.
        min_similarity: Minimum similarity score for expansion acceptance.
        deduplicate: Whether to deduplicate results.
        parallel_search: Whether to search all expansions in parallel.
    """

    strategies: List[ExpansionStrategy] = field(
        default_factory=lambda: [ExpansionStrategy.SEMANTIC]
    )
    max_expansions_per_strategy: int = 3
    min_similarity: float = 0.5
    deduplicate: bool = True
    parallel_search: bool = True


@dataclass
class ExpandedQuery:
    """Represents an expanded query variation.

    Attributes:
        query_id: Unique identifier for this expanded query.
        original_query: The original user query.
        expanded_query: The expanded/modified query.
        strategy: Strategy used to generate this expansion.
        similarity_score: Similarity score to original (0.0-1.0).
    """

    query_id: str
    original_query: str
    expanded_query: str
    strategy: ExpansionStrategy
    similarity_score: float


@dataclass
class QueryExpansionResult(PatternResult):
    """Result from query expansion pattern execution.

    Attributes:
        original_query: The original user query.
        expanded_queries: List of all expanded query variations.
        search_results: Aggregated search results from all queries.
        unique_results_found: Number of unique results after deduplication.
    """

    original_query: str = ""
    expanded_queries: List[ExpandedQuery] = field(default_factory=list)
    search_results: List[Any] = field(default_factory=list)
    unique_results_found: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        base = super().to_dict()
        base.update(
            {
                "original_query": self.original_query,
                "expanded_queries": [
                    {
                        "query_id": eq.query_id,
                        "original_query": eq.original_query,
                        "expanded_query": eq.expanded_query,
                        "strategy": eq.strategy.value,
                        "similarity_score": eq.similarity_score,
                    }
                    for eq in self.expanded_queries
                ],
                "unique_results_found": self.unique_results_found,
            }
        )
        return base


class LightRAGQueryExpander(BasePattern):
    """Query expansion pattern using LightRAG.

    Expands queries using multiple strategies and executes parallel searches
    to retrieve more comprehensive results.

    Example:
        >>> from promptchain.integrations.lightrag import LightRAGIntegration
        >>> integration = LightRAGIntegration()
        >>> expander = LightRAGQueryExpander(
        ...     lightrag_integration=integration,
        ...     config=QueryExpansionConfig(
        ...         strategies=[ExpansionStrategy.SEMANTIC, ExpansionStrategy.REFORMULATION],
        ...         max_expansions_per_strategy=3
        ...     )
        ... )
        >>> result = await expander.execute(
        ...     query="What is machine learning?",
        ...     strategies=[ExpansionStrategy.SEMANTIC]
        ... )
    """

    def __init__(
        self,
        search_interface: Optional["SearchInterface"] = None,
        lightrag_integration: Optional["LightRAGIntegration"] = None,
        config: Optional[QueryExpansionConfig] = None,
    ):
        """Initialize query expander.

        Args:
            search_interface: Direct SearchInterface instance (from hybridrag).
            lightrag_integration: LightRAGIntegration wrapper instance.
            config: Configuration for query expansion.

        Raises:
            ImportError: If hybridrag is not available.
            ValueError: If neither search_interface nor lightrag_integration provided.
        """
        super().__init__(config or QueryExpansionConfig())
        self.config: QueryExpansionConfig = self.config  # Type hint refinement

        # Accept either direct SearchInterface or LightRAGIntegration wrapper
        if search_interface is not None:
            self._search = search_interface
        elif lightrag_integration is not None:
            self._search = lightrag_integration.search
        else:
            raise ValueError(
                "Must provide either search_interface or lightrag_integration"
            )

        # Store integration for context extraction
        self._integration = lightrag_integration

    async def execute(  # type: ignore[override]
        self, query: str, strategies: Optional[List[ExpansionStrategy]] = None
    ) -> QueryExpansionResult:
        """Execute query expansion and parallel search.

        Args:
            query: Original user query to expand.
            strategies: Override expansion strategies from config.

        Returns:
            QueryExpansionResult with expanded queries and aggregated results.
        """
        start_time = time.perf_counter()
        expansion_strategies = strategies or self.config.strategies

        self.emit_event(
            "pattern.query_expansion.started",
            {
                "query": query,
                "strategies": [s.value for s in expansion_strategies],
            },
        )

        try:
            # Extract context from knowledge graph for semantic understanding
            context = await self._extract_query_context(query)

            # Generate query expansions using each strategy
            expanded_queries = await self._generate_expansions(
                query, expansion_strategies, context
            )

            self.emit_event(
                "pattern.query_expansion.expanded",
                {
                    "query": query,
                    "expansion_count": len(expanded_queries),
                    "strategies_used": [s.value for s in expansion_strategies],
                },
            )

            # Execute searches for all expanded queries
            search_results = await self._execute_searches(query, expanded_queries)

            self.emit_event(
                "pattern.query_expansion.searching",
                {
                    "query": query,
                    "total_queries": len(expanded_queries) + 1,  # +1 for original
                },
            )

            # Deduplicate and rank results
            unique_results = self._deduplicate_results(search_results)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            self.emit_event(
                "pattern.query_expansion.completed",
                {
                    "query": query,
                    "unique_results": len(unique_results),
                    "total_expansions": len(expanded_queries),
                    "execution_time_ms": elapsed_ms,
                },
            )

            return QueryExpansionResult(
                pattern_id=self.config.pattern_id,
                success=True,
                result=unique_results,
                execution_time_ms=elapsed_ms,
                original_query=query,
                expanded_queries=expanded_queries,
                search_results=unique_results,
                unique_results_found=len(unique_results),
                metadata={
                    "strategies_used": [s.value for s in expansion_strategies],
                    "total_queries": len(expanded_queries) + 1,
                    "parallel_execution": self.config.parallel_search,
                },
            )

        except ImportError as e:
            logger.error(f"hybridrag not available: {e}")
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return QueryExpansionResult(
                pattern_id=self.config.pattern_id,
                success=False,
                result=None,
                execution_time_ms=elapsed_ms,
                original_query=query,
                errors=[f"hybridrag not available: {e}"],
            )
        except Exception as e:
            logger.exception(f"Query expansion failed: {e}")
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return QueryExpansionResult(
                pattern_id=self.config.pattern_id,
                success=False,
                result=None,
                execution_time_ms=elapsed_ms,
                original_query=query,
                errors=[str(e)],
            )

    async def _extract_query_context(self, query: str) -> Dict[str, Any]:
        """Extract semantic context for query understanding.

        Args:
            query: The user query.

        Returns:
            Dictionary with extracted entities, relationships, concepts.
        """
        try:
            if self._integration is not None:
                context = await self._integration.extract_context(query)
                return context if context else {}
            return {}
        except Exception as e:
            logger.warning(f"Context extraction failed: {e}")
            return {}

    async def _generate_expansions(
        self, query: str, strategies: List[ExpansionStrategy], context: Dict[str, Any]
    ) -> List[ExpandedQuery]:
        """Generate query expansions using specified strategies.

        Args:
            query: Original query.
            strategies: Expansion strategies to use.
            context: Extracted semantic context.

        Returns:
            List of expanded query variations.
        """
        expanded_queries = []

        for strategy in strategies:
            try:
                expansions = await self._expand_with_strategy(query, strategy, context)
                expanded_queries.extend(expansions)
            except Exception as e:
                logger.warning(f"Expansion with {strategy.value} failed: {e}")

        # Filter by similarity threshold
        filtered = [
            eq
            for eq in expanded_queries
            if eq.similarity_score >= self.config.min_similarity
        ]

        return filtered

    async def _expand_with_strategy(
        self, query: str, strategy: ExpansionStrategy, context: Dict[str, Any]
    ) -> List[ExpandedQuery]:
        """Expand query using a specific strategy.

        Args:
            query: Original query.
            strategy: Expansion strategy.
            context: Semantic context.

        Returns:
            List of expanded queries for this strategy.
        """
        expansions = []

        if strategy == ExpansionStrategy.SYNONYM:
            expansions = await self._expand_synonyms(query, context)
        elif strategy == ExpansionStrategy.SEMANTIC:
            expansions = await self._expand_semantic(query, context)
        elif strategy == ExpansionStrategy.ACRONYM:
            expansions = await self._expand_acronyms(query, context)
        elif strategy == ExpansionStrategy.REFORMULATION:
            expansions = await self._expand_reformulations(query, context)

        # Limit to max_expansions_per_strategy
        return expansions[: self.config.max_expansions_per_strategy]

    async def _expand_synonyms(
        self, query: str, context: Dict[str, Any]
    ) -> List[ExpandedQuery]:
        """Expand using synonyms and related terms.

        Args:
            query: Original query.
            context: Semantic context with entities.

        Returns:
            List of synonym-based expansions.
        """
        # Simple synonym expansion based on context entities
        expansions = []
        entities = context.get("entities", [])

        for i, entity in enumerate(entities[: self.config.max_expansions_per_strategy]):
            expanded = query.replace(
                entity.get("name", ""), entity.get("type", entity.get("name", ""))
            )
            if expanded != query:
                expansions.append(
                    ExpandedQuery(
                        query_id=str(uuid.uuid4()),
                        original_query=query,
                        expanded_query=expanded,
                        strategy=ExpansionStrategy.SYNONYM,
                        similarity_score=0.8,  # Heuristic
                    )
                )

        return expansions

    async def _expand_semantic(
        self, query: str, context: Dict[str, Any]
    ) -> List[ExpandedQuery]:
        """Expand using semantic similarity.

        Args:
            query: Original query.
            context: Semantic context.

        Returns:
            List of semantically similar expansions.
        """
        # Semantic expansion using context relationships
        expansions = []
        relationships = context.get("relationships", [])

        for i, rel in enumerate(
            relationships[: self.config.max_expansions_per_strategy]
        ):
            # Create expansion based on relationship
            expanded = f"{query} {rel.get('type', '')} {rel.get('target', '')}"
            expansions.append(
                ExpandedQuery(
                    query_id=str(uuid.uuid4()),
                    original_query=query,
                    expanded_query=expanded.strip(),
                    strategy=ExpansionStrategy.SEMANTIC,
                    similarity_score=0.85,  # Heuristic
                )
            )

        return expansions

    async def _expand_acronyms(
        self, query: str, context: Dict[str, Any]
    ) -> List[ExpandedQuery]:
        """Expand acronyms and abbreviations.

        Args:
            query: Original query.
            context: Semantic context.

        Returns:
            List of acronym expansions.
        """
        # Simple acronym detection and expansion
        expansions = []
        words = query.split()

        for word in words:
            if word.isupper() and len(word) > 1:  # Likely acronym
                # Look for expansion in context
                for entity in context.get("entities", []):
                    if word in entity.get("name", "").upper():
                        expanded = query.replace(word, entity.get("name", word))
                        expansions.append(
                            ExpandedQuery(
                                query_id=str(uuid.uuid4()),
                                original_query=query,
                                expanded_query=expanded,
                                strategy=ExpansionStrategy.ACRONYM,
                                similarity_score=0.9,
                            )
                        )
                        break

        return expansions[: self.config.max_expansions_per_strategy]

    async def _expand_reformulations(
        self, query: str, context: Dict[str, Any]
    ) -> List[ExpandedQuery]:
        """Expand through query reformulation.

        Args:
            query: Original query.
            context: Semantic context.

        Returns:
            List of reformulated queries.
        """
        # Simple reformulations with question variations
        expansions = []
        reformulations = [
            f"What is {query}?",
            f"Explain {query}",
            f"How does {query} work?",
        ]

        for i, reformulated in enumerate(
            reformulations[: self.config.max_expansions_per_strategy]
        ):
            if reformulated.lower() != query.lower():
                expansions.append(
                    ExpandedQuery(
                        query_id=str(uuid.uuid4()),
                        original_query=query,
                        expanded_query=reformulated,
                        strategy=ExpansionStrategy.REFORMULATION,
                        similarity_score=0.75,  # Heuristic
                    )
                )

        return expansions

    async def _execute_searches(
        self, original_query: str, expanded_queries: List[ExpandedQuery]
    ) -> List[Any]:
        """Execute searches for all query variations.

        Args:
            original_query: Original user query.
            expanded_queries: List of expanded variations.

        Returns:
            Aggregated search results.
        """
        all_queries = [original_query] + [eq.expanded_query for eq in expanded_queries]

        if self.config.parallel_search:
            try:
                # Use multi_query_search for parallel execution
                results = await self._search.multi_query_search(
                    queries=all_queries, mode="hybrid"
                )
                return results if isinstance(results, list) else [results]
            except AttributeError:
                # Fallback if multi_query_search not available
                logger.warning(
                    "multi_query_search not available, falling back to sequential"
                )
                return await self._execute_sequential_searches(all_queries)
        else:
            return await self._execute_sequential_searches(all_queries)

    async def _execute_sequential_searches(self, queries: List[str]) -> List[Any]:
        """Execute searches sequentially.

        Args:
            queries: List of queries to search.

        Returns:
            List of search results.
        """
        results = []
        for query in queries:
            try:
                # Execute hybrid search for each query
                result = await self._search.hybrid_search(query)
                results.append(result)
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")

        return results

    def _deduplicate_results(self, search_results: List[Any]) -> List[Any]:
        """Deduplicate search results.

        Args:
            search_results: Raw search results.

        Returns:
            Deduplicated results.
        """
        if not self.config.deduplicate:
            return search_results

        # Simple deduplication by converting to set (requires hashable results)
        # In practice, this would use more sophisticated deduplication
        seen = set()
        unique = []

        for result in search_results:
            # Use string representation as dedup key
            result_key = str(result)
            if result_key not in seen:
                seen.add(result_key)
                unique.append(result)

        return unique
