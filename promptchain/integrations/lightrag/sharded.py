"""LightRAG Sharded Retrieval Pattern.

Provides parallel retrieval across multiple LightRAG database shards with
aggregation and fault tolerance.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from promptchain.patterns.base import BasePattern, PatternConfig, PatternResult
from promptchain.integrations.lightrag import LIGHTRAG_AVAILABLE

if TYPE_CHECKING:
    from hybridrag.src.lightrag_core import HybridLightRAGCore

logger = logging.getLogger(__name__)


class ShardType(Enum):
    """Type of database shard."""
    LIGHTRAG = "lightrag"
    VECTOR_DB = "vector_db"
    DOCUMENT_DB = "document_db"
    API = "api"


@dataclass
class ShardConfig:
    """Configuration for a single shard.

    Attributes:
        shard_id: Unique identifier for this shard.
        shard_type: Type of shard (LIGHTRAG, VECTOR_DB, etc.).
        working_dir: Working directory for LIGHTRAG type shards.
        connection_config: Configuration dict for non-LIGHTRAG shards.
        priority: Priority for result aggregation (higher = more important).
        timeout_seconds: Maximum query time before timeout.
        enabled: Whether this shard is active.
    """
    shard_id: str
    shard_type: ShardType
    working_dir: str = ""
    connection_config: Optional[Dict[str, Any]] = None
    priority: int = 0
    timeout_seconds: float = 10.0
    enabled: bool = True


@dataclass
class ShardResult:
    """Result from querying a single shard.

    Attributes:
        shard_id: ID of the shard that produced this result.
        results: List of results from the shard.
        query_time_ms: Time taken to query this shard in milliseconds.
        error: Error message if query failed, None otherwise.
    """
    shard_id: str
    results: List[Any]
    query_time_ms: float
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Whether the shard query succeeded."""
        return self.error is None


@dataclass
class ShardedRetrievalConfig(PatternConfig):
    """Configuration for sharded retrieval pattern.

    Extends PatternConfig with shard-specific settings.

    Attributes:
        parallel: Whether to query shards in parallel.
        fail_partial: If False, fail entire query if any shard fails.
        aggregate_top_k: Number of results to return after aggregation.
        normalize_scores: Whether to normalize scores across shards.
    """
    parallel: bool = True
    fail_partial: bool = True
    aggregate_top_k: int = 10
    normalize_scores: bool = True


@dataclass
class ShardedRetrievalResult(PatternResult):
    """Result from sharded retrieval pattern execution.

    Extends PatternResult with shard-specific information.

    Attributes:
        query: The original query.
        shard_results: Individual results from each shard.
        aggregated_results: Combined and ranked results.
        shards_queried: Number of shards that were queried.
        shards_failed: Number of shards that failed.
        warnings: List of warning messages.
    """
    query: str = ""
    shard_results: List[ShardResult] = field(default_factory=list)
    aggregated_results: List[Any] = field(default_factory=list)
    shards_queried: int = 0
    shards_failed: int = 0
    warnings: List[str] = field(default_factory=list)


class LightRAGShardRegistry:
    """Registry for managing multiple LightRAG shards.

    Handles shard registration, lifecycle, and health checking.

    Example:
        >>> registry = LightRAGShardRegistry()
        >>> registry.register_shard(ShardConfig(
        ...     shard_id="shard1",
        ...     shard_type=ShardType.LIGHTRAG,
        ...     working_dir="./shard1_data"
        ... ))
        >>> shard = registry.get_shard("shard1")
        >>> health = registry.health_check()
    """

    def __init__(self):
        """Initialize the shard registry."""
        self._shards: Dict[str, Any] = {}
        self._configs: Dict[str, ShardConfig] = {}

    def register_shard(self, config: ShardConfig) -> None:
        """Register a new shard.

        For LIGHTRAG type shards, creates a HybridLightRAGCore instance.
        For other types, stores the connection config for future use.

        Args:
            config: Shard configuration.

        Raises:
            ImportError: If LIGHTRAG type requested but hybridrag not available.
            ValueError: If shard_id already registered.
        """
        if config.shard_id in self._shards:
            raise ValueError(f"Shard {config.shard_id} already registered")

        if config.shard_type == ShardType.LIGHTRAG:
            if not LIGHTRAG_AVAILABLE:
                raise ImportError(
                    "hybridrag is not installed. Install with: "
                    "pip install git+https://github.com/gyasis/hybridrag.git"
                )

            from hybridrag.src.lightrag_core import HybridLightRAGCore

            # Create HybridLightRAGCore instance
            shard = HybridLightRAGCore(
                working_dir=config.working_dir,
                **(config.connection_config or {})
            )
            self._shards[config.shard_id] = shard
            logger.info(f"Registered LIGHTRAG shard: {config.shard_id}")

        else:
            # For non-LIGHTRAG shards, store config for custom implementation
            self._shards[config.shard_id] = config.connection_config or {}
            logger.info(f"Registered {config.shard_type.value} shard: {config.shard_id}")

        self._configs[config.shard_id] = config

    def get_shard(self, shard_id: str) -> Any:
        """Get a registered shard.

        Args:
            shard_id: ID of the shard to retrieve.

        Returns:
            The shard instance or connection config.

        Raises:
            KeyError: If shard_id not found.
        """
        if shard_id not in self._shards:
            raise KeyError(f"Shard {shard_id} not found")
        return self._shards[shard_id]

    def get_config(self, shard_id: str) -> ShardConfig:
        """Get configuration for a shard.

        Args:
            shard_id: ID of the shard.

        Returns:
            ShardConfig instance.

        Raises:
            KeyError: If shard_id not found.
        """
        if shard_id not in self._configs:
            raise KeyError(f"Shard {shard_id} not found")
        return self._configs[shard_id]

    def list_shards(self) -> List[str]:
        """List all registered shard IDs.

        Returns:
            List of shard IDs.
        """
        return list(self._shards.keys())

    def health_check(self) -> Dict[str, bool]:
        """Check availability of all shards.

        Returns:
            Dictionary mapping shard_id to availability status.
        """
        health = {}
        for shard_id, config in self._configs.items():
            if not config.enabled:
                health[shard_id] = False
                continue

            # For LIGHTRAG shards, check if instance exists
            if config.shard_type == ShardType.LIGHTRAG:
                health[shard_id] = shard_id in self._shards
            else:
                # For other types, mark as available if registered
                health[shard_id] = shard_id in self._shards

        return health


class LightRAGShardedRetriever(BasePattern):
    """Pattern for parallel retrieval across multiple LightRAG shards.

    Queries multiple shards in parallel, handles failures gracefully,
    and aggregates results with score normalization.

    Events emitted:
        - pattern.sharded.started: Query execution started
        - pattern.sharded.shard_queried: Individual shard query completed
        - pattern.sharded.shard_failed: Individual shard query failed
        - pattern.sharded.completed: All shards queried and results aggregated

    Example:
        >>> registry = LightRAGShardRegistry()
        >>> registry.register_shard(ShardConfig(
        ...     shard_id="shard1",
        ...     shard_type=ShardType.LIGHTRAG,
        ...     working_dir="./shard1"
        ... ))
        >>> retriever = LightRAGShardedRetriever(
        ...     registry=registry,
        ...     config=ShardedRetrievalConfig(parallel=True)
        ... )
        >>> result = await retriever.execute(query="What is AI?")
    """

    def __init__(
        self,
        registry: LightRAGShardRegistry,
        config: Optional[ShardedRetrievalConfig] = None
    ):
        """Initialize the sharded retriever.

        Args:
            registry: Registry containing registered shards.
            config: Configuration for sharded retrieval.
        """
        super().__init__(config or ShardedRetrievalConfig())
        self.registry = registry
        self.config: ShardedRetrievalConfig  # Type hint for IDE support

    async def execute(
        self,
        query: str,
        shard_ids: Optional[List[str]] = None,
        **kwargs
    ) -> ShardedRetrievalResult:
        """Execute sharded retrieval.

        Queries specified shards (or all shards if not specified) and
        aggregates results.

        Args:
            query: The search query.
            shard_ids: Optional list of specific shard IDs to query.
                      If None, queries all enabled shards.
            **kwargs: Additional arguments passed to shard queries.

        Returns:
            ShardedRetrievalResult with aggregated results.
        """
        start_time = time.perf_counter()

        # Determine which shards to query
        if shard_ids is None:
            shard_ids = [
                sid for sid in self.registry.list_shards()
                if self.registry.get_config(sid).enabled
            ]

        if not shard_ids:
            return ShardedRetrievalResult(
                pattern_id=self.config.pattern_id,
                success=False,
                result=None,
                execution_time_ms=0.0,
                query=query,
                errors=["No enabled shards available"],
            )

        self.emit_event("pattern.sharded.started", {
            "query": query,
            "shard_count": len(shard_ids),
            "parallel": self.config.parallel,
        })

        # Query shards
        if self.config.parallel:
            shard_results = await self._query_parallel(query, shard_ids, **kwargs)
        else:
            shard_results = await self._query_sequential(query, shard_ids, **kwargs)

        # Aggregate results
        aggregated = self._aggregate_results(shard_results)

        # Calculate statistics
        shards_failed = sum(1 for sr in shard_results if not sr.success)
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Check if we should fail
        success = True
        errors = []
        warnings = []

        if not self.config.fail_partial and shards_failed > 0:
            success = False
            errors.append(f"{shards_failed} shard(s) failed")
        elif shards_failed > 0:
            warnings.append(f"{shards_failed} shard(s) failed but continuing")

        self.emit_event("pattern.sharded.completed", {
            "query": query,
            "shards_queried": len(shard_ids),
            "shards_failed": shards_failed,
            "results_count": len(aggregated),
            "execution_time_ms": execution_time_ms,
        })

        return ShardedRetrievalResult(
            pattern_id=self.config.pattern_id,
            success=success,
            result=aggregated,
            execution_time_ms=execution_time_ms,
            query=query,
            shard_results=shard_results,
            aggregated_results=aggregated,
            shards_queried=len(shard_ids),
            shards_failed=shards_failed,
            warnings=warnings,
            errors=errors,
        )

    async def _query_parallel(
        self,
        query: str,
        shard_ids: List[str],
        **kwargs
    ) -> List[ShardResult]:
        """Query all shards in parallel.

        Args:
            query: The search query.
            shard_ids: List of shard IDs to query.
            **kwargs: Additional query arguments.

        Returns:
            List of ShardResult instances.
        """
        tasks = [
            self._query_single_shard(query, shard_id, **kwargs)
            for shard_id in shard_ids
        ]
        return await asyncio.gather(*tasks)

    async def _query_sequential(
        self,
        query: str,
        shard_ids: List[str],
        **kwargs
    ) -> List[ShardResult]:
        """Query shards sequentially.

        Args:
            query: The search query.
            shard_ids: List of shard IDs to query.
            **kwargs: Additional query arguments.

        Returns:
            List of ShardResult instances.
        """
        results = []
        for shard_id in shard_ids:
            result = await self._query_single_shard(query, shard_id, **kwargs)
            results.append(result)
        return results

    async def _query_single_shard(
        self,
        query: str,
        shard_id: str,
        **kwargs
    ) -> ShardResult:
        """Query a single shard with timeout handling.

        Args:
            query: The search query.
            shard_id: ID of the shard to query.
            **kwargs: Additional query arguments.

        Returns:
            ShardResult with query outcome.
        """
        start_time = time.perf_counter()

        try:
            config = self.registry.get_config(shard_id)
            shard = self.registry.get_shard(shard_id)

            # Query with timeout
            if config.shard_type == ShardType.LIGHTRAG:
                results = await asyncio.wait_for(
                    shard.hybrid_query(query, **kwargs),
                    timeout=config.timeout_seconds
                )
            else:
                # For non-LIGHTRAG shards, return empty results
                # Subclasses can override _query_single_shard for custom logic
                results = []

            query_time_ms = (time.perf_counter() - start_time) * 1000

            self.emit_event("pattern.sharded.shard_queried", {
                "shard_id": shard_id,
                "query_time_ms": query_time_ms,
                "results_count": len(results) if isinstance(results, list) else 1,
            })

            # Normalize results to list format
            if not isinstance(results, list):
                results = [results]

            return ShardResult(
                shard_id=shard_id,
                results=results,
                query_time_ms=query_time_ms,
            )

        except asyncio.TimeoutError:
            query_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"Query timed out after {config.timeout_seconds}s"
            logger.warning(f"Shard {shard_id}: {error_msg}")

            self.emit_event("pattern.sharded.shard_failed", {
                "shard_id": shard_id,
                "error": error_msg,
                "query_time_ms": query_time_ms,
            })

            return ShardResult(
                shard_id=shard_id,
                results=[],
                query_time_ms=query_time_ms,
                error=error_msg,
            )

        except Exception as e:
            query_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"Query failed: {str(e)}"
            logger.exception(f"Shard {shard_id}: {error_msg}")

            self.emit_event("pattern.sharded.shard_failed", {
                "shard_id": shard_id,
                "error": error_msg,
                "query_time_ms": query_time_ms,
            })

            return ShardResult(
                shard_id=shard_id,
                results=[],
                query_time_ms=query_time_ms,
                error=error_msg,
            )

    def _aggregate_results(self, shard_results: List[ShardResult]) -> List[Any]:
        """Aggregate results from multiple shards.

        Combines results, normalizes scores if enabled, and returns
        top-k results by score.

        Args:
            shard_results: Results from individual shards.

        Returns:
            Aggregated and ranked results.
        """
        all_results = []

        # Collect all successful results
        for shard_result in shard_results:
            if not shard_result.success:
                continue

            config = self.registry.get_config(shard_result.shard_id)

            for result in shard_result.results:
                # Extract score if available
                score = self._extract_score(result)

                # Apply priority weighting
                weighted_score = score * (1.0 + config.priority * 0.1)

                all_results.append({
                    "result": result,
                    "shard_id": shard_result.shard_id,
                    "score": weighted_score,
                    "original_score": score,
                })

        if not all_results:
            return []

        # Normalize scores if enabled
        if self.config.normalize_scores:
            all_results = self._normalize_scores(all_results)

        # Sort by score and return top-k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return [r["result"] for r in all_results[:self.config.aggregate_top_k]]

    def _extract_score(self, result: Any) -> float:
        """Extract score from a result object.

        Args:
            result: Result object from a shard.

        Returns:
            Extracted score or 1.0 if no score available.
        """
        # Try common score attributes
        if hasattr(result, "score"):
            return float(result.score)
        if isinstance(result, dict):
            if "score" in result:
                return float(result["score"])
            if "relevance" in result:
                return float(result["relevance"])
        return 1.0

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize scores across all results.

        Uses min-max normalization to scale scores to [0, 1].

        Args:
            results: List of result dictionaries with "score" key.

        Returns:
            Results with normalized scores.
        """
        if not results:
            return results

        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        if max_score == min_score:
            for result in results:
                result["score"] = 1.0
            return results

        # Min-max normalization
        for result in results:
            result["score"] = (result["score"] - min_score) / (max_score - min_score)

        return results
