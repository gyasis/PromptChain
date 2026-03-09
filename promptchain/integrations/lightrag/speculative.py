"""Speculative Execution Pattern for LightRAG.

Predicts and pre-executes likely LightRAG queries based on conversation patterns,
caching results to reduce latency.
"""

import asyncio
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from promptchain.patterns.base import BasePattern, PatternConfig, PatternResult

if TYPE_CHECKING:
    from promptchain.integrations.lightrag.core import LightRAGIntegration

try:
    from promptchain.integrations.lightrag import LIGHTRAG_AVAILABLE
except ImportError:
    LIGHTRAG_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ToolPrediction:
    """Represents a predicted LightRAG query.

    Attributes:
        prediction_id: Unique identifier for this prediction.
        query: The predicted query text.
        mode: Query mode ("local", "global", "hybrid").
        confidence: Prediction confidence score (0.0-1.0).
        pattern_matched: Name of the pattern that triggered this prediction.
        context_hash: Hash of the context used for prediction.
    """

    prediction_id: str
    query: str
    mode: str
    confidence: float
    pattern_matched: str
    context_hash: str


@dataclass
class SpeculativeResult:
    """Represents a cached speculative execution result.

    Attributes:
        prediction_id: ID of the prediction that generated this result.
        query: The query that was executed.
        result: The query result.
        cached_at: Timestamp when result was cached.
        ttl_seconds: Time-to-live for this cached result.
        hit: Whether this result was used (cache hit).
    """

    prediction_id: str
    query: str
    result: Any
    cached_at: datetime
    ttl_seconds: float
    hit: bool = False

    def is_expired(self) -> bool:
        """Check if this cached result has expired."""
        age_seconds = (datetime.utcnow() - self.cached_at).total_seconds()
        return age_seconds > self.ttl_seconds


@dataclass
class SpeculativeConfig(PatternConfig):
    """Configuration for speculative execution.

    Attributes:
        min_confidence: Minimum confidence threshold for executing predictions.
        max_concurrent: Maximum number of concurrent speculative executions.
        default_ttl: Default time-to-live for cached results in seconds.
        prediction_model: Model to use for predictions ("frequency", "pattern").
        history_window: Number of recent queries to consider for predictions.
    """

    min_confidence: float = 0.7
    max_concurrent: int = 3
    default_ttl: float = 60.0
    prediction_model: str = "frequency"
    history_window: int = 20


@dataclass
class SpeculativeExecutionResult(PatternResult):
    """Result from speculative execution pattern.

    Attributes:
        predictions: List of all predictions generated.
        executed: List of predictions that were actually executed.
        cached_results: List of all cached results.
        actual_call: The actual query that was requested (if any).
        hit: Whether the actual query was found in cache.
        latency_saved_ms: Estimated latency saved by cache hit.
    """

    predictions: List[ToolPrediction] = field(default_factory=list)
    executed: List[ToolPrediction] = field(default_factory=list)
    cached_results: List[SpeculativeResult] = field(default_factory=list)
    actual_call: Optional[str] = None
    hit: bool = False
    latency_saved_ms: float = 0.0


class LightRAGSpeculativeExecutor(BasePattern):
    """Speculative execution pattern for LightRAG queries.

    This pattern analyzes conversation history to predict likely next queries
    and pre-executes them speculatively. When an actual query arrives, the
    cached result can be returned immediately, reducing latency.

    Example:
        >>> executor = LightRAGSpeculativeExecutor(
        ...     lightrag_core=lightrag_integration,
        ...     config=SpeculativeConfig(min_confidence=0.8)
        ... )
        >>> result = await executor.execute(context="What is machine learning?")
    """

    def __init__(
        self,
        lightrag_core: "LightRAGIntegration",
        config: Optional[SpeculativeConfig] = None,
    ):
        """Initialize the speculative executor.

        Args:
            lightrag_core: LightRAGIntegration instance for executing queries.
            config: Configuration for speculative execution.

        Raises:
            ImportError: If hybridrag is not available.
        """
        if not LIGHTRAG_AVAILABLE:
            raise ImportError(
                "hybridrag is not installed. Install with: "
                "pip install git+https://github.com/gyasis/hybridrag.git"
            )

        super().__init__(config or SpeculativeConfig())
        self.config: SpeculativeConfig  # Type narrowing
        self.lightrag_core = lightrag_core

        # Query history tracking
        self.history: deque = deque(maxlen=self.config.history_window)

        # Result cache
        self.cache: Dict[str, SpeculativeResult] = {}

        # Execution tracking
        self._total_predictions = 0
        self._total_executions = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def record_call(self, query: str, mode: str) -> None:
        """Record a query in the history.

        Args:
            query: The query text.
            mode: The query mode ("local", "global", "hybrid").
        """
        self.history.append(
            {"query": query, "mode": mode, "timestamp": datetime.utcnow()}
        )
        logger.debug(f"Recorded query: {query[:50]}... (mode={mode})")

    def _compute_context_hash(self, context: str) -> str:
        """Compute hash of context for cache key generation.

        Args:
            context: Context string to hash.

        Returns:
            Hexadecimal hash string.
        """
        return hashlib.sha256(context.encode()).hexdigest()[:16]

    def _analyze_frequency_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze query frequency patterns in history.

        Returns:
            Dictionary mapping query patterns to their statistics.
        """
        patterns: Dict[str, Dict[str, Any]] = {}

        for entry in self.history:
            query = entry["query"].lower()
            mode = entry["mode"]
            key = f"{query[:30]}_{mode}"  # First 30 chars + mode

            if key not in patterns:
                patterns[key] = {
                    "query": entry["query"],
                    "mode": mode,
                    "count": 0,
                    "last_seen": entry["timestamp"],
                }

            patterns[key]["count"] += 1
            patterns[key]["last_seen"] = entry["timestamp"]

        return patterns

    def predict_next_queries(self, context: str) -> List[ToolPrediction]:
        """Predict likely next queries based on context and history.

        Args:
            context: Current conversation context.

        Returns:
            List of predicted queries with confidence scores.
        """
        predictions: List[ToolPrediction] = []
        context_hash = self._compute_context_hash(context)

        if self.config.prediction_model == "frequency":
            # Frequency-based prediction
            patterns = self._analyze_frequency_patterns()
            total_queries = len(self.history)

            if total_queries == 0:
                return predictions

            for key, stats in patterns.items():
                # Confidence based on frequency and recency
                frequency_score = stats["count"] / total_queries
                recency_score = 1.0 - (
                    (datetime.utcnow() - stats["last_seen"]).total_seconds() / 3600
                )
                recency_score = max(0.0, min(1.0, recency_score))

                confidence = (frequency_score * 0.7) + (recency_score * 0.3)

                if confidence >= self.config.min_confidence:
                    prediction = ToolPrediction(
                        prediction_id=f"pred_{len(predictions)}_{context_hash}",
                        query=stats["query"],
                        mode=stats["mode"],
                        confidence=confidence,
                        pattern_matched="frequency",
                        context_hash=context_hash,
                    )
                    predictions.append(prediction)

        elif self.config.prediction_model == "pattern":
            # Pattern-based prediction (detect follow-up question patterns)
            if len(self.history) >= 2:
                recent = list(self.history)[-2:]

                # Simple pattern: if last two queries were similar mode
                if recent[0]["mode"] == recent[1]["mode"]:
                    # Predict continuation with same mode
                    last_query = recent[-1]["query"]
                    mode = recent[-1]["mode"]

                    prediction = ToolPrediction(
                        prediction_id=f"pred_pattern_{context_hash}",
                        query=last_query,
                        mode=mode,
                        confidence=0.8,
                        pattern_matched="mode_continuation",
                        context_hash=context_hash,
                    )
                    predictions.append(prediction)

        # Sort by confidence
        predictions.sort(key=lambda p: p.confidence, reverse=True)

        # Limit to max_concurrent
        predictions = predictions[: self.config.max_concurrent]

        self._total_predictions += len(predictions)
        logger.info(f"Generated {len(predictions)} predictions")

        return predictions

    def check_cache(self, query: str, mode: str) -> Optional[SpeculativeResult]:
        """Check if a query result is cached.

        Args:
            query: The query text.
            mode: The query mode.

        Returns:
            Cached result if found and not expired, None otherwise.
        """
        cache_key = f"{query}_{mode}"

        if cache_key in self.cache:
            result = self.cache[cache_key]

            if not result.is_expired():
                result.hit = True
                self._cache_hits += 1
                logger.info(f"Cache hit for query: {query[:50]}...")
                return result
            else:
                # Remove expired entry
                del self.cache[cache_key]
                logger.debug(f"Removed expired cache entry: {cache_key}")

        self._cache_misses += 1
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None

    async def _execute_prediction(
        self, prediction: ToolPrediction
    ) -> SpeculativeResult:
        """Execute a prediction and cache the result.

        Args:
            prediction: The prediction to execute.

        Returns:
            Cached result.
        """
        try:
            # Execute query based on mode
            if prediction.mode == "local":
                result = await self.lightrag_core.local_query(prediction.query)
            elif prediction.mode == "global":
                result = await self.lightrag_core.global_query(prediction.query)
            elif prediction.mode == "hybrid":
                result = await self.lightrag_core.hybrid_query(prediction.query)
            else:
                logger.warning(f"Unknown mode: {prediction.mode}, defaulting to hybrid")
                result = await self.lightrag_core.hybrid_query(prediction.query)

            # Cache result
            cached = SpeculativeResult(
                prediction_id=prediction.prediction_id,
                query=prediction.query,
                result=result,
                cached_at=datetime.utcnow(),
                ttl_seconds=self.config.default_ttl,
                hit=False,
            )

            cache_key = f"{prediction.query}_{prediction.mode}"
            self.cache[cache_key] = cached

            logger.info(
                f"Executed and cached prediction: {prediction.query[:50]}... "
                f"(confidence={prediction.confidence:.2f})"
            )

            return cached

        except Exception as e:
            logger.error(f"Failed to execute prediction: {e}")
            raise

    async def execute(self, context: str, **kwargs) -> SpeculativeExecutionResult:  # type: ignore[override]
        """Execute speculative execution pattern.

        Generates predictions based on context, executes high-confidence
        predictions speculatively, and caches results.

        Args:
            context: Current conversation context.
            **kwargs: Additional arguments (unused).

        Returns:
            SpeculativeExecutionResult with predictions and cached results.
        """
        start_time = time.perf_counter()

        self.emit_event("pattern.speculative.started", {"context_length": len(context)})

        try:
            # Generate predictions
            predictions = self.predict_next_queries(context)

            self.emit_event(
                "pattern.speculative.predicted",
                {
                    "num_predictions": len(predictions),
                    "avg_confidence": (
                        sum(p.confidence for p in predictions) / len(predictions)
                        if predictions
                        else 0.0
                    ),
                },
            )

            # Execute predictions concurrently
            executed_predictions: List[ToolPrediction] = []
            tasks = []

            for prediction in predictions:
                if prediction.confidence >= self.config.min_confidence:
                    tasks.append(self._execute_prediction(prediction))
                    executed_predictions.append(prediction)

            if tasks:
                cached_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter out exceptions
                valid_results = [
                    r for r in cached_results if isinstance(r, SpeculativeResult)
                ]

                self._total_executions += len(valid_results)

                self.emit_event(
                    "pattern.speculative.executed",
                    {
                        "num_executed": len(valid_results),
                        "num_failed": len(cached_results) - len(valid_results),
                    },
                )
            else:
                valid_results = []

            # Cleanup expired entries
            self.cleanup()

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            result = SpeculativeExecutionResult(
                pattern_id=self.config.pattern_id,
                success=True,
                result=valid_results,
                execution_time_ms=execution_time_ms,
                predictions=predictions,
                executed=executed_predictions,
                cached_results=valid_results,
                metadata={
                    "cache_size": len(self.cache),
                    "history_size": len(self.history),
                    "total_predictions": self._total_predictions,
                    "total_executions": self._total_executions,
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses,
                    "cache_hit_rate": (
                        self._cache_hits / (self._cache_hits + self._cache_misses)
                        if (self._cache_hits + self._cache_misses) > 0
                        else 0.0
                    ),
                },
            )

            self.emit_event(
                "pattern.speculative.completed",
                {
                    "execution_time_ms": execution_time_ms,
                    "num_cached": len(valid_results),
                },
            )

            return result

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Speculative execution failed: {e}")

            return SpeculativeExecutionResult(
                pattern_id=self.config.pattern_id,
                success=False,
                result=None,
                execution_time_ms=execution_time_ms,
                errors=[str(e)],
            )

    def cleanup(self) -> None:
        """Remove expired cache entries."""
        expired_keys = [
            key for key, result in self.cache.items() if result.is_expired()
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics including cache performance.

        Returns:
            Dictionary with detailed statistics.
        """
        base_stats = super().get_stats()

        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0
            else 0.0
        )

        return {
            **base_stats,
            "total_predictions": self._total_predictions,
            "total_executions": self._total_executions,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache),
            "history_size": len(self.history),
        }
