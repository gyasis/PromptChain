"""Pattern executor functions for TUI and CLI integration.

This module provides standalone async functions for executing agentic patterns.
These functions are used by both:
- Click CLI commands (promptchain patterns branch "query")
- TUI slash commands (/branch "query")

Each executor returns a standardized dict:
{
    "success": bool,
    "result": dict,  # Pattern-specific results
    "error": str | None,
    "execution_time_ms": float
}
"""

import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from promptchain.cli.models import MessageBus, Blackboard


# Check if LightRAG integration is available
try:
    from promptchain.integrations.lightrag import LIGHTRAG_AVAILABLE
except ImportError:
    LIGHTRAG_AVAILABLE = False


class PatternNotAvailableError(Exception):
    """Raised when LightRAG/hybridrag is not installed."""
    pass


def _check_lightrag_available():
    """Check if LightRAG is available, raise helpful error if not."""
    if not LIGHTRAG_AVAILABLE:
        raise PatternNotAvailableError(
            "LightRAG integration not available.\n"
            "Install: pip install git+https://github.com/gyasis/hybridrag.git"
        )


async def execute_branch(
    query: str,
    count: int = 3,
    mode: str = "hybrid",
    deeplake_path: Optional[str] = None,
    verbose: bool = False,
    message_bus: Optional["MessageBus"] = None,
    blackboard: Optional["Blackboard"] = None,
) -> Dict[str, Any]:
    """Execute branching thoughts pattern.

    Generates multiple hypotheses for a problem and evaluates them.

    Args:
        query: The problem/question to analyze
        count: Number of hypotheses to generate (default: 3)
        mode: Search mode - local, global, or hybrid (default: hybrid)
        deeplake_path: Path to DeepLake dataset
        verbose: Enable verbose output
        message_bus: Optional MessageBus for event emission
        blackboard: Optional Blackboard for state sharing

    Returns:
        Dict with success, result (hypotheses), error, execution_time_ms
    """
    _check_lightrag_available()
    start_time = time.perf_counter()

    try:
        from promptchain.integrations.lightrag import LightRAGBranchingThoughts, PatternConfig

        config = PatternConfig(
            pattern_id="tui-branch",
            emit_events=message_bus is not None,
            use_blackboard=blackboard is not None,
        )

        pattern = LightRAGBranchingThoughts(
            config=config,
            deeplake_path=deeplake_path or "~/.lightrag/default",
            search_mode=mode.lower(),
            num_hypotheses=count,
            verbose=verbose,
        )

        if message_bus:
            pattern.connect_messagebus(message_bus)
        if blackboard:
            pattern.connect_blackboard(blackboard)

        result = await pattern.execute_with_timeout(query=query)

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "success": result.success,
            "result": result.result if result.success else None,
            "hypotheses": result.result.get("hypotheses", []) if result.success and isinstance(result.result, dict) else [],
            "error": result.errors[0] if result.errors else None,
            "execution_time_ms": execution_time_ms,
            "metadata": result.metadata,
        }

    except PatternNotAvailableError:
        raise
    except Exception as e:
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        return {
            "success": False,
            "result": None,
            "hypotheses": [],
            "error": str(e),
            "execution_time_ms": execution_time_ms,
            "metadata": {},
        }


async def execute_expand(
    query: str,
    strategies: Optional[List[str]] = None,
    max_expansions: int = 5,
    deeplake_path: Optional[str] = None,
    verbose: bool = False,
    message_bus: Optional["MessageBus"] = None,
    blackboard: Optional["Blackboard"] = None,
) -> Dict[str, Any]:
    """Execute query expansion pattern.

    Generates alternative query formulations to improve retrieval coverage.

    Args:
        query: The search query to expand
        strategies: Expansion strategies (semantic, synonym, acronym, contextual)
        max_expansions: Maximum variations to generate (default: 5)
        deeplake_path: Path to DeepLake dataset
        verbose: Enable verbose output
        message_bus: Optional MessageBus for event emission
        blackboard: Optional Blackboard for state sharing

    Returns:
        Dict with success, result (expansions), error, execution_time_ms
    """
    _check_lightrag_available()
    start_time = time.perf_counter()

    if strategies is None:
        strategies = ["semantic"]

    try:
        from promptchain.integrations.lightrag import LightRAGQueryExpander, PatternConfig

        config = PatternConfig(
            pattern_id="tui-expand",
            emit_events=message_bus is not None,
            use_blackboard=blackboard is not None,
        )

        pattern = LightRAGQueryExpander(
            config=config,
            deeplake_path=deeplake_path or "~/.lightrag/default",
            expansion_strategies=list(strategies),
            max_expansions=max_expansions,
            verbose=verbose,
        )

        if message_bus:
            pattern.connect_messagebus(message_bus)
        if blackboard:
            pattern.connect_blackboard(blackboard)

        result = await pattern.execute_with_timeout(query=query)

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "success": result.success,
            "result": result.result if result.success else None,
            "expansions": result.result.get("expansions", []) if result.success and isinstance(result.result, dict) else [],
            "error": result.errors[0] if result.errors else None,
            "execution_time_ms": execution_time_ms,
            "metadata": result.metadata,
        }

    except PatternNotAvailableError:
        raise
    except Exception as e:
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        return {
            "success": False,
            "result": None,
            "expansions": [],
            "error": str(e),
            "execution_time_ms": execution_time_ms,
            "metadata": {},
        }


async def execute_multihop(
    query: str,
    max_hops: int = 5,
    objective: Optional[str] = None,
    mode: str = "hybrid",
    deeplake_path: Optional[str] = None,
    verbose: bool = False,
    message_bus: Optional["MessageBus"] = None,
    blackboard: Optional["Blackboard"] = None,
) -> Dict[str, Any]:
    """Execute multi-hop retrieval pattern.

    Performs iterative reasoning across multiple retrieval steps.

    Args:
        query: The complex question requiring multi-hop reasoning
        max_hops: Maximum reasoning hops (default: 5)
        objective: Custom objective for reasoning
        mode: Search mode - local, global, or hybrid
        deeplake_path: Path to DeepLake dataset
        verbose: Enable verbose output
        message_bus: Optional MessageBus for event emission
        blackboard: Optional Blackboard for state sharing

    Returns:
        Dict with success, result (hops, answer), error, execution_time_ms
    """
    _check_lightrag_available()
    start_time = time.perf_counter()

    try:
        from promptchain.integrations.lightrag import LightRAGMultiHop, PatternConfig

        config = PatternConfig(
            pattern_id="tui-multihop",
            emit_events=message_bus is not None,
            use_blackboard=blackboard is not None,
        )

        pattern = LightRAGMultiHop(
            config=config,
            deeplake_path=deeplake_path or "~/.lightrag/default",
            search_mode=mode.lower(),
            max_hops=max_hops,
            objective=objective,
            verbose=verbose,
        )

        if message_bus:
            pattern.connect_messagebus(message_bus)
        if blackboard:
            pattern.connect_blackboard(blackboard)

        result = await pattern.execute_with_timeout(query=query)

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "success": result.success,
            "result": result.result if result.success else None,
            "hops": result.result.get("hops", []) if result.success and isinstance(result.result, dict) else [],
            "answer": result.result.get("answer", "") if result.success and isinstance(result.result, dict) else "",
            "error": result.errors[0] if result.errors else None,
            "execution_time_ms": execution_time_ms,
            "metadata": result.metadata,
        }

    except PatternNotAvailableError:
        raise
    except Exception as e:
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        return {
            "success": False,
            "result": None,
            "hops": [],
            "answer": "",
            "error": str(e),
            "execution_time_ms": execution_time_ms,
            "metadata": {},
        }


async def execute_hybrid(
    query: str,
    fusion: str = "rrf",
    weights: tuple = (0.5, 0.5),
    top_k: int = 10,
    deeplake_path: Optional[str] = None,
    verbose: bool = False,
    message_bus: Optional["MessageBus"] = None,
    blackboard: Optional["Blackboard"] = None,
) -> Dict[str, Any]:
    """Execute hybrid search fusion pattern.

    Combines local and global search results using fusion algorithms.

    Args:
        query: The search query
        fusion: Fusion algorithm - rrf, linear, or borda (default: rrf)
        weights: Weights for [local, global] search (default: 0.5, 0.5)
        top_k: Number of results to return (default: 10)
        deeplake_path: Path to DeepLake dataset
        verbose: Enable verbose output
        message_bus: Optional MessageBus for event emission
        blackboard: Optional Blackboard for state sharing

    Returns:
        Dict with success, result (results), error, execution_time_ms
    """
    _check_lightrag_available()
    start_time = time.perf_counter()

    try:
        from promptchain.integrations.lightrag import LightRAGHybridSearcher, PatternConfig

        config = PatternConfig(
            pattern_id="tui-hybrid",
            emit_events=message_bus is not None,
            use_blackboard=blackboard is not None,
        )

        pattern = LightRAGHybridSearcher(
            config=config,
            deeplake_path=deeplake_path or "~/.lightrag/default",
            fusion_algorithm=fusion.lower(),
            local_weight=weights[0],
            global_weight=weights[1],
            top_k=top_k,
            verbose=verbose,
        )

        if message_bus:
            pattern.connect_messagebus(message_bus)
        if blackboard:
            pattern.connect_blackboard(blackboard)

        result = await pattern.execute_with_timeout(query=query)

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "success": result.success,
            "result": result.result if result.success else None,
            "results": result.result.get("results", []) if result.success and isinstance(result.result, dict) else [],
            "error": result.errors[0] if result.errors else None,
            "execution_time_ms": execution_time_ms,
            "metadata": result.metadata,
        }

    except PatternNotAvailableError:
        raise
    except Exception as e:
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        return {
            "success": False,
            "result": None,
            "results": [],
            "error": str(e),
            "execution_time_ms": execution_time_ms,
            "metadata": {},
        }


async def execute_sharded(
    query: str,
    shards: List[str],
    aggregation: str = "rrf",
    mode: str = "hybrid",
    top_k: int = 10,
    verbose: bool = False,
    message_bus: Optional["MessageBus"] = None,
    blackboard: Optional["Blackboard"] = None,
) -> Dict[str, Any]:
    """Execute sharded retrieval pattern.

    Queries across multiple dataset shards and aggregates results.

    Args:
        query: The search query
        shards: List of shard names/paths to query
        aggregation: Aggregation method - merge, weighted, or rrf (default: rrf)
        mode: Search mode for each shard
        top_k: Number of results per shard (default: 10)
        verbose: Enable verbose output
        message_bus: Optional MessageBus for event emission
        blackboard: Optional Blackboard for state sharing

    Returns:
        Dict with success, result (shard_results, aggregated), error, execution_time_ms
    """
    _check_lightrag_available()
    start_time = time.perf_counter()

    try:
        from promptchain.integrations.lightrag import LightRAGShardedRetriever, PatternConfig

        config = PatternConfig(
            pattern_id="tui-sharded",
            emit_events=message_bus is not None,
            use_blackboard=blackboard is not None,
        )

        pattern = LightRAGShardedRetriever(
            config=config,
            shard_paths=list(shards),
            aggregation_method=aggregation.lower(),
            search_mode=mode.lower(),
            top_k_per_shard=top_k,
            verbose=verbose,
        )

        if message_bus:
            pattern.connect_messagebus(message_bus)
        if blackboard:
            pattern.connect_blackboard(blackboard)

        result = await pattern.execute_with_timeout(query=query)

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "success": result.success,
            "result": result.result if result.success else None,
            "shard_results": result.result.get("shard_results", {}) if result.success and isinstance(result.result, dict) else {},
            "aggregated": result.result.get("aggregated", []) if result.success and isinstance(result.result, dict) else [],
            "error": result.errors[0] if result.errors else None,
            "execution_time_ms": execution_time_ms,
            "metadata": result.metadata,
        }

    except PatternNotAvailableError:
        raise
    except Exception as e:
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        return {
            "success": False,
            "result": None,
            "shard_results": {},
            "aggregated": [],
            "error": str(e),
            "execution_time_ms": execution_time_ms,
            "metadata": {},
        }


async def execute_speculate(
    context: str,
    min_confidence: float = 0.7,
    prefetch_count: int = 3,
    mode: str = "hybrid",
    deeplake_path: Optional[str] = None,
    verbose: bool = False,
    message_bus: Optional["MessageBus"] = None,
    blackboard: Optional["Blackboard"] = None,
) -> Dict[str, Any]:
    """Execute speculative execution pattern.

    Predicts and pre-fetches results for likely follow-up queries.

    Args:
        context: Current conversation context or user intent
        min_confidence: Minimum prediction confidence threshold (default: 0.7)
        prefetch_count: Number of queries to prefetch (default: 3)
        mode: Search mode for speculative queries
        deeplake_path: Path to DeepLake dataset
        verbose: Enable verbose output
        message_bus: Optional MessageBus for event emission
        blackboard: Optional Blackboard for state sharing

    Returns:
        Dict with success, result (predictions), error, execution_time_ms
    """
    _check_lightrag_available()
    start_time = time.perf_counter()

    try:
        from promptchain.integrations.lightrag import LightRAGSpeculativeExecutor, PatternConfig

        config = PatternConfig(
            pattern_id="tui-speculate",
            emit_events=message_bus is not None,
            use_blackboard=blackboard is not None,
        )

        pattern = LightRAGSpeculativeExecutor(
            config=config,
            deeplake_path=deeplake_path or "~/.lightrag/default",
            search_mode=mode.lower(),
            min_confidence=min_confidence,
            max_speculative_queries=prefetch_count,
            verbose=verbose,
        )

        if message_bus:
            pattern.connect_messagebus(message_bus)
        if blackboard:
            pattern.connect_blackboard(blackboard)

        result = await pattern.execute_with_timeout(context=context)

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "success": result.success,
            "result": result.result if result.success else None,
            "predictions": result.result.get("predictions", []) if result.success and isinstance(result.result, dict) else [],
            "cache_info": result.result.get("cache_info", {}) if result.success and isinstance(result.result, dict) else {},
            "error": result.errors[0] if result.errors else None,
            "execution_time_ms": execution_time_ms,
            "metadata": result.metadata,
        }

    except PatternNotAvailableError:
        raise
    except Exception as e:
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        return {
            "success": False,
            "result": None,
            "predictions": [],
            "cache_info": {},
            "error": str(e),
            "execution_time_ms": execution_time_ms,
            "metadata": {},
        }


# Export all executors
__all__ = [
    "execute_branch",
    "execute_expand",
    "execute_multihop",
    "execute_hybrid",
    "execute_sharded",
    "execute_speculate",
    "PatternNotAvailableError",
    "LIGHTRAG_AVAILABLE",
]
