"""LightRAG integration via hybridrag project.

This module wraps the hybridrag project's LightRAG components as PromptChain patterns.
Installation: pip install git+https://github.com/gyasis/hybridrag.git

The hybridrag project provides:
- HybridLightRAGCore: Core LightRAG wrapper with local/global/hybrid query modes
- SearchInterface: Search with simple_search, agentic_search, multi_query_search
"""

from typing import TYPE_CHECKING

# Check if hybridrag is installed and import core components
try:
    from hybridrag.src.lightrag_core import HybridLightRAGCore
    from hybridrag.src.search_interface import SearchInterface
    LIGHTRAG_AVAILABLE = True
except ImportError:
    LIGHTRAG_AVAILABLE = False
    HybridLightRAGCore = None  # type: ignore
    SearchInterface = None  # type: ignore

# Import pattern implementations (lazy import to avoid circular dependencies)
if TYPE_CHECKING:
    from promptchain.integrations.lightrag.core import LightRAGIntegration
    from promptchain.integrations.lightrag.branching import LightRAGBranchingThoughts
    from promptchain.integrations.lightrag.query_expansion import LightRAGQueryExpander
    from promptchain.integrations.lightrag.sharded import LightRAGShardedRetriever, LightRAGShardRegistry
    from promptchain.integrations.lightrag.multi_hop import LightRAGMultiHop
    from promptchain.integrations.lightrag.hybrid_search import LightRAGHybridSearcher
    from promptchain.integrations.lightrag.speculative import LightRAGSpeculativeExecutor
    # Integration layer (Wave 3)
    from promptchain.integrations.lightrag.messaging import PatternMessageBusMixin, PatternEventBroadcaster
    from promptchain.integrations.lightrag.state import PatternBlackboardMixin, PatternStateCoordinator, StateSnapshot
    from promptchain.integrations.lightrag.events import PatternEvent, PATTERN_EVENTS, EventSeverity, EventLifecycle


def __getattr__(name: str):
    """Lazy import pattern implementations."""
    if name == "LightRAGIntegration":
        from promptchain.integrations.lightrag.core import LightRAGIntegration
        return LightRAGIntegration
    elif name == "LightRAGBranchingThoughts":
        from promptchain.integrations.lightrag.branching import LightRAGBranchingThoughts
        return LightRAGBranchingThoughts
    elif name == "LightRAGQueryExpander":
        from promptchain.integrations.lightrag.query_expansion import LightRAGQueryExpander
        return LightRAGQueryExpander
    elif name == "LightRAGShardedRetriever":
        from promptchain.integrations.lightrag.sharded import LightRAGShardedRetriever
        return LightRAGShardedRetriever
    elif name == "LightRAGShardRegistry":
        from promptchain.integrations.lightrag.sharded import LightRAGShardRegistry
        return LightRAGShardRegistry
    elif name == "LightRAGMultiHop":
        from promptchain.integrations.lightrag.multi_hop import LightRAGMultiHop
        return LightRAGMultiHop
    elif name == "LightRAGHybridSearcher":
        from promptchain.integrations.lightrag.hybrid_search import LightRAGHybridSearcher
        return LightRAGHybridSearcher
    elif name == "LightRAGSpeculativeExecutor":
        from promptchain.integrations.lightrag.speculative import LightRAGSpeculativeExecutor
        return LightRAGSpeculativeExecutor
    # Integration layer (Wave 3)
    elif name == "PatternMessageBusMixin":
        from promptchain.integrations.lightrag.messaging import PatternMessageBusMixin
        return PatternMessageBusMixin
    elif name == "PatternEventBroadcaster":
        from promptchain.integrations.lightrag.messaging import PatternEventBroadcaster
        return PatternEventBroadcaster
    elif name == "PatternBlackboardMixin":
        from promptchain.integrations.lightrag.state import PatternBlackboardMixin
        return PatternBlackboardMixin
    elif name == "PatternStateCoordinator":
        from promptchain.integrations.lightrag.state import PatternStateCoordinator
        return PatternStateCoordinator
    elif name == "StateSnapshot":
        from promptchain.integrations.lightrag.state import StateSnapshot
        return StateSnapshot
    elif name == "PatternEvent":
        from promptchain.integrations.lightrag.events import PatternEvent
        return PatternEvent
    elif name == "PATTERN_EVENTS":
        from promptchain.integrations.lightrag.events import PATTERN_EVENTS
        return PATTERN_EVENTS
    elif name == "EventSeverity":
        from promptchain.integrations.lightrag.events import EventSeverity
        return EventSeverity
    elif name == "EventLifecycle":
        from promptchain.integrations.lightrag.events import EventLifecycle
        return EventLifecycle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Availability flag
    "LIGHTRAG_AVAILABLE",
    # Core hybridrag components (may be None if not installed)
    "HybridLightRAGCore",
    "SearchInterface",
    # Pattern implementations
    "LightRAGIntegration",
    "LightRAGBranchingThoughts",
    "LightRAGQueryExpander",
    "LightRAGShardedRetriever",
    "LightRAGShardRegistry",
    "LightRAGMultiHop",
    "LightRAGHybridSearcher",
    "LightRAGSpeculativeExecutor",
    # Integration layer (Wave 3)
    "PatternMessageBusMixin",
    "PatternEventBroadcaster",
    "PatternBlackboardMixin",
    "PatternStateCoordinator",
    "StateSnapshot",
    "PatternEvent",
    "PATTERN_EVENTS",
    "EventSeverity",
    "EventLifecycle",
]
