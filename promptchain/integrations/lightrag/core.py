"""LightRAG Integration wrapper for PromptChain.

Provides a unified interface to hybridrag components.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING
import logging

from promptchain.integrations.lightrag import LIGHTRAG_AVAILABLE

if TYPE_CHECKING:
    from hybridrag.src.lightrag_core import HybridLightRAGCore
    from hybridrag.src.search_interface import SearchInterface

logger = logging.getLogger(__name__)


@dataclass
class LightRAGConfig:
    """Configuration for LightRAG integration.

    ``model_name`` and ``embedding_model`` have no defaults; the caller names
    them. They are validated when LightRAG is first used.
    """
    working_dir: str = "./lightrag_data"
    model_name: Optional[str] = None
    embedding_model: Optional[str] = None
    max_tokens: int = 4000
    chunk_size: int = 1000
    chunk_overlap: int = 200
    extra_params: Dict[str, Any] = field(default_factory=dict)


class LightRAGIntegration:
    """Unified wrapper for hybridrag/LightRAG components.

    This class provides a single entry point to access all LightRAG
    capabilities including:
    - Local queries (entity-specific retrieval)
    - Global queries (high-level conceptual retrieval)
    - Hybrid queries (combined approach)
    - Agentic search (multi-hop reasoning)
    - Multi-query search (query expansion)

    Example:
        >>> integration = LightRAGIntegration(config=LightRAGConfig(working_dir="./my_data"))
        >>> result = await integration.query("What is machine learning?")
    """

    def __init__(self, config: Optional[LightRAGConfig] = None):
        """Initialize LightRAG integration.

        Args:
            config: Configuration for LightRAG. Uses defaults if not provided.

        Raises:
            ImportError: If hybridrag is not installed.
        """
        if not LIGHTRAG_AVAILABLE:
            raise ImportError(
                "hybridrag is not installed. Install with: "
                "pip install git+https://github.com/gyasis/hybridrag.git"
            )

        self.config = config or LightRAGConfig()
        self._core: Optional["HybridLightRAGCore"] = None
        self._search: Optional["SearchInterface"] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of hybridrag components."""
        if self._initialized:
            return

        if not self.config.model_name:
            raise ValueError(
                "LightRAGConfig.model_name is required: name the LLM model "
                "(for example 'provider/model') when constructing LightRAGConfig."
            )
        if not self.config.embedding_model:
            raise ValueError(
                "LightRAGConfig.embedding_model is required: name the embedding "
                "model when constructing LightRAGConfig."
            )

        from hybridrag.src.lightrag_core import HybridLightRAGCore
        from hybridrag.src.search_interface import SearchInterface

        # Initialize core LightRAG
        self._core = HybridLightRAGCore(
            working_dir=self.config.working_dir,
            model_name=self.config.model_name,
            embedding_model=self.config.embedding_model,
            **self.config.extra_params
        )

        # Initialize search interface
        self._search = SearchInterface(lightrag_core=self._core)

        self._initialized = True
        logger.info(f"LightRAG initialized with working_dir={self.config.working_dir}")

    @property
    def core(self) -> "HybridLightRAGCore":
        """Get the underlying HybridLightRAGCore instance."""
        self._ensure_initialized()
        return self._core  # type: ignore

    @property
    def search(self) -> "SearchInterface":
        """Get the SearchInterface instance."""
        self._ensure_initialized()
        return self._search  # type: ignore

    async def local_query(self, query: str, top_k: int = 10, **kwargs) -> Any:
        """Execute a local (entity-specific) query.

        Args:
            query: The search query.
            top_k: Number of top results to return.
            **kwargs: Additional arguments passed to HybridLightRAGCore.local_query.

        Returns:
            Query result with entity-focused information.
        """
        self._ensure_initialized()
        return await self._core.local_query(query, top_k=top_k, **kwargs)  # type: ignore

    async def global_query(self, query: str, top_k: int = 10, **kwargs) -> Any:
        """Execute a global (high-level overview) query.

        Args:
            query: The search query.
            top_k: Number of top results to return.
            **kwargs: Additional arguments passed to HybridLightRAGCore.global_query.

        Returns:
            Query result with conceptual overview information.
        """
        self._ensure_initialized()
        return await self._core.global_query(query, top_k=top_k, **kwargs)  # type: ignore

    async def hybrid_query(self, query: str, top_k: int = 10, **kwargs) -> Any:
        """Execute a hybrid (local + global) query.

        Args:
            query: The search query.
            top_k: Number of top results to return.
            **kwargs: Additional arguments passed to HybridLightRAGCore.hybrid_query.

        Returns:
            Query result combining entity-specific and conceptual information.
        """
        self._ensure_initialized()
        return await self._core.hybrid_query(query, top_k=top_k, **kwargs)  # type: ignore

    async def agentic_search(
        self,
        query: str,
        objective: Optional[str] = None,
        max_steps: int = 5,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Execute an agentic (multi-hop) search.

        Uses AgenticStepProcessor for complex reasoning across multiple
        retrieval steps.

        Args:
            query: The search query.
            objective: Optional objective for the search (defaults to answering query).
            max_steps: Maximum number of reasoning steps.
            model_name: Model to use for agentic reasoning (falls back to
                ``config.model_name``, which must then be set).
            **kwargs: Additional arguments passed to SearchInterface.agentic_search.

        Returns:
            Search result with multi-hop reasoning.
        """
        self._ensure_initialized()
        return await self._search.agentic_search(  # type: ignore
            query=query,
            objective=objective or f"Comprehensively answer: {query}",
            max_steps=max_steps,
            model_name=model_name or self.config.model_name,
            **kwargs
        )

    async def multi_query_search(
        self,
        queries: list[str],
        mode: str = "hybrid",
        **kwargs
    ) -> Any:
        """Execute parallel search across multiple query variations.

        Args:
            queries: List of query variations to search.
            mode: Query mode ("local", "global", "hybrid").
            **kwargs: Additional arguments passed to SearchInterface.multi_query_search.

        Returns:
            Aggregated search results from all queries.
        """
        self._ensure_initialized()
        return await self._search.multi_query_search(  # type: ignore
            queries=queries,
            mode=mode,
            **kwargs
        )

    async def extract_context(self, query: str, **kwargs) -> Any:
        """Extract context from knowledge graph for a query.

        Args:
            query: The query to extract context for.
            **kwargs: Additional arguments.

        Returns:
            Extracted context including entities and relationships.
        """
        self._ensure_initialized()
        return await self._core.extract_context(query, **kwargs)  # type: ignore

    def is_available(self) -> bool:
        """Check if LightRAG is available and configured."""
        return LIGHTRAG_AVAILABLE
