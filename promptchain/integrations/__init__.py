"""Integration modules for external RAG systems.

This module provides integration wrappers for external projects like hybridrag/LightRAG
that can be used as PromptChain pattern implementations.
"""

# Import conditionally to avoid import errors when dependencies not installed
try:
    from promptchain.integrations.lightrag import (LIGHTRAG_AVAILABLE,
                                                   LightRAGIntegration)
except ImportError:
    LIGHTRAG_AVAILABLE = False
    LightRAGIntegration = None  # type: ignore[misc,assignment]

__all__ = [
    "LIGHTRAG_AVAILABLE",
    "LightRAGIntegration",
]
