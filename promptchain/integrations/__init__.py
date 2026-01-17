"""Integration modules for external RAG systems.

This module provides integration wrappers for external projects like hybridrag/LightRAG
that can be used as PromptChain pattern implementations.
"""

# Import conditionally to avoid import errors when dependencies not installed
try:
    from promptchain.integrations.lightrag import LightRAGIntegration, LIGHTRAG_AVAILABLE
except ImportError:
    LIGHTRAG_AVAILABLE = False
    LightRAGIntegration = None

__all__ = [
    "LIGHTRAG_AVAILABLE",
    "LightRAGIntegration",
]
