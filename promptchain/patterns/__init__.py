"""Advanced Agentic Patterns for PromptChain.

This module provides the base classes and configurations for implementing
agentic AI patterns including:
- Branching Thoughts (hypothesis generation + judge)
- Query Expansion (parallel query diversification)
- Sharded Retrieval (multi-source parallel queries)
- Multi-Hop Retrieval (question decomposition)
- Hybrid Search Fusion (technique combination)
- Speculative Execution (predictive tool calling)

For LightRAG-based implementations, see promptchain.integrations.lightrag.

For standalone executor functions (used by CLI and TUI), see:
    from promptchain.patterns.executors import execute_branch, execute_expand, ...
"""

from promptchain.patterns.base import (
    BasePattern,
    PatternConfig,
    PatternResult,
)

# Executor functions for CLI/TUI integration
from promptchain.patterns.executors import (
    execute_branch,
    execute_expand,
    execute_multihop,
    execute_hybrid,
    execute_sharded,
    execute_speculate,
    PatternNotAvailableError,
)

__all__ = [
    # Base classes
    "BasePattern",
    "PatternConfig",
    "PatternResult",
    # Executor functions
    "execute_branch",
    "execute_expand",
    "execute_multihop",
    "execute_hybrid",
    "execute_sharded",
    "execute_speculate",
    "PatternNotAvailableError",
]
