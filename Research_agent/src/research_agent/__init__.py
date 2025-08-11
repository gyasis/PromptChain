"""
Research Agent System

Advanced iterative research agent with 3-tier RAG processing and PromptChain orchestration.
"""

from .core.config import ResearchConfig
from .core.orchestrator import AdvancedResearchOrchestrator
from .core.session import ResearchSession

__version__ = "0.1.0"
__all__ = [
    "ResearchConfig",
    "AdvancedResearchOrchestrator", 
    "ResearchSession"
]