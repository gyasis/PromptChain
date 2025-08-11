"""
Research Agent System

Advanced research automation with 3-tier RAG processing.
"""

__version__ = "0.1.0.b"

from .core.config import ResearchConfig
from .core.orchestrator import AdvancedResearchOrchestrator
from .core.session import ResearchSession

__all__ = [
    "ResearchConfig",
    "AdvancedResearchOrchestrator", 
    "ResearchSession"
]