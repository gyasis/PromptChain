"""
ContextDistiller: LLM-based context compression for conversation history.

Re-exports ContextDistiller from execution_history_manager for backwards
compatibility and to satisfy import guards in test_janitor_agent.py.

FR-010: Context distillation integrated with JanitorAgent background monitoring.
"""
from .execution_history_manager import ContextDistiller  # noqa: F401

__all__ = ["ContextDistiller"]
