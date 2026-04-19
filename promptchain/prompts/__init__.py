"""Prompt-builder strategy objects for AgenticStepProcessor.

Public surface:
- ``BasePromptBuilder`` — structural Protocol (see ``base.py``).
- ``DynamicPromptGenerator`` — truthful default, renders only registered tools.
- ``LegacyTUIPromptGenerator`` — preserved v0.5.0 hardcoded TUI prompt.

See ``specs/011-agentic-prompt-builder/contracts/prompt_builder_protocol.md``
for the authoritative contract.
"""

from promptchain.prompts.base import BasePromptBuilder
from promptchain.prompts.dynamic import DynamicPromptGenerator
from promptchain.prompts.legacy_tui import LegacyTUIPromptGenerator

__all__ = [
    "BasePromptBuilder",
    "DynamicPromptGenerator",
    "LegacyTUIPromptGenerator",
]
