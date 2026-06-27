"""Prompt-builder strategy objects for AgenticStepProcessor.

Public surface:
- ``BasePromptBuilder`` — structural Protocol (see ``base.py``).
- ``DynamicPromptGenerator`` — truthful default, renders only registered tools.
- ``LegacyTUIPromptGenerator`` — preserved v0.5.0 hardcoded TUI prompt.
- ``DynamicTUIPromptGenerator`` — static curated TUI foundation + live tool list.
- ``TUI_FOUNDATION_PROMPT`` — the static base the TUI prompt is built from.

See ``specs/011-agentic-prompt-builder/contracts/prompt_builder_protocol.md``
for the authoritative contract.
"""

from promptchain.prompts.base import BasePromptBuilder
from promptchain.prompts.dynamic import DynamicPromptGenerator
from promptchain.prompts.family import family_of
from promptchain.prompts.legacy_tui import LegacyTUIPromptGenerator
from promptchain.prompts.longevity import DocumentAndClear, build_turn_context
from promptchain.prompts.model_dynamic import DynamicModelPromptGenerator
from promptchain.prompts.tiers import PromptTier, select_tier
from promptchain.prompts.tui_dynamic import (
    TUI_FOUNDATION_PROMPT,
    DynamicTUIPromptGenerator,
)

__all__ = [
    "BasePromptBuilder",
    "DynamicPromptGenerator",
    "LegacyTUIPromptGenerator",
    "DynamicTUIPromptGenerator",
    "TUI_FOUNDATION_PROMPT",
    "DynamicModelPromptGenerator",
    "PromptTier",
    "select_tier",
    "family_of",
    "DocumentAndClear",
    "build_turn_context",
]
