"""CLI/TUI-specific ``AgenticStepProcessor`` subclass.

This module defines :class:`TUIAgenticStepProcessor`, a thin wrapper around
:class:`promptchain.utils.agentic_step_processor.AgenticStepProcessor` that
bakes in the legacy v0.5.0 hardcoded TUI system prompt (via
:class:`promptchain.prompts.LegacyTUIPromptGenerator`).

This subclass exists to preserve byte-identical behavior for the PromptChain
CLI/TUI after the prompt-builder decoupling (spec 011). It is intended for
consumers of the CLI/TUI layer only.

Library consumers should use :class:`AgenticStepProcessor` directly — either
with the default :class:`DynamicPromptGenerator` (tool-accurate, minimal) or
with a custom :class:`BasePromptBuilder` implementation.

See ``specs/011-agentic-prompt-builder/contracts/prompt_builder_protocol.md``
(Section 5) for the authoritative contract.
"""

from __future__ import annotations

from typing import Any

from promptchain.prompts import LegacyTUIPromptGenerator
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


class TUIAgenticStepProcessor(AgenticStepProcessor):
    """``AgenticStepProcessor`` with the legacy TUI prompt pre-wired.

    This subclass is a transparent pass-through for every constructor
    argument except ``prompt_builder`` and ``instructions``, which are
    rejected to keep the legacy TUI behavior unambiguous.

    Raises:
        TypeError: If ``prompt_builder`` or ``instructions`` is supplied.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "prompt_builder" in kwargs:
            raise TypeError(
                "TUIAgenticStepProcessor does not accept prompt_builder=. "
                "Use AgenticStepProcessor directly if you need a custom builder."
            )
        if "instructions" in kwargs:
            raise TypeError(
                "TUIAgenticStepProcessor does not accept instructions=. "
                "Use AgenticStepProcessor(instructions=...) directly."
            )
        super().__init__(*args, prompt_builder=LegacyTUIPromptGenerator(), **kwargs)
