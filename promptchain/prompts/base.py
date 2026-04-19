"""Base protocol for prompt builders used by AgenticStepProcessor.

This module defines the `BasePromptBuilder` Protocol that any prompt-builder
implementation (shipped or third-party) must conform to. It imports only from
the stdlib `typing` module — zero internal PromptChain imports — so library
consumers can implement custom builders without pulling in the rest of the
package.
"""

from typing import Any, Dict, List, Optional, Protocol


class BasePromptBuilder(Protocol):
    """Structural protocol for prompt builders.

    Implementations construct the system prompt consumed by
    `AgenticStepProcessor`. They receive the runtime objective, the actually
    registered tools, and optional prior scratchpad context, and return a
    deterministic string suitable for a `{"role": "system"}` message.
    """

    def generate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> str:
        """Render the full system prompt.

        Args:
            objective: Non-empty string describing what the agent must achieve.
            tools: OpenAI-format function-tool schemas actually registered
                with the processor. Empty list is valid and must be accepted.
            context: Optional prior scratchpad rendered verbatim as the
                PRIOR CONTEXT block when provided.

        Returns:
            A non-empty string suitable for use as the system message content.
        """
        ...

    def get_token_estimate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
    ) -> int:
        """Return a non-negative token estimate for the generated prompt."""
        ...
