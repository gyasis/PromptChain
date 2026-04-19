"""Dynamic prompt builder that renders only the tools actually registered.

This module ships :class:`DynamicPromptGenerator`, the default strategy object
consumed by :class:`AgenticStepProcessor` (spec 011-agentic-prompt-builder).
It replaces the v0.5.0 hardcoded TUI prompt with a truthful, deterministic
prompt assembled from the caller's actual tool inventory.

Module-level constants
----------------------

``_REACT_TASKLIST_TOOLS``
    Allowlist of tool names that the ReAct scaffold considers task-list
    writers. The heuristic used by :meth:`DynamicPromptGenerator.generate`
    first checks whether any registered tool's ``function.name`` contains the
    substring ``task_list`` (case-insensitive); if no such match is found it
    falls back to this explicit set.

See ``specs/011-agentic-prompt-builder/contracts/prompt_builder_protocol.md``
for the authoritative contract.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

# Optional tiktoken import — mirrors the pattern used in
# ``promptchain/utils/execution_history_manager.py`` so that environments
# without tiktoken installed still get a functional (char-based) estimator.
try:  # pragma: no cover - import guard
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False


logger = logging.getLogger(__name__)

#: Tools that the ReAct scaffold treats as task-list writers when the
#: substring heuristic misses. Used by :meth:`DynamicPromptGenerator.generate`.
_REACT_TASKLIST_TOOLS: frozenset = frozenset(
    {"task_list_write_tool", "task_list_read_tool"}
)

_VALID_WORKFLOW_PATTERNS = ("standard", "react")


class DynamicPromptGenerator:
    """Render a system prompt built from the agent's real tool inventory.

    The generator is a pure strategy object: all configuration is supplied at
    construction time, and :meth:`generate` is deterministic with respect to
    its inputs. No process-global mutable state is held.

    Args:
        extra_instructions: Optional list of additional instruction strings.
            Each string is rendered as a bullet under an
            ``ADDITIONAL INSTRUCTIONS:`` header. ``None`` and ``[]`` are
            treated identically — neither emits the block.
        workflow_pattern: Either ``"standard"`` (default) or ``"react"``.
            Any other value raises :class:`ValueError` at construction time.
            ``"react"`` inserts a THINK/PLAN/ACT/OBSERVE scaffold between the
            tool inventory and any additional instructions.
        include_response_format_hint: When ``True`` (default) the prompt ends
            with the canonical ``FINAL ANSWER:`` block instructing the agent
            to include full tool-result content and avoid summarization.

    Raises:
        ValueError: If ``workflow_pattern`` is not one of ``"standard"`` or
            ``"react"``.

    Example:
        >>> gen = DynamicPromptGenerator()
        >>> prompt = gen.generate(
        ...     "ship the release",
        ...     [
        ...         {
        ...             "type": "function",
        ...             "function": {
        ...                 "name": "deploy_service",
        ...                 "description": "Deploy a service build.",
        ...                 "parameters": {"type": "object", "properties": {}},
        ...             },
        ...         }
        ...     ],
        ... )
        >>> "deploy_service" in prompt
        True
    """

    def __init__(
        self,
        *,
        extra_instructions: Optional[List[str]] = None,
        workflow_pattern: Literal["standard", "react"] = "standard",
        include_response_format_hint: bool = True,
    ) -> None:
        if workflow_pattern not in _VALID_WORKFLOW_PATTERNS:
            raise ValueError(
                "workflow_pattern must be one of "
                f"{_VALID_WORKFLOW_PATTERNS!r}, got {workflow_pattern!r}"
            )

        # Normalise empty list to None so downstream checks are a single
        # truthiness test.
        self._extra_instructions: Optional[List[str]] = (
            list(extra_instructions) if extra_instructions else None
        )
        self._workflow_pattern: str = workflow_pattern
        self._include_response_format_hint: bool = include_response_format_hint

        # Lazy-initialised tiktoken encoder; shared across calls on the same
        # instance. ``None`` if tiktoken is unavailable or initialisation
        # failed.
        self._tokenizer: Any = None
        self._tokenizer_checked: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> str:
        """Render the system prompt for the supplied objective and tools.

        Args:
            objective: Non-empty string describing the agent's goal.
            tools: OpenAI-format function-tool schemas. Empty list is valid
                and renders the empty-inventory sentinel. The list is **not**
                mutated — a local copy is made if ordering work is needed.
            context: Optional prior scratchpad text. If supplied, rendered
                verbatim under a ``PRIOR CONTEXT:`` header.

        Returns:
            A deterministic, non-empty string suitable for use as the
            ``content`` of a ``{"role": "system"}`` message.
        """
        blocks: List[str] = []

        # Block 1 — objective line.
        blocks.append(
            f"Your goal is to achieve the following objective: {objective}"
        )

        # Block 2 — available tools.
        blocks.append(self._render_tools_block(tools))

        # ReAct scaffold is inserted between blocks 2 and 3.
        if self._workflow_pattern == "react":
            blocks.append(self._render_react_scaffold(tools))

        # Block 3 — additional instructions (only when non-empty).
        if self._extra_instructions:
            blocks.append(self._render_extra_instructions_block())

        # Block 4 — prior context (only when the caller supplied one).
        if context is not None:
            blocks.append(self._render_context_block(context))

        # Block 5 — final-answer format hint.
        if self._include_response_format_hint:
            blocks.append(self._render_final_answer_block())

        # Blocks are separated by a single blank line for readability.
        return "\n\n".join(blocks)

    def get_token_estimate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
    ) -> int:
        """Return a non-negative token estimate for the generated prompt.

        Uses tiktoken's ``cl100k_base`` encoder when available, otherwise
        falls back to a character-count heuristic (``len(prompt) // 4``).
        Complexity is O(n_tools) in tool-inventory size.

        Args:
            objective: Same as :meth:`generate`.
            tools: Same as :meth:`generate`.

        Returns:
            A non-negative integer.
        """
        # Render with no prior context — token estimates are a floor for the
        # no-scratchpad case. Callers who need exact counts can call
        # ``generate(...)`` and measure the result themselves.
        prompt = self.generate(objective, tools, context=None)
        encoder = self._get_tokenizer()

        if encoder is not None:
            try:
                return max(0, len(encoder.encode(prompt)))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "tiktoken encode failed (%s); falling back to char heuristic",
                    exc,
                )

        # Character-based fallback: ~4 chars per token is the well-known
        # rough estimate for English text.
        return max(0, len(prompt) // 4)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_tools_block(tools: List[Dict[str, Any]]) -> str:
        """Render the ``AVAILABLE TOOLS`` block, including an empty sentinel."""
        lines = ["AVAILABLE TOOLS:"]
        if not tools:
            lines.append("- (no tools registered)")
            return "\n".join(lines)

        for tool in tools:
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = fn.get("name", "<unnamed>")
            description = fn.get("description", "")
            lines.append(f"- {name}: {description}".rstrip())
        return "\n".join(lines)

    def _render_react_scaffold(self, tools: List[Dict[str, Any]]) -> str:
        """Render the ReAct scaffold, with fallback when no task-list tool exists."""
        if self._has_tasklist_tool(tools):
            return (
                "REASONING PATTERN (ReAct):\n"
                "- THINK: state what you know and what you still need.\n"
                "- PLAN: write/update a task list using the task_list tool "
                "before acting.\n"
                "- ACT: call exactly one tool per step.\n"
                "- OBSERVE: record the tool result verbatim before the next "
                "THINK cycle."
            )

        logger.warning(
            "DynamicPromptGenerator: workflow_pattern='react' requested but no "
            "task-list-writer tool found in the inventory (checked substring "
            "'task_list' and allowlist %s). Falling back to a minimal "
            "thought/action scaffold.",
            sorted(_REACT_TASKLIST_TOOLS),
        )
        return (
            "REASONING PATTERN (minimal):\n"
            "- THOUGHT: briefly state your reasoning before each tool call.\n"
            "- ACTION: call one tool at a time and wait for its result "
            "before the next thought."
        )

    def _render_extra_instructions_block(self) -> str:
        assert self._extra_instructions is not None  # narrowed by caller
        lines = ["ADDITIONAL INSTRUCTIONS:"]
        for entry in self._extra_instructions:
            lines.append(f"- {entry}")
        return "\n".join(lines)

    @staticmethod
    def _render_context_block(context: str) -> str:
        return f"PRIOR CONTEXT:\n{context}"

    @staticmethod
    def _render_final_answer_block() -> str:
        # Compact 4-line block — keeps the standard-mode output well under
        # the 15-non-blank-line budget enforced by T013.
        return (
            "FINAL ANSWER:\n"
            "- Include the full content of tool results verbatim; do not "
            "summarize.\n"
            "- Do not announce what you are about to do (no 'I have "
            "explained' / 'I will now').\n"
            "- Answer only after the objective is satisfied."
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _has_tasklist_tool(tools: List[Dict[str, Any]]) -> bool:
        """Return True if any registered tool looks like a task-list writer."""
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function", {})
            if not isinstance(fn, dict):
                continue
            name = fn.get("name", "")
            if not isinstance(name, str):
                continue
            if "task_list" in name.lower():
                return True
            if name in _REACT_TASKLIST_TOOLS:
                return True
        return False

    def _get_tokenizer(self) -> Any:
        """Lazy-initialise the tiktoken encoder; return ``None`` if unavailable."""
        if self._tokenizer_checked:
            return self._tokenizer
        self._tokenizer_checked = True

        if not _TIKTOKEN_AVAILABLE or tiktoken is None:
            return None

        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to initialise tiktoken encoder (%s); using char fallback",
                exc,
            )
            self._tokenizer = None
        return self._tokenizer
