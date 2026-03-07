# execution_history_manager.py
"""Manages the state and history of complex agentic workflow executions."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

# Configure basic logging for this module
logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate counting
try:
    import tiktoken

    _tiktoken_found = True
except ImportError:
    tiktoken = None  # type: ignore[assignment]
    _tiktoken_found = False
    logger.warning(
        "tiktoken not found. History token limits will use character estimation."
    )

# Define possible entry types
HistoryEntryType = Literal[
    "user_input",
    "agent_output",
    "tool_call",
    "tool_result",
    "system_message",
    "error",
    "agentic_step",  # T053: Reasoning step progress
    "agentic_exhaustion",  # T055: Max steps exhaustion
    "custom",
]

# Define truncation strategies (T078: Added "keep_last")
TruncationStrategy = Literal["oldest_first", "keep_last"]

DEFAULT_TRUNCATION_STRATEGY: TruncationStrategy = "oldest_first"


class ExecutionHistoryManager:
    """Stores and provides access to the history of an execution flow.

    Manages a structured list of events/outputs from various sources
    (user, agents, tools) and allows retrieving formatted history
    with filtering options. Supports automatic truncation based on
    maximum number of entries or maximum token count.
    """

    def __init__(
        self,
        max_entries: Optional[int] = None,
        max_tokens: Optional[int] = None,
        truncation_strategy: TruncationStrategy = DEFAULT_TRUNCATION_STRATEGY,
        callback_manager: Optional[Any] = None,
    ):
        """Initializes the history manager.

        Args:
            max_entries: Optional max number of history entries to keep.
            max_tokens: Optional max number of tokens the history should contain.
                        Requires tiktoken for accuracy, falls back to char estimation.
                        Truncation prioritizes staying under this limit if set.
            truncation_strategy: Method used for truncation ('oldest_first' currently).
                                 Placeholder for future advanced strategies.
            callback_manager: Optional CallbackManager for emitting events.
        """
        self._history: List[Dict[str, Any]] = []
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self.truncation_strategy = truncation_strategy
        self.callback_manager = callback_manager
        self._tokenizer = None
        self._current_token_count = 0  # Track tokens efficiently

        if self.max_tokens is not None:
            try:
                if _tiktoken_found:
                    # Using cl100k_base as a common default
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
                    logger.info(
                        "Initialized with tiktoken encoder (cl100k_base) for token counting."
                    )
                else:
                    logger.warning(
                        "max_tokens specified, but tiktoken not found. Using character estimation."
                    )
            except Exception as e:
                logger.error(
                    f"Failed to initialize tiktoken encoder: {e}. Falling back to char count."
                )
                self._tokenizer = None

        logger.info(
            f"ExecutionHistoryManager initialized (max_entries: {max_entries or 'Unlimited'}, max_tokens: {max_tokens or 'Unlimited'}, strategy: {truncation_strategy})."
        )

    def _count_tokens(self, text: str) -> int:
        """Counts tokens using tiktoken if available, otherwise estimates from chars."""
        if not text:
            return 0
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except Exception as e:
                logger.warning(
                    f"Tiktoken encoding error: {e}. Falling back to char count estimate."
                )
                # Fallback estimate
                return len(text) // 4
        else:
            # Fallback estimate if tiktoken is not available
            return len(text) // 4

    def _count_entry_tokens(self, entry: Dict[str, Any]) -> int:
        """Calculates the token count for a history entry, focusing on content."""
        # Consider counting other fields later if needed (source, type, metadata keys?)
        content_str = str(entry.get("content", ""))
        return self._count_tokens(content_str)

    def add_entry(
        self,
        entry_type: HistoryEntryType,
        content: Any,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Adds a new entry and applies truncation rules (tokens first, then entries)."""
        entry = {
            "type": entry_type,
            "source": source,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        new_entry_tokens = self._count_entry_tokens(entry)

        # Append first, then truncate
        self._history.append(entry)
        self._current_token_count += new_entry_tokens
        logger.debug(
            f"History entry added: Type='{entry_type}', Source='{source}', Tokens={new_entry_tokens}, TotalTokens={self._current_token_count}"
        )

        # Apply Truncation Rules
        self._apply_truncation()

    def _apply_truncation(self):
        """Applies truncation rules based on tokens and entries."""
        truncated = False
        entries_removed = 0
        tokens_removed = 0

        # 1. Token Limit Truncation (if enabled)
        if self.max_tokens is not None and self._current_token_count > self.max_tokens:
            if self.truncation_strategy == "oldest_first":
                while (
                    self._current_token_count > self.max_tokens
                    and len(self._history) > 1
                ):
                    removed_entry = self._history.pop(0)  # Remove oldest
                    removed_tokens = self._count_entry_tokens(removed_entry)
                    self._current_token_count -= removed_tokens
                    tokens_removed += removed_tokens
                    entries_removed += 1
                    truncated = True
                    logger.debug(
                        f"Truncated oldest entry (tokens={removed_tokens}) due to token limit. New total: {self._current_token_count}"
                    )
            elif self.truncation_strategy == "keep_last":
                # T078: Keep only the most recent entries within token limit
                # Remove from the beginning until we're under the limit
                while (
                    self._current_token_count > self.max_tokens
                    and len(self._history) > 1
                ):
                    removed_entry = self._history.pop(
                        0
                    )  # Remove oldest (same as oldest_first)
                    removed_tokens = self._count_entry_tokens(removed_entry)
                    self._current_token_count -= removed_tokens
                    tokens_removed += removed_tokens
                    entries_removed += 1
                    truncated = True
                    logger.debug(
                        f"Truncated oldest entry (tokens={removed_tokens}) using keep_last strategy. New total: {self._current_token_count}"
                    )
            else:
                logger.warning(
                    f"Unsupported truncation_strategy: {self.truncation_strategy}. Using 'oldest_first'."
                )
                # Fallback to default strategy
                while (
                    self._current_token_count > self.max_tokens
                    and len(self._history) > 1
                ):
                    removed_entry = self._history.pop(0)
                    removed_tokens = self._count_entry_tokens(removed_entry)
                    self._current_token_count -= removed_tokens
                    tokens_removed += removed_tokens
                    entries_removed += 1
                    truncated = True
                    logger.debug(
                        f"Truncated oldest entry (tokens={removed_tokens}) due to token limit. New total: {self._current_token_count}"
                    )
            if truncated:
                logger.info(
                    f"History truncated due to max_tokens ({self.max_tokens}). Final token count: {self._current_token_count}"
                )

                # Emit HISTORY_TRUNCATED event for token-based truncation
                if self.callback_manager:
                    try:
                        from .execution_events import (ExecutionEvent,
                                                       ExecutionEventType)

                        self.callback_manager.emit_sync(
                            ExecutionEvent(
                                event_type=ExecutionEventType.HISTORY_TRUNCATED,
                                timestamp=datetime.now(),
                                metadata={
                                    "strategy": self.truncation_strategy,
                                    "reason": "max_tokens",
                                    "entries_removed": entries_removed,
                                    "tokens_removed": tokens_removed,
                                    "final_token_count": self._current_token_count,
                                    "max_tokens": self.max_tokens,
                                },
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to emit HISTORY_TRUNCATED event: {e}")
            # Ensure count is accurate after potential rounding/estimation issues
            if len(self._history) <= 1 and self._current_token_count > self.max_tokens:
                logger.warning(
                    f"Token count ({self._current_token_count}) still exceeds limit ({self.max_tokens}) after truncating all but one entry. This might happen with very large single entries."
                )
            elif self._current_token_count < 0:  # Safety check
                self._recalculate_token_count()  # Recalculate if count becomes inconsistent

        # 2. Max Entries Truncation (applied AFTER token truncation, if enabled)
        entries_truncated_for_max = False
        max_entries_removed = 0
        max_entries_tokens_removed = 0

        if self.max_entries is not None and len(self._history) > self.max_entries:
            if self.truncation_strategy == "oldest_first":
                num_to_remove = len(self._history) - self.max_entries
                entries_to_remove = self._history[:num_to_remove]
                self._history = self._history[num_to_remove:]
                # Recalculate token count if entries were removed here
                removed_tokens_count = sum(
                    self._count_entry_tokens(e) for e in entries_to_remove
                )
                self._current_token_count -= removed_tokens_count
                max_entries_removed = num_to_remove
                max_entries_tokens_removed = removed_tokens_count
                entries_truncated_for_max = True
                logger.debug(
                    f"Truncated {num_to_remove} oldest entries due to max_entries limit."
                )
            elif self.truncation_strategy == "keep_last":
                # T078: Keep only the most recent max_entries entries
                # Remove from the beginning (same as oldest_first for max_entries)
                num_to_remove = len(self._history) - self.max_entries
                entries_to_remove = self._history[:num_to_remove]
                self._history = self._history[num_to_remove:]
                removed_tokens_count = sum(
                    self._count_entry_tokens(e) for e in entries_to_remove
                )
                self._current_token_count -= removed_tokens_count
                max_entries_removed = num_to_remove
                max_entries_tokens_removed = removed_tokens_count
                entries_truncated_for_max = True
                logger.debug(
                    f"Truncated {num_to_remove} oldest entries using keep_last strategy (max_entries limit)."
                )
            else:
                logger.warning(
                    f"Unsupported truncation_strategy for max_entries: {self.truncation_strategy}. Using 'oldest_first'."
                )
                num_to_remove = len(self._history) - self.max_entries
                entries_to_remove = self._history[:num_to_remove]
                self._history = self._history[num_to_remove:]
                removed_tokens_count = sum(
                    self._count_entry_tokens(e) for e in entries_to_remove
                )
                self._current_token_count -= removed_tokens_count
                max_entries_removed = num_to_remove
                max_entries_tokens_removed = removed_tokens_count
                entries_truncated_for_max = True
                logger.debug(
                    f"Truncated {num_to_remove} oldest entries due to max_entries limit."
                )
            if entries_truncated_for_max:
                logger.info(
                    f"History truncated due to max_entries ({self.max_entries}). Final entry count: {len(self._history)}"
                )

                # Emit HISTORY_TRUNCATED event for max_entries truncation
                if self.callback_manager:
                    try:
                        from .execution_events import (ExecutionEvent,
                                                       ExecutionEventType)

                        self.callback_manager.emit_sync(
                            ExecutionEvent(
                                event_type=ExecutionEventType.HISTORY_TRUNCATED,
                                timestamp=datetime.now(),
                                metadata={
                                    "strategy": self.truncation_strategy,
                                    "reason": "max_entries",
                                    "entries_removed": max_entries_removed,
                                    "tokens_removed": max_entries_tokens_removed,
                                    "final_entry_count": len(self._history),
                                    "max_entries": self.max_entries,
                                },
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to emit HISTORY_TRUNCATED event: {e}")
            if self._current_token_count < 0:  # Safety check
                self._recalculate_token_count()

    def _recalculate_token_count(self):
        """Recalculates the total token count from scratch. Use if tracking gets inconsistent."""
        logger.warning("Recalculating total token count for history...")
        self._current_token_count = sum(
            self._count_entry_tokens(e) for e in self._history
        )
        logger.info(
            f"Recalculation complete. Total token count: {self._current_token_count}"
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the complete current history (list of entry dictionaries)."""
        return self._history.copy()  # Return a copy to prevent external modification

    def clear_history(self):
        """Clears all entries from the history."""
        self._history = []
        self._current_token_count = 0  # Reset token count
        logger.info("Execution history cleared.")

    def clear(self):
        """Alias for clear_history() for convenience."""
        self.clear_history()

    def add_exhaustion_entry(
        self,
        objective: str,
        max_steps: int,
        steps_completed: int,
        partial_result: Optional[str] = None,
        source: str = "agentic_step_processor",
    ) -> None:
        """Record agentic exhaustion in execution history (T055).

        Args:
            objective: The objective that wasn't completed
            max_steps: Maximum steps configured
            steps_completed: Actual steps completed
            partial_result: Partial results before exhaustion
            source: Source of the exhaustion event
        """
        content = f"Max steps ({max_steps}) reached for objective: {objective}"

        if partial_result:
            content += f"\n\nPartial result ({len(partial_result)} chars):\n{partial_result[:200]}..."

        metadata = {
            "objective": objective,
            "max_steps": max_steps,
            "steps_completed": steps_completed,
            "completion_status": "exhausted",
            "suggestions": [
                "Increase max_internal_steps in agent config",
                "Simplify the objective to be more specific",
                "Break complex objective into smaller sub-objectives",
                "Review if objective is achievable with available tools",
            ],
        }

        if partial_result:
            metadata["partial_result_preview"] = partial_result[:200]

        self.add_entry(
            entry_type="agentic_exhaustion",
            content=content,
            metadata=metadata,
            source=source,
        )

        logger.warning(
            f"Agentic exhaustion recorded: {steps_completed}/{max_steps} steps "
            f"completed for objective: {objective[:50]}..."
        )

    def get_formatted_history(
        self,
        include_types: Optional[List[HistoryEntryType]] = None,
        include_sources: Optional[List[str]] = None,
        exclude_types: Optional[List[HistoryEntryType]] = None,
        exclude_sources: Optional[List[str]] = None,
        max_entries: Optional[int] = None,
        max_tokens: Optional[int] = None,  # Added max_tokens filter here too
        format_style: Literal["chat", "full_json", "content_only"] = "chat",
    ) -> str:
        """Retrieves and formats the history based on filtering criteria and limits.

        Filtering Logic:
        1. Inclusion/Exclusion filters are applied.
        2. max_entries filter is applied (takes most recent N).
        3. max_tokens filter is applied to the result of step 2 (takes most recent fitting tokens).

        Args:
            include_types: List of entry types to include.
            include_sources: List of source identifiers to include.
            exclude_types: List of entry types to exclude.
            exclude_sources: List of source identifiers to exclude.
            max_entries: Max number of *most recent* entries to initially consider.
            max_tokens: Max token count for the final formatted output.
                        Iterates backwards from recent entries until limit is met.
            format_style: How to format the output string ('chat', 'full_json', 'content_only').

        Returns:
            A formatted string representation of the filtered and limited history.
        """
        # 1. Initial Filtering (Types/Sources)
        temp_history = self._history
        if include_types:
            temp_history = [e for e in temp_history if e.get("type") in include_types]
        if include_sources:
            temp_history = [
                e for e in temp_history if e.get("source") in include_sources
            ]
        if exclude_types:
            temp_history = [
                e for e in temp_history if e.get("type") not in exclude_types
            ]
        if exclude_sources:
            temp_history = [
                e for e in temp_history if e.get("source") not in exclude_sources
            ]

        # 2. Apply max_entries limit (from most recent)
        if max_entries is not None:
            temp_history = temp_history[-max_entries:]

        # 3. Apply max_tokens limit (from most recent backwards)
        final_filtered_entries: List[Dict[str, Any]] = []
        current_token_count = 0
        if max_tokens is not None:
            for entry in reversed(temp_history):
                entry_tokens = self._count_entry_tokens(entry)
                if current_token_count + entry_tokens <= max_tokens:
                    final_filtered_entries.insert(
                        0, entry
                    )  # Insert at beginning to maintain order
                    current_token_count += entry_tokens
                else:
                    logger.debug(
                        f"Token limit ({max_tokens}) reached during get_formatted_history. Stopping inclusion."
                    )
                    break  # Stop adding entries once limit exceeded
        else:
            # If no token limit for formatting, use all entries from step 2
            final_filtered_entries = temp_history

        # 4. Format output
        if not final_filtered_entries:
            return "No relevant history entries found matching criteria and limits."

        if format_style == "full_json":
            try:
                return json.dumps(final_filtered_entries, indent=2)
            except TypeError as e:
                logger.error(
                    f"Error serializing history to JSON: {e}. Returning basic string representation."
                )
                return "\n".join([str(e) for e in final_filtered_entries])
        elif format_style == "content_only":
            return "\n".join(
                [str(e.get("content", "")) for e in final_filtered_entries]
            )
        elif format_style == "chat":
            formatted_lines = []
            for entry in final_filtered_entries:
                prefix = ""  # Default prefix
                content_str = str(entry.get("content", ""))
                entry_type = entry.get("type")
                source = entry.get("source")

                if entry_type == "user_input":
                    prefix = "User:"
                elif entry_type == "agent_output":
                    prefix = f"Agent ({source or 'Unknown'}):"
                elif entry_type == "tool_result":
                    prefix = f"Tool Result ({source or 'Unknown'}):"
                elif entry_type == "error":
                    prefix = f"Error ({source or 'System'}):"
                elif entry_type == "system_message":
                    prefix = "System:"
                # Add more formatting rules as needed

                formatted_lines.append(f"{prefix} {content_str}".strip())
            return "\n".join(formatted_lines)
        else:
            logger.warning(
                f"Unknown format_style: '{format_style}'. Defaulting to chat format."
            )
            return self.get_formatted_history(
                # Pass original filters, apply default format
                include_types=include_types,
                include_sources=include_sources,
                exclude_types=exclude_types,
                exclude_sources=exclude_sources,
                max_entries=max_entries,
                max_tokens=max_tokens,
                format_style="chat",
            )

    def __len__(self):
        """Return the number of entries in the history."""
        return len(self._history)

    # Public API Properties (v0.4.1a)
    @property
    def current_token_count(self) -> int:
        """Public getter for current token count.

        Returns:
            int: Current total token count across all history entries.
        """
        return self._current_token_count

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Public getter for history entries (read-only copy).

        Returns:
            List[Dict[str, Any]]: Copy of all history entries.
        """
        return self._history.copy()

    @property
    def history_size(self) -> int:
        """Get number of entries in history.

        Returns:
            int: Number of history entries currently stored.
        """
        return len(self._history)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive history statistics.

        Returns:
            Dict[str, Any]: Statistics dictionary containing:
                - total_tokens (int): Current total token count
                - total_entries (int): Number of history entries
                - max_tokens (int|None): Maximum token limit (None if unlimited)
                - max_entries (int|None): Maximum entry limit (None if unlimited)
                - utilization_pct (float): Percentage of token limit used (0.0 if unlimited)
                - entry_types (Dict[str, int]): Count of entries by type
                - truncation_strategy (str): Current truncation strategy
        """
        # Calculate entry type distribution
        entry_types: Dict[str, int] = {}
        for entry in self._history:
            entry_type = entry.get("type", "unknown")
            entry_types[entry_type] = entry_types.get(entry_type, 0) + 1

        # Calculate utilization percentage
        utilization_pct = 0.0
        if self.max_tokens is not None and self.max_tokens > 0:
            utilization_pct = (self._current_token_count / self.max_tokens) * 100.0

        return {
            "total_tokens": self._current_token_count,
            "total_entries": len(self._history),
            "max_tokens": self.max_tokens,
            "max_entries": self.max_entries,
            "utilization_pct": utilization_pct,
            "entry_types": entry_types,
            "truncation_strategy": self.truncation_strategy,
        }

    def __str__(self):
        """Return a simple string representation of the manager."""
        return f"ExecutionHistoryManager(entries={len(self._history)}, max_entries={self.max_entries or 'Unlimited'}, max_tokens={self.max_tokens or 'Unlimited'})"


# ============================================================================
# Context Distillation (Issue #6 - wU2 Pattern from Claude Code)
# ============================================================================


class ContextDistiller:
    """Implements context compression using LLM-based summarization.

    Issue #6 Enhancement: Implements wU2 pattern from Claude Code for
    automatic context compression at 70% token capacity. Instead of
    truncating old messages, generates "Current State of Knowledge" summary
    for better context preservation.

    Benefits:
    - 30-50% token reduction while maintaining context quality
    - Preserved conversation continuity
    - Reduced information loss compared to truncation
    """

    def __init__(
        self, llm_caller: Optional[Any] = None, distillation_threshold: float = 0.7
    ):
        """Initialize context distiller.

        Args:
            llm_caller: Optional LLM caller for generating summaries.
                       If None, uses simple extraction without LLM summarization.
            distillation_threshold: Token capacity threshold (0.0-1.0) that triggers
                                   distillation. Default 0.7 (70% capacity).
        """
        self.llm_caller = llm_caller
        self.distillation_threshold = distillation_threshold
        logger.info(
            f"ContextDistiller initialized (threshold: {distillation_threshold*100}%)"
        )

    def should_distill(self, history_manager: ExecutionHistoryManager) -> bool:
        """Check if history should be distilled based on token utilization.

        Args:
            history_manager: History manager to check

        Returns:
            True if token usage exceeds threshold, False otherwise
        """
        if history_manager.max_tokens is None:
            return False  # No token limit, no need to distill

        stats = history_manager.get_statistics()
        utilization = stats["utilization_pct"] / 100.0  # Convert to 0-1 range

        return utilization >= self.distillation_threshold

    def distill_history(
        self, history: List[Dict[str, Any]], max_tokens: int, preserve_recent: int = 3
    ) -> Dict[str, Any]:
        """Distill conversation history into compact state representation.

        Args:
            history: List of history entries to distill
            max_tokens: Target token count after distillation
            preserve_recent: Number of recent messages to keep verbatim (default: 3)

        Returns:
            Dictionary containing:
                - state_summary: LLM-generated summary of knowledge state
                - recent_messages: Last N messages preserved verbatim
                - key_decisions: Extracted important decisions
                - open_issues: Extracted unresolved issues
                - tokens_before: Token count before distillation
                - tokens_after: Token count after distillation
        """
        if not history:
            return {
                "state_summary": "No history to distill",
                "recent_messages": [],
                "key_decisions": [],
                "open_issues": [],
                "tokens_before": 0,
                "tokens_after": 0,
            }

        # Preserve recent messages
        recent_messages = (
            history[-preserve_recent:] if len(history) >= preserve_recent else history
        )
        older_messages = (
            history[:-preserve_recent] if len(history) > preserve_recent else []
        )

        # Extract key information from older messages
        key_decisions = self._extract_key_decisions(older_messages)
        open_issues = self._extract_open_issues(older_messages)

        # Generate state summary
        if self.llm_caller and older_messages:
            state_summary = self._generate_llm_summary(
                older_messages, key_decisions, open_issues
            )
        else:
            state_summary = self._generate_simple_summary(
                older_messages, key_decisions, open_issues
            )

        # Calculate token savings (approximate)
        tokens_before = sum(len(str(e.get("content", ""))) // 4 for e in history)
        tokens_after = len(state_summary) // 4 + sum(
            len(str(e.get("content", ""))) // 4 for e in recent_messages
        )

        logger.info(
            f"Context distilled: {len(older_messages)} older messages compressed into summary. "
            f"Token reduction: {tokens_before} → {tokens_after} "
            f"({((tokens_before - tokens_after) / tokens_before * 100):.1f}% savings)"
        )

        return {
            "state_summary": state_summary,
            "recent_messages": recent_messages,
            "key_decisions": key_decisions,
            "open_issues": open_issues,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
        }

    def _extract_key_decisions(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract key decisions from message history.

        Args:
            messages: List of history entries

        Returns:
            List of key decision strings
        """
        decisions = []
        decision_keywords = [
            "decided",
            "chose",
            "selected",
            "agreed",
            "confirmed",
            "approved",
        ]

        for entry in messages:
            content = str(entry.get("content", "")).lower()
            for keyword in decision_keywords:
                if keyword in content:
                    # Extract the sentence containing the decision
                    sentences = content.split(".")
                    for sentence in sentences:
                        if keyword in sentence:
                            decisions.append(
                                sentence.strip()[:200]
                            )  # Truncate long sentences
                            break

        return decisions[:5]  # Return top 5 decisions

    def _extract_open_issues(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract unresolved issues from message history.

        Args:
            messages: List of history entries

        Returns:
            List of open issue strings
        """
        issues = []
        issue_keywords = [
            "error",
            "failed",
            "issue",
            "problem",
            "bug",
            "todo",
            "need to",
            "must fix",
        ]

        for entry in messages:
            content = str(entry.get("content", "")).lower()
            entry_type = entry.get("type", "")

            # Prioritize error entries
            if entry_type == "error":
                issues.append(f"Error: {str(entry.get('content', ''))[:200]}")
            else:
                for keyword in issue_keywords:
                    if keyword in content:
                        sentences = content.split(".")
                        for sentence in sentences:
                            if keyword in sentence:
                                issues.append(sentence.strip()[:200])
                                break

        return issues[:5]  # Return top 5 issues

    def _generate_simple_summary(
        self,
        messages: List[Dict[str, Any]],
        key_decisions: List[str],
        open_issues: List[str],
    ) -> str:
        """Generate simple summary without LLM (fallback).

        Args:
            messages: Message history to summarize
            key_decisions: Extracted decisions
            open_issues: Extracted issues

        Returns:
            Summary string
        """
        summary_parts = [
            "=== Current State of Knowledge (Distilled) ===",
            f"\nCompressed {len(messages)} older messages into this summary.\n",
        ]

        if key_decisions:
            summary_parts.append("\nKey Decisions:")
            for i, decision in enumerate(key_decisions, 1):
                summary_parts.append(f"  {i}. {decision}")

        if open_issues:
            summary_parts.append("\nOpen Issues:")
            for i, issue in enumerate(open_issues, 1):
                summary_parts.append(f"  {i}. {issue}")

        # Add brief context about what was discussed
        topics = set()
        for entry in messages[-10:]:  # Look at last 10 for topics
            content_lower = str(entry.get("content", "")).lower()
            # Extract potential topics (simplified)
            words = content_lower.split()
            for word in words:
                if len(word) > 8 and word.isalpha():  # Longer words likely topics
                    topics.add(word)

        if topics:
            summary_parts.append(
                f"\nRecent Discussion Topics: {', '.join(list(topics)[:5])}"
            )

        return "\n".join(summary_parts)

    def _generate_llm_summary(
        self,
        messages: List[Dict[str, Any]],
        key_decisions: List[str],
        open_issues: List[str],
    ) -> str:
        """Generate LLM-powered summary of conversation history.

        Args:
            messages: Message history to summarize
            key_decisions: Extracted decisions
            open_issues: Extracted issues

        Returns:
            LLM-generated summary string
        """
        # Prepare context for LLM
        context_parts = []
        for entry in messages:
            entry_type = entry.get("type", "unknown")
            content = str(entry.get("content", ""))
            context_parts.append(f"[{entry_type}] {content}")

        context = "\n".join(context_parts[-20:])  # Last 20 messages for context

        prompt = f"""Summarize the following conversation history into a "Current State of Knowledge" summary.
Focus on:
1. What was accomplished
2. Current understanding and context
3. Decisions made
4. Open issues or next steps

Conversation History:
{context}

Key Decisions Identified:
{chr(10).join(f'- {d}' for d in key_decisions) if key_decisions else 'None'}

Open Issues Identified:
{chr(10).join(f'- {i}' for i in open_issues) if open_issues else 'None'}

Provide a concise summary (2-3 paragraphs max):"""

        try:
            if self.llm_caller:
                # Call LLM to generate summary
                summary = self.llm_caller(prompt)
                return f"=== Current State of Knowledge (Distilled) ===\n\n{summary}"
        except Exception as e:
            logger.error(
                f"LLM summary generation failed: {e}. Falling back to simple summary."
            )

        # Fallback to simple summary
        return self._generate_simple_summary(messages, key_decisions, open_issues)

    def apply_distillation(self, history_manager: ExecutionHistoryManager) -> bool:
        """Apply distillation to a history manager in-place.

        Args:
            history_manager: History manager to distill

        Returns:
            True if distillation was applied, False otherwise
        """
        if not self.should_distill(history_manager):
            return False

        # Get current history
        current_history = history_manager.get_history()

        # Perform distillation
        distilled = self.distill_history(
            current_history,
            max_tokens=history_manager.max_tokens or 10000,
            preserve_recent=3,
        )

        # Clear and rebuild history with distilled content
        history_manager.clear_history()

        # Add distilled summary as system message
        history_manager.add_entry(
            entry_type="system_message",
            content=distilled["state_summary"],
            source="context_distiller",
            metadata={
                "distillation_applied": True,
                "tokens_before": distilled["tokens_before"],
                "tokens_after": distilled["tokens_after"],
                "key_decisions": distilled["key_decisions"],
                "open_issues": distilled["open_issues"],
            },
        )

        # Add back recent messages
        for entry in distilled["recent_messages"]:
            history_manager.add_entry(
                entry_type=entry.get("type", "custom"),
                content=entry.get("content"),
                source=entry.get("source"),
                metadata=entry.get("metadata", {}),
            )

        logger.info(
            f"Context distillation applied successfully. "
            f"Token savings: {distilled['tokens_before'] - distilled['tokens_after']} "
            f"({((distilled['tokens_before'] - distilled['tokens_after']) / distilled['tokens_before'] * 100):.1f}%)"
        )

        return True
