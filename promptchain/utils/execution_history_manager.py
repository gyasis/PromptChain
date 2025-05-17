# execution_history_manager.py
"""Manages the state and history of complex agentic workflow executions."""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
import logging

# Configure basic logging for this module
logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate counting
try:
    import tiktoken
    _tiktoken_found = True
except ImportError:
    tiktoken = None
    _tiktoken_found = False
    logger.warning("tiktoken not found. History token limits will use character estimation.")

# Define possible entry types
HistoryEntryType = Literal[
    "user_input",
    "agent_output",
    "tool_call",
    "tool_result",
    "system_message",
    "error",
    "custom"
]

# Define truncation strategies (only 'oldest_first' implemented)
TruncationStrategy = Literal["oldest_first"]

DEFAULT_TRUNCATION_STRATEGY: TruncationStrategy = "oldest_first"

class ExecutionHistoryManager:
    """Stores and provides access to the history of an execution flow.

    Manages a structured list of events/outputs from various sources
    (user, agents, tools) and allows retrieving formatted history
    with filtering options. Supports automatic truncation based on
    maximum number of entries or maximum token count.
    """

    def __init__(self,
                 max_entries: Optional[int] = None,
                 max_tokens: Optional[int] = None,
                 truncation_strategy: TruncationStrategy = DEFAULT_TRUNCATION_STRATEGY):
        """Initializes the history manager.

        Args:
            max_entries: Optional max number of history entries to keep.
            max_tokens: Optional max number of tokens the history should contain.
                        Requires tiktoken for accuracy, falls back to char estimation.
                        Truncation prioritizes staying under this limit if set.
            truncation_strategy: Method used for truncation ('oldest_first' currently).
                                 Placeholder for future advanced strategies.
        """
        self._history: List[Dict[str, Any]] = []
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self.truncation_strategy = truncation_strategy
        self._tokenizer = None
        self._current_token_count = 0 # Track tokens efficiently

        if self.max_tokens is not None:
            try:
                if _tiktoken_found:
                    # Using cl100k_base as a common default
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
                    logger.info("Initialized with tiktoken encoder (cl100k_base) for token counting.")
                else:
                    logger.warning("max_tokens specified, but tiktoken not found. Using character estimation.")
            except Exception as e:
                logger.error(f"Failed to initialize tiktoken encoder: {e}. Falling back to char count.")
                self._tokenizer = None

        logger.info(f"ExecutionHistoryManager initialized (max_entries: {max_entries or 'Unlimited'}, max_tokens: {max_tokens or 'Unlimited'}, strategy: {truncation_strategy}).")

    def _count_tokens(self, text: str) -> int:
        """Counts tokens using tiktoken if available, otherwise estimates from chars."""
        if not text:
            return 0
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except Exception as e:
                 logger.warning(f"Tiktoken encoding error: {e}. Falling back to char count estimate.")
                 # Fallback estimate
                 return len(text) // 4
        else:
            # Fallback estimate if tiktoken is not available
            return len(text) // 4

    def _count_entry_tokens(self, entry: Dict[str, Any]) -> int:
        """Calculates the token count for a history entry, focusing on content."""
        # Consider counting other fields later if needed (source, type, metadata keys?)
        content_str = str(entry.get('content', ''))
        return self._count_tokens(content_str)

    def add_entry(
        self,
        entry_type: HistoryEntryType,
        content: Any,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Adds a new entry and applies truncation rules (tokens first, then entries)."""
        entry = {
            "type": entry_type,
            "source": source,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        new_entry_tokens = self._count_entry_tokens(entry)

        # Append first, then truncate
        self._history.append(entry)
        self._current_token_count += new_entry_tokens
        logger.debug(f"History entry added: Type='{entry_type}', Source='{source}', Tokens={new_entry_tokens}, TotalTokens={self._current_token_count}")

        # Apply Truncation Rules
        self._apply_truncation()

    def _apply_truncation(self):
        """Applies truncation rules based on tokens and entries."""
        truncated = False

        # 1. Token Limit Truncation (if enabled)
        if self.max_tokens is not None and self._current_token_count > self.max_tokens:
            if self.truncation_strategy == "oldest_first":
                while self._current_token_count > self.max_tokens and len(self._history) > 1:
                    removed_entry = self._history.pop(0) # Remove oldest
                    removed_tokens = self._count_entry_tokens(removed_entry)
                    self._current_token_count -= removed_tokens
                    truncated = True
                    logger.debug(f"Truncated oldest entry (tokens={removed_tokens}) due to token limit. New total: {self._current_token_count}")
            # Add other strategies here later
            else:
                logger.warning(f"Unsupported truncation_strategy: {self.truncation_strategy}. Using 'oldest_first'.")
                # Fallback to default strategy
                while self._current_token_count > self.max_tokens and len(self._history) > 1:
                    removed_entry = self._history.pop(0)
                    removed_tokens = self._count_entry_tokens(removed_entry)
                    self._current_token_count -= removed_tokens
                    truncated = True
                    logger.debug(f"Truncated oldest entry (tokens={removed_tokens}) due to token limit. New total: {self._current_token_count}")
            if truncated:
                logger.info(f"History truncated due to max_tokens ({self.max_tokens}). Final token count: {self._current_token_count}")
            # Ensure count is accurate after potential rounding/estimation issues
            if len(self._history) <= 1 and self._current_token_count > self.max_tokens:
                 logger.warning(f"Token count ({self._current_token_count}) still exceeds limit ({self.max_tokens}) after truncating all but one entry. This might happen with very large single entries.")
            elif self._current_token_count < 0: # Safety check
                 self._recalculate_token_count() # Recalculate if count becomes inconsistent

        # 2. Max Entries Truncation (applied AFTER token truncation, if enabled)
        if self.max_entries is not None and len(self._history) > self.max_entries:
            if self.truncation_strategy == "oldest_first":
                num_to_remove = len(self._history) - self.max_entries
                entries_to_remove = self._history[:num_to_remove]
                self._history = self._history[num_to_remove:]
                # Recalculate token count if entries were removed here
                removed_tokens_count = sum(self._count_entry_tokens(e) for e in entries_to_remove)
                self._current_token_count -= removed_tokens_count
                truncated = True
                logger.debug(f"Truncated {num_to_remove} oldest entries due to max_entries limit.")
            # Add other strategies here later
            else:
                 logger.warning(f"Unsupported truncation_strategy for max_entries: {self.truncation_strategy}. Using 'oldest_first'.")
                 num_to_remove = len(self._history) - self.max_entries
                 entries_to_remove = self._history[:num_to_remove]
                 self._history = self._history[num_to_remove:]
                 removed_tokens_count = sum(self._count_entry_tokens(e) for e in entries_to_remove)
                 self._current_token_count -= removed_tokens_count
                 truncated = True
                 logger.debug(f"Truncated {num_to_remove} oldest entries due to max_entries limit.")
            if truncated:
                logger.info(f"History truncated due to max_entries ({self.max_entries}). Final entry count: {len(self._history)}")
            if self._current_token_count < 0: # Safety check
                 self._recalculate_token_count()

    def _recalculate_token_count(self):
        """Recalculates the total token count from scratch. Use if tracking gets inconsistent."""
        logger.warning("Recalculating total token count for history...")
        self._current_token_count = sum(self._count_entry_tokens(e) for e in self._history)
        logger.info(f"Recalculation complete. Total token count: {self._current_token_count}")

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the complete current history (list of entry dictionaries)."""
        return self._history.copy() # Return a copy to prevent external modification

    def clear_history(self):
        """Clears all entries from the history."""
        self._history = []
        self._current_token_count = 0 # Reset token count
        logger.info("Execution history cleared.")

    def get_formatted_history(
        self,
        include_types: Optional[List[HistoryEntryType]] = None,
        include_sources: Optional[List[str]] = None,
        exclude_types: Optional[List[HistoryEntryType]] = None,
        exclude_sources: Optional[List[str]] = None,
        max_entries: Optional[int] = None,
        max_tokens: Optional[int] = None, # Added max_tokens filter here too
        format_style: Literal['chat', 'full_json', 'content_only'] = 'chat'
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
            temp_history = [e for e in temp_history if e.get('type') in include_types]
        if include_sources:
            temp_history = [e for e in temp_history if e.get('source') in include_sources]
        if exclude_types:
            temp_history = [e for e in temp_history if e.get('type') not in exclude_types]
        if exclude_sources:
            temp_history = [e for e in temp_history if e.get('source') not in exclude_sources]

        # 2. Apply max_entries limit (from most recent)
        if max_entries is not None:
            temp_history = temp_history[-max_entries:]

        # 3. Apply max_tokens limit (from most recent backwards)
        final_filtered_entries = []
        current_token_count = 0
        if max_tokens is not None:
            for entry in reversed(temp_history):
                entry_tokens = self._count_entry_tokens(entry)
                if current_token_count + entry_tokens <= max_tokens:
                    final_filtered_entries.insert(0, entry) # Insert at beginning to maintain order
                    current_token_count += entry_tokens
                else:
                    logger.debug(f"Token limit ({max_tokens}) reached during get_formatted_history. Stopping inclusion.")
                    break # Stop adding entries once limit exceeded
        else:
            # If no token limit for formatting, use all entries from step 2
            final_filtered_entries = temp_history

        # 4. Format output
        if not final_filtered_entries:
            return "No relevant history entries found matching criteria and limits."

        if format_style == 'full_json':
            try:
                return json.dumps(final_filtered_entries, indent=2)
            except TypeError as e:
                logger.error(f"Error serializing history to JSON: {e}. Returning basic string representation.")
                return "\n".join([str(e) for e in final_filtered_entries])
        elif format_style == 'content_only':
            return "\n".join([str(e.get('content', '')) for e in final_filtered_entries])
        elif format_style == 'chat':
            formatted_lines = []
            for entry in final_filtered_entries:
                prefix = "" # Default prefix
                content_str = str(entry.get('content', ''))
                entry_type = entry.get('type')
                source = entry.get('source')

                if entry_type == 'user_input':
                    prefix = "User:"
                elif entry_type == 'agent_output':
                    prefix = f"Agent ({source or 'Unknown'}):"
                elif entry_type == 'tool_result':
                    prefix = f"Tool Result ({source or 'Unknown'}):"
                elif entry_type == 'error':
                    prefix = f"Error ({source or 'System'}):"
                elif entry_type == 'system_message':
                    prefix = "System:"
                # Add more formatting rules as needed

                formatted_lines.append(f"{prefix} {content_str}".strip())
            return "\n".join(formatted_lines)
        else:
            logger.warning(f"Unknown format_style: '{format_style}'. Defaulting to chat format.")
            return self.get_formatted_history(
                # Pass original filters, apply default format
                include_types=include_types,
                include_sources=include_sources,
                exclude_types=exclude_types,
                exclude_sources=exclude_sources,
                max_entries=max_entries,
                max_tokens=max_tokens,
                format_style='chat'
            )

    def __len__(self):
        """Return the number of entries in the history."""
        return len(self._history)

    def __str__(self):
        """Return a simple string representation of the manager."""
        return f"ExecutionHistoryManager(entries={len(self._history)}, max_entries={self.max_entries or 'Unlimited'}, max_tokens={self.max_tokens or 'Unlimited'})" 