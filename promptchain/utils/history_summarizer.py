"""History Summarizer for PromptChain.

Provides LLM-powered summarization of conversation history to prevent
context window overflow while preserving critical information.

Uses a fast/cheap model (gpt-4.1-mini) to compress old conversation turns
into concise summaries, preserving errors, decisions, and key tool results.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import tiktoken

from promptchain.observability import track_llm_call

logger = logging.getLogger(__name__)


@dataclass
class SummarizationResult:
    """Result of a history summarization operation."""

    summary: str
    original_tokens: int
    summary_tokens: int
    entries_summarized: int
    entries_preserved: int

    @property
    def savings_percent(self) -> float:
        """Calculate percentage of tokens saved."""
        if self.original_tokens == 0:
            return 0.0
        return ((self.original_tokens - self.summary_tokens) / self.original_tokens) * 100


# Summarization prompt template
SUMMARIZATION_PROMPT = """You are a conversation summarizer. Summarize the following conversation history concisely while preserving:

1. **Key decisions made** - What was decided and why
2. **Important results** - Tool outputs, code execution results, API responses
3. **Errors and failures** - Any errors encountered and their context
4. **Progress milestones** - What tasks were completed
5. **Unresolved items** - What still needs to be done

Format your summary as structured bullet points. Be extremely concise - aim for maximum information density.

CONVERSATION TO SUMMARIZE:
{history}

SUMMARY (max {max_tokens} tokens):"""


class HistorySummarizer:
    """LLM-powered history summarization for context management.

    Uses a fast, cheap model to compress conversation history while
    preserving critical information like errors, decisions, and results.

    Attributes:
        model: LiteLLM model string for summarization (default: gpt-4.1-mini)
        max_summary_tokens: Maximum tokens for the summary output
        encoding: Tiktoken encoding for token counting
    """

    def __init__(
        self,
        model: str = "openai/gpt-4.1-mini-2025-04-14",
        max_summary_tokens: int = 500,
    ):
        """Initialize the history summarizer.

        Args:
            model: LiteLLM model string for summarization
            max_summary_tokens: Maximum tokens for summary output
        """
        self.model = model
        self.max_summary_tokens = max_summary_tokens

        # Initialize tiktoken for token counting
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        if not text:
            return 0
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback: rough estimate
            return len(text) // 4

    def _format_entry(self, entry: Dict[str, Any]) -> str:
        """Format a single history entry for summarization.

        Args:
            entry: History entry dict with 'role' and 'content'

        Returns:
            Formatted string representation
        """
        role = entry.get("role", "unknown")
        content = entry.get("content", "")

        # Handle tool calls specially
        if role == "assistant" and entry.get("tool_calls"):
            tool_calls = entry.get("tool_calls", [])
            tool_names = [tc.get("function", {}).get("name", "unknown") for tc in tool_calls]
            return f"[ASSISTANT] Called tools: {', '.join(tool_names)}"

        if role == "tool":
            tool_name = entry.get("name", "unknown")
            # Truncate long tool results
            if len(content) > 500:
                content = content[:400] + f"\n... [{len(content) - 400} chars truncated]"
            return f"[TOOL:{tool_name}] {content}"

        # Standard message formatting
        role_label = role.upper()
        return f"[{role_label}] {content}"

    def _format_history(self, entries: List[Dict[str, Any]]) -> str:
        """Format history entries into a single string for summarization.

        Args:
            entries: List of history entry dicts

        Returns:
            Formatted history string
        """
        formatted_parts = []
        for i, entry in enumerate(entries, 1):
            formatted = self._format_entry(entry)
            formatted_parts.append(f"{i}. {formatted}")

        return "\n\n".join(formatted_parts)

    @track_llm_call(
        model_param="model",
        extract_args=["max_tokens"]
    )
    async def summarize_history(
        self,
        entries: List[Dict[str, Any]],
        preserve_last_n: int = 2,
    ) -> SummarizationResult:
        """Summarize conversation history, preserving recent entries.

        Args:
            entries: List of history entry dicts
            preserve_last_n: Number of recent entries to preserve verbatim

        Returns:
            SummarizationResult with summary and token metrics
        """
        if len(entries) <= preserve_last_n:
            # Nothing to summarize
            return SummarizationResult(
                summary="",
                original_tokens=0,
                summary_tokens=0,
                entries_summarized=0,
                entries_preserved=len(entries),
            )

        # Split entries
        entries_to_summarize = entries[:-preserve_last_n] if preserve_last_n > 0 else entries
        entries_preserved = entries[-preserve_last_n:] if preserve_last_n > 0 else []

        # Format history for summarization
        history_text = self._format_history(entries_to_summarize)
        original_tokens = self.count_tokens(history_text)

        # Build summarization prompt
        prompt = SUMMARIZATION_PROMPT.format(
            history=history_text,
            max_tokens=self.max_summary_tokens,
        )

        # Call LLM for summarization
        try:
            import litellm

            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_summary_tokens,
                temperature=0.3,  # Lower for more consistent summaries
            )

            summary = response.choices[0].message.content.strip()
            summary_tokens = self.count_tokens(summary)

            logger.info(
                f"Summarized {len(entries_to_summarize)} entries: "
                f"{original_tokens} -> {summary_tokens} tokens "
                f"({((original_tokens - summary_tokens) / original_tokens * 100):.1f}% reduction)"
            )

            return SummarizationResult(
                summary=summary,
                original_tokens=original_tokens,
                summary_tokens=summary_tokens,
                entries_summarized=len(entries_to_summarize),
                entries_preserved=len(entries_preserved),
            )

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: return truncated history as summary
            fallback_summary = f"[Summarization failed] Previous {len(entries_to_summarize)} exchanges covered various topics."
            return SummarizationResult(
                summary=fallback_summary,
                original_tokens=original_tokens,
                summary_tokens=self.count_tokens(fallback_summary),
                entries_summarized=len(entries_to_summarize),
                entries_preserved=len(entries_preserved),
            )

    def summarize_history_sync(
        self,
        entries: List[Dict[str, Any]],
        preserve_last_n: int = 2,
    ) -> SummarizationResult:
        """Synchronous wrapper for summarize_history.

        Args:
            entries: List of history entry dicts
            preserve_last_n: Number of recent entries to preserve verbatim

        Returns:
            SummarizationResult with summary and token metrics
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new loop for sync call
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.summarize_history(entries, preserve_last_n),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.summarize_history(entries, preserve_last_n)
                )
        except RuntimeError:
            return asyncio.run(self.summarize_history(entries, preserve_last_n))

    def create_summary_message(self, summary: str) -> Dict[str, Any]:
        """Create a system message containing the summary.

        Args:
            summary: The summarized history text

        Returns:
            Message dict suitable for insertion into history
        """
        return {
            "role": "system",
            "content": f"[Summary of previous conversation]\n{summary}",
        }

    def estimate_tokens_after_summarization(
        self,
        entries: List[Dict[str, Any]],
        preserve_last_n: int = 2,
    ) -> int:
        """Estimate token count after summarization without actually summarizing.

        Args:
            entries: List of history entry dicts
            preserve_last_n: Number of recent entries to preserve

        Returns:
            Estimated token count after summarization
        """
        if len(entries) <= preserve_last_n:
            # No summarization needed
            total = 0
            for entry in entries:
                total += self.count_tokens(str(entry.get("content", "")))
            return total

        # Estimate summary tokens (roughly 10% of original)
        entries_to_summarize = entries[:-preserve_last_n] if preserve_last_n > 0 else entries
        original_tokens = sum(
            self.count_tokens(str(e.get("content", "")))
            for e in entries_to_summarize
        )
        estimated_summary = min(self.max_summary_tokens, max(100, original_tokens // 10))

        # Add preserved entries
        preserved_tokens = 0
        if preserve_last_n > 0:
            for entry in entries[-preserve_last_n:]:
                preserved_tokens += self.count_tokens(str(entry.get("content", "")))

        return estimated_summary + preserved_tokens
