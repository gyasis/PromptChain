"""Observe panel widget for verbose observability mode.

Shows detailed internal steps, tool calls, LLM responses, and execution
results in real-time when --verbose flag is enabled.

T118: Verbose observability mode implementation.
"""

from typing import List, Optional, Union
from datetime import datetime
from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import Static
from rich.text import Text


class ObservePanel(Container):
    """Panel displaying detailed execution observability information.

    Shows:
    - Internal reasoning steps
    - Tool calls with arguments and results
    - LLM requests and responses
    - Execution timing
    - Token usage

    Designed for developers to understand and debug agent behavior.
    """

    DEFAULT_CSS = """
    ObservePanel {
        display: none;
        height: auto;
        max-height: 20;
        border: none;
        background: transparent;
        padding: 0;
        margin: 0;
    }

    ObservePanel.visible {
        display: block;
    }

    #observe-header {
        height: 1;
        color: #888888;
        background: transparent;
    }

    #observe-content {
        height: auto;
        max-height: 18;
        padding: 0;
        background: transparent;
        overflow-y: auto;
        scrollbar-size: 1 1;
    }

    .observe-entry {
        height: auto;
        padding: 0;
    }

    .observe-entry.tool-call {
        color: #6699cc;
    }

    .observe-entry.tool-result {
        color: #66cc99;
    }

    .observe-entry.llm-request {
        color: #cc9966;
    }

    .observe-entry.llm-response {
        color: #99cc66;
    }

    .observe-entry.reasoning {
        color: #9966cc;
    }

    .observe-entry.error {
        color: #cc6666;
    }

    .observe-entry.info {
        color: #666666;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize observe panel."""
        super().__init__(*args, **kwargs)
        self._entries: List[dict] = []
        self._max_entries = 100  # Keep last 100 entries

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Static("[Verbose Mode] Observability Panel", id="observe-header")
        yield ScrollableContainer(id="observe-content")

    def log_entry(
        self,
        entry_type: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> None:
        """Log a new observability entry.

        Args:
            entry_type: Type of entry (tool-call, tool-result, llm-request, llm-response, reasoning, error, info)
            content: Entry content
            metadata: Optional metadata (timing, tokens, etc.)
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        entry = {
            "timestamp": timestamp,
            "type": entry_type,
            "content": content,
            "metadata": metadata or {}
        }

        self._entries.append(entry)

        # Trim old entries
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        self._update_display()

    def log_tool_call(self, tool_name: str, args: dict) -> None:
        """Log a tool call with arguments."""
        args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items())
        self.log_entry(
            "tool-call",
            f"CALL {tool_name}({args_str})"
        )

    def log_tool_result(self, tool_name: str, result: str, success: bool = True) -> None:
        """Log a tool result."""
        status = "OK" if success else "ERROR"
        # Truncate long results
        result_preview = result[:200] + "..." if len(result) > 200 else result
        self.log_entry(
            "tool-result" if success else "error",
            f"RESULT [{status}] {tool_name}: {result_preview}"
        )

    def log_llm_request(self, model: str, prompt_preview: str) -> None:
        """Log an LLM request."""
        # Truncate long prompts
        prompt_preview = prompt_preview[:100] + "..." if len(prompt_preview) > 100 else prompt_preview
        self.log_entry(
            "llm-request",
            f"LLM [{model}] Request: {prompt_preview}"
        )

    def log_llm_response(self, model: str, response_preview: str, tokens: Optional[int] = None) -> None:
        """Log an LLM response."""
        # Truncate long responses
        response_preview = response_preview[:100] + "..." if len(response_preview) > 100 else response_preview
        token_str = f" ({tokens} tokens)" if tokens else ""
        self.log_entry(
            "llm-response",
            f"LLM [{model}] Response{token_str}: {response_preview}"
        )

    def log_reasoning(self, step: Union[int, str], thought: str) -> None:
        """Log a reasoning step.

        Args:
            step: Step number or hierarchical step (e.g., "1.5" for processor 1, step 5)
            thought: Reasoning content to log
        """
        # Truncate long thoughts
        thought_preview = thought[:150] + "..." if len(thought) > 150 else thought
        self.log_entry(
            "reasoning",
            f"THINK [Step {step}]: {thought_preview}"
        )

    def log_info(self, message: str) -> None:
        """Log an informational message."""
        self.log_entry("info", message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.log_entry("error", f"ERROR: {message}")

    def _update_display(self) -> None:
        """Update the display with current entries."""
        try:
            content = self.query_one("#observe-content", ScrollableContainer)
        except Exception:
            return

        content.remove_children()

        # Show last 30 entries in the scrollable area
        display_entries = self._entries[-30:]

        for entry in display_entries:
            text = Text()
            # Escape square brackets in content to prevent Rich markup conflicts
            content_text = entry['content'].replace('[', '\\[').replace(']', '\\]')
            text.append(f"[{entry['timestamp']}] ", style="dim")
            text.append(content_text)

            entry_static = Static(text)
            entry_static.add_class("observe-entry")
            entry_static.add_class(entry['type'])
            content.mount(entry_static)

        # Auto-scroll to bottom
        content.scroll_end(animate=False)

    def show_panel(self) -> None:
        """Show the observe panel."""
        self.add_class("visible")

    def hide_panel(self) -> None:
        """Hide the observe panel."""
        self.remove_class("visible")

    def toggle_panel(self) -> None:
        """Toggle panel visibility."""
        if "visible" in self.classes:
            self.hide_panel()
        else:
            self.show_panel()

    def clear(self) -> None:
        """Clear all entries."""
        self._entries = []
        try:
            content = self.query_one("#observe-content", ScrollableContainer)
            content.remove_children()
        except Exception:
            pass
