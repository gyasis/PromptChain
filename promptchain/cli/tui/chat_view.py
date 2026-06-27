"""ChatView widget for displaying conversation messages."""

import asyncio
from typing import Any, List, Optional, Union

import pyperclip
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.markdown import Heading, Markdown
from rich.padding import Padding
from rich.syntax import Syntax
from rich.text import Text

# Map a file extension → a Pygments lexer name so we can syntax-highlight code
# that a read/write tool returns — WE detect + style it (the agent never formats).
_EXT_LEXER = {
    "py": "python", "pyi": "python", "js": "javascript", "mjs": "javascript",
    "ts": "typescript", "jsx": "jsx", "tsx": "tsx", "java": "java", "go": "go",
    "rs": "rust", "cpp": "cpp", "cc": "cpp", "c": "c", "h": "c", "hpp": "cpp",
    "rb": "ruby", "php": "php", "sh": "bash", "bash": "bash", "zsh": "bash",
    "json": "json", "yaml": "yaml", "yml": "yaml", "toml": "toml", "ini": "ini",
    "md": "markdown", "html": "html", "css": "css", "scss": "scss", "sql": "sql",
    "xml": "xml", "svelte": "html", "vue": "html", "lua": "lua", "r": "r",
}


def _lexer_for_path(path: Optional[str]) -> str:
    if not path or "." not in path:
        return "text"
    return _EXT_LEXER.get(path.rsplit(".", 1)[-1].lower(), "text")


def _esc(s: str) -> str:
    return str(s).replace("[", "\\[").replace("]", "\\]")


def _parse_file_read(result: str):
    """If ``result`` is the file_read summary, return (path, code_preview)."""
    if not result or not result.lstrip().startswith("[FILE READ]"):
        return None, None
    first = result.splitlines()[0]
    path = first.replace("[FILE READ]", "").strip()
    if "Preview:" in result:
        code = result.split("Preview:", 1)[1].strip("\n")
        return path, code
    return path, None


def _is_diffish(text: str) -> bool:
    n = sum(1 for ln in text.splitlines() if ln[:1] in ("+", "-"))
    return n >= 2


def _render_diff(text: str) -> Text:
    """Color a unified-diff-ish blob: + added (green), - removed (red)."""
    out = Text()
    for ln in text.splitlines():
        if ln.startswith("+"):
            out.append(ln + "\n", style="#7ee787")
        elif ln.startswith("-"):
            out.append(ln + "\n", style="#f47067")
        else:
            out.append(ln + "\n", style="#6b7480")
    return out
from textual import events
from textual.message import Message as TextualMessage
from textual.reactive import reactive
from textual.widgets import ListItem, ListView

from ..models import Message

# Event types that count as "thinking / tool detail" — these are the dim,
# high-volume streaming breadcrumbs that can be toggled off to declutter chat.
# Errors are intentionally NOT included: they always stay visible.
DETAIL_EVENT_TYPES = {"thinking", "tool_call", "tool_result"}


class _ChatHeading(Heading):
    """Render markdown headings as clean bold-accent text.

    Rich's default boxes an h1 in a full-width HEAVY panel, which is far too
    loud for a chat TUI (and adds a background-ish frame the dark theme avoids).
    We render every heading as left-aligned bold teal text instead.
    """

    def __rich_console__(self, console, options):  # type: ignore[override]
        text = self.text
        text.justify = "left"
        text.stylize("bold #4ec9b0")
        yield text


class ChatMarkdown(Markdown):
    """Markdown tuned for the dark chat TUI: plain accent headings, no h1 box."""

    elements = {**Markdown.elements, "heading_open": _ChatHeading}


def _looks_like_markdown(text: str) -> bool:
    """Check if text appears to contain markdown formatting.

    Args:
        text: Text to check

    Returns:
        True if text likely contains markdown
    """
    # Common markdown patterns
    markdown_indicators = [
        "**",  # Bold
        "__",  # Bold alt
        "```",  # Code block
        "`",  # Inline code
        "# ",  # Headers
        "## ",
        "### ",
        "- ",  # Lists
        "1. ",  # Numbered lists
        "[",  # Links
        "> ",  # Blockquotes
    ]
    # Note: Removed single "*" and "_" as they cause false positives
    return any(indicator in text for indicator in markdown_indicators)


def _has_significant_formatting(text: str) -> bool:
    """Check if text has significant formatting that needs preservation.

    This includes newlines, multiple paragraphs, or markdown.

    Args:
        text: Text to check

    Returns:
        True if text has formatting worth preserving
    """
    # Check for multiple newlines (paragraphs)
    if "\n\n" in text:
        return True
    # Check for any newlines
    if "\n" in text:
        return True
    # Check for markdown
    return _looks_like_markdown(text)


class MessageItem(ListItem):
    """A single message item in the chat view.

    Features:
    - Clean, minimal display without visual clutter
    - Click to select/deselect for copying
    - Individual copy support per message
    """

    # Message copied event
    class MessageCopied(TextualMessage):
        """Posted when a message is copied."""

        def __init__(self, message: Message) -> None:
            self.message = message
            super().__init__()

    def __init__(self, message: Message, index: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = message
        self.index = index
        # Codex-blend theme: tag each turn with a role class so the app CSS can
        # draw the colored gutter bar (.role-user / .role-assistant / .role-tool / ...).
        _meta = getattr(message, "metadata", None) or {}
        _etype = _meta.get("event_type")
        if _etype in ("tool_call", "tool_result"):
            self.add_class("role-tool")
        elif _etype == "thinking":
            self.add_class("role-think")
        elif getattr(message, "role", None):
            self.add_class(f"role-{message.role}")
        self.selected = reactive(False)
        self.is_processing = reactive(False)  # Keep for API compat but no visual
        self.spin_task: Optional[asyncio.Task[None]] = None

    def on_mount(self) -> None:
        """Mount handler - no longer starts spinner."""
        pass

    def start_spinner(self) -> None:
        """No-op - spinners removed for cleaner UI."""
        self.is_processing = True  # type: ignore[assignment]

    def stop_spinner(self) -> None:
        """Stop processing state."""
        self.is_processing = False  # type: ignore[assignment]
        if self.spin_task:
            self.spin_task.cancel()
            self.spin_task = None

    def on_click(self, event: events.Click):
        """Click handler.

        For collapsible reasoning/tool BLOCKS (#2): toggle expanded/collapsed in
        place. For all other messages: disabled, so normal terminal text
        selection still works.
        """
        meta = getattr(self.message, "metadata", None) or {}
        if meta.get("block"):
            meta["collapsed"] = not meta.get("collapsed", False)
            # Re-measure: collapsed is one line, expanded is N — needs layout.
            self.refresh(layout=True)
            event.stop()

    def on_key(self, event) -> None:
        """Handle key events for copy shortcut."""
        if event.key == "c":
            self.run_worker(self.copy_message())
            event.stop()

    async def copy_message(self):
        """Copy this message to clipboard."""
        role = self.message.role.upper()
        if self.message.role == "assistant" and self.message.agent_name:
            role = f"{self.message.agent_name.upper()}"
            if self.message.model_name:
                role += f" ({self.message.model_name})"

        text = f"{role}: {self.message.content}"
        try:
            pyperclip.copy(text)
            self.post_message(self.MessageCopied(self.message))
        except Exception:
            pass

    def _render_block(self, meta: dict) -> Union[Text, Group]:
        """Render a collapsible reasoning/tool block (#2).

        Collapsed -> a single truncated summary line (``▸ … · click to expand``).
        Expanded  -> the full body as rich bullet points (one per streamed line).
        The app mutates ``meta`` (lines / collapsed / summary) then refreshes us,
        so we always read live from metadata rather than caching at construction.
        """
        if meta.get("collapsed", False):
            summary = meta.get("summary") or self.message.content
            try:
                return Text.from_markup(summary)
            except Exception:
                return Text(summary)

        kind = meta.get("block_kind")
        parts: List[Any] = []
        header = meta.get("expanded_header")
        if header:
            try:
                parts.append(Text.from_markup(header))
            except Exception:
                parts.append(Text(str(header)))

        if kind == "reasoning":
            # Word-for-word, ITALIC reasoning (reasoning == thinking). Number the
            # model's own reasoning lines, but don't double-number "Step N" status.
            for i, ln in enumerate(meta.get("lines") or [], 1):
                mark = "" if ln.lstrip().lower().startswith("step ") else f"{i}. "
                try:
                    parts.append(Text.from_markup(f"  [dim italic]{mark}{ln}[/]"))
                except Exception:
                    parts.append(Text(f"  {mark}{ln}", style="dim italic"))
        elif kind == "tool":
            parts.extend(self._render_tool_body(meta))
        else:
            for ln in meta.get("lines") or []:
                try:
                    parts.append(Text.from_markup(f"  • {ln}"))
                except Exception:
                    parts.append(Text(f"  • {ln}"))

        if not parts:
            return Text("")
        return Group(*parts)

    def _render_tool_body(self, meta: dict) -> List[Any]:
        """Render a tool call's body RICHLY — syntax-highlighted code for file
        reads, colored +/- diffs for edits, else the result as dim text. We
        recognise the shape ourselves; the agent never formats anything."""
        parts: List[Any] = []
        args = meta.get("tool_args_str") or ""
        if args:
            parts.append(Text.from_markup(f"  [dim]{_esc(args)}[/]"))
        result = meta.get("result_raw")
        if not result:
            return parts
        path, code = _parse_file_read(result)
        if code is not None:
            lexer = _lexer_for_path(path or meta.get("file_path"))
            try:
                parts.append(Padding(
                    Syntax(code, lexer, theme="ansi_dark", line_numbers=True,
                           word_wrap=True, background_color="default"),
                    (0, 0, 0, 2)))
            except Exception:
                parts.append(Text(code, style="dim"))
        elif _is_diffish(result):
            parts.append(Padding(_render_diff(result), (0, 0, 0, 2)))
        else:
            txt = result if len(result) <= 2000 else result[:2000] + " …"
            parts.append(Text.from_markup(f"  [dim]{_esc(txt)}[/]"))
        return parts

    def render(self) -> Union[Text, Group]:
        """Render the message with markdown support for assistant messages."""
        # Indentation is handled by the gutter + padding in app CSS now.
        prefix_text = Text("")

        # Collapsible reasoning/tool block (#2) — handled before role dispatch.
        _meta = getattr(self.message, "metadata", None) or {}
        if _meta.get("block"):
            return self._render_block(_meta)

        # For system messages, content might already be formatted - render directly
        if self.message.role == "system":
            # System messages may have pre-formatted content (like shell output)
            try:
                return Text.from_markup(self.message.content)
            except Exception:
                return Text(self.message.content)

        # Create role indicator for user/assistant
        if self.message.role == "user":
            role_text = Text()
            role_text.append("You: ", style="bold cyan")
            role_text.append(self.message.content)
            return Group(prefix_text, role_text)

        elif self.message.role == "assistant":
            # Build role header
            agent_name = self.message.agent_name or "Assistant"
            model_name = self.message.model_name or ""

            role_text = Text()
            role_text.append(agent_name, style="bold green")
            if model_name:
                role_text.append(f" ({model_name})", style="dim")
            role_text.append(":")

            # Get content and check for formatting
            content = self.message.content

            # Always use Markdown for content with significant formatting
            # This preserves newlines, paragraphs, and any markdown
            if _has_significant_formatting(content):
                # Render as markdown - this properly handles newlines and formatting
                try:
                    # ansi_dark code theme → code uses the terminal's own dark
                    # background instead of Pygments' gray box (no highlight).
                    md = ChatMarkdown(content, code_theme="ansi_dark")
                    return Group(prefix_text, role_text, md)
                except Exception:
                    # Fallback: render content as separate Text to preserve newlines
                    content_text = Text("\n" + content)
                    return Group(prefix_text, role_text, content_text)
            else:
                # Short single-line content - append to role text
                role_text.append(" ")
                role_text.append(content)
                return Group(prefix_text, role_text)
        else:
            # Other roles
            role_text = Text()
            role_text.append(f"{self.message.role}: ", style="bold")
            role_text.append(self.message.content)
            return Group(prefix_text, role_text)


class ChatView(ListView):
    """Widget for displaying conversation messages.

    Features:
    - Auto-scrolls to latest message
    - Displays messages with role indicators
    - Supports user/assistant/system message types
    - Shows agent name and model for assistant messages
    - Select All button to copy entire conversation
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages: List[Message] = []
        # Pagination for large conversations (T149)
        self.max_displayed_messages = 100  # Display last 100 messages for performance
        self.total_messages = 0  # Track total including hidden messages
        # Whether thinking/tool-call detail messages are shown (toggled via Ctrl+T).
        # Default True preserves existing behaviour; toggling hides the dim
        # streaming breadcrumbs without removing them from history.
        self.detail_visible = True

    @staticmethod
    def _is_detail(message: Message) -> bool:
        """True if a message is a toggleable thinking/tool-call breadcrumb."""
        metadata = getattr(message, "metadata", None) or {}
        return metadata.get("event_type") in DETAIL_EVENT_TYPES

    def add_message(self, message: Message):
        """Add a message to the chat view.

        Args:
            message: Message object to display
        """
        self.messages.append(message)

        # Create and append message item with index
        item = MessageItem(message, index=len(self.messages) - 1)
        self.append(item)

        # Respect the detail-visibility toggle (Ctrl+T): hide thinking/tool
        # breadcrumbs on arrival when detail is collapsed, without dropping them.
        if not self.detail_visible and self._is_detail(message):
            item.display = False
            return

        # Auto-scroll to latest message
        self.index = len(self.messages) - 1

    def set_detail_visible(self, visible: bool) -> None:
        """Show or hide all thinking/tool-call detail messages (Ctrl+T).

        Toggles ``display`` on existing detail items so they collapse/expand
        in place — this respects the ListView architecture (no nested
        Collapsible widgets, which ListView does not support).
        """
        self.detail_visible = visible
        for item in self.children:
            if isinstance(item, MessageItem) and self._is_detail(item.message):
                item.display = visible

    def remove_message(self, message: Message) -> bool:
        """Remove a specific message by IDENTITY from the view + backing list.

        Reliable even when later messages were appended after ``message`` (e.g.
        streaming thinking/tool breadcrumbs added during an await), where the
        old index/``pop()``-based removal targeted the wrong item — leaving the
        'Processing…' indicator stranded and silently dropping a real message.
        Stops any spinner on the item. Returns True if the widget was found.
        """
        removed = False
        for item in list(self.children):
            if isinstance(item, MessageItem) and item.message is message:
                try:
                    item.stop_spinner()
                except Exception:
                    pass
                item.remove()
                removed = True
                break
        try:
            self.messages.remove(message)
        except ValueError:
            pass
        return removed

    def item_for(self, message: Message) -> Optional["MessageItem"]:
        """Return the MessageItem widget backing ``message`` (by identity), or None."""
        for item in self.children:
            if isinstance(item, MessageItem) and item.message is message:
                return item
        return None

    def refresh_block(self, message: Message) -> None:
        """Re-render the collapsible block backing ``message`` after its
        metadata (lines / collapsed / summary) was mutated in place (#2)."""
        item = self.item_for(message)
        if item is not None:
            item.refresh(layout=True)
            # Keep the freshly-grown block in view while it streams.
            try:
                self.scroll_end(animate=False)
            except Exception:
                pass

    def clear_messages(self):
        """Clear all messages from the view."""
        self.clear()
        self.messages = []

    def load_messages(self, messages: List[Message]):
        """Load multiple messages at once with pagination (T149).

        For performance with large conversations, only displays the most recent
        max_displayed_messages messages. Older messages are still accessible
        in session storage.

        Args:
            messages: List of Message objects to display
        """
        self.clear_messages()
        self.total_messages = len(messages)

        # Apply pagination for large conversations (T149)
        if len(messages) > self.max_displayed_messages:
            # Show most recent messages
            display_messages = messages[-self.max_displayed_messages :]
            # Store all messages for get_all_text functionality
            self.messages = messages
            # But only display recent ones in UI. index = the message's real
            # position in the full list (offset + i), matching add_message's
            # convention. Computing it from len(self) BEFORE append was off by
            # one for every paginated item (audit F10). Also honour the
            # detail-visibility toggle so a hidden state survives a reload.
            offset = len(messages) - len(display_messages)
            for i, message in enumerate(display_messages):
                item = MessageItem(message, index=offset + i)
                self.append(item)
                if not self.detail_visible and self._is_detail(message):
                    item.display = False
        else:
            # Display all messages normally
            for message in messages:
                self.add_message(message)

    def get_all_text(self) -> str:
        """Get all conversation text for select all functionality.

        Returns:
            String with entire conversation formatted
        """
        lines = []
        for msg in self.messages:
            role = msg.role.upper()
            if msg.role == "assistant" and msg.agent_name:
                role = f"{msg.agent_name.upper()}"
                if msg.model_name:
                    role += f" ({msg.model_name})"
            lines.append(f"{role}: {msg.content}")
            lines.append("")  # Empty line between messages
        return "\n".join(lines)

    def get_selected_text(self) -> str:
        """Get text from selected messages only.

        Returns:
            String with selected messages formatted, or empty string if none selected
        """
        lines = []
        for item in self.children:
            if isinstance(item, MessageItem) and item.selected:
                msg = item.message
                role = msg.role.upper()
                if msg.role == "assistant" and msg.agent_name:
                    role = f"{msg.agent_name.upper()}"
                    if msg.model_name:
                        role += f" ({msg.model_name})"
                lines.append(f"{role}: {msg.content}")
                lines.append("")  # Empty line between messages
        return "\n".join(lines)

    def get_selected_count(self) -> int:
        """Get count of selected messages.

        Returns:
            Number of selected messages
        """
        count = 0
        for item in self.children:
            if isinstance(item, MessageItem) and item.selected:
                count += 1
        return count

    def clear_selection(self):
        """Clear all message selections."""
        for item in self.children:
            if isinstance(item, MessageItem):
                item.selected = False

    async def copy_selected_messages(self):
        """Copy all selected messages to clipboard."""
        selected_text = self.get_selected_text()
        if selected_text:
            try:
                pyperclip.copy(selected_text)
                return True
            except Exception:
                return False
        return False

    def on_message_item_message_copied(self, message: MessageItem.MessageCopied):
        """Handle individual message copy events."""
        # Could show a notification or update UI here
        pass
