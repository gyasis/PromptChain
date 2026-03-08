"""ChatView widget for displaying conversation messages."""

import asyncio
from typing import Any, List, Optional, Union

import pyperclip
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.markdown import Markdown
from rich.text import Text
from textual import events
from textual.message import Message as TextualMessage
from textual.reactive import reactive
from textual.widgets import ListItem, ListView

from ..models import Message


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
        """Click handler - disabled for normal terminal selection."""
        # Selection disabled - allow normal terminal text selection
        pass

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

    def render(self) -> Union[Text, Group]:
        """Render the message with markdown support for assistant messages."""
        # Simple consistent prefix (no selection highlighting)
        prefix_text = Text("  ")

        # For system messages, content might already be formatted - render directly
        if self.message.role == "system":
            # System messages may have pre-formatted content (like shell output)
            try:
                return Text.from_markup("  " + self.message.content)
            except Exception:
                return Text("  " + self.message.content)

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
                    md = Markdown(content)
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

    def add_message(self, message: Message):
        """Add a message to the chat view.

        Args:
            message: Message object to display
        """
        self.messages.append(message)

        # Create and append message item with index
        item = MessageItem(message, index=len(self.messages) - 1)
        self.append(item)

        # Auto-scroll to latest message
        self.index = len(self.messages) - 1

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
            # But only display recent ones in UI
            for message in display_messages:
                item = MessageItem(message, index=len(self) - 1)
                self.append(item)
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
