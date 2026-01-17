"""InputWidget for user message input with autocomplete support."""

from typing import Optional

from textual.message import Message as TextualMessage
from textual.widgets import TextArea

from ..command_handler import get_command_suggestions


class InputWidget(TextArea):
    """Widget for user input with Enter to submit.

    Features:
    - Multi-line text input support
    - Enter key submits message
    - Shift+Enter adds newline
    - Auto-clears after submit
    - Emits MessageSubmitted event
    - Command history navigation with Up/Down (T144)
    - Slash command autocomplete popup (T145 enhanced)
    """

    class MessageSubmitted(TextualMessage):
        """Event emitted when user submits a message."""

        def __init__(self, content: str):
            super().__init__()
            self.content = content

    class AutocompleteRequest(TextualMessage):
        """Request to update autocomplete popup with suggestions."""

        def __init__(self, suggestions: list):
            super().__init__()
            self.suggestions = suggestions

    class AutocompleteNavigate(TextualMessage):
        """Request to navigate autocomplete popup."""

        def __init__(self, direction: str):
            super().__init__()
            self.direction = direction  # "up" or "down"

    class AutocompleteSelect(TextualMessage):
        """Request to select current autocomplete item."""
        pass

    class AutocompleteDismiss(TextualMessage):
        """Request to dismiss autocomplete popup."""
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_line_numbers = False

        # Command history for Up/Down navigation (T144)
        self.command_history = []  # List of previous commands
        self.history_index = -1  # Current position in history (-1 = not navigating)
        self.current_draft = ""  # Save current input when navigating history

        # Autocomplete state (T145 enhanced)
        self.autocomplete_active = False  # Is popup currently shown
        self._last_autocomplete_text = ""  # Track text changes

    def action_submit(self):
        """Submit the current message content."""
        content = self.text.strip()

        if content:
            # Add to command history (T144)
            self.command_history.append(content)
            # Keep history limited to last 100 commands
            if len(self.command_history) > 100:
                self.command_history.pop(0)
            # Reset history navigation
            self.history_index = -1
            self.current_draft = ""

            # Dismiss autocomplete
            self.autocomplete_active = False
            self.post_message(self.AutocompleteDismiss())

            # Emit message submitted event
            self.post_message(self.MessageSubmitted(content))

            # Clear the input
            self.clear()

    def navigate_history_up(self):
        """Navigate to previous command in history (T144)."""
        if not self.command_history:
            return

        # Save current input as draft when first navigating
        if self.history_index == -1:
            self.current_draft = self.text

        # Navigate up in history
        if self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            # Get command from history (reverse order - most recent first)
            command = self.command_history[-(self.history_index + 1)]
            self.text = command

    def navigate_history_down(self):
        """Navigate to next command in history (T144)."""
        if self.history_index == -1:
            return  # Not navigating history

        # Navigate down in history
        if self.history_index > 0:
            self.history_index -= 1
            # Get command from history
            command = self.command_history[-(self.history_index + 1)]
            self.text = command
        else:
            # Back to draft or empty
            self.history_index = -1
            self.text = self.current_draft
            self.current_draft = ""

    def update_autocomplete(self):
        """Update autocomplete suggestions based on current input.

        Note: Any errors here are caught silently to ensure input remains responsive.
        """
        try:
            current_text = self.text.strip()

            # Only show autocomplete if input starts with /
            if not current_text.startswith("/"):
                if self.autocomplete_active:
                    self.autocomplete_active = False
                    self.post_message(self.AutocompleteDismiss())
                return

            # Get suggestions from command registry
            suggestions = get_command_suggestions(current_text)

            if suggestions:
                self.autocomplete_active = True
                self.post_message(self.AutocompleteRequest(suggestions))
            else:
                if self.autocomplete_active:
                    self.autocomplete_active = False
                    self.post_message(self.AutocompleteDismiss())
        except Exception:
            # Silently ignore autocomplete errors to keep input responsive
            self.autocomplete_active = False

    def set_text_from_autocomplete(self, command: str):
        """Set text from autocomplete selection.

        Args:
            command: The command string to set
        """
        self.text = command
        self.autocomplete_active = False
        # Move cursor to end
        self.move_cursor((0, len(command)))

    async def on_key(self, event):
        """Handle key events.

        Enter: Submit message (or select from autocomplete)
        Shift+Enter: Insert newline
        Up Arrow: Navigate popup or history (T144)
        Down Arrow: Navigate popup or history (T144)
        Tab: Select from autocomplete (T145)
        Escape: Dismiss autocomplete

        Note: Regular character keys are explicitly passed through to TextArea.
        """
        # Escape to dismiss autocomplete
        if event.key == "escape":
            if self.autocomplete_active:
                event.prevent_default()
                event.stop()
                self.autocomplete_active = False
                self.post_message(self.AutocompleteDismiss())
                return

        # Tab key to select from autocomplete
        if event.key == "tab":
            event.prevent_default()
            event.stop()
            if self.autocomplete_active:
                self.post_message(self.AutocompleteSelect())
            return

        # Up/Down arrow navigation
        if event.key == "up":
            event.prevent_default()
            event.stop()
            if self.autocomplete_active:
                # Navigate autocomplete popup
                self.post_message(self.AutocompleteNavigate("up"))
            else:
                # Navigate command history
                self.navigate_history_up()
            return

        if event.key == "down":
            event.prevent_default()
            event.stop()
            if self.autocomplete_active:
                # Navigate autocomplete popup
                self.post_message(self.AutocompleteNavigate("down"))
            else:
                # Navigate command history
                self.navigate_history_down()
            return

        # Enter key handling - ALWAYS submit the message
        # (Tab is used for autocomplete selection, Enter is for execution)
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            # Dismiss autocomplete if active
            if self.autocomplete_active:
                self.autocomplete_active = False
                self.post_message(self.AutocompleteDismiss())
            # Always submit the message
            self.action_submit()
            return

        # For all other keys (regular characters, backspace, delete, etc.),
        # let the parent TextArea handle them normally by NOT stopping the event.
        # This ensures input remains responsive even with autocomplete active.

    def on_text_area_changed(self, event) -> None:
        """Handle text changes to update autocomplete.

        Note: Errors are caught to prevent autocomplete from breaking input.
        """
        try:
            # Reset history navigation when text changes
            if self.history_index != -1:
                self.history_index = -1
                self.current_draft = ""

            # Update autocomplete suggestions
            self.update_autocomplete()
        except Exception:
            # Never let autocomplete errors break text input
            pass
