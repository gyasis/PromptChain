"""Autocomplete popup widget for slash command suggestions."""

from typing import List, Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message as TextualMessage
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class SuggestionItem(Static):
    """A single suggestion item in the autocomplete popup."""

    DEFAULT_CSS = """
    SuggestionItem {
        height: 1;
        padding: 0 1;
        background: $surface;
    }
    SuggestionItem.highlighted {
        background: $accent;
        color: $text;
    }
    """

    def __init__(self, command: str, description: str, **kwargs):
        super().__init__(**kwargs)
        self.command = command
        self.description = description

    def compose(self) -> ComposeResult:
        yield Static(f"{self.command}  [dim]{self.description}[/dim]")


class AutocompletePopup(Widget):
    """Popup widget showing command suggestions with keyboard navigation.

    Features:
    - Shows matching commands when user types /
    - Up/Down arrow navigation
    - Enter/Tab to select
    - Escape to dismiss
    - Auto-updates as user types

    Note: This widget is non-focusable to keep focus on InputWidget.
    """

    # Prevent this widget from stealing focus
    can_focus = False

    DEFAULT_CSS = """
    AutocompletePopup {
        layer: autocomplete;
        width: auto;
        max-width: 60;
        height: auto;
        max-height: 10;
        background: $surface;
        border: solid $primary;
        padding: 0;
        display: none;
    }
    AutocompletePopup.visible {
        display: block;
    }
    AutocompletePopup > Vertical {
        height: auto;
        max-height: 10;
    }
    """

    # Reactive properties
    suggestions: reactive[List[dict]] = reactive(list, init=False)
    selected_index: reactive[int] = reactive(0)
    visible: reactive[bool] = reactive(False)

    class CommandSelected(TextualMessage):
        """Emitted when a command is selected."""

        def __init__(self, command: str):
            super().__init__()
            self.command = command

    class PopupDismissed(TextualMessage):
        """Emitted when popup is dismissed."""
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._suggestions: List[dict] = []
        self._selected_index = 0

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("", id="suggestion-list")

    def update_suggestions(self, suggestions: List[dict]) -> None:
        """Update the list of suggestions.

        Args:
            suggestions: List of dicts with 'command' and 'description' keys
        """
        self._suggestions = suggestions[:10]  # Limit to 10 suggestions
        self._selected_index = 0

        if not self._suggestions:
            self.hide()
            return

        # Build suggestion display
        lines = []
        for i, s in enumerate(self._suggestions):
            prefix = ">" if i == self._selected_index else " "
            cmd = s["command"]
            desc = s["description"]
            # Truncate description if too long
            if len(desc) > 35:
                desc = desc[:32] + "..."
            lines.append(f"{prefix} {cmd:<20} {desc}")

        # Update the static content
        suggestion_list = self.query_one("#suggestion-list", Static)
        suggestion_list.update("\n".join(lines))

        self.show()

    def show(self) -> None:
        """Show the popup."""
        self.add_class("visible")
        self.visible = True

    def hide(self) -> None:
        """Hide the popup."""
        self.remove_class("visible")
        self.visible = False
        self._suggestions = []
        self._selected_index = 0

    def navigate_up(self) -> None:
        """Move selection up."""
        if not self._suggestions:
            return
        self._selected_index = (self._selected_index - 1) % len(self._suggestions)
        self._refresh_display()

    def navigate_down(self) -> None:
        """Move selection down."""
        if not self._suggestions:
            return
        self._selected_index = (self._selected_index + 1) % len(self._suggestions)
        self._refresh_display()

    def select_current(self) -> Optional[str]:
        """Select the currently highlighted command.

        Returns:
            The selected command string, or None if no selection
        """
        if self._suggestions and 0 <= self._selected_index < len(self._suggestions):
            command = self._suggestions[self._selected_index]["command"]
            self.post_message(self.CommandSelected(command))
            self.hide()
            return command
        return None

    def dismiss(self) -> None:
        """Dismiss the popup without selecting."""
        self.hide()
        self.post_message(self.PopupDismissed())

    def _refresh_display(self) -> None:
        """Refresh the display with current selection highlighted."""
        if not self._suggestions:
            return

        lines = []
        for i, s in enumerate(self._suggestions):
            prefix = ">" if i == self._selected_index else " "
            cmd = s["command"]
            desc = s["description"]
            if len(desc) > 35:
                desc = desc[:32] + "..."
            lines.append(f"{prefix} {cmd:<20} {desc}")

        suggestion_list = self.query_one("#suggestion-list", Static)
        suggestion_list.update("\n".join(lines))

    @property
    def is_visible(self) -> bool:
        """Check if popup is currently visible."""
        return self.visible and bool(self._suggestions)

    @property
    def current_command(self) -> Optional[str]:
        """Get the currently highlighted command."""
        if self._suggestions and 0 <= self._selected_index < len(self._suggestions):
            return self._suggestions[self._selected_index]["command"]
        return None
