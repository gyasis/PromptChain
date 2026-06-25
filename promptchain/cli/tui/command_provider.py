"""Command-palette Provider exposing app-level actions (Ctrl+P).

This COMPLEMENTS — it does not replace — the inline ``/command`` parsing in
``InputWidget`` + ``AutocompletePopup``. The pattern mirrors production Textual
chat TUIs (e.g. oterm), which keep inline ``/cmd`` in the input box and use the
command palette only for app-level actions (new session, toggle panels, export).

Registered via ``App.COMMANDS = App.COMMANDS | {PromptChainCommands}`` in
``app.py``, so the built-in Textual system commands are preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Tuple, cast

from textual.command import DiscoveryHit, Hit, Hits, Provider

if TYPE_CHECKING:
    from .app import PromptChainApp


class PromptChainCommands(Provider):
    """Fuzzy command palette for PromptChain app-level actions."""

    def _commands(self) -> List[Tuple[str, Callable[[], None], str]]:
        """Build the (display, runner, help) tuples for the palette.

        Built lazily per search so the runners close over the live app.
        """
        app = cast("PromptChainApp", self.app)
        return [
            ("Help", lambda: app.run_palette_command("/help"), "Show help overview"),
            (
                "Help: commands",
                lambda: app.run_palette_command("/help commands"),
                "Slash command reference",
            ),
            (
                "Session info",
                lambda: app.run_palette_command("/session"),
                "Show current session details",
            ),
            (
                "Active agent",
                lambda: app.run_palette_command("/agent"),
                "Show the active agent",
            ),
            (
                "Clear chat history",
                lambda: app.run_palette_command("/clear"),
                "Clear the conversation view",
            ),
            (
                "Clear Python cache",
                lambda: app.run_palette_command("/cache clear"),
                "Remove __pycache__ and .pyc files",
            ),
            (
                "Toggle observe panel",
                app.action_toggle_observe,
                "Show/hide execution observability",
            ),
            (
                "Toggle activity logs",
                app.action_toggle_log_view,
                "Show/hide the activity log viewer",
            ),
            (
                "Toggle thinking / tool detail",
                app.action_toggle_thinking,
                "Show/hide inline thinking + tool-call messages",
            ),
            (
                "Exit",
                lambda: app.run_palette_command("/exit"),
                "Save session and quit",
            ),
        ]

    async def discover(self) -> Hits:
        """Yield default commands shown when the palette opens empty."""
        for display, runner, help_text in self._commands():
            yield DiscoveryHit(display, runner, help=help_text)

    async def search(self, query: str) -> Hits:
        """Yield fuzzy-matched commands for the user's query."""
        matcher = self.matcher(query)
        for display, runner, help_text in self._commands():
            score = matcher.match(display)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(display),
                    runner,
                    help=help_text,
                )
