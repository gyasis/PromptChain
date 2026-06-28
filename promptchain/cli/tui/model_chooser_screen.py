"""Interactive model chooser modal.

A ``ModalScreen`` that lists every selectable model from the live catalog
(Mac-Studio-local Ollama, Ollama Cloud, curated OpenAI/Gemini), lets the user
filter-as-they-type, and returns the chosen :class:`ModelEntry`.

Compatible with the app's existing routing (``_model_call_params`` resolves a
model_name against ``self._model_catalog`` to add ``api_base``/``api_key``): on
pick the screen hands its **already-fetched full catalog** (incl. local) to
``self.app._model_catalog`` so the chosen model — local OR cloud — routes
correctly with no extra network round-trip.

Usage from the app (push + callback, like ApprovalScreen)::

    self.push_screen(ModelChooserScreen(current=agent.model_name), self._on_model_chosen)

The catalog is fetched OFF the event loop (``asyncio.to_thread``) so opening the
chooser never blocks the TUI while the network sources respond.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from ..model_catalog import ModelEntry, build_catalog

# Display order + tag per source.
_SOURCE_ORDER = {"ollama-local": 0, "ollama-cloud": 1, "openai": 2, "gemini": 3}
_SOURCE_TAG = {
    "ollama-local": ("local", "green"),
    "ollama-cloud": ("cloud", "cyan"),
    "openai": ("openai", "magenta"),
    "gemini": ("gemini", "yellow"),
}


class ModelChooserScreen(ModalScreen[Optional[ModelEntry]]):
    """Filterable picker over the live model catalog. Dismisses with a ModelEntry."""

    DEFAULT_CSS = """
    ModelChooserScreen {
        align: center middle;
        background: $background 70%;
    }
    ModelChooserScreen #mc-box {
        width: 92;
        max-width: 95%;
        height: 80%;
        max-height: 40;
        padding: 1 2;
        background: $surface;
        border: round $accent;
    }
    ModelChooserScreen #mc-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    ModelChooserScreen #mc-filter {
        margin-bottom: 1;
    }
    ModelChooserScreen #mc-results {
        height: 1fr;
        border: round $panel;
    }
    ModelChooserScreen #mc-hint {
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("down", "cursor_down", "Down"),
        ("up", "cursor_up", "Up"),
    ]

    def __init__(self, current: Optional[str] = None) -> None:
        super().__init__()
        self._current = current
        self._entries: List[ModelEntry] = []
        self._visible: List[ModelEntry] = []
        self._loaded = False

    def compose(self) -> ComposeResult:
        with Vertical(id="mc-box"):
            cur = f"  [dim](current: {self._current})[/dim]" if self._current else ""
            yield Static(f"Choose a model{cur}", id="mc-title")
            yield Input(
                placeholder="type to filter…  (↑↓ move · Enter pick · Esc cancel)",
                id="mc-filter",
            )
            ol: OptionList = OptionList(id="mc-results")
            ol.add_option(Option(Text("Loading models…", style="dim italic")))
            yield ol
            yield Static("", id="mc-hint")

    def on_mount(self) -> None:
        self.query_one("#mc-filter", Input).focus()
        self._load_catalog()

    # --- catalog load (off the event loop) ------------------------------- #
    @work(exclusive=True)
    async def _load_catalog(self) -> None:
        try:
            entries = await asyncio.to_thread(build_catalog)
        except Exception as exc:  # never let discovery break the modal
            entries = []
            self.query_one("#mc-hint", Static).update(f"[red]catalog error: {exc}[/red]")
        entries.sort(key=lambda e: (_SOURCE_ORDER.get(e.source, 9), e.model_name.lower()))
        self._entries = entries
        self._loaded = True
        self._apply_filter(self.query_one("#mc-filter", Input).value)

    # --- filtering ------------------------------------------------------- #
    @on(Input.Changed, "#mc-filter")
    def _on_filter_changed(self, event: Input.Changed) -> None:
        self._apply_filter(event.value)

    def _apply_filter(self, query: str) -> None:
        terms = query.lower().split()
        if terms:
            self._visible = [
                e
                for e in self._entries
                if all(
                    t in f"{e.label} {e.model_name} {e.source}".lower() for t in terms
                )
            ]
        else:
            self._visible = list(self._entries)
        self._repopulate()

    def _repopulate(self) -> None:
        ol = self.query_one("#mc-results", OptionList)
        ol.clear_options()
        if not self._loaded:
            ol.add_option(Option(Text("Loading models…", style="dim italic")))
            return
        if not self._visible:
            ol.add_option(
                Option(
                    Text(
                        "No models found — check OLLAMA_HOST / OLLAMA_API_KEY / OPENAI_API_KEY.",
                        style="dim italic",
                    )
                )
            )
            self.query_one("#mc-hint", Static).update(
                f"{len(self._entries)} models in catalog"
            )
            return
        for e in self._visible:
            ol.add_option(Option(self._format(e)))
        ol.highlighted = 0
        self.query_one("#mc-hint", Static).update(
            f"{len(self._visible)}/{len(self._entries)} models"
        )

    def _format(self, e: ModelEntry) -> Text:
        tag, color = _SOURCE_TAG.get(e.source, (e.source, "white"))
        line = Text()
        line.append(f"{tag:<6} ", style=color)
        line.append(e.model_name, style="bold")
        if e.size_gb:
            line.append(f"  {e.size_gb}GB", style="dim")
        if e.reasoning.get("supports"):
            line.append(f"  think:{e.reasoning.get('extract')}", style="italic dim")
        if e.model_name == self._current:
            line.append("  ✓ current", style="green")
        return line

    # --- selection ------------------------------------------------------- #
    @on(OptionList.OptionSelected, "#mc-results")
    def _on_selected(self, event: OptionList.OptionSelected) -> None:
        self._pick(event.option_index)

    @on(Input.Submitted, "#mc-filter")
    def _on_submit(self, event: Input.Submitted) -> None:
        ol = self.query_one("#mc-results", OptionList)
        idx = ol.highlighted if ol.highlighted is not None else 0
        self._pick(idx)

    def _pick(self, index: Optional[int]) -> None:
        if not self._loaded or not self._visible or index is None:
            return
        if 0 <= index < len(self._visible):
            # Hand the freshly-fetched full catalog (incl. local) to the app so
            # _model_call_params() can route the pick without another probe.
            try:
                self.app._model_catalog = self._entries
            except Exception:
                pass
            self.dismiss(self._visible[index])

    # --- navigation while the filter Input holds focus ------------------- #
    def action_cursor_down(self) -> None:
        self.query_one("#mc-results", OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#mc-results", OptionList).action_cursor_up()

    def action_cancel(self) -> None:
        self.dismiss(None)
