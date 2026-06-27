"""Approval modal for gating risky shell/tool actions before execution.

A ``ModalScreen[bool]`` that shows the proposed command (with any matched
danger reason) and blocks the caller until the user approves or rejects.

Use from a worker context::

    approved = await self.push_screen_wait(ApprovalScreen(command, reason))

``push_screen_wait`` must be awaited from a worker (a ``@work`` method) or an
already-async message handler that Textual runs as a task; see app.py wiring.
"""

from __future__ import annotations

from typing import Optional

from rich.syntax import Syntax
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ApprovalScreen(ModalScreen[bool]):
    """Yes/No approval dialog for a single risky action."""

    DEFAULT_CSS = """
    ApprovalScreen {
        align: center middle;
        background: #0b0e14;
    }
    ApprovalScreen #approval-box {
        width: 80;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        background: #0b0e14;
        border: round $warning;
    }
    ApprovalScreen #approval-title {
        text-style: bold;
        color: $warning;
        padding-bottom: 1;
    }
    ApprovalScreen #approval-reason {
        color: $text-muted;
        padding-bottom: 1;
    }
    ApprovalScreen #approval-buttons {
        height: auto;
        align: right middle;
        padding-top: 1;
    }
    ApprovalScreen Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        ("y", "approve", "Approve"),
        ("n", "reject", "Reject"),
        ("escape", "reject", "Reject"),
    ]

    def __init__(
        self,
        command: str,
        reason: Optional[str] = None,
        language: str = "bash",
    ) -> None:
        super().__init__()
        self._command = command
        self._reason = reason
        self._language = language

    def compose(self) -> ComposeResult:
        title = "Approve this action?"
        with Vertical(id="approval-box"):
            yield Static(title, id="approval-title")
            if self._reason:
                yield Static(f"⚠ {self._reason}", id="approval-reason")
            yield Static(
                Syntax(
                    self._command,
                    self._language,
                    theme="ansi_dark",
                    word_wrap=True,
                ),
                id="approval-command",
            )
            with Horizontal(id="approval-buttons"):
                yield Button("Reject [n]", id="reject", variant="error")
                yield Button("Approve [y]", id="approve", variant="success")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_reject(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#approve")
    def _on_approve(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#reject")
    def _on_reject(self) -> None:
        self.dismiss(False)
