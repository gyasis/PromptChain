"""Weak-model longevity for F3 / US3 — Document-&-Clear (FR-011..FR-014).

Two pieces:

- :func:`build_turn_context` — the ``<turn-context>`` block that re-injects the
  goal verbatim each turn so a weak model never loses the objective (FR-011).
- :class:`DocumentAndClear` — pure decision helpers (compress / stall /
  escalate) plus the one I/O method ``document_and_clear`` that writes a
  progress doc and returns a reset, doc-seeded resumed history. When the working
  dir is not writable it falls back lossily and never raises (FR-012/013/014).

The pure decision functions are deterministic + side-effect-free; only
``document_and_clear`` touches the filesystem. See
``specs/014-dynamic-prompt-layer/contracts/generator-api.md`` ("Longevity (US3)")
and ``contracts/prompt-layout.md`` for the rendered shapes.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional


def build_turn_context(goal: str, *, turn: int, extra: Optional[str] = None) -> str:
    """Return the ``<turn-context>`` block re-injecting the goal for this turn.

    Per ``contracts/prompt-layout.md`` (FR-011): a ``<turn-context turn="N">``
    block that re-injects ``GOAL: <goal>`` verbatim and, when supplied, appends
    the ``extra`` line (e.g. current step / last result summary).
    """
    lines = [f'<turn-context turn="{turn}">', f"GOAL: {goal}"]
    if extra:
        lines.append(extra)
    lines.append("</turn-context>")
    return "\n".join(lines)


class DocumentAndClear:
    """Document-&-Clear longevity controller (FR-012/013/014)."""

    def __init__(
        self,
        *,
        compress_at: float = 0.60,
        min_turns: int = 10,
        jacket=None,
    ) -> None:
        self.compress_at = compress_at
        self.min_turns = min_turns
        self.jacket = jacket

    # -- pure decision functions (deterministic, side-effect-free) ----------- #
    def should_compress(self, context_usage_fraction: float) -> bool:
        """True once context usage reaches the compress threshold (inclusive)."""
        return context_usage_fraction >= self.compress_at

    def is_stalled(self, progress_signals: list) -> bool:
        """True when there is no measurable progress across the recent window.

        Empty signals → stalled. Otherwise the last ``min(3, len)`` signals being
        all identical means no movement (e.g. ``[5, 5, 5]`` → True, ``[1, 2, 3]``
        → False).
        """
        if not progress_signals:
            return True
        if len(progress_signals) < 2:
            return False
        window = progress_signals[-min(3, len(progress_signals)):]
        return all(s == window[0] for s in window)

    def should_escalate(self, *, stalled: bool) -> bool:
        """Escalate only on stall AND when the jacket permits it (FR-013)."""
        if not stalled:
            return False
        return self.jacket is not None and bool(getattr(self.jacket, "escalate", False))

    # -- the only I/O method ------------------------------------------------- #
    def document_and_clear(self, working_dir: str, state: dict) -> List[Dict[str, str]]:
        """Write a progress doc, then return a reset, doc-seeded resumed history.

        Falls back to a minimal lossy resumed history (no raise) when the
        working dir is not writable (FR-014).
        """
        goal = state.get("goal", "")
        doc = self._render_progress_doc(state)

        try:
            os.makedirs(working_dir, exist_ok=True)
            path = os.path.join(working_dir, "PROGRESS.md")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(doc)
        except OSError:
            # Lossy fallback — never raise, still return a goal-anchored history.
            return [{"role": "system", "content": f"[lossy fallback] GOAL: {goal}"}]

        return [
            {"role": "system", "content": doc},
            {"role": "user", "content": f"Resume toward GOAL: {goal}"},
        ]

    # -- helpers ------------------------------------------------------------- #
    @staticmethod
    def _render_progress_doc(state: dict) -> str:
        goal = state.get("goal", "")

        def _section(title: str, key: str) -> str:
            items = state.get(key) or []
            body = "\n".join(f"- {item}" for item in items)
            return f"## {title}\n{body}".rstrip()

        return "\n".join(
            [
                f"# Progress — {goal}",
                _section("Plan", "plan"),
                _section("Decisions", "decisions"),
                _section("Done / State", "progress"),
                "## Resume",
                "Continue toward GOAL from the state above.",
            ]
        )
