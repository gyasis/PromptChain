"""Regression tests for the TUI history-collection gap.

Bug (observed 2026-07-06): TUI sessions lost most of their conversation
transcript. In one real session the router had handled 38 distinct user
queries but only 6 turns survived in ``messages.jsonl`` (84% loss).

Root cause: the TUI turn loop never triggered persistence. The built-in
``Session.check_autosave`` (save every N messages / T seconds) and the
app's ``on_exit`` cleanup were both effectively dead code — ``on_exit`` is
not a Textual lifecycle event so Textual never called it, and no turn handler
called ``check_autosave``. The transcript was flushed to disk only when the
user literally typed ``/exit``; Ctrl+C / Ctrl+D / terminal-close lost every
turn since the last (rare) save.

Fix: ``handle_user_message`` / ``_handle_workflow_message`` now call
``self.session.check_autosave(self.session_manager)`` after each turn, and a
real ``on_unmount`` hook delegates to ``on_exit`` so every exit path flushes.

These tests pin the persistence-layer behaviour (turns survive without a
clean ``/exit``) and guard that the TUI turn handlers keep the wiring.
"""

import inspect
import json
import shutil
import tempfile
from pathlib import Path

import pytest


class TestHistoryPersistenceGap:
    @pytest.fixture
    def temp_sessions_dir(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        from promptchain.cli.session_manager import SessionManager

        return SessionManager(sessions_dir=temp_sessions_dir)

    def _read_persisted_messages(self, session_manager, session_id):
        path = session_manager.sessions_dir / session_id / "messages.jsonl"
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    def _run_turns(self, session, session_manager, n_turns):
        """Drive the TUI turn loop: per turn add user+assistant, then autosave."""
        for i in range(n_turns):
            session.add_message(role="user", content=f"user query {i}")
            session.add_message(
                role="assistant",
                content=f"assistant reply {i}",
                metadata={"model_name": "openai/gpt-4.1-mini"},
            )
            # The line the TUI turn handlers were missing (the fix).
            session.check_autosave(session_manager)

    def test_periodic_autosave_persists_bulk_without_clean_exit(self, session_manager):
        """Reproduces the 84%-loss scenario: many turns, NO /exit.

        Before the fix nothing called ``check_autosave`` so ``messages.jsonl``
        stayed empty/stale until a rare ``/exit``. With per-turn autosave the
        bulk of the transcript is on disk mid-session — here every turn past the
        first autosave interval is already durable.
        """
        session = session_manager.create_session("gap-repro", Path.cwd())
        n_turns = 8
        self._run_turns(session, session_manager, n_turns)

        persisted = self._read_persisted_messages(session_manager, session.id)
        user_turns = [m for m in persisted if m["role"] == "user"]

        # Interval autosave (every 5 msgs) leaves only a small tail unflushed;
        # the bulk is durable mid-session (was 0 before the fix).
        assert len(user_turns) >= n_turns - 2, (
            f"expected the bulk of {n_turns} turns persisted mid-session, got "
            f"{len(user_turns)} (persistence gap regressed)"
        )

    def test_full_history_persists_after_shutdown_flush(self, session_manager):
        """End-to-end: per-turn autosave + shutdown flush == complete history.

        Models the whole fix — the turn loop calls ``check_autosave``, and
        teardown (app ``on_unmount`` → ``on_exit``, mirrored here by the
        model's ``Session.on_exit``) does the final flush. Result: zero loss,
        even for the tail turns the interval autosave hadn't reached yet.
        """
        session = session_manager.create_session("gap-repro-full", Path.cwd())
        n_turns = 8
        self._run_turns(session, session_manager, n_turns)

        # Final flush on shutdown (what on_unmount now guarantees).
        session.on_exit(session_manager)

        persisted = self._read_persisted_messages(session_manager, session.id)
        user_turns = [m for m in persisted if m["role"] == "user"]

        assert len(user_turns) == n_turns, (
            f"expected all {n_turns} user turns persisted after shutdown flush, "
            f"got {len(user_turns)} (persistence gap regressed)"
        )
        assert [m["content"] for m in user_turns] == [
            f"user query {i}" for i in range(n_turns)
        ]

    def test_tui_turn_handlers_invoke_check_autosave(self):
        """Guard the wiring: both TUI turn handlers must trigger persistence.

        A source-level check so that if someone removes the ``check_autosave``
        call from a turn handler, this test fails instead of silently
        re-introducing the history gap. (Full async-app execution is covered by
        the integration suite; this stays fast and harness-independent.)
        """
        from promptchain.cli.tui.app import PromptChainApp

        for method_name in ("handle_user_message", "_handle_workflow_message"):
            src = inspect.getsource(getattr(PromptChainApp, method_name))
            assert "check_autosave" in src, (
                f"{method_name} no longer calls check_autosave — the TUI history "
                f"persistence gap has regressed"
            )

    def test_on_unmount_is_wired_to_shutdown_save(self):
        """``on_unmount`` (the real Textual hook) must exist and reach on_exit.

        ``on_exit`` alone is dead code (not a Textual event); on_unmount is what
        Textual actually fires on teardown, so it must drive the shutdown save.
        """
        from promptchain.cli.tui.app import PromptChainApp

        assert hasattr(PromptChainApp, "on_unmount"), "on_unmount hook missing"
        src = inspect.getsource(PromptChainApp.on_unmount)
        assert "on_exit" in src, "on_unmount does not delegate to on_exit"
