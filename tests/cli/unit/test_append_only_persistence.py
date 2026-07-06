"""Tests for append-only messages.jsonl persistence + slash-command turn recording.

Follow-ups to the TUI history-persistence fix:

1. save_session now APPENDS only the messages added since the last save
   (tracked via session._persisted_count) instead of rewriting the whole
   transcript each turn — O(new) instead of O(n) per save, killing the O(n^2)
   cost on long sessions. Full rewrite is a fallback only when the file is
   missing or the in-memory list shrank (clear/compaction).

2. Slash-command turns previously appended to session.messages directly,
   bypassing add_message's autosave bookkeeping. app._record_command_turn now
   bumps the counter and runs check_autosave so those turns are counted + saved.
"""

from __future__ import annotations

import inspect
import shutil
import tempfile
from pathlib import Path

import pytest


class TestAppendOnlyPersistence:
    @pytest.fixture
    def temp_sessions_dir(self):
        d = Path(tempfile.mkdtemp())
        yield d
        shutil.rmtree(d)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        from promptchain.cli.session_manager import SessionManager

        return SessionManager(sessions_dir=temp_sessions_dir)

    def _lines(self, session_manager, session_id):
        p = session_manager.sessions_dir / session_id / "messages.jsonl"
        return [l for l in p.read_text().splitlines() if l.strip()]

    def test_no_duplication_across_saves(self, session_manager):
        """Two saves with new messages between them → each message once."""
        s = session_manager.create_session("append1", Path.cwd())
        s.add_message(role="user", content="one")
        s.add_message(role="assistant", content="two")
        session_manager.save_session(s)
        s.add_message(role="user", content="three")
        s.add_message(role="assistant", content="four")
        session_manager.save_session(s)

        lines = self._lines(session_manager, s.id)
        assert len(lines) == 4
        contents = [l for l in lines]
        # each unique content appears exactly once
        for token in ("one", "two", "three", "four"):
            assert sum(token in c for c in contents) == 1

    def test_second_save_appends_not_rewrites(self, session_manager):
        """The bytes written by the first save are preserved verbatim as a prefix."""
        s = session_manager.create_session("append2", Path.cwd())
        s.add_message(role="user", content="first")
        session_manager.save_session(s)
        path = session_manager.sessions_dir / s.id / "messages.jsonl"
        after_first = path.read_text()

        s.add_message(role="assistant", content="second")
        session_manager.save_session(s)
        after_second = path.read_text()

        assert after_second.startswith(after_first)  # true append, no rewrite
        assert "second" in after_second[len(after_first):]

    def test_noop_save_writes_nothing(self, session_manager):
        """Saving with no new messages must not append anything."""
        s = session_manager.create_session("append3", Path.cwd())
        s.add_message(role="user", content="only")
        session_manager.save_session(s)
        path = session_manager.sessions_dir / s.id / "messages.jsonl"
        size1 = path.stat().st_size
        session_manager.save_session(s)  # no new messages
        assert path.stat().st_size == size1
        assert len(self._lines(session_manager, s.id)) == 1

    def test_reload_then_save_no_duplication(self, session_manager):
        """Loading a session sets _persisted_count so later saves don't re-append."""
        s = session_manager.create_session("append4", Path.cwd())
        s.add_message(role="user", content="alpha")
        s.add_message(role="assistant", content="beta")
        session_manager.save_session(s)

        reloaded = session_manager.load_session("append4")
        assert getattr(reloaded, "_persisted_count", None) == 2
        reloaded.add_message(role="user", content="gamma")
        session_manager.save_session(reloaded)

        lines = self._lines(session_manager, s.id)
        assert len(lines) == 3  # alpha, beta, gamma — no duplicates
        assert sum("alpha" in l for l in lines) == 1

    def test_full_rewrite_fallback_on_shrink(self, session_manager):
        """If the in-memory list is shorter than persisted, do a full rewrite."""
        s = session_manager.create_session("append5", Path.cwd())
        for i in range(4):
            s.add_message(role="user", content=f"m{i}")
        session_manager.save_session(s)
        assert len(self._lines(session_manager, s.id)) == 4

        # Simulate a clear/compaction: list shrinks below persisted count.
        s.messages = s.messages[:1]
        session_manager.save_session(s)

        lines = self._lines(session_manager, s.id)
        assert len(lines) == 1  # file reflects the shrunk list, not stale 4
        assert "m0" in lines[0]


class TestCommandTurnRecording:
    def test_no_direct_append_sites_remain(self):
        """All slash-command turns must route through _record_command_turn."""
        from promptchain.cli.tui.app import PromptChainApp

        src = inspect.getsource(PromptChainApp)
        # The only remaining direct appends are the two inside the helper itself.
        assert src.count("self.session.messages.append") == 2

    def test_helper_counts_and_saves(self):
        """_record_command_turn must bump the counter AND trigger check_autosave."""
        from promptchain.cli.tui.app import PromptChainApp

        src = inspect.getsource(PromptChainApp._record_command_turn)
        assert "messages_since_save" in src
        assert "check_autosave" in src
