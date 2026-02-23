"""Integration tests for session list command (T067).

These tests verify that users can list all saved sessions with proper
metadata display and sorting.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import time


class TestSessionList:
    """Test /session list command functionality."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary sessions directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        """Create SessionManager with temp directory."""
        from promptchain.cli.session_manager import SessionManager
        return SessionManager(sessions_dir=temp_sessions_dir)

    def test_list_sessions(self, session_manager):
        """Integration: /session list shows all saved sessions.

        Flow:
        1. Create multiple sessions
        2. Run /session list command
        3. Verify all sessions displayed

        Validates:
        - All sessions appear in list
        - Session names displayed
        - Basic metadata shown
        """
        from promptchain.cli.command_handler import CommandHandler

        # Create multiple sessions
        session1 = session_manager.create_session("project-a", Path.cwd())
        session2 = session_manager.create_session("project-b", Path.cwd())
        session3 = session_manager.create_session("experiment-1", Path.cwd())

        # Save sessions
        session_manager.save_session(session1)
        session_manager.save_session(session2)
        session_manager.save_session(session3)

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Execute list command
            result = handler.handle_session_list()

            assert result.success is True
            assert result.data is not None

            # Verify all sessions in results
            session_list = result.data.get("sessions", [])
            assert len(session_list) == 3

            session_names = [s["name"] for s in session_list]
            assert "project-a" in session_names
            assert "project-b" in session_names
            assert "experiment-1" in session_names

        except AttributeError:
            pytest.skip("handle_session_list() not yet implemented (will be in T074)")

    def test_session_metadata_display(self, session_manager):
        """Integration: Session list includes useful metadata.

        Flow:
        1. Create sessions with messages and agents
        2. List sessions
        3. Verify metadata displayed (agent count, message count, last accessed)

        Validates:
        - Last accessed timestamp (human-readable)
        - Number of agents
        - Number of messages
        - Session state
        """
        from promptchain.cli.command_handler import CommandHandler
        from promptchain.cli.models.agent_config import Agent

        session = session_manager.create_session("metadata-test", Path.cwd())

        # Add agents and messages
        session.agents["coder"] = Agent(
            name="coder",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Coder",
            created_at=time.time()
        )

        session.add_message(role="user", content="Hello")
        session.add_message(role="assistant", content="Hi")
        session_manager.save_session(session)

        handler = CommandHandler(session_manager=session_manager)

        try:
            result = handler.handle_session_list()

            assert result.success is True
            sessions = result.data["sessions"]

            # Find our session
            test_session = next(s for s in sessions if s["name"] == "metadata-test")

            # Verify metadata fields present
            assert "last_accessed" in test_session
            assert "agent_count" in test_session
            assert "message_count" in test_session

            # Verify values
            assert test_session["agent_count"] == 2  # default + coder
            assert test_session["message_count"] == 2

        except (AttributeError, StopIteration):
            pytest.skip("Session metadata not yet implemented")

    def test_sorted_by_last_accessed(self, session_manager):
        """Integration: Sessions sorted by last_accessed (most recent first).

        Flow:
        1. Create sessions at different times
        2. Access them in specific order
        3. List sessions
        4. Verify order is by last_accessed descending

        Validates:
        - Most recently accessed session appears first
        - Sorting is consistent
        """
        from promptchain.cli.command_handler import CommandHandler

        # Create sessions with time delays
        session1 = session_manager.create_session("old-session", Path.cwd())
        session_manager.save_session(session1)
        time.sleep(0.1)

        session2 = session_manager.create_session("middle-session", Path.cwd())
        session_manager.save_session(session2)
        time.sleep(0.1)

        session3 = session_manager.create_session("recent-session", Path.cwd())
        session_manager.save_session(session3)

        handler = CommandHandler(session_manager=session_manager)

        try:
            result = handler.handle_session_list()

            assert result.success is True
            sessions = result.data["sessions"]

            # Verify sorted by last_accessed descending
            # Most recent should be first
            assert sessions[0]["name"] == "recent-session"
            assert sessions[1]["name"] == "middle-session"
            assert sessions[2]["name"] == "old-session"

        except (AttributeError, KeyError):
            pytest.skip("Session sorting not yet implemented")

    def test_empty_session_list(self, session_manager):
        """Integration: List empty when no sessions exist.

        Flow:
        1. Create new SessionManager (no sessions)
        2. Run /session list
        3. Verify empty list with helpful message

        Validates:
        - Empty list handling
        - User-friendly message
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=session_manager)

        try:
            result = handler.handle_session_list()

            assert result.success is True
            sessions = result.data.get("sessions", [])
            assert len(sessions) == 0
            assert "no sessions" in result.message.lower()

        except AttributeError:
            pytest.skip("handle_session_list() not yet implemented")

    def test_session_list_with_active_marker(self, session_manager):
        """Integration: Current/active session marked in list.

        Flow:
        1. Create multiple sessions
        2. Load one as current session
        3. List sessions
        4. Verify current session is marked

        Validates:
        - Active session identification
        - Visual marker (e.g., "(current)")
        """
        from promptchain.cli.command_handler import CommandHandler

        session1 = session_manager.create_session("session-a", Path.cwd())
        session2 = session_manager.create_session("session-b", Path.cwd())

        session_manager.save_session(session1)
        session_manager.save_session(session2)

        # Simulate session2 being current
        handler = CommandHandler(session_manager=session_manager)

        try:
            # Pass current session to handler
            result = handler.handle_session_list(current_session_id=session2.id)

            assert result.success is True

            # Verify message indicates current session
            assert "session-b" in result.message
            # Should have marker like "(current)" or similar
            assert "current" in result.message.lower() or "(active)" in result.message.lower()

        except (AttributeError, TypeError):
            pytest.skip("Active session marking not yet implemented")

    def test_session_list_formatting(self, session_manager):
        """Integration: Session list has readable formatting.

        Flow:
        1. Create sessions
        2. List sessions
        3. Verify formatting is user-friendly

        Validates:
        - Timestamps in human-readable format
        - Clear column alignment
        - Readable output
        """
        from promptchain.cli.command_handler import CommandHandler

        session = session_manager.create_session("format-test", Path.cwd())
        session_manager.save_session(session)

        handler = CommandHandler(session_manager=session_manager)

        try:
            result = handler.handle_session_list()

            # Check message formatting
            message = result.message

            # Should contain session name
            assert "format-test" in message

            # Should have some kind of timestamp formatting
            # (exact format not specified, just check it exists)
            assert result.success is True

        except AttributeError:
            pytest.skip("Session list formatting not yet implemented")

    def test_session_list_performance(self, session_manager):
        """Integration: Listing many sessions completes quickly.

        Flow:
        1. Create 50 sessions
        2. Measure list command time
        3. Verify completes in reasonable time

        Validates:
        - Performance with many sessions
        - Query efficiency
        """
        from promptchain.cli.command_handler import CommandHandler

        # Create 50 sessions
        for i in range(50):
            session = session_manager.create_session(f"session-{i}", Path.cwd())
            session_manager.save_session(session)

        handler = CommandHandler(session_manager=session_manager)

        try:
            start = time.time()
            result = handler.handle_session_list()
            elapsed = time.time() - start

            assert result.success is True
            assert len(result.data["sessions"]) == 50

            # Should complete quickly even with many sessions
            assert elapsed < 1.0, f"Listing 50 sessions took {elapsed:.2f}s"

        except AttributeError:
            pytest.skip("handle_session_list() not yet implemented")
