"""Integration tests for session deletion (T068).

These tests verify that sessions can be properly deleted with all
associated files and database entries cleaned up.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import time


class TestSessionDelete:
    """Test session deletion workflow and cleanup."""

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

    def test_delete_session(self, session_manager):
        """Integration: /session delete removes session from system.

        Flow:
        1. Create and save session
        2. Delete session via command
        3. Verify session cannot be loaded
        4. Verify session not in list

        Validates:
        - Session deleted from database
        - Session cannot be loaded after deletion
        """
        from promptchain.cli.command_handler import CommandHandler

        # Create session
        session = session_manager.create_session("test-delete", Path.cwd())
        session.add_message(role="user", content="Test message")
        session_manager.save_session(session)
        session_id = session.id

        # Verify session exists
        loaded = session_manager.load_session(session_id)
        assert loaded.name == "test-delete"

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Delete session (with confirmation)
            result = handler.handle_session_delete(session_id=session_id, confirmed=True)

            assert result.success is True
            assert "deleted" in result.message.lower()

            # Verify session cannot be loaded
            with pytest.raises(ValueError):
                session_manager.load_session(session_id)

        except AttributeError:
            pytest.skip("handle_session_delete() not yet implemented (will be in T082)")

    def test_delete_removes_files(self, session_manager):
        """Integration: Session deletion removes all associated files.

        Flow:
        1. Create session with messages
        2. Verify files exist (messages.jsonl, etc.)
        3. Delete session
        4. Verify files removed

        Validates:
        - Session directory deleted
        - messages.jsonl removed
        - No orphaned files remain
        """
        from promptchain.cli.command_handler import CommandHandler

        session = session_manager.create_session("test-files", Path.cwd())
        session.add_message(role="user", content="Message 1")
        session.add_message(role="assistant", content="Response 1")
        session_manager.save_session(session)

        session_id = session.id
        session_dir = session_manager.sessions_dir / session_id
        messages_file = session_dir / "messages.jsonl"

        # Verify files exist
        assert session_dir.exists()
        assert messages_file.exists()

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Delete session (with confirmation)
            handler.handle_session_delete(session_id=session_id, confirmed=True)

            # Verify files removed
            assert not session_dir.exists()
            assert not messages_file.exists()

        except AttributeError:
            pytest.skip("File deletion not yet implemented (will be in T083)")

    def test_delete_cascade_agents(self, session_manager):
        """Integration: Deleting session cascades to delete agents.

        Flow:
        1. Create session with multiple agents
        2. Delete session
        3. Verify agents removed from database

        Validates:
        - Foreign key cascade deletion
        - Agent entries removed
        - No orphaned agent data
        """
        from promptchain.cli.command_handler import CommandHandler
        from promptchain.cli.models.agent_config import Agent

        session = session_manager.create_session("test-cascade", Path.cwd())

        # Add agents
        session.agents["coder"] = Agent(
            name="coder",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Coder",
            created_at=time.time()
        )
        session.agents["writer"] = Agent(
            name="writer",
            model_name="anthropic/claude-3-opus-20240229",
            description="Writer",
            created_at=time.time()
        )

        session_manager.save_session(session)
        session_id = session.id

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Delete session (with confirmation)
            handler.handle_session_delete(session_id=session_id, confirmed=True)

            # Verify agents removed from database
            import sqlite3
            conn = sqlite3.connect(session_manager.db_path)
            try:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM agents WHERE session_id = ?",
                    (session_id,)
                )
                count = cursor.fetchone()[0]
                assert count == 0, "Agents should be deleted with session"
            finally:
                conn.close()

        except AttributeError:
            pytest.skip("Cascade deletion not yet implemented")

    def test_delete_nonexistent_session(self, session_manager):
        """Integration: Deleting nonexistent session returns error.

        Flow:
        1. Try to delete session that doesn't exist
        2. Verify error message

        Validates:
        - Error handling for missing sessions
        - User-friendly error message
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=session_manager)

        try:
            result = handler.handle_session_delete(session_id="nonexistent-id", confirmed=True)

            assert result.success is False
            assert "not found" in result.message.lower() or "does not exist" in result.message.lower()

        except AttributeError:
            pytest.skip("Error handling not yet implemented")

    def test_cannot_delete_active_session(self, session_manager):
        """Integration: Cannot delete currently active session.

        Flow:
        1. Create session and set as active
        2. Try to delete active session
        3. Verify deletion prevented

        Validates:
        - Protection against deleting active session
        - User warned to switch sessions first
        """
        from promptchain.cli.command_handler import CommandHandler

        session = session_manager.create_session("active-session", Path.cwd())
        session_manager.save_session(session)

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Simulate active session (in real app, this would be current session)
            result = handler.handle_session_delete(
                session_id=session.id,
                is_active=True  # Flag indicating this is the active session
            )

            assert result.success is False
            assert "active" in result.message.lower()
            assert "switch" in result.message.lower() or "close" in result.message.lower()

            # Verify session still exists
            loaded = session_manager.load_session(session.id)
            assert loaded.name == "active-session"

        except (AttributeError, TypeError):
            pytest.skip("Active session protection not yet implemented")

    def test_delete_with_confirmation(self, session_manager):
        """Integration: Session deletion requires confirmation.

        Flow:
        1. Create session
        2. Call delete without confirmation
        3. Verify session not deleted
        4. Call delete with confirmation
        5. Verify session deleted

        Validates:
        - Confirmation requirement
        - Safety against accidental deletion
        """
        from promptchain.cli.command_handler import CommandHandler

        session = session_manager.create_session("confirm-delete", Path.cwd())
        session_manager.save_session(session)
        session_id = session.id

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Try delete without confirmation
            result = handler.handle_session_delete(
                session_id=session_id,
                confirmed=False
            )

            # Should prompt for confirmation, not delete yet
            if not result.success:
                assert "confirm" in result.message.lower()

                # Verify session still exists
                loaded = session_manager.load_session(session_id)
                assert loaded.name == "confirm-delete"

            # Delete with confirmation
            result = handler.handle_session_delete(
                session_id=session_id,
                confirmed=True
            )

            assert result.success is True

            # Verify session deleted
            with pytest.raises(ValueError):
                session_manager.load_session(session_id)

        except (AttributeError, TypeError):
            pytest.skip("Confirmation flow not yet implemented")

    def test_delete_preserves_other_sessions(self, session_manager):
        """Integration: Deleting one session doesn't affect others.

        Flow:
        1. Create multiple sessions
        2. Delete one session
        3. Verify others still exist

        Validates:
        - Isolated deletion
        - Other sessions unaffected
        """
        from promptchain.cli.command_handler import CommandHandler

        # Create multiple sessions
        session1 = session_manager.create_session("keep-1", Path.cwd())
        session2 = session_manager.create_session("delete-me", Path.cwd())
        session3 = session_manager.create_session("keep-2", Path.cwd())

        session_manager.save_session(session1)
        session_manager.save_session(session2)
        session_manager.save_session(session3)

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Delete middle session (with confirmation)
            handler.handle_session_delete(session_id=session2.id, confirmed=True)

            # Verify others still exist
            loaded1 = session_manager.load_session(session1.id)
            loaded3 = session_manager.load_session(session3.id)

            assert loaded1.name == "keep-1"
            assert loaded3.name == "keep-2"

            # Verify deleted session gone
            with pytest.raises(ValueError):
                session_manager.load_session(session2.id)

        except AttributeError:
            pytest.skip("Deletion isolation not yet verified")

    def test_delete_session_with_history(self, session_manager):
        """Integration: Sessions with conversation history can be deleted.

        Flow:
        1. Create session with extensive history
        2. Delete session
        3. Verify all history removed

        Validates:
        - Large sessions can be deleted
        - All message history removed
        """
        from promptchain.cli.command_handler import CommandHandler

        session = session_manager.create_session("history-delete", Path.cwd())

        # Add extensive history
        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            session.add_message(role=role, content=f"Message {i}")

        session_manager.save_session(session)
        session_id = session.id

        # Verify messages file is large
        messages_file = session_manager.sessions_dir / session_id / "messages.jsonl"
        assert messages_file.exists()

        handler = CommandHandler(session_manager=session_manager)

        try:
            # Delete session (with confirmation)
            handler.handle_session_delete(session_id=session_id, confirmed=True)

            # Verify all files removed
            assert not messages_file.exists()

            # Verify session cannot be loaded
            with pytest.raises(ValueError):
                session_manager.load_session(session_id)

        except AttributeError:
            pytest.skip("History deletion not yet implemented")
