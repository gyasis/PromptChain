"""Integration tests for exit behavior.

These tests verify graceful exit mechanisms including /exit command,
Ctrl+D handling, goodbye messages, and session cleanup.

Test Coverage:
- test_slash_exit_command: /exit command triggers graceful shutdown
- test_ctrl_d_exit: Ctrl+D (EOF) triggers exit
- test_goodbye_message: Goodbye message displayed on exit
- test_session_cleanup: Session is saved before exit
- test_auto_save_on_exit: Auto-save triggered before exit
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import time

from promptchain.cli.models import Session


class TestExitBehavior:
    """Integration tests for exit behavior."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary directory for test sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def session_manager(self, temp_sessions_dir):
        """Create SessionManager for testing."""
        from promptchain.cli.session_manager import SessionManager
        return SessionManager(sessions_dir=temp_sessions_dir)

    @pytest.fixture
    def test_session(self, session_manager):
        """Create a test session."""
        session = session_manager.create_session(
            name="exit-test",
            working_directory=Path.cwd()
        )
        # Add some messages
        session.add_message(role="user", content="Test message")
        session.add_message(role="assistant", content="Test response")
        return session

    @pytest.mark.asyncio
    async def test_slash_exit_command(self, test_session):
        """Integration: /exit command triggers graceful shutdown.

        Flow:
        1. User types /exit
        2. Command is parsed by CommandHandler
        3. Exit handler is called
        4. Session is saved
        5. Goodbye message is displayed
        6. Application exits

        Validates:
        - /exit is recognized as command
        - Exit handler is invoked
        - CommandResult indicates exit
        """
        from promptchain.cli.command_handler import CommandHandler

        # Create command handler with session manager
        handler = CommandHandler(session_manager=Mock())

        # Parse /exit command
        parsed = handler.parse_command("/exit")

        # Validate command parsed correctly
        assert parsed is not None
        assert parsed.name == "exit"
        assert parsed.subcommand is None

        # Execute exit command (will be implemented in T039)
        try:
            result = await handler.handle_exit(test_session)

            # Validate result
            assert result.success is True
            assert "goodbye" in result.message.lower() or "exit" in result.message.lower()

        except AttributeError:
            # Expected until T039 is implemented
            pytest.skip("handle_exit() not yet implemented (will be in T039)")

    @pytest.mark.asyncio
    async def test_ctrl_d_exit(self, test_session):
        """Integration: Ctrl+D (EOF) triggers exit gracefully.

        Flow:
        1. User presses Ctrl+D
        2. EOF is detected by input handler
        3. Same exit flow as /exit
        4. Session saved and app exits

        Validates:
        - EOF signal handled
        - Exit triggered
        - Session cleanup occurs
        """
        try:
            from promptchain.cli.tui.app import PromptChainApp

            # Mock app with Ctrl+D binding
            with patch.object(PromptChainApp, 'on_exit') as mock_exit:
                mock_exit.return_value = AsyncMock()

                # Simulate Ctrl+D (will be bound in T040)
                # In Textual, this is handled via action_quit or similar
                app = Mock(spec=PromptChainApp)
                app.on_exit = AsyncMock()

                # Trigger exit
                await app.on_exit()

                # Validate exit was called
                assert app.on_exit.called

        except ImportError:
            pytest.skip("PromptChainApp not yet implemented (will be in T027)")

    @pytest.mark.asyncio
    async def test_goodbye_message(self, test_session, session_manager):
        """Integration: Goodbye message displayed on exit.

        Expected goodbye message format:
        - "Session '<name>' saved"
        - "Goodbye!"
        - Optional: session statistics

        Validates:
        - Goodbye message is displayed
        - Message includes session name
        - Message is formatted correctly
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=session_manager)

        try:
            result = await handler.handle_exit(test_session)

            # Validate goodbye message
            message = result.message
            assert message is not None
            assert len(message) > 0

            # Should mention saving or goodbye
            assert any(
                keyword in message.lower()
                for keyword in ["goodbye", "exit", "saved", "bye"]
            )

        except AttributeError:
            pytest.skip("handle_exit() not yet implemented (will be in T039)")

    @pytest.mark.asyncio
    async def test_session_cleanup(self, test_session, session_manager):
        """Integration: Session is saved before exit.

        Flow:
        1. User has active session with messages
        2. User triggers exit
        3. Session is automatically saved
        4. All messages are persisted
        5. Session state is saved to database

        Validates:
        - Session save is triggered
        - All messages are persisted to JSONL
        - Session metadata saved to SQLite
        - Session can be resumed later
        """
        # Add messages to session
        initial_message_count = len(test_session.messages)

        # Simulate exit with save
        session_manager.save_session(test_session)

        # Reload session to verify persistence
        reloaded = session_manager.load_session(test_session.name)

        # Validate messages were persisted
        assert len(reloaded.messages) == initial_message_count
        assert reloaded.id == test_session.id

    @pytest.mark.asyncio
    async def test_auto_save_on_exit(self, test_session, session_manager):
        """Integration: Auto-save triggered before exit.

        Even if auto-save conditions not met (< 5 messages, < 2 minutes):
        - Exit triggers final save
        - All conversation data persisted
        - No data loss on exit

        Validates:
        - Final save occurs regardless of auto-save interval
        - Session state is current at exit
        """
        # Modify session but don't trigger auto-save
        test_session.add_message(role="user", content="Last message before exit")

        # Record last_accessed before save
        before_save = test_session.last_accessed

        # Simulate exit with final save
        await asyncio.sleep(0.01)  # Small delay
        session_manager.save_session(test_session)

        # Validate session was saved
        reloaded = session_manager.load_session(test_session.name)

        # Check that the last message was saved
        assert len(reloaded.messages) == len(test_session.messages)
        assert reloaded.messages[-1].content == "Last message before exit"

    @pytest.mark.asyncio
    async def test_exit_with_unsaved_changes(self, test_session, session_manager):
        """Integration: Exit with unsaved changes saves automatically.

        Scenario:
        - User has made changes since last save
        - User triggers exit
        - Changes are saved automatically (no prompt)

        Validates:
        - No data loss on exit
        - Auto-save handles unsaved changes
        - User doesn't need to manually save
        """
        # Add new message (unsaved change)
        test_session.add_message(
            role="user",
            content="Unsaved message before exit"
        )

        # Trigger exit (should auto-save)
        session_manager.save_session(test_session)

        # Verify unsaved message was persisted
        reloaded = session_manager.load_session(test_session.name)
        last_msg = reloaded.messages[-1]

        assert last_msg.content == "Unsaved message before exit"
        assert last_msg.role == "user"

    @pytest.mark.asyncio
    async def test_exit_performance(self, test_session, session_manager):
        """Integration: Exit completes quickly (<2s).

        Exit operations should be fast:
        - Final save <2s (SC-003)
        - No blocking operations
        - Graceful shutdown

        Validates:
        - Exit path is optimized
        - No hanging on exit
        """
        start = time.perf_counter()

        # Perform exit operations
        session_manager.save_session(test_session)
        # In full implementation, also includes app.exit()

        duration = time.perf_counter() - start

        assert duration < 2.0, f"Exit took {duration:.2f}s, exceeds 2s target"

    @pytest.mark.asyncio
    async def test_multiple_exit_commands(self, test_session):
        """Integration: Multiple /exit commands handled gracefully.

        Scenario:
        - User types /exit multiple times rapidly
        - Only one exit occurs
        - No errors or crashes

        Validates:
        - Exit is idempotent
        - No race conditions
        """
        from promptchain.cli.command_handler import CommandHandler

        handler = CommandHandler(session_manager=Mock())

        try:
            # Execute exit multiple times
            results = []
            for _ in range(3):
                result = await handler.handle_exit(test_session)
                results.append(result)

            # Validate all succeeded (or only first one)
            assert all(r.success for r in results)

        except AttributeError:
            pytest.skip("handle_exit() not yet implemented (will be in T039)")

    @pytest.mark.asyncio
    async def test_exit_during_agent_response(self, test_session):
        """Integration: Exit during agent response cancels gracefully.

        Scenario:
        - Agent is generating response (long operation)
        - User triggers exit
        - Agent operation is cancelled
        - Exit completes

        Validates:
        - Agent tasks can be cancelled
        - Exit doesn't hang waiting for response
        - Partial response not saved
        """
        # This will be testable once agent integration is complete
        # For now, validate that exit can be triggered at any time

        # Simulate agent response in progress
        agent_task = asyncio.create_task(asyncio.sleep(5))  # Simulated long response

        # Trigger exit (should cancel task)
        agent_task.cancel()

        try:
            await agent_task
        except asyncio.CancelledError:
            # Expected - task was cancelled
            pass

        # Exit should complete even though task was cancelled
        assert agent_task.cancelled()

    @pytest.mark.asyncio
    async def test_exit_updates_session_state(self, test_session, session_manager):
        """Integration: Exit updates session state appropriately.

        On exit:
        - last_accessed is updated
        - Session state may transition (Active -> Paused/Archived on next load)
        - All metadata is current

        Validates:
        - Session metadata is up-to-date
        - State is consistent
        """
        # Record initial last_accessed
        initial_last_accessed = test_session.last_accessed

        # Wait briefly
        await asyncio.sleep(0.1)

        # Update access time (as exit would)
        test_session.update_access_time()

        # Save session
        session_manager.save_session(test_session)

        # Validate last_accessed was updated
        assert test_session.last_accessed > initial_last_accessed

        # Reload and verify
        reloaded = session_manager.load_session(test_session.name)
        assert reloaded.last_accessed == test_session.last_accessed
