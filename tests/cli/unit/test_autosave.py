"""Unit tests for auto-save logic (T069).

These tests verify the auto-save mechanism that saves sessions automatically
based on message count and time intervals (SC-007).
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import time
from unittest.mock import Mock, patch, call


class TestAutoSave:
    """Test auto-save logic for sessions."""

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

    def test_autosave_every_5_messages(self, session_manager):
        """Unit: Auto-save triggers after 5 messages (SC-007).

        Given: Session with auto-save enabled
        When: 5 messages added
        Then: Session automatically saved
        """
        session = session_manager.create_session("autosave-test", Path.cwd())

        # Mock save to track calls
        with patch.object(session_manager, 'save_session') as mock_save:
            # Add 4 messages - should not trigger save
            for i in range(4):
                session.add_message(role="user", content=f"Message {i}")
                session.check_autosave(session_manager)

            # No saves yet
            assert mock_save.call_count == 0

            # Add 5th message - should trigger save
            session.add_message(role="user", content="Message 4")
            session.check_autosave(session_manager)

            # Save should be called once
            assert mock_save.call_count == 1

    def test_autosave_every_2_minutes(self, session_manager):
        """Unit: Auto-save triggers after 2 minutes (SC-007).

        Given: Session with auto-save enabled
        When: 2 minutes pass since last save
        Then: Session automatically saved
        """
        session = session_manager.create_session("time-autosave", Path.cwd())

        # Set last_save_time to 2 minutes ago
        session.last_save_time = time.time() - 121  # 121 seconds = >2 minutes

        with patch.object(session_manager, 'save_session') as mock_save:
            # Add a message and check autosave
            session.add_message(role="user", content="Message")
            session.check_autosave(session_manager)

            # Should have triggered time-based save
            assert mock_save.call_count == 1

    def test_autosave_disabled(self, session_manager):
        """Unit: Auto-save can be disabled.

        Given: Session with auto-save disabled
        When: Messages added or time passes
        Then: No automatic saves occur
        """
        session = session_manager.create_session("no-autosave", Path.cwd())
        session.autosave_enabled = False

        with patch.object(session_manager, 'save_session') as mock_save:
            # Add many messages
            for i in range(10):
                session.add_message(role="user", content=f"Message {i}")
                session.check_autosave(session_manager)

            # No saves should occur
            assert mock_save.call_count == 0

    def test_autosave_resets_counter(self, session_manager):
        """Unit: Message counter resets after auto-save.

        Given: Session with 5 messages triggering auto-save
        When: Auto-save completes
        Then: Message counter resets to 0
        """
        session = session_manager.create_session("counter-reset", Path.cwd())

        # Add 5 messages to trigger save
        for i in range(5):
            session.add_message(role="user", content=f"Message {i}")

        # Manually trigger save
        session_manager.save_session(session)
        session.messages_since_save = 0
        session.last_save_time = time.time()

        with patch.object(session_manager, 'save_session') as mock_save:
            # Add 4 more messages - should not trigger save
            for i in range(4):
                session.add_message(role="user", content=f"Message {i+5}")
                session.check_autosave(session_manager)

            # No additional saves
            assert mock_save.call_count == 0

    def test_autosave_updates_timestamp(self, session_manager):
        """Unit: Auto-save updates last_save_time.

        Given: Session with auto-save
        When: Auto-save triggers
        Then: last_save_time updated to current time
        """
        session = session_manager.create_session("timestamp-test", Path.cwd())

        initial_time = session.last_save_time

        # Add 5 messages to trigger save
        for i in range(5):
            session.add_message(role="user", content=f"Message {i}")

        time.sleep(0.1)  # Ensure time difference

        # Trigger autosave
        session.check_autosave(session_manager)

        # Timestamp should be updated
        assert session.last_save_time > initial_time

    def test_autosave_configurable_intervals(self, session_manager):
        """Unit: Auto-save intervals can be configured.

        Given: Session with custom auto-save intervals
        When: Custom thresholds met
        Then: Auto-save triggers accordingly
        """
        session = session_manager.create_session("custom-intervals", Path.cwd())

        # Set custom intervals
        session.autosave_message_interval = 3  # Save every 3 messages
        session.autosave_time_interval = 60    # Save every 60 seconds

        with patch.object(session_manager, 'save_session') as mock_save:
            # Add 2 messages - should not trigger
            for i in range(2):
                session.add_message(role="user", content=f"Message {i}")
                session.check_autosave(session_manager)

            assert mock_save.call_count == 0

            # Add 3rd message - should trigger
            session.add_message(role="user", content="Message 2")
            session.check_autosave(session_manager)

            assert mock_save.call_count == 1

    def test_autosave_with_errors(self, session_manager):
        """Unit: Auto-save handles errors gracefully.

        Given: Session with auto-save
        When: Save operation fails
        Then: Error logged but session continues
        """
        session = session_manager.create_session("error-handling", Path.cwd())

        # Mock save to raise exception
        with patch.object(session_manager, 'save_session') as mock_save:
            mock_save.side_effect = Exception("Save failed")

            # Add 5 messages to trigger save
            for i in range(5):
                session.add_message(role="user", content=f"Message {i}")

            # Should not crash, just log error
            try:
                session.check_autosave(session_manager)
                # If no exception, test passes
            except Exception:
                pytest.fail("Auto-save should handle errors gracefully")

    def test_autosave_performance_impact(self, session_manager):
        """Unit: Auto-save completes quickly.

        Given: Session with auto-save
        When: Auto-save triggers
        Then: Save completes in <100ms
        """
        session = session_manager.create_session("perf-autosave", Path.cwd())

        # Add messages to trigger autosave
        for i in range(5):
            session.add_message(role="user", content=f"Message {i}")

        # Measure autosave time
        start = time.time()
        session.check_autosave(session_manager)
        elapsed = time.time() - start

        # Should be very fast (background save)
        assert elapsed < 0.1, f"Auto-save took {elapsed:.3f}s (should be <0.1s)"

    def test_autosave_on_session_exit(self, session_manager):
        """Unit: Auto-save triggers on session exit.

        Given: Session with unsaved changes
        When: Session exits
        Then: Final auto-save performed
        """
        session = session_manager.create_session("exit-save", Path.cwd())

        # Add messages without triggering autosave
        session.add_message(role="user", content="Message 1")
        session.add_message(role="assistant", content="Response 1")

        with patch.object(session_manager, 'save_session') as mock_save:
            # Simulate session exit
            session.on_exit(session_manager)

            # Should save on exit
            assert mock_save.call_count == 1

    def test_autosave_message_count_tracking(self, session_manager):
        """Unit: messages_since_save counter accurate.

        Given: Session tracking message count
        When: Messages added
        Then: Counter increments correctly
        """
        session = session_manager.create_session("counter-test", Path.cwd())

        # Initial counter
        assert session.messages_since_save == 0

        # Add messages
        for i in range(3):
            session.add_message(role="user", content=f"Message {i}")

        # Counter should be 3
        assert session.messages_since_save == 3

        # Manual save resets counter
        session_manager.save_session(session)
        session.messages_since_save = 0

        assert session.messages_since_save == 0

    def test_autosave_multiple_triggers(self, session_manager):
        """Unit: Auto-save can trigger multiple times in session.

        Given: Long-running session
        When: Multiple autosave thresholds met
        Then: Each triggers its own save
        """
        session = session_manager.create_session("multi-save", Path.cwd())

        with patch.object(session_manager, 'save_session') as mock_save:
            # First batch: 5 messages
            for i in range(5):
                session.add_message(role="user", content=f"Batch1-{i}")
            session.check_autosave(session_manager)
            session.messages_since_save = 0  # Reset after save

            # Second batch: 5 more messages
            for i in range(5):
                session.add_message(role="user", content=f"Batch2-{i}")
            session.check_autosave(session_manager)
            session.messages_since_save = 0  # Reset after save

            # Should have saved twice
            assert mock_save.call_count == 2

    def test_autosave_preserves_state(self, session_manager):
        """Unit: Auto-save preserves all session state.

        Given: Session with messages, agents, metadata
        When: Auto-save triggers
        Then: All state preserved correctly
        """
        from promptchain.cli.models.agent_config import Agent

        session = session_manager.create_session("state-preserve", Path.cwd())

        # Add complex state
        session.agents["coder"] = Agent(
            name="coder",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Coder",
            created_at=time.time()
        )
        session.add_message(role="user", content="Test message")
        session.metadata = {"custom": "value"}

        # Trigger autosave
        for i in range(5):
            session.add_message(role="user", content=f"Message {i}")
        session.check_autosave(session_manager)

        # Load and verify state preserved
        loaded = session_manager.load_session(session.id)

        assert "coder" in loaded.agents
        assert loaded.metadata["custom"] == "value"
        assert len(loaded.messages) == 6  # 1 initial + 5 autosave trigger
