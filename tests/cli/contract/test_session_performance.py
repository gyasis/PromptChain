"""Contract tests for session performance (T066).

These tests verify that session operations meet performance requirements
specified in success criteria SC-003.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import time


class TestSessionPerformance:
    """Test session save/load performance requirements."""

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

    def test_save_under_2_seconds(self, session_manager):
        """Contract: save_session() completes in <2 seconds (SC-003).

        Given: Session with 100 messages and 5 agents
        When: save_session() is called
        Then: Operation completes in less than 2 seconds
        """
        from promptchain.cli.models.agent_config import Agent

        session = session_manager.create_session(
            name="perf-save",
            working_directory=Path.cwd()
        )

        # Add 5 agents
        for i in range(5):
            agent_name = f"agent{i}"
            session.agents[agent_name] = Agent(
                name=agent_name,
                model_name=f"openai/gpt-{4-i}",
                description=f"Agent {i}",
                created_at=time.time()
            )

        # Add 100 messages
        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            session.add_message(
                role=role,
                content=f"Message {i} with some content to make it realistic"
            )

        # Measure save time
        start = time.time()
        session_manager.save_session(session)
        elapsed = time.time() - start

        # Assert performance requirement (SC-003)
        assert elapsed < 2.0, f"save_session took {elapsed:.2f}s (should be <2s)"

    def test_load_under_3_seconds(self, session_manager):
        """Contract: load_session() completes in <3 seconds (SC-003).

        Given: Saved session with 100 messages and 5 agents
        When: load_session() is called
        Then: Operation completes in less than 3 seconds
        """
        from promptchain.cli.models.agent_config import Agent

        # Create and save session with lots of data
        session = session_manager.create_session(
            name="perf-load",
            working_directory=Path.cwd()
        )

        # Add agents and messages
        for i in range(5):
            agent_name = f"agent{i}"
            session.agents[agent_name] = Agent(
                name=agent_name,
                model_name=f"openai/gpt-{4-i}",
                description=f"Agent {i}",
                created_at=time.time()
            )

        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            session.add_message(
                role=role,
                content=f"Message {i} with realistic content length"
            )

        session_manager.save_session(session)
        session_id = session.id

        # Measure load time
        start = time.time()
        loaded_session = session_manager.load_session(session_id)
        elapsed = time.time() - start

        # Assert performance requirement (SC-003)
        assert elapsed < 3.0, f"load_session took {elapsed:.2f}s (should be <3s)"

        # Verify data loaded correctly
        assert len(loaded_session.messages) == 100
        assert len(loaded_session.agents) == 6  # 5 + default

    def test_save_scales_linearly(self, session_manager):
        """Contract: Save time scales approximately linearly with messages.

        Given: Sessions with 10, 50, 100 messages
        When: save_session() is called for each
        Then: Time scales roughly linearly (not exponential)
        """
        timings = {}

        for msg_count in [10, 50, 100]:
            session = session_manager.create_session(
                name=f"scale-{msg_count}",
                working_directory=Path.cwd()
            )

            # Add messages
            for i in range(msg_count):
                role = "user" if i % 2 == 0 else "assistant"
                session.add_message(role=role, content=f"Message {i}")

            # Measure save time
            start = time.time()
            session_manager.save_session(session)
            elapsed = time.time() - start

            timings[msg_count] = elapsed

        # Check approximate linear scaling
        # 100 messages should not take more than 15x the time of 10 messages
        ratio = timings[100] / timings[10]
        assert ratio < 15, f"Save time scaling non-linear: 10msg={timings[10]:.3f}s, 100msg={timings[100]:.3f}s (ratio={ratio:.1f}x)"

    def test_load_with_large_messages(self, session_manager):
        """Contract: Load handles messages with large content efficiently.

        Given: Session with messages containing large text (>10KB each)
        When: load_session() is called
        Then: Completes within reasonable time (<5s)
        """
        session = session_manager.create_session(
            name="large-messages",
            working_directory=Path.cwd()
        )

        # Add messages with large content (simulate code snippets, logs)
        large_content = "x" * 10000  # 10KB of content
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            session.add_message(role=role, content=f"Message {i}\n{large_content}")

        session_manager.save_session(session)

        # Measure load time
        start = time.time()
        loaded = session_manager.load_session(session.id)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Load with large messages took {elapsed:.2f}s"
        assert len(loaded.messages) == 10

    def test_multiple_concurrent_saves(self, session_manager):
        """Contract: Multiple rapid saves don't degrade performance.

        Given: Session saved 10 times rapidly
        When: save_session() called repeatedly
        Then: Each save completes quickly (no lock contention)
        """
        session = session_manager.create_session(
            name="concurrent-saves",
            working_directory=Path.cwd()
        )

        # Perform 10 rapid saves
        save_times = []
        for i in range(10):
            session.add_message(role="user", content=f"Message {i}")

            start = time.time()
            session_manager.save_session(session)
            elapsed = time.time() - start

            save_times.append(elapsed)

        # No save should take more than 1 second
        max_time = max(save_times)
        assert max_time < 1.0, f"Slowest save took {max_time:.2f}s"

        # Average should be well under 1 second
        avg_time = sum(save_times) / len(save_times)
        assert avg_time < 0.5, f"Average save time {avg_time:.2f}s too high"

    def test_session_directory_size_reasonable(self, session_manager):
        """Contract: Session storage doesn't grow unreasonably.

        Given: Session with 100 messages
        When: Session saved to disk
        Then: Total directory size is reasonable (<1MB for text messages)
        """
        session = session_manager.create_session(
            name="size-test",
            working_directory=Path.cwd()
        )

        # Add 100 messages with typical content
        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"This is message {i} with some typical conversational content that might span a few sentences."
            session.add_message(role=role, content=content)

        session_manager.save_session(session)

        # Calculate session directory size
        session_dir = session_manager.sessions_dir / session.id
        total_size = sum(f.stat().st_size for f in session_dir.rglob('*') if f.is_file())

        # Should be well under 1MB for text-only messages
        assert total_size < 1024 * 1024, f"Session directory size {total_size} bytes too large"
