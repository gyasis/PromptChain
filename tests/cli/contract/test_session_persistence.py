"""Contract tests for session persistence (T065).

These tests define the expected behavior for saving and loading sessions
with full conversation history and agent configurations.
"""

import pytest
from pathlib import Path
import tempfile
import shutil
import time


class TestSessionPersistence:
    """Test session save/load roundtrip with history preservation."""

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

    def test_save_session(self, session_manager):
        """Contract: save_session() persists session to SQLite and JSONL.

        Given: Session with messages and agents
        When: save_session() is called
        Then: Session metadata saved to SQLite
        And: Messages saved to JSONL file
        And: Agent configurations saved to SQLite agents table
        """
        from promptchain.cli.models.agent_config import Agent

        # Create session with conversation history
        session = session_manager.create_session(
            name="test-save",
            working_directory=Path.cwd()
        )

        # Add multiple agents
        session.agents["coder"] = Agent(
            name="coder",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Coding assistant",
            created_at=time.time()
        )

        # Add conversation messages
        session.add_message(role="user", content="Hello world")
        session.add_message(role="assistant", content="Hi there!")
        session.add_message(role="user", content="How are you?")

        # Save session
        session_manager.save_session(session)

        # Verify SQLite contains session
        import sqlite3
        conn = sqlite3.connect(session_manager.db_path)
        try:
            cursor = conn.execute(
                "SELECT name, active_agent FROM sessions WHERE id = ?",
                (session.id,)
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "test-save"
            assert row[1] == "default"

            # Verify agents table contains both agents
            cursor = conn.execute(
                "SELECT name, model_name FROM agents WHERE session_id = ?",
                (session.id,)
            )
            agents = cursor.fetchall()
            assert len(agents) == 2  # default + coder
            agent_names = [a[0] for a in agents]
            assert "default" in agent_names
            assert "coder" in agent_names

        finally:
            conn.close()

        # Verify JSONL file exists and contains messages
        messages_file = session_manager.sessions_dir / session.id / "messages.jsonl"
        assert messages_file.exists()

        import json
        with open(messages_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3  # 3 messages

            # Verify first message
            msg1 = json.loads(lines[0])
            assert msg1["role"] == "user"
            assert msg1["content"] == "Hello world"

    def test_load_session(self, session_manager):
        """Contract: load_session() reconstructs session from persistence.

        Given: Saved session in SQLite and JSONL
        When: load_session() is called
        Then: Session object reconstructed with all metadata
        And: Messages loaded from JSONL
        And: Agents loaded from SQLite
        """
        # Create and save session
        session = session_manager.create_session(
            name="test-load",
            working_directory=Path.cwd()
        )

        session.add_message(role="user", content="Test message")
        session.add_message(role="assistant", content="Test response")
        session_manager.save_session(session)

        session_id = session.id

        # Load session
        loaded_session = session_manager.load_session(session_id)

        # Verify session metadata
        assert loaded_session.id == session_id
        assert loaded_session.name == "test-load"
        assert loaded_session.active_agent == "default"

        # Verify messages loaded
        assert len(loaded_session.messages) == 2
        assert loaded_session.messages[0].role == "user"
        assert loaded_session.messages[0].content == "Test message"
        assert loaded_session.messages[1].role == "assistant"
        assert loaded_session.messages[1].content == "Test response"

        # Verify agents loaded
        assert "default" in loaded_session.agents
        assert loaded_session.agents["default"].model_name == "gpt-4.1-mini-2025-04-14"

    def test_history_preserved(self, session_manager):
        """Contract: Conversation history preserved across save/load.

        Given: Session with multi-turn conversation
        When: Session saved and reloaded
        Then: All messages preserved in correct order
        And: Message metadata preserved (agent_name, timestamps)
        """
        session = session_manager.create_session(
            name="test-history",
            working_directory=Path.cwd()
        )

        # Multi-turn conversation
        messages = [
            ("user", "What is Python?"),
            ("assistant", "Python is a programming language."),
            ("user", "Tell me more"),
            ("assistant", "Python is interpreted and dynamically typed."),
            ("user", "Thanks!"),
        ]

        for role, content in messages:
            session.add_message(role=role, content=content)

        # Save
        session_manager.save_session(session)
        session_id = session.id

        # Load
        loaded = session_manager.load_session(session_id)

        # Verify all messages preserved
        assert len(loaded.messages) == 5

        for i, (role, content) in enumerate(messages):
            assert loaded.messages[i].role == role
            assert loaded.messages[i].content == content
            assert loaded.messages[i].timestamp is not None

    def test_agents_restored(self, session_manager):
        """Contract: Agent configurations restored with full details.

        Given: Session with multiple agents with usage stats
        When: Session loaded
        Then: All agents restored with correct models
        And: Usage statistics preserved (usage_count, last_used)
        And: Active agent correctly set
        """
        from promptchain.cli.models.agent_config import Agent

        session = session_manager.create_session(
            name="test-agents",
            working_directory=Path.cwd()
        )

        # Add agents with usage stats
        now = time.time()
        session.agents["fast"] = Agent(
            name="fast",
            model_name="anthropic/claude-3-haiku-20240307",
            description="Fast agent",
            created_at=now,
            last_used=now + 100,
            usage_count=5
        )

        session.agents["smart"] = Agent(
            name="smart",
            model_name="anthropic/claude-3-opus-20240229",
            description="Smart agent",
            created_at=now,
            last_used=now + 200,
            usage_count=10
        )

        # Set active agent to non-default
        session.active_agent = "smart"

        # Save and load
        session_manager.save_session(session)
        loaded = session_manager.load_session(session.id)

        # Verify all agents restored
        assert len(loaded.agents) == 3  # default + fast + smart
        assert "fast" in loaded.agents
        assert "smart" in loaded.agents

        # Verify agent details
        fast_agent = loaded.agents["fast"]
        assert fast_agent.model_name == "anthropic/claude-3-haiku-20240307"
        assert fast_agent.usage_count == 5
        assert fast_agent.last_used is not None

        smart_agent = loaded.agents["smart"]
        assert smart_agent.model_name == "anthropic/claude-3-opus-20240229"
        assert smart_agent.usage_count == 10

        # Verify active agent
        assert loaded.active_agent == "smart"

    def test_empty_session_save_load(self, session_manager):
        """Contract: Empty session (no messages) can be saved and loaded.

        Given: New session with no messages
        When: Session saved and loaded
        Then: Session restored with empty message list
        And: Default agent present
        """
        session = session_manager.create_session(
            name="empty-session",
            working_directory=Path.cwd()
        )

        # Don't add any messages
        session_manager.save_session(session)

        loaded = session_manager.load_session(session.id)

        assert loaded.name == "empty-session"
        assert len(loaded.messages) == 0
        assert "default" in loaded.agents

    def test_session_metadata_preserved(self, session_manager):
        """Contract: Session metadata preserved across save/load.

        Given: Session with custom metadata
        When: Session saved and loaded
        Then: All metadata fields preserved
        """
        session = session_manager.create_session(
            name="test-metadata",
            working_directory=Path.cwd()
        )

        # Set custom metadata
        session.metadata = {
            "project": "PromptChain",
            "version": "0.5.0",
            "custom_field": "test_value"
        }

        session_manager.save_session(session)
        loaded = session_manager.load_session(session.id)

        assert loaded.metadata == session.metadata
        assert loaded.metadata["project"] == "PromptChain"

    def test_concurrent_saves(self, session_manager):
        """Contract: Multiple saves don't corrupt session data.

        Given: Session that is saved multiple times
        When: save_session() called repeatedly
        Then: Latest state always preserved
        And: No data corruption
        """
        session = session_manager.create_session(
            name="test-concurrent",
            working_directory=Path.cwd()
        )

        # Save multiple times with different states
        session.add_message(role="user", content="Message 1")
        session_manager.save_session(session)

        session.add_message(role="assistant", content="Response 1")
        session_manager.save_session(session)

        session.add_message(role="user", content="Message 2")
        session_manager.save_session(session)

        # Load and verify latest state
        loaded = session_manager.load_session(session.id)
        assert len(loaded.messages) == 3
        assert loaded.messages[-1].content == "Message 2"
