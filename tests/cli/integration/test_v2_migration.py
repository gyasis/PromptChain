"""Integration tests for V1 → V2 schema migration (T015).

This module tests the database migration from V1 to V2 schema, verifying:
1. V1 session creation
2. Migration execution
3. V2 schema verification
4. Data preservation
5. Foreign key integrity
6. Backward compatibility

RED → GREEN → REFACTOR approach:
- RED: Test fails because V2 tables don't exist
- GREEN: Migration script creates tables and preserves data
- REFACTOR: Extract migration logic into reusable helper functions
"""

import json
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from promptchain.cli.migrations.v2_schema import (
    migrate_v1_to_v2,
    rollback_migration,
    verify_migration,
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_sessions.db"
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()
    os.rmdir(temp_dir)


@pytest.fixture
def v1_database(temp_db):
    """Create a V1 schema database with sample data.

    This represents the pre-migration state with V1 schema only.
    """
    conn = sqlite3.connect(temp_db)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        # Create V1 schema (from schema.sql, version 1)
        conn.execute(
            """
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at REAL NOT NULL,
                description TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO schema_version (version, applied_at, description)
            VALUES (1, ?, 'Initial CLI schema')
            """,
            (datetime.now().timestamp(),),
        )

        # V1 sessions table (no orchestration_config, schema_version columns)
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                working_directory TEXT NOT NULL,
                active_agent TEXT,
                default_model TEXT NOT NULL DEFAULT 'gpt-4.1-mini-2025-04-14',
                auto_save_enabled INTEGER NOT NULL DEFAULT 1,
                auto_save_interval INTEGER NOT NULL DEFAULT 120,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )

        # V1 agents table (no instruction_chain, tools, history_config columns)
        conn.execute(
            """
            CREATE TABLE agents (
                session_id TEXT NOT NULL,
                name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                last_used REAL,
                usage_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (session_id, name),
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """
        )

        # Insert sample V1 session
        session_id = str(uuid.uuid4())
        now = datetime.now().timestamp()
        conn.execute(
            """
            INSERT INTO sessions (id, name, created_at, last_accessed, working_directory)
            VALUES (?, 'test-session', ?, ?, ?)
            """,
            (session_id, now, now, os.getcwd()),
        )

        # Insert sample V1 agent
        conn.execute(
            """
            INSERT INTO agents (session_id, name, model_name, description, created_at)
            VALUES (?, 'default', 'gpt-4.1-mini-2025-04-14', 'Default agent', ?)
            """,
            (session_id, now),
        )

        conn.commit()
        return session_id  # Return for verification later

    finally:
        conn.close()


class TestV2Migration:
    """Test suite for V1 → V2 schema migration."""

    def test_1_v1_session_creation(self, v1_database, temp_db):
        """RED: Verify V1 schema exists without V2 columns.

        This test should PASS initially, confirming V1 state before migration.
        """
        conn = sqlite3.connect(temp_db)
        try:
            # Verify V1 schema version
            cursor = conn.execute("SELECT MAX(version) FROM schema_version")
            version = cursor.fetchone()[0]
            assert version == 1, "Expected V1 schema version"

            # Verify sessions table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            )
            assert cursor.fetchone(), "sessions table should exist"

            # Verify V2 columns DO NOT exist (V1 state)
            cursor = conn.execute("PRAGMA table_info(sessions)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "orchestration_config" not in columns, "V2 column should not exist yet"
            assert "schema_version" not in columns, "V2 column should not exist yet"

            # Verify agents table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='agents'"
            )
            assert cursor.fetchone(), "agents table should exist"

            # Verify V2 columns DO NOT exist in agents (V1 state)
            cursor = conn.execute("PRAGMA table_info(agents)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "instruction_chain" not in columns, "V2 column should not exist yet"
            assert "tools" not in columns, "V2 column should not exist yet"
            assert "history_config" not in columns, "V2 column should not exist yet"

            # Verify V2 tables DO NOT exist (V1 state)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='mcp_servers'"
            )
            assert not cursor.fetchone(), "mcp_servers table should not exist yet"

            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_states'"
            )
            assert not cursor.fetchone(), "workflow_states table should not exist yet"

            print("✅ V1 schema confirmed (no V2 columns or tables)")

        finally:
            conn.close()

    def test_2_migration_execution(self, v1_database, temp_db):
        """GREEN: Execute migration and verify V2 schema created.

        This test should FAIL initially (RED), then PASS after migration implementation (GREEN).
        """
        # Execute migration
        stats = migrate_v1_to_v2(temp_db)

        # Verify migration statistics
        assert stats["migration_version"] == 2
        assert stats["sessions_migrated"] == 1
        assert stats["agents_migrated"] == 1
        assert stats["statements_executed"] > 0

        conn = sqlite3.connect(temp_db)
        try:
            # Verify schema version updated to 2
            cursor = conn.execute("SELECT MAX(version) FROM schema_version")
            version = cursor.fetchone()[0]
            assert version == 2, "Schema version should be 2 after migration"

            # Verify sessions table has V2 columns
            cursor = conn.execute("PRAGMA table_info(sessions)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "orchestration_config" in columns, "orchestration_config column missing"
            assert "schema_version" in columns, "schema_version column missing"

            # Verify agents table has V2 columns
            cursor = conn.execute("PRAGMA table_info(agents)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "instruction_chain" in columns, "instruction_chain column missing"
            assert "tools" in columns, "tools column missing"
            assert "history_config" in columns, "history_config column missing"

            # Verify mcp_servers table created
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='mcp_servers'"
            )
            assert cursor.fetchone(), "mcp_servers table not created"

            # Verify workflow_states table created
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_states'"
            )
            assert cursor.fetchone(), "workflow_states table not created"

            print("✅ V2 schema created successfully")

        finally:
            conn.close()

    def test_3_v2_schema_verification(self, v1_database, temp_db):
        """Verify V2 schema integrity using verify_migration function."""
        # Execute migration first
        migrate_v1_to_v2(temp_db)

        # Run verification
        assert verify_migration(temp_db), "Migration verification failed"

        print("✅ V2 schema verification passed")

    def test_4_data_preservation(self, v1_database, temp_db):
        """Verify existing V1 data preserved after migration."""
        session_id = v1_database

        # Execute migration
        migrate_v1_to_v2(temp_db)

        conn = sqlite3.connect(temp_db)
        try:
            # Verify session data preserved
            cursor = conn.execute(
                "SELECT id, name, working_directory FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
            assert row, "Session data lost during migration"
            assert row[1] == "test-session", "Session name changed during migration"
            assert row[2] == os.getcwd(), "Working directory changed during migration"

            # Verify agent data preserved
            cursor = conn.execute(
                "SELECT name, model_name, description FROM agents WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            assert row, "Agent data lost during migration"
            assert row[0] == "default", "Agent name changed during migration"
            assert row[1] == "gpt-4.1-mini-2025-04-14", "Agent model changed during migration"

            # Verify V2 fields initialized with defaults
            cursor = conn.execute(
                "SELECT orchestration_config, schema_version FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            assert row[0], "orchestration_config not initialized"
            assert row[1] == "2.0", "schema_version not set correctly"

            # Verify orchestration_config is valid JSON
            config = json.loads(row[0])
            assert config["execution_mode"] == "router", "Default execution_mode incorrect"
            assert (
                config["auto_include_history"] is True
            ), "Default auto_include_history incorrect"

            # Verify agent V2 fields initialized
            cursor = conn.execute(
                "SELECT instruction_chain, tools, history_config FROM agents WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            instruction_chain = json.loads(row[0])
            tools = json.loads(row[1])
            history_config = json.loads(row[2])

            assert instruction_chain == ["{input}"], "Default instruction_chain incorrect"
            assert tools == [], "Default tools incorrect"
            assert history_config["enabled"] is True, "Default history_config incorrect"
            assert history_config["max_tokens"] == 4000, "Default max_tokens incorrect"

            print("✅ All V1 data preserved and V2 defaults initialized")

        finally:
            conn.close()

    def test_5_foreign_key_integrity(self, v1_database, temp_db):
        """Verify foreign key relationships work correctly after migration."""
        session_id = v1_database

        # Execute migration
        migrate_v1_to_v2(temp_db)

        conn = sqlite3.connect(temp_db)
        conn.execute("PRAGMA foreign_keys = ON")

        try:
            # Test CASCADE DELETE: deleting session should delete related records
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()

            # Verify agents deleted (CASCADE)
            cursor = conn.execute("SELECT COUNT(*) FROM agents WHERE session_id = ?", (session_id,))
            assert cursor.fetchone()[0] == 0, "Agents not deleted on session CASCADE DELETE"

            # Test INSERT with valid foreign key
            new_session_id = str(uuid.uuid4())
            now = datetime.now().timestamp()
            conn.execute(
                """
                INSERT INTO sessions (id, name, created_at, last_accessed, working_directory)
                VALUES (?, 'test-session-2', ?, ?, ?)
                """,
                (new_session_id, now, now, os.getcwd()),
            )

            # Insert MCP server with valid foreign key
            conn.execute(
                """
                INSERT INTO mcp_servers (id, session_id, type, auto_connect, state)
                VALUES ('fs', ?, 'stdio', 1, 'disconnected')
                """,
                (new_session_id,),
            )
            conn.commit()

            # Verify MCP server inserted
            cursor = conn.execute(
                "SELECT id FROM mcp_servers WHERE session_id = ?", (new_session_id,)
            )
            assert cursor.fetchone(), "MCP server not inserted"

            # Test INSERT with invalid foreign key (should fail)
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO mcp_servers (id, session_id, type)
                    VALUES ('test', 'invalid-session-id', 'stdio')
                    """
                )
                conn.commit()

            print("✅ Foreign key integrity verified")

        finally:
            conn.close()

    def test_6_backward_compatibility(self, v1_database, temp_db):
        """Verify V1 code can still read basic session data after migration."""
        session_id = v1_database

        # Execute migration
        migrate_v1_to_v2(temp_db)

        conn = sqlite3.connect(temp_db)
        try:
            # Simulate V1 code reading session (ignores new V2 columns)
            cursor = conn.execute(
                """
                SELECT id, name, working_directory, created_at, last_accessed
                FROM sessions WHERE id = ?
                """,
                (session_id,),
            )
            row = cursor.fetchone()

            # V1 code expectations
            assert row, "Session not readable after migration"
            assert row[1] == "test-session", "Session name not readable"
            assert row[2] == os.getcwd(), "Working directory not readable"

            # Verify V1 queries still work
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            assert cursor.fetchone()[0] == 1, "Session count query failed"

            cursor = conn.execute("SELECT COUNT(*) FROM agents")
            assert cursor.fetchone()[0] == 1, "Agent count query failed"

            print("✅ Backward compatibility verified - V1 code can read migrated data")

        finally:
            conn.close()

    def test_7_idempotent_migration(self, v1_database, temp_db):
        """Verify migration can be run multiple times without errors (idempotency)."""
        # Execute migration twice
        stats1 = migrate_v1_to_v2(temp_db)
        stats2 = migrate_v1_to_v2(temp_db)

        # Second migration should be skipped
        assert stats2["statements_executed"] == 0, "Migration not idempotent"

        # Verify schema still valid after second run
        assert verify_migration(temp_db), "Schema invalid after second migration attempt"

        print("✅ Migration is idempotent")

    def test_8_rollback_migration(self, v1_database, temp_db):
        """Test migration rollback (for development/testing)."""
        # Execute migration
        migrate_v1_to_v2(temp_db)

        # Verify V2 tables exist
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mcp_servers'"
        )
        assert cursor.fetchone(), "mcp_servers table should exist before rollback"
        conn.close()

        # Rollback migration
        assert rollback_migration(temp_db), "Rollback failed"

        # Verify V2 tables removed
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mcp_servers'"
        )
        assert not cursor.fetchone(), "mcp_servers table should be removed after rollback"

        # Verify schema version reset
        cursor = conn.execute("SELECT MAX(version) FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == 1, "Schema version should be reset to 1 after rollback"
        conn.close()

        print("✅ Rollback successful")


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
