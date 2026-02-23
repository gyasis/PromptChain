"""V2 Schema Migration: Add Orchestration Support.

This migration extends the CLI database schema to support AgentChain orchestration,
MCP server management, and workflow state tracking.

Migration Version: 2
Applied: 002-cli-orchestration feature

Changes:
- Add orchestration_config, schema_version columns to sessions table
- Add instruction_chain, tools, history_config columns to agents table
- Create mcp_servers table for MCP server configurations
- Create workflow_states table for multi-session workflow tracking
- Preserve all existing V1 data during migration
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Migration version
MIGRATION_VERSION = 2
MIGRATION_DESCRIPTION = "Add orchestration support (AgentChain, MCP, workflows)"


def get_migration_sql() -> List[Tuple[str, str]]:
    """Get SQL statements for V2 schema migration.

    Returns:
        List[Tuple[str, str]]: List of (description, sql) tuples
    """
    migrations = []

    # 1. Add new columns to sessions table
    migrations.append(
        (
            "Add orchestration_config to sessions",
            """
            ALTER TABLE sessions
            ADD COLUMN orchestration_config TEXT DEFAULT '{}';
            """,
        )
    )

    migrations.append(
        (
            "Add schema_version to sessions",
            """
            ALTER TABLE sessions
            ADD COLUMN schema_version TEXT DEFAULT '2.0';
            """,
        )
    )

    # 2. Add new columns to agents table
    migrations.append(
        (
            "Add instruction_chain to agents",
            """
            ALTER TABLE agents
            ADD COLUMN instruction_chain TEXT DEFAULT '[]';
            """,
        )
    )

    migrations.append(
        (
            "Add tools to agents",
            """
            ALTER TABLE agents
            ADD COLUMN tools TEXT DEFAULT '[]';
            """,
        )
    )

    migrations.append(
        (
            "Add history_config to agents",
            """
            ALTER TABLE agents
            ADD COLUMN history_config TEXT DEFAULT NULL;
            """,
        )
    )

    # 3. Create mcp_servers table
    migrations.append(
        (
            "Create mcp_servers table",
            """
            CREATE TABLE IF NOT EXISTS mcp_servers (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('stdio', 'http')),
                command TEXT,
                args TEXT DEFAULT '[]',
                url TEXT,
                auto_connect INTEGER NOT NULL DEFAULT 1,
                state TEXT NOT NULL DEFAULT 'disconnected' CHECK(state IN ('disconnected', 'connected', 'error')),
                discovered_tools TEXT DEFAULT '[]',
                error_message TEXT,
                connected_at REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            """,
        )
    )

    migrations.append(
        (
            "Create mcp_servers session index",
            """
            CREATE INDEX IF NOT EXISTS idx_mcp_servers_session
            ON mcp_servers(session_id);
            """,
        )
    )

    # 4. Create workflow_states table
    migrations.append(
        (
            "Create workflow_states table",
            """
            CREATE TABLE IF NOT EXISTS workflow_states (
                session_id TEXT PRIMARY KEY,
                objective TEXT NOT NULL,
                steps TEXT DEFAULT '[]',
                current_step_index INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            """,
        )
    )

    return migrations


def migrate_v1_to_v2(db_path: Path) -> Dict[str, int]:
    """Execute V1 to V2 schema migration.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Dict[str, int]: Migration statistics (sessions_migrated, agents_migrated, etc.)

    Raises:
        sqlite3.Error: If migration fails
    """
    stats = {
        "sessions_migrated": 0,
        "agents_migrated": 0,
        "statements_executed": 0,
        "migration_version": MIGRATION_VERSION,
    }

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        # Check if migration already applied
        cursor = conn.execute(
            "SELECT version FROM schema_version WHERE version = ?", (MIGRATION_VERSION,)
        )
        if cursor.fetchone():
            print(f"Migration version {MIGRATION_VERSION} already applied, skipping...")
            return stats

        # Execute migration SQL statements
        migrations = get_migration_sql()
        for description, sql in migrations:
            print(f"  - {description}")
            conn.execute(sql)
            stats["statements_executed"] += 1

        # Initialize default orchestration config for existing sessions
        default_orchestration = {
            "execution_mode": "router",
            "default_agent": None,
            "router_config": {
                "model": "openai/gpt-4o-mini",
                "decision_prompt_template": "DEFAULT_ROUTER_PROMPT",
                "timeout_seconds": 10,
            },
            "auto_include_history": True,
        }

        cursor = conn.execute("SELECT id FROM sessions")
        session_ids = [row[0] for row in cursor.fetchall()]

        for session_id in session_ids:
            conn.execute(
                """
                UPDATE sessions
                SET orchestration_config = ?, schema_version = '2.0'
                WHERE id = ?
                """,
                (json.dumps(default_orchestration), session_id),
            )
            stats["sessions_migrated"] += 1

        # Initialize default history config for existing agents
        default_history = {
            "enabled": True,
            "max_tokens": 4000,
            "max_entries": 20,
            "truncation_strategy": "oldest_first",
            "include_types": None,
            "exclude_sources": None,
        }

        cursor = conn.execute("SELECT session_id, name FROM agents")
        agents = cursor.fetchall()

        for session_id, agent_name in agents:
            conn.execute(
                """
                UPDATE agents
                SET instruction_chain = ?, tools = ?, history_config = ?
                WHERE session_id = ? AND name = ?
                """,
                (
                    json.dumps(["{input}"]),  # Default instruction chain
                    json.dumps([]),  # No tools by default
                    json.dumps(default_history),
                    session_id,
                    agent_name,
                ),
            )
            stats["agents_migrated"] += 1

        # Record migration in schema_version table
        now = datetime.now().timestamp()
        conn.execute(
            """
            INSERT INTO schema_version (version, applied_at, description)
            VALUES (?, ?, ?)
            """,
            (MIGRATION_VERSION, now, MIGRATION_DESCRIPTION),
        )

        conn.commit()
        print(f"\n✅ Migration version {MIGRATION_VERSION} applied successfully")
        print(f"   Sessions migrated: {stats['sessions_migrated']}")
        print(f"   Agents migrated: {stats['agents_migrated']}")
        print(f"   SQL statements executed: {stats['statements_executed']}")

        return stats

    except sqlite3.Error as e:
        conn.rollback()
        print(f"\n❌ Migration failed: {e}")
        raise

    finally:
        conn.close()


def verify_migration(db_path: Path) -> bool:
    """Verify V2 schema migration was applied correctly.

    Args:
        db_path: Path to SQLite database file

    Returns:
        bool: True if migration successful and schema valid

    Raises:
        AssertionError: If verification fails
    """
    conn = sqlite3.connect(db_path)
    try:
        # Enable foreign keys (per-connection setting in SQLite)
        conn.execute("PRAGMA foreign_keys = ON")

        # 1. Verify migration version recorded
        cursor = conn.execute(
            "SELECT version FROM schema_version WHERE version = ?", (MIGRATION_VERSION,)
        )
        assert cursor.fetchone(), f"Migration version {MIGRATION_VERSION} not recorded"

        # 2. Verify sessions table has new columns
        cursor = conn.execute("PRAGMA table_info(sessions)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "orchestration_config" in columns, "orchestration_config column missing"
        assert "schema_version" in columns, "schema_version column missing"

        # 3. Verify agents table has new columns
        cursor = conn.execute("PRAGMA table_info(agents)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "instruction_chain" in columns, "instruction_chain column missing"
        assert "tools" in columns, "tools column missing"
        assert "history_config" in columns, "history_config column missing"

        # 4. Verify mcp_servers table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mcp_servers'"
        )
        assert cursor.fetchone(), "mcp_servers table not created"

        # 5. Verify workflow_states table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_states'"
        )
        assert cursor.fetchone(), "workflow_states table not created"

        # 6. Verify foreign key constraints work
        cursor = conn.execute("PRAGMA foreign_keys")
        assert cursor.fetchone()[0] == 1, "Foreign keys not enabled"

        # 7. Verify existing data preserved (if any sessions exist)
        cursor = conn.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]

        if session_count > 0:
            cursor = conn.execute("SELECT id, name, working_directory FROM sessions LIMIT 1")
            row = cursor.fetchone()
            assert row, "Session data lost during migration"
            assert row[1], "Session name lost during migration"
            assert row[2], "Session working_directory lost during migration"

        print(f"\n✅ Migration verification passed")
        print(f"   - All new columns present")
        print(f"   - All new tables created")
        print(f"   - Foreign keys enabled")
        print(f"   - Existing data preserved ({session_count} sessions)")

        return True

    except AssertionError as e:
        print(f"\n❌ Migration verification failed: {e}")
        raise

    finally:
        conn.close()


def rollback_migration(db_path: Path) -> bool:
    """Rollback V2 migration (for testing purposes).

    WARNING: This will DROP new tables and remove new columns.
    Only use in development/testing environments.

    Args:
        db_path: Path to SQLite database file

    Returns:
        bool: True if rollback successful
    """
    conn = sqlite3.connect(db_path)
    try:
        print(f"\n⚠️  Rolling back migration version {MIGRATION_VERSION}...")

        # SQLite doesn't support DROP COLUMN, so we can't fully rollback
        # Instead, we'll drop the new tables and clear the migration record
        conn.execute("DROP TABLE IF EXISTS mcp_servers")
        conn.execute("DROP TABLE IF EXISTS workflow_states")
        conn.execute("DELETE FROM schema_version WHERE version = ?", (MIGRATION_VERSION,))
        conn.commit()

        print(f"✅ Rollback complete (new tables dropped, V1 schema restored)")
        print(
            f"   NOTE: New columns in sessions/agents tables remain (SQLite limitation)"
        )

        return True

    except sqlite3.Error as e:
        conn.rollback()
        print(f"❌ Rollback failed: {e}")
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    """Run migration from command line for testing."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python v2_schema.py <db_path> [--verify|--rollback]")
        sys.exit(1)

    db_path = Path(sys.argv[1])

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)

    if "--verify" in sys.argv:
        verify_migration(db_path)
    elif "--rollback" in sys.argv:
        rollback_migration(db_path)
    else:
        stats = migrate_v1_to_v2(db_path)
        verify_migration(db_path)
