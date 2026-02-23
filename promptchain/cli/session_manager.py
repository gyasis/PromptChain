"""Session manager for PromptChain CLI.

This module handles session lifecycle operations including creation, persistence,
loading, and auto-save functionality.

Includes error logging to JSONL session logs (T143).
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain.observability import track_task

from .models import Agent, Message, Session
from .models.agent_config import HistoryConfig
from .utils.error_logger import ErrorLogger


class SessionManager:
    """Manages session lifecycle and persistence.

    Responsibilities:
    - Create and initialize new sessions
    - Load existing sessions from SQLite
    - Save session state to SQLite and JSONL
    - Auto-save based on message count and time interval
    - List and delete sessions
    """

    def __init__(self, sessions_dir: Path, db_path: Optional[Path] = None):
        """Initialize session manager.

        Args:
            sessions_dir: Directory to store session files
            db_path: Path to SQLite database (default: sessions_dir/sessions.db)
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path or (self.sessions_dir / "sessions.db")

        # Initialize database schema (will be implemented in T016)
        self._init_database()

    def _init_database(self):
        """Initialize database schema.

        Creates tables if they don't exist and ensures schema version is tracked.
        Uses schema.sql for table definitions.

        T014: Implements automatic V1→V2 migration detection and execution.
        """
        # Read schema SQL file
        schema_path = Path(__file__).parent / "schema.sql"

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r") as f:
            schema_sql = f.read()

        # Execute schema creation
        conn = sqlite3.connect(self.db_path)
        try:
            # Enable foreign keys (required for CASCADE DELETE)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(schema_sql)
            conn.commit()
        finally:
            conn.close()

        # T014: Check if V2 migration is needed
        self._check_and_migrate_v2()

        # 003-T008: Check if V3 migration is needed (multi-agent communication)
        self._check_and_migrate_v3()

    def get_schema_version(self) -> int:
        """Get current database schema version.

        Returns:
            int: Current schema version number
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT MAX(version) FROM schema_version")
            version = cursor.fetchone()[0]
            return version if version else 0
        finally:
            conn.close()

    def _check_and_migrate_v2(self):
        """Check if V2 migration is needed and execute if necessary (T014).

        Detects V1 schema by checking for missing orchestration_config column
        in sessions table. If V1 detected, automatically runs V2 migration.
        """
        current_version = self.get_schema_version()

        # Check if V2 migration already applied
        if current_version >= 2:
            return  # Already on V2 or later

        # Check if sessions table has orchestration_config column (V2 indicator)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("PRAGMA table_info(sessions)")
            columns = {row[1] for row in cursor.fetchall()}

            # V1 detected: missing V2 columns
            if "orchestration_config" not in columns:
                print("\n🔄 Detecting V1 schema, auto-migrating to V2...")

                # Import and run migration
                from .migrations.v2_schema import migrate_v1_to_v2

                migrate_v1_to_v2(self.db_path)
                print("✅ V2 migration complete\n")

        finally:
            conn.close()

    def _check_and_migrate_v3(self):
        """Check if V3 migration is needed and execute if necessary (003-T008).

        Detects V2 schema by checking for missing task_queue table.
        If V2 detected, automatically creates V3 multi-agent communication tables.
        """
        current_version = self.get_schema_version()

        # Check if V3 migration already applied
        if current_version >= 3:
            return  # Already on V3 or later

        # Check if task_queue table exists (V3 indicator)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='task_queue'"
            )

            if not cursor.fetchone():
                print("\n🔄 Detecting V2 schema, auto-migrating to V3 (multi-agent communication)...")

                # Create V3 tables
                v3_migration_sql = """
                -- Task queue for agent delegation (FR-006 to FR-010)
                CREATE TABLE IF NOT EXISTS task_queue (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source_agent TEXT NOT NULL,
                    target_agent TEXT NOT NULL,
                    priority TEXT NOT NULL DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high')),
                    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'failed')),
                    context_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    result_json TEXT,
                    error_message TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_task_queue_session ON task_queue(session_id);
                CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(session_id, status);
                CREATE INDEX IF NOT EXISTS idx_task_queue_target ON task_queue(session_id, target_agent, status);
                CREATE INDEX IF NOT EXISTS idx_task_queue_priority ON task_queue(session_id, priority DESC, created_at ASC);

                -- Blackboard for shared data (FR-011 to FR-015)
                CREATE TABLE IF NOT EXISTS blackboard (
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    written_by TEXT NOT NULL,
                    written_at REAL NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (session_id, key),
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_blackboard_session ON blackboard(session_id);
                CREATE INDEX IF NOT EXISTS idx_blackboard_writer ON blackboard(session_id, written_by);

                -- Workflow state for multi-agent workflows (FR-021 to FR-025)
                CREATE TABLE IF NOT EXISTS workflow_state (
                    workflow_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    stage TEXT NOT NULL DEFAULT 'planning' CHECK (stage IN ('planning', 'execution', 'review', 'complete')),
                    agents_involved_json TEXT NOT NULL DEFAULT '[]',
                    completed_tasks_json TEXT NOT NULL DEFAULT '[]',
                    current_task TEXT,
                    context_json TEXT NOT NULL DEFAULT '{}',
                    started_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_session ON workflow_state(session_id);
                CREATE INDEX IF NOT EXISTS idx_workflow_stage ON workflow_state(session_id, stage);

                -- Message log for agent communication (FR-016 to FR-020)
                CREATE TABLE IF NOT EXISTS message_log (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    receiver TEXT NOT NULL,
                    message_type TEXT NOT NULL CHECK (message_type IN ('request', 'response', 'broadcast', 'delegation', 'status')),
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    timestamp REAL NOT NULL,
                    delivered INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_message_log_session ON message_log(session_id);
                CREATE INDEX IF NOT EXISTS idx_message_log_receiver ON message_log(session_id, receiver, delivered);
                CREATE INDEX IF NOT EXISTS idx_message_log_type ON message_log(session_id, message_type);
                """

                conn.executescript(v3_migration_sql)

                # Record V3 migration
                conn.execute(
                    "INSERT OR IGNORE INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                    (3, time.time(), "Multi-agent communication tables")
                )
                conn.commit()

                print("✅ V3 migration complete\n")

        finally:
            conn.close()

    def apply_migration(self, version: int, description: str, sql: str):
        """Apply a database migration.

        Args:
            version: Migration version number
            description: Migration description
            sql: Migration SQL statements
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Check if migration already applied
            cursor = conn.execute(
                "SELECT version FROM schema_version WHERE version = ?", (version,)
            )
            if cursor.fetchone():
                return  # Already applied

            # Apply migration
            conn.executescript(sql)

            # Record migration
            import time

            conn.execute(
                "INSERT INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                (version, time.time(), description),
            )

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Migration {version} failed: {e}")
        finally:
            conn.close()

    # === Core Session Operations (T023-T026) ===

    def create_session(
        self,
        name: str,
        working_directory: Optional[Path] = None,
        default_model: Optional[str] = None,
        mcp_servers: Optional[List] = None,
    ) -> Session:
        """Create a new session (T023, T062).

        Args:
            name: Session name (1-64 chars, alphanumeric+dashes/underscores)
            working_directory: Working directory for session (default: cwd)
            default_model: Default LLM model (default: "openai/gpt-4")
            mcp_servers: List of MCPServerConfig objects to add to session (T062)

        Returns:
            Session: Created session object

        Raises:
            ValueError: If session name already exists or validation fails
        """
        # Use cwd if no working directory provided
        if working_directory is None:
            working_directory = Path.cwd()
        else:
            working_directory = Path(working_directory)

        # Set default model
        if default_model is None:
            default_model = "openai/gpt-4.1-mini-2025-04-14"

        # Check if session name already exists
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT id FROM sessions WHERE name = ?", (name,))
            if cursor.fetchone():
                raise ValueError(f"Session '{name}' already exists")
        finally:
            conn.close()

        # Generate UUID for session
        session_id = str(uuid.uuid4())
        now = time.time()

        # Create default agent with agentic capabilities (AgenticStepProcessor)
        # instruction_chain triggers multi-hop reasoning in app.py:1040
        default_agent = Agent(
            name="default",
            model_name=default_model,
            description="Default agent with agentic reasoning and tool access",
            created_at=now,
            instruction_chain=[
                "You are an EXECUTION agent, not an explanation agent. Your job is to COMPLETE tasks by USING TOOLS, not by explaining how to do them.\n\n"
                "CRITICAL RULES:\n"
                "1. NEVER explain what tools to use - ACTUALLY USE THEM\n"
                "2. For code execution: ALWAYS use sandbox_provision_uv or sandbox_provision_docker to create isolated environments\n"
                "3. For file operations: ALWAYS use file_write, file_read, file_edit tools\n"
                "4. For terminal commands: ALWAYS use terminal_execute tool\n"
                "5. Use multi-hop reasoning: Call multiple tools in sequence to complete complex tasks\n"
                "6. If a task requires packages: provision sandbox → install packages → execute code → return results\n"
                "7. ONLY respond with results after completing the task using tools\n\n"
                "Available sandbox tools: sandbox_provision_uv, sandbox_provision_docker, sandbox_execute, sandbox_list, sandbox_cleanup\n"
                "Example: User asks for pygame script → provision_uv sandbox → install pygame → write script → execute → show output\n\n"
                "COMPLETE the user's request by EXECUTING it with tools. DO NOT explain how to do it."
            ],
            tools=[  # All 19 registered tools available
                "create_sandbox", "list_sandboxes", "delete_sandbox", "execute_in_sandbox", "get_sandbox_status",
                "code_reader", "code_writer", "code_editor", "code_analyzer", "code_search",
                "code_refactor", "code_test_generator", "code_doc_generator", "code_complexity",
                "code_security_scan", "code_formatter", "dependency_analyzer", "ast_analyzer", "linter"
            ],
            history_config=HistoryConfig(
                enabled=True,
                max_tokens=8000,  # Generous limit for agentic reasoning
                max_entries=50,
                truncation_strategy="oldest_first"
            )
        )

        # Create session object (will validate inputs)
        session = Session(
            id=session_id,
            name=name,
            created_at=now,
            last_accessed=now,
            working_directory=working_directory,
            active_agent="default",
            default_model=default_model,
            agents={"default": default_agent},
            mcp_servers=mcp_servers if mcp_servers else [],  # T062: Add MCP servers
        )

        # Create session directory
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # T062: Auto-connect MCP servers with auto_connect=True
        if mcp_servers:
            from promptchain.cli.utils.mcp_manager import MCPManager
            import asyncio
            import nest_asyncio

            # Allow nested event loops (needed for async contexts)
            nest_asyncio.apply()

            mcp_manager = MCPManager(session)

            # Connect servers with auto_connect=True
            async def auto_connect_servers():
                for server in mcp_servers:
                    if server.auto_connect:
                        try:
                            await mcp_manager.connect_server(server.id)
                        except Exception as e:
                            # Server will be marked with error state, continue with others
                            pass

            # Run auto-connect synchronously (works in both sync and async contexts)
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(auto_connect_servers())

        # Create messages file (empty JSONL)
        messages_file = session_dir / "messages.jsonl"
        messages_file.touch()

        # Insert session into database
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO sessions (
                    id, name, created_at, last_accessed, working_directory,
                    active_agent, default_model, auto_save_enabled, auto_save_interval,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.name,
                    session.created_at,
                    session.last_accessed,
                    str(session.working_directory),
                    session.active_agent,
                    session.default_model,
                    1 if session.auto_save_enabled else 0,
                    session.auto_save_interval,
                    json.dumps(session.metadata),
                ),
            )

            # Insert default agent
            conn.execute(
                """
                INSERT INTO agents (
                    session_id, name, model_name, description, created_at,
                    last_used, usage_count, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    default_agent.name,
                    default_agent.model_name,
                    default_agent.description,
                    default_agent.created_at,
                    default_agent.last_used,
                    default_agent.usage_count,
                    json.dumps(default_agent.metadata),
                ),
            )

            conn.commit()
        except Exception as e:
            conn.rollback()

            # Log error to session's errors.jsonl (T143)
            try:
                error_logger = ErrorLogger(session_dir)
                error_logger.log_error(
                    error=e,
                    context="creating session",
                    metadata={"session_name": name, "working_directory": str(working_directory)},
                )
            except Exception as log_error:
                # BUG-012 fix: Log the logging failure to stderr as last resort
                import sys
                print(
                    f"WARNING: Failed to log session creation error to file: {log_error}",
                    file=sys.stderr
                )
                print(f"Original error was: {e}", file=sys.stderr)

            raise RuntimeError(f"Failed to create session: {e}")
        finally:
            conn.close()

        # ✅ Phase 3: Initialize ActivityLogger for the session
        # Creates activity logging directory and ActivityLogger instance
        activity_log_dir = session_dir / "activity_logs"
        activity_log_dir.mkdir(exist_ok=True)

        try:
            from promptchain.cli.activity_logger import ActivityLogger

            session._activity_logger = ActivityLogger(
                session_name=session.name,
                log_dir=activity_log_dir,
                db_path=session_dir / "activities.db",
                enable_console=False  # CLI will handle console output
            )
        except Exception as e:
            # Log error but don't fail session creation
            try:
                error_logger = ErrorLogger(session_dir)
                error_logger.log_error(
                    error=e,
                    context="initializing activity logger",
                    metadata={"session_name": name},
                )
            except Exception:
                pass

        return session

    def load_session(self, name_or_id: str) -> Session:
        """Load an existing session (T024, T062, T070).

        Args:
            name_or_id: Session name or ID to load

        Returns:
            Session: Loaded session object

        Raises:
            ValueError: If session not found
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Load session metadata - try by ID first, then by name (T070)
            cursor = conn.execute(
                """
                SELECT id, name, created_at, last_accessed, working_directory,
                       active_agent, default_model, auto_save_enabled,
                       auto_save_interval, metadata_json
                FROM sessions WHERE id = ? OR name = ?
                """,
                (name_or_id, name_or_id),
            )
            row = cursor.fetchone()

            if not row:
                raise ValueError(f"Session '{name_or_id}' not found")

            # Load agents (V2 schema with orchestration fields - T063)
            agent_cursor = conn.execute(
                """
                SELECT name, model_name, description, created_at, last_used,
                       usage_count, metadata_json, instruction_chain, tools, history_config
                FROM agents WHERE session_id = ?
                """,
                (row[0],),
            )
            agents = {}
            for agent_row in agent_cursor.fetchall():
                # Deserialize V2 fields from JSON
                instruction_chain = json.loads(agent_row[7]) if agent_row[7] else []
                tools = json.loads(agent_row[8]) if agent_row[8] else []
                history_config_json = agent_row[9]
                history_config = None
                if history_config_json:
                    from .models.agent_config import HistoryConfig
                    history_config_dict = json.loads(history_config_json)
                    history_config = HistoryConfig(**history_config_dict)

                agent = Agent(
                    name=agent_row[0],
                    model_name=agent_row[1],
                    description=agent_row[2],
                    created_at=agent_row[3],
                    last_used=agent_row[4],
                    usage_count=agent_row[5],
                    metadata=json.loads(agent_row[6]),
                    instruction_chain=instruction_chain,
                    tools=tools,
                    history_config=history_config,
                )
                agents[agent.name] = agent

            # T062: Load MCP servers from database
            mcp_servers = self.load_mcp_servers(row[0])

            # Create session object
            session = Session(
                id=row[0],
                name=row[1],
                created_at=row[2],
                last_accessed=row[3],
                working_directory=Path(row[4]),
                active_agent=row[5],
                default_model=row[6],
                auto_save_enabled=bool(row[7]),
                auto_save_interval=row[8],
                metadata=json.loads(row[9]),
                agents=agents,
                mcp_servers=mcp_servers,  # T062: Add loaded MCP servers
            )

            # Load messages from JSONL
            messages_file = self.sessions_dir / session.id / "messages.jsonl"
            if messages_file.exists():
                with open(messages_file, "r") as f:
                    for line in f:
                        if line.strip():
                            session.messages.append(Message.from_jsonl(line))

            # ✅ Phase 3: Initialize ActivityLogger for loaded session
            # Reconnects to existing activity log database
            session_dir = self.sessions_dir / session.id
            activity_log_dir = session_dir / "activity_logs"
            activity_db_path = session_dir / "activities.db"

            # Create activity log directory if it doesn't exist (migration support)
            activity_log_dir.mkdir(exist_ok=True)

            try:
                from promptchain.cli.activity_logger import ActivityLogger

                session._activity_logger = ActivityLogger(
                    session_name=session.name,
                    log_dir=activity_log_dir,
                    db_path=activity_db_path,
                    enable_console=False  # CLI will handle console output
                )
            except Exception as e:
                # Log error but don't fail session load
                try:
                    error_logger = ErrorLogger(session_dir)
                    error_logger.log_error(
                        error=e,
                        context="initializing activity logger on load",
                        metadata={"session_id": session.id, "session_name": session.name},
                    )
                except Exception:
                    pass

            # T062: Auto-connect MCP servers with auto_connect=True
            if session.mcp_servers:
                from promptchain.cli.utils.mcp_manager import MCPManager
                import asyncio
                import nest_asyncio

                # Allow nested event loops (needed for async contexts)
                nest_asyncio.apply()

                mcp_manager = MCPManager(session)

                # Connect servers with auto_connect=True
                async def auto_connect_servers():
                    for server in session.mcp_servers:
                        if server.auto_connect:
                            try:
                                await mcp_manager.connect_server(server.id)
                            except Exception as e:
                                # Server will be marked with error state, continue with others
                                pass

                # Run auto-connect synchronously (works in both sync and async contexts)
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                loop.run_until_complete(auto_connect_servers())

            return session

        finally:
            conn.close()

    def save_session(self, session: Session):
        """Save session state (T026, T062).

        Args:
            session: Session object to save

        Saves:
            - Session metadata to SQLite
            - Agents to SQLite
            - MCP servers to SQLite (T062)
            - Messages to JSONL (append-only)
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Update session metadata
            conn.execute(
                """
                UPDATE sessions
                SET last_accessed = ?,
                    active_agent = ?,
                    default_model = ?,
                    auto_save_enabled = ?,
                    auto_save_interval = ?,
                    metadata_json = ?
                WHERE id = ?
                """,
                (
                    session.last_accessed,
                    session.active_agent,
                    session.default_model,
                    1 if session.auto_save_enabled else 0,
                    session.auto_save_interval,
                    json.dumps(session.metadata),
                    session.id,
                ),
            )

            # Update agents (delete and re-insert for simplicity - V2 schema - T063)
            conn.execute("DELETE FROM agents WHERE session_id = ?", (session.id,))
            for agent in session.agents.values():
                # Serialize V2 fields to JSON
                instruction_chain_json = json.dumps(agent.instruction_chain)
                tools_json = json.dumps(agent.tools)
                history_config_json = None
                if agent.history_config:
                    history_config_json = json.dumps({
                        "enabled": agent.history_config.enabled,
                        "max_tokens": agent.history_config.max_tokens,
                        "max_entries": agent.history_config.max_entries,
                        "truncation_strategy": agent.history_config.truncation_strategy,
                        "include_types": agent.history_config.include_types,
                        "exclude_sources": agent.history_config.exclude_sources,
                    })

                conn.execute(
                    """
                    INSERT INTO agents (
                        session_id, name, model_name, description, created_at,
                        last_used, usage_count, metadata_json,
                        instruction_chain, tools, history_config
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.id,
                        agent.name,
                        agent.model_name,
                        agent.description,
                        agent.created_at,
                        agent.last_used,
                        agent.usage_count,
                        json.dumps(agent.metadata),
                        instruction_chain_json,
                        tools_json,
                        history_config_json,
                    ),
                )

            conn.commit()

            # T062: Save MCP servers
            if session.mcp_servers:
                self.save_mcp_servers(session.id, session.mcp_servers)
        except Exception as e:
            conn.rollback()

            # Log error to session's errors.jsonl (T143)
            try:
                session_dir = self.sessions_dir / session.id
                error_logger = ErrorLogger(session_dir)
                error_logger.log_error(
                    error=e,
                    context="saving session",
                    metadata={
                        "session_id": session.id,
                        "session_name": session.name,
                        "message_count": len(session.messages),
                    },
                )
            except Exception:
                pass  # Don't fail on logging error

            raise RuntimeError(f"Failed to save session: {e}")
        finally:
            conn.close()

        # Save messages to JSONL (append-only)
        messages_file = self.sessions_dir / session.id / "messages.jsonl"
        try:
            with open(messages_file, "w") as f:
                for message in session.messages:
                    f.write(message.to_jsonl() + "\n")
        except Exception as e:
            # Log file writing error
            try:
                session_dir = self.sessions_dir / session.id
                error_logger = ErrorLogger(session_dir)
                error_logger.log_error(
                    error=e,
                    context="saving messages to JSONL",
                    metadata={"session_id": session.id, "messages_file": str(messages_file)},
                )
            except Exception:
                pass
            raise RuntimeError(f"Failed to save messages: {e}")

    def list_sessions(self) -> List[Session]:
        """List all sessions (lightweight, metadata only).

        Returns:
            List[Session]: List of session objects (without messages)
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, name, created_at, last_accessed, working_directory,
                       active_agent, default_model, auto_save_enabled,
                       auto_save_interval, metadata_json
                FROM sessions
                ORDER BY last_accessed DESC
                """
            )

            sessions = []
            for row in cursor.fetchall():
                session = Session(
                    id=row[0],
                    name=row[1],
                    created_at=row[2],
                    last_accessed=row[3],
                    working_directory=Path(row[4]),
                    active_agent=row[5],
                    default_model=row[6],
                    auto_save_enabled=bool(row[7]),
                    auto_save_interval=row[8],
                    metadata=json.loads(row[9]),
                    agents={},  # Don't load agents for list view
                    messages=[],  # Don't load messages for list view
                )
                sessions.append(session)

            return sessions

        finally:
            conn.close()

    # T028: Agent Configuration Persistence
    def save_agent_configs(self, session_id: str, agents: Dict[str, "Agent"]) -> None:
        """Save agent configurations to database (T028).

        Args:
            session_id: Session identifier
            agents: Dictionary of agent name to Agent object

        Raises:
            RuntimeError: If database operation fails
        """
        from .models.agent_config import Agent

        conn = sqlite3.connect(self.db_path)
        try:
            # Delete existing agents for this session
            conn.execute("DELETE FROM agents WHERE session_id = ?", (session_id,))

            # Insert all agents with V2 schema fields
            for agent_name, agent in agents.items():
                # Serialize instruction_chain and tools to JSON
                instruction_chain_json = json.dumps(agent.instruction_chain)
                tools_json = json.dumps(agent.tools)

                # Serialize history_config to JSON
                history_config_json = None
                if agent.history_config:
                    history_config_json = json.dumps({
                        "enabled": agent.history_config.enabled,
                        "max_tokens": agent.history_config.max_tokens,
                        "max_entries": agent.history_config.max_entries,
                        "truncation_strategy": agent.history_config.truncation_strategy,
                        "include_types": agent.history_config.include_types,
                        "exclude_sources": agent.history_config.exclude_sources,
                    })

                conn.execute(
                    """
                    INSERT INTO agents (
                        session_id, name, model_name, description, created_at,
                        last_used, usage_count, metadata_json,
                        instruction_chain, tools, history_config
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        agent.name,
                        agent.model_name,
                        agent.description,
                        agent.created_at,
                        agent.last_used,
                        agent.usage_count,
                        json.dumps(agent.metadata),
                        instruction_chain_json,
                        tools_json,
                        history_config_json,
                    ),
                )

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to save agent configs: {e}")
        finally:
            conn.close()

    def load_agent_configs(self, session_id: str) -> Dict[str, "Agent"]:
        """Load agent configurations from database (T028).

        Args:
            session_id: Session identifier

        Returns:
            Dict[str, Agent]: Dictionary of agent name to Agent object

        Raises:
            RuntimeError: If database operation fails
        """
        from .models.agent_config import Agent, HistoryConfig

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT name, model_name, description, created_at, last_used,
                       usage_count, metadata_json, instruction_chain, tools, history_config
                FROM agents
                WHERE session_id = ?
                """,
                (session_id,),
            )

            agents = {}
            for row in cursor.fetchall():
                # Deserialize JSON fields
                metadata = json.loads(row[6]) if row[6] else {}
                instruction_chain = json.loads(row[7]) if row[7] else []
                tools = json.loads(row[8]) if row[8] else []

                # Deserialize history_config
                history_config = None
                if row[9]:
                    history_data = json.loads(row[9])
                    history_config = HistoryConfig(
                        enabled=history_data.get("enabled", True),
                        max_tokens=history_data.get("max_tokens", 4000),
                        max_entries=history_data.get("max_entries", 20),
                        truncation_strategy=history_data.get("truncation_strategy", "oldest_first"),
                        include_types=history_data.get("include_types"),
                        exclude_sources=history_data.get("exclude_sources"),
                    )

                agent = Agent(
                    name=row[0],
                    model_name=row[1],
                    description=row[2],
                    created_at=row[3],
                    last_used=row[4],
                    usage_count=row[5],
                    metadata=metadata,
                    instruction_chain=instruction_chain,
                    tools=tools,
                    history_config=history_config,
                )
                agents[agent.name] = agent

            return agents

        finally:
            conn.close()

    # T029: MCP Server Configuration Persistence
    def save_mcp_servers(self, session_id: str, servers: List["MCPServerConfig"]) -> None:
        """Save MCP server configurations to database (T029).

        Args:
            session_id: Session identifier
            servers: List of MCPServerConfig objects

        Raises:
            RuntimeError: If database operation fails
        """
        from .models.mcp_config import MCPServerConfig

        conn = sqlite3.connect(self.db_path)
        try:
            # Delete existing MCP servers for this session
            conn.execute("DELETE FROM mcp_servers WHERE session_id = ?", (session_id,))

            # Insert all MCP servers
            for server in servers:
                conn.execute(
                    """
                    INSERT INTO mcp_servers (
                        session_id, id, type, command, args, url,
                        auto_connect, state, discovered_tools, error_message, connected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        server.id,
                        server.type,
                        server.command,
                        json.dumps(server.args),
                        server.url,
                        1 if server.auto_connect else 0,
                        server.state,
                        json.dumps(server.discovered_tools),
                        server.error_message,
                        server.connected_at,
                    ),
                )

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to save MCP servers: {e}")
        finally:
            conn.close()

    def load_mcp_servers(self, session_id: str) -> List["MCPServerConfig"]:
        """Load MCP server configurations from database (T029).

        Args:
            session_id: Session identifier

        Returns:
            List[MCPServerConfig]: List of MCP server configurations

        Raises:
            RuntimeError: If database operation fails
        """
        from .models.mcp_config import MCPServerConfig

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT id, type, command, args, url, auto_connect,
                       state, discovered_tools, error_message, connected_at
                FROM mcp_servers
                WHERE session_id = ?
                """,
                (session_id,),
            )

            servers = []
            for row in cursor.fetchall():
                server = MCPServerConfig(
                    id=row[0],
                    type=row[1],
                    command=row[2],
                    args=json.loads(row[3]) if row[3] else [],
                    url=row[4],
                    auto_connect=bool(row[5]),
                    state=row[6],
                    discovered_tools=json.loads(row[7]) if row[7] else [],
                    error_message=row[8],
                    connected_at=row[9],
                )
                servers.append(server)

            return servers

        finally:
            conn.close()

    # T030: Workflow State Persistence
    def save_workflow(self, session_id: str, workflow: "WorkflowState") -> None:
        """Save workflow state to database (T030).

        Args:
            session_id: Session identifier
            workflow: WorkflowState object

        Raises:
            RuntimeError: If database operation fails
        """
        from .models.workflow import WorkflowState

        conn = sqlite3.connect(self.db_path)
        try:
            # Serialize workflow steps to JSON
            steps_json = json.dumps([
                {
                    "description": step.description,
                    "status": step.status,
                    "agent_name": step.agent_name,
                    "started_at": step.started_at,
                    "completed_at": step.completed_at,
                    "result": step.result,
                    "error_message": step.error_message,
                    "retry_count": step.retry_count,
                }
                for step in workflow.steps
            ])

            # Check if workflow already exists
            cursor = conn.execute(
                "SELECT 1 FROM workflow_states WHERE session_id = ?", (session_id,)
            )
            exists = cursor.fetchone() is not None

            if exists:
                # Update existing workflow
                conn.execute(
                    """
                    UPDATE workflow_states
                    SET objective = ?, steps = ?, current_step_index = ?,
                        updated_at = ?, completed_at = ?, metadata = ?
                    WHERE session_id = ?
                    """,
                    (
                        workflow.objective,
                        steps_json,
                        workflow.current_step_index,
                        workflow.updated_at,
                        workflow.completed_at,
                        json.dumps(workflow.metadata),
                        session_id,
                    ),
                )
            else:
                # Insert new workflow
                conn.execute(
                    """
                    INSERT INTO workflow_states (
                        session_id, objective, steps, current_step_index,
                        created_at, updated_at, completed_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        workflow.objective,
                        steps_json,
                        workflow.current_step_index,
                        workflow.created_at,
                        workflow.updated_at,
                        workflow.completed_at,
                        json.dumps(workflow.metadata),
                    ),
                )

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to save workflow: {e}")
        finally:
            conn.close()

    def load_workflow(self, session_id: str) -> Optional["WorkflowState"]:
        """Load workflow state from database (T030).

        Args:
            session_id: Session identifier

        Returns:
            Optional[WorkflowState]: Workflow state or None if not found

        Raises:
            RuntimeError: If database operation fails
        """
        from .models.workflow import WorkflowState, WorkflowStep

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT objective, steps, current_step_index, created_at,
                       updated_at, completed_at, metadata
                FROM workflow_states
                WHERE session_id = ?
                """,
                (session_id,),
            )

            row = cursor.fetchone()
            if row is None:
                return None

            # Deserialize steps from JSON
            steps_data = json.loads(row[1])
            steps = [
                WorkflowStep(
                    description=step["description"],
                    status=step["status"],
                    agent_name=step.get("agent_name"),
                    started_at=step.get("started_at"),
                    completed_at=step.get("completed_at"),
                    result=step.get("result"),
                    error_message=step.get("error_message"),
                    retry_count=step.get("retry_count", 0),
                )
                for step in steps_data
            ]

            workflow = WorkflowState(
                objective=row[0],
                steps=steps,
                current_step_index=row[2],
                created_at=row[3],
                updated_at=row[4],
                completed_at=row[5],
                metadata=json.loads(row[6]) if row[6] else {},
            )

            return workflow

        finally:
            conn.close()

    def resume_workflow(self, session_id: str) -> Optional["WorkflowState"]:
        """Resume workflow from last incomplete step (T030).

        Args:
            session_id: Session identifier

        Returns:
            Optional[WorkflowState]: Workflow ready to resume, or None if completed

        Raises:
            RuntimeError: If database operation fails
        """
        workflow = self.load_workflow(session_id)

        if workflow is None:
            return None

        # Check if workflow is already completed
        if workflow.completed_at is not None:
            return None

        # Reset any in_progress steps to pending (crashed recovery)
        for step in workflow.steps:
            if step.status == "in_progress":
                step.status = "pending"
                step.started_at = None

        # Find first pending step
        for i, step in enumerate(workflow.steps):
            if step.status == "pending":
                workflow.current_step_index = i
                break

        # Update workflow timestamp
        from datetime import datetime
        workflow.updated_at = datetime.now().timestamp()

        return workflow

    # T043: Router Decision Logging
    def log_router_decision(
        self,
        session_id: str,
        user_query: str,
        selected_agent: str,
        rationale: Optional[str] = None,
        confidence: Optional[float] = None,
        all_agents: Optional[List[str]] = None,
    ) -> None:
        """Log router agent selection decision to JSONL (T043).

        Args:
            session_id: Session identifier
            user_query: User's input query
            selected_agent: Name of selected agent
            rationale: Optional rationale for selection
            confidence: Optional confidence score (0.0-1.0)
            all_agents: List of all available agents at decision time

        Logs to: ~/.promptchain/sessions/<session-id>/history.jsonl
        """
        from datetime import datetime

        # Create router decision entry
        decision_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "router_decision",
            "session_id": session_id,
            "user_query": user_query,
            "selected_agent": selected_agent,
            "rationale": rationale,
            "confidence": confidence,
            "available_agents": all_agents,
        }

        # Write to history.jsonl file (append mode)
        session_dir = self.sessions_dir / session_id
        history_log_path = session_dir / "history.jsonl"

        try:
            # Ensure session directory exists
            session_dir.mkdir(parents=True, exist_ok=True)

            # Append to JSONL file
            with open(history_log_path, "a") as f:
                json.dump(decision_entry, f)
                f.write("\n")
        except Exception as log_error:
            # If logging fails, write to stderr but don't crash
            import sys

            print(
                f"ERROR: Failed to log router decision to {history_log_path}: {log_error}",
                file=sys.stderr,
            )

    # T044: Router Failure Logging
    def log_router_failure(
        self,
        session_id: str,
        error_type: str,
        reason: str,
        user_query: str,
        fallback_agent: Optional[str] = None,
    ) -> None:
        """Log router failure event to JSONL (T044).

        Args:
            session_id: Session identifier
            error_type: Type of failure (timeout, invalid_json, agent_not_found, etc.)
            reason: Detailed failure reason
            user_query: User query that triggered failure
            fallback_agent: Fallback agent used (if any)

        Logs to: ~/.promptchain/sessions/<session-id>/history.jsonl
        """
        from datetime import datetime

        # Create router failure entry
        failure_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "router_failure",
            "session_id": session_id,
            "error_type": error_type,
            "reason": reason,
            "user_query": user_query,
            "fallback_agent": fallback_agent,
        }

        # Write to history.jsonl file (append mode)
        session_dir = self.sessions_dir / session_id
        history_log_path = session_dir / "history.jsonl"

        try:
            # Ensure session directory exists
            session_dir.mkdir(parents=True, exist_ok=True)

            # Append to JSONL file
            with open(history_log_path, "a") as f:
                json.dump(failure_entry, f)
                f.write("\n")
        except Exception as log_error:
            # If logging fails, write to stderr but don't crash
            import sys

            print(
                f"ERROR: Failed to log router failure to {history_log_path}: {log_error}",
                file=sys.stderr,
            )

    def log_agentic_exhaustion(
        self,
        session_id: str,
        agent_name: str,
        objective: str,
        max_steps: int,
        steps_completed: int,
        partial_result: Optional[str] = None
    ) -> None:
        """Log AgenticStepProcessor max steps exhaustion event (T055).

        Args:
            session_id: Session identifier
            agent_name: Name of agent that exhausted steps
            objective: The objective that wasn't completed
            max_steps: Maximum steps configured
            steps_completed: Actual steps completed (usually == max_steps)
            partial_result: Partial results generated before exhaustion
        """
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "agentic_exhaustion",
            "session_id": session_id,
            "agent_name": agent_name,
            "objective": objective,
            "max_steps": max_steps,
            "steps_completed": steps_completed,
            "partial_result_length": len(partial_result) if partial_result else 0,
            "completion_detected": False  # Exhausted before completion
        }

        # Get session log path
        session_dir = self.sessions_dir / session_id
        log_file = session_dir / "history.jsonl"

        try:
            # Ensure session directory exists
            session_dir.mkdir(parents=True, exist_ok=True)

            # Append to JSONL
            with open(log_file, "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
        except Exception as log_error:
            # If logging fails, write to stderr but don't crash
            import sys
            print(
                f"ERROR: Failed to log agentic exhaustion to {log_file}: {log_error}",
                file=sys.stderr,
            )

    def get_exhaustion_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get agentic exhaustion events from session history (T055).

        Args:
            session_id: Session identifier
            limit: Maximum number of events to return

        Returns:
            List of exhaustion events, newest first
        """
        session_dir = self.sessions_dir / session_id
        log_file = session_dir / "history.jsonl"

        if not log_file.exists():
            return []

        exhaustions = []
        with open(log_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line.strip())
                    if entry.get("event_type") == "agentic_exhaustion":
                        exhaustions.append(entry)
                except json.JSONDecodeError:
                    continue

        # Return newest first
        exhaustions.reverse()

        if limit:
            return exhaustions[:limit]
        return exhaustions

    def get_router_decision_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[dict]:
        """Get router decision history from JSONL log (T043).

        Args:
            session_id: Session identifier
            limit: Maximum number of entries to return (newest first)

        Returns:
            List of router decision log entries
        """
        session_dir = self.sessions_dir / session_id
        history_file = session_dir / "history.jsonl"

        if not history_file.exists():
            return []

        decisions = []
        with open(history_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line.strip())
                    if entry.get("event_type") == "router_decision":
                        decisions.append(entry)
                except json.JSONDecodeError:
                    continue

        # Return newest first
        decisions.reverse()

        if limit:
            return decisions[:limit]
        return decisions

    # T090: Step Tracking Detection Logic
    def detect_step_transition(self, session: "Session", agent_output: str) -> Optional["WorkflowStep"]:
        """Detect if agent output signals step completion.

        Args:
            session: Current CLI session
            agent_output: Latest agent message content

        Returns:
            Completed WorkflowStep if detected, None otherwise

        Logic:
            1. Load current workflow (if exists)
            2. Get current pending step
            3. Check for completion signals in agent_output:
               - Keywords: "completed", "done", "finished", "✅"
               - Step reference: "Step N completed" or similar
            4. If detected, mark step as completed
            5. Return completed step
        """
        from .models.workflow import WorkflowState, WorkflowStep

        # Load workflow
        workflow = self.load_workflow(session.id)
        if not workflow:
            return None

        # Get current step
        current_step = workflow.current_step
        if not current_step or current_step.status != "pending":
            return None

        # Simple keyword matching (can upgrade to LLM later)
        completion_keywords = ["completed", "done", "finished", "✅", "✓"]
        agent_lower = agent_output.lower()

        # Check if any completion signal present
        if any(keyword in agent_lower for keyword in completion_keywords):
            # Mark step completed
            current_step.mark_completed(result=agent_output[:200])  # First 200 chars as result

            # Advance workflow to next step
            workflow.advance_step()

            # Save updated workflow
            self.save_workflow(session.id, workflow)

            return current_step

        return None

    # T087: Auto-Update Workflow on Message
    def update_workflow_on_message(self, session: "Session", message: "Message") -> Optional[Dict[str, Any]]:
        """Update workflow state after agent message.

        Called from TUI after each agent response to check for step transitions.

        Args:
            session: Current CLI session
            message: Latest message from agent

        Returns:
            Optional dict with step completion info:
                {
                    "step_description": str,
                    "progress_percentage": float,
                    "completed_count": int,
                    "total_steps": int
                }
            Returns None if no workflow active or no step completed.
        """
        import logging

        logger = logging.getLogger(__name__)

        if message.role != "assistant":
            return None

        # Detect step completion
        completed_step = self.detect_step_transition(session, message.content)

        if completed_step:
            # Log step completion
            logger.info(f"Workflow step completed: {completed_step.description}")

            # Load workflow to get updated state
            workflow = self.load_workflow(session.id)
            if workflow:
                # Show progress update
                completed_count = len([s for s in workflow.steps if s.status == "completed"])
                logger.info(
                    f"Workflow progress: {workflow.progress_percentage:.0f}% "
                    f"({completed_count}/{len(workflow.steps)} steps)"
                )

                # Return completion info for TUI display
                return {
                    "step_description": completed_step.description,
                    "progress_percentage": workflow.progress_percentage,
                    "completed_count": completed_count,
                    "total_steps": len(workflow.steps)
                }

        return None

    def list_all_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows across all sessions (T093).

        Returns:
            List of workflow dictionaries with session and progress information

        Returns workflow data sorted by creation date (most recent first) including:
        - session_name: Name of session containing workflow
        - objective: Workflow objective
        - progress: Percentage completed (0.0-100.0)
        - status: Workflow status ('active' or 'complete')
        - created_at: Creation timestamp
        - step_count: Total number of steps
        - completed_count: Number of completed steps
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT s.name, w.objective, w.steps, w.completed_at, w.created_at
                FROM workflow_states w
                JOIN sessions s ON w.session_id = s.id
                ORDER BY w.created_at DESC
                """
            )

            workflows = []
            for row in cursor.fetchall():
                session_name = row[0]
                objective = row[1]
                steps_json = row[2]
                completed_at = row[3]
                created_at = row[4]

                # Parse steps to calculate progress
                steps_data = json.loads(steps_json) if steps_json else []
                total_steps = len(steps_data)
                completed_count = sum(
                    1 for step in steps_data if step.get("status") == "completed"
                )

                # Calculate progress percentage
                if total_steps > 0:
                    progress = (completed_count / total_steps) * 100.0
                else:
                    progress = 0.0

                # Determine status
                if completed_at is not None:
                    status = "complete"
                else:
                    status = "active"

                workflows.append({
                    "session_name": session_name,
                    "objective": objective,
                    "progress": progress,
                    "status": status,
                    "created_at": created_at,
                    "step_count": total_steps,
                    "completed_count": completed_count
                })

            return workflows

        finally:
            conn.close()

    # ==========================================================================
    # V3: Multi-Agent Communication CRUD Methods (003-T009 to T014)
    # ==========================================================================

    # --- Task Queue CRUD (FR-006 to FR-010) ---

    @track_task(operation_type="CREATE")
    def create_task(
        self,
        session_id: str,
        description: str,
        source_agent: str,
        target_agent: str,
        priority: str = "medium",
        context: Optional[Dict[str, Any]] = None
    ) -> "Task":
        """Create a new task in the queue (003-T009).

        Args:
            session_id: Session identifier
            description: Task description
            source_agent: Agent delegating the task
            target_agent: Agent to execute the task
            priority: Task priority (low, medium, high)
            context: Additional context data

        Returns:
            Task: Created task object

        Raises:
            RuntimeError: If database operation fails
        """
        from .models.task import Task, TaskPriority

        task = Task.create(
            description=description,
            source_agent=source_agent,
            target_agent=target_agent,
            priority=TaskPriority(priority),
            context=context
        )

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO task_queue (
                    task_id, session_id, description, source_agent, target_agent,
                    priority, status, context_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    session_id,
                    task.description,
                    task.source_agent,
                    task.target_agent,
                    task.priority.value,
                    task.status.value,
                    json.dumps(task.context),
                    task.created_at,
                ),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to create task: {e}")
        finally:
            conn.close()

        return task

    def get_task(self, task_id: str) -> Optional["Task"]:
        """Get a task by ID (003-T009).

        Args:
            task_id: Task identifier

        Returns:
            Task or None if not found
        """
        from .models.task import Task

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT task_id, session_id, description, source_agent, target_agent,
                       priority, status, context_json, created_at, started_at,
                       completed_at, result_json, error_message
                FROM task_queue WHERE task_id = ?
                """,
                (task_id,),
            )
            row = cursor.fetchone()
            if row:
                return Task.from_db_row(row)
            return None
        finally:
            conn.close()

    def list_tasks(
        self,
        session_id: str,
        status: Optional[str] = None,
        target_agent: Optional[str] = None,
        limit: int = 100
    ) -> List["Task"]:
        """List tasks with optional filters (003-T009).

        Args:
            session_id: Session identifier
            status: Filter by status
            target_agent: Filter by target agent
            limit: Maximum tasks to return

        Returns:
            List of Task objects
        """
        from .models.task import Task

        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT * FROM task_queue WHERE session_id = ?"
            params = [session_id]

            if status:
                query += " AND status = ?"
                params.append(status)

            if target_agent:
                query += " AND target_agent = ?"
                params.append(target_agent)

            query += " ORDER BY priority DESC, created_at ASC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [Task.from_db_row(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    @track_task(operation_type="STATE_CHANGE")
    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update task status (003-T009).

        Args:
            task_id: Task identifier
            status: New status
            result: Task result (for completed)
            error_message: Error message (for failed)

        Returns:
            True if updated, False if task not found
        """
        conn = sqlite3.connect(self.db_path)
        try:
            now = time.time()

            if status == "in_progress":
                conn.execute(
                    "UPDATE task_queue SET status = ?, started_at = ? WHERE task_id = ?",
                    (status, now, task_id),
                )
            elif status in ("completed", "failed"):
                conn.execute(
                    """UPDATE task_queue
                       SET status = ?, completed_at = ?, result_json = ?, error_message = ?
                       WHERE task_id = ?""",
                    (status, now, json.dumps(result) if result else None, error_message, task_id),
                )
            else:
                conn.execute(
                    "UPDATE task_queue SET status = ? WHERE task_id = ?",
                    (status, task_id),
                )

            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()

    # --- Blackboard CRUD (FR-011 to FR-015) ---

    def write_blackboard(
        self,
        session_id: str,
        key: str,
        value: Any,
        written_by: str
    ) -> "BlackboardEntry":
        """Write to blackboard (003-T010).

        Args:
            session_id: Session identifier
            key: Entry key
            value: Entry value (JSON-serializable)
            written_by: Agent writing the entry

        Returns:
            BlackboardEntry object

        Raises:
            RuntimeError: If database operation fails
        """
        from .models.blackboard import BlackboardEntry

        now = time.time()
        conn = sqlite3.connect(self.db_path)
        try:
            # Check if key exists
            cursor = conn.execute(
                "SELECT version FROM blackboard WHERE session_id = ? AND key = ?",
                (session_id, key),
            )
            row = cursor.fetchone()

            if row:
                # Update existing entry
                new_version = row[0] + 1
                conn.execute(
                    """UPDATE blackboard
                       SET value_json = ?, written_by = ?, written_at = ?, version = ?
                       WHERE session_id = ? AND key = ?""",
                    (json.dumps(value), written_by, now, new_version, session_id, key),
                )
                entry = BlackboardEntry(
                    key=key,
                    value=value,
                    written_by=written_by,
                    written_at=now,
                    version=new_version
                )
            else:
                # Insert new entry
                conn.execute(
                    """INSERT INTO blackboard (session_id, key, value_json, written_by, written_at, version)
                       VALUES (?, ?, ?, ?, ?, 1)""",
                    (session_id, key, json.dumps(value), written_by, now),
                )
                entry = BlackboardEntry(
                    key=key,
                    value=value,
                    written_by=written_by,
                    written_at=now,
                    version=1
                )

            conn.commit()
            return entry
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to write to blackboard: {e}")
        finally:
            conn.close()

    def read_blackboard(self, session_id: str, key: str) -> Optional["BlackboardEntry"]:
        """Read from blackboard (003-T010).

        Args:
            session_id: Session identifier
            key: Entry key

        Returns:
            BlackboardEntry or None if not found
        """
        from .models.blackboard import BlackboardEntry

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """SELECT session_id, key, value_json, written_by, written_at, version
                   FROM blackboard WHERE session_id = ? AND key = ?""",
                (session_id, key),
            )
            row = cursor.fetchone()
            if row:
                return BlackboardEntry.from_db_row(row)
            return None
        finally:
            conn.close()

    def list_blackboard_keys(self, session_id: str) -> List[str]:
        """List all blackboard keys (003-T010).

        Args:
            session_id: Session identifier

        Returns:
            List of key strings
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT key FROM blackboard WHERE session_id = ? ORDER BY written_at DESC",
                (session_id,),
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def delete_blackboard_entry(self, session_id: str, key: str) -> bool:
        """Delete a blackboard entry (003-T010).

        Args:
            session_id: Session identifier
            key: Entry key

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "DELETE FROM blackboard WHERE session_id = ? AND key = ?",
                (session_id, key),
            )
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()

    # --- MultiAgentWorkflow CRUD (FR-021 to FR-025) ---

    def create_multi_agent_workflow(
        self,
        session_id: str,
        agents: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> "MultiAgentWorkflow":
        """Create a new multi-agent workflow (003-T011).

        Args:
            session_id: Session identifier
            agents: List of agent names involved
            context: Additional workflow context

        Returns:
            MultiAgentWorkflow object
        """
        from .models.workflow import MultiAgentWorkflow

        workflow = MultiAgentWorkflow.create(agents=agents, context=context)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """INSERT INTO workflow_state (
                    workflow_id, session_id, stage, agents_involved_json,
                    completed_tasks_json, current_task, context_json, started_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    workflow.workflow_id,
                    session_id,
                    workflow.stage.value,
                    json.dumps(workflow.agents_involved),
                    json.dumps(workflow.completed_tasks),
                    workflow.current_task,
                    json.dumps(workflow.context),
                    workflow.started_at,
                    workflow.updated_at,
                ),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise RuntimeError(f"Failed to create workflow: {e}")
        finally:
            conn.close()

        return workflow

    def get_multi_agent_workflow(self, session_id: str) -> Optional["MultiAgentWorkflow"]:
        """Get multi-agent workflow for session (003-T011).

        Args:
            session_id: Session identifier

        Returns:
            MultiAgentWorkflow or None if not found
        """
        from .models.workflow import MultiAgentWorkflow

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """SELECT workflow_id, session_id, stage, agents_involved_json,
                          completed_tasks_json, current_task, context_json, started_at, updated_at
                   FROM workflow_state WHERE session_id = ?""",
                (session_id,),
            )
            row = cursor.fetchone()
            if row:
                return MultiAgentWorkflow.from_db_row(row)
            return None
        finally:
            conn.close()

    def update_multi_agent_workflow(
        self,
        session_id: str,
        stage: Optional[str] = None,
        current_task: Optional[str] = None,
        completed_task_id: Optional[str] = None,
        new_agent: Optional[str] = None
    ) -> bool:
        """Update multi-agent workflow (003-T011).

        Args:
            session_id: Session identifier
            stage: New workflow stage
            current_task: Current task description
            completed_task_id: Task ID to add to completed list
            new_agent: Agent name to add to involved list

        Returns:
            True if updated, False if workflow not found
        """
        workflow = self.get_multi_agent_workflow(session_id)
        if not workflow:
            return False

        if stage:
            from .models.workflow import WorkflowStage
            workflow.set_stage(WorkflowStage(stage))

        if current_task is not None:
            workflow.set_current_task(current_task)

        if completed_task_id:
            workflow.complete_task(completed_task_id)

        if new_agent:
            workflow.add_agent(new_agent)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """UPDATE workflow_state
                   SET stage = ?, current_task = ?, agents_involved_json = ?,
                       completed_tasks_json = ?, updated_at = ?
                   WHERE session_id = ?""",
                (
                    workflow.stage.value,
                    workflow.current_task,
                    json.dumps(workflow.agents_involved),
                    json.dumps(workflow.completed_tasks),
                    workflow.updated_at,
                    session_id,
                ),
            )
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()

    # --- Message Log CRUD (FR-016 to FR-020) ---

    def log_message(
        self,
        session_id: str,
        sender: str,
        receiver: str,
        message_type: str,
        payload: Dict[str, Any]
    ) -> str:
        """Log an agent message (003-T012).

        Args:
            session_id: Session identifier
            sender: Sending agent
            receiver: Receiving agent
            message_type: Message type
            payload: Message payload

        Returns:
            Message ID
        """
        import uuid

        message_id = str(uuid.uuid4())
        now = time.time()

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """INSERT INTO message_log (
                    message_id, session_id, sender, receiver, message_type,
                    payload_json, timestamp, delivered
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
                (message_id, session_id, sender, receiver, message_type, json.dumps(payload), now),
            )
            conn.commit()
        finally:
            conn.close()

        return message_id

    def get_messages(
        self,
        session_id: str,
        sender: Optional[str] = None,
        receiver: Optional[str] = None,
        message_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get messages with optional filters (003-T012).

        Args:
            session_id: Session identifier
            sender: Filter by sender
            receiver: Filter by receiver
            message_type: Filter by type
            limit: Maximum messages

        Returns:
            List of message dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT * FROM message_log WHERE session_id = ?"
            params = [session_id]

            if sender:
                query += " AND sender = ?"
                params.append(sender)
            if receiver:
                query += " AND receiver = ?"
                params.append(receiver)
            if message_type:
                query += " AND message_type = ?"
                params.append(message_type)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()

    def mark_message_delivered(self, message_id: str) -> bool:
        """Mark a message as delivered (003-T012).

        Args:
            message_id: Message identifier

        Returns:
            True if updated
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE message_log SET delivered = 1 WHERE message_id = ?",
                (message_id,),
            )
            conn.commit()
            return conn.total_changes > 0
        finally:
            conn.close()
