"""
Comprehensive agent activity logging system.

This module provides the ActivityLogger class that captures ALL agent interactions,
reasoning steps, tool calls, and decisions to persistent storage (JSONL + SQLite)
for searchability without affecting chat history or token usage.
"""

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from uuid import uuid4

logger = logging.getLogger(__name__)


class ActivityLogger:
    """Comprehensive agent activity logging system.

    Captures ALL agent interactions, reasoning steps, tool calls, and decisions
    to persistent storage (JSONL + SQLite) for searchability without affecting
    chat history or token usage.

    This logger is designed to work independently from chat history and token
    limits, providing a complete audit trail of agent activities that can be
    searched via grep or SQL queries.

    Attributes:
        session_name: Current session identifier
        log_dir: Directory for JSONL activity logs
        db_path: Path to SQLite database
        enable_console: Whether to also log to console
        current_chain_id: ID of current interaction chain
        current_parent_id: ID of parent activity in current chain
        current_depth: Current nesting depth in interaction chain
    """

    # Activity types
    ACTIVITY_TYPES = {
        "user_input",
        "agent_input",
        "agent_output",
        "tool_call",
        "tool_result",
        "reasoning_step",
        "router_decision",
        "router_fallback",
        "error",
        "system_message"
    }

    def __init__(
        self,
        session_name: str,
        log_dir: Path,
        db_path: Path,
        enable_console: bool = False
    ):
        """Initialize activity logger.

        Args:
            session_name: Current session identifier
            log_dir: Directory for JSONL activity logs
            db_path: Path to SQLite database
            enable_console: Whether to also log to console (default: False)
        """
        self.session_name = session_name
        self.log_dir = Path(log_dir)
        self.db_path = Path(db_path)
        self.enable_console = enable_console

        # Create log directory structure
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.chains_dir = self.log_dir / "chains"
        self.chains_dir.mkdir(exist_ok=True)

        # Activity log file (main JSONL file)
        self.activity_log_path = self.log_dir / "activities.jsonl"

        # Current interaction chain tracking
        self.current_chain_id: Optional[str] = None
        self.current_parent_id: Optional[str] = None
        self.current_depth: int = 0

        # Activity buffer (for batch writes)
        self._activity_buffer: List[Dict[str, Any]] = []
        self._buffer_size = 10  # Flush after 10 activities

        # Initialize database
        self._init_database()

        logger.info(
            f"ActivityLogger initialized for session '{session_name}' "
            f"(log_dir: {self.log_dir}, db: {self.db_path})"
        )

    def _init_database(self):
        """Initialize SQLite database with activity tables."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Agent activities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_activities (
                    activity_id TEXT PRIMARY KEY,
                    session_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    activity_type TEXT NOT NULL,
                    agent_name TEXT,
                    agent_model TEXT,
                    parent_activity_id TEXT,
                    interaction_chain_id TEXT NOT NULL,
                    depth_level INTEGER DEFAULT 0,
                    content_preview TEXT,
                    full_log_path TEXT,
                    tags TEXT,
                    FOREIGN KEY (parent_activity_id) REFERENCES agent_activities(activity_id)
                )
            """)

            # Indexes for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_activities_session
                ON agent_activities(session_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_activities_timestamp
                ON agent_activities(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_activities_type
                ON agent_activities(activity_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_activities_agent
                ON agent_activities(agent_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_activities_chain
                ON agent_activities(interaction_chain_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_activities_parent
                ON agent_activities(parent_activity_id)
            """)

            # Interaction chains table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interaction_chains (
                    chain_id TEXT PRIMARY KEY,
                    session_name TEXT NOT NULL,
                    root_activity_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    total_activities INTEGER DEFAULT 0,
                    total_agents_involved INTEGER DEFAULT 0,
                    max_depth_level INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (root_activity_id) REFERENCES agent_activities(activity_id)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chains_session
                ON interaction_chains(session_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chains_status
                ON interaction_chains(status)
            """)

            conn.commit()
            conn.close()

            logger.info("ActivityLogger database tables initialized")

        except Exception as e:
            logger.error(f"Failed to initialize ActivityLogger database: {e}")
            raise

    def _generate_activity_id(self) -> str:
        """Generate unique activity ID with timestamp prefix."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid4())[:8]
        return f"act_{timestamp}_{unique_id}"

    def _generate_chain_id(self) -> str:
        """Generate unique interaction chain ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid4())[:8]
        return f"chain_{timestamp}_{unique_id}"

    def start_interaction_chain(self, root_activity_id: Optional[str] = None) -> str:
        """Start a new interaction chain (root user input).

        Args:
            root_activity_id: Optional pre-generated activity ID for root

        Returns:
            Generated chain ID
        """
        chain_id = self._generate_chain_id()
        self.current_chain_id = chain_id
        self.current_parent_id = None
        self.current_depth = 0

        # If root activity ID not provided, generate one
        if root_activity_id is None:
            root_activity_id = self._generate_activity_id()

        # Create chain record in database
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO interaction_chains
                (chain_id, session_name, root_activity_id, started_at, status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                chain_id,
                self.session_name,
                root_activity_id,
                datetime.now().isoformat(),
                "active"
            ))

            conn.commit()
            conn.close()

            logger.debug(f"Started interaction chain: {chain_id}")

        except Exception as e:
            logger.error(f"Failed to create interaction chain: {e}")

        return chain_id

    def log_activity(
        self,
        activity_type: str,
        agent_name: Optional[str],
        content: Dict[str, Any],
        agent_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[str] = None
    ) -> str:
        """Log a single activity entry.

        Args:
            activity_type: Type of activity (must be in ACTIVITY_TYPES)
            agent_name: Name of agent performing activity (None for system)
            content: Activity content (input, output, etc.)
            agent_model: Model identifier for agent
            metadata: Optional metadata (tokens, duration, etc.)
            tags: Optional tags for categorization
            parent_id: Optional parent activity ID (overrides current_parent_id)

        Returns:
            Generated activity ID

        Raises:
            ValueError: If activity_type is invalid
        """
        if activity_type not in self.ACTIVITY_TYPES:
            raise ValueError(
                f"Invalid activity_type: {activity_type}. "
                f"Must be one of: {', '.join(self.ACTIVITY_TYPES)}"
            )

        # Generate activity ID
        activity_id = self._generate_activity_id()
        timestamp = datetime.now().isoformat()

        # Use current chain ID or start new chain if none exists
        chain_id = self.current_chain_id
        if chain_id is None:
            chain_id = self.start_interaction_chain(root_activity_id=activity_id)

        # Determine parent ID (explicit override or current parent)
        actual_parent_id = parent_id if parent_id is not None else self.current_parent_id

        # Build searchable text for grep
        searchable_parts = []
        if agent_name:
            searchable_parts.append(f"Agent: {agent_name}")
        searchable_parts.append(f"Type: {activity_type}")
        if isinstance(content, dict):
            for key, value in content.items():
                searchable_parts.append(f"{key}: {value}")
        else:
            searchable_parts.append(str(content))
        searchable_text = " | ".join(searchable_parts)

        # Create content preview (first 200 chars)
        content_str = json.dumps(content) if isinstance(content, dict) else str(content)
        content_preview = (content_str[:200] + "...") if len(content_str) > 200 else content_str

        # Build full activity entry
        activity_entry = {
            "activity_id": activity_id,
            "session_name": self.session_name,
            "timestamp": timestamp,
            "activity_type": activity_type,
            "agent_name": agent_name,
            "agent_model": agent_model,
            "parent_activity_id": actual_parent_id,
            "interaction_chain_id": chain_id,
            "depth_level": self.current_depth,
            "content": content,
            "metadata": metadata or {},
            "searchable_text": searchable_text,
            "tags": tags or []
        }

        # Write to JSONL file immediately (don't buffer for now - can optimize later)
        self._write_to_jsonl(activity_entry)

        # Write to database (with buffering optimization)
        self._write_to_database(activity_entry, content_preview)

        # Console logging if enabled
        if self.enable_console:
            logger.info(
                f"[ActivityLog] {activity_type} | Agent: {agent_name or 'System'} | "
                f"Chain: {chain_id[:12]}... | Depth: {self.current_depth}"
            )

        # Update parent ID for next activity in chain
        self.current_parent_id = activity_id

        # Update chain statistics
        self._update_chain_stats(chain_id, activity_type, agent_name)

        return activity_id

    def _write_to_jsonl(self, activity_entry: Dict[str, Any]):
        """Write activity entry to JSONL file."""
        try:
            with open(self.activity_log_path, 'a', encoding='utf-8') as f:
                json.dump(activity_entry, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write activity to JSONL: {e}")

    def _write_to_database(self, activity_entry: Dict[str, Any], content_preview: str):
        """Write activity entry to SQLite database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO agent_activities
                (activity_id, session_name, timestamp, activity_type, agent_name,
                 agent_model, parent_activity_id, interaction_chain_id, depth_level,
                 content_preview, full_log_path, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                activity_entry["activity_id"],
                activity_entry["session_name"],
                activity_entry["timestamp"],
                activity_entry["activity_type"],
                activity_entry["agent_name"],
                activity_entry["agent_model"],
                activity_entry["parent_activity_id"],
                activity_entry["interaction_chain_id"],
                activity_entry["depth_level"],
                content_preview,
                str(self.activity_log_path),
                ",".join(activity_entry["tags"]) if activity_entry["tags"] else ""
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to write activity to database: {e}")

    def _update_chain_stats(
        self,
        chain_id: str,
        activity_type: str,
        agent_name: Optional[str]
    ):
        """Update statistics for interaction chain."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Increment total activities
            cursor.execute("""
                UPDATE interaction_chains
                SET total_activities = total_activities + 1,
                    max_depth_level = MAX(max_depth_level, ?)
                WHERE chain_id = ?
            """, (self.current_depth, chain_id))

            # Update unique agent count if agent activity
            if agent_name:
                # Get current unique agents
                cursor.execute("""
                    SELECT COUNT(DISTINCT agent_name)
                    FROM agent_activities
                    WHERE interaction_chain_id = ?
                      AND agent_name IS NOT NULL
                """, (chain_id,))
                unique_agents = cursor.fetchone()[0]

                cursor.execute("""
                    UPDATE interaction_chains
                    SET total_agents_involved = ?
                    WHERE chain_id = ?
                """, (unique_agents, chain_id))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update chain stats: {e}")

    def end_interaction_chain(self, status: str = "completed"):
        """Mark current interaction chain as completed.

        Args:
            status: Final status ('completed', 'error', etc.)
        """
        if self.current_chain_id is None:
            logger.warning("No active interaction chain to end")
            return

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE interaction_chains
                SET completed_at = ?,
                    status = ?
                WHERE chain_id = ?
            """, (datetime.now().isoformat(), status, self.current_chain_id))

            conn.commit()
            conn.close()

            logger.debug(
                f"Ended interaction chain: {self.current_chain_id} "
                f"(status: {status})"
            )

        except Exception as e:
            logger.error(f"Failed to end interaction chain: {e}")

        finally:
            # Reset current chain tracking
            self.current_chain_id = None
            self.current_parent_id = None
            self.current_depth = 0

    def increase_depth(self):
        """Increase nesting depth (entering nested agent/reasoning)."""
        self.current_depth += 1

    def decrease_depth(self):
        """Decrease nesting depth (exiting nested agent/reasoning)."""
        self.current_depth = max(0, self.current_depth - 1)

    def get_chain_activities(
        self,
        chain_id: str,
        include_content: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all activities in a specific interaction chain.

        Args:
            chain_id: Interaction chain ID
            include_content: Whether to load full content from JSONL

        Returns:
            List of activity dictionaries (ordered by timestamp)
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT *
                FROM agent_activities
                WHERE interaction_chain_id = ?
                ORDER BY timestamp ASC
            """, (chain_id,))

            activities = [dict(row) for row in cursor.fetchall()]
            conn.close()

            # If full content requested, load from JSONL
            if include_content and activities:
                activities = self._enrich_with_full_content(activities)

            return activities

        except Exception as e:
            logger.error(f"Failed to get chain activities: {e}")
            return []

    def _enrich_with_full_content(
        self,
        activities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich activity records with full content from JSONL."""
        # Read entire JSONL file
        activity_map = {}
        try:
            with open(self.activity_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        activity_map[entry["activity_id"]] = entry
        except Exception as e:
            logger.error(f"Failed to read JSONL for content enrichment: {e}")
            return activities

        # Merge full content
        enriched = []
        for activity in activities:
            activity_id = activity["activity_id"]
            if activity_id in activity_map:
                full_entry = activity_map[activity_id]
                activity["content"] = full_entry.get("content", {})
                activity["metadata"] = full_entry.get("metadata", {})
                activity["searchable_text"] = full_entry.get("searchable_text", "")
            enriched.append(activity)

        return enriched

    def get_agent_activities(
        self,
        agent_name: str,
        limit: int = 100,
        include_content: bool = False
    ) -> List[Dict[str, Any]]:
        """Get recent activities for a specific agent.

        Args:
            agent_name: Agent name to filter by
            limit: Maximum number of activities to return
            include_content: Whether to load full content from JSONL

        Returns:
            List of activity dictionaries (most recent first)
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT *
                FROM agent_activities
                WHERE agent_name = ? AND session_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (agent_name, self.session_name, limit))

            activities = [dict(row) for row in cursor.fetchall()]
            conn.close()

            if include_content and activities:
                activities = self._enrich_with_full_content(activities)

            return activities

        except Exception as e:
            logger.error(f"Failed to get agent activities: {e}")
            return []

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure current chain is ended."""
        if self.current_chain_id is not None:
            status = "error" if exc_type else "completed"
            self.end_interaction_chain(status=status)
